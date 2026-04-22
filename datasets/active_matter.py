"""Active Matter dataset for DynaMo SSL, aligned with the reference pipeline.

Reference (reference_jepa_physics/data.py::WellDatasetForSequence) specifics:
  * Reads HDF5 shards directly with ``h5py`` (per-worker LRU of open handles
    + per-dataset chunk cache) -- much cheaper than loading whole trajectories.
  * Does NOT normalize channels; raw physical fields are fed to the encoder
    (whose first block should handle per-channel scale via LayerNorm/BN).
  * Each ``__getitem__`` returns a single F-frame contiguous window.
  * Supports configurable ``stride`` (default=F, i.e. non-overlapping).
  * Optional Gaussian-noise augmentation.

This wrapper adapts that loader to DynaMo's ``TrajectoryDataset`` contract:
  * ``__getitem__`` returns ``(obs, actions, mask)`` where
      - obs:     Tensor[T, V=1, C=11, H, W]  float32 (raw physical values)
      - actions: Tensor[T, 1]  zeros (DynaMo SSL does not use actions)
      - mask:    Tensor[T]     all True
  * This lets us plug into ``train.py`` by passing our dataset instance straight
    as ``train_set`` / ``test_set`` (i.e. we SKIP ``TrajectorySlicerDataset``;
    each of our items is already a window of length ``num_frames``).

Physical parameters (alpha, zeta) are kept available via ``get_physical_params``
for the downstream linear-probe / kNN evaluation. They are NEVER returned from
``__getitem__`` to avoid any chance of leaking into the SSL loss.
"""

from __future__ import annotations

import random
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


class ActiveMatterWindowDataset(Dataset):
    """Active Matter windows for DynaMo SSL pretraining.

    Args:
        data_dir: path containing ``data/<split>/*.hdf5``. For The Well's
            on-disk layout this is ``<well_base_path>/active_matter``.
        num_frames: temporal length ``T`` of each window.
        split: ``"train" | "valid" | "test"``.
        resolution: spatial size to bilinear-resize to (``None`` keeps 256).
        stride: step between consecutive window starts (defaults to ``num_frames``,
            i.e. non-overlapping -- matches reference's ``WellDatasetForSequence``).
        noise_std: std of additive Gaussian noise (0 disables).
        subset_indices: optional explicit list of window indices to use
            (e.g. for matching reference's fixed subsets).
        max_open_files: per-worker LRU size of open HDF5 handles.
        rdcc_nbytes / rdcc_nslots / rdcc_w0: HDF5 chunk cache tuning.

    Returned tuple (DynaMo-compatible):
        obs:     (T, V=1, C=11, H, W) float32, raw (unnormalized) physical fields
        actions: (T, 1) zeros
        mask:    (T,) bool, all True
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        num_frames: int,
        split: str = "train",
        resolution: Optional[int] = 224,
        stride: Optional[int] = None,
        noise_std: float = 0.0,
        subset_indices: Optional[List[int]] = None,
        max_open_files: int = 6,
        rdcc_nbytes: int = 512 * 1024 ** 2,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
    ):
        super().__init__()
        split = _SPLIT_ALIASES.get(split, split)
        data_dir = Path(data_dir)
        self.data_dir = data_dir / "data" / split
        self.dataset_name = data_dir.stem
        self.split = split
        self.num_frames = int(num_frames)
        assert self.num_frames > 0
        self.stride = int(stride) if stride is not None else self.num_frames
        self.resolution = resolution
        self.noise_std = float(noise_std)
        self.subset_indices = list(subset_indices) if subset_indices is not None else None

        self._open: Optional[OrderedDict[str, Tuple[h5py.File, dict]]] = None
        self._max_open_files = int(max_open_files)
        self._rdcc = (int(rdcc_nbytes), int(rdcc_nslots), float(rdcc_w0))

        self.index, self.physical_params_idx = self._build_index()
        if len(self.index) == 0:
            raise ValueError(
                f"No valid windows found under {self.data_dir} with "
                f"num_frames={self.num_frames}, stride={self.stride}."
            )
        self._build_global_field_schema(self.data_dir / self.index[0][0])

        print(
            f"[ActiveMatterWindowDataset split={split}] "
            f"{len(self.index)} windows across {len(self.physical_params_idx)} files "
            f"(num_frames={self.num_frames}, stride={self.stride}, C={self._C_total}, "
            f"HxW={self._spatial_shape})",
            flush=True,
        )

    def _build_index(self) -> Tuple[
        List[Tuple[str, int, int]], Dict[str, List[np.ndarray]]
    ]:
        """Scan HDF5 files and build a flat ``(file_name, obj_id, t0)`` list."""
        idx: List[Tuple[str, int, int]] = []
        params: Dict[str, List[np.ndarray]] = {}
        F_ = self.num_frames
        paths = sorted(
            list(self.data_dir.rglob("*.h5")) + list(self.data_dir.rglob("*.hdf5"))
        )
        for path in paths:
            with h5py.File(path, "r") as f:
                first_key = next(iter(f["t0_fields"]))
                example = f["t0_fields"][first_key]
                T = int(example.shape[1])      # shape: (num_objs, T, H, W)
                num_objs = int(example.shape[0])
                max_t0 = T - F_
                if max_t0 < 0:
                    continue
                for obj_id in range(num_objs):
                    for t0 in range(0, max_t0 + 1, self.stride):
                        idx.append((path.name, obj_id, t0))
                # Store per-file physical params (exclude L -- constant for active_matter).
                # Shape per entry: (num_objs,)
                params[path.name] = [
                    f["scalars"][key][()]
                    for key in f["scalars"].keys()
                    if key != "L"
                ]
        return idx, params

    def _build_global_field_schema(self, sample_path: Path) -> None:
        """Cache field paths + component shapes from a sample file."""
        field_paths: List[str] = []
        d_sizes: List[int] = []
        comp_shapes: List[Tuple[int, ...]] = []
        with h5py.File(sample_path, "r") as f:
            for group in ("t0_fields", "t1_fields", "t2_fields"):
                if group in f:
                    for name, ds in f[group].items():
                        if not isinstance(ds, h5py.Dataset):
                            continue
                        comp = tuple(ds.shape[4:])        # () / (2,) / (2,2)
                        d_sizes.append(int(np.prod(comp) or 1))
                        comp_shapes.append(comp)
                        field_paths.append(f"{group}/{name}")
            if not field_paths:
                raise RuntimeError(f"No fields in {sample_path}")
            _, _, H, W = f[field_paths[0]].shape
            self._dtype = f[field_paths[0]].dtype

        d_sizes_arr = np.asarray(d_sizes, dtype=np.int64)
        self._field_paths = tuple(field_paths)
        self._d_sizes = d_sizes_arr
        self._comp_shapes = comp_shapes
        self._chan_offsets = np.concatenate(([0], np.cumsum(d_sizes_arr)))
        self._C_total = int(self._chan_offsets[-1])
        self._spatial_shape = (int(H), int(W))

    def __len__(self) -> int:
        return len(self.subset_indices) if self.subset_indices is not None else len(self.index)

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual = self.subset_indices[i] if self.subset_indices is not None else i
        file_name, obj_id, t0 = self.index[actual]
        F_ = self.num_frames

        f, state = self._open_file(file_name)
        H, W = self._spatial_shape
        C = self._C_total
        seq = np.empty((F_, H, W, C), dtype=self._dtype, order="C")

        sel_prefix = (obj_id, slice(t0, t0 + F_), slice(None), slice(None))
        tmp_cache = state.setdefault("buf_cache", {})

        c0 = 0
        for path, dsize, comp_shape in zip(
            self._field_paths, self._d_sizes, self._comp_shapes
        ):
            ds = self._get_ds_handle(f, state, path)
            need_shape = (F_, H, W) + comp_shape
            buf = tmp_cache.get(comp_shape)
            if buf is None or buf.shape != need_shape or buf.dtype != self._dtype:
                buf = np.empty(need_shape, dtype=self._dtype, order="C")
                tmp_cache[comp_shape] = buf

            sel = sel_prefix + (slice(None),) * len(comp_shape)
            ds.read_direct(buf, source_sel=sel)
            view = buf.reshape(F_, H, W, int(dsize))
            c1 = c0 + int(dsize)
            seq[..., c0:c1] = view
            c0 = c1

        # -> torch (C, T, H, W)
        seq_t = torch.from_numpy(seq).permute(3, 0, 1, 2).contiguous().to(torch.float32)

        # Optional spatial resize.
        if self.resolution is not None and seq_t.shape[-1] != self.resolution:
            seq_t = F.interpolate(
                seq_t,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )

        # Optional noise (applied to raw scale).
        if self.noise_std > 0:
            seq_t = seq_t + torch.randn_like(seq_t) * self.noise_std

        # -> DynaMo: (T, V=1, C, H, W)
        obs = seq_t.permute(1, 0, 2, 3).unsqueeze(1).contiguous()
        t = obs.shape[0]
        actions = torch.zeros(t, 1, dtype=torch.float32)
        mask = torch.ones(t, dtype=torch.bool)
        return obs, actions, mask

    def get_physical_params(self, i: int) -> torch.Tensor:
        """Return ``(alpha, zeta)`` for window ``i`` (eval only, never in SSL batch).

        In The Well's active_matter layout each HDF5 file corresponds to ONE
        (alpha, zeta) pair shared across all objects in the file (``L`` is
        excluded as it's constant across the dataset). So we key by file only.
        """
        actual = self.subset_indices[i] if self.subset_indices is not None else i
        file_name, _, _ = self.index[actual]
        per_key = self.physical_params_idx[file_name]   # list of numpy scalars
        return torch.tensor([float(v) for v in per_key], dtype=torch.float32)

    # ---- per-worker HDF5 handle pool ----

    def _get_ds_handle(self, f: h5py.File, state: dict, path: str) -> h5py.Dataset:
        ds_cache = state.setdefault("ds_cache", {})
        if path in ds_cache:
            return ds_cache[path]
        ds = f[path]
        try:
            ds.id.set_chunk_cache(self._rdcc[1], self._rdcc[0], self._rdcc[2])
        except Exception:
            pass
        ds_cache[path] = ds
        return ds

    def _ensure_worker_state(self) -> None:
        if self._open is None:
            self._open = OrderedDict()
            weakref.finalize(self, self._close_all)

    def _close_all(self) -> None:
        if self._open:
            for (fp, _) in self._open.values():
                try:
                    fp.close()
                except Exception:
                    pass
            self._open.clear()

    def _open_file(self, file_name: str) -> Tuple[h5py.File, dict]:
        self._ensure_worker_state()
        assert self._open is not None
        if file_name in self._open:
            entry = self._open.pop(file_name)
            self._open[file_name] = entry
            return entry
        while len(self._open) >= self._max_open_files:
            _, (old_f, _) = self._open.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass
        fp = h5py.File(
            self.data_dir / file_name,
            mode="r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=self._rdcc[0],
            rdcc_nslots=self._rdcc[1],
            rdcc_w0=self._rdcc[2],
        )
        entry = (fp, {})
        self._open[file_name] = entry
        return entry

    def __getstate__(self):
        # Drop open HDF5 handles when DataLoader pickles us to workers.
        st = self.__dict__.copy()
        st["_open"] = None
        return st


def build_active_matter_train_val(
    data_dir: Union[str, Path],
    num_frames: int,
    resolution: Optional[int] = 224,
    stride_train: Optional[int] = None,
    stride_val: Optional[int] = None,
    noise_std: float = 0.0,
) -> Tuple[ActiveMatterWindowDataset, ActiveMatterWindowDataset]:
    """Build official (train, valid) window datasets -- no random re-splitting."""
    train = ActiveMatterWindowDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        split="train",
        resolution=resolution,
        stride=stride_train,
        noise_std=noise_std,
    )
    val = ActiveMatterWindowDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        split="valid",
        resolution=resolution,
        stride=stride_val,
        noise_std=0.0,
    )
    return train, val
