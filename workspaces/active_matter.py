"""Downstream probe monitor for The Well's active_matter during SSL training.

This is the *in-training* (online) eval that runs once per epoch from
``Trainer.eval()`` and writes metrics to wandb under ``env_offline_eval/*``.

Design choices, aligned with ``reference_jepa_physics/eval_frozen_regression.py``
so the numbers are directly comparable to the course baseline:

* Feature per window = mean-pool encoder output over (T, V). With our 2D ResNet
  backbone the spatial pooling is already done internally, so this matches the
  reference's ``pool_encoder_output`` which means over (T, H, W).
* Labels are z-scored using the **official The Well global stats** hard-coded
  in the reference (alpha: mu=-3.0 sigma=1.41, zeta: mu=9.0 sigma=5.16), NOT
  per-split stats. This keeps MSE numbers comparable across checkpoints / runs.
* Linear probe: sklearn ``Ridge`` with a small alpha grid. This is a *fast*
  proxy for the reference's SGD ``nn.Linear`` + AdamW + early-stopping setup
  (which would add ~2 min per epoch). The authoritative evaluation for the
  final report is ``eval_thewell.py``, which matches the reference exactly.
* kNN: sklearn ``KNeighborsRegressor(n_neighbors=5, weights="distance")``,
  default L2 metric.

Test split is NEVER touched here (PDF rule 8.4). Only train + valid.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader

from datasets.active_matter import ActiveMatterWindowDataset
from workspaces import base


log = logging.getLogger(__name__)


# Official The Well label stats for active_matter, copied verbatim from
# reference_jepa_physics/eval_frozen_regression.py::LABEL_STATS so our
# z-scored MSE is directly comparable to the course baseline.
LABEL_STATS = {
    "means": np.array([-3.0, 9.0], dtype=np.float32),   # alpha, zeta
    "stds":  np.array([1.41, 5.16], dtype=np.float32),
    "names": ("alpha", "zeta"),
}


def _zscore_labels(y: np.ndarray) -> np.ndarray:
    return (y - LABEL_STATS["means"]) / LABEL_STATS["stds"]


class ActiveMatterWorkspace(base.Workspace):
    """Online Ridge + kNN probes on frozen encoder features."""

    def __init__(self, cfg, work_dir):
        super().__init__(cfg, work_dir)
        # Fast online defaults. Override via cfg for ablations.
        self.ridge_alpha: float = float(getattr(cfg, "probe_ridge_alpha", 1.0))
        self.knn_k: int = int(getattr(cfg, "probe_knn_k", 5))
        self.probe_batch_size: int = int(
            getattr(cfg, "probe_batch_size", max(1, cfg.batch_size))
        )
        self.probe_num_workers: int = int(
            getattr(cfg, "probe_num_workers", 4)
        )
        self._val_dataset: Optional[ActiveMatterWindowDataset] = None

    # ---- dataset helpers ----

    def _get_val_dataset(self) -> ActiveMatterWindowDataset:
        if self._val_dataset is None:
            env = self.cfg.env
            self._val_dataset = ActiveMatterWindowDataset(
                data_dir=env.dataset.data_dir,
                num_frames=self.cfg.window_size,
                split="valid",
                resolution=env.dataset.resolution,
                stride=env.get("stride_val", None),
                noise_std=0.0,
            )
        return self._val_dataset

    # ---- feature extraction ----

    @torch.no_grad()
    def _extract_window_features(
        self,
        dataset: ActiveMatterWindowDataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run encoder over every window and mean-pool the per-frame features.

        Returns:
            X: ``(N_windows, D)`` float32 features
            y: ``(N_windows, 2)`` float32 RAW (unnormalised) (alpha, zeta) labels.
               The caller is responsible for z-scoring with LABEL_STATS.
        """
        encoder = self.accelerator.unwrap_model(self.encoder)
        encoder.eval()
        device = self.accelerator.device

        loader = DataLoader(
            dataset,
            batch_size=self.probe_batch_size,
            shuffle=False,
            num_workers=self.probe_num_workers,
            pin_memory=True,
            persistent_workers=(self.probe_num_workers > 0),
            prefetch_factor=(4 if self.probe_num_workers > 0 else None),
        )

        feats: list[torch.Tensor] = []
        labels: list[np.ndarray] = []

        global_i = 0
        for batch in loader:
            obs, _, _ = batch                          # (B, T, V, C, H, W)
            obs = obs.to(device, non_blocking=True)
            enc = encoder(obs)                         # (B, T, V, E)
            feat = enc.mean(dim=(1, 2)).detach().cpu() # (B, E)
            feats.append(feat)

            B = feat.shape[0]
            for b in range(B):
                y_b = dataset.get_physical_params(global_i + b).numpy()  # (2,)
                labels.append(y_b)
            global_i += B

        X = torch.cat(feats, dim=0).numpy().astype(np.float32)
        y = np.stack(labels, axis=0).astype(np.float32)
        return X, y

    # ---- probes ----

    def _ridge_probe(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> Dict[str, float]:
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(X_tr, y_tr)
        tr_mse = ((ridge.predict(X_tr) - y_tr) ** 2).mean(axis=0)
        val_mse = ((ridge.predict(X_val) - y_val) ** 2).mean(axis=0)
        return {
            "lp_alpha_train_mse_z": float(tr_mse[0]),
            "lp_zeta_train_mse_z": float(tr_mse[1]),
            "lp_alpha_val_mse_z": float(val_mse[0]),
            "lp_zeta_val_mse_z": float(val_mse[1]),
            "lp_val_mean_mse_z": float(val_mse.mean()),
            "lp_ridge_alpha": float(self.ridge_alpha),
        }

    def _knn_probe(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> Dict[str, float]:
        k = max(1, min(self.knn_k, len(X_tr)))
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_tr, y_tr)
        val_mse = ((knn.predict(X_val) - y_val) ** 2).mean(axis=0)
        return {
            "knn_alpha_val_mse_z": float(val_mse[0]),
            "knn_zeta_val_mse_z": float(val_mse[1]),
            "knn_val_mean_mse_z": float(val_mse.mean()),
            "knn_k": float(k),
        }

    # ---- entrypoint ----

    def run_offline_eval(self) -> Dict[str, float]:
        # Run probes on rank 0 only; others block on the final barrier.
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return {"loss": 0.0}

        log.info("[probe] extracting train features (%d windows)", len(self.dataset))
        X_tr, y_tr = self._extract_window_features(self.dataset)
        val_dataset = self._get_val_dataset()
        log.info("[probe] extracting val features (%d windows)", len(val_dataset))
        X_val, y_val = self._extract_window_features(val_dataset)
        log.info(
            "[probe] shapes: X_tr=%s X_val=%s", X_tr.shape, X_val.shape,
        )

        y_tr_z = _zscore_labels(y_tr)
        y_val_z = _zscore_labels(y_val)

        metrics: Dict[str, float] = {"loss": 0.0}
        metrics.update(self._ridge_probe(X_tr, y_tr_z, X_val, y_val_z))
        metrics.update(self._knn_probe(X_tr, y_tr_z, X_val, y_val_z))
        metrics["probe_n_train"] = float(len(X_tr))
        metrics["probe_n_val"] = float(len(X_val))

        log.info(
            "[probe] lp val mse(z): alpha=%.4f zeta=%.4f mean=%.4f",
            metrics["lp_alpha_val_mse_z"],
            metrics["lp_zeta_val_mse_z"],
            metrics["lp_val_mean_mse_z"],
        )
        log.info(
            "[probe] knn val mse(z) k=%d: alpha=%.4f zeta=%.4f mean=%.4f",
            int(metrics["knn_k"]),
            metrics["knn_alpha_val_mse_z"],
            metrics["knn_zeta_val_mse_z"],
            metrics["knn_val_mean_mse_z"],
        )

        self.accelerator.wait_for_everyone()
        return metrics
