"""DynaMo SSL training entrypoint for The Well's active_matter dataset.

Reuses ``train.py::Trainer`` verbatim; only the dataset-construction path is
overridden so we:
  * load the OFFICIAL train/valid split from The Well (no random re-splitting);
  * skip ``TrajectorySlicerDataset`` (our dataset already returns windows of
    length ``window_size``).

Run:
    accelerate launch train_thewell.py
"""

from __future__ import annotations

import hydra
from torch.utils.data import DataLoader

from train import Trainer
from datasets.active_matter import ActiveMatterWindowDataset


class TheWellTrainer(Trainer):
    def _split_and_slice_dataset(self, dataset):
        # ``dataset`` is the train split instantiated via hydra (cfg.env.dataset).
        # We only need to build the matching valid split here -- both are already
        # windowed (each __getitem__ returns obs of shape (T, V, C, H, W)), so
        # no TrajectorySlicerDataset wrapping is required.
        env_cfg = self.cfg.env
        val_set = ActiveMatterWindowDataset(
            data_dir=env_cfg.dataset.data_dir,
            num_frames=self.cfg.window_size,
            split="valid",
            resolution=env_cfg.dataset.resolution,
            stride=env_cfg.get("stride_val", None),
            noise_std=0.0,   # never augment the eval split
        )
        return dataset, val_set

    def _setup_loaders(self, batch_size=None, pin_memory=True, num_workers=None):
        # Override the base Trainer loader setup to keep HDF5 file handles /
        # chunk caches alive across epochs -- without this each epoch re-forks
        # workers and re-opens files, which was causing 40-60ms idle gaps per
        # step (GPU util dropping to 0 periodically).
        if num_workers is None:
            num_workers = self.cfg.num_workers
        bs = (batch_size or self.cfg.batch_size)
        assert bs % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Got {bs} and {self.accelerator.num_processes}."
        )
        per_proc_bs = bs // self.accelerator.num_processes
        loader_kwargs = dict(
            batch_size=per_proc_bs,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
        )
        self.train_loader = DataLoader(self.train_set, shuffle=True, drop_last=True, **loader_kwargs)
        self.test_loader = DataLoader(self.test_set, shuffle=False, drop_last=False, **loader_kwargs)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)


@hydra.main(version_base="1.2", config_path="configs", config_name="train_thewell")
def main(cfg):
    trainer = TheWellTrainer(cfg)
    eval_loss = trainer.run()
    return eval_loss


if __name__ == "__main__":
    main()
