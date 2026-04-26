"""Standalone frozen-encoder evaluation for active_matter.

Mirrors ``reference_jepa_physics/eval_frozen_regression.py`` so results are
directly comparable to the course baseline:

    Linear probe:  ``nn.Linear`` + AdamW + MSELoss, early-stopping on val
                   (NO weight decay on output scale, standard SSL-style probe).
    kNN regression: sklearn ``KNeighborsRegressor(weights="distance")`` with
                    k grid, best k selected on val.
    Label z-score:  GLOBAL stats from reference (alpha=-3/1.41, zeta=9/5.16).
    Feature:        per-window, mean-pool encoder output over (T, V).

Usage:
    python eval_thewell.py \\
        --ckpt exp_local/2026.04.22/114527_thewell_dynamo/encoder.pt \\
        --data_dir /home_shared/grail_andre/Datasets/the_well/datasets/active_matter

Outputs JSON next to the checkpoint (or to --save_path).

Rules (PDF):
    * Frozen encoder: requires_grad=False, eval() mode.
    * Single linear layer (no MLP / attention pooling).
    * Test split used ONLY at the end, for the final number.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader, TensorDataset

from datasets.active_matter import ActiveMatterWindowDataset


# Hard-coded global stats from reference_jepa_physics/eval_frozen_regression.py
LABEL_STATS = {
    "means": np.array([-3.0, 9.0], dtype=np.float32),   # alpha, zeta
    "stds":  np.array([1.41, 5.16], dtype=np.float32),
    "names": ("alpha", "zeta"),
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s [eval] %(message)s")
log = logging.getLogger("eval_thewell")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    dataset: ActiveMatterWindowDataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (X: (N, D), y: (N, 2)) -- y is RAW (unnormalised) (alpha, zeta)."""
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    feats: List[torch.Tensor] = []
    labels: List[np.ndarray] = []
    i = 0
    for batch in loader:
        obs, _, _ = batch                          # (B, T, V, C, H, W)
        obs = obs.to(device, non_blocking=True)
        enc = encoder(obs)                         # (B, T, V, E)
        feat = enc.mean(dim=(1, 2)).detach().cpu() # (B, E)
        feats.append(feat)
        B = feat.shape[0]
        for b in range(B):
            labels.append(dataset.get_physical_params(i + b).numpy())
        i += B
    X = torch.cat(feats, dim=0).float()
    y = torch.from_numpy(np.stack(labels, axis=0)).float()
    return X, y


def zscore_labels(y: torch.Tensor) -> torch.Tensor:
    m = torch.from_numpy(LABEL_STATS["means"])
    s = torch.from_numpy(LABEL_STATS["stds"])
    return (y - m) / s


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def normalize_features(
    X_tr: torch.Tensor, X_val: torch.Tensor, X_test: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dimension z-score on train stats; apply to val/test. Standard SSL eval."""
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (X_tr - mu) / std, (X_val - mu) / std, (X_test - mu) / std, mu, std


def fit_linear_probe(
    X_tr: torch.Tensor, y_tr: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 200,
) -> Tuple[nn.Linear, List[dict]]:
    """Train a single ``nn.Linear`` head with AdamW + MSE, early-stop on val.

    Matches reference_jepa_physics/eval_frozen_regression.py::fit_linear_regressor.
    """
    in_dim = X_tr.shape[1]
    out_dim = y_tr.shape[1]
    model = nn.Linear(in_dim, out_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    history: List[dict] = []
    best_val = float("inf")
    best_state = None

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)
    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_dev), y_val_dev).item()
        history.append({
            "epoch": epoch + 1,
            "train_mse": float(np.mean(losses)),
            "val_mse": float(val_loss),
        })
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def fit_knn_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    ks: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> Tuple[KNeighborsRegressor, int, List[dict]]:
    """Grid over k, pick best val-MSE. ``weights="distance"`` like reference."""
    trials: List[dict] = []
    best_k = None
    best_val = float("inf")
    best_model = None
    for k in ks:
        k_eff = max(1, min(k, len(X_tr)))
        m = KNeighborsRegressor(n_neighbors=k_eff, weights="distance")
        m.fit(X_tr, y_tr)
        val_mse = float(((m.predict(X_val) - y_val) ** 2).mean())
        trials.append({"k": k_eff, "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_k = k_eff
            best_model = m
    assert best_model is not None and best_k is not None
    return best_model, best_k, trials


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def mse_report(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    per_dim = ((pred - target) ** 2).mean(axis=0)
    total = float(((pred - target) ** 2).mean())
    names = LABEL_STATS["names"]
    return {
        "mse": total,
        **{f"mse_{names[i]}": float(per_dim[i]) for i in range(len(names))},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to encoder.pt (saved via Trainer.save_snapshot).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Active matter root -- contains data/{train,valid,test}/")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--stride", type=int, default=None,
                        help="Windowing stride; default = window_size (non-overlapping).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--linear_epochs", type=int, default=200)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--linear_weight_decay", type=float, default=1e-4)
    parser.add_argument("--linear_batch_size", type=int, default=256)
    parser.add_argument("--knn_ks", type=int, nargs="*",
                        default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda / cpu); auto by default.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info("device=%s", device)

    log.info("loading encoder from %s", args.ckpt)
    encoder = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(encoder, nn.Module):
        raise TypeError(
            f"Expected nn.Module in {args.ckpt}, got {type(encoder)}. "
            "Are you passing an encoder.pt saved by Trainer.save_snapshot?"
        )
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval().to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    log.info("encoder params: %.2fM", n_params / 1e6)

    common_ds_kw = dict(
        data_dir=args.data_dir,
        num_frames=args.window_size,
        resolution=args.resolution,
        stride=args.stride,
        noise_std=0.0,
    )

    splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "valid", "test"):
        log.info("building %s dataset", split)
        ds = ActiveMatterWindowDataset(split=split, **common_ds_kw)
        X, y = extract_features(
            encoder, ds, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        splits[split] = (X, zscore_labels(y))
        log.info("  %s: X=%s y=%s", split, tuple(X.shape), tuple(y.shape))

    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["valid"]
    X_test, y_test = splits["test"]

    # Feature normalization: per-dim z-score on train, apply to val/test.
    # This is standard SSL eval practice — keeps AdamW stable regardless of
    # encoder output scale (our ResNet outputs norm ~33).
    X_tr, X_val, X_test, feat_mu, feat_std = normalize_features(X_tr, X_val, X_test)
    log.info("feature norm after z-score: train mean=%.3f std=%.3f",
             X_tr.mean().item(), X_tr.std().item())

    # --- Linear probe ---
    log.info("fitting linear probe (SGD nn.Linear, %d epochs)", args.linear_epochs)
    lin_model, lin_history = fit_linear_probe(
        X_tr, y_tr, X_val, y_val,
        device=device,
        lr=args.linear_lr,
        weight_decay=args.linear_weight_decay,
        batch_size=args.linear_batch_size,
        epochs=args.linear_epochs,
    )
    with torch.no_grad():
        lin_val_pred = lin_model(X_val.to(device)).cpu().numpy()
        lin_test_pred = lin_model(X_test.to(device)).cpu().numpy()
    lin_report = {
        "val": mse_report(lin_val_pred, y_val.numpy()),
        "test": mse_report(lin_test_pred, y_test.numpy()),
        "best_val_mse": min(h["val_mse"] for h in lin_history),
        "best_epoch": int(
            np.argmin([h["val_mse"] for h in lin_history]) + 1
        ),
        "hparams": {
            "lr": args.linear_lr,
            "weight_decay": args.linear_weight_decay,
            "batch_size": args.linear_batch_size,
            "epochs": args.linear_epochs,
        },
    }
    log.info("linear val MSE: %.4f (alpha=%.4f zeta=%.4f)",
             lin_report["val"]["mse"],
             lin_report["val"]["mse_alpha"],
             lin_report["val"]["mse_zeta"])
    log.info("linear test MSE: %.4f (alpha=%.4f zeta=%.4f)",
             lin_report["test"]["mse"],
             lin_report["test"]["mse_alpha"],
             lin_report["test"]["mse_zeta"])

    # --- kNN probe ---
    log.info("fitting kNN probe (ks=%s)", args.knn_ks)
    X_tr_np = X_tr.numpy()
    y_tr_np = y_tr.numpy()
    X_val_np = X_val.numpy()
    y_val_np = y_val.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    knn_model, best_k, trials = fit_knn_probe(
        X_tr_np, y_tr_np, X_val_np, y_val_np, ks=args.knn_ks,
    )
    knn_val_pred = knn_model.predict(X_val_np)
    knn_test_pred = knn_model.predict(X_test_np)
    knn_report = {
        "best_k": int(best_k),
        "trials": trials,
        "val": mse_report(knn_val_pred, y_val_np),
        "test": mse_report(knn_test_pred, y_test_np),
    }
    log.info("kNN (k=%d) val MSE: %.4f  test MSE: %.4f",
             best_k, knn_report["val"]["mse"], knn_report["test"]["mse"])

    # --- Assemble JSON report ---
    results = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "window_size": args.window_size,
        "resolution": args.resolution,
        "stride": args.stride if args.stride is not None else args.window_size,
        "label_stats": {
            "means": LABEL_STATS["means"].tolist(),
            "stds": LABEL_STATS["stds"].tolist(),
            "names": list(LABEL_STATS["names"]),
        },
        "encoder_params": int(n_params),
        "n_samples": {
            "train": int(len(X_tr)),
            "valid": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "linear": lin_report,
        "knn": knn_report,
    }
    print(json.dumps({
        "linear": {"val": results["linear"]["val"], "test": results["linear"]["test"]},
        "knn":    {"best_k": results["knn"]["best_k"],
                   "val": results["knn"]["val"], "test": results["knn"]["test"]},
    }, indent=2))

    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        save_path = Path(args.ckpt).with_name(
            Path(args.ckpt).stem + "_eval.json"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", save_path)


if __name__ == "__main__":
    main()
