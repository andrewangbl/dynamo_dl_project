import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    get_train_dataloader_from_cfg,
    get_val_dataloader_from_cfg,
    get_test_dataloader_from_cfg,
)
from .model import get_model_and_loss_cnn
from .utils.data_utils import normalize_labels
from .utils.hydra import compose


LABEL_STATS = {
    "active_matter": {
        "means": [-3.0, 9.0],  # alpha, zeta
        "stds": [1.41, 5.16],
        "names": ["alpha", "zeta"],
    },
    "shear_flow": {
        "means": [4.85, 2.69],
        "stds": [0.61, 3.38],
        "compression": ["log", None],
        "names": ["rayleigh", "schmidt"],
    },
    "rayleigh_benard": {
        "means": [2.69, 8.0],
        "stds": [3.38, 1.41],
        "compression": [None, "log"],
        "names": ["prandtl", "rayleigh"],
    },
}


def pool_encoder_output(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        return x.mean(dim=(2, 3, 4))
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 2:
        return x
    raise ValueError(f"Unexpected encoder output shape: {tuple(x.shape)}")


def build_encoder(cfg, checkpoint_path: str | None, device: torch.device):
    encoder, _, _ = get_model_and_loss_cnn(
        cfg.model.dims,
        cfg.model.num_res_blocks,
        cfg.dataset.num_frames,
        in_chans=cfg.dataset.num_chans,
    )
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(device)
    return encoder


@torch.no_grad()
def extract_embeddings(loader, encoder, label_stats, device):
    all_x = []
    all_y = []
    for batch in loader:
        ctx = batch["context"].to(device)
        if ctx.shape[2] < 4:
            ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))
        labels = normalize_labels(batch["physical_params"], stats=label_stats).float()
        enc = encoder(ctx)
        pooled = pool_encoder_output(enc).cpu()
        all_x.append(pooled)
        all_y.append(labels.cpu())
    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)


def mse_report(pred: np.ndarray, target: np.ndarray, names: list[str]):
    per_dim = ((pred - target) ** 2).mean(axis=0)
    report = {"mse": float(((pred - target) ** 2).mean())}
    for idx, name in enumerate(names):
        report[f"mse_{name}"] = float(per_dim[idx])
    return report


def fit_linear_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
):
    model = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_x.float(), train_y.float()),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x.float().to(device))
            val_loss = loss_fn(val_pred, val_y.float().to(device)).item()
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(np.mean(epoch_losses)),
                "val_mse": float(val_loss),
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def fit_knn_regressor(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, ks: list[int]):
    best_model = None
    best_k = None
    best_val = float("inf")
    trials = []
    for k in ks:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(train_x, train_y)
        val_pred = model.predict(val_x)
        val_mse = float(((val_pred - val_y) ** 2).mean())
        trials.append({"k": k, "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_model = model
            best_k = k
    return best_model, best_k, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--linear_epochs", type=int, default=100)
    parser.add_argument("--linear_lr", type=float, default=1e-3)
    parser.add_argument("--linear_weight_decay", type=float, default=1e-4)
    parser.add_argument("--knn_k", type=int, nargs="*", default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)

    if cfg.dataset.name not in LABEL_STATS:
        raise ValueError(f"No label stats registered for dataset {cfg.dataset.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(cfg, args.trained_model_path, device)
    label_stats = LABEL_STATS[cfg.dataset.name]

    train_loader = get_train_dataloader_from_cfg(cfg, stage="ft")
    val_loader = get_val_dataloader_from_cfg(cfg, stage="ft")
    test_loader = get_test_dataloader_from_cfg(cfg, stage="ft")

    train_x, train_y = extract_embeddings(train_loader, encoder, label_stats, device)
    val_x, val_y = extract_embeddings(val_loader, encoder, label_stats, device)
    test_x, test_y = extract_embeddings(test_loader, encoder, label_stats, device)

    linear_model, linear_history = fit_linear_regressor(
        train_x,
        train_y,
        val_x,
        val_y,
        lr=args.linear_lr,
        weight_decay=args.linear_weight_decay,
        batch_size=cfg.ft.batch_size,
        epochs=args.linear_epochs,
        device=device,
    )
    with torch.no_grad():
        linear_val_pred = linear_model(val_x.float().to(device)).cpu().numpy()
        linear_test_pred = linear_model(test_x.float().to(device)).cpu().numpy()

    train_x_np = train_x.numpy()
    train_y_np = train_y.numpy()
    val_x_np = val_x.numpy()
    val_y_np = val_y.numpy()
    test_x_np = test_x.numpy()
    test_y_np = test_y.numpy()

    knn_model, best_k, knn_trials = fit_knn_regressor(
        train_x_np,
        train_y_np,
        val_x_np,
        val_y_np,
        ks=args.knn_k,
    )
    knn_val_pred = knn_model.predict(val_x_np)
    knn_test_pred = knn_model.predict(test_x_np)

    results = {
        "dataset": cfg.dataset.name,
        "checkpoint": args.trained_model_path,
        "embedding_shape": list(train_x.shape[1:]),
        "encoder_input": {
            "channels": int(cfg.dataset.num_chans),
            "frames": int(cfg.dataset.num_frames),
            "resolution": int(cfg.dataset.resolution),
        },
        "label_names": label_stats["names"],
        "linear": {
            "val": mse_report(linear_val_pred, val_y_np, label_stats["names"]),
            "test": mse_report(linear_test_pred, test_y_np, label_stats["names"]),
            "history": linear_history,
        },
        "knn": {
            "best_k": best_k,
            "val": mse_report(knn_val_pred, val_y_np, label_stats["names"]),
            "test": mse_report(knn_test_pred, test_y_np, label_stats["names"]),
            "trials": knn_trials,
        },
    }

    print(json.dumps(results, indent=2))

    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        save_path = Path(cfg.out_path) / f"{cfg.dataset.name}_frozen_regression_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
