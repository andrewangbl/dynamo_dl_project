"""Minimal smoke test: build DynaMo encoder+projector+SSL on random tensors,
run one forward+backward step. No dataset, no hydra, no wandb.
Run from repo root:  python smoke_test.py
"""
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from models.encoder.resnet import resnet18
from models.projector.inverse_dynamics import InverseDynamicsProjector
from models.ssl.dynamo import DynaMoSSL


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    N, T, V, C, H, W = 2, 5, 1, 3, 64, 64
    FEATURE_DIM, PROJ_DIM = 512, 64

    encoder = resnet18(pretrained=False, output_dim=FEATURE_DIM).to(device)
    projector = InverseDynamicsProjector(
        window_size=T,
        input_dim=FEATURE_DIM,
        n_layer=2,
        n_head=4,
        n_embd=128,
        output_dim=PROJ_DIM,
        dropout=0.0,
    ).to(device)

    ssl = DynaMoSSL(
        encoder=encoder,
        projector=projector,
        window_size=T,
        feature_dim=FEATURE_DIM,
        projection_dim=PROJ_DIM,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        covariance_reg_coef=0.04,
        dynamics_loss_coef=1.0,
        ema_beta=None,
        projector_use_ema=False,
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        separate_single_views=True,
    )

    obs = torch.rand(N, T, V, C, H, W, device=device)
    print(f"[smoke] input obs shape = {tuple(obs.shape)}")

    obs_enc, obs_proj, total_loss, parts = ssl(obs)
    print(f"[smoke] obs_enc  = {tuple(obs_enc.shape)}")
    print(f"[smoke] obs_proj = {tuple(obs_proj.shape)}")
    print(f"[smoke] total_loss = {total_loss.item():.4f}")
    for k, v in parts.items():
        if torch.is_tensor(v):
            print(f"        {k}: {v.item():.4f}")

    total_loss.backward()
    ssl.step()
    assert torch.isfinite(total_loss), "loss is not finite"
    print("[smoke] forward + backward + step OK")


if __name__ == "__main__":
    main()
