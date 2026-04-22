"""Smoke test: ActiveMatterWindowDataset + ResNet18Physics + DynaMoSSL forward.

Verifies the whole stack end-to-end without Hydra / accelerate, using tiny
batch sizes so it runs on CPU or a single GPU in a few seconds.
"""

from __future__ import annotations

import time
import torch
from torch.utils.data import DataLoader

from datasets.active_matter import ActiveMatterWindowDataset, build_active_matter_train_val
from models.encoder.resnet_physics import resnet18_physics
from models.projector.inverse_dynamics import InverseDynamicsProjector
from models.ssl.dynamo import DynaMoSSL


DATA_DIR = "/home_shared/grail_andre/Datasets/the_well/datasets/active_matter"
WINDOW = 8          # keep tiny for CPU smoke test
RES = 64            # ditto


def banner(msg):
    print("\n" + "=" * 8 + f" {msg} " + "=" * 8)


banner("1. ActiveMatterWindowDataset (train, non-overlapping)")
t0 = time.time()
ds = ActiveMatterWindowDataset(
    data_dir=DATA_DIR,
    num_frames=WINDOW,
    split="train",
    resolution=RES,
    stride=None,          # == WINDOW
    noise_std=0.0,
)
print(f"  built in {time.time() - t0:.2f}s")
print(f"  len = {len(ds)}  (175 trajs * floor(81/{WINDOW}) = {175 * (81 // WINDOW)})")

t0 = time.time()
obs, actions, mask = ds[0]
print(f"  __getitem__[0] took {time.time() - t0:.3f}s")
print(f"  obs:  shape={tuple(obs.shape)}, dtype={obs.dtype}, "
      f"range=[{obs.min().item():+.3f}, {obs.max().item():+.3f}]")
print(f"  actions: {tuple(actions.shape)}, mask: {tuple(mask.shape)} (all True={bool(mask.all())})")
print(f"  physical_params (alpha, zeta?): {ds.get_physical_params(0).tolist()}")


banner("2. Official (train, valid) splits")
train_ds, val_ds = build_active_matter_train_val(
    data_dir=DATA_DIR, num_frames=WINDOW, resolution=RES
)
print(f"  train: {len(train_ds)} windows    valid: {len(val_ds)} windows")


banner("3. DataLoader (batch_size=2, num_workers=0)")
loader = DataLoader(train_ds, batch_size=2, num_workers=0, shuffle=True)
obs, actions, mask = next(iter(loader))
print(f"  batch obs: {tuple(obs.shape)}   (N=2, T={WINDOW}, V=1, C=11, {RES}, {RES})")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[device = {device}]")
obs = obs.to(device)

banner("4. ResNet18Physics on (N, T, V, C, H, W)")
encoder = resnet18_physics(in_channels=11, output_dim=512).to(device)
n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
print(f"  params: {n_params:.2f}M")
with torch.no_grad():
    feat = encoder(obs)
print(f"  out:   {tuple(feat.shape)}  (expect (2, {WINDOW}, 1, 512))")


banner("5. Full DynaMoSSL forward")
projector = InverseDynamicsProjector(
    window_size=WINDOW,
    input_dim=512,
    n_layer=2,
    n_head=2,
    n_embd=64,
    output_dim=16,
    dropout=0.0,
).to(device)
ssl = DynaMoSSL(
    encoder=encoder,
    projector=projector,
    window_size=WINDOW,
    feature_dim=512,
    projection_dim=16,
    n_layer=2,
    n_head=2,
    n_embd=64,
    dropout=0.0,
    covariance_reg_coef=0.04,
    dynamics_loss_coef=1.0,
    ema_beta=0.99,
    beta_scheduling=True,
    projector_use_ema=True,
    lr=1e-4,
    weight_decay=0.0,
    betas=(0.9, 0.999),
    separate_single_views=True,
)
obs_enc, obs_proj, loss, loss_components = ssl.forward(obs)
print(f"  obs_enc:  {tuple(obs_enc.shape)}   (N, T, V, E=512)")
print(f"  obs_proj: {tuple(obs_proj.shape)}  (N, T, V, Z=16)")
print(f"  loss: {loss.item():.4f}")
for k, v in loss_components.items():
    vv = v.item() if hasattr(v, "item") else float(v)
    print(f"    {k}: {vv:.4f}")

print("\nAll smoke tests passed.")
