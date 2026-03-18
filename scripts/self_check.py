#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal self-check (no datasets required).

This script verifies that:
  - imports work
  - the DDI-E model can run a forward pass on synthetic data

Run:
  python scripts/self_check.py
"""
from __future__ import annotations

import os
import sys

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: torch.\n"
        "Please install dependencies first:\n"
        "  pip install -r requirements.txt\n"
        "\n"
        "If you are on Windows and pip cannot resolve a compatible wheel, install PyTorch\n"
        "from the official selector first, then re-run the command above.\n"
    ) from e


def main() -> None:
    # Ensure project root is on sys.path when running from repo root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)

    from Models.ddie.ddie_model import DDI_E

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    b, t, c = 2, 128, 1
    x = torch.randn(b, t, c, device=device)
    mask = (torch.rand(b, t, 1, device=device) > 0.3).float()

    model = DDI_E(
        seq_length=t,
        feature_size=c,
        n_layer_enc=2,
        n_layer_dec=1,
        d_model=64,
        timesteps=50,
        sampling_timesteps=10,
        beta_schedule="cosine",
        n_heads=4,
        mlp_hidden_times=2,
        use_mask_condition=True,
        freq_loss_weight=0.0,
        use_trend_cycle=False,
    ).to(device)

    model.train()
    loss = model(x, mask=mask)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Self-check failed: non-finite loss: {loss}")

    model.eval()
    with torch.no_grad():
        x_obs = x.clone()
        x_obs[mask < 0.5] = 0.0
        y = model.fast_sample_infill(x.shape, x_obs, mask, clip_denoised=True)
        if y.shape != x.shape:
            raise RuntimeError(f"Self-check failed: output shape {y.shape} != {x.shape}")

    print("Self-check OK")
    print("  device:", device)
    print("  loss:", float(loss.detach().cpu()))
    print("  sample output mean/std:", float(y.mean().cpu()), float(y.std().cpu()))


if __name__ == "__main__":
    main()

