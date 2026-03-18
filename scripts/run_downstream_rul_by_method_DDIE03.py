#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute downstream RUL metrics by imputation method (LI / KNN / DDI-E) for DDIE03.

Goal: produce real data for Figure 4 (grouped bars: RMSE + PHM Score).

Pipeline (NASA battery, test=B0018 windows):
1) Create MCAR missing mask at a fixed rate (default 50%) for N masks per seed.
2) Impute with LI, KNN, and DDI-E (DDIE03 checkpoint).
3) Build the same RUL dataset (last m points) and evaluate the same LSTM for each method.
4) Report mean±std across masks (and seeds) for each method.

Outputs:
- results_DDIE03/downstream_rul_by_method_DDIE03.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from Utils.io_utils import instantiate_from_config, load_yaml_config  # noqa: E402
from Data.build_dataloader import build_dataloader  # noqa: E402
from downstream_rul import build_rul_dataset, evaluate_rul, train_rul_model  # noqa: E402


def _to_serializable(obj):
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _denormalize_if_available(dataset, arr: np.ndarray) -> np.ndarray:
    if hasattr(dataset, "denormalize"):
        try:
            return dataset.denormalize(arr)
        except Exception:
            return arr
    return arr


def make_mcar_mask(shape: Tuple[int, int], missing_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # mask: 1 observed, 0 missing
    m = (rng.random(shape) > missing_rate).astype(np.float32)
    return m


def impute_li(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """x: (N,L,1), mask: (N,L)"""
    n, L, _ = x.shape
    out = x.copy()
    for i in range(n):
        xi = out[i, :, 0]
        mi = mask[i].astype(bool)
        if mi.sum() < 2:
            # not enough points: fallback to zero/nearest
            out[i, :, 0] = np.where(mi, xi, xi[mi][0] if mi.any() else 0.0)
            continue
        t_obs = np.where(mi)[0]
        v_obs = xi[mi]
        f = interp1d(t_obs, v_obs, kind="linear", fill_value="extrapolate", bounds_error=False)
        xi_imp = f(np.arange(L))
        out[i, :, 0] = np.where(mi, xi, xi_imp)
    return out


def impute_knn(x: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Simple KNN imputer over each sequence independently (feature=1)."""
    n, L, _ = x.shape
    out = x.copy()
    imputer = KNNImputer(n_neighbors=k, weights="distance")
    for i in range(n):
        xi = out[i, :, 0].copy()
        xi[mask[i] < 0.5] = np.nan
        xi_imp = imputer.fit_transform(xi.reshape(-1, 1)).reshape(-1)
        out[i, :, 0] = np.where(mask[i] > 0.5, out[i, :, 0], xi_imp)
    return out


@torch.no_grad()
def impute_ddie(
    model,
    x: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    eval_batch_size: int = 16,
    seed: int | None = None,
) -> np.ndarray:
    """DDI-E infill using model.fast_sample_infill (same as run_experiments_DDIE03)."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    n, L, C = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    n_batches = (n + eval_batch_size - 1) // eval_batch_size
    for bi in range(n_batches):
        start = bi * eval_batch_size
        end = min((bi + 1) * eval_batch_size, n)
        xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
        mb = torch.tensor(mask[start:end], dtype=torch.float32, device=device).unsqueeze(-1)
        target = xb.clone()
        target[mb.expand_as(xb) < 0.5] = 0.0
        pred = model.fast_sample_infill(xb.shape, target, mb, clip_denoised=True)
        out[start:end] = pred.detach().cpu().numpy()
    return out


def compute_rul_metrics_for_Xy(
    X: np.ndarray,
    y: np.ndarray,
    split_idx: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    tr_idx, te_idx = split_idx
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    model = train_rul_model(X_tr, y_tr, epochs=epochs, lr=lr)
    metrics = evaluate_rul(model, X_te, y_te)
    return {"RMSE": float(metrics["RMSE"]), "PHM_Score": float(metrics["PHM_Score"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/nasa_battery_DDIE03.yaml")
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--n_masks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="base seed; each mask uses seed+mi")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="DDI-E inference batch size (lower for 4GB GPUs)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:0 ... (default: auto)")
    parser.add_argument("--rul_epochs", type=int, default=50)
    parser.add_argument("--rul_lr", type=float, default=1e-3)
    parser.add_argument("--m_last", type=int, default=10, help="use last m points as RUL features")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test windows (B0018) from dataset (same as DDIE03 imputation eval)
    test = build_dataloader(config, "test")
    dataset = test["dataset"]

    # dataset[i] may be tensor or dict depending on dataset implementation
    xs = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict) and "data" in item:
            xi = item["data"]
        else:
            xi = item
        xi = np.array(xi).reshape(-1)  # (L,)
        xs.append(xi)

    data = np.stack(xs, axis=0).astype(np.float32)  # (N,L)
    N, L = data.shape
    data3 = data[:, :, None]  # (N,L,1)

    # Labels: follow existing DDIE03 downstream script, but compute from COMPLETE sequence (no missing)
    rul_labels = [(1.0 - float(seq.mean())) * 100 for seq in data]
    rul_labels = [max(0.0, r) for r in rul_labels]

    # Baseline (complete data)
    X_base, y_base = build_rul_dataset([d for d in data], rul_labels, m=args.m_last)
    if len(X_base) < 3:
        out_path = ROOT / "results_DDIE03" / "downstream_rul_by_method_DDIE03.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "skipped", "reason": "too_few_samples", "n": int(len(X_base))}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(out_path)
        return 0

    # Use the SAME split for all methods
    idx_all = np.arange(len(X_base))
    tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=42)
    split_idx = (tr_idx, te_idx)

    # Load DDI-E model checkpoint
    ckpt_path = Path(config["solver"]["results_folder"]) / "best.pt"
    model = None
    if ckpt_path.exists():
        model = instantiate_from_config(config["model"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

    # Collect metrics across masks
    methods = ["LI", "KNN", "DDI-E"]
    metrics_runs: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

    for mi in range(args.n_masks):
        mask_seed = args.seed + mi
        mask = make_mcar_mask((N, L), args.missing_rate, seed=mask_seed)

        x_obs3 = data3.copy()
        x_obs3[mask < 0.5, 0] = 0.0

        # LI
        imp_li = impute_li(x_obs3, mask)
        # KNN
        imp_knn = impute_knn(x_obs3, mask, k=5)
        # DDI-E (if checkpoint exists); else skip
        imp_ddie = None
        if model is not None:
            imp_ddie = impute_ddie(model, x_obs3, mask, device=device, eval_batch_size=args.eval_batch_size, seed=mask_seed)

        # For RUL, we keep features in the same scale as dataset provides (normalized).
        # Use last m points.
        X_li, y = build_rul_dataset([imp_li[i, :, 0] for i in range(N)], rul_labels, m=args.m_last)
        X_knn, _ = build_rul_dataset([imp_knn[i, :, 0] for i in range(N)], rul_labels, m=args.m_last)

        metrics_runs["LI"].append(compute_rul_metrics_for_Xy(X_li, y, split_idx, epochs=args.rul_epochs, lr=args.rul_lr))
        metrics_runs["KNN"].append(compute_rul_metrics_for_Xy(X_knn, y, split_idx, epochs=args.rul_epochs, lr=args.rul_lr))

        if imp_ddie is not None:
            X_ddie, _ = build_rul_dataset([imp_ddie[i, :, 0] for i in range(N)], rul_labels, m=args.m_last)
            metrics_runs["DDI-E"].append(compute_rul_metrics_for_Xy(X_ddie, y, split_idx, epochs=args.rul_epochs, lr=args.rul_lr))

    # Aggregate mean/std
    out = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "missing_rate": args.missing_rate,
        "n_masks": args.n_masks,
        "seed_base": args.seed,
        "eval_batch_size": args.eval_batch_size,
        "device": str(device),
        "rul": {
            "epochs": args.rul_epochs,
            "lr": args.rul_lr,
            "m_last": args.m_last,
            "split": {"train_n": int(len(tr_idx)), "test_n": int(len(te_idx)), "random_state": 42},
        },
        "baseline_complete": compute_rul_metrics_for_Xy(X_base, y_base, split_idx, epochs=args.rul_epochs, lr=args.rul_lr),
        "by_method": {},
        "runs": metrics_runs,
        "note": "Metrics are computed by running the same downstream LSTM on features built from last m points of each (imputed) window. Baseline is complete data (no missing).",
    }

    for m in methods:
        arr_rmse = np.array([r["RMSE"] for r in metrics_runs[m]], dtype=np.float64) if metrics_runs[m] else np.array([])
        arr_phm = np.array([r["PHM_Score"] for r in metrics_runs[m]], dtype=np.float64) if metrics_runs[m] else np.array([])
        out["by_method"][m] = {
            "RMSE_mean": float(arr_rmse.mean()) if arr_rmse.size else None,
            "RMSE_std": float(arr_rmse.std(ddof=0)) if arr_rmse.size else None,
            "PHM_Score_mean": float(arr_phm.mean()) if arr_phm.size else None,
            "PHM_Score_std": float(arr_phm.std(ddof=0)) if arr_phm.size else None,
            "n_runs": int(arr_rmse.size),
        }

    out_path = ROOT / "results_DDIE03" / "downstream_rul_by_method_DDIE03.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_serializable(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

