"""Evaluation metrics for DDI-E experiments"""

import numpy as np


def compute_imputation_metrics(original, imputed, mask):
    """
    Compute MAE, RMSE, MAPE only at missing positions.
    mask: 1=observed, 0=missing. We evaluate on missing (mask==0).
    """
    # mask: 1=observed, 0=missing. 在缺失位置评估插补误差
    missing = (mask < 0.5)
    if original.ndim == 3 and mask.ndim == 2:
        missing = np.expand_dims(missing, axis=-1)
    orig_flat = np.array(original).flatten()
    imp_flat = np.array(imputed).flatten()
    mask_flat = np.array(missing).flatten()
    idx = mask_flat  # True = 缺失位置，只在这些位置算 MAE/RMSE/MAPE
    if idx.sum() == 0:
        return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0}
    o, p = orig_flat[idx], imp_flat[idx]
    mae = np.mean(np.abs(p - o))
    rmse = np.sqrt(np.mean((p - o) ** 2))
    valid = np.abs(o) > 1e-8
    mape = np.mean(np.abs((p[valid] - o[valid]) / o[valid])) * 100 if valid.any() else 0.0
    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}


def compute_uncertainty_metrics(gt, samples, mask):
    """
    samples: (N_s, ...) - N_s samples per test point
    Returns: coverage (%), mean interval width
    """
    missing = (mask < 0.5)
    if mask.ndim == 2 and gt.ndim == 3:
        missing = np.broadcast_to(missing[:, :, np.newaxis], gt.shape)
    mu = np.mean(samples, axis=0)
    sigma = np.std(samples, axis=0) + 1e-8
    lo, hi = mu - 1.96 * sigma, mu + 1.96 * sigma
    gt_a = np.array(gt)
    inside = (gt_a >= lo) & (gt_a <= hi) & missing
    total = missing.sum()
    coverage = 100.0 * inside.sum() / total if total > 0 else 0.0
    width = np.mean(2 * 1.96 * sigma[missing]) if total > 0 else 0.0
    return {'coverage': coverage, 'mean_interval_width': width}


def phm_score(pred_rul, true_rul):
    """PHM 2008 Challenge scoring function."""
    d = np.array(pred_rul) - np.array(true_rul)
    pos = d >= 0
    neg = ~pos
    s = np.sum(np.exp(d[neg] / 13) - 1) + np.sum(np.exp(d[pos] / 10) - 1)
    return float(s)
