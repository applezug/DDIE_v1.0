"""
Simple baseline imputation: Linear Interpolation (LI) and K-Nearest Neighbors (KNN)
"""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer


def linear_impute(data, mask):
    """
    data: (N, L) or (N, L, C)
    mask: 1=observed, 0=missing
    """
    data = np.array(data, dtype=np.float64)
    mask = np.array(mask)
    if data.ndim == 3:
        out = np.zeros_like(data)
        for i in range(data.shape[0]):
            for c in range(data.shape[2]):
                out[i, :, c] = _linear_1d(data[i, :, c], mask[i] if mask.ndim == 3 else mask[i, :])
        return out
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        out[i] = _linear_1d(data[i], mask[i] if mask.ndim == 2 else mask[i, :])
    return out


def _linear_1d(x, m):
    x = np.array(x).flatten()
    m = np.array(m).flatten()
    if m.ndim > 1:
        m = m[:, 0]
    obs = np.where(m > 0.5)[0]
    mis = np.where(m < 0.5)[0]
    if len(obs) < 2 or len(mis) == 0:
        return x.copy()
    f = interp1d(obs, x[obs], kind='linear', bounds_error=False, fill_value=(x[obs[0]], x[obs[-1]]))
    out = x.copy()
    out[mis] = f(mis)
    return out


def knn_impute(data, mask, n_neighbors=10):
    """
    data: (N, L) or (N, L*C) - flatten if 3D
    mask: 1=observed, 0=missing
    """
    data = np.array(data, dtype=np.float64)
    mask = np.array(mask)
    orig_shape = data.shape
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1) if mask.ndim == 3 else np.broadcast_to(mask, (data.shape[0], orig_shape[1] * orig_shape[2]))
    masked = data.copy()
    masked[mask < 0.5] = np.nan
    n_neighbors_use = max(1, min(n_neighbors, data.shape[0] - 1))
    imp = KNNImputer(n_neighbors=n_neighbors_use)
    out = imp.fit_transform(masked)
    # 保证与 data 同形状（个别 sklearn 版本可能返回不同列数）
    if out.shape != data.shape:
        out_full = np.zeros_like(data)
        c = min(out.shape[1], data.shape[1])
        out_full[:, :c] = out[:, :c]
        if out.shape[1] < data.shape[1]:
            out_full[:, c:] = data[:, c:]
        out = out_full
    out = np.where(mask > 0.5, data, out)
    return out.reshape(orig_shape)
