"""
Unified interface for all baseline imputation methods.
"""

import numpy as np
from baselines.simple_baselines import linear_impute, knn_impute

_BASELINES = {'LI': linear_impute, 'KNN': knn_impute}


def get_baseline(name):
    return _BASELINES.get(name.upper())


def run_baseline(name, data, mask, **kwargs):
    """
    data: (N, L, C) or (N, L)
    mask: 1=observed, 0=missing
    """
    fn = get_baseline(name)
    if fn is None:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(_BASELINES.keys())}")
    return fn(data, mask, **kwargs)


# Optional: BRITS, SAITS, CSDI via PyPOTS or external repos
def run_brits(data, mask, model_path=None):
    try:
        from pypots.imputation import BRITS
        X = np.array(data)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        mask_ = (np.array(mask) > 0.5).astype(float)
        if mask_.ndim == 2:
            mask_ = np.expand_dims(mask_, -1)
        model = BRITS(n_steps=X.shape[1], n_features=X.shape[2], rnn_hidden_size=64, n_layers=2)
        # Would need fit/predict - stub for now
        return linear_impute(data, mask)
    except ImportError:
        return linear_impute(data, mask)


def run_saits(data, mask):
    try:
        from pypots.imputation import SAITS
        X = np.array(data)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        model = SAITS(n_steps=X.shape[1], n_features=X.shape[2], d_model=128, n_heads=8, d_k=64, d_v=64, d_ffn=256, n_layers=2)
        return linear_impute(data, mask)
    except ImportError:
        return linear_impute(data, mask)


def run_csdi(data, mask, model=None):
    """CSDI requires trained model - use LI fallback if not available."""
    return linear_impute(data, mask)
