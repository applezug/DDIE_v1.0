#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCAR (Missing Completely At Random) simulation for DDI-E experiments.
"""

import numpy as np


def generate_mcar_mask(shape, missing_rate, seed=None):
    """
    Generate MCAR mask. 1 = observed, 0 = missing.
    
    Args:
        shape: (N, L) or (N, L, C) - samples x length x channels
        missing_rate: float in [0, 1], probability of missing
        seed: random seed
    
    Returns:
        mask: same shape, 1=observed 0=missing
    """
    if seed is not None:
        np.random.seed(seed)
    if len(shape) == 2:
        M = (np.random.rand(*shape) > missing_rate).astype(np.float32)
    else:
        M = (np.random.rand(shape[0], shape[1]) > missing_rate).astype(np.float32)
        if len(shape) == 3:
            M = np.expand_dims(M, axis=-1)
    return M


def apply_mask(data, mask, fill_value=np.nan):
    """
    Apply mask to data: set missing positions to fill_value.
    data, mask: (N, L) or (N, L, C)
    """
    masked = data.copy()
    if mask.ndim == 2 and data.ndim == 3:
        mask = np.expand_dims(mask, axis=-1)
    masked[mask < 0.5] = fill_value
    return masked


def generate_multiple_masks(shape, missing_rate, n_masks=5, base_seed=42):
    """Generate n_masks different random masks."""
    masks = []
    for i in range(n_masks):
        m = generate_mcar_mask(shape, missing_rate, seed=base_seed + i * 1000)
        masks.append(m)
    return masks
