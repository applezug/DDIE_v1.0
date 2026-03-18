#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NASA Lithium-Ion Battery Dataset for DDI-E
Extracts capacity degradation sequences from NASA Ames Prognostics Data Repository.
Batteries B0005, B0006, B0007 (train/val), B0018 (test)
"""

import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy import io


def extract_capacity_from_mat(mat_path):
    """
    Extract capacity (Ah) per discharge cycle from NASA battery .mat file.
    Supports both nested structure and flat structure formats.
    
    Returns:
        capacity_list: list of capacity values, one per discharge cycle
    """
    try:
        data = io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        raise IOError(f"Cannot load {mat_path}: {e}")
    
    capacity_list = []
    
    # Try common NASA battery mat formats
    # Format 1: Top-level 'cycle' array
    if 'cycle' in data:
        cycle_struct = data['cycle']
        if np.isscalar(cycle_struct):
            cycle_struct = [cycle_struct]
        for i, cyc in enumerate(cycle_struct):
            if hasattr(cyc, 'type') and 'discharge' in str(cyc.type).lower():
                if hasattr(cyc, 'data') and hasattr(cyc.data, 'Capacity'):
                    cap = np.array(cyc.data.Capacity).flatten()
                    if cap.size > 0:
                        capacity_list.append(float(cap[-1]))  # Last value is final capacity
            elif hasattr(cyc, 'data'):
                # Sometimes type is in data
                dd = cyc.data
                if hasattr(dd, 'Capacity'):
                    cap = np.array(dd.Capacity).flatten()
                    if cap.size > 0:
                        capacity_list.append(float(cap[-1]))
    
    # Format 2: Direct 'Capacity' or similar
    for key in ['Capacity', 'capacity']:
        if key in data and not key.startswith('_'):
            arr = np.array(data[key]).flatten()
            if arr.size > 0:
                capacity_list = arr.tolist()
                break
    
    # Format 3: B0005, B0006 style - nested under battery name
    for k, v in data.items():
        if k.startswith('B') and not k.startswith('__'):
            if hasattr(v, 'cycle'):
                cycle_struct = v.cycle
                if np.isscalar(cycle_struct):
                    cycle_struct = [cycle_struct]
                for cyc in cycle_struct:
                    if hasattr(cyc, 'data') and hasattr(cyc.data, 'Capacity'):
                        cap = np.array(cyc.data.Capacity).flatten()
                        if cap.size > 0:
                            capacity_list.append(float(cap[-1]))
                if capacity_list:
                    break
    
    if not capacity_list:
        # Fallback: look for any numeric array that could be capacity
        for k, v in data.items():
            if k.startswith('__'):
                continue
            arr = np.array(v)
            if arr.ndim <= 2 and arr.size > 10 and np.issubdtype(arr.dtype, np.number):
                flat = arr.flatten()
                if flat.min() > 0 and flat.max() < 10:  # Capacity in Ah range
                    capacity_list = flat.tolist()
                    break
    
    return capacity_list


def load_nasa_battery_sequences(data_root, battery_ids):
    """
    Load capacity sequences for multiple batteries.
    
    Returns:
        dict: {battery_id: np.array of capacity values}
        global_min, global_max: for normalization across train set
    """
    all_caps = []
    sequences = {}
    
    for bid in battery_ids:
        # Try multiple naming conventions
        candidates = [
            os.path.join(data_root, f"{bid}.mat"),
            os.path.join(data_root, f"{bid}.MAT"),
            os.path.join(data_root, bid, "data.mat"),
        ]
        found = False
        for path in candidates:
            if os.path.isfile(path):
                cap_list = extract_capacity_from_mat(path)
                if cap_list:
                    arr = np.array(cap_list, dtype=np.float64)
                    sequences[bid] = arr
                    all_caps.extend(arr.tolist())
                    found = True
                    print(f"  Loaded {bid}: {len(arr)} cycles")
                    break
        if not found:
            print(f"  Warning: No data found for {bid}, creating synthetic placeholder")
            # Synthetic capacity degradation for testing
            n_cycles = 168 if bid != 'B0018' else 132
            t = np.arange(n_cycles, dtype=float)
            arr = 2.0 - 0.6 * (t / n_cycles) ** 1.2 + np.random.RandomState(42 + ord(bid[-1])).randn(n_cycles) * 0.02
            arr = np.clip(arr, 1.2, 2.1)
            sequences[bid] = arr
            all_caps.extend(arr.tolist())
    
    global_min = float(np.min(all_caps)) if all_caps else 1.0
    global_max = float(np.max(all_caps)) if all_caps else 2.0
    return sequences, global_min, global_max


class NASABatteryDataset(Dataset):
    """
    NASA battery capacity degradation dataset.
    - Train: B0005, B0006, B0007
    - Val: 20% of train indices (by sequence position)
    - Test: B0018
    - Window length L=128, stride=1, forward padding if sequence < 128
    - 测试集归一化使用训练集的 global_min/global_max，与训练分布一致（见 Report 20260205）。
    """
    
    def __init__(
        self,
        data_root,
        battery_ids=None,
        window=128,
        seed=42,
        period='train',  # 'train', 'val', 'test'
        val_ratio=0.2,
        neg_one_to_one=True,
        train_battery_ids_for_norm=None,  # LOOCV: 测试集归一化用到的训练电池列表，仅 period=='test' 时有效
    ):
        super().__init__()
        self.data_root = data_root
        self.battery_ids = battery_ids or ['B0005', 'B0006', 'B0007']
        self.window = window
        self.seed = seed
        self.period = period
        self.val_ratio = val_ratio
        self.neg_one_to_one = neg_one_to_one
        self.train_battery_ids_for_norm = train_battery_ids_for_norm

        # Determine train vs test batteries
        if period == 'test' and train_battery_ids_for_norm is not None:
            train_ids = list(train_battery_ids_for_norm)
            test_ids = list(self.battery_ids) if self.battery_ids else ['B0018']
        else:
            train_ids = [b for b in self.battery_ids if b != 'B0018']
            if not train_ids:
                train_ids = ['B0005', 'B0006', 'B0007']
            test_ids = [b for b in self.battery_ids if b == 'B0018']
            if not test_ids:
                test_ids = ['B0018']

        if period == 'test':
            # 测试集必须使用训练集的 global_min/max 归一化，与 20260205 报告一致
            _, self.global_min, self.global_max = load_nasa_battery_sequences(data_root, train_ids)
            self.sequences, _, _ = load_nasa_battery_sequences(data_root, test_ids if test_ids else ['B0018'])
        else:
            load_ids = train_ids
            self.sequences, self.global_min, self.global_max = load_nasa_battery_sequences(data_root, load_ids)
        
        if not self.sequences:
            raise ValueError("No battery data loaded. Check data_root and battery_ids.")
        
        # Build window samples: (N, L, 1)
        self.samples = []
        self.battery_indices = []
        self.cycle_indices = []
        
        for bid, cap_arr in self.sequences.items():
            seq = cap_arr.reshape(-1, 1)
            n = seq.shape[0]
            if n < window:
                # Forward fill
                first_val = seq[0:1]
                pad = np.repeat(first_val, window - n, axis=0)
                seq = np.concatenate([seq, pad], axis=0)
                n = window
            
            for i in range(n - window + 1):
                win = seq[i:i + window]
                self.samples.append(win)
                self.battery_indices.append(bid)
                self.cycle_indices.append(i)
        
        self.samples = np.array(self.samples, dtype=np.float32)
        
        # Min-Max normalize on [0,1] then optionally to [-1,1]
        self.samples = (self.samples - self.global_min) / (self.global_max - self.global_min + 1e-8)
        self.samples = np.clip(self.samples, 0, 1)
        
        if neg_one_to_one:
            self.samples = self.samples * 2 - 1
        
        # Train/Val split: 20% of positions (by global index)
        if period in ['train', 'val'] and len(load_ids) > 0 and 'B0018' not in load_ids:
            np.random.seed(seed)
            n_total = len(self.samples)
            indices = np.random.permutation(n_total)
            n_val = int(n_total * val_ratio)
            val_idx = set(indices[:n_val])
            if period == 'train':
                keep = [i for i in range(n_total) if i not in val_idx]
            else:
                keep = [i for i in range(n_total) if i in val_idx]
            self.samples = self.samples[keep]
            self.battery_indices = [self.battery_indices[i] for i in keep]
            self.cycle_indices = [self.cycle_indices[i] for i in keep]
        
        self.sample_num = len(self.samples)
    
    def get_scaler_params(self):
        return {'min': self.global_min, 'max': self.global_max}
    
    def denormalize(self, x):
        """Reverse normalization to original capacity scale."""
        if self.neg_one_to_one:
            x = (x + 1) / 2
        x = x * (self.global_max - self.global_min) + self.global_min
        return x
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, idx):
        return self.samples[idx]
