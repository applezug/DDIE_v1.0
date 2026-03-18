#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NASA IGBT 老化数据集：从 SMU 表征 CSV（LeakageIV 等）提取退化指标，构造等长序列。
支持「测试集使用训练集 global_min/max 归一化」与报告中的优化策略一致。
"""

import os
import re
import numpy as np
from torch.utils.data import Dataset


def _parse_part_key(name):
    """Part 文件夹名排序用：Part 1, Part 2, ... Part 19, Part 1A, Part 1B 等。"""
    m = re.match(r"^Part\s*(\d+)([A-Za-z]*)$", name.strip(), re.IGNORECASE)
    if m:
        num = int(m.group(1))
        suffix = (m.group(2) or "").upper()
        return (num, suffix)
    return (0, "")


def _read_leakage_scalar(csv_path):
    """从 LeakageIV.csv 读两列 (V,I)，取第二列绝对值均值作为该 Part 的退化代理。"""
    if not os.path.isfile(csv_path):
        return None
    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
        if data.shape[0] == 0 or data.shape[1] < 2:
            return None
        # 电流列取绝对值后均值，避免符号影响
        i_col = np.abs(data[:, 1])
        return float(np.mean(i_col))
    except Exception:
        return None


def load_igbt_sequences(data_root, device_ids=None):
    """
    扫描 data_root 下各设备文件夹，每个设备下 Part 1,2,... 的 LeakageIV.csv 提取一标量，
    得到每设备一条序列。返回序列列表、全局 min/max（用于归一化）。
    data_root: 例如 "Data/datasets/NASA IGBT/Data/SMU Data for new devices"
    或包含多个设备子目录的根路径。
    """
    if device_ids is None:
        device_ids = []
        for name in sorted(os.listdir(data_root)):
            path = os.path.join(data_root, name)
            if os.path.isdir(path) and not name.startswith("."):
                device_ids.append(name)
    if not device_ids:
        return [], 0.0, 1.0

    all_values = []
    sequences = {}  # device_id -> np.array of scalars per Part

    for dev_id in device_ids:
        dev_path = os.path.join(data_root, dev_id)
        if not os.path.isdir(dev_path):
            continue
        part_dirs = []
        for p in os.listdir(dev_path):
            full = os.path.join(dev_path, p)
            if os.path.isdir(full) and p.strip().lower().startswith("part"):
                part_dirs.append(p)
        part_dirs.sort(key=lambda x: _parse_part_key(x))
        values = []
        for part_name in part_dirs:
            csv_path = os.path.join(dev_path, part_name, "LeakageIV.csv")
            v = _read_leakage_scalar(csv_path)
            if v is not None:
                values.append(v)
        if values:
            arr = np.array(values, dtype=np.float64)
            sequences[dev_id] = arr
            all_values.extend(arr.tolist())

    global_min = float(np.min(all_values)) if all_values else 0.0
    global_max = float(np.max(all_values)) if all_values else 1.0
    return sequences, global_min, global_max


class NASAIGBTDataset(Dataset):
    """
    NASA IGBT 退化序列数据集。
    - 每设备一条原始序列（Part 1..K 的标量），插值到 window 长度。
    - 训练/验证集：使用自身 global_min/max 归一化。
    - 测试集：使用训练集设备的 global_min/max 归一化（与 20260205 报告一致）。
    - 输出形状 (N, window, 1)，归一化到 [0,1] 或 [-1,1]。
    """

    def __init__(
        self,
        data_root,
        device_ids=None,
        window=128,
        seed=42,
        period="train",
        val_ratio=0.2,
        test_device_ids=None,
        neg_one_to_one=True,
    ):
        super().__init__()
        self.data_root = data_root
        self.window = window
        self.seed = seed
        self.period = period
        self.val_ratio = val_ratio
        self.neg_one_to_one = neg_one_to_one
        self.test_device_ids = test_device_ids or []

        # 确定训练用设备与测试用设备
        all_ids = device_ids
        if all_ids is None:
            all_ids = []
            for name in sorted(os.listdir(data_root)):
                p = os.path.join(data_root, name)
                if os.path.isdir(p) and not name.startswith("."):
                    all_ids.append(name)
        train_ids = [d for d in all_ids if d not in self.test_device_ids]
        if not train_ids:
            train_ids = all_ids[: max(1, len(all_ids) - 1)]
        test_ids = self.test_device_ids if self.test_device_ids else [d for d in all_ids if d not in train_ids]
        if not test_ids and all_ids:
            test_ids = [all_ids[-1]]

        if period == "test":
            _, self.global_min, self.global_max = load_igbt_sequences(data_root, train_ids)
            self.sequences, _, _ = load_igbt_sequences(data_root, test_ids)
        else:
            self.sequences, self.global_min, self.global_max = load_igbt_sequences(data_root, train_ids)

        if not self.sequences:
            raise ValueError("No IGBT data loaded. Check data_root and device_ids.")

        self.samples = []
        self.device_ids_list = []
        for dev_id, arr in self.sequences.items():
            if arr.size < 2:
                continue
            # 插值到 window 长度
            x_old = np.linspace(0, 1, arr.size)
            x_new = np.linspace(0, 1, window)
            interp = np.interp(x_new, x_old, arr)
            seq = interp.reshape(-1, 1).astype(np.float32)
            self.samples.append(seq)
            self.device_ids_list.append(dev_id)
        if not self.samples:
            raise ValueError("No valid sequences after interpolation.")
        self.samples = np.array(self.samples, dtype=np.float32)

        # 归一化
        self.samples = (self.samples - self.global_min) / (self.global_max - self.global_min + 1e-8)
        self.samples = np.clip(self.samples, 0, 1)
        if neg_one_to_one:
            self.samples = self.samples * 2 - 1

        if period in ["train", "val"] and len(self.samples) > 1:
            np.random.seed(seed)
            n = len(self.samples)
            idx = np.random.permutation(n)
            n_val = max(1, int(n * val_ratio))
            val_idx = set(idx[:n_val])
            if period == "train":
                keep = [i for i in range(n) if i not in val_idx]
            else:
                keep = [i for i in range(n) if i in val_idx]
            self.samples = self.samples[keep]
            self.device_ids_list = [self.device_ids_list[i] for i in keep]

        self.sample_num = len(self.samples)

    def get_scaler_params(self):
        return {"min": self.global_min, "max": self.global_max}

    def denormalize(self, x):
        if self.neg_one_to_one:
            x = (x + 1) / 2
        x = x * (self.global_max - self.global_min) + self.global_min
        return x

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return self.samples[idx]
