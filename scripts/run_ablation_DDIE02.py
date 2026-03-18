#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDIE-02 消融实验：训练并评估 DDI-E 变体（无条件、无频域损失、完整模型）。
产出：results_DDIE02/ablation/ablation_metrics_DDIE02.json
用法: python scripts/run_ablation_DDIE02.py --config Config/nasa_battery_DDIE02.yaml [--skip_train]
"""

import os
import sys
import argparse
import json
import copy
import numpy as np
import torch
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from Utils.io_utils import load_yaml_config, instantiate_from_config
from Utils.Data_utils.missing_simulation import generate_multiple_masks
from Utils.metric_utils import compute_imputation_metrics
from Data.build_dataloader import build_dataloader
from train_ddie import train

EXP_SUFFIX = "DDIE02"
RESULTS_BASE = Path("results_DDIE02")
ABLATION_DIR = RESULTS_BASE / "ablation"

# 变体：(名称, use_mask_condition, freq_loss_weight)
VARIANTS = [
    ("full", True, 0.1),
    ("no_mask_condition", False, 0.1),
    ("no_freq_loss", True, 0.0),
]


def _denormalize_if_available(dataset, arr):
    if hasattr(dataset, "denormalize") and callable(getattr(dataset, "denormalize")):
        return np.array(dataset.denormalize(arr), dtype=np.float64)
    return arr


def evaluate_variant(model, dataset, data, mask, report_original_scale=True):
    """单次插补并计算 MAE（仅缺失位置）。"""
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).to(device)
        m = torch.tensor(mask, dtype=torch.float32).to(device)
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        target = x.clone()
        target[m.expand_as(x) < 0.5] = 0.0
        pred = model.fast_sample_infill(x.shape, target, m, clip_denoised=True)
        imp = pred.cpu().numpy()
    orig_eval = _denormalize_if_available(dataset, data) if report_original_scale else data
    imp_eval = _denormalize_if_available(dataset, imp) if report_original_scale else imp
    return compute_imputation_metrics(orig_eval, imp_eval, mask)["MAE"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/nasa_battery_DDIE02.yaml")
    parser.add_argument("--skip_train", action="store_true", help="不训练，仅评估已有 checkpoint")
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--n_masks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50, help="消融时可用较少 epoch 快速跑")
    args = parser.parse_args()

    base_config = load_yaml_config(args.config)
    report_original_scale = base_config.get("experiment", {}).get("report_original_scale", True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    test_dl = build_dataloader(base_config, "test")
    dataset = test_dl["dataset"]
    data = np.array([dataset[i] for i in range(len(dataset))])
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for variant_name, use_mask, freq_w in VARIANTS:
        print(f"\n--- Variant: {variant_name} (use_mask={use_mask}, freq_loss_weight={freq_w}) ---")
        config = copy.deepcopy(base_config)
        config["model"]["params"]["use_mask_condition"] = use_mask
        config["model"]["params"]["freq_loss_weight"] = freq_w
        base = Path(config["solver"]["results_folder"])
        ckpt_dir = base.parent / f"{base.name}_ablation_{variant_name}"
        config["solver"]["results_folder"] = str(ckpt_dir)
        config["solver"]["max_epochs"] = args.max_epochs

        ckpt_path = Path(ckpt_dir) / "best.pt"
        if not ckpt_path.exists() and not args.skip_train:
            # 训练
            class AblationArgs:
                pass
            a = AblationArgs()
            a.missing_rate = args.missing_rate
            train(config, a)
        elif ckpt_path.exists():
            pass
        else:
            print(f"  Skip (no checkpoint and --skip_train)")
            results[variant_name] = {"MAE": None, "delta_MAE_pct": None}
            continue

        if not ckpt_path.exists():
            results[variant_name] = {"MAE": None, "delta_MAE_pct": None}
            continue

        model = instantiate_from_config(config["model"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        mae_list = []
        masks = generate_multiple_masks(data.shape, args.missing_rate, args.n_masks, base_seed=args.seed)
        for mask in masks:
            mae_list.append(evaluate_variant(model, dataset, data, mask, report_original_scale))
        mae_mean = float(np.mean(mae_list))
        results[variant_name] = {"MAE": mae_mean, "MAE_std": float(np.std(mae_list)), "n_masks": len(mae_list)}

    # ΔMAE 以 full 为基准
    if "full" in results and results["full"].get("MAE") is not None:
        base_mae = results["full"]["MAE"]
        for k, v in results.items():
            if v.get("MAE") is not None and base_mae > 0:
                v["delta_MAE_pct"] = (v["MAE"] - base_mae) / base_mae * 100
            else:
                v["delta_MAE_pct"] = None
        results["_reference"] = "full"
        results["_missing_rate"] = args.missing_rate

    out_file = ABLATION_DIR / f"ablation_metrics_{EXP_SUFFIX}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
