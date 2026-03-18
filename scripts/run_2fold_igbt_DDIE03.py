#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NASA IGBT 两折交叉验证：2 个设备轮流作测试集，汇总 2 折插补指标（均值±标准差）。
产出：results_DDIE03/loocv_nasa_igbt/imputation_results_2fold_DDIE03.json
"""

import os
import sys
import json
import copy
import re
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICES = ["IGBT-IRG4BC30K", "MOSFET-IRF520Npbf"]
RESULTS_DIR = ROOT / "results_DDIE03" / "loocv_nasa_igbt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_metric(s):
    if not s or s == "-":
        return None
    m = re.match(r"([-\d.]+)\s*±", str(s).strip())
    return float(m.group(1)) if m else None


def get_2fold_config(base_config, test_device):
    """指定 test_device 为测试设备，另一设备为训练。"""
    config = copy.deepcopy(base_config)
    train_device = [d for d in DEVICES if d != test_device][0]
    fold_name = f"2fold_test_{test_device.replace('-', '_')}"
    config["solver"]["results_folder"] = f"./Checkpoints_ddie_igbt_DDIE03_{fold_name}"
    for key in ["train_dataset", "val_dataset", "test_dataset"]:
        config["dataloader"][key]["params"]["device_ids"] = DEVICES
        config["dataloader"][key]["params"]["test_device_ids"] = [test_device]
    return config


def run_one_fold(test_device, base_config, exp_cfg, device, max_epochs=200, skip_train=False):
    from Utils.io_utils import instantiate_from_config
    from Data.build_dataloader import build_dataloader
    from train_ddie import train

    config = get_2fold_config(base_config, test_device)
    missing_rates = exp_cfg.get("missing_rates", [0.1, 0.3, 0.5, 0.7, 0.9])
    n_masks = exp_cfg.get("n_masks_per_sample", 5)
    seeds = exp_cfg.get("seeds", [42, 123, 2024])
    eval_batch_size = exp_cfg.get("eval_batch_size", 8)
    report_original_scale = exp_cfg.get("report_original_scale", True)

    if not skip_train:
        class Args:
            config = ""
            missing_rate = 0.3
        train(config, Args())
    ckpt_path = Path(config["solver"]["results_folder"]) / "best.pt"
    test_dl = build_dataloader(config, "test")
    dataset = test_dl["dataset"]
    model = None
    if ckpt_path.exists():
        model = instantiate_from_config(config["model"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    import run_experiments_DDIE03 as exp03
    return exp03.evaluate_imputation(
        model, dataset, config, device, missing_rates, n_masks, seeds,
        eval_batch_size=eval_batch_size,
        report_original_scale=report_original_scale,
    )


def run_one_fold_train_only(test_device, base_config, max_epochs=200):
    """仅训练单折并保存 checkpoint，不进行评估。"""
    from train_ddie import train
    config = get_2fold_config(base_config, test_device)
    config["solver"]["max_epochs"] = max_epochs
    class Args:
        config = ""
        missing_rate = 0.3
    train(config, Args())


def aggregate_2fold(fold_results):
    missing_rates = [k for k in fold_results[0] if not (isinstance(k, str) and k.startswith("_"))]
    methods = ["LI", "KNN", "DDI-E"]
    metrics = ["MAE", "RMSE", "MAPE"]
    agg = {}
    for mr in missing_rates:
        agg[mr] = {}
        for method in methods:
            agg[mr][method] = {}
            for met in metrics:
                values = [_parse_metric(r.get(mr, {}).get(method, {}).get(met)) for r in fold_results]
                values = [v for v in values if v is not None]
                if values:
                    agg[mr][method][met] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                else:
                    agg[mr][method][met] = "-"
    agg["_scale"] = fold_results[0].get("_scale", "original_scale")
    agg["_n_folds"] = len(fold_results)
    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NASA IGBT 两折交叉验证")
    parser.add_argument("--skip_train", action="store_true", help="不训练，仅用已有 checkpoint 做评估并汇总保存（仅评估）")
    parser.add_argument("--train_only", action="store_true", help="仅训练 2 折并保存 checkpoint，不进行评估与汇总")
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()

    from Utils.io_utils import load_yaml_config
    base_config = load_yaml_config(ROOT / "Config" / "nasa_igbt_DDIE03.yaml")
    base_config["solver"]["max_epochs"] = args.max_epochs
    exp_cfg = base_config.get("experiment", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== NASA IGBT 2-Fold (DDIE03) ==========")

    if args.train_only:
        print("[仅训练] 依次训练 2 折并保存 checkpoint，不进行评估。完成后可单独运行 --skip_train 做评估与汇总。")
        for test_device in DEVICES:
            print(f"\n--- 折: 测试设备 {test_device} ---")
            run_one_fold_train_only(test_device, base_config, args.max_epochs)
        print("\n[仅训练] 2 折训练已全部完成。运行本脚本并加 --skip_train 可仅做评估与汇总。")
        return 0

    fold_results = []
    for test_device in DEVICES:
        print(f"\n--- 折: 测试设备 {test_device} ---")
        res = run_one_fold(test_device, base_config, exp_cfg, device, args.max_epochs, args.skip_train)
        fold_results.append(res)
    aggregated = aggregate_2fold(fold_results)
    out = {
        "folds": {f"test_{d}": fold_results[i] for i, d in enumerate(DEVICES)},
        "aggregate": aggregated,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / "imputation_results_2fold_DDIE03.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n汇总 (2 折 均值±标准差):")
    for mr in aggregated:
        if isinstance(mr, str) and mr.startswith("_"):
            continue
        print(f"  缺失率 {mr}:", aggregated[mr])
    print(f"\n已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
