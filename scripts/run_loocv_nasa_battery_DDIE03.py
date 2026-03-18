#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NASA 电池 LOOCV（留一电池交叉验证）：4 折，每折留一个电池作测试，其余 3 个训练。
训练 DDI-E 并评估 LI/KNN/DDI-E 插补，汇总 4 折均值±标准差。产出：results_DDIE03/loocv_nasa_battery/。
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

ALL_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RESULTS_DIR = ROOT / "results_DDIE03" / "loocv_nasa_battery"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_metric(s):
    """从 '0.4321 ± 0.0197' 解析出第一个浮点数。"""
    if not s or s == "-":
        return None
    m = re.match(r"([-\d.]+)\s*±", str(s).strip())
    return float(m.group(1)) if m else None


def get_loocv_config(base_config, test_battery):
    """为 LOOCV 某一折生成配置：训练集为其余 3 块电池，测试集为 test_battery。"""
    config = copy.deepcopy(base_config)
    train_ids = [b for b in ALL_BATTERIES if b != test_battery]
    fold_name = f"loocv_fold_{test_battery}"
    config["solver"]["results_folder"] = f"./Checkpoints_ddie_nasa_DDIE03_{fold_name}"
    for key in ["train_dataset", "val_dataset"]:
        config["dataloader"][key]["params"]["battery_ids"] = train_ids
    config["dataloader"]["test_dataset"]["params"]["battery_ids"] = [test_battery]
    config["dataloader"]["test_dataset"]["params"]["train_battery_ids_for_norm"] = train_ids
    return config


def run_one_fold_train_only(test_battery, base_config, max_epochs=200):
    """仅训练单折并保存 checkpoint，不进行评估。用于分步执行时先完成全部训练。"""
    from train_ddie import train
    config = get_loocv_config(base_config, test_battery)
    config["solver"]["max_epochs"] = max_epochs
    class Args:
        config = ""
        missing_rate = 0.3
    train(config, Args())


def run_one_fold(test_battery, base_config, exp_cfg, device, max_epochs=200, skip_train=False):
    """训练并评估单折，返回该折的插补指标（格式与 run_experiments_DDIE03 一致）。"""
    from Utils.io_utils import load_yaml_config, instantiate_from_config
    from Data.build_dataloader import build_dataloader
    from train_ddie import train

    config = get_loocv_config(base_config, test_battery)
    missing_rates = exp_cfg.get("missing_rates", [0.1, 0.3, 0.5, 0.7, 0.9])
    n_masks = exp_cfg.get("n_masks_per_sample", 5)
    seeds = exp_cfg.get("seeds", [42, 123, 2024])
    eval_batch_size = exp_cfg.get("eval_batch_size", 32)
    report_original_scale = exp_cfg.get("report_original_scale", True)

    # 训练
    if not skip_train:
        class Args:
            config = ""
            missing_rate = 0.3
        train(config, Args())
    else:
        ckpt = Path(config["solver"]["results_folder"]) / "best.pt"
        if not ckpt.exists():
            print(f"  [LOOCV] 跳过训练但无 checkpoint: {ckpt}，该折仅基线")
    # 评估
    test_dl = build_dataloader(config, "test")
    dataset = test_dl["dataset"]
    ckpt_path = Path(config["solver"]["results_folder"]) / "best.pt"
    model = None
    if ckpt_path.exists():
        model = instantiate_from_config(config["model"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    import run_experiments_DDIE03 as exp03
    imputation_results = exp03.evaluate_imputation(
        model, dataset, config, device, missing_rates, n_masks, seeds,
        eval_batch_size=eval_batch_size,
        report_original_scale=report_original_scale,
    )
    return imputation_results


def aggregate_loocv(fold_results):
    """将 4 折的插补结果汇总为均值±标准差。"""
    # 键可能是 float (0.1, 0.3...) 或 str；只排除以 "_" 开头的元数据键
    missing_rates = [k for k in fold_results[0] if not (isinstance(k, str) and k.startswith("_"))]
    methods = ["LI", "KNN", "DDI-E"]
    metrics = ["MAE", "RMSE", "MAPE"]
    agg = {}
    for mr in missing_rates:
        agg[mr] = {}
        for method in methods:
            agg[mr][method] = {}
            for met in metrics:
                values = []
                for res in fold_results:
                    v = _parse_metric(res.get(mr, {}).get(method, {}).get(met))
                    if v is not None:
                        values.append(v)
                if values:
                    agg[mr][method][met] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                else:
                    agg[mr][method][met] = "-"
    agg["_scale"] = fold_results[0].get("_scale", "original_scale")
    agg["_n_folds"] = len(fold_results)
    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NASA 电池 LOOCV（4 折）")
    parser.add_argument("--skip_train", action="store_true", help="不训练，仅用已有各折 checkpoint 做评估并汇总保存（仅评估）")
    parser.add_argument("--train_only", action="store_true", help="仅训练 4 折并保存 checkpoint，不进行评估与汇总")
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()

    from Utils.io_utils import load_yaml_config
    base_config = load_yaml_config(ROOT / "Config" / "nasa_battery_DDIE03.yaml")
    base_config["solver"]["max_epochs"] = args.max_epochs
    exp_cfg = base_config.get("experiment", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== NASA 电池 LOOCV (DDIE03) ==========")
    print("测试电池顺序:", ALL_BATTERIES)

    if args.train_only:
        print("[仅训练] 依次训练 4 折并保存 checkpoint，不进行评估。完成后可单独运行 --skip_train 做评估与汇总。")
        for test_battery in ALL_BATTERIES:
            print(f"\n--- 折: 测试电池 {test_battery} ---")
            run_one_fold_train_only(test_battery, base_config, args.max_epochs)
        print("\n[仅训练] 4 折训练已全部完成。运行本脚本并加 --skip_train 可仅做评估与汇总。")
        return 0

    fold_results = []
    for test_battery in ALL_BATTERIES:
        print(f"\n--- 折: 测试电池 {test_battery} ---")
        res = run_one_fold(
            test_battery, base_config, exp_cfg, device,
            max_epochs=args.max_epochs, skip_train=args.skip_train,
        )
        fold_results.append(res)
    aggregated = aggregate_loocv(fold_results)
    out = {
        "loocv_folds": {f"test_{b}": fold_results[i] for i, b in enumerate(ALL_BATTERIES)},
        "aggregate": aggregated,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / "imputation_results_loocv_DDIE03.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n汇总 (4 折 均值±标准差):")
    for mr in aggregated:
        if isinstance(mr, str) and mr.startswith("_"):
            continue
        print(f"  缺失率 {mr}:", aggregated[mr])
    print(f"\n已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
