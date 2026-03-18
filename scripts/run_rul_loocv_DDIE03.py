#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NASA 电池 RUL 留一电池交叉验证（4 折）：每折留一块电池作 RUL 测试，其余 3 块训练，汇总 4 折 RMSE/PHM。
产出：results_DDIE03/rul_loocv_DDIE03.json
"""

import os
import sys
import json
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ALL_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RESULTS_DIR = ROOT / "results_DDIE03"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_loocv_config(base_config, test_battery):
    train_ids = [b for b in ALL_BATTERIES if b != test_battery]
    config = copy.deepcopy(base_config)
    for key in ["train_dataset", "val_dataset"]:
        config["dataloader"][key]["params"]["battery_ids"] = train_ids
    config["dataloader"]["test_dataset"]["params"]["battery_ids"] = [test_battery]
    config["dataloader"]["test_dataset"]["params"]["train_battery_ids_for_norm"] = train_ids
    return config


def get_battery_xy(base_config, battery_as_test):
    """对指定电池作为测试集加载，得到 (X, y) 用于 RUL（每窗口取最后 m=10，标签 (1-mean)*100）。"""
    from Data.build_dataloader import build_dataloader
    from downstream_rul import build_rul_dataset

    cfg = get_loocv_config(base_config, battery_as_test)
    test_dl = build_dataloader(cfg, "test")
    dataset = test_dl["dataset"]
    capacity_sequences = [dataset[i].squeeze() for i in range(len(dataset))]
    rul_labels = [(1.0 - float(seq.mean())) * 100 for seq in capacity_sequences]
    rul_labels = [max(0, r) for r in rul_labels]
    X, y = build_rul_dataset(capacity_sequences, rul_labels, m=10)
    return X, y


def main():
    from Utils.io_utils import load_yaml_config
    from downstream_rul import train_rul_model, evaluate_rul

    base_config = load_yaml_config(ROOT / "Config" / "nasa_battery_DDIE03.yaml")
    # 预计算每块电池的 (X_b, y_b)
    per_battery = {}
    for b in ALL_BATTERIES:
        X_b, y_b = get_battery_xy(base_config, b)
        per_battery[b] = (X_b, y_b)
        print(f"  电池 {b}: {len(X_b)} 样本")

    fold_results = []
    for test_battery in ALL_BATTERIES:
        train_ids = [x for x in ALL_BATTERIES if x != test_battery]
        X_train = np.concatenate([per_battery[b][0] for b in train_ids], axis=0)
        y_train = np.concatenate([per_battery[b][1] for b in train_ids], axis=0)
        X_test, y_test = per_battery[test_battery][0], per_battery[test_battery][1]
        if len(X_test) < 1 or len(X_train) < 2:
            fold_results.append({"status": "skipped", "reason": "too_few_samples", "test": test_battery})
            continue
        model = train_rul_model(X_train, y_train, epochs=50, lr=1e-3)
        metrics = evaluate_rul(model, X_test, y_test)
        fold_results.append({
            "test_battery": test_battery,
            "n_train": len(X_train), "n_test": len(X_test),
            "RMSE": float(metrics["RMSE"]), "PHM_Score": float(metrics["PHM_Score"]),
        })
        print(f"  折 test={test_battery} RMSE={metrics['RMSE']:.4f} PHM={metrics['PHM_Score']:.4f}")

    rmse_list = [f["RMSE"] for f in fold_results if "RMSE" in f]
    phm_list = [f["PHM_Score"] for f in fold_results if "PHM_Score" in f]
    aggregate = {}
    if rmse_list:
        aggregate["RMSE_mean"] = float(np.mean(rmse_list))
        aggregate["RMSE_std"] = float(np.std(rmse_list))
    if phm_list:
        aggregate["PHM_Score_mean"] = float(np.mean(phm_list))
        aggregate["PHM_Score_std"] = float(np.std(phm_list))
    aggregate["n_folds"] = len(ALL_BATTERIES)

    out = {
        "folds": fold_results,
        "aggregate": aggregate,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / "rul_loocv_DDIE03.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nRUL LOOCV 汇总:", aggregate)
    print(f"已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
