#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDIE-03 下游任务：仅 NASA 电池 RUL（本实验不包含 UMass，无负荷预测下游；IGBT 无 RUL 下游）。
产出：results_DDIE03/downstream_rul_DDIE03.json。不覆写 DDIE02。
用法: python scripts/run_downstream_DDIE03.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
os.chdir(ROOT)

EXP_SUFFIX = "DDIE03"
RESULTS_DIR = ROOT / "results_DDIE03"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NASA = ROOT / "Config" / "nasa_battery_DDIE03.yaml"


def _to_serializable(obj):
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def run_rul():
    """NASA 容量退化 → RUL 预测（使用 DDIE03 配置）。"""
    from Utils.io_utils import load_yaml_config
    from Data.build_dataloader import build_dataloader
    from downstream_rul import build_rul_dataset, train_rul_model, evaluate_rul
    from sklearn.model_selection import train_test_split

    config = load_yaml_config(CONFIG_NASA)
    test_dl = build_dataloader(config, "test")
    dataset = test_dl["dataset"]
    capacity_sequences = [dataset[i].squeeze() for i in range(len(dataset))]
    rul_labels = [(1.0 - float(seq.mean())) * 100 for seq in capacity_sequences]
    rul_labels = [max(0, r) for r in rul_labels]

    X, y = build_rul_dataset(capacity_sequences, rul_labels, m=10)
    if len(X) < 3:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(X)}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_rul_model(X_tr, y_tr, epochs=50, lr=1e-3)
    metrics = evaluate_rul(model, X_te, y_te)
    return {"status": "ok", "metrics": metrics, "n_train": len(X_tr), "n_test": len(X_te), "experiment": EXP_SUFFIX}


def main():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] DDIE-03 下游任务：NASA RUL")

    try:
        rul_result = run_rul()
        out_path = RESULTS_DIR / f"downstream_rul_{EXP_SUFFIX}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"time": stamp, "result": _to_serializable(rul_result)}, f, indent=2, ensure_ascii=False)
        print("  RUL ->", out_path, rul_result)
    except Exception as e:
        rul_result = {"status": "error", "error": str(e)}
        with open(RESULTS_DIR / f"downstream_rul_{EXP_SUFFIX}.json", "w", encoding="utf-8") as f:
            json.dump({"time": stamp, "result": rul_result}, f, indent=2, ensure_ascii=False)
        print("  RUL 异常:", e)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DDIE-03 下游任务结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
