#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDIE-03 实验计划自动执行：nasa_battery + NASA IGBT 两数据集（本实验不包含 UMass 数据集）。
训练 → 插补评估（原始尺度）；可选下游 NASA RUL。
日志写入 log/（文件名带 DDIE03），结果 results_DDIE03/，报告 reports_DDIE03/。不覆写 DDIE02/DDIE01。
沿用 20260205 报告优化：测试集与训练集归一化一致、report_original_scale。
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
os.chdir(ROOT)

LOG_DIR = ROOT / "log"
RESULTS_DIR = ROOT / "results_DDIE03"
REPORTS_DIR = ROOT / "reports_DDIE03"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, log_path):
        self.console = sys.stdout
        self.log_path = str(log_path)

    def write(self, data):
        self.console.write(data)
        if data:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(data if isinstance(data, str) else data.decode("utf-8", errors="replace"))

    def flush(self):
        self.console.flush()


def log_print(msg, log_file):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def verify_data_sources(config_paths_and_names, log_file=None):
    """
    核查各配置对应的数据路径是否存在（符合 20260205 报告 3.3、4.4）。
    返回 dict：{ name: { data_root, exists, note } }，用于写入报告。
    """
    from Utils.io_utils import load_yaml_config
    result = {}
    for config_path, name in config_paths_and_names:
        if not config_path.exists():
            result[name] = {"data_root": str(config_path), "exists": False, "note": "配置文件不存在"}
            continue
        cfg = load_yaml_config(config_path)
        train_cfg = cfg.get("dataloader", {}).get("train_dataset", {}).get("params", {})
        data_root = train_cfg.get("data_root", "")
        if not data_root:
            result[name] = {"data_root": "", "exists": False, "note": "未配置 data_root"}
            continue
        root_abs = (ROOT / data_root.replace("./", "")).resolve()
        if not root_abs.exists():
            result[name] = {"data_root": str(root_abs), "exists": False, "note": "路径不存在，可能使用合成/占位数据"}
            if log_file:
                log_print(f"  [数据来源] {name}: 路径不存在 {root_abs}", log_file)
            continue
        # 数据集类型简要核查（报告 3.3：确认数据来源并在报告中说明）
        note = "路径存在"
        if "nasa_battery" in name or "battery" in name.lower():
            found = sum(1 for bid in ["B0005", "B0006", "B0007", "B0018"]
                       if (root_abs / f"{bid}.mat").exists() or (root_abs / f"{bid}.MAT").exists())
            note = "真实数据（.mat 已找到）" if found >= 4 else "路径存在，但部分 .mat 未找到，可能使用合成数据"
        elif "igbt" in name.lower():
            devs = [d.name for d in root_abs.iterdir() if d.is_dir() and not d.name.startswith(".")]
            csv_found = False
            for d in devs[:3]:
                for part in (root_abs / d).iterdir():
                    if part.is_dir() and (part / "LeakageIV.csv").exists():
                        csv_found = True
                        break
            note = "真实数据（设备目录与 LeakageIV.csv 已找到）" if csv_found else "路径存在，未验证 CSV"
        result[name] = {"data_root": str(root_abs), "exists": True, "note": note}
        if log_file:
            log_print(f"  [数据来源] {name}: {note}", log_file)
    return result


def run_training(config_path, missing_rate=0.3, max_epochs_override=None, log_file=None):
    from Utils.io_utils import load_yaml_config
    from train_ddie import train

    config = load_yaml_config(config_path)
    if max_epochs_override is not None:
        config["solver"]["max_epochs"] = max_epochs_override
        log_print(f"  max_epochs 覆盖为 {max_epochs_override}", log_file)
    class Args:
        pass
    args_obj = Args()
    args_obj.config = str(config_path)
    args_obj.missing_rate = missing_rate
    train(config, args_obj)


def run_experiments_ddie03(config_path, skip_train=True, uncertainty=False, log_file=None):
    import run_experiments_DDIE03 as mod
    argv = ["run_experiments_DDIE03.py", "--config", str(config_path)]
    if skip_train:
        argv.append("--skip_train")
    if uncertainty:
        argv.append("--uncertainty")
    old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old_argv


def run_downstream_ddie03_nasa_only(log_file=None):
    """仅运行 NASA RUL 下游，写入 results_DDIE03/downstream_rul_DDIE03.json"""
    import run_downstream_DDIE03 as mod
    log_print("运行下游任务: NASA RUL (DDIE03)...", log_file)
    mod.main()


def main():
    parser = argparse.ArgumentParser(description="DDIE-03 实验计划：nasa_battery + NASA IGBT")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，仅评估已有 checkpoint")
    parser.add_argument("--battery_only", action="store_true", help="仅跑 NASA 电池配置")
    parser.add_argument("--igbt_only", action="store_true", help="仅跑 NASA IGBT 配置")
    parser.add_argument("--uncertainty", action="store_true", help="插补后对 DDI-E 做 N=20 不确定性量化")
    parser.add_argument("--no_downstream", action="store_true", help="不运行下游 NASA RUL")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"experiment_plan_DDIE03_{stamp}.log"
    try:
        def log(msg):
            log_print(msg, str(log_path))

        log("========== DDIE-03 实验计划（nasa_battery + NASA IGBT）==========")
        log(f"日志: {log_path}")
        log(f"结果: {RESULTS_DIR}")
        log(f"报告: {REPORTS_DIR}")

        configs = []
        if not args.igbt_only:
            configs.append((ROOT / "Config" / "nasa_battery_DDIE03.yaml", "nasa_battery"))
        if not args.battery_only:
            configs.append((ROOT / "Config" / "nasa_igbt_DDIE03.yaml", "nasa_igbt"))

        # 20260205 报告 3.3 / 4.4：数据来源核查并记录
        log("--- 数据来源核查（报告 3.3）---")
        data_sources = verify_data_sources(configs, log_file=str(log_path))

        for config_path, name in configs:
            if not config_path.exists():
                log(f"跳过不存在的配置: {config_path}")
                continue
            log(f"--- 配置: {name} ({config_path.name}) ---")
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = Tee(log_path)
                sys.stderr = Tee(log_path)
                if not args.skip_train:
                    log(f"训练 DDI-E: {config_path}")
                    try:
                        run_training(
                            config_path,
                            missing_rate=0.3,
                            max_epochs_override=args.max_epochs,
                            log_file=log_path,
                        )
                    except Exception as e:
                        log(f"训练异常: {e}")
                        import traceback
                        log(traceback.format_exc())
                else:
                    log("跳过训练 (--skip_train)")

                log(f"运行 DDIE03 插补评估: {config_path}")
                try:
                    run_experiments_ddie03(
                        config_path,
                        skip_train=True,
                        uncertainty=args.uncertainty,
                        log_file=log_path,
                    )
                except Exception as e:
                    log(f"插补评估异常: {e}")
                    import traceback
                    log(traceback.format_exc())
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        if not args.no_downstream:
            log("--- 下游任务 (NASA RUL) ---")
            try:
                run_downstream_ddie03_nasa_only(log_file=log_path)
            except Exception as e:
                log(f"下游任务异常: {e}")
                import traceback
                log(traceback.format_exc())

        # 读取实验参数（报告 4.4：实验记录）
        exp_seeds = [42, 123, 2024]
        max_epochs_used = args.max_epochs
        if configs:
            from Utils.io_utils import load_yaml_config
            first_cfg = load_yaml_config(configs[0][0])
            exp_seeds = first_cfg.get("experiment", {}).get("seeds", exp_seeds)
            if max_epochs_used is None:
                max_epochs_used = first_cfg.get("solver", {}).get("max_epochs", 200)
        else:
            max_epochs_used = max_epochs_used or 200

        report_path = REPORTS_DIR / f"experiment_summary_DDIE03_{stamp}.md"
        lines = [
            "# DDIE-03 实验计划执行摘要",
            "",
            f"**执行时间**: {datetime.now().isoformat()}",
            f"**日志文件**: `log/experiment_plan_DDIE03_{stamp}.log`",
            "",
            "## 实验记录（符合 20260205 报告 4.4）",
            "",
            "| 项目 | 说明 |",
            "|------|------|",
            "| 配置 | " + ", ".join(c[1] for c in configs) + " |",
            "| 随机种子 | " + str(exp_seeds) + " |",
            "| 训练轮次 | " + str(max_epochs_used) + " |",
            "| 结果目录 | results_DDIE03/ |",
            "",
            "### 数据来源核查（报告 3.3）",
            "",
        ]
        for name in data_sources:
            ds = data_sources[name]
            lines.append(f"- **{name}**: {ds.get('note', '')}  （路径: {ds.get('data_root', '')}）")
        lines.extend(["", "## 结果目录 (results_DDIE03/)", ""])
        for _, name in configs:
            res_dir = RESULTS_DIR / name
            json_path = res_dir / "imputation_results_DDIE03.json"
            lines.append(f"- **{name}**: `results_DDIE03/{name}/`")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lines.append("")
                lines.append("### 插补指标 (MAE / RMSE / MAPE, 原始尺度)")
                lines.append("")
                for k, v in data.items():
                    if k.startswith("_"):
                        continue
                    lines.append(f"- 缺失率 {k}:")
                    for method, metrics in (v if isinstance(v, dict) else {}).items():
                        if isinstance(metrics, dict):
                            lines.append(f"  - {method}: {metrics}")
                lines.append("")
        lines.append("- **下游 RUL (NASA)**: `results_DDIE03/downstream_rul_DDIE03.json`")
        lines.append("")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log(f"报告已写入: {report_path}")
        log("========== DDIE-03 执行结束 ==========")
    finally:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
