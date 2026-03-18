#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDIE-03 实验主脚本：插补评估（原始尺度）、不确定性量化。
使用 nasa_battery + NASA IGBT 两数据集（本实验不包含 UMass）；产出与 DDIE02/DDIE01 区分：results_DDIE03/、*_DDIE03.json、reports_DDIE03/。
沿用 20260205 报告优化：测试集与训练集归一化一致、report_original_scale。
用法:
  python scripts/run_experiments_DDIE03.py --config Config/nasa_battery_DDIE03.yaml
  python scripts/run_experiments_DDIE03.py --config Config/nasa_igbt_DDIE03.yaml
  python scripts/run_experiments_DDIE03.py --config Config/nasa_igbt_DDIE03.yaml --skip_train
  python scripts/run_experiments_DDIE03.py --config Config/nasa_battery_DDIE03.yaml --uncertainty
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.io_utils import load_yaml_config, instantiate_from_config
from Utils.Data_utils.missing_simulation import generate_multiple_masks
from Utils.metric_utils import compute_imputation_metrics, compute_uncertainty_metrics
from Data.build_dataloader import build_dataloader
from baselines.baseline_runner import run_baseline

# DDIE-03 专用产出目录与后缀（不覆写 DDIE02/DDIE01）
EXP_SUFFIX = "DDIE03"
RESULTS_BASE = Path("results_DDIE03")
REPORTS_BASE = Path("reports_DDIE03")


def _denormalize_if_available(dataset, arr):
    """若 dataset 有 denormalize 方法则反归一化到原始尺度。"""
    if hasattr(dataset, "denormalize") and callable(getattr(dataset, "denormalize")):
        return np.array(dataset.denormalize(arr), dtype=np.float64)
    return arr


def evaluate_imputation(
    model,
    dataset,
    config,
    device,
    missing_rates,
    n_masks,
    seeds,
    eval_batch_size=32,
    report_original_scale=True,
):
    """评估插补：可选在原始尺度上计算 MAE/RMSE/MAPE。"""
    methods = ["LI", "KNN", "DDI-E"]
    results = {}
    for mr in missing_rates:
        results[mr] = {}
        for method in methods:
            results[mr][method] = {"MAE": [], "RMSE": [], "MAPE": []}

    data = np.array([dataset[i] for i in range(len(dataset))])
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    n_samples = data.shape[0]
    scale_note = "original_scale" if (report_original_scale and hasattr(dataset, "denormalize")) else "normalized"

    for seed in seeds:
        for mr in missing_rates:
            masks = generate_multiple_masks(data.shape, mr, n_masks, base_seed=seed)
            for mi, mask in enumerate(masks):
                print(f"  [eval] seed={seed} mr={mr} mask={mi+1}/{n_masks}")
                if mask.ndim == 2:
                    mask_3d = np.expand_dims(mask, -1)
                else:
                    mask_3d = mask
                masked = data.copy()
                masked[mask_3d < 0.5] = np.nan

                # LI
                imp_li = run_baseline("LI", data, mask)
                orig_eval = _denormalize_if_available(dataset, data) if report_original_scale else data
                imp_li_eval = _denormalize_if_available(dataset, imp_li) if report_original_scale else imp_li
                for k, v in compute_imputation_metrics(orig_eval, imp_li_eval, mask).items():
                    results[mr]["LI"][k].append(v)

                # KNN
                imp_knn = run_baseline("KNN", data, mask, n_neighbors=10)
                imp_knn_eval = _denormalize_if_available(dataset, imp_knn) if report_original_scale else imp_knn
                for k, v in compute_imputation_metrics(orig_eval, imp_knn_eval, mask).items():
                    results[mr]["KNN"][k].append(v)

                # DDI-E
                if model is not None:
                    imp_ddie = np.zeros_like(data, dtype=np.float32)
                    n_batches = (n_samples + eval_batch_size - 1) // eval_batch_size
                    for bi, start in enumerate(range(0, n_samples, eval_batch_size)):
                        if n_batches >= 4 and (bi + 1) % max(1, n_batches // 4) == 0:
                            print(f"    DDI-E batch {bi+1}/{n_batches}")
                        end = min(start + eval_batch_size, n_samples)
                        batch_data = data[start:end]
                        batch_mask = mask[start:end]
                        with torch.no_grad():
                            x = torch.tensor(batch_data, dtype=torch.float32).to(device)
                            m = torch.tensor(batch_mask, dtype=torch.float32).to(device)
                            if m.dim() == 2:
                                m = m.unsqueeze(-1)
                            target = x.clone()
                            target[m.expand_as(x) < 0.5] = 0.0
                            pred = model.fast_sample_infill(x.shape, target, m, clip_denoised=True)
                            imp_ddie[start:end] = pred.cpu().numpy()
                    imp_ddie_eval = _denormalize_if_available(dataset, imp_ddie) if report_original_scale else imp_ddie
                    for k, v in compute_imputation_metrics(orig_eval, imp_ddie_eval, mask).items():
                        results[mr]["DDI-E"][k].append(v)

    out = {}
    for mr in missing_rates:
        out[mr] = {}
        for method in methods:
            m = results[mr][method]
            out[mr][method] = {
                "MAE": f"{np.mean(m['MAE']):.4f} ± {np.std(m['MAE']):.4f}" if m["MAE"] else "-",
                "RMSE": f"{np.mean(m['RMSE']):.4f} ± {np.std(m['RMSE']):.4f}" if m["RMSE"] else "-",
                "MAPE": f"{np.mean(m['MAPE']):.2f} ± {np.std(m['MAPE']):.2f}" if m["MAPE"] else "-",
            }
    out["_scale"] = scale_note
    return out


def run_uncertainty_eval(model, dataset, config, device, missing_rates, n_masks, seeds, eval_batch_size, n_samples=20):
    """对 DDI-E 做 N 次采样，计算 95% 区间覆盖率与平均区间宽度。"""
    results = {}
    data = np.array([dataset[i] for i in range(len(dataset))])
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    n_total = data.shape[0]

    for mr in missing_rates:
        coverage_list = []
        width_list = []
        for seed in seeds:
            masks = generate_multiple_masks(data.shape, mr, n_masks, base_seed=seed)
            for mask in masks:
                if mask.ndim == 2:
                    mask_3d = np.expand_dims(mask, -1)
                else:
                    mask_3d = mask
                samples_list = []
                for _ in range(n_samples):
                    imp = np.zeros_like(data, dtype=np.float32)
                    for start in range(0, n_total, eval_batch_size):
                        end = min(start + eval_batch_size, n_total)
                        batch_data = data[start:end]
                        batch_mask = mask[start:end]
                        with torch.no_grad():
                            x = torch.tensor(batch_data, dtype=torch.float32).to(device)
                            m = torch.tensor(batch_mask, dtype=torch.float32).to(device)
                            if m.dim() == 2:
                                m = m.unsqueeze(-1)
                            target = x.clone()
                            target[m.expand_as(x) < 0.5] = 0.0
                            pred = model.fast_sample_infill(x.shape, target, m, clip_denoised=True)
                            imp[start:end] = pred.cpu().numpy()
                    samples_list.append(imp)
                samples_arr = np.stack(samples_list, axis=0)
                gt = data
                met = compute_uncertainty_metrics(gt, samples_arr, mask)
                coverage_list.append(met["coverage"])
                width_list.append(met["mean_interval_width"])
        results[mr] = {
            "coverage_mean": float(np.mean(coverage_list)),
            "coverage_std": float(np.std(coverage_list)),
            "mean_interval_width_mean": float(np.mean(width_list)),
            "mean_interval_width_std": float(np.std(width_list)),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/nasa_battery_DDIE03.yaml")
    parser.add_argument("--skip_train", action="store_true", help="不训练，仅评估（需已有 checkpoint）")
    parser.add_argument("--uncertainty", action="store_true", help="对 DDI-E 做 N=20 不确定性量化并保存")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    exp_cfg = config.get("experiment", {})
    missing_rates = exp_cfg.get("missing_rates", [0.1, 0.3, 0.5, 0.7, 0.9])
    n_masks = exp_cfg.get("n_masks_per_sample", 5)
    seeds = exp_cfg.get("seeds", [42, 123, 2024])
    eval_batch_size = exp_cfg.get("eval_batch_size", 32)
    report_original_scale = exp_cfg.get("report_original_scale", True)
    uncertainty_samples = exp_cfg.get("uncertainty_samples", 20)

    config_name = Path(args.config).stem
    if "igbt" in config_name.lower():
        dataset_key = "nasa_igbt"
    elif "nasa" in config_name.lower():
        dataset_key = "nasa_battery"
    else:
        dataset_key = config_name

    out_dir = RESULTS_BASE / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dl = build_dataloader(config, "test")
    dataset = test_dl["dataset"]

    model = None
    ckpt_path = Path(config["solver"]["results_folder"]) / "best.pt"
    if not args.skip_train:
        print("Training DDI-E (DDIE-03)...")
        model = instantiate_from_config(config["model"]).to(device)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print("  No checkpoint found. Run train_ddie.py with DDIE03 config first or use --skip_train.")
    else:
        if ckpt_path.exists():
            model = instantiate_from_config(config["model"]).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print("  No checkpoint at", ckpt_path, "- baselines only.")

    print("Evaluating imputation (DDIE-03, scale:", "original" if report_original_scale else "normalized", ")...")
    imputation_results = evaluate_imputation(
        model, dataset, config, device, missing_rates, n_masks, seeds,
        eval_batch_size=eval_batch_size,
        report_original_scale=report_original_scale,
    )
    out_file = out_dir / f"imputation_results_{EXP_SUFFIX}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(imputation_results, f, indent=2, ensure_ascii=False)
    print("  Saved:", out_file)

    if args.uncertainty and model is not None:
        print("Running uncertainty evaluation (N=%d)..." % uncertainty_samples)
        unc_results = run_uncertainty_eval(
            model, dataset, config, device, missing_rates, n_masks, seeds,
            eval_batch_size=eval_batch_size,
            n_samples=uncertainty_samples,
        )
        unc_file = out_dir / f"uncertainty_metrics_{EXP_SUFFIX}.json"
        with open(unc_file, "w", encoding="utf-8") as f:
            json.dump(unc_results, f, indent=2)
        print("  Saved:", unc_file)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_BASE / f"experiment_summary_{EXP_SUFFIX}_{dataset_key}_{stamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# DDIE-03 实验摘要 ({dataset_key})\n\n")
        f.write(f"时间: {datetime.now().isoformat()}\n\n")
        f.write(f"## 插补结果 (imputation_results_{EXP_SUFFIX}.json)\n\n")
        f.write("```json\n")
        f.write(json.dumps(imputation_results, indent=2, ensure_ascii=False))
        f.write("\n```\n")
    print("Report:", report_path)
    print("Done (DDIE-03).")


if __name__ == "__main__":
    main()
