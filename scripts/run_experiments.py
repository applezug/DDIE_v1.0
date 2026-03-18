#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run full DDI-E experiment pipeline: train, evaluate imputation, baselines, downstream tasks.
Usage: python scripts/run_experiments.py --config Config/nasa_battery.yaml
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.io_utils import load_yaml_config, instantiate_from_config
from Utils.Data_utils.missing_simulation import generate_mcar_mask, generate_multiple_masks
from Utils.metric_utils import compute_imputation_metrics, compute_uncertainty_metrics
from Data.build_dataloader import build_dataloader
from baselines.baseline_runner import run_baseline


def evaluate_imputation(model, dataset, config, device, missing_rates, n_masks, seeds, eval_batch_size=32):
    """
    eval_batch_size: DDI-E 推理时每批样本数，避免整表一次进 GPU 导致 OOM 或极慢。
    """
    results = {}
    for mr in missing_rates:
        results[mr] = {}
        for method in ['LI', 'KNN', 'DDI-E']:
            results[mr][method] = {'MAE': [], 'RMSE': [], 'MAPE': []}

    data = np.array([dataset[i] for i in range(len(dataset))])
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    n_samples = data.shape[0]
    n_loops = len(seeds) * len(missing_rates) * n_masks
    print(f"  Test samples: {n_samples}, eval_batch_size: {eval_batch_size}, total loops: {n_loops} (seed×mr×masks)")

    for seed in seeds:
        for mr in missing_rates:
            masks = generate_multiple_masks(data.shape, mr, n_masks, base_seed=seed)

            for mi, mask in enumerate(masks):
                print(f"  [eval] seed={seed} mr={mr} mask={mi+1}/{n_masks}")
                masked = data.copy()
                if mask.ndim == 2:
                    mask_3d = np.expand_dims(mask, -1)
                else:
                    mask_3d = mask
                masked[mask_3d < 0.5] = np.nan

                # LI
                imp_li = run_baseline('LI', data, mask)
                for k, v in compute_imputation_metrics(data, imp_li, mask).items():
                    results[mr]['LI'][k].append(v)

                # KNN
                imp_knn = run_baseline('KNN', data, mask, n_neighbors=10)
                for k, v in compute_imputation_metrics(data, imp_knn, mask).items():
                    results[mr]['KNN'][k].append(v)

                # DDI-E: 按批推理，避免整表一次进 GPU（OOM/极慢）
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
                    for k, v in compute_imputation_metrics(data, imp_ddie, mask).items():
                        results[mr]['DDI-E'][k].append(v)

    # Aggregate
    out = {}
    for mr in missing_rates:
        out[mr] = {}
        for method in ['LI', 'KNN', 'DDI-E']:
            m = results[mr][method]
            out[mr][method] = {
                'MAE': f"{np.mean(m['MAE']):.4f} ± {np.std(m['MAE']):.4f}" if m['MAE'] else '-',
                'RMSE': f"{np.mean(m['RMSE']):.4f} ± {np.std(m['RMSE']):.4f}" if m['RMSE'] else '-',
                'MAPE': f"{np.mean(m['MAPE']):.2f} ± {np.std(m['MAPE']):.2f}" if m['MAPE'] else '-',
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config/nasa_battery.yaml')
    parser.add_argument('--output', type=str, default='./experiment_results')
    parser.add_argument('--skip_train', action='store_true')
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    exp_cfg = config.get('experiment', {})
    missing_rates = exp_cfg.get('missing_rates', [0.1, 0.3, 0.5, 0.7, 0.9])
    n_masks = exp_cfg.get('n_masks_per_sample', 5)
    seeds = exp_cfg.get('seeds', [42, 123, 2024])
    eval_batch_size = exp_cfg.get('eval_batch_size', 32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dl = build_dataloader(config, 'test')
    dataset = test_dl['dataset']

    model = None
    if not args.skip_train:
        print('Training DDI-E...')
        model = instantiate_from_config(config['model']).to(device)
        ckpt = Path(config['solver']['results_folder']) / 'best.pt'
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print('  No checkpoint found. Run train_ddie.py first or use --skip_train to evaluate baselines only.')
    else:
        ckpt = Path(config['solver']['results_folder']) / 'best.pt'
        if ckpt.exists():
            model = instantiate_from_config(config['model']).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))

    print('Evaluating imputation...')
    results = evaluate_imputation(model, dataset, config, device, missing_rates, n_masks, seeds, eval_batch_size=eval_batch_size)
    with open(out_dir / 'imputation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Done. Results saved to', out_dir / 'imputation_results.json')


if __name__ == '__main__':
    main()
