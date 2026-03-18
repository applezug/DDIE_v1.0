#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train DDI-E model for NASA Battery or NASA IGBT dataset.
Usage: python scripts/train_ddie.py --config Config/nasa_battery.yaml [--missing_rate 0.3]
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.io_utils import load_yaml_config, instantiate_from_config
from Data.build_dataloader import build_dataloader


def train(config, args):
    model = instantiate_from_config(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_dl = build_dataloader(config, 'train', missing_rate=getattr(args, 'missing_rate', 0.3))
    val_dl = build_dataloader(config, 'val', missing_rate=0.3)  # Validate on 30% missing

    opt = torch.optim.Adam(model.parameters(), lr=config['solver']['base_lr'], betas=(0.9, 0.999))
    sc_cfg = config['solver']['scheduler']
    sc_cfg['params']['optimizer'] = opt
    scheduler = instantiate_from_config(sc_cfg)

    results_folder = Path(config['solver']['results_folder'])
    results_folder.mkdir(parents=True, exist_ok=True)
    patience = config['solver'].get('early_stop_patience', 20)
    best_mae = float('inf')
    no_improve = 0

    for epoch in range(config['solver']['max_epochs']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dl['dataloader'], desc=f'Epoch {epoch+1}'):
            data = batch['data'].to(device)
            mask = batch['mask'].to(device)
            opt.zero_grad()
            loss = model(data, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl['dataloader'])
        scheduler.step(avg_loss)

        model.eval()
        val_mae = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_dl['dataloader']:
                data = batch['data'].to(device)
                mask = batch['mask'].to(device)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                target = data.clone()
                target[mask.expand_as(data) < 0.5] = 0.0
                pred = model.fast_sample_infill(data.shape, target, mask, clip_denoised=True)
                miss = (mask < 0.5)
                if miss.any():
                    val_mae += (pred[miss] - data[miss]).abs().sum().item()
                    n_val += miss.sum().item()
        val_mae = val_mae / max(n_val, 1)

        if val_mae < best_mae:
            best_mae = val_mae
            no_improve = 0
            torch.save(model.state_dict(), results_folder / 'best.pt')
        else:
            no_improve += 1

        print(f'Epoch {epoch+1} | Loss: {avg_loss:.6f} | Val MAE: {val_mae:.6f} | Best: {best_mae:.6f}')
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print('Training complete.')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Config/nasa_battery.yaml')
    parser.add_argument('--missing_rate', type=float, default=0.3)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    train(config, args)
