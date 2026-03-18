"""Build dataloaders for DDI-E experiments"""

import torch
from torch.utils.data import DataLoader
from Utils.io_utils import instantiate_from_config


class DDICollateFn:
    """Collate that optionally applies random MCAR mask during training."""

    def __init__(self, missing_rate=0.0, seed=None):
        self.missing_rate = missing_rate
        self.seed = seed

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            data = torch.stack([b['data'] for b in batch])
            mask = torch.stack([b['mask'] for b in batch]) if 'mask' in batch[0] else None
        else:
            data = torch.stack([torch.tensor(b, dtype=torch.float32) for b in batch])
            mask = None
        if self.missing_rate > 0:
            m = (torch.rand_like(data[:, :, 0]) > self.missing_rate).float()
            mask = m.unsqueeze(-1) if data.dim() == 3 else m
        elif mask is None:
            mask = torch.ones_like(data[:, :, 0]).unsqueeze(-1) if data.dim() == 3 else torch.ones_like(data)
        return {'data': data, 'mask': mask}


def build_dataloader(config, mode='train', missing_rate=0.0):
    if mode == 'train':
        cfg = config['dataloader']['train_dataset']
    elif mode == 'val':
        cfg = config['dataloader'].get('val_dataset', config['dataloader']['train_dataset'])
    else:
        cfg = config['dataloader']['test_dataset']
    dataset = instantiate_from_config(cfg)
    batch_size = config['dataloader'].get('batch_size', 32)
    shuffle = (mode == 'train')
    collate = DDICollateFn(missing_rate=missing_rate if mode == 'train' else 0.0)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['dataloader'].get('num_workers', 0),
        drop_last=config['dataloader'].get('drop_last', mode == 'train'),
        pin_memory=config['dataloader'].get('pin_memory', torch.cuda.is_available()),
        collate_fn=collate,
    )
    return {'dataloader': dl, 'dataset': dataset}
