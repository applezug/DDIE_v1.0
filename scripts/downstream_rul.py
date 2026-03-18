#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RUL (Remaining Useful Life) prediction downstream task.
Uses last m=10 capacity values from imputed sequence to predict RUL.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.metric_utils import phm_score


class RULLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def build_rul_dataset(capacity_sequences, rul_labels, m=10):
    """capacity_sequences: list of (L,) arrays. rul_labels: list of RUL values."""
    X, y = [], []
    for seq, rul in zip(capacity_sequences, rul_labels):
        if len(seq) < m:
            continue
        X.append(seq[-m:].reshape(m, -1))
        y.append(rul)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_rul_model(X_train, y_train, epochs=100, lr=1e-3):
    model = RULLSTM(input_size=X_train.shape[2] if X_train.ndim == 3 else 1, hidden_size=64, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def evaluate_rul(model, X_test, y_test):
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32))
        pred = pred.numpy().flatten()
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    score = phm_score(pred, y_test)
    return {'RMSE': rmse, 'PHM_Score': score}
