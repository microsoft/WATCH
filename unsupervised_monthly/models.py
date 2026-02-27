# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        z = self.proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        return z

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, depth: int = 4, nhead: int = 8, ff: int = 512, dropout: float = 0.1, mask_ratio: float = 0.3):
        super().__init__()
        self.backbone = TemporalTransformer(input_dim, d_model, nhead, depth, ff, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(),
            nn.Linear(ff, input_dim)
        )
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, T, Fdim = x.shape
        z = self.backbone(x)
        mask = torch.rand(B, T, device=x.device) < self.mask_ratio
        recon = self.decoder(z)
        if mask.any().item():
            loss = F.mse_loss(recon[mask], x[mask])
        else:
            # Fallback to full reconstruction loss to keep gradients flowing
            loss = F.mse_loss(recon, x)
        return loss, recon, mask

class NextMonthForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, depth: int = 3, nhead: int = 8, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.backbone = TemporalTransformer(input_dim, d_model, nhead, depth, ff, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(),
            nn.Linear(ff, input_dim)
        )

    def forward(self, x):
        z = self.backbone(x)
        pred = self.head(z[:, :-1, :])
        target = x[:, 1:, :]
        loss = F.mse_loss(pred, target)
        return loss, pred
