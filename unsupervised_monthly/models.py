# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Transformer-based models for unsupervised monthly change detection."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 512):
        """Initialize PositionalEncoding.

        Args:
            d_model: Dimension of the model embeddings.
            max_len: Maximum sequence length supported.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Tensor of the same shape with positional encoding added.
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TemporalTransformer(nn.Module):
    """Transformer encoder that maps a (B, T, input_dim) time series to (B, T, d_model) embeddings."""

    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        """Initialize TemporalTransformer.

        Args:
            input_dim: Dimensionality of the input features per time step.
            d_model: Hidden dimension of the transformer.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of the feed-forward network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=1024)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        """Encode the input time series.

        Args:
            x: Input tensor of shape (B, T, input_dim).

        Returns:
            Encoded tensor of shape (B, T, d_model).
        """
        z = self.proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        return z

class MaskedAutoencoder(nn.Module):
    """Masked autoencoder that reconstructs randomly masked time steps."""

    def __init__(self, input_dim: int, d_model: int = 256, depth: int = 4, nhead: int = 8, ff: int = 512, dropout: float = 0.1, mask_ratio: float = 0.3):
        """Initialize MaskedAutoencoder.

        Args:
            input_dim: Dimensionality of input features per time step.
            d_model: Hidden dimension.
            depth: Number of transformer encoder layers.
            nhead: Number of attention heads.
            ff: Feed-forward dimension.
            dropout: Dropout rate.
            mask_ratio: Fraction of time steps to mask during training.
        """
        super().__init__()
        self.backbone = TemporalTransformer(input_dim, d_model, nhead, depth, ff, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(),
            nn.Linear(ff, input_dim)
        )
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """Run forward pass with random masking.

        Args:
            x: Input tensor of shape (B, T, input_dim).

        Returns:
            Tuple of (loss, reconstruction, mask) where loss is the MSE on
            masked positions, reconstruction is the full (B, T, input_dim)
            output, and mask is a boolean (B, T) tensor.
        """
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
    """Predicts the next month's features from preceding time steps."""

    def __init__(self, input_dim: int, d_model: int = 256, depth: int = 3, nhead: int = 8, ff: int = 512, dropout: float = 0.1):
        """Initialize NextMonthForecaster.

        Args:
            input_dim: Dimensionality of input features per time step.
            d_model: Hidden dimension.
            depth: Number of transformer encoder layers.
            nhead: Number of attention heads.
            ff: Feed-forward dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.backbone = TemporalTransformer(input_dim, d_model, nhead, depth, ff, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(),
            nn.Linear(ff, input_dim)
        )

    def forward(self, x):
        """Predict each next time step from the preceding context.

        Args:
            x: Input tensor of shape (B, T, input_dim).

        Returns:
            Tuple of (loss, predictions) where loss is the MSE between
            predicted and actual next-month features, and predictions has
            shape (B, T-1, input_dim).
        """
        z = self.backbone(x)
        pred = self.head(z[:, :-1, :])
        target = x[:, 1:, :]
        loss = F.mse_loss(pred, target)
        return loss, pred
