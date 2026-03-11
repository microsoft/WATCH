# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LSTM-based model for weakly-supervised monthly change detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """MLP encoder used before the LSTM.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor, return_proj: bool = False) -> torch.Tensor:
        h = self.backbone(x)
        if return_proj:
            return self.proj(h)
        return h


class LSTMChangeDetector(nn.Module):
    """Sequence model with an optional per-time change head.

    For this monthly pipeline we primarily use `time_head` to produce per-month
    logits (B, T), which we sigmoid to probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        enc_hidden: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=enc_hidden, proj_dim=enc_hidden)
        self.lstm = nn.LSTM(
            input_size=enc_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.pre_head_norm = nn.LayerNorm(lstm_hidden)
        self.time_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden // 2, 1),
        )

        for m in self.time_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)

    def forward_time_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-month logits with shape (B, T)."""
        B, TT, F = x.shape
        h = self.encoder(x.view(B * TT, F), return_proj=False).view(B, TT, -1)
        out, _ = self.lstm(h)
        # Optional normalization of last hidden isn't needed for time head
        t_logits = self.time_head(out).squeeze(-1)
        return t_logits


def infer_arch_from_state_dict(sd: dict) -> dict:
    """Infer model architecture from a saved state_dict."""
    enc_w = sd.get("encoder.backbone.0.weight")
    if enc_w is None:
        raise ValueError("Cannot infer architecture: missing encoder.backbone.0.weight")
    enc_hidden, input_dim = enc_w.shape

    lstm_w0 = sd.get("lstm.weight_ih_l0")
    if lstm_w0 is None:
        raise ValueError("Cannot infer architecture: missing lstm.weight_ih_l0")
    lstm_hidden = lstm_w0.shape[0] // 4

    lstm_layers = len([k for k in sd.keys() if k.startswith("lstm.weight_ih_l")])

    return {
        "input_dim": int(input_dim),
        "enc_hidden": int(enc_hidden),
        "lstm_hidden": int(lstm_hidden),
        "lstm_layers": int(lstm_layers),
    }
