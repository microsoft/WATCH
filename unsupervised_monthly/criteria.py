"""Losses and criteria."""

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_offdiag: float = 5e-3):
        super().__init__()
        self.lmb = lambda_offdiag

    def forward(self, z1, z2):
        N, D = z1.shape
        if N < 2:
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        mean1 = z1.mean(dim=0)
        mean2 = z2.mean(dim=0)
        std1 = z1.std(dim=0, unbiased=False).clamp_min(1e-6)
        std2 = z2.std(dim=0, unbiased=False).clamp_min(1e-6)
        z1n = (z1 - mean1) / std1
        z2n = (z2 - mean2) / std2
        c = (z1n.T @ z2n) / float(N)
        on = (torch.diagonal(c) - 1.0).pow(2).sum()
        off = (c - torch.eye(D, device=z1.device, dtype=c.dtype)).pow(2).sum() - on
        return on + self.lmb * off

class TemporalOrderLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.tau = temperature

    def forward(self, z):
        B, T, D = z.shape
        if T < 3:
            return torch.tensor(0.0, device=z.device)
        loss = 0.0
        steps = 0
        for t in range(1, T-1):
            q = z[:, t, :]
            pos = (z[:, t-1, :] + z[:, t+1, :]) / 2.0
            neg_idx = list(range(0, max(0, t-2))) + list(range(min(T, t+3), T))
            if len(neg_idx) == 0:
                continue
            neg = z[:, neg_idx, :].reshape(B, -1, D)
            qn = F.normalize(q, dim=1)
            pn = F.normalize(pos, dim=1)
            nnorm = F.normalize(neg, dim=2)
            logits_pos = (qn * pn).sum(dim=1, keepdim=True) / self.tau
            logits_neg = torch.bmm(nnorm, qn.unsqueeze(-1)).squeeze(-1) / self.tau
            logits = torch.cat([logits_pos, logits_neg], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=z.device)
            loss += F.cross_entropy(logits, labels)
            steps += 1
        return loss / max(1, steps)
