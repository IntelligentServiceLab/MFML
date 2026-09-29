"""Residual temporal CNN attribute view with masked attention pooling."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class MaskedGroupNorm1d(nn.Module):
    """GroupNorm over valid time steps only.

    Native GroupNorm includes right-padding positions in its statistics, which
    makes a prefix representation change when the batch happens to contain a
    longer peer.  This equivalent masked form preserves padding invariance.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels % num_groups:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor, event_mask: Tensor) -> Tensor:
        batch, channels, length = x.shape
        grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, length)
        mask = event_mask[:, None, None, :].to(dtype=x.dtype)
        count = mask.sum(dim=(2, 3), keepdim=True) * (channels // self.num_groups)
        mean = (grouped * mask).sum(dim=(2, 3), keepdim=True) / count.clamp_min(1.0)
        variance = ((grouped - mean).square() * mask).sum(dim=(2, 3), keepdim=True)
        variance = variance / count.clamp_min(1.0)
        normalized = ((grouped - mean) * torch.rsqrt(variance + self.eps)).reshape_as(x)
        normalized = normalized * self.weight[None, :, None] + self.bias[None, :, None]
        return normalized * event_mask[:, None, :].to(dtype=x.dtype)


class MaskedResidualTCNBlock(nn.Module):
    def __init__(
        self, channels: int, groups: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, 3, padding=padding, dilation=dilation
        )
        self.norm1 = MaskedGroupNorm1d(groups, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, 3, padding=padding, dilation=dilation
        )
        self.norm2 = MaskedGroupNorm1d(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, event_mask: Tensor) -> Tensor:
        mask = event_mask[:, None, :].to(dtype=x.dtype)
        residual = x * mask
        y = self.conv1(residual) * mask
        y = self.dropout(self.activation(self.norm1(y, event_mask))) * mask
        y = self.conv2(y) * mask
        y = self.dropout(self.activation(self.norm2(y, event_mask))) * mask
        return (residual + y) * mask


class AttributeBranch(nn.Module):
    """Masked residual 1-D TCN with attention pooling."""

    def __init__(
        self,
        event_input_dim: int,
        channels: int,
        groups: int,
        dilations: Sequence[int],
        branch_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(event_input_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            MaskedResidualTCNBlock(channels, groups, dilation, dropout)
            for dilation in dilations
        )
        self.attention = nn.Linear(channels, 1)
        self.output_projection = nn.Linear(channels, branch_dim)
        self.output_norm = nn.LayerNorm(branch_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_features: Tensor, event_mask: Tensor) -> Tensor:
        mask_channels = event_mask[:, None, :].to(dtype=event_features.dtype)
        encoded = self.input_projection(event_features.transpose(1, 2)) * mask_channels
        for block in self.blocks:
            encoded = block(encoded, event_mask)
        sequence = encoded.transpose(1, 2)
        scores = self.attention(sequence).squeeze(-1)
        scores = scores.masked_fill(~event_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * event_mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        result = self.output_projection(pooled)
        return self.output_norm(self.dropout(self.activation(result)))
