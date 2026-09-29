"""Transformer-LSTM order view with positional encoding and padding masks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class SinusoidalPositionEncoding(nn.Module):
    """Parameter-free positional signal that grows with the current batch."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        position = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(
            1
        )
        exponent = torch.arange(
            0, self.dimension, 2, device=x.device, dtype=torch.float32
        )
        exponent = torch.exp(exponent * (-math.log(10000.0) / self.dimension))
        encoding = torch.zeros(
            length, self.dimension, device=x.device, dtype=torch.float32
        )
        encoding[:, 0::2] = torch.sin(position * exponent)
        if self.dimension > 1:
            encoding[:, 1::2] = torch.cos(
                position * exponent[: encoding[:, 1::2].shape[1]]
            )
        return x + encoding.to(dtype=x.dtype).unsqueeze(0)


class OrderBranch(nn.Module):
    """Transformer context followed by the last valid LSTM hidden state."""

    def __init__(self, event_input_dim: int, branch_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(event_input_dim, branch_dim)
        self.position = SinusoidalPositionEncoding(branch_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.lstm = nn.LSTM(branch_dim, 128, num_layers=1, batch_first=True)
        self.output_projection = nn.Linear(128, branch_dim)
        self.output_norm = nn.LayerNorm(branch_dim)
        self.activation = nn.GELU()

    def forward(
        self, event_features: Tensor, event_mask: Tensor, lengths: Tensor
    ) -> Tensor:
        mask = event_mask.unsqueeze(-1).to(event_features.dtype)
        encoded = self.input_projection(event_features) * mask
        encoded = self.position(encoded) * mask
        encoded = self.transformer(encoded, src_key_padding_mask=~event_mask) * mask
        packed = pack_padded_sequence(
            encoded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output_norm(self.activation(self.output_projection(hidden[-1])))
