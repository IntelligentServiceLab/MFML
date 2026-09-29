"""Medium-BERT semantic view of the observed event prefix."""

from __future__ import annotations

from torch import Tensor, nn


class SemanticBranch(nn.Module):
    """Masked-mean language-model encoder projected to the fusion space."""

    def __init__(self, encoder: nn.Module, branch_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("the injected BERT encoder must expose config.hidden_size")
        self.projection = nn.Linear(int(hidden_size), branch_dim)
        self.norm = nn.LayerNorm(branch_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.dropout(self.norm(self.activation(self.projection(pooled))))
