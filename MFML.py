"""MFML: three-view gated fusion for next activity and remaining time."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from attribute import AttributeBranch
from order import OrderBranch
from semantic import SemanticBranch


class PredictionHead(nn.Module):
    def __init__(self, branch_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Intentionally no output activation: CrossEntropyLoss expects logits,
        # and negative standardized log-time predictions must retain gradients.
        return self.layers(x)


class MFML(nn.Module):
    """Full MFML model with the paper's architecture defaults.

    Attribute order follows ``categorical_cardinalities``. Each vocabulary
    includes PAD=0 and UNK=1. Optional CBOW matrices must include those rows.
    ``remaining_time_z`` predicts standardized log1p remaining days.
    """

    def __init__(
        self,
        categorical_cardinalities: Mapping[str, int],
        num_activities: int,
        bert_model_name_or_path: str = "prajjwal1/bert-medium",
        *,
        embedding_dim: int = 32,
        branch_dim: int = 256,
        dropout: float = 0.2,
        embedding_initial_weights: Optional[Mapping[str, Tensor]] = None,
        bert_encoder: Optional[nn.Module] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if not categorical_cardinalities or any(
            size < 2 for size in categorical_cardinalities.values()
        ):
            raise ValueError("Each categorical vocabulary must include PAD=0 and UNK=1")
        if num_activities < 2 or embedding_dim < 1 or branch_dim < 1 or branch_dim % 4:
            raise ValueError(
                "Use at least two activities, positive dimensions, and branch_dim divisible by 4"
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.attribute_names = tuple(categorical_cardinalities)

        if bert_encoder is None:
            from transformers import AutoModel

            bert_encoder = AutoModel.from_pretrained(
                bert_model_name_or_path,
                local_files_only=local_files_only,
            )
        # Medium-BERT embeddings and the bottom four layers are frozen.
        for parameter in bert_encoder.embeddings.parameters():
            parameter.requires_grad = False
        for layer in bert_encoder.encoder.layer[:4]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        self.semantic_branch = SemanticBranch(bert_encoder, branch_dim, dropout)

        weights = embedding_initial_weights or {}
        self.categorical_embeddings = nn.ModuleDict()
        for name, size in categorical_cardinalities.items():
            embedding = nn.Embedding(size, embedding_dim, padding_idx=0)
            if name in weights:
                matrix = torch.as_tensor(weights[name], dtype=embedding.weight.dtype)
                if matrix.shape != embedding.weight.shape:
                    raise ValueError(
                        f"CBOW weights for {name} must have shape {(size, embedding_dim)}"
                    )
                with torch.no_grad():
                    embedding.weight.copy_(matrix)
            with torch.no_grad():
                embedding.weight[0].zero_()
            self.categorical_embeddings[name] = embedding

        self.numeric_projection = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Dropout(dropout)
        )
        event_dim = len(self.attribute_names) * embedding_dim + 32
        self.order_branch = OrderBranch(event_dim, branch_dim, dropout)
        self.attribute_branch = AttributeBranch(
            event_input_dim=event_dim,
            channels=128,
            groups=8,
            dilations=(1, 2, 4),
            branch_dim=branch_dim,
            dropout=dropout,
        )
        self.branch_norms = nn.ModuleList(nn.LayerNorm(branch_dim) for _ in range(3))
        self.gate = nn.Linear(3 * branch_dim, 3)
        self.concat_residual = nn.Linear(3 * branch_dim, branch_dim)
        self.fusion_norm = nn.LayerNorm(branch_dim)
        self.activity_head = PredictionHead(branch_dim, num_activities, dropout)
        self.remaining_time_head = PredictionHead(branch_dim, 1, dropout)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        categorical_ids: Tensor,
        numeric_features: Tensor,
        event_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Encode aligned prefix views; valid events must precede right padding.

        Text inputs: [B, S]. Categorical inputs: [B, T, A]. Numeric inputs:
        [B, T, 2]. Event mask: [B, T]. Numeric columns are log1p-transformed,
        train-standardized seconds since the previous event and case start.
        """
        if categorical_ids.ndim != 3 or categorical_ids.shape[-1] != len(
            self.attribute_names
        ):
            raise ValueError(
                "categorical_ids must have shape [batch, events, attributes]"
            )
        batch, events, _ = categorical_ids.shape
        if numeric_features.shape != (batch, events, 2):
            raise ValueError("numeric_features must have shape [batch, events, 2]")
        if event_mask.shape != (batch, events):
            raise ValueError("event_mask must have shape [batch, events]")
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != batch
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "Text IDs and attention mask must have matching [batch, tokens] shapes"
            )

        device = categorical_ids.device
        categorical_ids = categorical_ids.long()
        numeric_features = numeric_features.to(device=device, dtype=torch.float32)
        event_mask = event_mask.to(device=device, dtype=torch.bool)
        lengths = event_mask.sum(dim=1)
        expected = torch.arange(events, device=device)[None, :] < lengths[:, None]
        if torch.any(lengths < 1) or not torch.equal(event_mask, expected):
            raise ValueError(
                "Every prefix needs at least one event and contiguous right padding"
            )
        input_ids = input_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device)
        if not torch.all(attention_mask.bool().any(dim=1)):
            raise ValueError("Every text prefix must contain at least one valid token")

        mask = event_mask.unsqueeze(-1).to(numeric_features.dtype)
        embedded = (
            torch.cat(
                [
                    self.categorical_embeddings[name](categorical_ids[..., index])
                    for index, name in enumerate(self.attribute_names)
                ],
                dim=-1,
            )
            * mask
        )
        numeric = self.numeric_projection(numeric_features) * mask
        event_features = torch.cat((embedded, numeric), dim=-1) * mask

        semantic = self.semantic_branch(input_ids, attention_mask)
        order = self.order_branch(event_features, event_mask, lengths)
        attribute = self.attribute_branch(event_features, event_mask)
        branches = [
            norm(view)
            for norm, view in zip(self.branch_norms, (semantic, order, attribute))
        ]
        concatenated = torch.cat(branches, dim=-1)
        gates = torch.softmax(self.gate(concatenated), dim=-1)
        weighted = (torch.stack(branches, dim=1) * gates.unsqueeze(-1)).sum(dim=1)
        fused = self.fusion_norm(weighted + self.concat_residual(concatenated))
        return {
            "activity_logits": self.activity_head(fused),
            "remaining_time_z": self.remaining_time_head(fused).squeeze(-1),
            "gates": gates,
        }
