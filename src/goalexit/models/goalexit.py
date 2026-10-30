from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class GlobalStatsMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class CausalGazeGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        head_size: int = 64,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if hidden_size < 1 or num_layers < 1 or head_size < 1:
            raise ValueError("GRU dimensions and layer count must be positive")
        if not 0.0 <= recurrent_dropout < 1.0:
            raise ValueError("Recurrent dropout must be in [0, 1)")
        if bidirectional:
            raise ValueError("Deployment GRU must remain causal")
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def encode(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(self.input_norm(values))
        positions = lengths.to(device=values.device, dtype=torch.long) - 1
        return output[torch.arange(len(values), device=values.device), positions]

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(values, lengths)).squeeze(-1)


class PrivilegedTeacher(nn.Module):
    def __init__(
        self,
        *,
        maximum_words: int,
        text_only: bool = False,
        gaze_only: bool = False,
        dropout: float = 0.1,
        position_dim: int = 32,
    ):
        super().__init__()
        if text_only and gaze_only:
            raise ValueError("Teacher cannot be both text-only and gaze-only")
        self.text_only = bool(text_only)
        self.gaze_only = bool(gaze_only)
        self.text_projection = nn.Linear(768, 128)
        self.gaze_projection = nn.Linear(8, 64)
        self.position_embedding = nn.Embedding(maximum_words, position_dim)
        self.fusion_projection = nn.Linear(128 + 64 + position_dim, 192)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=384,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.classifier = nn.Linear(192, 1)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        gaze_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if text_embeddings.ndim != 3 or gaze_features.ndim != 3:
            raise ValueError("Teacher inputs must be B x words x features")
        if text_embeddings.shape[:2] != gaze_features.shape[:2]:
            raise ValueError("Teacher text/gaze word dimensions differ")
        batch, words, _ = text_embeddings.shape
        if words > self.position_embedding.num_embeddings:
            raise ValueError("Teacher word sequence exceeds frozen position capacity")
        text_input = torch.zeros_like(text_embeddings) if self.gaze_only else text_embeddings
        text = self.text_projection(text_input)
        gaze_input = torch.zeros_like(gaze_features) if self.text_only else gaze_features
        gaze = self.gaze_projection(gaze_input)
        positions = self.position_embedding(
            torch.arange(words, device=text_embeddings.device)
        ).unsqueeze(0).expand(batch, -1, -1)
        fused = self.fusion_projection(torch.cat([text, gaze, positions], dim=-1))
        cls = self.cls_token.expand(batch, -1, -1)
        values = torch.cat([cls, fused], dim=1)
        mask = torch.cat(
            [
                torch.zeros((batch, 1), dtype=torch.bool, device=padding_mask.device),
                padding_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(values, src_key_padding_mask=mask)
        return self.classifier(encoded[:, 0]).squeeze(-1)


def make_deployment_model(
    model: str,
    input_dim: int,
    *,
    dropout: float = 0.2,
    student_architecture: Mapping[str, Any] | None = None,
) -> nn.Module:
    if model == "b0":
        return GlobalStatsMLP(input_dim, dropout=dropout)
    if model in {"b1", "d", "a", "d_gaze", "ls"}:
        architecture = dict(student_architecture or {})
        head = architecture.get("head")
        head_size = (
            int(architecture["head_size"])
            if "head_size" in architecture
            else int(head[-2])
            if isinstance(head, (list, tuple)) and len(head) >= 3
            else 64
        )
        return CausalGazeGRU(
            input_dim=input_dim,
            hidden_size=int(architecture.get("hidden_size", 64)),
            num_layers=int(architecture.get("gru_layers", 2)),
            head_size=head_size,
            dropout=dropout,
            recurrent_dropout=float(architecture.get("recurrent_dropout", 0.0)),
            bidirectional=bool(architecture.get("bidirectional", False)),
        )
    raise ValueError(f"Unauthorized deployment model {model}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }
