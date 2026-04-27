from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


class Level1BinaryClassifier(nn.Module):
    """Lightweight edge model for normal vs suspicious detection (Reptile meta-learned).

    Architecture must mirror the training notebook exactly:
    LayerNorm (not BatchNorm), GELU (not ReLU), dropout 0.10/0.05.
    """

    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class Level2OpenSetClassifier(nn.Module):
    """Cloud model: cosine-similarity classifier with L2-normalised embeddings.

    Architecture must mirror the training notebook exactly:
    LayerNorm, GELU, dropout 0.15/0.10, L2-normalised embeddings,
    weight-normalised classifier with scale factor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        embedding_dim: int,
        num_classes: int,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim_2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.scale = scale

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Run encoder and L2-normalise the output (unit-sphere embedding)."""
        feat = self.encoder(features)
        return F.normalize(feat, p=2, dim=1)

    def forward_with_embedding(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, logits) with normalised-weight cosine scoring."""
        feat = self.encode(features)
        weight = F.normalize(self.classifier.weight, p=2, dim=1)
        logits = self.scale * F.linear(feat, weight)
        return feat, logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, logits = self.forward_with_embedding(features)
        return logits


def build_level1_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Level1BinaryClassifier:
    input_dim = int(state_dict["net.0.weight"].shape[1])
    hidden_dim_1 = int(state_dict["net.0.weight"].shape[0])
    hidden_dim_2 = int(state_dict["net.4.weight"].shape[0])
    output_dim = int(state_dict["net.8.weight"].shape[0])

    model = Level1BinaryClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        output_dim=output_dim,
    )
    model.load_state_dict(dict(state_dict), strict=False)
    model.eval()
    return model


def build_level2_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Level2OpenSetClassifier:
    input_dim = int(state_dict["encoder.0.weight"].shape[1])
    hidden_dim_1 = int(state_dict["encoder.0.weight"].shape[0])
    hidden_dim_2 = int(state_dict["encoder.4.weight"].shape[0])
    embedding_dim = int(state_dict["encoder.8.weight"].shape[0])
    num_classes = int(state_dict["classifier.weight"].shape[0])

    model = Level2OpenSetClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )
    model.load_state_dict(dict(state_dict), strict=False)
    model.eval()
    return model
