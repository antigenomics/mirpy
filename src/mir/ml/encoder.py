"""Forward codec: a CNN that maps a CDR3 sequence to its TCREMP embedding.

The GPU-friendly approximation of the prototype mapping (irrm-codec forward model).
Trained with **free supervision** — targets are TCREMP embeddings *computed* by
:class:`mir.embedding.tcremp.TCREmp`, so labelled data is never the bottleneck.
"""

from __future__ import annotations

import torch
from torch import nn

from mir.ml.tokenize import FIXED_LEN, N_TOKENS


class SequenceEncoder(nn.Module):
    """1-D CNN over the length-40 one-hot CDR3 → a fixed embedding vector.

    Args:
        embed_dim: Output dimensionality (the target embedding width).
        channels: Conv channel sizes.
        kernel: Conv kernel size.
        hidden: Width of the dense head.
        dropout: Dropout on the dense head.
    """

    def __init__(
        self,
        embed_dim: int,
        channels: tuple[int, ...] = (64, 128),
        kernel: int = 3,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = N_TOKENS
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * FIXED_LEN, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, FIXED_LEN, N_TOKENS) -> conv wants (B, N_TOKENS, FIXED_LEN)
        return self.head(self.conv(x.transpose(1, 2)))
