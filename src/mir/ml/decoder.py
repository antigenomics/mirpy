"""Inverse codec: reconstruct a CDR3 sequence from its (compact) embedding.

A **parallel** decoder (irrm-codec style): the compact PCA code is expanded and
refined by 1-D convolutions into length-40 per-position token logits in one shot
(no autoregression). Trained with cross-entropy against the fixed-length-40
tokenization; the reconstructed string drops gap tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from mir.ml.tokenize import AA, FIXED_LEN, N_TOKENS

_GAP_IDX = N_TOKENS - 1


class SequenceDecoder(nn.Module):
    """Compact embedding code → ``(B, 40, 21)`` per-position token logits.

    Args:
        code_dim: Input code width (e.g. #PCs of the compact embedding).
        hidden: Width of the expansion MLP.
        channels: Conv refinement channels (first is the reshaped depth).
        kernel: Conv kernel size.
    """

    def __init__(
        self,
        code_dim: int,
        hidden: int = 512,
        channels: tuple[int, ...] = (128, 64),
        kernel: int = 3,
    ):
        super().__init__()
        c0 = channels[0]
        self._c0 = c0
        self.fc = nn.Sequential(
            nn.Linear(code_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c0 * FIXED_LEN),
            nn.ReLU(),
        )
        layers: list[nn.Module] = []
        c_in = c0
        for c_out in channels[1:]:
            layers += [
                nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            ]
            c_in = c_out
        layers += [nn.Conv1d(c_in, N_TOKENS, kernel, padding=kernel // 2)]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self._c0, FIXED_LEN)
        return self.conv(h).transpose(1, 2)  # (B, FIXED_LEN, N_TOKENS)


def tokens_to_seq(idx_row) -> str:
    """Map a length-40 token-index row back to an amino-acid string (gaps dropped)."""
    return "".join(AA[i] for i in idx_row if 0 <= int(i) < len(AA))
