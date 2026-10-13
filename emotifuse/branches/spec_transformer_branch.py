"""Spectrogram Transformer branch.

Implements a small Vision-Transformer-style encoder over log-mel spectrograms.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "SpecTransformerBranch",
]


def _make_encoder(embed_dim: int = 256, num_layers: int = 2, num_heads: int = 4) -> nn.Module:
    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class SpecTransformerBranch(nn.Module):
    """Patchified log-mel spectrogram → Transformer Encoder."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Linear(patch_size, embed_dim)
        self.encoder = _make_encoder(embed_dim, num_layers, num_heads)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Input log-mel spectrogram of shape ``(B, F, T)``.

        Breaks frequency dimension into patches of ``patch_size``.
        Returns output embeddings per frame ``(B, T, embed_dim)``.
        """

        if spec.ndim != 3 or min(spec.shape) < 1:
            raise ValueError("spectrogram must have shape (B, F, T) with nonempty axes")
        B, F, T = spec.shape
        # Project each frequency patch, then aggregate patches per time frame.
        padding = (-F) % self.patch_size
        spec = nn.functional.pad(spec, (0, 0, 0, padding))
        n_patches = spec.size(1) // self.patch_size
        x = spec.reshape(B, n_patches, self.patch_size, T).permute(0, 3, 1, 2)
        x = self.proj(x).mean(dim=2)

        # Sinusoidal time positions use O(T * D) memory and work at any length.
        positions = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, self.embed_dim, 2, device=x.device,
                                              dtype=x.dtype) * (-torch.log(x.new_tensor(10000.0)) / self.embed_dim))
        encoding = x.new_zeros(T, self.embed_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:self.embed_dim // 2])
        return self.encoder(x + encoding.unsqueeze(0))
