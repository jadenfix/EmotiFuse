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

        B, F, T = spec.shape
        if F % self.patch_size != 0:
            raise ValueError("Frequency bins must be divisible by patch_size")

        # Reshape into patches along frequency: (B, T, n_patches, patch_size)
        n_patches = F // self.patch_size
        x = spec.view(B, n_patches, self.patch_size, T)  # (B, n_patches, patch_size, T)
        x = x.permute(0, 3, 1, 2)  # (B, T, n_patches, patch_size)
        x = x.flatten(2)  # (B, T, n_patches * patch_size)

        # Project
        x = self.proj(x)  # (B, T, embed_dim)

        # Positional encoding (learned)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_embed = nn.functional.embedding(pos, torch.eye(T, device=x.device, dtype=x.dtype))
        pos_embed = pos_embed.expand(B, -1, -1)  # (B, T, T)
        x = x + pos_embed[..., : self.embed_dim]

        x = self.encoder(x)
        return x 