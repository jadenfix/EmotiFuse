"""Classical acoustic feature MLP branch."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

# Assuming features extracted via features.extract_features -> numpy arrays
__all__ = [
    "ClassicalBranch",
]

class ClassicalBranch(nn.Module):
    """Map handcrafted features to frame-wise embeddings using an MLP."""

    def __init__(self, in_dim: int = 39, hidden_dim: int = 128, emb_dim: int = 256):
        """Parameters
        ----------
        in_dim
            Input feature dimension (e.g., 13 MFCC * 3 = 39).
        hidden_dim
            Hidden units of MLP.
        emb_dim
            Output embedding dimension.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward per-frame features dict → embedding.

        Expects keys ``mfcc``, ``delta``, ``delta2`` of shape ``(B, T, 13)``.
        Concatenates along last dimension to form ``in_dim``.
        """
        mfcc = feats["mfcc"]
        delta = feats["delta"]
        delta2 = feats["delta2"]
        x = torch.cat([mfcc, delta, delta2], dim=-1)
        return self.mlp(x) 