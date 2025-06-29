"""High-level EmotiFuse model."""

from __future__ import annotations

import torch
from torch import nn

from .branches import Wav2VecBranch, ClassicalBranch, SpecTransformerBranch
from .fusion import GatedCrossAttention
from .ncde_model import EmotiCDE

__all__ = [
    "EmotiFuse",
]


class EmotiFuse(nn.Module):
    """Full multi-branch model as described in the design doc."""

    def __init__(self, emb_dim: int = 256, hidden_dim: int = 512, n_classes: int = 8):
        super().__init__()
        self.wav2vec = Wav2VecBranch(proj_dim=emb_dim)
        self.classical = ClassicalBranch(emb_dim=emb_dim)
        self.spect = SpecTransformerBranch(embed_dim=emb_dim)

        self.fusion = GatedCrossAttention(dim=emb_dim)
        self.cde = EmotiCDE(input_dim=emb_dim, hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, wavs: torch.Tensor, feats: dict[str, torch.Tensor], specs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward three modalities in parallel.

        Parameters
        ----------
        wavs
            Waveforms of shape ``(B, T)`` at 16 kHz.
        feats
            Dict of handcrafted features ``(B, T, f)``.
        specs
            Log-mel spectrograms ``(B, F, T)``.
        """

        e = self.wav2vec(wavs)
        m = self.classical(feats)
        s = self.spect(specs)

        h = self.fusion(e, m, s)
        z = self.cde(h)
        out = self.classifier(z)
        return out 