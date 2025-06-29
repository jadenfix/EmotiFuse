"""Wav2Vec-based branch.

Wraps a *transformers* `Wav2Vec2Model` (or fine-tuned classifier variant) and
outputs per-frame embeddings suitable for fusion.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import Wav2Vec2Model, AutoConfig

__all__ = [
    "Wav2VecBranch",
]


def _load_pretrained(name_or_path: str, output_hidden_states: bool = True) -> Wav2Vec2Model:
    cfg = AutoConfig.from_pretrained(name_or_path, output_hidden_states=output_hidden_states)
    return Wav2Vec2Model.from_pretrained(name_or_path, config=cfg)


class Wav2VecBranch(nn.Module):
    """Return frame-level embeddings from a frozen (or trainable) Wav2Vec model."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", trainable: bool = False, proj_dim: Optional[int] = None):
        super().__init__()
        self.wav2vec = _load_pretrained(model_name)
        self.trainable = trainable
        for p in self.wav2vec.parameters():
            p.requires_grad = trainable

        hidden_size = self.wav2vec.config.hidden_size
        self.proj = nn.Identity() if proj_dim is None else nn.Linear(hidden_size, proj_dim)

    def forward(self, wavs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401
        """*wavs* shape: ``(batch, time)``  in 16 kHz."""

        outputs = self.wav2vec(wavs, attention_mask=attention_mask, output_hidden_states=True)
        # Take last hidden state: (B, T, H)
        x = outputs.last_hidden_state
        x = self.proj(x)
        return x 