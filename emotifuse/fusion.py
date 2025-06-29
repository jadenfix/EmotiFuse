"""Cross-modal fusion with gated cross-attention."""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "GatedCrossAttention",
]


class GatedCrossAttention(nn.Module):
    """Fuse three modalities (e, m, s) with learned gating."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, e: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """All inputs shape ``(B, T, dim)``.

        Returns fused representation of same shape.
        """

        stacked = torch.cat([e, m, s], dim=-1)
        g = self.gate(stacked)  # (B, T, dim)

        # Use *e* as query, key/value are concat of m & s
        kv = torch.cat([m, s], dim=1)  # (B, 2T, dim)
        q = e * g  # modulate query

        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out 