"""Neural CDE utilities."""

from __future__ import annotations

import torch
from torch import nn
import torchcde

__all__ = [
    "CDEFunc",
    "EmotiCDE",
]


class CDEFunc(nn.Module):
    """Simple linear vector field parameterisation."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim * input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # z shape: (batch, hidden_dim)
        return self.linear(z).view(z.size(0), self.hidden_dim, self.input_dim)


class EmotiCDE(nn.Module):
    """Wrapper around torchcde.cdeint for NCDE integration."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.func = CDEFunc(input_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        """Integrate latent trajectory.

        Parameters
        ----------
        h
            Input sequence ``(B, T, input_dim)``.
        times
            1-D tensor of shape ``(T,)`` with time stamps (default evenly spaced
            0..T-1).
        Returns
        -------
        torch.Tensor
            Final hidden state ``(B, hidden_dim)``.
        """

        if times is None:
            times = torch.linspace(0, h.size(1) - 1, h.size(1), device=h.device)

        X = torchcde.LinearInterpolation(h, t=times)
        z0 = self.initial_linear(h[:, 0])
        z_T = torchcde.cdeint(X=X, func=self.func, z0=z0, t=times, method="rk4")
        return z_T[:, -1]  # final state 