"""Utility helpers shared across the library."""

from __future__ import annotations

import torch

__all__ = [
    "get_default_device",
]


def get_default_device() -> torch.device:  # pragma: no cover
    """Return *cuda* if available, else *cpu*."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 