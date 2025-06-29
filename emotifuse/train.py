"""Training script (library style, not CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import EmotiFuse
from .utils import get_default_device

__all__ = [
    "train",
]


def train(dataset: Dataset, epochs: int = 10, batch_size: int = 4, lr: float = 1e-4, ckpt_dir: str | Path = "checkpoints") -> None:  # noqa: D401
    device = get_default_device()

    model = EmotiFuse().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for sample in loader:
            wavs = sample["wav"].to(device)
            feats = {k: v.to(device) for k, v in sample["feats"].items()}
            specs = sample["spec"].to(device)
            labels = sample["label"].to(device)

            optim.zero_grad()
            logits = model(wavs=wavs, feats=feats, specs=specs)
            loss = crit(logits, labels)
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * wavs.size(0)

        epoch_loss /= len(loader.dataset)
        print(f"Epoch {epoch:02d}: loss={epoch_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_dir / f"emotifuse_epoch{epoch:02d}.pt") 