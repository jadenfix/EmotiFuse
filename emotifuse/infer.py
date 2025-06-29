"""Convenience inference API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .model import EmotiFuse
from .preprocessing import resample_mono, pre_emphasis, voice_activity_detection
from .utils import get_default_device
from . import features as _features

__all__ = [
    "infer",
]

_model: EmotiFuse | None = None
_device = get_default_device()


def _load_model(checkpoint: str | None = None) -> EmotiFuse:  # pragma: no cover
    global _model
    if _model is None:
        _model = EmotiFuse().to(_device).eval()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=_device)
            _model.load_state_dict(state)
    return _model


def infer(wav_path: str | Path, checkpoint: str | None = None) -> str:  # noqa: D401
    """Return predicted emotion label for *wav_path*."""

    wav, sr = sf.read(str(wav_path))
    wav = torch.tensor(wav).float()

    wav, _ = resample_mono(wav, sr, 16_000)
    wav = pre_emphasis(wav)
    wav = voice_activity_detection(wav)
    if wav.numel() == 0:
        raise ValueError("No speech detected in input signal.")

    device = _device
    wav = wav.to(device)

    # Classical features using NumPy API
    feats = _features.extract_features(wav.cpu().numpy(), 16_000)
    feats = {k: torch.tensor(v).unsqueeze(0).to(device).float().transpose(1, 2) for k, v in feats.items()}

    # Spectrogram
    spec = torch.log1p(torch.stft(
        wav,
        n_fft=400,
        hop_length=160,
        window=torch.hann_window(400, device=device),
        return_complex=True,
    ).abs() ** 2)
    spec = spec.unsqueeze(0)  # (1, F, T)

    model = _load_model(checkpoint)
    with torch.no_grad():
        logits = model(wavs=wav.unsqueeze(0), feats=feats, specs=spec)
        pred = logits.softmax(dim=-1).argmax(dim=-1).item()

    # Map to label names (placeholder)
    idx2label = {
        0: "neutral",
        1: "happy",
        2: "sad",
        3: "angry",
        4: "fearful",
        5: "disgust",
        6: "surprised",
        7: "calm",
    }

    return idx2label.get(pred, str(pred)) 