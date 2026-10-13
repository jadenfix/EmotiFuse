"""Convenience inference API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .model import EmotiFuse
from .preprocessing import resample_mono, pre_emphasis, voice_activity_detection
from .utils import get_default_device
from . import features as _features

__all__ = [
    "infer",
]

_model: EmotiFuse | None = None
_checkpoint: str | None = None
_device = get_default_device()


def _load_model(checkpoint: str | None = None) -> EmotiFuse:  # pragma: no cover
    global _model, _checkpoint
    if checkpoint is None:
        raise ValueError("A trained EmotiFuse checkpoint is required for emotion prediction")
    resolved = str(Path(checkpoint).resolve())
    if _model is None or _checkpoint != resolved:
        # Publish a new cache entry only after the checkpoint loads completely.
        state = torch.load(resolved, map_location=_device, weights_only=True)
        candidate = EmotiFuse().to(_device).eval()
        candidate.load_state_dict(state)
        _model, _checkpoint = candidate, resolved
    return _model


def infer(wav_path: str | Path, checkpoint: str | None = None) -> str:  # noqa: D401
    """Return predicted emotion label for *wav_path*."""

    if checkpoint is None:
        raise ValueError("A trained EmotiFuse checkpoint is required for emotion prediction")
    wav, sr = sf.read(str(wav_path))
    wav = torch.tensor(wav).float()
    if wav.ndim == 2:
        wav = wav.transpose(0, 1)  # soundfile returns (time, channels)

    wav, _ = resample_mono(wav, sr, 16_000)
    wav = pre_emphasis(wav)
    wav = voice_activity_detection(wav)
    if wav.numel() == 0:
        raise ValueError("No speech detected in input signal.")

    device = _device
    wav = wav.to(device)

    # Classical features using NumPy API
    feats = _features.extract_features(wav.cpu().numpy(), 16_000)
    feats = {k: torch.tensor(feats[k]).transpose(0, 1).unsqueeze(0).to(device).float()
             for k in ("mfcc", "delta", "delta2")}

    # Spectrogram
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16_000, n_fft=400,
                                              hop_length=160, n_mels=64).to(device)
    spec = torch.log1p(mel(wav)).unsqueeze(0)

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