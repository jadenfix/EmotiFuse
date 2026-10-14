"""Convenience inference API."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

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
_checkpoint_fingerprint: tuple | None = None
_cache_lock = RLock()
_device = get_default_device()


def _file_fingerprint(path: str) -> tuple:
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _load_model(checkpoint: str | None = None) -> EmotiFuse:
    global _model, _checkpoint, _checkpoint_fingerprint
    if checkpoint is None:
        raise ValueError("A trained EmotiFuse checkpoint is required for emotion prediction")
    resolved = str(Path(checkpoint).resolve())
    with _cache_lock:
        fingerprint = _file_fingerprint(resolved)
        if (_model is None or _checkpoint != resolved
                or _checkpoint_fingerprint != fingerprint):
            # Publish only a fully loaded, unchanged file. Replacement files
            # at the same path invalidate the cache; failed loads preserve it.
            state = torch.load(resolved, map_location=_device, weights_only=True)
            candidate = EmotiFuse().to(_device).eval()
            candidate.load_state_dict(state)
            if _file_fingerprint(resolved) != fingerprint:
                raise RuntimeError("Checkpoint changed while loading; retry with a stable file")
            _model, _checkpoint, _checkpoint_fingerprint = candidate, resolved, fingerprint
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
        if logits.ndim != 2 or logits.shape != (1, 8):
            raise ValueError("Checkpoint must produce one prediction over the eight emotion labels")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite emotion logits; checkpoint or acoustic inputs require validation")
        probabilities = logits.softmax(dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Non-finite emotion probabilities; refusing an invalid prediction")
        pred = probabilities.argmax(dim=-1).item()

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