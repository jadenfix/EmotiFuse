"""Audio preprocessing utilities.

This module normalises input waveforms before feature extraction or model
forward passes.  All functions operate on PyTorch tensors for downstream
compatibility but transparently accept ``numpy.ndarray`` inputs as well.

Key steps implemented:

1. ``resample_mono`` – resample audio to a target sampling rate and convert
   to mono.
2. ``pre_emphasis`` – apply a first-order high-pass (pre-emphasis) filter.
3. ``voice_activity_detection`` – simple energy-threshold VAD that masks
   silent frames (< \-40 dB).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchaudio

__all__ = [
    "resample_mono",
    "pre_emphasis",
    "voice_activity_detection",
]

# -----------------------------------------------------------------------------
# Resampling / Channel handling
# -----------------------------------------------------------------------------

def resample_mono(wav: torch.Tensor, orig_sr: int, target_sr: int = 16_000) -> Tuple[torch.Tensor, int]:
    """Resample *wav* to *target_sr* and ensure mono channel.

    Parameters
    ----------
    wav
        Tensor of shape ``(channels, time)`` or ``(time,)``.
    orig_sr
        Original sampling rate in Hertz.
    target_sr
        Desired sampling rate (default 16 kHz).

    Returns
    -------
    tuple
        ``(mono_wav, target_sr)`` where *mono_wav* is a 1-D tensor.
    """

    if wav.ndim == 2 and wav.size(0) > 1:
        wav = torch.mean(wav, dim=0)  # naive down-mix to mono
    elif wav.ndim == 1:
        pass  # already mono
    else:
        raise ValueError("Expected tensor with shape (channels, time) or (time,)")

    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=target_sr)

    return wav, target_sr


# -----------------------------------------------------------------------------
# Pre-emphasis
# -----------------------------------------------------------------------------

def pre_emphasis(wav: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """Apply a first-order pre-emphasis filter.

    ``y[n] = x[n] - coeff * x[n-1]``
    """

    if wav.ndim != 1:
        raise ValueError("pre_emphasis expects a 1-D mono signal")

    # Pad with zero for x[n-1] when n = 0
    padded = torch.nn.functional.pad(wav.unsqueeze(0), (1, 0), mode="constant").squeeze(0)
    emphasized = wav - coeff * padded[:-1]
    return emphasized


# -----------------------------------------------------------------------------
# Voice Activity Detection (energy-based)
# -----------------------------------------------------------------------------


@torch.no_grad()
def voice_activity_detection(wav: torch.Tensor, threshold_db: float = -40.0, frame_length: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """Return VAD-trimmed waveform.

    A simple RMS-energy threshold is used for demonstration.  For production
    deployments consider WebRTC VAD or pyannote.audio segmentation.
    """

    if wav.ndim != 1:
        raise ValueError("voice_activity_detection expects a 1-D mono signal")

    # Compute frame-wise RMS (energy)
    frames = wav.unfold(0, frame_length, hop_length)  # shape: (n_frames, frame_length)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-10)
    rms_db = 20 * torch.log10(rms)

    # Determine voiced frames
    voiced = rms_db > threshold_db
    if not voiced.any():
        return torch.zeros(0)

    start_frame = voiced.argmax().item()
    end_frame = len(voiced) - voiced.flip(0).argmax().item()  # last True

    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, wav.size(0))

    return wav[start_sample:end_sample] 