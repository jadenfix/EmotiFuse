"""Handcrafted acoustic feature extraction.

A thin wrapper around *librosa* to compute MFCCs, pitch, energy and spectral
statistics for the *ClassicalBranch*.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import librosa

__all__ = [
    "extract_features",
]


def extract_features(wav: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, np.ndarray]:
    """Return a dictionary with MFCCs, their deltas, pitch, and energy.

    Parameters
    ----------
    wav
        1-D NumPy array.
    sr
        Sampling rate.

    Returns
    -------
    dict
        Keys: ``mfcc``, ``delta``, ``delta2``, ``pitch``, ``energy``.
    """

    # MFCCs
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Pitch using autocorrelation (pyin more robust but slower)
    pitches, _ = librosa.piptrack(y=wav, sr=sr)
    pitch = pitches.max(axis=0)  # take max per frame

    # Frame-wise RMS energy
    energy = librosa.feature.rms(y=wav).squeeze(0)

    return {
        "mfcc": mfcc,
        "delta": delta,
        "delta2": delta2,
        "pitch": pitch,
        "energy": energy,
    } 