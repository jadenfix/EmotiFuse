"""Test MFCC extraction on a sine wave.
Verifies output dimensions and basic sanity (variance across frames)."""

import importlib
import math
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module("emotifuse")


def _write_wav(path: Path, freq=440.0, sr=16000, duration=0.3):
    n = int(sr * duration)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            sample = int(0.6 * 32767 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(sample.to_bytes(2, "little", signed=True))


def test_mfcc(tmp_path, mod):
    wav = tmp_path / "tone.wav"
    _write_wav(wav)

    audioio = mod.AudioIO()
    signal = audioio.load_wav_mono(str(wav))
    fb = audioio.framing(signal)

    fe = mod.FeatureExtractor()
    feats = fe.extract(fb)
    dim = fe.feature_dim
    nframes = fb.frames

    assert len(feats) == dim * nframes
    # Ensure variance across time (not all zeros)
    arr = np.array(feats).reshape(nframes, dim)
    assert arr.std() > 0.001 