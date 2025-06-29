"""Unit test for the AudioIO front-end using a synthetically generated sine wave."""

import wave
import math
from pathlib import Path

import importlib
import pytest


@pytest.fixture(scope="module")
def emotifuse_module():
    return importlib.import_module("emotifuse")


def _write_sine_wav(path: Path, freq=440.0, sr=16000, duration=0.25):
    n_samples = int(sr * duration)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for n in range(n_samples):
            # Simple sine wave, 0.5 amplitude
            sample = int(0.5 * 32767 * math.sin(2 * math.pi * freq * n / sr))
            wf.writeframes(sample.to_bytes(2, "little", signed=True))


def test_audioio_pipeline(tmp_path, emotifuse_module):
    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path)

    AudioIO = emotifuse_module.AudioIO
    io = AudioIO()
    signal = io.load_wav_mono(str(wav_path))
    assert len(signal) > 0

    fb = io.framing(signal)
    assert fb.frames > 0

    energies = io.frame_energy(fb)
    assert len(energies) == fb.frames
    # Energies should be above -40 dB for pure tone
    assert max(energies) > -20 