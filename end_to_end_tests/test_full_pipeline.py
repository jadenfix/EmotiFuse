"""End-to-end smoke tests exercising the full EmotiFuse pipeline.

These tests generate dummy ONNX identity models for every branch, synthesise
short WAV snippets, and verify that `EmotiFuse.predict` returns a valid label
for multiple inputs. While the models are placeholders, this ensures that all
C++ → Python glue, ONNX Runtime execution, and feature extraction interact
correctly in an integrated scenario.
"""

import importlib
import math
import tempfile
from pathlib import Path
import wave

import numpy as np
import pytest

# Skip if onnx python package missing.
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

LABELS = {"neutral", "happy", "sad", "angry", "other"}


def _write_wav(path: Path, freq: float, sr: int = 16000, dur: float = 0.2) -> None:
    n_samples = int(sr * dur)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for n in range(n_samples):
            s = int(0.4 * 32767 * math.sin(2 * math.pi * freq * n / sr))
            wf.writeframes(s.to_bytes(2, "little", signed=True))


def _identity_model(out_dim: int, path: Path):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, out_dim])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, out_dim])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    model = helper.make_model(graph)
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Create a temporary directory containing identity ONNX models."""
    d = Path(tmp_path_factory.mktemp("models"))
    _identity_model(10, d / "wav2vec_emoti.onnx")
    _identity_model(10, d / "mlp_emoti.onnx")
    _identity_model(10, d / "spec_transformer.onnx")
    _identity_model(10, d / "ncde_emoti.onnx")
    _identity_model(10, d / "classifier.onnx")
    return d


@pytest.fixture(scope="module")
def emotifuse(model_dir):
    mod = importlib.import_module("emotifuse")
    return mod.EmotiFuse(str(model_dir))


def test_multiple_predictions(tmp_path, emotifuse):
    """Run predict on several synthetic tones and verify label validity."""
    freqs = [220.0, 440.0, 660.0]
    for i, f in enumerate(freqs):
        wav = tmp_path / f"tone_{i}.wav"
        _write_wav(wav, f)
        label = emotifuse.predict(str(wav))
        assert label in LABELS

    # Check deterministic outcome on repeated call
    wav = tmp_path / "repeat.wav"
    _write_wav(wav, 330.0)
    label1 = emotifuse.predict(str(wav))
    label2 = emotifuse.predict(str(wav))
    assert label1 == label2 