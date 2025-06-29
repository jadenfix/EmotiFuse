"""Test that EmotiFuse loads three branch models from a directory."""

import importlib
import tempfile
from pathlib import Path

import numpy as np
import wave
import math

import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto


def _create_identity(path: Path, dim):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, dim])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, dim])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    model = helper.make_model(graph)
    onnx.save(model, str(path))


def _write_dummy_wav(path: Path, sr=16000):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(sr // 10):  # 0.1 s
            val = int(0.1 * 32767 * math.sin(2 * math.pi * 220 * i / sr))
            w.writeframes(val.to_bytes(2, "little", signed=True))


def test_branch_loading():
    emo = importlib.import_module("emotifuse")

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        _create_identity(d / "wav2vec_emoti.onnx", 10)
        _create_identity(d / "mlp_emoti.onnx", 10)
        _create_identity(d / "spec_transformer.onnx", 10)

        wav = d / "test.wav"
        _write_dummy_wav(wav)

        model = emo.EmotiFuse(str(d))
        assert model.predict(str(wav)) == "neutral" 