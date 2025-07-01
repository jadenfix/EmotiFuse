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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 10  # Use compatible IR version with current ONNX Runtime
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
        # The FeatureExtractor outputs n_mfcc * 3 = 13 * 3 = 39 features by default
        _create_identity(d / "wav2vec_emoti.onnx", 39)
        _create_identity(d / "mlp_emoti.onnx", 39)
        _create_identity(d / "spec_transformer.onnx", 39)
        _create_identity(d / "ncde_emoti.onnx", 39)
        _create_identity(d / "classifier.onnx", 39)

        wav = d / "test.wav"
        _write_dummy_wav(wav)

        model = emo.EmotiFuse(str(d))
        pred = model.predict(str(wav))
        assert pred in {"neutral", "happy", "sad", "angry", "other"} 