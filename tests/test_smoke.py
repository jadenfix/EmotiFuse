"""Smoke test for the EmotiFuse Python bindings.

This test simply verifies that the compiled extension can be imported and that
calling `predict` on a dummy WAV path returns a string. The heavy‐lifting logic
will be exercised in later unit tests once the pipeline is implemented.
"""

import importlib
import subprocess
from pathlib import Path
import tempfile

import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto


def _create_identity_model(path: Path, dim: int = 10):
    """Create a simple identity ONNX model for testing."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, dim])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, dim])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "ident_graph", [X], [Y])
    model = helper.make_model(graph, producer_name="test", opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 8  # Use a compatible IR version
    onnx.save(model, str(path))


def test_import_and_predict(tmp_path):
    """Ensure that the module can be imported and the stub returns a string."""
    emotifuse = importlib.import_module("emotifuse")

    # Create dummy ONNX model files that EmotiFuse expects
    _create_identity_model(tmp_path / "wav2vec_emoti.onnx", 10)
    _create_identity_model(tmp_path / "mlp_emoti.onnx", 10)
    _create_identity_model(tmp_path / "spec_transformer.onnx", 10)
    _create_identity_model(tmp_path / "ncde_emoti.onnx", 10)
    _create_identity_model(tmp_path / "classifier.onnx", 10)

    # Create an empty dummy WAV file so that the path exists.
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.touch()

    model = emotifuse.EmotiFuse(str(tmp_path))
    label = model.predict(str(dummy_wav))
    assert isinstance(label, str) 