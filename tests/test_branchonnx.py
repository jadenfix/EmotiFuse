"""BranchONNX integration test using a tiny identity ONNX model generated at runtime.
Requires `onnx` Python package; test is skipped if not present."""

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto


def _create_identity_model(path: Path, dim: int = 4):
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, dim])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, dim])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "ident_graph", [X], [Y])
    model = helper.make_model(graph, producer_name="test")
    onnx.save(model, str(path))


def test_identity():
    mod = importlib.import_module("emotifuse")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "id.onnx"
        _create_identity_model(model_path)

        branch = mod.BranchONNX(str(model_path))
        x = np.random.rand(3, 4).astype(np.float32)
        y = branch.run(x)
        np.testing.assert_allclose(x, y, atol=1e-6) 