"""Fusion gating unit test."""

import importlib
import numpy as np


def test_fusion_shape_and_gate_sum():
    mod = importlib.import_module("emotifuse")
    dim = 6
    fusion = mod.Fusion(dim, seed=123)

    e1 = np.random.rand(dim).astype(np.float32)
    e2 = np.random.rand(dim).astype(np.float32)
    e3 = np.random.rand(dim).astype(np.float32)

    fused = fusion.fuse(e1, e2, e3)
    assert len(fused) == dim

    gates = fusion.last_gates
    assert abs(sum(gates) - 1.0) < 1e-5, "gates should softmax to 1"

    # Manually compute weighted average to verify implementation
    fused_manual = gates[0] * e1 + gates[1] * e2 + gates[2] * e3
    np.testing.assert_allclose(fused, fused_manual, atol=1e-6) 