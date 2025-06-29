"""Smoke test for the EmotiFuse Python bindings.

This test simply verifies that the compiled extension can be imported and that
calling `predict` on a dummy WAV path returns a string. The heavy‐lifting logic
will be exercised in later unit tests once the pipeline is implemented.
"""

import importlib
import subprocess
from pathlib import Path

import pytest


def test_import_and_predict(tmp_path):
    """Ensure that the module can be imported and the stub returns a string."""
    emotifuse = importlib.import_module("emotifuse")

    # Create an empty dummy WAV file so that the path exists.
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.touch()

    model = emotifuse.EmotiFuse(str(tmp_path))
    label = model.predict(str(dummy_wav))
    assert isinstance(label, str) 