"""Component integration tests for EmotiFuse.

These tests verify that individual components can be instantiated and used
independently, ensuring the Python bindings work correctly for each C++ class.
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


def _write_test_wav(path: Path, freq: float = 440.0, sr: int = 16000, dur: float = 0.5) -> None:
    """Write a test WAV file."""
    n_samples = int(sr * dur)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for n in range(n_samples):
            s = int(0.4 * 32767 * math.sin(2 * math.pi * freq * n / sr))
            wf.writeframes(s.to_bytes(2, "little", signed=True))


def _create_identity_model(dim: int, path: Path):
    """Create an identity ONNX model."""
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, dim])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, dim])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "identity", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 10
    onnx.save(model, str(path))


class TestAudioIO:
    """Test AudioIO component."""
    
    def test_audioio_instantiation(self):
        """Test that AudioIO can be instantiated."""
        mod = importlib.import_module("emotifuse")
        audio_io = mod.AudioIO()
        assert audio_io is not None
    
    def test_audioio_load_wav(self, tmp_path):
        """Test AudioIO can load WAV files."""
        mod = importlib.import_module("emotifuse")
        audio_io = mod.AudioIO()
        
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        _write_test_wav(wav_path)
        
        # Load audio
        audio_data = audio_io.load_wav_mono(str(wav_path))
        assert isinstance(audio_data, list)
        assert len(audio_data) > 0
        
        # Values should be floats in a reasonable range
        assert all(isinstance(x, float) for x in audio_data[:10])  # Check first 10 samples
    
    def test_audioio_different_files(self, tmp_path):
        """Test AudioIO with different audio files."""
        mod = importlib.import_module("emotifuse")
        audio_io = mod.AudioIO()
        
        # Test different frequencies and durations
        test_cases = [
            (220.0, 0.3),
            (440.0, 0.5),
            (880.0, 1.0),
        ]
        
        for freq, duration in test_cases:
            wav_path = tmp_path / f"test_{freq}_{duration}.wav"
            _write_test_wav(wav_path, freq=freq, dur=duration)
            
            audio_data = audio_io.load_wav_mono(str(wav_path))
            assert len(audio_data) > 0
            # Longer durations should result in more samples
            expected_samples = int(16000 * duration)  # 16kHz sample rate
            # Allow some tolerance for WAV header and processing
            assert abs(len(audio_data) - expected_samples) < 100


class TestFeatureExtractor:
    """Test FeatureExtractor component."""
    
    def test_feature_extractor_instantiation(self):
        """Test that FeatureExtractor can be instantiated."""
        mod = importlib.import_module("emotifuse")
        feature_extractor = mod.FeatureExtractor()
        assert feature_extractor is not None
    
    def test_feature_extraction(self, tmp_path):
        """Test feature extraction from audio."""
        mod = importlib.import_module("emotifuse")
        audio_io = mod.AudioIO()
        feature_extractor = mod.FeatureExtractor()
        
        # Create test audio
        wav_path = tmp_path / "test_features.wav"
        _write_test_wav(wav_path, dur=1.0)  # Longer duration for more features
        
        # Load audio and extract features
        audio_data = audio_io.load_wav_mono(str(wav_path))
        features = feature_extractor.extract(audio_data)
        
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Check feature dimensions - should be 39 (13 MFCC * 3)
        expected_dim = 39
        assert len(features) == expected_dim
        
        # Features should be floats
        assert all(isinstance(f, float) for f in features)
    
    def test_feature_extractor_properties(self):
        """Test FeatureExtractor properties."""
        mod = importlib.import_module("emotifuse")
        feature_extractor = mod.FeatureExtractor()
        
        # Test featureDim property
        dim = feature_extractor.feature_dim
        assert dim == 39  # 13 MFCC * 3 (static + delta + delta-delta)
        assert isinstance(dim, int)


class TestBranchONNX:
    """Test BranchONNX component."""
    
    def test_branchonnx_instantiation(self, tmp_path):
        """Test that BranchONNX can be instantiated with a model."""
        mod = importlib.import_module("emotifuse")
        
        # Create test ONNX model
        model_path = tmp_path / "test_model.onnx"
        _create_identity_model(39, model_path)
        
        branch = mod.BranchONNX(str(model_path))
        assert branch is not None
    
    def test_branchonnx_inference(self, tmp_path):
        """Test BranchONNX inference."""
        mod = importlib.import_module("emotifuse")
        
        # Create test ONNX model
        model_path = tmp_path / "test_inference.onnx"
        _create_identity_model(39, model_path)
        
        branch = mod.BranchONNX(str(model_path))
        
        # Create test input - BranchONNX expects 2D input (batch_size, features)
        test_input = np.array([[1.0] * 39])  # Shape: (1, 39)
        
        # Run inference
        output = branch.run(test_input)
        
        assert isinstance(output, np.ndarray)
        assert output.shape == (1, 39)  # Should preserve shape
        
        # For identity model, output should equal input
        np.testing.assert_allclose(output[0], test_input[0], rtol=1e-5)
    
    def test_branchonnx_different_inputs(self, tmp_path):
        """Test BranchONNX with different input patterns."""
        mod = importlib.import_module("emotifuse")
        
        model_path = tmp_path / "test_various.onnx"
        _create_identity_model(39, model_path)
        branch = mod.BranchONNX(str(model_path))
        
        # Test various input patterns - need to be 2D arrays
        test_cases = [
            np.array([[0.0] * 39]),              # All zeros
            np.array([[1.0] * 39]),              # All ones
            np.array([list(range(39))]),         # Sequential numbers
            np.array([[(-1)**i for i in range(39)]]),  # Alternating signs
        ]
        
        for test_input in test_cases:
            output = branch.run(test_input)
            assert output.shape == (1, 39)
            np.testing.assert_allclose(output[0], test_input[0], rtol=1e-5)


class TestFusion:
    """Test Fusion component."""
    
    def test_fusion_instantiation(self):
        """Test that Fusion can be instantiated."""
        mod = importlib.import_module("emotifuse")
        fusion = mod.Fusion(39)  # 39-dimensional features
        assert fusion is not None
    
    def test_fusion_computation(self):
        """Test fusion computation."""
        mod = importlib.import_module("emotifuse")
        fusion = mod.Fusion(39)
        
        # Create test embeddings for three branches
        emb1 = [1.0] * 39
        emb2 = [2.0] * 39
        emb3 = [3.0] * 39
        
        # Compute fusion
        fused = fusion.fuse(emb1, emb2, emb3)
        
        assert isinstance(fused, list)
        assert len(fused) == 39
        assert all(isinstance(x, float) for x in fused)
    
    def test_fusion_different_inputs(self):
        """Test fusion with different input combinations."""
        mod = importlib.import_module("emotifuse")
        fusion = mod.Fusion(39)
        
        # Test with different embedding patterns
        test_cases = [
            ([0.0] * 39, [0.0] * 39, [0.0] * 39),  # All zeros
            ([1.0] * 39, [0.0] * 39, [0.0] * 39),  # Only first active
            ([0.0] * 39, [1.0] * 39, [0.0] * 39),  # Only second active
            ([0.0] * 39, [0.0] * 39, [1.0] * 39),  # Only third active
            ([1.0, 0.0] * 19 + [1.0], [0.0, 1.0] * 19 + [0.0], [0.5] * 39),  # Mixed patterns
        ]
        
        for emb1, emb2, emb3 in test_cases:
            fused = fusion.fuse(emb1, emb2, emb3)
            assert len(fused) == 39
            assert all(isinstance(x, float) for x in fused)


class TestIntegration:
    """Test integration between components."""
    
    def test_audio_to_features_pipeline(self, tmp_path):
        """Test the complete audio -> features pipeline."""
        mod = importlib.import_module("emotifuse")
        
        # Create components
        audio_io = mod.AudioIO()
        feature_extractor = mod.FeatureExtractor()
        
        # Create test audio
        wav_path = tmp_path / "integration_test.wav"
        _write_test_wav(wav_path, freq=800.0, dur=0.8)
        
        # Full pipeline
        audio_data = audio_io.load_wav_mono(str(wav_path))
        features = feature_extractor.extract(audio_data)
        
        assert len(features) == 39
        assert all(isinstance(f, float) for f in features)
    
    def test_features_to_branch_pipeline(self, tmp_path):
        """Test features -> branch inference pipeline."""
        mod = importlib.import_module("emotifuse")
        
        # Create components
        audio_io = mod.AudioIO()
        feature_extractor = mod.FeatureExtractor()
        
        # Create ONNX model
        model_path = tmp_path / "branch_test.onnx"
        _create_identity_model(39, model_path)
        branch = mod.BranchONNX(str(model_path))
        
        # Create test audio and extract features
        wav_path = tmp_path / "branch_pipeline.wav"
        _write_test_wav(wav_path)
        
        audio_data = audio_io.load_wav_mono(str(wav_path))
        features = feature_extractor.extract(audio_data)
        
        # Run through branch - need to reshape features to 2D
        features_2d = np.array([features])  # Convert to shape (1, 39)
        output = branch.run(features_2d)
        
        assert output.shape == (1, 39)
        # For identity model, output should match input features
        np.testing.assert_allclose(output[0], features, rtol=1e-5)
    
    def test_full_component_chain(self, tmp_path):
        """Test the complete chain: audio -> features -> branches -> fusion."""
        mod = importlib.import_module("emotifuse")
        
        # Create all components
        audio_io = mod.AudioIO()
        feature_extractor = mod.FeatureExtractor()
        fusion = mod.Fusion(39)
        
        # Create three ONNX models for branches
        branch_models = []
        for i in range(3):
            model_path = tmp_path / f"branch_{i}.onnx"
            _create_identity_model(39, model_path)
            branch_models.append(mod.BranchONNX(str(model_path)))
        
        # Create test audio
        wav_path = tmp_path / "full_chain.wav"
        _write_test_wav(wav_path, freq=600.0, dur=0.6)
        
        # Full pipeline
        audio_data = audio_io.load_wav_mono(str(wav_path))
        features = feature_extractor.extract(audio_data)
        
        # Run through all branches
        branch_outputs = []
        features_2d = np.array([features])  # Convert to shape (1, 39)
        for branch in branch_models:
            output = branch.run(features_2d)
            branch_outputs.append(output[0].tolist())  # Convert to list for fusion
        
        # Fuse outputs
        fused = fusion.fuse(branch_outputs[0], branch_outputs[1], branch_outputs[2])
        
        assert len(fused) == 39
        assert all(isinstance(x, float) for x in fused)
    
    def test_component_reuse(self, tmp_path):
        """Test that components can be reused across multiple calls."""
        mod = importlib.import_module("emotifuse")
        
        # Create components once
        audio_io = mod.AudioIO()
        feature_extractor = mod.FeatureExtractor()
        
        model_path = tmp_path / "reuse_test.onnx"
        _create_identity_model(39, model_path)
        branch = mod.BranchONNX(str(model_path))
        
        # Use them multiple times
        for i in range(5):
            wav_path = tmp_path / f"reuse_{i}.wav"
            _write_test_wav(wav_path, freq=400 + i * 100)
            
            audio_data = audio_io.load_wav_mono(str(wav_path))
            features = feature_extractor.extract(audio_data)
            features_2d = np.array([features])  # Convert to shape (1, 39)
            output = branch.run(features_2d)
            
            assert len(features) == 39
            assert output.shape == (1, 39)


class TestErrorHandling:
    """Test error handling in components."""
    
    def test_audioio_invalid_file(self, tmp_path):
        """Test AudioIO with invalid files."""
        mod = importlib.import_module("emotifuse")
        audio_io = mod.AudioIO()
        
        # Non-existent file
        with pytest.raises(Exception):
            audio_io.load_wav_mono(str(tmp_path / "nonexistent.wav"))
        
        # Invalid file
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("not a wav file")
        with pytest.raises(Exception):
            audio_io.load_wav_mono(str(invalid_file))
    
    def test_branchonnx_invalid_model(self, tmp_path):
        """Test BranchONNX with invalid models."""
        mod = importlib.import_module("emotifuse")
        
        # Non-existent model
        with pytest.raises(Exception):
            mod.BranchONNX(str(tmp_path / "nonexistent.onnx"))
        
        # Invalid model file
        invalid_model = tmp_path / "invalid.onnx"
        invalid_model.write_text("not an onnx model")
        with pytest.raises(Exception):
            mod.BranchONNX(str(invalid_model))
    
    def test_branchonnx_wrong_input_size(self, tmp_path):
        """Test BranchONNX with wrong input dimensions."""
        mod = importlib.import_module("emotifuse")
        
        model_path = tmp_path / "size_test.onnx"
        _create_identity_model(39, model_path)
        branch = mod.BranchONNX(str(model_path))
        
        # Wrong input size - need 2D arrays
        with pytest.raises(Exception):
            branch.run(np.array([[1.0] * 10]))  # Wrong dimension
        
        with pytest.raises(Exception):
            branch.run(np.array([[1.0] * 100]))  # Wrong dimension
    
    def test_fusion_wrong_input_size(self):
        """Test Fusion with wrong input dimensions."""
        mod = importlib.import_module("emotifuse")
        fusion = mod.Fusion(39)
        
        # Wrong input sizes
        with pytest.raises(Exception):
            fusion.fuse([1.0] * 10, [1.0] * 39, [1.0] * 39)  # First wrong
        
        with pytest.raises(Exception):
            fusion.fuse([1.0] * 39, [1.0] * 10, [1.0] * 39)  # Second wrong
        
        with pytest.raises(Exception):
            fusion.fuse([1.0] * 39, [1.0] * 39, [1.0] * 10)  # Third wrong 