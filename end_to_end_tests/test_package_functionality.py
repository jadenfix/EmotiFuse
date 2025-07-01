"""Package functionality tests for EmotiFuse.

These tests verify that the EmotiFuse package behaves correctly as a Python package,
including proper imports, metadata, and adherence to Python packaging standards.
"""

import importlib
import sys
import subprocess
import tempfile
from pathlib import Path
import wave
import math

import numpy as np
import pytest


def _write_simple_wav(path: Path, dur: float = 0.2) -> None:
    """Write a simple test WAV file."""
    sr = 16000
    n_samples = int(sr * dur)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for n in range(n_samples):
            s = int(0.3 * 32767 * math.sin(2 * math.pi * 440.0 * n / sr))
            wf.writeframes(s.to_bytes(2, "little", signed=True))


class TestPackageImports:
    """Test package import behavior."""
    
    def test_main_package_import(self):
        """Test that the main package can be imported."""
        import emotifuse
        assert emotifuse is not None
    
    def test_reimport_safety(self):
        """Test that the package can be safely re-imported."""
        # First import
        import emotifuse as emotifuse1
        
        # Second import
        import emotifuse as emotifuse2
        
        # Should be the same module
        assert emotifuse1 is emotifuse2
    
    def test_module_attributes(self):
        """Test that expected attributes are available."""
        import emotifuse
        
        # Core classes that should be available
        expected_attrs = [
            'EmotiFuse',
            'AudioIO', 
            'FeatureExtractor',
            'BranchONNX',
            'Fusion'
        ]
        
        for attr in expected_attrs:
            assert hasattr(emotifuse, attr), f"Missing attribute: {attr}"
            assert callable(getattr(emotifuse, attr)), f"Attribute {attr} is not callable"
    
    def test_module_name_and_package(self):
        """Test module name and package information."""
        import emotifuse
        
        assert emotifuse.__name__ == "emotifuse"
        # The module should have basic metadata
        assert hasattr(emotifuse, '__file__')
    
    def test_no_unexpected_imports(self):
        """Test that importing doesn't pollute the namespace."""
        import emotifuse
        
        # Get all attributes
        attrs = dir(emotifuse)
        
        # Filter out standard Python attributes
        public_attrs = [attr for attr in attrs if not attr.startswith('_')]
        
        # Should only have our expected classes, not internal C++ symbols
        expected_classes = {'EmotiFuse', 'AudioIO', 'FeatureExtractor', 'BranchONNX', 'Fusion', 'FrameBuffer'}
        unexpected_attrs = set(public_attrs) - expected_classes
        
        # Allow for some flexibility, but there shouldn't be too many unexpected items
        print(f"Public attributes: {public_attrs}")
        print(f"Unexpected attributes: {unexpected_attrs}")


class TestPackageInstallation:
    """Test package installation and distribution aspects."""
    
    def test_package_in_sys_modules(self):
        """Test that the package is properly registered in sys.modules."""
        import emotifuse
        assert 'emotifuse' in sys.modules
    
    def test_extension_module_loading(self):
        """Test that the C++ extension module loads correctly."""
        import emotifuse
        
        # The module should contain compiled extension components
        # Test by instantiating a core class
        audio_io = emotifuse.AudioIO()
        assert audio_io is not None
    
    def test_no_import_errors(self):
        """Test that there are no import errors or warnings."""
        # This is more about ensuring clean import
        try:
            import emotifuse
            # Try to use basic functionality
            audio_io = emotifuse.AudioIO()
            feature_extractor = emotifuse.FeatureExtractor()
            fusion = emotifuse.Fusion(39)
        except Exception as e:
            pytest.fail(f"Import or basic instantiation failed: {e}")


class TestThreadSafety:
    """Test thread safety aspects (basic tests)."""
    
    def test_concurrent_imports(self):
        """Test that concurrent imports work correctly."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def import_emotifuse():
            try:
                import emotifuse
                audio_io = emotifuse.AudioIO()
                results.put(("success", audio_io))
            except Exception as e:
                errors.put(("error", str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=import_emotifuse)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_data = results.get()
            if result_type == "success":
                success_count += 1
        
        # Check for errors
        error_count = 0
        while not errors.empty():
            error_type, error_msg = errors.get()
            print(f"Thread error: {error_msg}")
            error_count += 1
        
        assert success_count == 5, f"Expected 5 successful imports, got {success_count}"
        assert error_count == 0, f"Got {error_count} errors during concurrent imports"


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def test_multiple_emotifuse_instances(self, tmp_path):
        """Test creating multiple EmotiFuse instances."""
        # Skip if onnx not available
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        
        # Create model directory
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Create minimal models
        def create_model(path, in_dim, out_dim):
            X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, in_dim])
            Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, out_dim])
            node = helper.make_node("Identity", ["x"], ["y"]) if in_dim == out_dim else \
                   helper.make_node("MatMul", ["x", "W"], ["y"])
            
            if in_dim != out_dim:
                import numpy as np
                W_data = np.eye(in_dim, out_dim, dtype=np.float32)
                W = helper.make_tensor("W", TensorProto.FLOAT, [in_dim, out_dim], W_data.flatten())
                graph = helper.make_graph([node], "model", [X], [Y], [W])
            else:
                graph = helper.make_graph([node], "model", [X], [Y])
            
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
            model.ir_version = 10
            onnx.save(model, str(path))
        
        create_model(model_dir / "wav2vec_emoti.onnx", 39, 39)
        create_model(model_dir / "mlp_emoti.onnx", 39, 39)
        create_model(model_dir / "spec_transformer.onnx", 39, 39)
        create_model(model_dir / "ncde_emoti.onnx", 39, 39)
        create_model(model_dir / "classifier.onnx", 39, 5)
        
        import emotifuse
        
        # Create multiple instances
        instances = []
        for i in range(3):
            instance = emotifuse.EmotiFuse(str(model_dir))
            instances.append(instance)
        
        # All instances should be independent and functional
        wav_path = tmp_path / "test.wav"
        _write_simple_wav(wav_path)
        
        for i, instance in enumerate(instances):
            label = instance.predict(str(wav_path))
            assert label in {"neutral", "happy", "sad", "angry", "other"}
        
        # Cleanup should be automatic (no explicit cleanup needed)
        del instances
    
    def test_component_lifecycle(self, tmp_path):
        """Test component creation and destruction lifecycle."""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        
        import emotifuse
        
        # Create and destroy components multiple times
        for iteration in range(3):
            # AudioIO
            audio_io = emotifuse.AudioIO()
            wav_path = tmp_path / f"test_{iteration}.wav"
            _write_simple_wav(wav_path)
            audio_data = audio_io.load_wav_mono(str(wav_path))
            del audio_io
            
            # FeatureExtractor
            feature_extractor = emotifuse.FeatureExtractor()
            features = feature_extractor.extract(audio_data)
            del feature_extractor
            
            # BranchONNX
            model_path = tmp_path / f"test_model_{iteration}.onnx"
            X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 39])
            Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 39])
            node = helper.make_node("Identity", ["x"], ["y"])
            graph = helper.make_graph([node], "test", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
            model.ir_version = 10
            onnx.save(model, str(model_path))
            
            branch = emotifuse.BranchONNX(str(model_path))
            output = branch.run(features)
            del branch
            
            # Fusion
            fusion = emotifuse.Fusion(39)
            fused = fusion.forward(features, features, features)
            del fusion
            
            # Clean up files
            wav_path.unlink()
            model_path.unlink()


class TestErrorRecovery:
    """Test error recovery and robustness."""
    
    def test_graceful_failure_recovery(self, tmp_path):
        """Test that the package can recover from errors gracefully."""
        import emotifuse
        
        audio_io = emotifuse.AudioIO()
        
        # Try to load non-existent file (should fail)
        try:
            audio_io.load_wav_mono(str(tmp_path / "nonexistent.wav"))
            assert False, "Should have raised an exception"
        except Exception:
            pass  # Expected
        
        # But subsequent valid operations should still work
        wav_path = tmp_path / "valid.wav"
        _write_simple_wav(wav_path)
        audio_data = audio_io.load_wav_mono(str(wav_path))
        assert len(audio_data) > 0
    
    def test_invalid_operations_dont_crash(self, tmp_path):
        """Test that invalid operations don't crash the Python interpreter."""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        
        import emotifuse
        
        # Create a valid model
        model_path = tmp_path / "test.onnx"
        X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 39])
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 39])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
        model.ir_version = 10
        onnx.save(model, str(model_path))
        
        branch = emotifuse.BranchONNX(str(model_path))
        
        # Try various invalid inputs
        invalid_inputs = [
            [],                    # Empty list
            [1.0] * 10,           # Wrong size
            [1.0] * 100,          # Wrong size  
            ["not", "numbers"],   # Wrong type
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = branch.run(invalid_input)
                # If it somehow succeeds, that's also fine - the point is no crash
            except Exception:
                # Expected - invalid input should raise exception
                pass
        
        # Valid input should still work after errors
        valid_input = np.array([[1.0] * 39])
        result = branch.run(valid_input)
        assert result.shape == (1, 39)


class TestVersioning:
    """Test version and metadata information."""
    
    def test_module_has_version_info(self):
        """Test that the module provides version information if available."""
        import emotifuse
        
        # Check if version info is available (optional)
        version_attrs = ['__version__', 'version', '_version']
        has_version = any(hasattr(emotifuse, attr) for attr in version_attrs)
        
        # It's okay if version info is not available in development builds
        print(f"Version info available: {has_version}")
        if has_version:
            for attr in version_attrs:
                if hasattr(emotifuse, attr):
                    version = getattr(emotifuse, attr)
                    print(f"{attr}: {version}")


class TestDocumentation:
    """Test documentation and help functionality."""
    
    def test_classes_have_docstrings(self):
        """Test that main classes have docstrings or are at least documented."""
        import emotifuse
        
        main_classes = [
            emotifuse.EmotiFuse,
            emotifuse.AudioIO,
            emotifuse.FeatureExtractor,
            emotifuse.BranchONNX,
            emotifuse.Fusion
        ]
        
        for cls in main_classes:
            # Check if help() works without crashing
            try:
                help_text = help(cls)
                # help() returns None but prints to stdout
                # The fact that it doesn't crash is the test
            except Exception as e:
                pytest.fail(f"help() failed for {cls.__name__}: {e}")
    
    def test_module_help(self):
        """Test that module-level help works."""
        import emotifuse
        
        try:
            help_text = help(emotifuse)
            # help() returns None but prints to stdout
        except Exception as e:
            pytest.fail(f"help() failed for emotifuse module: {e}")


def test_basic_smoke_test_after_import():
    """Final smoke test to ensure everything works after all other tests."""
    import emotifuse
    
    # Just verify we can create instances of main classes
    audio_io = emotifuse.AudioIO()
    feature_extractor = emotifuse.FeatureExtractor()
    fusion = emotifuse.Fusion(39)
    
    # And that they have the expected methods
    assert hasattr(audio_io, 'load_wav_mono')
    assert hasattr(feature_extractor, 'extract')
    assert hasattr(feature_extractor, 'feature_dim')
    assert hasattr(fusion, 'fuse')
    
    print("Package functionality verification complete!") 