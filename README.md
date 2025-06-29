# EmotiFuse (C++ Scaffold)

This repository hosts the first iteration of **EmotiFuse**, a C++‐native speech–emotion recognition core wrapped as a Python package via [pybind11](https://github.com/pybind/pybind11).

> ⚠️ **Early scaffold:** The current version only sets up the build system and exposes a stub `EmotiFuse` class. The actual inference pipeline (audio I/O, feature extraction, ONNX models, fusion, NCDE) will be implemented incrementally.

## Getting Started

1. **Prerequisites** (macOS/Linux/Windows):
   - C++17 compiler (e.g., Clang ≥ 10, GCC ≥ 9, MSVC ≥ 19.28).
   - CMake ≥ 3.15.
   - Python 3.8-3.12 with development headers.
   - `pybind11` and `onnxruntime` *development* packages installed system-wide **or** available via environment variables `PYBIND11_DIR`, `ONNXRUNTIME_DIR`.

2. **Build & install**:

   ```bash
   git clone https://github.com/yourname/emotifuse_cpp.git
   cd emotifuse_cpp
   pip install .  # invokes setup.py → CMake → build
   ```

3. **Quick test**:

   ```python
   import emotifuse
   model = emotifuse.EmotiFuse("./models")
   print(model.predict("example.wav"))  # → "neutral" (stub)
   ```

## Development Roadmap

- [ ] Audio I/O & preprocessing (`AudioIO.*`)
- [ ] Classic feature extraction (`FeatureExtractor.*`)
- [ ] ONNX Runtime wrappers (`BranchONNX.*`)
- [ ] Gated cross-attention fusion (`Fusion.*`)
- [ ] NCDE post-processing (`NCDE.*`)
- [ ] Unit tests & CI (GitHub Actions)

Contributions are welcome! 