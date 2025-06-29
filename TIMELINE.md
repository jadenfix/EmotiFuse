# EmotiFuse Development Timeline

> This document tracks the incremental milestones required to evolve the current C++ scaffold into a production-ready, pip-installable speech-emotion recogniser. Check off items as they are merged.

## Milestone 0 — Scaffold (✔ complete)
- Build system (`CMakeLists.txt`) and Python packaging (`setup.py`, `pyproject.toml`)
- Continuous Integration: GitHub Actions building wheels, publishing to PyPI
- Namespace package, stub C++ sources, smoke tests

---

## Milestone 1 — Audio Front-End
1. **AudioIO**
   - Choose backend: libsndfile (preferred) with dr_wav fallback
   - WAV load ➜ mono, 16 kHz (SoX-style linear-phase resampler)
   - Pre-emphasis filter `y[n] = x[n] − 0.97 x[n−1]`
   - Frame blocking: 25 ms window, 10 ms hop, Hamming window
   - Simple energy-based VAD

**Deliverables**: `src/AudioIO.{hpp,cpp}`, unit tests with golden WAV fixtures.

---

## Milestone 2 — Classic Feature Extraction
- Embed KissFFT (BSD) in `third_party/kissfft`
- MFCC pipeline: 512-pt FFT → 40 Mel bins → DCT-II → 13 coeffs
- Compute Δ/ΔΔ; add pitch (ACF) & RMS energy per frame

**Deliverables**: `src/FeatureExtractor.{hpp,cpp}`, unit tests vs. librosa.

---

## Milestone 3 — ONNX Runtime Wrapper
- Implement `BranchONNX` (lazy `Ort::Session` loader)
- Helper for `std::vector<float>` ⇄ ONNX tensor

**Deliverables**: `src/BranchONNX.{hpp,cpp}`, tests using identity ONNX model.

---

## Milestone 4 — Per-Branch Inference
- Instantiate three branches:
  1. `wav2vec_emoti.onnx` (raw waveform)
  2. `mlp_emoti.onnx` (classic features)
  3. `spec_transformer.onnx` (spectrogram)
- Ensure embeddings align to common dimension `H`

---

## Milestone 5 — Fusion (Gated Cross-Attention)
- Use Eigen for lightweight linear / matmul
- Gate: `g = σ(Wg · concat(e₁,e₂,e₃))`
- Attention: compute Q/K/V, produce fused embedding sequence

**Deliverables**: `src/Fusion.{hpp,cpp}`, deterministic unit tests.

---

## Milestone 6 — NCDE Post-Processing
- Option A: train NCDE and export to ONNX (preferred short-term)
- (Option B: native Euler/RK4 solver with learned params)

---

## Milestone 7 — Classifier & End-to-End Glue
- Final MLP classifier in ONNX
- Complete `EmotiFuse::predict` pipeline
- Map output vector → label string via YAML/JSON

---

## Milestone 8 — Polish & Distribution
- Remove placeholders; enable `-Werror`, clang-tidy, CI static analysis
- Extend CI Python matrix to 3.8-3.12, manylinux wheels
- README badges, GitHub Releases notes

---

**Legend**
- `[ ]` not started `[►]` in progress `[✔]` done

Feel free to adjust scope or order as the project evolves. 