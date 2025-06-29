# EmotiFuse

EmotiFuse is a research-grade audio emotion recognition library that blends state-of-the-art speech encoders (e.g., Wav2Vec 2.0), handcrafted statistical descriptors, and spectrogram transformers.  These heterogeneous representations are fused with gated cross-attention and processed by a Neural Controlled Differential Equation (NCDE) backbone.

## Features

* Multi-branch architecture combining:
  * Pre-trained Wav2Vec 2.0 branch.
  * Classical acoustic-feature MLP branch.
  * Spectrogram Transformer branch.
* Gated cross-attention fusion.
* Temporal modeling with Neural CDEs.
* Simple training & inference interfaces.
* PyTorch ≥ 2.0, compatible with CUDA and CPU.

## Quickstart

```bash
pip install emotifuse  # Not on PyPI yet → install from source
```

```python
from emotifuse import EmotiFuse, infer

label = infer("audio/happy_utterance.wav")
print(label)
```

## Development

1. Clone and install dependencies:
   ```bash
   git clone https://github.com/yourname/emotifuse.git
   cd emotifuse
   poetry install  # or pip install -e .
   ```
2. Run tests:
   ```bash
   pytest -q
   ```
3. Format & lint:
   ```bash
   black . && flake8
   ```

## License

MIT © Your Name 