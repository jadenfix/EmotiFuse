# EmotiFuse

**EmotiFuse** is a research-grade audio emotion recognition library that blends state-of-the-art speech encoders (Wav2Vec 2.0), handcrafted statistical descriptors, and spectrogram transformers. These heterogeneous representations are fused with gated cross-attention and processed by a Neural Controlled Differential Equation (NCDE) backbone for continuous-time modeling.

---

## 🔍 Features

- **Multi-branch architecture** combining:  
  - **Pre-trained Wav2Vec 2.0** branch (frame-level contextual embeddings, *E* = 768).  
  - **Classical acoustic-feature MLP** branch (MFCC + Δ/ΔΔ + pitch + energy).  
  - **Spectrogram Transformer** branch (ViTSER-style with deformable attention).  
- **Gated cross-attention fusion** of branch embeddings.  
- **Temporal modeling** via Neural CDEs.  
- **Simple training & inference** interfaces (PyTorch ≥ 2.0, CUDA & CPU).

---

## 🏗 Architecture

```mermaid
graph LR
  A[Raw WAV<br/>$begin:math:text$x(t)$end:math:text$] --> B(Preprocessing)
  B --> C1[Wav2Vec 2.0 Branch<br/>$begin:math:text$\\mathbf{e}_t\\in\\mathbb{R}^{768}$end:math:text$]
  B --> C2[Classical MLP Branch<br/>$begin:math:text$\\mathbf{m}_t\\in\\mathbb{R}^{256}$end:math:text$]
  B --> C3[Spectrogram Transformer<br/>$begin:math:text$\\mathbf{s}_t\\in\\mathbb{R}^{256}$end:math:text$]
  C1 --> D[Gated Cross-Attention<br/>$begin:math:text$\\mathbf{h}_t\\in\\mathbb{R}^{256}$end:math:text$]
  C2 --> D
  C3 --> D
  D --> E[Neural CDE<br/>$begin:math:text$\\mathbf{z}(T)\\in\\mathbb{R}^{128}$end:math:text$]
  E --> F[Classifier → Softmax<br/>$begin:math:text$\\hat{\\mathbf{y}}\\in\\mathbb{R}^{C}$end:math:text$]


⸻

🔢 Mathematical Formulation

1. Preprocessing
	•	Resample to 16 kHz mono.
	•	Pre-emphasis:
[
y[n] = x[n] - 0.97,x[n-1]
]
	•	VAD: drop frames with RMS < −40 dB.

2. Branch Embeddings

2.1 Wav2Vec 2.0 Branch

Fine-tuned wav2vec2.0 yields frame-level embeddings
(\mathbf{e}_t\in\mathbb{R}^{768}).

2.2 Classical Acoustic MLP Branch
	1.	Feature extraction per frame (t):
	•	MFCCs: (\mathrm{MFCC}_{t,j}) for (j=1..13)
	•	Δ and ΔΔ → total MFCC dims = 39
	•	Pitch (p_t) via autocorrelation
	•	Energy (e_t) (RMS)
	2.	Feature vector
[
\mathbf{f}_t \in \mathbb{R}^{41}.
]
	3.	MLP
[
\mathbf{m}_t = \mathrm{ReLU}\bigl(W_2,\mathrm{ReLU}(W_1,\mathbf{f}_t + b_1) + b_2\bigr),
\quad \mathbf{m}_t\in\mathbb{R}^{256}.
]

2.3 Spectrogram Transformer Branch
	1.	Log-mel spectrogram (S\in\mathbb{R}^{T\times80}) (80 mel bins).
	2.	Two Transformer blocks with deformable self-attention →
[
\mathbf{s}_t = \mathrm{TransformerBlock}(S)\in\mathbb{R}^{256}.
]

3. Gated Cross-Attention Fusion

Stack embeddings:
[
H_t = [\mathbf{e}_t;,\mathbf{m}t;,\mathbf{s}t]\in\mathbb{R}^{3\times256}.
]
Compute gate logits:
[
\mathbf{g}t = W_g,H_t + b_g\in\mathbb{R}^{3},\quad
\boldsymbol{\alpha}t = \mathrm{softmax}(\mathbf{g}t).
]
Fuse:
[
\mathbf{h}t = \sum{i=1}^3 \alpha{t,i},\mathbf{b}{t,i},\quad
[\mathbf{b}{t,1},\mathbf{b}{t,2},\mathbf{b}{t,3}]
= [\mathbf{e}_t,\mathbf{m}_t,\mathbf{s}_t].
]

4. Neural CDE Backbone

Control path ({\mathbf{h}t}):
[
\mathrm{d}\mathbf{z}(t) = f\theta\bigl(\mathbf{z}(t)\bigr),\mathrm{d}X(t),
\quad \mathbf{z}(0)=0.
]
Euler update:
[
\mathbf{z}_{n+1} = \mathbf{z}n + f\theta(\mathbf{z}n),\bigl(\mathbf{h}{n+1}-\mathbf{h}_n\bigr),
\quad \mathbf{z}\in\mathbb{R}^{128}.
]

5. Final Classification

Readout (\mathbf{z}(T)):
[
\hat{\mathbf{y}} = \mathrm{softmax}(W_c,\mathbf{z}(T) + b_c)\in\mathbb{R}^{C}.
]

⸻

📊 Hyperparameters & Model Size

Component	Output Dim	Params
Wav2Vec 2.0 (frozen)	768	95 M
Classical MLP (41→256→256)	256	~85 K
Spectrogram Transformer (2×)	256	~2 M
Gating & Fusion (3×256→3→256)	256	~2 K
NCDE MLP (128)	128	~16 K
Classifier (128→64→C)	C	~10 K
Total (excl. Wav2Vec)	—	~~2.1 M


⸻

⚙️ Quickstart

# Install from source (not yet on PyPI)
git clone https://github.com/yourname/emotifuse.git
cd emotifuse
poetry install     # or pip install -e .

# In Python:
from emotifuse import EmotiFuse, infer
model = EmotiFuse("path/to/checkpoint_dir")
label = infer("audio/happy_utterance.wav")
print(f"Predicted emotion: {label}")


⸻

🛠 Development

# Clone & install
git clone https://github.com/yourname/emotifuse.git
cd emotifuse
poetry install   # or pip install -e .

# Run tests
pytest -q

# Format & lint
black . && flake8


⸻

📄 License

MIT © Jaden Fix

