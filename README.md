# EmotiFuse

**EmotiFuse** is a research-grade audio–emotion-recognition library that blends

* a self-supervised speech encoder (Wav2Vec 2.0),
* handcrafted acoustic descriptors,
* and a spectrogram Transformer,

then fuses these heterogeneous representations via **gated cross-attention** and feeds them to a **Neural Controlled Differential Equation (NCDE)** backbone for continuous-time modeling.

---

## Features

1. **Wav2Vec 2.0 branch** – frame-level contextual embeddings, dimension $E=768$  
2. **Classical acoustic-feature MLP branch** – MFCC + $\Delta/\Delta\Delta$ + pitch + energy (dimension $256$)  
3. **Spectrogram-Transformer branch** – ViTSER-style with deformable attention (dimension $256$)  
4. **Gated cross-attention fusion** of the three branches  
5. **Neural CDE backbone** for temporal dynamics  
6. Simple training / inference API (PyTorch ≥ 2.0; CPU & CUDA)

---

## High-Level Architecture

```mermaid
graph LR
  A[Raw WAV] --> B(Pre-processing)
  B --> C1[Wav2Vec 2.0<br>(768-D)]
  B --> C2[Acoustic MLP<br>(256-D)]
  B --> C3[Spec Transformer<br>(256-D)]
  C1 --> D[Gated Cross-Attention<br>(256-D)]
  C2 --> D
  C3 --> D
  D --> E[Neural CDE<br>(128-D)]
  E --> F[Classifier → Softmax]
```

⸻

Mathematical Formulation

1. Pre-processing
	•	Resample to 16 kHz mono.
	•	Pre-emphasis filter
[
y[n] = x[n] ;-; 0.97 , x[n-1].
]
	•	Voice-activity detection: drop frames whose RMS (<-40\ \mathrm{dB}).

2. Branch Embeddings

2.1 Wav2Vec 2.0 branch

A fine-tuned model outputs per-frame embeddings
[
\mathbf{e}_t \in \mathbb{R}^{768}.
]

2.2 Classical acoustic-feature MLP branch
	•	Per-frame feature vector
[
\mathbf{f}t = \bigl[\text{MFCC}{1\ldots 39},;p_t,;e_t\bigr] \in \mathbb{R}^{41},
]
where (p_t) is pitch and (e_t) is energy.
	•	Two-layer MLP
[
\mathbf{m}_t
= \operatorname{ReLU}!\bigl(
W_2,\operatorname{ReLU}(W_1 \mathbf{f}_t + b_1) + b_2
\bigr)
\in \mathbb{R}^{256}.
]

2.3 Spectrogram-Transformer branch
	•	Log-mel spectrogram (S \in \mathbb{R}^{T\times 80}).
	•	Two deformable-attention Transformer blocks produce
[
\mathbf{s}_t = \operatorname{Transformer}(S)_t \in \mathbb{R}^{256}.
]

3. Gated Cross-Attention Fusion
	•	Stack branch embeddings
[
H_t = [\mathbf{e}_t;,\mathbf{m}_t;,\mathbf{s}_t] \in \mathbb{R}^{3\times 256}.
]
	•	Gates
[
\mathbf{g}t = W_g H_t + b_g \in \mathbb{R}^3,
\qquad
\alpha{t,i} = \frac{e^{g_{t,i}}}{\sum_{j=1}^3 e^{g_{t,j}}}.
]
	•	Fuse
[
\mathbf{h}t = \sum{i=1}^3 \alpha_{t,i},\mathbf{b}{t,i},
\quad
(\mathbf{b}{t,1},\mathbf{b}{t,2},\mathbf{b}{t,3})
= (\mathbf{e}_t,\mathbf{m}_t,\mathbf{s}_t).
]

4. Neural CDE backbone

Treat (\mathbf{h}t) as a control path (X(t)).
[
\mathrm{d}\mathbf{z}(t)
= f\theta\bigl(\mathbf{z}(t)\bigr),\mathrm{d}X(t),
\qquad
\mathbf{z}(0)=\mathbf{0},
\qquad
\mathbf{z}(T)\in\mathbb{R}^{128}.
]

Euler step for implementation:
[
\mathbf{z}_{n+1}
= \mathbf{z}n
+ f\theta(\mathbf{z}n),\bigl(\mathbf{h}{n+1}-\mathbf{h}_n\bigr).
]

5. Final classifier

[
\hat{\mathbf{y}}
= \operatorname{softmax}(W_c,\mathbf{z}(T) + b_c)
\in \mathbb{R}^C.
]

⸻

Parameter Budget (excludes frozen Wav2Vec)

Module	Output dim	Parameters
Acoustic MLP (41→256→256)	256	8.5 × 10⁴
Spectrogram Transformer (2 blocks)	256	2 × 10⁶
Gating + fusion	256	2 × 10³
NCDE MLP ((128\rightarrow128))	128	1.6 × 10⁴
Classifier ((128\rightarrow64\rightarrow C))	C	1.0 × 10⁴
Total (frozen encoder not counted)		≈ 2.1 M


⸻

Quick-Start

# clone (not yet on PyPI)
git clone https://github.com/yourname/emotifuse.git
cd emotifuse
poetry install        # or: pip install -e .

from emotifuse import EmotiFuse, infer

model = EmotiFuse("checkpoints/")
print(infer("audio/happy.wav"))


⸻

Development

git clone https://github.com/yourname/emotifuse.git
cd emotifuse
poetry install        # or: pip install -e .

pytest -q             # run tests
black . && flake8     # format and lint


⸻

License

MIT © Jaden Fix

