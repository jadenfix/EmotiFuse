"""Download-free synthetic forward/backward check, not emotion prediction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2Model
from .features import extract_features
from .model import EmotiFuse


def run_smoke():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    samples = np.arange(1600, dtype=np.float32) / 16000
    waveform = torch.from_numpy(.5 * np.sin(2 * np.pi * 220 * samples))
    features = extract_features(waveform.numpy(), 16000)
    features = {key: torch.tensor(features[key]).T.unsqueeze(0).float()
                for key in ("mfcc", "delta", "delta2")}
    spectrum = torch.log1p(torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)(waveform)).unsqueeze(0)
    encoder = Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8, 8), conv_stride=(8, 8),
        conv_kernel=(10, 5), num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8, mask_time_prob=0))
    model = EmotiFuse(emb_dim=16, hidden_dim=8, speech_encoder=encoder)
    logits = model(waveform.unsqueeze(0), features, spectrum)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    if not torch.isfinite(logits).all() or not torch.isfinite(model.spect.proj.weight.grad).all():
        raise RuntimeError("non-finite model output or gradient")
    return {"mode": "synthetic-random-weight-smoke", "samples": len(waveform),
            "sample_rate": 16000, "spectrum_shape": list(spectrum.shape),
            "mfcc_shape": list(features["mfcc"].shape), "logit_shape": list(logits.shape),
            "loss": float(loss.detach()), "backward": "finite"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
