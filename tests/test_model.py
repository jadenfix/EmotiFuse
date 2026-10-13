import importlib
import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model
from emotifuse.model import EmotiFuse
from emotifuse.branches.spec_transformer_branch import SpecTransformerBranch
from emotifuse.ncde_model import EmotiCDE
from emotifuse.preprocessing import resample_mono, voice_activity_detection


def tiny_encoder():
    config = Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=32, conv_dim=(8, 8), conv_stride=(4, 4),
                            conv_kernel=(8, 4), num_conv_pos_embedding_groups=2,
                            num_conv_pos_embeddings=8, mask_time_prob=0)
    return Wav2Vec2Model(config)


def test_complete_model_forward_backward_without_download():
    torch.manual_seed(7)
    model = EmotiFuse(emb_dim=16, hidden_dim=8, speech_encoder=tiny_encoder())
    wavs = torch.randn(2, 512)
    feats = {k: torch.randn(2, 9, 13) for k in ("mfcc", "delta", "delta2")}
    logits = model(wavs, feats, torch.randn(2, 201, 11))
    assert logits.shape == (2, 8) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert model.classifier[0].weight.grad is not None
    assert torch.isfinite(model.spect.proj.weight.grad).all()


@pytest.mark.parametrize("frames", [3, 35])
def test_spectrogram_arbitrary_frequency_and_time_shapes(frames):
    branch = SpecTransformerBranch(embed_dim=16)
    assert branch(torch.randn(1, 201, frames)).shape == (1, frames, 16)


def test_single_channel_and_short_vad():
    mono, sr = resample_mono(torch.ones(1, 128), 16000)
    assert mono.shape == (128,) and sr == 16000
    assert voice_activity_detection(mono).shape == mono.shape
    assert voice_activity_detection(torch.zeros(128)).numel() == 0
    assert voice_activity_detection(torch.empty(0)).numel() == 0


def test_cde_single_frame():
    model = EmotiCDE(4, 8)
    assert model(torch.randn(2, 1, 4)).shape == (2, 8)


def test_inference_requires_trained_checkpoint():
    module = importlib.import_module("emotifuse.infer")
    with pytest.raises(ValueError, match="checkpoint"):
        module.infer("unused.wav")


def test_stereo_file_feature_and_model_pipeline(tmp_path, monkeypatch):
    module = importlib.import_module("emotifuse.infer")
    t = np.arange(16000) / 16000
    audio = .5 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([audio, audio], axis=1), 16000)
    calls = []
    class CheckedModel:
        def __call__(self, wavs, feats, specs):
            assert wavs.shape == (1, 16000)
            assert all(value.ndim == 3 and value.size(-1) == 13 for value in feats.values())
            assert specs.ndim == 3 and specs.size(1) == 64
            calls.append(True)
            return torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]])
    monkeypatch.setattr(module, "_load_model", lambda checkpoint: CheckedModel())
    assert module.infer(path, checkpoint="test-only.pt") == "happy"
    assert calls == [True]


def test_cde_late_control_frames_receive_training_gradients():
    torch.manual_seed(7)
    model = EmotiCDE(2, 4)
    control = torch.randn(1, 3, 2, requires_grad=True)
    model(control).sum().backward()
    assert control.grad[:, 1:].abs().sum() > 0
    assert torch.isfinite(control.grad).all()
