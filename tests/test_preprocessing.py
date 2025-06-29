import torch

from emotifuse.preprocessing import resample_mono, pre_emphasis, voice_activity_detection


def test_resample_mono_identity():
    # Generate 1 kHz sine at 16kHz
    sr = 16_000
    t = torch.linspace(0, 1, sr)
    wav = torch.sin(2 * torch.pi * 1000 * t)
    out, sr_out = resample_mono(wav, sr, sr)
    assert sr_out == sr
    assert torch.allclose(wav, out, atol=1e-5)


def test_pre_emphasis_shape():
    wav = torch.randn(10_000)
    out = pre_emphasis(wav)
    assert out.shape == wav.shape


def test_vad_silence():
    wav = torch.zeros(8000)
    out = voice_activity_detection(wav)
    assert out.numel() == 0 