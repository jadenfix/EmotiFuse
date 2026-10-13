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

def test_single_channel_audio_and_short_clip():
    wav = torch.ones(1, 100)
    mono, rate = resample_mono(wav, 16000)
    assert mono.shape == (100,)
    assert rate == 16000
    assert torch.equal(voice_activity_detection(mono), mono)
    silence = torch.zeros(100, dtype=torch.float64)
    trimmed = voice_activity_detection(silence)
    assert trimmed.numel() == 0
    assert trimmed.dtype == silence.dtype


def test_vad_exact_voiced_interval():
    wav = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4)])
    trimmed = voice_activity_detection(wav, frame_length=4, hop_length=4)
    assert torch.equal(trimmed, torch.ones(4))


def test_vad_empty_audio():
    empty = torch.empty(0, dtype=torch.float64)
    assert voice_activity_detection(empty).dtype == empty.dtype
