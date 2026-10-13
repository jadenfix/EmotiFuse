"""Actual neural checkpoint and acoustic-pipeline regressions; no accuracy claim."""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2Model

from emotifuse.features import extract_features
from emotifuse.model import EmotiFuse
from emotifuse.preprocessing import pre_emphasis, voice_activity_detection


def tiny_model(emb_dim=8):
    config = Wav2Vec2Config(hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, conv_dim=(4, 4),
        conv_stride=(4, 4), conv_kernel=(8, 4), num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8, mask_time_prob=0)
    return EmotiFuse(emb_dim=emb_dim, hidden_dim=4, speech_encoder=Wav2Vec2Model(config))


@pytest.fixture
def cache(monkeypatch):
    module = importlib.import_module("emotifuse.infer")
    monkeypatch.setattr(module, "EmotiFuse", tiny_model)
    monkeypatch.setattr(module, "_model", None)
    monkeypatch.setattr(module, "_checkpoint", None)
    monkeypatch.setattr(module, "_checkpoint_fingerprint", None, raising=False)
    monkeypatch.setattr(module, "_device", torch.device("cpu"))
    return module


@pytest.mark.parametrize("replace_atomically", [False, True])
def test_same_path_replacement_loads_new_actual_weights(tmp_path, cache, replace_atomically):
    path = tmp_path / "model.pt"
    model = tiny_model()
    torch.save(model.state_dict(), path)
    original = cache._load_model(str(path))
    replacement = tmp_path / "replacement.pt"
    with torch.no_grad():
        model.classifier[-1].bias.add_(10)
    torch.save(model.state_dict(), replacement)
    if replace_atomically:
        os.replace(replacement, path)
    else:
        torch.save(model.state_dict(), path)
    refreshed = cache._load_model(str(path))
    assert refreshed is not original
    assert torch.equal(refreshed.classifier[-1].bias, model.classifier[-1].bias)
    assert not torch.equal(original.classifier[-1].bias, refreshed.classifier[-1].bias)


def test_file_modified_during_load_does_not_replace_cached_model(tmp_path, cache, monkeypatch):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(tiny_model().state_dict(), a)
    original = cache._load_model(str(a))
    torch.save(tiny_model().state_dict(), b)
    load = torch.load
    def changed_load(path, **kwargs):
        result = load(path, **kwargs)
        Path(path).write_bytes(b"changed during load")
        return result
    monkeypatch.setattr(cache.torch, "load", changed_load)
    with pytest.raises(RuntimeError, match="changed while loading"):
        cache._load_model(str(b))
    assert cache._model is original
    assert cache._checkpoint == str(a.resolve())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_logits_never_become_emotion_labels(tmp_path, cache, monkeypatch, bad):
    waveform = .6 * np.sin(2 * np.pi * 220 * np.arange(2048) / 16000)
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, waveform, 16000)
    class InvalidModel:
        def __call__(self, **kwargs):
            return torch.full((1, 8), bad)
    monkeypatch.setattr(cache, "_load_model", lambda _: InvalidModel())
    with pytest.raises(RuntimeError, match="Non-finite emotion logits"):
        cache.infer(audio_path, "test.pt")


def test_real_acoustic_training_checkpoint_restore_and_inference(tmp_path, cache, monkeypatch):
    # A small local encoder and gently initialized CDE allow a bounded CPU
    # workflow. Arbitrary labels and one optimizer step do not establish SER quality.
    torch.manual_seed(8)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny_model()
        with torch.no_grad():
            model.cde.func.linear.weight.mul_(.01)
            model.cde.func.linear.bias.mul_(.01)
        before = model.classifier[0].weight.detach().clone()
        waveform = (.6 * np.sin(2 * np.pi * 220 * np.arange(2048) / 16000)).astype(np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(path, np.stack([waveform, waveform], axis=1), 16000, subtype="FLOAT")
        wav = voice_activity_detection(pre_emphasis(torch.from_numpy(waveform)))
        raw = extract_features(wav.numpy(), 16000)
        features = {key: torch.from_numpy(raw[key]).T.float() for key in ("mfcc", "delta", "delta2")}
        spectrum = torch.log1p(torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)(wav))
        dataset = [{"wav": wav, "feats": features, "spec": spectrum,
                    "label": torch.tensor(label)} for label in (0, 1)]
        training = importlib.import_module("emotifuse.train")
        monkeypatch.setattr(training, "EmotiFuse", lambda: model)
        monkeypatch.setattr(training, "get_default_device", lambda: torch.device("cpu"))
        loader = training.DataLoader
        monkeypatch.setattr(training, "DataLoader", lambda *args, **kwargs:
                            loader(*args, **{**kwargs, "num_workers": 0}))
        training.train(dataset, epochs=1, batch_size=2, ckpt_dir=tmp_path)
        assert not torch.equal(before, model.classifier[0].weight)
        assert all(torch.isfinite(value).all() for value in model.parameters())
        assert all(value.grad is None or torch.isfinite(value.grad).all() for value in model.parameters())
        checkpoint = tmp_path / "emotifuse_epoch01.pt"
        restored = cache._load_model(str(checkpoint))
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
        logits = []
        handle = restored.register_forward_hook(lambda module, inputs, output: logits.append(output.detach()))
        try:
            label = cache.infer(path, str(checkpoint))
        finally:
            handle.remove()
        probabilities = logits[0].softmax(-1)
        assert torch.isfinite(probabilities).all()
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        labels = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised", "calm"]
        assert label == labels[probabilities.argmax(-1).item()]
    finally:
        torch.set_num_threads(previous_threads)


def test_actual_incompatible_architecture_keeps_cached_checkpoint(tmp_path, cache):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(tiny_model().state_dict(), a)
    original = cache._load_model(str(a))
    torch.save(tiny_model(emb_dim=16).state_dict(), b)
    with pytest.raises(RuntimeError, match="size mismatch"):
        cache._load_model(str(b))
    assert cache._model is original
    assert cache._checkpoint == str(a.resolve())
    assert cache._load_model(str(a)) is original


def test_checkpoint_output_label_dimension_must_match_fixed_labels(tmp_path, cache, monkeypatch):
    waveform = .6 * np.sin(2 * np.pi * 220 * np.arange(2048) / 16000)
    path = tmp_path / "audio.wav"
    sf.write(path, waveform, 16000)
    class WrongLabels:
        def __call__(self, **kwargs):
            return torch.zeros(1, 7)
    monkeypatch.setattr(cache, "_load_model", lambda _: WrongLabels())
    with pytest.raises(ValueError, match="eight emotion labels"):
        cache.infer(path, "test.pt")
