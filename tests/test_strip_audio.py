"""Tests for strip_audio_encoder utility.

These tests build a mock nn.Module with the Gemma 4 attribute layout and
verify that strip_audio_encoder correctly identifies and removes audio-related
submodules without affecting decoder or vision weights.
"""

from __future__ import annotations

import torch.nn as nn

from dhurandhar.finetune import strip_audio_encoder


class MockGemma4Config:
    """Stand-in for transformers PretrainedConfig with Gemma 4 fields."""

    def __init__(self):
        self.has_audio_encoder = True
        self.audio_config = {"some": "audio_config"}
        self.vocab_size = 262_144


class MockAudioTower(nn.Module):
    """Mimics the ~300 MB Gemma 4 audio encoder at small scale."""

    def __init__(self):
        super().__init__()
        # Realistic-ish: conv front-end + a few transformer-like blocks
        self.frontend = nn.Conv1d(80, 1024, kernel_size=3)
        self.blocks = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(4)
        ])
        self.output_proj = nn.Linear(1024, 2048)


class MockVisionTower(nn.Module):
    """Vision encoder — must NOT be stripped."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.layers = nn.ModuleList([nn.Linear(768, 768) for _ in range(2)])


class MockDecoder(nn.Module):
    """Text decoder — must NOT be stripped."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(6)])


class MockGemma4Model(nn.Module):
    """Mock Gemma 4 multimodal model with audio + vision + decoder."""

    def __init__(self, audio_at_top: bool = True):
        super().__init__()
        if audio_at_top:
            # Multimodal checkpoints often expose audio_tower at top level
            self.audio_tower = MockAudioTower()
            self.audio_projector = nn.Linear(2048, 2048)
        else:
            # Other variants nest it under `model`
            self.model = nn.Module()
            self.model.audio_tower = MockAudioTower()
        self.vision_tower = MockVisionTower()
        self.decoder = MockDecoder()
        self.config = MockGemma4Config()


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_strip_removes_audio_tower_at_top_level() -> None:
    model = MockGemma4Model(audio_at_top=True)
    total_before = _count_params(model)
    audio_params = _count_params(model.audio_tower) + _count_params(model.audio_projector)

    model, diag = strip_audio_encoder(model)

    # Audio tower should be replaced with Identity (0 params)
    assert isinstance(model.audio_tower, nn.Identity)
    assert isinstance(model.audio_projector, nn.Identity)

    # Diagnostic is correct
    assert "audio_tower" in diag["stripped"]
    assert "audio_projector" in diag["stripped"]
    assert diag["params_removed"] == audio_params
    assert diag["bytes_removed"] > 0

    total_after = _count_params(model)
    assert total_after == total_before - audio_params


def test_strip_removes_nested_audio_tower() -> None:
    """Some Gemma 4 variants nest audio under model.audio_tower."""
    model = MockGemma4Model(audio_at_top=False)
    audio_params_before = _count_params(model.model.audio_tower)

    model, diag = strip_audio_encoder(model)

    assert isinstance(model.model.audio_tower, nn.Identity)
    assert any("audio_tower" in s for s in diag["stripped"])
    assert diag["params_removed"] == audio_params_before


def test_strip_preserves_vision_tower() -> None:
    """Vision encoder must not be touched."""
    model = MockGemma4Model()
    vision_params_before = _count_params(model.vision_tower)

    model, _ = strip_audio_encoder(model)

    vision_params_after = _count_params(model.vision_tower)
    assert vision_params_after == vision_params_before
    assert isinstance(model.vision_tower, MockVisionTower)  # class intact


def test_strip_preserves_decoder() -> None:
    """Text decoder must remain completely intact."""
    model = MockGemma4Model()
    decoder_params_before = _count_params(model.decoder)

    model, _ = strip_audio_encoder(model)

    decoder_params_after = _count_params(model.decoder)
    assert decoder_params_after == decoder_params_before


def test_strip_updates_config_flags() -> None:
    model = MockGemma4Model()
    model, _ = strip_audio_encoder(model)

    assert model.config.has_audio_encoder is False
    assert model.config.audio_config is None


def test_strip_is_idempotent() -> None:
    """Running strip twice should be safe — second call finds nothing."""
    model = MockGemma4Model()

    model, diag1 = strip_audio_encoder(model)
    assert diag1["params_removed"] > 0

    model, diag2 = strip_audio_encoder(model)
    # Second call: audio_tower is now Identity, so no params to remove
    assert diag2["params_removed"] == 0


def test_strip_handles_text_only_model_gracefully() -> None:
    """Text-only checkpoints have no audio stack — strip should no-op cleanly."""
    class TextOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = MockDecoder()
            self.config = MockGemma4Config()
            # Explicitly remove audio_config so the config-flag clearing is a no-op
            self.config.has_audio_encoder = False

    model = TextOnlyModel()
    params_before = _count_params(model)

    model, diag = strip_audio_encoder(model)

    assert diag["stripped"] == []
    assert diag["params_removed"] == 0
    assert diag["skipped"] is not None
    assert "text-only" in diag["skipped"] or "already stripped" in diag["skipped"]
    assert _count_params(model) == params_before


def test_strip_diagnostic_dict_structure() -> None:
    model = MockGemma4Model()
    _, diag = strip_audio_encoder(model)
    # Required keys
    assert set(diag.keys()) == {"stripped", "params_removed", "bytes_removed", "skipped"}
    assert isinstance(diag["stripped"], list)
    assert isinstance(diag["params_removed"], int)
    assert isinstance(diag["bytes_removed"], int)
