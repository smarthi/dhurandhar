"""Tests for the built-in model registry."""

from __future__ import annotations

from dhurandhar.models import EMBEDDINGGEMMA2, get_model, list_models


def test_embeddinggemma2_registered() -> None:
    assert "embeddinggemma2" in list_models()
    assert get_model("embeddinggemma2") is EMBEDDINGGEMMA2


def test_embeddinggemma2_layer_pattern_matches_config() -> None:
    """HF config layer_types: full attention at layers 5, 11, 17, 23."""
    arch = EMBEDDINGGEMMA2
    assert arch.global_layer_indices() == [5, 11, 17, 23]
    assert len(arch.local_layer_indices()) == 20


def test_embeddinggemma2_kv_geometry() -> None:
    """Local: 2 KV heads x 256; global: 1 KV head x 512, at 2 bytes."""
    arch = EMBEDDINGGEMMA2
    ctx = 4096
    local = 20 * min(ctx, arch.sliding_window) * 2 * 256 * 2 * 2
    glob = 4 * ctx * 1 * 512 * 2 * 2
    assert arch.kv_cache_bytes(ctx, 16) == local + glob
