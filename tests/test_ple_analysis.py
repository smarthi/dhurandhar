"""Tests for PLE footprint analysis."""

from __future__ import annotations

import pytest

from dhurandhar.config import DEVICE_PROFILES
from dhurandhar.models import GEMMA4_E2B
from dhurandhar.ple_analysis import PLEFootprintAnalyzer


def test_architecture_layer_partitions_are_complete() -> None:
    """Every layer is either fresh-KV or shared-KV, never both."""
    arch = GEMMA4_E2B
    fresh = set(arch.fresh_kv_layer_indices())
    shared = set(arch.shared_kv_layer_indices())

    assert fresh.isdisjoint(shared)
    assert fresh | shared == set(range(arch.num_hidden_layers))


def test_final_layer_is_global() -> None:
    """Architectural constraint: the final layer is always global attention."""
    arch = GEMMA4_E2B
    globals_ = arch.global_layer_indices()
    assert arch.num_hidden_layers - 1 in globals_


def test_global_and_local_are_disjoint() -> None:
    arch = GEMMA4_E2B
    g = set(arch.global_layer_indices())
    local_layers = set(arch.local_layer_indices())
    assert g.isdisjoint(local_layers)
    assert g | local_layers == set(range(arch.num_hidden_layers))


def test_ple_table_larger_than_decoder() -> None:
    """Core ADR claim: PLE table is larger than the decoder weights.

    This is the key architectural fact driving the mmap-vs-resident decision.
    """
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    breakdown = analyzer.compute_breakdown(quant_bits=4)
    assert breakdown.ple_table_mb > breakdown.decoder_mb, (
        f"PLE table ({breakdown.ple_table_mb:.0f} MB) should exceed "
        f"decoder ({breakdown.decoder_mb:.0f} MB). This is the central "
        f"fact driving the mmap decision."
    )


def test_memory_breakdown_matches_published_sizes() -> None:
    """Published LiteRT-LM sizes: decoder=0.79 GB, embeddings=1.12 GB."""
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    breakdown = analyzer.compute_breakdown(quant_bits=4)

    # Published decoder is 0.79 GB = 809 MB
    assert 700 <= breakdown.decoder_mb <= 900
    # Published embeddings are 1.12 GB = 1147 MB
    assert 1000 <= breakdown.ple_table_mb <= 1250


def test_strip_audio_reduces_footprint() -> None:
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    with_audio = analyzer.compute_breakdown(strip_audio=False)
    without_audio = analyzer.compute_breakdown(strip_audio=True)

    assert without_audio.resident_total_mb < with_audio.resident_total_mb
    assert (
        with_audio.resident_total_mb - without_audio.resident_total_mb
        == pytest.approx(300.0, abs=1.0)
    )


def test_all_device_profiles_are_assessable() -> None:
    """Every device profile produces a coherent feasibility assessment."""
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    for key in DEVICE_PROFILES:
        f = analyzer.assess_device(key)
        assert f.mode in {"resident", "mmap", "infeasible"}
        assert f.device.name
        assert f.rationale


def test_low_end_emmc_flagged_high_risk() -> None:
    """Low-end eMMC device should be flagged as flash-bound or infeasible.

    Devices on eMMC 5.1 with ~0.4 GB/s sustained random-read bandwidth
    likely cannot sustain the PLE mmap decode rate at the 15 tok/s target.
    """
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    f = analyzer.assess_device(
        "low_tier_mobile_emmc",
        context_tokens=32768,
        quant_bits=4,
        decode_tokens_per_sec_target=15.0,
    )
    assert f.mode in {"mmap", "infeasible"}, (
        "Low-end eMMC should at minimum require mmap, "
        "and may be infeasible at 15 tok/s target."
    )


def test_laptop_nvme_has_ample_headroom() -> None:
    """NVMe laptop profile should run E2B resident with lots of headroom."""
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    f = analyzer.assess_device("laptop_nvme")
    assert f.mode == "resident"
    assert f.headroom_mb > 3000  # plenty of slack


def test_unknown_device_raises() -> None:
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    with pytest.raises(KeyError, match="Unknown device"):
        analyzer.assess_device("nonexistent_device")


def test_kv_cache_scales_with_context() -> None:
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    b_short = analyzer.compute_breakdown(context_tokens=2048)
    b_long = analyzer.compute_breakdown(context_tokens=32768)
    assert b_long.kv_cache_mb > b_short.kv_cache_mb


def test_kv_cache_local_layers_bounded_by_window() -> None:
    """Local-attention layers' KV should not scale beyond the sliding window."""
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    # At contexts much larger than the 512-token local window, KV should scale
    # sub-linearly because local layers cap out.
    b1 = analyzer.compute_breakdown(context_tokens=512)
    b2 = analyzer.compute_breakdown(context_tokens=4096)
    ratio = b2.kv_cache_mb / b1.kv_cache_mb
    # Pure-linear would be 8x; we expect < 8x because local layers cap at 512
    assert ratio < 8.0
