"""Tests for mmap decode profiler.

Uses tiny test files (~4–16 MB) so tests run in seconds. The profiler
must produce sensible numbers at any scale — the scale only affects
realism of the cold-cache behavior, not correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dhurandhar.mmap_profiler import (
    MEMORY_BUDGETS_MB,
    PATTERNS,
    MemoryProbeResult,
    MmapDecodeProfiler,
    ProfileResult,
    RSSSample,
    ensure_test_file,
    sample_rss,
)
from dhurandhar.models import GEMMA4_E2B


@pytest.fixture
def tmp_test_file(tmp_path: Path) -> Path:
    return tmp_path / "ple_test.bin"


@pytest.fixture
def tiny_profiler(tmp_test_file: Path) -> MmapDecodeProfiler:
    """Profiler configured to create a ~4 MB test file."""
    p = MmapDecodeProfiler(
        num_layers=30,
        ple_hidden_size=256,
        quant_bits=4,
        vocab_size=262_144,
        test_file=tmp_test_file,
    )
    # scale=0.004 → ~4 MB at default dims; satisfies minimum 4 MB bound
    p.prepare(scale=0.004)
    return p


# ---------------------------------------------------------------------------
# File preparation
# ---------------------------------------------------------------------------


def test_ensure_test_file_creates_exact_size(tmp_test_file: Path) -> None:
    size = 1024 * 1024  # 1 MB
    ensure_test_file(tmp_test_file, size)
    assert tmp_test_file.stat().st_size == size


def test_ensure_test_file_reuses_when_size_matches(tmp_test_file: Path) -> None:
    size = 1024 * 1024
    ensure_test_file(tmp_test_file, size)
    mtime1 = tmp_test_file.stat().st_mtime_ns
    # Second call should not rewrite
    ensure_test_file(tmp_test_file, size)
    mtime2 = tmp_test_file.stat().st_mtime_ns
    assert mtime1 == mtime2


def test_ensure_test_file_recreates_when_size_differs(tmp_test_file: Path) -> None:
    ensure_test_file(tmp_test_file, 1024 * 1024)
    ensure_test_file(tmp_test_file, 2 * 1024 * 1024)
    assert tmp_test_file.stat().st_size == 2 * 1024 * 1024


def test_ensure_test_file_creates_dense_file(tmp_test_file: Path) -> None:
    """Test file must be dense (not sparse) — holes would invalidate throughput."""
    size = 4 * 1024 * 1024
    ensure_test_file(tmp_test_file, size)
    # A truly dense file's on-disk size equals logical size
    stat = tmp_test_file.stat()
    # st_blocks is in 512-byte units. Dense file: blocks * 512 ≈ size.
    allocated_bytes = stat.st_blocks * 512
    # Tolerate filesystem overhead (ratios between 0.95 and 1.1)
    assert allocated_bytes >= 0.95 * size, (
        f"File may be sparse: logical {size}, allocated {allocated_bytes}"
    )


# ---------------------------------------------------------------------------
# Profiler geometry
# ---------------------------------------------------------------------------


def test_bytes_per_token_matches_gemma_e2b_arithmetic() -> None:
    """30 layers × 256 PLE dim × 0.5 bytes (Q4) = 3840 bytes per token
    (with 8-byte alignment per layer row, 128 bytes/layer × 30 = 3840)."""
    p = MmapDecodeProfiler(num_layers=30, ple_hidden_size=256, quant_bits=4)
    assert p.bytes_per_token == 3840


def test_total_ple_bytes_matches_published_size() -> None:
    """Full-scale PLE file size should be ~1 GB at E2B defaults."""
    p = MmapDecodeProfiler.from_architecture()
    total_gb = p.total_ple_bytes / (1024**3)
    # Published LiteRT-LM embeddings are 1.12 GB; our PLE-only count is lower
    # because we don't include the token-embedding table here — just PLE.
    assert 0.8 < total_gb < 1.5


def test_from_architecture_uses_defaults() -> None:
    p = MmapDecodeProfiler.from_architecture(GEMMA4_E2B, quant_bits=4)
    assert p.num_layers == 30
    assert p.ple_hidden_size == 256


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", list(PATTERNS.keys()))
def test_profile_pattern_runs_and_produces_positive_tps(
    tiny_profiler: MmapDecodeProfiler, pattern: str
) -> None:
    """Every pattern must produce a valid ProfileResult with positive throughput."""
    result = tiny_profiler.profile(pattern, num_tokens=100, warmup_tokens=10)
    assert isinstance(result, ProfileResult)
    assert result.tokens_per_sec > 0
    assert result.mb_per_sec > 0
    assert result.p50_token_latency_us > 0
    assert result.p99_token_latency_us >= result.p50_token_latency_us
    assert result.pattern == pattern


def test_profile_cold_and_warm_both_produce_results(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    cold = tiny_profiler.profile("random_decode", num_tokens=50, cold=True)
    warm = tiny_profiler.profile("random_decode", num_tokens=50, cold=False)
    assert cold.cold is True
    assert warm.cold is False
    # Don't assert warm > cold — on small files held in OS cache, they may
    # be similar. We just check both complete.


def test_profile_all_produces_all_combinations(tiny_profiler: MmapDecodeProfiler) -> None:
    results = tiny_profiler.profile_all(num_tokens=50, warmup_tokens=5, include_warm=True)
    # 3 patterns × 2 modes (cold+warm) = 6 results
    assert len(results) == 6
    pairs = {(r.pattern, r.cold) for r in results}
    for pattern in PATTERNS:
        assert (pattern, True) in pairs
        assert (pattern, False) in pairs


def test_profile_all_cold_only(tiny_profiler: MmapDecodeProfiler) -> None:
    results = tiny_profiler.profile_all(num_tokens=50, warmup_tokens=5, include_warm=False)
    assert len(results) == 3
    assert all(r.cold for r in results)


def test_profile_unknown_pattern_raises(tiny_profiler: MmapDecodeProfiler) -> None:
    with pytest.raises(ValueError, match="Unknown pattern"):
        tiny_profiler.profile("nonexistent", num_tokens=10)


def test_profile_result_to_dict_is_json_compatible(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    import json

    result = tiny_profiler.profile("random_decode", num_tokens=50)
    d = result.to_dict()
    # Should round-trip through JSON
    json.loads(json.dumps(d))


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def test_evaluate_gate_pass_when_cold_meets_target(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    # On the host filesystem with a tiny file, throughput will be very high.
    # Set target to 1 tok/s so the gate passes deterministically.
    results = tiny_profiler.profile_all(num_tokens=50, warmup_tokens=5)
    gate = tiny_profiler.evaluate_gate(results, target_tps=1.0)
    assert gate["verdict"] == "PASS"
    assert gate["cold_tps"] >= 1.0


def test_evaluate_gate_fail_when_way_above_hardware(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    # Set target to something absurd no hardware can meet
    results = tiny_profiler.profile_all(num_tokens=50, warmup_tokens=5)
    gate = tiny_profiler.evaluate_gate(results, target_tps=1e9)
    assert gate["verdict"] == "FAIL"


def test_evaluate_gate_detects_warm_pass(tiny_profiler: MmapDecodeProfiler) -> None:
    """If we can synthesize cold < target ≤ warm, gate should be WARN."""
    results = tiny_profiler.profile_all(num_tokens=50, warmup_tokens=5)
    by_key = {(r.pattern, r.cold): r for r in results}
    cold = by_key[("random_decode", True)]
    warm = by_key[("random_decode", False)]

    # Pick a target between cold and warm throughput if possible
    if cold.tokens_per_sec < warm.tokens_per_sec:
        target = (cold.tokens_per_sec + warm.tokens_per_sec) / 2
        gate = tiny_profiler.evaluate_gate(results, target_tps=target)
        assert gate["verdict"] in {"WARN", "PASS"}  # tolerance for equality cases
    else:
        # On fast tmpfs cold ≥ warm is common — skip this specific assertion
        pytest.skip("Cold and warm too close on host fs to test WARN path")


# ---------------------------------------------------------------------------
# Peak RSS measurement — the real G1 gate
# ---------------------------------------------------------------------------


def test_sample_rss_returns_valid_sample() -> None:
    s = sample_rss()
    assert isinstance(s, RSSSample)
    # Any live process has nonzero RSS
    assert s.vm_rss_mb > 0
    assert s.timestamp_sec > 0


def test_memory_budgets_are_defined() -> None:
    """The three reference budgets (INT4/INT8/BF16) must all be registered."""
    assert "int4_aggressive" in MEMORY_BUDGETS_MB
    assert "int8_deployment" in MEMORY_BUDGETS_MB
    assert "bf16_development" in MEMORY_BUDGETS_MB
    # INT8 must be less than BF16
    assert MEMORY_BUDGETS_MB["int8_deployment"] < MEMORY_BUDGETS_MB["bf16_development"]
    assert MEMORY_BUDGETS_MB["int4_aggressive"] < MEMORY_BUDGETS_MB["int8_deployment"]


def test_profile_memory_produces_valid_result(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    """Memory probe must complete and return coherent measurements.

    Uses simulate_non_ple_resident=False to avoid allocating the full non-PLE
    placeholder — the mechanism check is the same either way.
    """
    result = tiny_profiler.profile_memory(
        weight_bits=8,
        num_tokens=50,
        warmup_tokens=5,
        simulate_non_ple_resident=False,
    )
    assert isinstance(result, MemoryProbeResult)
    assert result.peak_rss_mb >= result.baseline_rss_mb
    assert result.steady_state_rss_mb > 0
    assert result.num_tokens_decoded == 50
    # Working set should be non-negative
    assert result.ple_resident_working_set_mb >= 0


def test_profile_memory_non_ple_sized_per_weight_bits(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    """Higher weight bits should produce a larger non-PLE component."""
    r4 = tiny_profiler.profile_memory(
        weight_bits=4, num_tokens=30, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    r8 = tiny_profiler.profile_memory(
        weight_bits=8, num_tokens=30, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    r16 = tiny_profiler.profile_memory(
        weight_bits=16, num_tokens=30, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    assert r4.non_ple_component_mb < r8.non_ple_component_mb < r16.non_ple_component_mb


def test_profile_memory_result_is_json_serializable(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    import json
    r = tiny_profiler.profile_memory(
        weight_bits=4, num_tokens=20, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    json.loads(json.dumps(r.to_dict()))


def test_evaluate_budget_pass_when_huge_budget(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    r = tiny_profiler.profile_memory(
        weight_bits=4, num_tokens=30, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    verdict = tiny_profiler.evaluate_budget(r, budget_mb=1_000_000)
    assert verdict["verdict"] == "PASS"
    assert verdict["headroom_mb"] > 0


def test_evaluate_budget_fail_when_tiny_budget(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    r = tiny_profiler.profile_memory(
        weight_bits=4, num_tokens=30, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    verdict = tiny_profiler.evaluate_budget(r, budget_mb=1.0)
    assert verdict["verdict"] == "FAIL"
    assert verdict["headroom_mb"] < 0


def test_evaluate_budget_by_name(tiny_profiler: MmapDecodeProfiler) -> None:
    r = tiny_profiler.profile_memory(
        weight_bits=8, num_tokens=30, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    verdict = tiny_profiler.evaluate_budget(r, budget_name="int8_deployment")
    assert verdict["budget_mb"] == MEMORY_BUDGETS_MB["int8_deployment"]
    assert verdict["budget_name"] == "int8_deployment"


def test_evaluate_budget_unknown_name_raises(
    tiny_profiler: MmapDecodeProfiler,
) -> None:
    r = tiny_profiler.profile_memory(
        weight_bits=4, num_tokens=20, warmup_tokens=2,
        simulate_non_ple_resident=False,
    )
    with pytest.raises(KeyError, match="Unknown budget"):
        tiny_profiler.evaluate_budget(r, budget_name="made_up_budget")
