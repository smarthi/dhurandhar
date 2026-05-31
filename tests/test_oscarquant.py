"""Tests for OScaR KV compression.

Exercises the canalized rotation, omni-token scaling, and groupwise quant
end-to-end on synthetic KV tensors. No GPU / model weights required.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from dhurandhar.oscarquant import (
    OScaRCodec,
    OScaRConfig,
    fma_cost_comparison,
)
from dhurandhar.turboquant import (
    apply_randomized_hadamard,
    hadamard_matrix,
    invert_randomized_hadamard,
    synthesize_kv_tensor,
)

# ---------------------------------------------------------------------------
# Reused rotation primitives — sanity that the OScaR construction matches
# TurboQuant's invariants. (OScaR shares the helpers; this guards the wiring.)
# ---------------------------------------------------------------------------


def test_canalized_rotation_is_invertible() -> None:
    """The (randomized) Hadamard used by OScaR must round-trip exactly."""
    torch.manual_seed(0)
    d = 256
    x = torch.randn(8, d)
    signs = torch.randint(0, 2, (d,)).float() * 2 - 1
    H = hadamard_matrix(d)  # noqa: N806

    x_rot = apply_randomized_hadamard(x, signs, H)
    x_back = invert_randomized_hadamard(x_rot, signs, H)
    assert torch.allclose(x, x_back, atol=1e-5)


def test_canalized_rotation_preserves_inner_product() -> None:
    """Q·K must be preserved under the canalized rotation — the point of
    applying it to both queries and keys at attention time."""
    torch.manual_seed(0)
    d = 128
    q = torch.randn(4, d)
    k = torch.randn(4, d)
    signs = torch.randint(0, 2, (d,)).float() * 2 - 1
    H = hadamard_matrix(d)  # noqa: N806

    qk = (q * k).sum(dim=-1)
    qk_rot = (
        apply_randomized_hadamard(q, signs, H)
        * apply_randomized_hadamard(k, signs, H)
    ).sum(dim=-1)
    assert torch.allclose(qk, qk_rot, atol=1e-4)


# ---------------------------------------------------------------------------
# Codec round-trip — shapes
# ---------------------------------------------------------------------------


def test_codec_key_path_shapes_match_input() -> None:
    codec = OScaRCodec(head_dim=128)
    kv = torch.randn(16, 4, 128)
    q = codec.compress(kv)
    kv_approx = codec.decompress(q)
    assert kv_approx.shape == kv.shape
    assert q.tau is not None
    assert q.tau.shape == (16, 4)
    assert q.scale.shape == (16, 4, 128 // 32)


def test_codec_value_path_shapes_match_input() -> None:
    codec = OScaRCodec(head_dim=128, config=OScaRConfig(is_value=True))
    kv = torch.randn(16, 4, 128)
    q = codec.compress(kv)
    kv_approx = codec.decompress(q)
    assert kv_approx.shape == kv.shape
    assert q.tau is None
    assert q.scale.shape == (16, 4)  # per-token scalar


def test_codec_handles_head_dim_non_power_of_two() -> None:
    """Padding must transparently handle a non-power-of-two head_dim."""
    codec = OScaRCodec(head_dim=200)  # pads to 256
    kv = torch.randn(10, 200)
    q = codec.compress(kv)
    kv_approx = codec.decompress(q)
    assert kv_approx.shape == kv.shape


def test_codec_rejects_wrong_head_dim() -> None:
    codec = OScaRCodec(head_dim=128)
    with pytest.raises(ValueError, match="Expected last dim"):
        codec.compress(torch.randn(10, 64))


def test_codec_rejects_padded_dim_not_divisible_by_group_size() -> None:
    # head_dim=64 pads to 64; group_size=48 doesn't divide it.
    with pytest.raises(ValueError, match="divisible by"):
        OScaRCodec(head_dim=64, config=OScaRConfig(group_size=48))


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------


def test_reconstruction_metrics_are_finite() -> None:
    """All quality metrics should be finite numbers (no NaNs/infs)."""
    torch.manual_seed(0)
    codec = OScaRCodec(head_dim=128)
    kv = synthesize_kv_tensor(seq_len=256, num_kv_heads=2, head_dim=128, seed=1)
    metrics = codec.reconstruction_error(kv)
    for k, v in metrics.items():
        assert math.isfinite(v), f"metric {k}={v} is not finite"


def test_codec_preserves_direction_on_heavy_tail() -> None:
    """Cosine similarity should stay high on realistic heavy-tail KV."""
    torch.manual_seed(0)
    codec = OScaRCodec(head_dim=128, config=OScaRConfig(key_bits=4))
    kv = synthesize_kv_tensor(seq_len=512, num_kv_heads=4, head_dim=128, seed=1)
    metrics = codec.reconstruction_error(kv)
    assert metrics["cos_sim"] > 0.90, (
        f"cosine similarity should be > 0.90, got {metrics['cos_sim']:.4f}"
    )
    assert 0.85 < metrics["norm_ratio"] < 1.15


def test_value_path_quality_at_int4() -> None:
    """Value path (rotation + per-token quant) should reconstruct well at INT4."""
    torch.manual_seed(0)
    codec = OScaRCodec(
        head_dim=128,
        config=OScaRConfig(is_value=True, value_bits=4),
    )
    kv = synthesize_kv_tensor(seq_len=512, num_kv_heads=4, head_dim=128, seed=2)
    metrics = codec.reconstruction_error(kv)
    assert metrics["cos_sim"] > 0.90


@pytest.mark.parametrize("key_bits", [2, 4, 8])
def test_quality_improves_with_more_bits(key_bits: int) -> None:
    """Higher key_bits should not produce a worse MSE than 2-bit."""
    torch.manual_seed(0)
    kv = synthesize_kv_tensor(seq_len=256, num_kv_heads=2, head_dim=128, seed=3)
    low = OScaRCodec(head_dim=128, config=OScaRConfig(key_bits=2)).reconstruction_error(kv)
    high = OScaRCodec(
        head_dim=128, config=OScaRConfig(key_bits=key_bits)
    ).reconstruction_error(kv)
    if key_bits > 2:
        assert high["mse"] <= low["mse"] * 1.05


# ---------------------------------------------------------------------------
# INT2 sanity vs naive per-channel baseline
# ---------------------------------------------------------------------------


def _naive_per_channel_int_quant(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """Naive per-channel symmetric INT quantization (no rotation, no scaling).

    Each channel gets a single scale computed over all leading dims —
    the most common quantization baseline before any rotation-based
    flattening. OScaR should beat this on heavy-tail KV at the same
    bit budget.
    """
    qmax = (1 << (bits - 1)) - 1
    flat = kv.reshape(-1, kv.shape[-1])
    abs_max = flat.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)  # (1, d)
    scale = abs_max / qmax
    q = torch.round(flat / scale).clamp(-qmax - 1, qmax)
    deq = q * scale
    return deq.reshape(kv.shape)


def test_oscar_int2_no_worse_than_naive_per_channel_baseline() -> None:
    """The whole point of canalized rotation + omni-token scaling at low
    bit budgets: OScaR's MSE at INT2 should be no worse than the naive
    per-channel INT2 baseline on heavy-tail KV."""
    torch.manual_seed(0)
    kv = synthesize_kv_tensor(seq_len=512, num_kv_heads=4, head_dim=128, seed=11)

    naive_recon = _naive_per_channel_int_quant(kv, bits=2)
    naive_mse = F.mse_loss(naive_recon, kv).item()

    codec = OScaRCodec(head_dim=128, config=OScaRConfig(key_bits=2))
    oscar_mse = codec.reconstruction_error(kv)["mse"]

    # OScaR should be at least as good as the naive baseline (small tolerance
    # for stochastic noise from the random Hadamard signs).
    assert oscar_mse <= naive_mse * 1.05, (
        f"OScaR INT2 mse={oscar_mse:.5f} worse than naive per-channel "
        f"baseline mse={naive_mse:.5f}"
    )


# ---------------------------------------------------------------------------
# FMA cost
# ---------------------------------------------------------------------------


def test_fma_cost_structure() -> None:
    cost = fma_cost_comparison(128)
    assert "head_dim" in cost
    assert "turboquant_fmas" in cost
    assert "oscar_fmas" in cost
    assert "overhead_ratio" in cost
    # OScaR adds a small constant-factor overhead per vector
    assert cost["oscar_fmas"] > cost["turboquant_fmas"]
    assert cost["overhead_ratio"] >= 1.0


def test_fma_cost_overhead_is_modest() -> None:
    """OScaR's per-vector overhead vs TurboQuant should stay well below 2x
    for typical head_dims — the work is dominated by the shared Hadamard."""
    for d in [64, 128, 256, 512]:
        c = fma_cost_comparison(d)
        assert c["overhead_ratio"] < 2.0
