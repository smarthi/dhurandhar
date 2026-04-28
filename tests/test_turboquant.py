"""Tests for TurboQuant KV compression.

These tests exercise the numerical code end-to-end. They do NOT require
a GPU or model weights — they use synthetic tensors that mimic real KV
statistics.
"""

from __future__ import annotations

import pytest
import torch

from dhurandhar.turboquant import (
    KVCacheCompressor,
    TurboQuantCodec,
    TurboQuantConfig,
    apply_randomized_hadamard,
    hadamard_matrix,
    invert_randomized_hadamard,
    synthesize_kv_tensor,
)

# ---------------------------------------------------------------------------
# Hadamard properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128, 256])
def test_hadamard_is_orthogonal(n: int) -> None:
    H = hadamard_matrix(n)  # noqa: N806
    identity = H @ H.T
    assert torch.allclose(identity, torch.eye(n), atol=1e-5), (
        f"H @ H.T should be identity for n={n}"
    )


def test_hadamard_non_power_of_two_raises() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        hadamard_matrix(5)


def test_randomized_hadamard_is_invertible() -> None:
    torch.manual_seed(0)
    d = 256
    x = torch.randn(10, d)
    signs = torch.randint(0, 2, (d,)).float() * 2 - 1
    H = hadamard_matrix(d)  # noqa: N806

    x_rot = apply_randomized_hadamard(x, signs, H)
    x_back = invert_randomized_hadamard(x_rot, signs, H)

    assert torch.allclose(x, x_back, atol=1e-5), "Rotation should be exactly invertible"


def test_randomized_hadamard_preserves_norms() -> None:
    """Orthogonal transformations preserve L2 norms."""
    torch.manual_seed(0)
    d = 128
    x = torch.randn(50, d)
    signs = torch.randint(0, 2, (d,)).float() * 2 - 1
    H = hadamard_matrix(d)  # noqa: N806

    x_rot = apply_randomized_hadamard(x, signs, H)
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-4)


# ---------------------------------------------------------------------------
# Bit packing / unpacking
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip() -> None:
    torch.manual_seed(0)
    d = 256
    bits = (torch.rand(4, d) > 0.5).to(torch.uint8)
    packed = TurboQuantCodec._pack_bits(bits)
    unpacked = TurboQuantCodec._unpack_bits(packed, length=d)
    assert torch.equal(bits, unpacked), "Bit pack/unpack must be exact"


def test_pack_handles_non_multiple_of_8() -> None:
    torch.manual_seed(0)
    d = 250  # not a multiple of 8
    bits = (torch.rand(3, d) > 0.5).to(torch.uint8)
    packed = TurboQuantCodec._pack_bits(bits)
    unpacked = TurboQuantCodec._unpack_bits(packed, length=d)
    assert torch.equal(bits, unpacked)


# ---------------------------------------------------------------------------
# Codec round-trip quality
# ---------------------------------------------------------------------------


def test_codec_shapes_match_input() -> None:
    """Decompressed output must have the same shape as input."""
    codec = TurboQuantCodec(head_dim=256)
    kv = torch.randn(16, 4, 256)  # (seq, heads, head_dim)
    q = codec.compress(kv)
    kv_approx = codec.decompress(q)
    assert kv_approx.shape == kv.shape


def test_codec_preserves_direction() -> None:
    """Reconstruction should have high cosine similarity with original."""
    torch.manual_seed(0)
    codec = TurboQuantCodec(head_dim=256)
    kv = synthesize_kv_tensor(seq_len=512, num_kv_heads=4, head_dim=256, seed=1)
    metrics = codec.reconstruction_error(kv)

    # TurboQuant at 4-bit residual should preserve direction well
    assert metrics["cos_sim"] > 0.95, (
        f"Cosine similarity should be > 0.95, got {metrics['cos_sim']:.4f}"
    )
    assert 0.90 < metrics["norm_ratio"] < 1.10, (
        f"Norm preservation should be within 10%, got {metrics['norm_ratio']:.4f}"
    )


def test_codec_better_than_random_baseline() -> None:
    """TurboQuant reconstruction must beat a random-matrix baseline."""
    torch.manual_seed(0)
    codec = TurboQuantCodec(head_dim=128)
    kv = synthesize_kv_tensor(seq_len=256, num_kv_heads=2, head_dim=128, seed=2)
    metrics = codec.reconstruction_error(kv)

    # Random reconstruction would have cos_sim near 0
    assert metrics["cos_sim"] > 0.5


@pytest.mark.parametrize("residual_bits", [2, 4, 8])
def test_codec_quality_improves_with_residual_bits(residual_bits: int) -> None:
    """Higher residual precision must produce better reconstruction."""
    torch.manual_seed(0)
    kv = synthesize_kv_tensor(seq_len=256, num_kv_heads=2, head_dim=128, seed=3)

    codec_low = TurboQuantCodec(head_dim=128, config=TurboQuantConfig(residual_bits=2))
    codec_high = TurboQuantCodec(head_dim=128, config=TurboQuantConfig(residual_bits=residual_bits))

    m_low = codec_low.reconstruction_error(kv)
    m_high = codec_high.reconstruction_error(kv)

    if residual_bits > 2:
        assert m_high["mse"] <= m_low["mse"] * 1.01, (
            f"residual_bits={residual_bits} should match or beat residual_bits=2 MSE"
        )


def test_codec_handles_head_dim_non_power_of_two() -> None:
    """Padding should handle head_dims that aren't powers of 2."""
    codec = TurboQuantCodec(head_dim=200)
    kv = torch.randn(10, 200)
    q = codec.compress(kv)
    kv_approx = codec.decompress(q)
    assert kv_approx.shape == kv.shape


def test_codec_rejects_wrong_head_dim() -> None:
    codec = TurboQuantCodec(head_dim=256)
    kv_wrong = torch.randn(10, 128)
    with pytest.raises(ValueError, match="Expected last dim"):
        codec.compress(kv_wrong)


# ---------------------------------------------------------------------------
# Gemma 4 KV cache integration
# ---------------------------------------------------------------------------


def test_shared_kv_layers_return_none() -> None:
    """Shared-KV layers must not be compressed independently."""
    c = KVCacheCompressor(num_layers=30, head_dim=256, shared_kv_last_n=6)

    kv = torch.randn(64, 4, 256)
    # First 24 layers are fresh-KV
    assert c.compress_layer(0, kv) is not None
    assert c.compress_layer(23, kv) is not None
    # Last 6 layers are shared-KV — return None
    assert c.compress_layer(24, kv) is None
    assert c.compress_layer(29, kv) is None


def test_memory_savings_estimate_structure() -> None:
    c = KVCacheCompressor(num_layers=30, head_dim=256, shared_kv_last_n=6)
    savings = c.memory_savings_estimate(seq_len=32768, num_kv_heads=4)

    assert savings["fresh_kv_layers"] == 24
    assert savings["shared_kv_layers"] == 6
    assert savings["baseline_mb"] > savings["quantized_mb"]
    assert savings["savings_ratio"] > 3.0  # ~4.5x expected at 3.5 effective bits


def test_shared_layer_decompress_raises() -> None:
    c = KVCacheCompressor(num_layers=30, head_dim=256, shared_kv_last_n=6)
    # Use a dummy QuantizedVector from a fresh layer
    kv = torch.randn(8, 256)
    q = c.compress_layer(0, kv)
    # Trying to decompress on a shared-KV layer should error
    with pytest.raises(ValueError, match="shared-KV layer"):
        c.decompress_layer(28, q)


# ---------------------------------------------------------------------------
# Synthetic KV distribution
# ---------------------------------------------------------------------------


def test_synthesize_kv_heavy_tail_has_outliers() -> None:
    """Heavy-tail distribution should have more extreme values than pure gaussian."""
    torch.manual_seed(0)
    kv_normal = synthesize_kv_tensor(1000, 4, 128, distribution="gaussian", seed=0)
    kv_heavy = synthesize_kv_tensor(1000, 4, 128, distribution="gaussian_heavy_tail", seed=0)

    # Heavy-tail should have larger max absolute value
    assert kv_heavy.abs().max() > kv_normal.abs().max()


def test_synthesize_kv_deterministic() -> None:
    """Same seed must produce identical tensors."""
    a = synthesize_kv_tensor(100, 4, 64, seed=42)
    b = synthesize_kv_tensor(100, 4, 64, seed=42)
    assert torch.equal(a, b)
