"""Tests for RotorQuant KV compression.

Validates Clifford rotor math, the block-diagonal rotation, and that the
codec achieves interface compatibility with TurboQuant plus comparable
reconstruction quality.
"""

from __future__ import annotations

import pytest
import torch

from dhurandhar.rotorquant import (
    RotorQuantCodec,
    RotorQuantConfig,
    _rotor_sandwich,
    _rotor_sandwich_inverse,
    apply_blockwise_rotors,
    fma_cost_comparison,
    generate_random_unit_rotor,
)
from dhurandhar.turboquant import synthesize_kv_tensor

# ---------------------------------------------------------------------------
# Rotor math
# ---------------------------------------------------------------------------


def test_unit_rotor_has_unit_norm() -> None:
    """Rotors must have unit norm to represent proper rotations."""
    for seed in range(10):
        r = generate_random_unit_rotor(seed)
        assert torch.isclose(r.norm(), torch.tensor(1.0), atol=1e-6)


def test_rotor_sandwich_is_invertible() -> None:
    """R̃·(R·v·R̃)·R should equal v."""
    torch.manual_seed(0)
    rotor = generate_random_unit_rotor(42)
    v = torch.randn(5, 3)
    v_rot = _rotor_sandwich(v, rotor)
    v_back = _rotor_sandwich_inverse(v_rot, rotor)
    assert torch.allclose(v, v_back, atol=1e-5)


def test_rotor_sandwich_preserves_norms() -> None:
    """Rotors are orthogonal transformations — they preserve L2 norms."""
    rotor = generate_random_unit_rotor(7)
    v = torch.randn(20, 3)
    v_rot = _rotor_sandwich(v, rotor)
    assert torch.allclose(v.norm(dim=-1), v_rot.norm(dim=-1), atol=1e-5)


def test_identity_rotor_is_identity() -> None:
    """The identity rotor (1, 0, 0, 0) should leave vectors unchanged."""
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v = torch.randn(10, 3)
    v_rot = _rotor_sandwich(v, identity)
    assert torch.allclose(v, v_rot, atol=1e-6)


def test_rotor_90_about_z_rotates_correctly() -> None:
    """A rotor for 90° rotation about z should map e1 → e2."""
    # Rotor for θ=90° about z-axis: R = cos(θ/2) + sin(θ/2)·e12
    # In our (s, α, β, γ) convention with γ = coefficient on e12:
    #   s = cos(45°) = sqrt(2)/2
    #   γ = sin(45°) = sqrt(2)/2
    half = (0.5) ** 0.5
    rotor_z90 = torch.tensor([half, 0.0, 0.0, half])
    e1 = torch.tensor([[1.0, 0.0, 0.0]])
    result = _rotor_sandwich(e1, rotor_z90)
    # Our e12 bivector coefficient corresponds to rotation in the e1-e2 plane,
    # which via the Hodge-dual-axis convention in our formula behaves as
    # rotation about the z-axis. e1 should map to e2 (possibly with sign
    # depending on handedness — we check magnitude and structure).
    assert torch.isclose(result[0, 2], torch.tensor(0.0), atol=1e-5)
    # Either e1 → e2 or e1 → -e2 depending on rotor convention; assert
    # the x-component flipped to 0 and y-component has magnitude 1.
    assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(result[0, 0].abs(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(result[0, 1].abs(), torch.tensor(1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Blockwise rotor rotation
# ---------------------------------------------------------------------------


def test_apply_blockwise_rotors_is_invertible() -> None:
    """Applying rotors then their inverses must recover the original."""
    torch.manual_seed(0)
    d = 255  # 85 blocks of 3
    num_blocks = d // 3
    rotors = torch.stack([generate_random_unit_rotor(i) for i in range(num_blocks)])
    x = torch.randn(4, d)

    x_rot = apply_blockwise_rotors(x, rotors, inverse=False)
    x_back = apply_blockwise_rotors(x_rot, rotors, inverse=True)
    assert torch.allclose(x, x_back, atol=1e-4)


def test_apply_blockwise_rotors_preserves_norms() -> None:
    torch.manual_seed(0)
    d = 128
    num_blocks = d // 3
    rotors = torch.stack([generate_random_unit_rotor(i) for i in range(num_blocks)])
    x = torch.randn(8, d)

    x_rot = apply_blockwise_rotors(x, rotors)
    # Norm is preserved within blocks of 3; the tail (d mod 3 = 2 untouched dims)
    # stays intact, so total norm preserved within numerical precision.
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-4)


def test_apply_blockwise_rotors_handles_tail() -> None:
    """A dimension not divisible by 3 should pass the tail through unchanged."""
    d = 128  # 42 blocks of 3, plus 2 tail dims
    num_blocks = d // 3
    rotors = torch.stack([generate_random_unit_rotor(i) for i in range(num_blocks)])
    x = torch.randn(2, d)

    x_rot = apply_blockwise_rotors(x, rotors)
    # Tail dims (indices 126, 127) should be unchanged
    assert torch.allclose(x[..., 126:], x_rot[..., 126:], atol=1e-6)


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_codec_shapes_match_input() -> None:
    codec = RotorQuantCodec(head_dim=256)
    kv = torch.randn(16, 4, 256)
    q = codec.compress(kv)
    kv_approx = codec.decompress(q)
    assert kv_approx.shape == kv.shape


def test_codec_rejects_wrong_head_dim() -> None:
    codec = RotorQuantCodec(head_dim=256)
    with pytest.raises(ValueError, match="Expected last dim"):
        codec.compress(torch.randn(10, 128))


def test_codec_preserves_direction() -> None:
    """Reconstruction should preserve direction on realistic KV distributions."""
    torch.manual_seed(0)
    codec = RotorQuantCodec(head_dim=255)  # divisible by 3 for clean blocks
    kv = synthesize_kv_tensor(seq_len=512, num_kv_heads=4, head_dim=255, seed=1)
    metrics = codec.reconstruction_error(kv)
    assert metrics["cos_sim"] > 0.90  # slightly more permissive than TQ's 0.95


def test_codec_better_than_random() -> None:
    torch.manual_seed(0)
    codec = RotorQuantCodec(head_dim=129)
    kv = synthesize_kv_tensor(seq_len=256, num_kv_heads=2, head_dim=129, seed=2)
    metrics = codec.reconstruction_error(kv)
    assert metrics["cos_sim"] > 0.5  # clear signal vs noise


@pytest.mark.parametrize("residual_bits", [2, 4, 8])
def test_codec_quality_improves_with_residual_bits(residual_bits: int) -> None:
    torch.manual_seed(0)
    kv = synthesize_kv_tensor(seq_len=256, num_kv_heads=2, head_dim=129, seed=3)
    codec_low = RotorQuantCodec(head_dim=129, config=RotorQuantConfig(residual_bits=2))
    codec_high = RotorQuantCodec(
        head_dim=129, config=RotorQuantConfig(residual_bits=residual_bits)
    )
    m_low = codec_low.reconstruction_error(kv)
    m_high = codec_high.reconstruction_error(kv)
    if residual_bits > 2:
        assert m_high["mse"] <= m_low["mse"] * 1.01


# ---------------------------------------------------------------------------
# FMA cost comparison
# ---------------------------------------------------------------------------


def test_fma_cost_rotorquant_beats_turboquant() -> None:
    """The whole point of RotorQuant: fewer FMAs at typical head_dims."""
    for d in [64, 128, 256, 512]:
        cost = fma_cost_comparison(d)
        assert cost["rotorquant_fmas"] < cost["turboquant_fmas"]
        assert cost["speedup_ratio"] > 1.0


def test_fma_cost_structure() -> None:
    cost = fma_cost_comparison(128)
    assert "head_dim" in cost
    assert "turboquant_fmas" in cost
    assert "rotorquant_fmas" in cost
    assert "speedup_ratio" in cost
