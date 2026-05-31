"""OScaR KV cache compression — Omni-Scaled Canalized Rotation.

Based on: "OScaR: Omni-Scaled Canalized Rotation for KV Cache Compression"
(arXiv:2605.19660)

Three training-free stages:

  Stage 1 — Canalized Rotation:
    Apply a (normalized) randomized Hadamard transform to keys to
    redistribute outlier-channel energy before scaling. The same
    transform is applied to queries at attention time so the inner
    product Q·K is preserved (rotations are orthogonal).

  Stage 2 — Omni-Token Scaling:
    Compute each token's L2 norm `tau` after rotation and divide it
    out so every token sits on the unit sphere. The per-token tau is
    stored as 16-bit metadata and restored at dequant. This fixes
    Token Norm Imbalance — the phenomenon where a few tokens with
    very large norm dominate the per-tensor quantization scale and
    push the rest of the sequence toward the quantization noise floor.

  Stage 3 — Quantization:
    * Keys: per-channel INT (group size 32, matching the paper) on the
      rotated, unit-normalized tensor. Channel grouping captures the
      residual non-uniformity Hadamard cannot perfectly flatten.
    * Values: offline Hadamard rotation only (no token scaling), then
      per-token INT quantization. Token scaling on V is harmful — the
      attention output is V · softmax(...), and re-injecting per-token
      norm would double-scale the projected output.

At INT4 keys / INT4 values, OScaR typically lands within ~0.1pp PPL of
full bf16 on standard LLM benchmarks; at INT2 keys it preserves usable
quality where naive per-channel INT2 collapses.

Gemma 4 adaptations (same as TurboQuant):
  * Shared KV Cache: layers in the shared-KV tail are skipped — their
    cached tensors reference earlier layers' (already compressed) KV
    and must not be re-compressed.
  * Sliding-window local layers: compressed only within the active window.
  * p-RoPE on global layers: quantization happens POST-RoPE to preserve
    positional encoding fidelity. (Pre-RoPE quant is quality-lossy.)
  * GQA: compression operates per KV-head, not per query-head.

Reference implementation — CPU/GPU tensor math. For on-device deployment
via LiteRT-LM, a C++ port is required; this module serves as the golden
reference for validation and quality measurement.

Compatibility note:
    The codec mirrors the TurboQuantCodec interface (compress / decompress /
    reconstruction_error) so it can be slotted into the same KVCacheCompressor
    and dashboard harnesses without changes downstream.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel, ConfigDict

from .turboquant import (
    TurboQuantCodec,
    _next_power_of_two,
    apply_randomized_hadamard,
    hadamard_matrix,
    invert_randomized_hadamard,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class OScaRConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    """Configuration for OScaR KV compression."""

    key_bits: int = 4              # INT bits for the keys' per-channel-group quant
    value_bits: int = 4            # INT bits for the values' per-token quant
    group_size: int = 32           # Per-channel group size for keys (paper default: 32)
    rotation_seed: int = 20260421  # Deterministic random signs for reproducibility
    is_value: bool = False         # False = key path; True = value path
    rotate_post_rope: bool = True  # Quantize after RoPE on global layers

    @property
    def effective_bits(self) -> float:
        """Approximate bits/channel including scale + tau overhead.

        Key path: key_bits + 16/group_size (group scale) + 16/d_head (tau).
        Value path: value_bits + 16/d_head (per-token scale).
        Computed at the paper's reference head_dim=128 for a single number.
        """
        ref_head_dim = 128
        if self.is_value:
            return float(self.value_bits) + 16.0 / ref_head_dim
        return float(self.key_bits) + 16.0 / self.group_size + 16.0 / ref_head_dim


# ---------------------------------------------------------------------------
# Compressed representation
# ---------------------------------------------------------------------------


class OScaRQuantizedVector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    """Compressed KV representation.

    Key path populates `tau` (per-token L2 norm after rotation) and stores
    a per-group scale of shape (..., num_groups). Value path leaves `tau`
    as None and stores a per-token scalar scale.
    """

    kv_q: torch.Tensor              # (..., padded_dim) int8 quantized values
    scale: torch.Tensor             # key path: (..., num_groups); value path: (...,)
    tau: torch.Tensor | None        # key path only: (...,) per-token L2 norm
    head_dim: int
    padded_dim: int
    is_value: bool


# ---------------------------------------------------------------------------
# OScaR codec
# ---------------------------------------------------------------------------


class OScaRCodec:
    """Reference OScaR codec. Stateful per (layer, head) for rotation params.

    Typical use:
        codec = OScaRCodec(head_dim=128, config=OScaRConfig(key_bits=4))
        q = codec.compress(kv_tensor)        # (..., head_dim) -> quantized
        kv_approx = codec.decompress(q)      # quantized -> (..., head_dim)

    For the value cache, pass `OScaRConfig(is_value=True, value_bits=...)`.
    The two paths share the same canalized rotation; the difference is the
    omni-token scaling and the quant granularity.
    """

    def __init__(self, head_dim: int, config: OScaRConfig | None = None):
        self.head_dim = head_dim
        self.config = config or OScaRConfig()

        # Pad to next power of two for the Hadamard. Padding is zeroed at
        # encode time and stripped at decode.
        self._padded_dim = _next_power_of_two(head_dim)

        if self._padded_dim % self.config.group_size != 0:
            raise ValueError(
                f"padded head_dim ({self._padded_dim}) must be divisible by "
                f"group_size ({self.config.group_size})"
            )
        self._num_groups = self._padded_dim // self.config.group_size

        # Deterministic random signs seeded for reproducibility
        g = torch.Generator().manual_seed(self.config.rotation_seed)
        signs = torch.randint(0, 2, (self._padded_dim,), generator=g).float() * 2 - 1
        self.signs = signs
        self.H = hadamard_matrix(self._padded_dim)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, kv: torch.Tensor) -> OScaRQuantizedVector:
        """Compress a KV tensor of shape (..., head_dim).

        Operates on the last dim. Batching supported over leading dims
        (e.g., (seq_len, head_dim) or (batch, seq_len, head_dim)).
        """
        if kv.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected last dim = {self.head_dim}, got {kv.shape[-1]}"
            )

        # Pad to power of 2 if needed
        if self._padded_dim != self.head_dim:
            pad_amount = self._padded_dim - self.head_dim
            kv_padded = F.pad(kv, (0, pad_amount))
        else:
            kv_padded = kv

        # Stage 1: canalized rotation (randomized Hadamard)
        kv_rot = apply_randomized_hadamard(kv_padded, self.signs, self.H)

        if self.config.is_value:
            # Value path: rotation only, then per-token quantization.
            # Reuses TurboQuant's per-vector int quantizer.
            kv_q, kv_scale = TurboQuantCodec._quantize_int(kv_rot, self.config.value_bits)
            return OScaRQuantizedVector(
                kv_q=kv_q,
                scale=kv_scale,
                tau=None,
                head_dim=self.head_dim,
                padded_dim=self._padded_dim,
                is_value=True,
            )

        # Key path: rotation + omni-token scaling + per-group quantization.
        # Stage 2: omni-token scaling — divide out each token's L2 norm.
        tau = kv_rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
        kv_unit = kv_rot / tau

        # Stage 3: per-channel-group INT quantization.
        kv_q, kv_scale = self._quantize_groupwise(
            kv_unit, self.config.key_bits, self.config.group_size
        )

        return OScaRQuantizedVector(
            kv_q=kv_q,
            scale=kv_scale,
            tau=tau.squeeze(-1),
            head_dim=self.head_dim,
            padded_dim=self._padded_dim,
            is_value=False,
        )

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    def decompress(self, q: OScaRQuantizedVector) -> torch.Tensor:
        """Decompress back to (..., head_dim) approximation of original."""
        if q.is_value:
            # Per-token dequantization
            kv_rot_approx = q.kv_q.float() * q.scale.unsqueeze(-1)
        else:
            # Per-group dequantization
            kv_unit_approx = self._dequantize_groupwise(
                q.kv_q, q.scale, self.config.group_size
            )
            # Restore the per-token norm
            assert q.tau is not None
            kv_rot_approx = kv_unit_approx * q.tau.unsqueeze(-1)

        # Invert rotation
        kv_padded_approx = invert_randomized_hadamard(kv_rot_approx, self.signs, self.H)

        # Remove padding
        if self._padded_dim != self.head_dim:
            return kv_padded_approx[..., : self.head_dim]
        return kv_padded_approx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_groupwise(
        x: torch.Tensor, bits: int, group_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric per-channel-group linear quantization along the last dim.

        Reshapes (..., d) into (..., num_groups, group_size), then uses
        TurboQuant's per-vector quantizer on the inner dim — equivalent to
        treating each group of `group_size` channels as its own vector for
        the purpose of scale computation.
        """
        d = x.shape[-1]
        if d % group_size != 0:
            raise ValueError(
                f"last dim {d} must be divisible by group_size {group_size}"
            )
        num_groups = d // group_size
        x_grouped = x.reshape(*x.shape[:-1], num_groups, group_size)
        x_q_grouped, scale = TurboQuantCodec._quantize_int(x_grouped, bits)
        x_q = x_q_grouped.reshape(*x.shape[:-1], d)
        return x_q, scale  # scale shape: (..., num_groups)

    @staticmethod
    def _dequantize_groupwise(
        x_q: torch.Tensor, scale: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        """Inverse of `_quantize_groupwise`."""
        d = x_q.shape[-1]
        num_groups = d // group_size
        x_grouped = x_q.float().reshape(*x_q.shape[:-1], num_groups, group_size)
        x_dequant = x_grouped * scale.unsqueeze(-1)
        return x_dequant.reshape(*x_q.shape[:-1], d)

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def reconstruction_error(self, kv: torch.Tensor) -> dict[str, float]:
        """Compute reconstruction quality metrics for a batch of KV vectors.

        Returns:
            mse: mean squared error
            cos_sim: mean cosine similarity to original
            norm_ratio: mean ratio of reconstructed / original L2 norm
            effective_bits: approximate bits/channel including overhead
            compression_ratio: vs bf16
        """
        q = self.compress(kv)
        kv_approx = self.decompress(q)

        mse = F.mse_loss(kv_approx, kv).item()
        cos = F.cosine_similarity(
            kv_approx.reshape(-1, self.head_dim),
            kv.reshape(-1, self.head_dim),
            dim=-1,
        ).mean().item()
        norm_orig = kv.norm(dim=-1)
        norm_approx = kv_approx.norm(dim=-1)
        norm_ratio = (norm_approx / norm_orig.clamp(min=1e-8)).mean().item()

        return {
            "mse": mse,
            "cos_sim": cos,
            "norm_ratio": norm_ratio,
            "effective_bits": self.config.effective_bits,
            "compression_ratio": 16.0 / self.config.effective_bits,
        }


# ---------------------------------------------------------------------------
# Arithmetic cost comparison
# ---------------------------------------------------------------------------


def fma_cost_comparison(head_dim: int) -> dict[str, int | float]:
    """Approximate FMA count per vector for OScaR vs TurboQuant stage-1.

    OScaR uses the same randomized Hadamard as TurboQuant, so the rotation
    cost is identical (d·log2(d) with FWHT, d² with dense matmul). The
    omni-token scaling adds O(d) for the norm + division, and the groupwise
    quant scan is another O(d). Net: a small constant-factor overhead vs
    TurboQuant, with the quality win coming from canalization + scaling.

    Reported numbers use FWHT for the Hadamard, matching the convention in
    the TurboQuant and RotorQuant cost reports for direct comparison.
    """
    tq_fmas = int(head_dim * math.log2(head_dim))
    # OScaR: Hadamard + norm reduce (d) + per-channel div (d) + groupwise scan (d)
    oscar_fmas = tq_fmas + 3 * head_dim

    return {
        "head_dim": head_dim,
        "turboquant_fmas": tq_fmas,
        "oscar_fmas": oscar_fmas,
        "overhead_ratio": round(oscar_fmas / max(tq_fmas, 1), 3),
    }
