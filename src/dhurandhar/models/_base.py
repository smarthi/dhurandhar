"""Base ModelProfile dataclass — the single contract every model must satisfy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelProfile:
    """
    Architectural facts for a single model variant.

    All analysis modules (PLE, TurboQuant, RotorQuant, mmap, feasibility)
    are driven purely by this profile — nothing hardcoded downstream.

    Parameters
    ----------
    name
        Canonical slug, e.g. "gemma4-e2b", "qwen2.5-0.5b", "granite-3.3-2b".
    param_count_b
        Total parameter count in billions.
    weight_bytes
        On-disk / in-memory weight footprint in bytes (BF16 by default).
    num_layers
        Total transformer layers (attention + any SSM/DeltaNet layers).
    num_attention_layers
        Layers that actually produce a KV cache.  Critical for TurboQuant
        scope — on Qwen3-style hybrids this is << num_layers.
    num_kv_heads
        KV head count per attention layer (GQA-aware).
    head_dim
        Per-head dimension.
    local_attn_window
        Token window for sliding-window local attention layers (e.g. 512 for
        Gemma4).  None = all attention layers are full/global.
    global_attn_freq
        Every Nth layer uses global (full) attention.  None = all layers full.
    supports_mmap
        Whether weights can be memory-mapped rather than fully resident.
    dtype_default
        Weight dtype string: "bfloat16" | "float16" | "int8" | "int4".
    kv_cache_dtype
        KV cache accumulation dtype.
    architecture_family
        High-level family for family-specific analysis quirks.
    notes
        Free-text provenance / caveats.
    """

    name:                  str
    param_count_b:         float
    weight_bytes:          int
    num_layers:            int
    num_attention_layers:  int
    num_kv_heads:          int
    head_dim:              int
    local_attn_window:     int | None   = None
    global_attn_freq:      int | None   = None
    supports_mmap:         bool         = True
    dtype_default:         str          = "bfloat16"
    kv_cache_dtype:        str          = "float16"
    architecture_family:   str          = "unknown"
    notes:                 str          = ""

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def weight_gb(self) -> float:
        return self.weight_bytes / (1024 ** 3)

    @property
    def kv_layers_fraction(self) -> float:
        """Fraction of layers that contribute to KV cache (relevant for compression scope)."""
        return self.num_attention_layers / self.num_layers

    def kv_cache_bytes(self, context_len: int, batch_size: int = 1) -> int:
        """
        Peak KV cache footprint in bytes.

        For models with local_attn_window, local layers cap at the window size;
        global layers use full context_len.
        """
        bytes_per_element = 2  # float16

        if self.local_attn_window is None or self.global_attn_freq is None:
            # All attention layers are full-context
            effective_len = context_len
            tokens_per_layer = effective_len
        else:
            global_layers = self.num_attention_layers // self.global_attn_freq
            local_layers  = self.num_attention_layers - global_layers
            tokens_per_layer = (
                (
                    global_layers * context_len
                    + local_layers * min(context_len, self.local_attn_window)
                )
                / self.num_attention_layers
            )

        kv_bytes = (
            self.num_attention_layers   # layers
            * 2                          # K + V
            * self.num_kv_heads
            * self.head_dim
            * tokens_per_layer
            * batch_size
            * bytes_per_element
        )
        return int(kv_bytes)

    def __repr__(self) -> str:
        return (
            f"ModelProfile(name={self.name!r}, "
            f"params={self.param_count_b}B, "
            f"weights={self.weight_gb:.2f}GB, "
            f"attn_layers={self.num_attention_layers}/{self.num_layers})"
        )
