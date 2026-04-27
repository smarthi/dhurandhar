"""
Model profile registry.

Usage
-----
    from dhurandhar.models import get_profile, list_profiles, REGISTRY

    profile = get_profile("gemma4-e2b")          # built-in
    profile = get_profile("path/to/custom.yaml")  # YAML override
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dhurandhar.models._base import ModelProfile

# ------------------------------------------------------------------ #
# Reference profiles                                                   #
# ------------------------------------------------------------------ #

_GEMMA4_E2B = ModelProfile(
    name                 = "gemma4-e2b",
    param_count_b        = 2.0,
    weight_bytes         = int(2.0e9 * 2),   # BF16
    num_layers           = 26,
    num_attention_layers = 26,
    num_kv_heads         = 4,
    head_dim             = 256,
    local_attn_window    = 512,
    global_attn_freq     = 6,
    supports_mmap        = True,
    dtype_default        = "bfloat16",
    kv_cache_dtype       = "float16",
    architecture_family  = "gemma",
    notes                = "Gemma4 E2B — sliding-window local attn (512) + global every 6th layer. "
                           "p-RoPE on global layers; quantization applied post-RoPE.",
)

_QWEN25_0_5B = ModelProfile(
    name                 = "qwen2.5-0.5b",
    param_count_b        = 0.5,
    weight_bytes         = int(0.5e9 * 2),
    num_layers           = 24,
    num_attention_layers = 24,
    num_kv_heads         = 2,
    head_dim             = 64,
    supports_mmap        = True,
    dtype_default        = "bfloat16",
    kv_cache_dtype       = "float16",
    architecture_family  = "qwen",
    notes                = "Qwen2.5-0.5B — standard full-attention, GQA 2 KV heads. "
                           "Aggressive GQA limits TurboQuant savings scope.",
)

_QWEN25_1_5B = ModelProfile(
    name                 = "qwen2.5-1.5b",
    param_count_b        = 1.5,
    weight_bytes         = int(1.5e9 * 2),
    num_layers           = 28,
    num_attention_layers = 28,
    num_kv_heads         = 2,
    head_dim             = 128,
    supports_mmap        = True,
    dtype_default        = "bfloat16",
    kv_cache_dtype       = "float16",
    architecture_family  = "qwen",
)

_GRANITE_3_3_2B = ModelProfile(
    name                 = "granite-3.3-2b",
    param_count_b        = 2.0,
    weight_bytes         = int(2.0e9 * 2),
    num_layers           = 40,
    num_attention_layers = 40,
    num_kv_heads         = 8,
    head_dim             = 64,
    supports_mmap        = True,
    dtype_default        = "bfloat16",
    kv_cache_dtype       = "float16",
    architecture_family  = "granite",
    notes                = "IBM Granite 3.3-2B — standard MHA, well-suited for TurboQuant benchmarking.",  # noqa: E501
)

_LLAMA_3_2_1B = ModelProfile(
    name                 = "llama-3.2-1b",
    param_count_b        = 1.0,
    weight_bytes         = int(1.0e9 * 2),
    num_layers           = 16,
    num_attention_layers = 16,
    num_kv_heads         = 8,
    head_dim             = 64,
    supports_mmap        = True,
    dtype_default        = "bfloat16",
    kv_cache_dtype       = "float16",
    architecture_family  = "llama",
)

REGISTRY: dict[str, ModelProfile] = {
    p.name: p for p in [
        _GEMMA4_E2B,
        _QWEN25_0_5B,
        _QWEN25_1_5B,
        _GRANITE_3_3_2B,
        _LLAMA_3_2_1B,
    ]
}

# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def list_profiles() -> list[str]:
    """Return sorted list of built-in profile names."""
    return sorted(REGISTRY.keys())


def get_profile(name_or_path: str) -> ModelProfile:
    """
    Return a ModelProfile by name or YAML path.

    Parameters
    ----------
    name_or_path
        Either a registry key (e.g. "gemma4-e2b") or a path to a YAML file.

    Raises
    ------
    KeyError
        If the name is not in the registry and the path does not exist.
    """
    if name_or_path in REGISTRY:
        return REGISTRY[name_or_path]

    path = Path(name_or_path)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        return _from_yaml(path)

    raise KeyError(
        f"Unknown model {name_or_path!r}. "
        f"Built-ins: {list_profiles()}. "
        f"Or pass a path to a .yaml profile."
    )


def _from_yaml(path: Path) -> ModelProfile:
    with path.open() as f:
        data = yaml.safe_load(f)
    return ModelProfile(**data)


__all__ = ["ModelProfile", "REGISTRY", "get_profile", "list_profiles"]
