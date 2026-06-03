"""Model registry — pre-built ModelArchitecture instances + registry API.

Usage
-----
    from dhurandhar.models import get_model, list_models

    arch = get_model("gemma4-e2b")          # built-in
    arch = get_model("path/to/model.yaml")  # custom YAML

All analysis modules accept any ModelArchitecture; the registry just
provides convenient named instances so you don't have to fill in all
fields by hand.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dhurandhar.models._base import ModelArchitecture

# ------------------------------------------------------------------ #
# Built-in profiles                                                    #
# ------------------------------------------------------------------ #

# --- Gemma 4 ---

GEMMA4_E2B = ModelArchitecture(
    name                    = "gemma4-e2b",
    family                  = "gemma",
    param_count_b           = 5.1,       # total; 2.3B active per token
    num_hidden_layers       = 30,
    num_attention_layers    = 30,
    hidden_size             = 2048,
    intermediate_size       = 8192,
    vocab_size              = 262_144,
    num_attention_heads     = 8,
    num_key_value_heads     = 4,
    head_dim                = 256,
    local_to_global_ratio   = 5,         # 5 local : 1 global
    sliding_window          = 512,
    shared_kv_last_n_layers = 6,
    has_ple                 = True,
    ple_hidden_size         = 256,
    ple_vocab_size          = 262_144,
    vision_encoder_mb       = 150.0,
    audio_encoder_mb        = 300.0,
    published_decoder_gb    = 0.79,      # LiteRT-LM checkpoint
    published_embeddings_gb = 1.12,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 128_000,
    runtime_overhead_mb     = 128.0,
)

GEMMA4_E4B = ModelArchitecture(
    name                    = "gemma4-e4b",
    family                  = "gemma",
    param_count_b           = 9.0,
    num_hidden_layers       = 32,
    num_attention_layers    = 32,
    hidden_size             = 2560,
    intermediate_size       = 10240,
    vocab_size              = 262_144,
    num_attention_heads     = 8,
    num_key_value_heads     = 4,
    head_dim                = 256,
    local_to_global_ratio   = 5,
    sliding_window          = 512,
    shared_kv_last_n_layers = 6,
    has_ple                 = True,
    ple_hidden_size         = 256,
    ple_vocab_size          = 262_144,
    vision_encoder_mb       = 150.0,
    audio_encoder_mb        = 300.0,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 128_000,
    runtime_overhead_mb     = 128.0,
)

# Gemma 4 12B — the standalone (non-PLE) variant announced 2026-06-03 at
# https://huggingface.co/google/gemma-4-12B. All values below verified
# against the published config.json (model_type=gemma4_unified, dtype=bf16,
# transformers_version=5.10.0.dev0). Notable architectural features:
#   * attention_k_eq_v=true       — K and V share storage (kv_unified)
#   * head_dim=256 on local, 512 on global (sliding vs full attention)
#   * num_kv_heads=8 on local, 1 on global — extreme MQA on the global path
#   * layer_types: explicit 5 sliding + 1 full pattern, repeated 8× = 48
#   * tie_word_embeddings=true    — embedding & output share weights
#   * final_logit_softcapping=30.0, p-RoPE on full-attention layers
#   * unified multimodal: vision (mm_embed_dim=3840, patch_size=16, 280 soft
#     tokens) and audio (audio_embed_dim=640) are integrated, not separate
#     encoders — vision_encoder_mb / audio_encoder_mb stay 0
GEMMA4_12B = ModelArchitecture(
    name                       = "gemma4-12b",
    family                     = "gemma",
    param_count_b              = 11.95,    # fully dense — active == total
    num_hidden_layers          = 48,
    num_attention_layers       = 48,
    hidden_size                = 3840,
    intermediate_size          = 15360,
    vocab_size                 = 262_144,
    num_attention_heads        = 16,
    num_key_value_heads        = 8,        # local-layer KV heads
    head_dim                   = 256,      # local-layer head_dim
    local_to_global_ratio      = 5,        # 5 sliding : 1 full, last layer full
    sliding_window             = 1024,
    global_head_dim            = 512,      # global layers use head_dim=512
    num_global_key_value_heads = 1,        # extreme MQA on global path
    kv_unified                 = True,     # attention_k_eq_v — halves KV cache
                                           # AND saves one KV proj per layer
    embeddings_tied            = True,     # tie_word_embeddings=true in config
    shared_kv_last_n_layers    = 0,        # num_kv_shared_layers=0 in config
    has_ple                    = False,    # this is the non-PLE variant
    vision_encoder_mb          = 0.0,      # unified — no separate encoder
    audio_encoder_mb           = 0.0,      # unified — no separate encoder
    weight_dtype_bits          = 16,
    kv_dtype_bits              = 16,
    max_context_tokens         = 131_072,  # max_position_embeddings=128K
    runtime_overhead_mb        = 160.0,
)

# --- Zyphra ZAYA1 (MoE + Mamba hybrid) ---

ZAYA1_8B = ModelArchitecture(
    name                    = "zaya1-8b",
    family                  = "zaya",
    param_count_b           = 8.4,       # total across all experts
    active_param_count_b    = 0.76,      # 760M active per token
    num_hidden_layers       = 32,
    num_attention_layers    = 16,        # half are Mamba layers, half attention
    hidden_size             = 4096,
    intermediate_size       = 14336,
    vocab_size              = 131_072,
    num_attention_heads     = 32,
    num_key_value_heads     = 8,
    head_dim                = 128,
    is_moe                  = True,
    num_experts             = 64,
    num_active_experts      = 8,
    has_mamba               = True,
    # Mamba state: d_state(128) * d_model(4096) + d_conv(4) * d_model(4096) per layer
    # = 524,288 + 16,384 = 540,672 floats per layer
    mamba_state_dim         = 540_672,
    mamba_num_layers        = 16,
    mamba_state_dtype_bits  = 32,        # float32 required for SSM recurrence
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 32_768,
    runtime_overhead_mb     = 128.0,
)

# --- Qwen 2.5 ---

QWEN25_0_5B = ModelArchitecture(
    name                    = "qwen2.5-0.5b",
    family                  = "qwen",
    param_count_b           = 0.5,
    num_hidden_layers       = 24,
    num_attention_layers    = 24,
    hidden_size             = 896,
    intermediate_size       = 4864,
    vocab_size              = 151_936,
    num_attention_heads     = 14,
    num_key_value_heads     = 2,
    head_dim                = 64,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 32_768,
    runtime_overhead_mb     = 64.0,
)

QWEN25_1_5B = ModelArchitecture(
    name                    = "qwen2.5-1.5b",
    family                  = "qwen",
    param_count_b           = 1.5,
    num_hidden_layers       = 28,
    num_attention_layers    = 28,
    hidden_size             = 1536,
    intermediate_size       = 8960,
    vocab_size              = 151_936,
    num_attention_heads     = 12,
    num_key_value_heads     = 2,
    head_dim                = 128,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 32_768,
    runtime_overhead_mb     = 64.0,
)

QWEN25_3B = ModelArchitecture(
    name                    = "qwen2.5-3b",
    family                  = "qwen",
    param_count_b           = 3.0,
    num_hidden_layers       = 36,
    num_attention_layers    = 36,
    hidden_size             = 2048,
    intermediate_size       = 11008,
    vocab_size              = 151_936,
    num_attention_heads     = 16,
    num_key_value_heads     = 8,
    head_dim                = 128,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 32_768,
    runtime_overhead_mb     = 96.0,
)

# --- IBM Granite ---

GRANITE_3_3_2B = ModelArchitecture(
    name                    = "granite-3.3-2b",
    family                  = "granite",
    param_count_b           = 2.0,
    num_hidden_layers       = 40,
    num_attention_layers    = 40,
    hidden_size             = 2048,
    intermediate_size       = 8192,
    vocab_size              = 49_152,
    num_attention_heads     = 32,
    num_key_value_heads     = 8,
    head_dim                = 64,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 128_000,
    runtime_overhead_mb     = 96.0,
)

# --- Llama 3.2 ---

LLAMA_3_2_1B = ModelArchitecture(
    name                    = "llama-3.2-1b",
    family                  = "llama",
    param_count_b           = 1.0,
    num_hidden_layers       = 16,
    num_attention_layers    = 16,
    hidden_size             = 2048,
    intermediate_size       = 8192,
    vocab_size              = 128_256,
    num_attention_heads     = 32,
    num_key_value_heads     = 8,
    head_dim                = 64,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 131_072,
    runtime_overhead_mb     = 64.0,
)

LLAMA_3_2_3B = ModelArchitecture(
    name                    = "llama-3.2-3b",
    family                  = "llama",
    param_count_b           = 3.0,
    num_hidden_layers       = 28,
    num_attention_layers    = 28,
    hidden_size             = 3072,
    intermediate_size       = 8192,
    vocab_size              = 128_256,
    num_attention_heads     = 24,
    num_key_value_heads     = 8,
    head_dim                = 128,
    weight_dtype_bits       = 16,
    kv_dtype_bits           = 16,
    max_context_tokens      = 131_072,
    runtime_overhead_mb     = 96.0,
)

# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

REGISTRY: dict[str, ModelArchitecture] = {
    m.name: m for m in [
        GEMMA4_E2B,
        GEMMA4_E4B,
        GEMMA4_12B,
        ZAYA1_8B,
        QWEN25_0_5B,
        QWEN25_1_5B,
        QWEN25_3B,
        GRANITE_3_3_2B,
        LLAMA_3_2_1B,
        LLAMA_3_2_3B,
    ]
}


def list_models() -> list[str]:
    """Return sorted list of built-in model names."""
    return sorted(REGISTRY.keys())


def get_model(name_or_path: str) -> ModelArchitecture:
    """Return a ModelArchitecture by registry name or YAML path.

    Parameters
    ----------
    name_or_path
        Either a registry key (e.g. "gemma4-e2b") or a path to a YAML file
        whose keys match the ModelArchitecture field names.

    Raises
    ------
    KeyError
        If the name is not in the registry and the path does not exist.

    Examples
    --------
    >>> arch = get_model("gemma4-e2b")
    >>> arch = get_model("my_models/phi3_mini.yaml")
    """
    if name_or_path in REGISTRY:
        return REGISTRY[name_or_path]
    path = Path(name_or_path)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        with path.open() as f:
            data = yaml.safe_load(f)
        return ModelArchitecture(**data)
    raise KeyError(
        f"Unknown model {name_or_path!r}. "
        f"Built-ins: {list_models()}. "
        f"Or pass a path to a YAML file with ModelArchitecture fields."
    )


__all__ = [
    "ModelArchitecture",
    "REGISTRY",
    "get_model",
    "list_models",
    # Named instances
    "GEMMA4_E2B",
    "GEMMA4_E4B",
    "GEMMA4_12B",
    "ZAYA1_8B",
    "QWEN25_0_5B",
    "QWEN25_1_5B",
    "QWEN25_3B",
    "GRANITE_3_3_2B",
    "LLAMA_3_2_1B",
    "LLAMA_3_2_3B",
]
