"""Generic ModelArchitecture — the single contract all analysis modules consume."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelArchitecture(BaseModel):
    """Portable architecture specification for edge deployment analysis.

    Parameters
    ----------
    name
        Registry slug. e.g. "gemma4-e2b", "llama-3.2-1b".
    family
        Architecture family. e.g. "gemma", "qwen", "granite", "llama".
    param_count_b
        Total parameter count in billions.
    num_hidden_layers
        Total transformer layers (including SSM/DeltaNet layers without KV cache).
    num_attention_layers
        Layers that produce a KV cache. Equal to num_hidden_layers for standard
        transformers; less for hybrid SSM models (e.g. Qwen3-series).
    hidden_size
        Residual stream dimension.
    intermediate_size
        FFN intermediate dimension (gate/up width for SwiGLU/GeGLU).
    vocab_size
        Vocabulary size.
    num_attention_heads
        Query head count per attention layer.
    num_key_value_heads
        KV head count per attention layer (GQA). Equal to num_attention_heads for MHA.
    head_dim
        Per-head key/value dimension.
    local_to_global_ratio
        Hybrid local/global attention: every (ratio+1)-th layer is global.
        0 = all global (standard).
    sliding_window
        Token window for local-attention layers. 0 if no sliding window.
    shared_kv_last_n_layers
        Last N layers reuse KV from earlier layers (Gemma4). 0 = all fresh KV.
    has_ple
        Whether the model uses Per-Layer Embeddings (Gemma4-specific).
    ple_hidden_size
        PLE vector dimension per layer (Gemma4: 256). 0 if has_ple=False.
    ple_vocab_size
        PLE table vocabulary size. 0 if has_ple=False.
    vision_encoder_mb
        Vision encoder footprint in MB. 0 if absent.
    audio_encoder_mb
        Audio encoder footprint in MB. 0 if absent.
    published_decoder_gb
        Optional published decoder checkpoint size for cross-checking.
    published_embeddings_gb
        Optional published embeddings checkpoint size for cross-checking.
    weight_dtype_bits
        Native weight dtype bit width (16 = bf16/fp16).
    kv_dtype_bits
        KV cache accumulation dtype bit width.
    max_context_tokens
        Maximum supported context length.
    runtime_overhead_mb
        Estimated runtime framework overhead (tokenizer, runtime, misc).
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name:                    str
    family:                  str

    # Scale
    param_count_b:           float

    # Transformer geometry
    num_hidden_layers:       int
    num_attention_layers:    int
    hidden_size:             int
    intermediate_size:       int
    vocab_size:              int

    # Attention
    num_attention_heads:     int
    num_key_value_heads:     int
    head_dim:                int

    # Hybrid attention
    local_to_global_ratio:   int   = 0
    sliding_window:          int   = 0

    # Shared KV
    shared_kv_last_n_layers: int   = 0

    # Per-Layer Embeddings
    has_ple:                 bool  = False
    ple_hidden_size:         int   = 0
    ple_vocab_size:          int   = 0

    # Encoders
    vision_encoder_mb:       float = 0.0
    audio_encoder_mb:        float = 0.0

    # Published checkpoint sizes
    published_decoder_gb:    float = 0.0
    published_embeddings_gb: float = 0.0

    # Runtime
    weight_dtype_bits:       int   = 16
    kv_dtype_bits:           int   = 16
    max_context_tokens:      int   = 128_000
    runtime_overhead_mb:     float = 128.0

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    def global_layer_indices(self) -> list[int]:
        if self.local_to_global_ratio == 0:
            return list(range(self.num_hidden_layers))
        globals_: set[int] = set()
        stride = self.local_to_global_ratio + 1
        for idx in range(self.num_hidden_layers):
            if (idx + 1) % stride == 0 or idx == self.num_hidden_layers - 1:
                globals_.add(idx)
        return sorted(globals_)

    def local_layer_indices(self) -> list[int]:
        g = set(self.global_layer_indices())
        return [i for i in range(self.num_hidden_layers) if i not in g]

    def shared_kv_layer_indices(self) -> list[int]:
        n = self.shared_kv_last_n_layers
        if n == 0:
            return []
        return list(range(self.num_hidden_layers - n, self.num_hidden_layers))

    def fresh_kv_layer_indices(self) -> list[int]:
        shared = set(self.shared_kv_layer_indices())
        return [i for i in range(self.num_hidden_layers) if i not in shared]

    def kv_cache_bytes(self, context_tokens: int, kv_bits: int) -> int:
        kv_element_bytes = kv_bits / 8.0
        per_token_per_layer = self.num_key_value_heads * self.head_dim * 2
        total = 0
        global_set = set(self.global_layer_indices())
        for layer_idx in self.fresh_kv_layer_indices():
            effective = (
                context_tokens
                if layer_idx in global_set or self.sliding_window == 0
                else min(context_tokens, self.sliding_window)
            )
            total += int(effective * per_token_per_layer * kv_element_bytes)
        return total

    def decoder_params(self) -> int:
        per_q    = self.hidden_size * self.num_attention_heads * self.head_dim
        per_kv   = self.hidden_size * self.num_key_value_heads * self.head_dim
        per_o    = self.num_attention_heads * self.head_dim * self.hidden_size
        per_ffn  = 3 * self.hidden_size * self.intermediate_size
        per_norm = 4 * self.hidden_size
        fresh    = set(self.fresh_kv_layer_indices())
        n_fresh  = sum(1 for i in range(self.num_hidden_layers) if i in fresh)
        n_shared = self.num_hidden_layers - n_fresh
        return (
            n_fresh  * (per_q + 2 * per_kv + per_o + per_ffn + per_norm)
            + n_shared * (per_q           + per_o + per_ffn + per_norm)
            + self.hidden_size
        )

    def embedding_params(self) -> int:
        token_emb = self.vocab_size * self.hidden_size
        if not self.has_ple:
            return token_emb
        ple_table = self.ple_vocab_size * self.num_hidden_layers * self.ple_hidden_size
        ple_proj  = self.num_hidden_layers * self.ple_hidden_size * self.hidden_size
        return token_emb + ple_table + ple_proj

    def ple_table_params(self) -> int:
        if not self.has_ple:
            return 0
        return self.ple_vocab_size * self.num_hidden_layers * self.ple_hidden_size

    def ple_bytes(self, quant_bits: int) -> float:
        return self.ple_table_params() * (quant_bits / 8.0)

    def ple_bytes_per_decode_token(self, quant_bits: int) -> float:
        if not self.has_ple:
            return 0.0
        return self.num_hidden_layers * self.ple_hidden_size * (quant_bits / 8.0)

    @property
    def is_hybrid_attention(self) -> bool:
        return self.local_to_global_ratio > 0

    @property
    def has_shared_kv(self) -> bool:
        return self.shared_kv_last_n_layers > 0

    @property
    def kv_compression_eligible_layers(self) -> int:
        return len(self.fresh_kv_layer_indices())

    def __repr__(self) -> str:
        return (
            f"ModelArchitecture(name={self.name!r}, family={self.family!r}, "
            f"params={self.param_count_b}B, layers={self.num_hidden_layers}, "
            f"attn={self.num_attention_layers}, kv_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, ple={self.has_ple})"
        )
