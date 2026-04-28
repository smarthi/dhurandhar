# Decision Log: Adopt Gemma 4 E2B as Reference Model for Edge Deployment

| | |
|---|---|
| **Status** | Reference document |
| **Date** | April 2026 |
| **Decision class** | Architectural — on-device foundation model selection |

---

## Summary

This document captures the rationale for using **Gemma 4 E2B** as the reference
model for `dhurandhar`'s analytical framework. The framework is model-agnostic —
any model can be profiled by providing an architecture config — but Gemma 4 E2B
is the primary reference implementation because of its novel PLE architecture,
which makes the mmap-vs-resident tradeoff the central deployment decision.

---

## Why Gemma 4 E2B as the reference model

**PLE makes the deployment tradeoff interesting.** On the LiteRT-LM E2B checkpoint,
the PLE embedding table is 1.12 GB — *larger than the 0.79 GB text decoder*. This
means the "< 1.5 GB RAM" claim depends entirely on whether PLE can be safely
memory-mapped from flash at acceptable throughput. No other current tiny LLM has
this property, making it the most analytically interesting edge deployment case.

**Architecture covers all the important tradeoffs:**
- Shared KV Cache (architectural, not optional) — affects compression scope
- Hybrid local/global attention with sliding window — affects KV cache sizing
- GQA (4 KV heads vs 8 query heads) — affects TurboQuant yield
- p-RoPE on global layers — affects post-RoPE quantization requirement
- Native multimodal with optional audio strip — affects total footprint

**Apache 2.0 license.** No restrictions on use, modification, or distribution.

---

## Acceptance gates (reference targets)

The mmap profiler implements three reference acceptance gates:

- **`int4_aggressive`** — ≤ 1.5 GB peak RSS (low-tier mobile target)
- **`int8_deployment`** — ≤ 2 GB peak RSS (primary on-device target)
- **`bf16_development`** — ≤ 4 GB peak RSS (dev/eval workstation)

These are reference values. Adjust for your deployment envelope.

---

## Using dhurandhar with other models

Set `Gemma4E2BArchitecture` defaults to match your model's config, or derive
directly from a HuggingFace checkpoint:

```python
from transformers import AutoConfig
from dhurandhar.config import Gemma4E2BArchitecture

cfg = AutoConfig.from_pretrained("your-model-id")
arch = Gemma4E2BArchitecture(
    num_hidden_layers=cfg.num_hidden_layers,
    hidden_size=cfg.hidden_size,
    num_key_value_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim,
    # ... etc
)
```

A first-class `ModelProfile` abstraction with registry and YAML loading
is planned for a future release.
