# Dhurandhar — धुरंधर

> *dhura* (धुर, burden) + *dhara* (धर, one who bears)
>
> **"Bearer of burdens"** — a model-agnostic framework for deploying large
> multimodal models on memory-constrained edge devices where they have no
> right to survive.

[![PyPI version](https://shields.io/pypi/v/dhurandhar)](https://pypi.org/project/dhurandhar/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## Why this exists

Gemma 4 E2B's "< 1.5 GB RAM" deployment story depends on **memory-mapping
the Per-Layer Embedding (PLE) table from flash**. On the LiteRT-LM E2B
checkpoint, PLE is **1.12 GB — actually larger than the 0.79 GB text
decoder**. Whether mmap'd PLE sustains acceptable decode throughput on
target silicon is the single highest-risk item in any edge deployment plan.

Most teams find this out too late — after committing to a model, a
compression strategy, and a deployment architecture. `dhurandhar` moves
that conversation to the front of the process.

This toolkit lets you:

1. **Predict** memory feasibility per device profile before hardware arrives
2. **Measure** TurboQuant KV cache compression quality against hybrid-attention
   architectures (shared KV + GQA + sliding window)
3. **Fine-tune** LoRA adapters on frozen-PLE base models via QLoRA

---

## What it does

Given a model architecture and a target device, `dhurandhar` answers the
questions that matter *before* you ship:

| Module | Question answered |
|---|---|
| **PLE Analysis** | What is the true peak memory footprint at context length N? |
| **Device Feasibility** | Will this model run resident, mmap'd, or not at all on this device? |
| **TurboQuant Sweep** | What is the quality/memory tradeoff at 2/3/4/6/8-bit KV compression? |
| **RotorQuant Comparison** | TurboQuant vs RotorQuant — quality vs arithmetic cost? |
| **Mmap Profiler** | What is the real mmap throughput and peak RSS on this hardware? |

All five analyses are exposed as a **CLI**, a **Python API**, and a
**5-tab Gradio dashboard**.

---

## Supported models (built-in)

| Slug | Family | Params | PLE | Hybrid Attn |
|---|---|---|---|---|
| `gemma4-e2b` | Gemma 4 | 5.1B (2.3B active) | ✅ | ✅ local/global |
| `gemma4-e4b` | Gemma 4 | 9B | ✅ | ✅ local/global |
| `qwen2.5-0.5b` | Qwen 2.5 | 0.5B | ❌ | ❌ |
| `qwen2.5-1.5b` | Qwen 2.5 | 1.5B | ❌ | ❌ |
| `qwen2.5-3b` | Qwen 2.5 | 3B | ❌ | ❌ |
| `granite-3.3-2b` | IBM Granite | 2B | ❌ | ❌ |
| `llama-3.2-1b` | Llama 3.2 | 1B | ❌ | ❌ |
| `llama-3.2-3b` | Llama 3.2 | 3B | ❌ | ❌ |

Any model can be added via a [YAML profile](#bring-your-own-model) — no
code required.

---

## Installation

```bash
# Core (analysis + CLI)
uv add dhurandhar

# With interactive dashboard
uv add "dhurandhar[dashboard]"

# With GPU support (flash-attn, Linux only)
uv add "dhurandhar[gpu]"

# Everything
uv add "dhurandhar[dashboard,gpu]"
```

For local development:

```bash
git clone https://github.com/smarthi/dhurandhar.git
cd dhurandhar
uv sync --extra dev
uv run pytest tests/ -v
```

---

## Quick start

### 1. PLE memory footprint + device feasibility

```bash
dhurandhar-analyze-ple --model gemma4-e2b --context-tokens 32768 --quant-bits 4
```

```
Component                  Size      Notes
-------------------------  --------  -------------------------
Text decoder weights       809 MB    Q4
PLE embedding table        1,147 MB  Q4
KV cache @ 32,768 tokens   138 MB    shared + GQA + TurboQuant
Vision encoder             150 MB    bf16
Audio encoder (STRIPPED)   0 MB      bf16
Activations (peak)         64 MB
Runtime overhead           128 MB
-------------------------  --------
Total (PLE resident):      2,436 MB
Total (PLE mmap'd):        1,321 MB
PLE/Decoder ratio:         1.42x

[low_tier_mobile_emmc] Low-tier Mobile (eMMC 5.1)
  RAM budget:    1024 MB
  Flash bw:      0.40 GB/s
  Mode:          infeasible
  Notes:         Insufficient RAM even with mmap. Short by 297 MB.

[high_end_mobile_ufs4] High-end Mobile (UFS 4.0)
  RAM budget:    2048 MB
  Flash bw:      4.20 GB/s
  Mode:          mmap
  Notes:         PLE must be mmap'd (727 MB headroom). Flash bound = 185.2
                 tok/s (target 15.0). Viable but measure on device.

[laptop_nvme] Laptop (NVMe PCIe 4.0)
  RAM budget:    8192 MB
  Flash bw:      7.00 GB/s
  Mode:          resident
  Notes:         Fits resident with 5756 MB headroom. mmap not required.
```

Add `--json-out report.json` to emit a machine-readable report for
architecture review decks.

### 2. TurboQuant KV cache compression benchmark

```bash
dhurandhar-benchmark-kv --model gemma4-e2b --seq-len 32768 --residual-bits 4
```

```
Quality (synthetic KV reconstruction):
  MSE:                 0.003053
  Cosine similarity:   0.9972
  Norm preservation:   1.0029
  Effective bits/chan: 3.50
  Compression ratio:   4.57x vs bf16

KV cache footprint @ 32,768 tokens:
  Baseline (bf16):       3,072.0 MB
  TurboQuant:              672.0 MB
  Savings:               2,400.0 MB (4.57x reduction)
  Fresh-KV layers:     24
  Shared-KV layers:    6 (skipped)
```

The compressor correctly skips the 6 shared-KV layers — the tail of the
decoder that reuses earlier layers' KV tensors. Compressing those
independently would corrupt the attention pattern.

### 3. Real mmap decode throughput

```bash
# Quick run — creates a small test file, takes ~15s
dhurandhar-profile-mmap --scale 0.1 --num-tokens 1000 --target-tps 15

# Full-fidelity — creates a ~1 GB test file, realistic cold-mmap numbers
dhurandhar-profile-mmap --scale 1.0 --num-tokens 5000 --measure-memory
```

```
Pattern              Mode    tok/s      MB/s    p50 µs    p99 µs
sequential_prefill   cold    183,466    672     4.2       12.0
random_decode        cold    120,422    441     7.5       20.2
random_scatter       cold    127,155    466     7.1       16.2

Mmap gate verdict (target 15 tok/s)
  PASS
  Cold mmap decode throughput 120,422.2 tok/s ≥ target 15.0 tok/s.
```

**Important:** these numbers reflect the *host* filesystem, not target-device
flash. The profiler is designed to run on target silicon during the
feasibility phase — the methodology and code port directly, only the
measurement environment changes.

### 4. Codec comparison: TurboQuant vs RotorQuant

```bash
dhurandhar-compare-codecs --head-dim 255 --seq-len 2048 \
                           --residual-bits 2,3,4,6,8

# JSON output for archival
dhurandhar-compare-codecs --head-dim 255 --json-out codec_comparison.json
```

### 5. LoRA fine-tuning

```bash
# Dry run: build model + trainer, report param counts, don't train
dhurandhar-train-lora --config configs/gemma4_lora.yaml --dry-run

# Real training (requires GPU + HF_TOKEN)
HF_TOKEN=hf_... dhurandhar-train-lora --config configs/gemma4_lora.yaml
```

Default config: QLoRA (4-bit NF4 base) + r=16 LoRA on Q/K/V/O + SwiGLU
MLP. Expect ~2.5% trainable parameters on E2B. Fits on a single
L4 / A10G / RTX 4090-class GPU.

### 6. Interactive dashboard (5 tabs)

```bash
uv sync --extra dashboard
dhurandhar-dashboard
dhurandhar-dashboard --server-name 0.0.0.0 --port 7860   # LAN access
```

Five tabs:

1. **📊 PLE Memory Analysis** — model selector + interactive sliders for
   context, quantization, audio-strip; live component breakdown + stacked
   bar chart with 1.5 GB target line
2. **📱 Device Feasibility** — all built-in device profiles + custom device
   row; color-coded resident 🟢 / mmap 🟡 / infeasible 🔴 verdicts
3. **🗜️ TurboQuant KV** — compression quality sweep across residual bits;
   live quality-vs-bits chart and memory savings estimate
4. **⚡ Mmap Profiler** — real mmap throughput + peak RSS probe against
   configurable memory budgets
5. **🔄 TurboQuant vs RotorQuant** — side-by-side codec comparison:
   reconstruction quality sweep + stage-1 arithmetic cost bar chart

The dashboard is the recommended artifact for architecture review sessions —
interactive tradeoff exploration beats static slides for deployment decisions.

---

## Bring your own model

Any model not in the built-in registry can be added via YAML:

```yaml
# my_model.yaml
name: phi-3-mini-4k
family: phi
param_count_b: 3.8
num_hidden_layers: 32
num_attention_layers: 32
hidden_size: 3072
intermediate_size: 8192
vocab_size: 32064
num_attention_heads: 32
num_key_value_heads: 32
head_dim: 96
weight_dtype_bits: 16
kv_dtype_bits: 16
max_context_tokens: 4096
runtime_overhead_mb: 96.0
```

```bash
dhurandhar-analyze-ple --model my_model.yaml --context-tokens 4096
```

Or derive directly from a HuggingFace checkpoint to verify constants:

```python
from transformers import AutoConfig
from dhurandhar.models import ModelArchitecture

cfg = AutoConfig.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print(cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads, cfg.head_dim)
```

## Bring your own device

```python
from dhurandhar.config import DEVICE_PROFILES, DeploymentProfile

DEVICE_PROFILES["snapdragon_8_gen4"] = DeploymentProfile(
    name="Snapdragon 8 Gen 4 (UFS 4.0)",
    ram_budget_mb=3072,
    flash_read_gbps=4.5,
    supports_npu=True,
    notes="Measured on <device> on <date>. Update with real RSS budget.",
)
```

---

## Reference device profiles

| Profile | RAM budget | Flash bw | NPU |
|---|---|---|---|
| `high_end_mobile_ufs4` | 2048 MB | 4.2 GB/s | yes |
| `mid_tier_mobile_ufs3` | 1536 MB | 2.1 GB/s | yes |
| `low_tier_mobile_emmc` | 1024 MB | 0.4 GB/s | no |
| `tablet_ufs3` | 3072 MB | 2.0 GB/s | no |
| `laptop_nvme` | 8192 MB | 7.0 GB/s | yes |

All numbers are representative estimates — update with measured values for
your specific SKUs.

---

## Architecture

```
src/dhurandhar/
├── models/
│   ├── _base.py       # ModelArchitecture Pydantic model — the central contract
│   └── __init__.py    # Built-in registry + get_model() / list_models()
├── config.py          # DeploymentProfile registry + QuantizationProfile
├── ple_analysis.py    # PLE memory math + per-device feasibility (analytical)
├── mmap_profiler.py   # Real mmap decode throughput + peak RSS probe (empirical)
├── turboquant.py      # TurboQuant codec (Hadamard + sign + residual)
├── rotorquant.py      # RotorQuant codec (blockwise 3D Clifford rotors + residual)
├── finetune.py        # QLoRA training pipeline + audio-encoder strip
├── dashboard.py       # Gradio 5-tab UI combining all of the above
└── cli.py             # Click-based CLI entry points
```

### `models/` — the central contract

`ModelArchitecture` is a frozen Pydantic model that every analysis module
consumes. It captures the full architectural geometry of a model: layer
counts, attention head configuration, GQA parameters, hybrid attention
ratios, sliding window sizes, shared KV tail, and PLE dimensions.
All analytical methods (`kv_cache_bytes()`, `decoder_params()`,
`ple_bytes_per_decode_token()`, etc.) live here as pure functions of
the architecture — nothing downstream hard-codes model-specific constants.

New models can be added in three lines of YAML or a single
`ModelArchitecture(...)` constructor call.

### `ple_analysis.py` — memory math

Where `mmap_profiler.py` *measures*, this module *predicts*. Computes
component sizes from a `ModelArchitecture` instance and cross-checks
against any published checkpoint sizes provided. Accounts for:

- GQA (KV heads ≪ query heads) in KV cache sizing
- Sliding-window local layers (KV capped at `sliding_window` tokens)
- Shared-KV layers (zero additional KV bytes — they reference earlier layers)
- PLE mmap working-set estimate (~64 MB page-cache steady state) vs full
  resident PLE footprint

For non-PLE models the mmap analysis degenerates to a conventional
weights-resident breakdown — the PLE mmap question simply doesn't arise.

### `mmap_profiler.py` — empirical mmap gate

Where `ple_analysis.py` *predicts* mmap feasibility, this module *measures*
it. Creates a PLE-shaped dense file on disk, memory-maps it, and runs three
access patterns under both cold and warm conditions:

- **`sequential_prefill`** — sequential stride, models the prefill phase
- **`random_decode`** — random token-ID lookup, models autoregressive decode
- **`random_scatter`** — scattered multi-token decode, worst-case fragmentation

`mmap.MADV_DONTNEED` between measurements evicts pages for realistic
cold-mmap semantics. Outputs decode tokens/sec, effective MB/s, and p50/p99
latencies, plus a PASS/WARN/FAIL verdict against a configurable target.

The module is cross-platform — identical Python code runs on Linux, macOS,
and Android Termux for preliminary target-silicon measurement. For production
deployment a C++ port is expected; the access patterns and measurement
methodology stay identical.

### `turboquant.py` — KV compression codec

Implements the two-stage online vector quantization from
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874):

1. **Stage 1 — Randomized Hadamard rotation.** Flattens heavy-tailed
   per-coordinate distributions so that sign-bit quantization becomes
   effective. Implemented via the Fast Walsh-Hadamard Transform (FWHT)
   with random sign flips — O(d log d), no matrix multiplication.

2. **Stage 2 — Sign + L2 norm + residual int-quant.** Stores 1 sign bit
   per coordinate plus the vector's L2 norm, then quantizes the
   reconstruction residual at higher precision. Net: ≈ 3.5 bits/channel,
   ~4.5× compression vs bf16, cosine similarity > 0.99 on realistic
   heavy-tail KV distributions.

`KVCacheCompressor` applies this per-layer while respecting three
model-specific constraints that `ModelArchitecture` encodes:

- **Shared KV layers** — skipped entirely. A shared layer's KV is a
  reference to an earlier layer's already-quantized bytes; re-compressing
  would corrupt the attention pattern.
- **GQA** — compression operates per KV-head, not per query-head.
- **Sliding window** — local-attention layers' KV caps at `sliding_window`
  tokens. Memory savings accounting reflects this.
- **p-RoPE on global layers** — quantization applied post-RoPE so
  positional encoding fidelity is preserved.

### `rotorquant.py` — Clifford-rotor alternative

Implements a RotorQuant variant with the same two-stage pipeline but a
different stage-1 rotation:

1. **Stage 1 — Blockwise 3D Clifford rotor sandwich products.** Splits
   `head_dim` into groups of 3, embeds each as a Cl(3,0) vector, applies
   a rotor R·v·R̃ per block. Each rotor is 4 floats (unit quaternion
   equivalent), so the rotation is sparse by construction — blocks are
   independent with no butterfly-network dependencies.

2. **Stage 2** — identical to TurboQuant: sign + L2 + residual quantization.

Trade-off vs TurboQuant on this reference implementation:

| Metric | TurboQuant | RotorQuant |
|---|---|---|
| Stage-1 cost @ d=128 | 896 FMAs (FWHT) | 630 FMAs |
| Stage-1 cost @ d=256 | 2048 FMAs | 1275 FMAs |
| Quality @ 4-bit residual (synth) | cos_sim 0.997 | cos_sim 0.971 |
| Kernel complexity | butterfly network | embarrassingly parallel |
| Inter-coord dependencies | yes | no (block-independent) |

The real RotorQuant case for edge is kernel simplicity and block
independence for NPU/SIMD deployment — not raw FMA count. The quality gap
narrows substantially at 6+ residual bits, and block independence makes
RotorQuant easier to pipeline on hardware without scatter-gather support.

Use `dhurandhar-compare-codecs` or Tab 5 of the dashboard to run
side-by-side benchmarks on your target geometry.

### `finetune.py` — QLoRA training pipeline

Follows Google's recommended Gemma 4 QLoRA recipe, generalized for any
HuggingFace-compatible model:

- Base model loaded in 4-bit NF4 with double quantization via bitsandbytes
- LoRA adapters on Q/K/V/O attention projections + SwiGLU MLP (`gate_proj`,
  `up_proj`, `down_proj`)
- PEFT adapters trained in bf16
- TRL `SFTTrainer` handles chat templating, packing, and evaluation
- Paged 8-bit AdamW optimizer to keep VRAM flat during training

PLE stays frozen by default. If downstream quality needs PLE adaptation
that is a separate, deliberate decision.

Audio encoder strip (`strip_audio_encoder=True`) removes the ~300 MB audio
encoder on load for models that include one. ASR is typically handled by
a standalone pipeline; keeping the audio encoder in-model during fine-tuning
adds memory cost with no benefit unless specifically evaluating audio
capability.

---

## Testing

```bash
uv run pytest                         # all tests, ~15s
uv run pytest tests/test_turboquant.py -v
uv run pytest tests/test_rotorquant.py -v
uv run pytest tests/test_ple_analysis.py -v
uv run pytest tests/test_mmap_profiler.py -v
uv run pytest tests/test_strip_audio.py -v
```

90 tests cover:

- **TurboQuant** (26 tests) — Hadamard orthogonality at d ∈ {2…256},
  pack/unpack round-trip, codec quality (cos_sim > 0.95, norm
  preservation), shared-KV skipping, quality monotone in residual bits
- **RotorQuant** (17 tests) — Clifford rotor unit norm, sandwich
  invertibility, blockwise rotation on head_dim ∈ {64, 128, 129, 255,
  256}, codec quality, FMA cost beats TurboQuant at typical dims
- **PLE analysis** (12 tests) — `PLE > decoder` invariant for Gemma4,
  device feasibility (eMMC infeasible, NVMe resident), KV scales with
  context but caps at sliding window, non-PLE models produce valid
  resident-only breakdowns
- **Mmap profiler** (27 tests) — dense file creation, all three patterns
  produce valid throughput, peak RSS measurement, weights-only vs
  full-process budget verdicts, cross-platform madvise fallback
- **Strip audio encoder** (8 tests) — removes nested/top-level audio
  modules, preserves vision encoder and decoder, idempotent on repeated
  calls, handles text-only models gracefully

---

## Status and caveats

**Validated:**
- All 90 unit tests pass (Python 3.11, torch 2.5+, numpy 2.x)
- CLI tools produce output consistent with published LiteRT-LM checkpoint
  sizes (decoder = 0.79 GB, embeddings = 1.12 GB)
- TurboQuant delivers 4.57× compression at 0.997 cos-sim on synthetic
  heavy-tail KV distributions
- RotorQuant stage-1 FMA cost is 30–38% lower than TurboQuant at
  head_dim ∈ {128, 256}

**Unvalidated against real checkpoints (requires GPU + HF access):**
- `finetune.py` has not been run against actual model weights. The QLoRA
  recipe follows the documented pattern but `target_modules` may need
  adjustment if a model's `named_modules()` differ from expectations.
  Use `--dry-run` first to confirm adapter attachment before committing
  to a training run.
- The `ModelArchitecture` defaults for each built-in profile are
  best-effort reads of public model cards. Verify against
  `AutoConfig.from_pretrained(model_id)` before trusting absolute numbers;
  the *relative* claims (e.g. PLE > decoder for Gemma4) are sound because
  they derive from published checkpoint sizes directly.
- Flash-bandwidth ceiling in `assess_device()` computes PLE-only mmap
  reads and does not model decoder-weight bandwidth. On devices where
  decoder weights are also mmap'd (rather than resident), the real ceiling
  is tighter. Acceptable for initial feasibility gating; refine when
  device-level profiling data is available.

---

## License

Apache 2.0
