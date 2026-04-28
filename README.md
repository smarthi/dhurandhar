# dhurandhar — धुरंधर

> *dhura* (धुर, burden) + *dhara* (धर, one who bears)
>
> **"Bearer of burdens"** — a model-agnostic framework for deploying large multimodal models
> on memory-constrained edge devices where they have no right to survive.

[![PyPI version](https://badge.fury.io/py/dhurandhar.svg)](https://pypi.org/project/dhurandhar/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## What it does

Given a model architecture and a target device, `dhurandhar` answers the questions
that matter *before* you ship:

| Module | Question answered |
|---|---|
| **PLE Analysis** | What is the true peak memory footprint at context length N? |
| **Device Feasibility** | Will this model run resident, mmap'd, or not at all on this device? |
| **TurboQuant Sweep** | What is the quality/memory tradeoff at 2/3/4/6/8-bit KV compression? |
| **RotorQuant Comparison** | TurboQuant vs RotorQuant — quality vs arithmetic cost? |
| **Mmap Profiler** | What is the real mmap throughput and peak RSS on this hardware? |

All five analyses are exposed as a **CLI**, a **Python API**, and a **5-tab Gradio dashboard**.

---

## Why this exists

Gemma 4 E2B's "< 1.5 GB RAM" deployment story depends on **memory-mapping the
Per-Layer Embedding (PLE) table from flash**. On the LiteRT-LM E2B checkpoint,
PLE is **1.12 GB — larger than the 0.79 GB text decoder**. Whether mmap'd PLE
sustains acceptable decode throughput on your target silicon is the single
highest-risk item in any edge deployment plan.

`dhurandhar` lets you:

1. **Predict** memory feasibility per device profile before hardware arrives
2. **Measure** TurboQuant KV cache compression quality against Gemma 4's
   hybrid-attention architecture (shared KV + GQA + sliding window)
3. **Fine-tune** LoRA adapters on the frozen-PLE base model via QLoRA

---

## Installation

```bash
# Core (analysis + CLI)
uv add dhurandhar

# With interactive dashboard
uv add "dhurandhar[dashboard]"

# With GPU support (flash-attn, Linux only)
uv add "dhurandhar[gpu]"
```

---

## Quickstart

### PLE memory footprint + device feasibility

```bash
dhurandhar-analyze-ple --context-tokens 32768 --quant-bits 4
```

```
Component                  Size      Notes
-------------------------  --------  -------------------------
Text decoder weights       809 MB    Q4
PLE embedding table        1,147 MB  Q4
KV cache @ 32,768 tokens   138 MB    shared + GQA + TurboQuant
Vision encoder             150 MB    bf16
Audio encoder (STRIPPED)   0 MB      bf16
...
Total (PLE resident): 2,404 MB
Total (PLE mmap'd):  1,321 MB
PLE/Decoder ratio:   1.42x

[low_tier_mobile_emmc] Low-tier Mobile (eMMC 5.1)
  RAM budget:    1024 MB
  Mode:        infeasible
  Notes:       Insufficient RAM even with mmap. Short by 297 MB.

[laptop_nvme] Laptop (NVMe PCIe 4.0)
  RAM budget:    8192 MB
  Mode:        resident
  Notes:       PLE fits resident with 5788 MB headroom.
```

### TurboQuant KV cache compression

```bash
dhurandhar-benchmark-kv --seq-len 32768 --residual-bits 4
```

```
Quality (synthetic KV reconstruction):
  Cosine similarity:   0.9972
  Compression ratio:   4.57x vs bf16
  Fresh-KV layers:     24
  Shared-KV layers:    6 (skipped)
```

### Real mmap decode throughput

```bash
# Quick run — small test file, ~15s
dhurandhar-profile-mmap --scale 0.1 --num-tokens 1000 --target-tps 15

# Full-fidelity — ~1 GB test file, realistic cold-mmap numbers
dhurandhar-profile-mmap --scale 1.0 --num-tokens 5000 --measure-memory
```

### Codec comparison: TurboQuant vs RotorQuant

```bash
dhurandhar-compare-codecs --head-dim 255 --residual-bits 2,3,4,6,8
```

### LoRA fine-tuning

```bash
# Dry run — confirm adapter attachment without training
dhurandhar-train-lora --config configs/gemma4_lora.yaml --dry-run

# Real training (requires GPU + HF_TOKEN)
HF_TOKEN=hf_... dhurandhar-train-lora --config configs/gemma4_lora.yaml
```

### Interactive dashboard (5 tabs)

```bash
uv sync --extra dashboard
dhurandhar-dashboard
dhurandhar-dashboard --server-name 0.0.0.0 --port 7860  # LAN access
```

Five tabs:

1. **📊 PLE Memory Analysis** — component breakdown + stacked bar chart vs 1.5 GB target
2. **📱 Device Feasibility** — resident 🟢 / mmap 🟡 / infeasible 🔴 verdicts + custom device
3. **🗜️ TurboQuant KV** — quality sweep across residual bits + memory savings estimate
4. **⚡ Mmap Profiler** — real mmap throughput + peak RSS vs deployment budget
5. **🔄 TurboQuant vs RotorQuant** — quality sweep + stage-1 arithmetic cost comparison

---

## Custom model architecture

Override architectural constants in `config.py` or verify against a live checkpoint:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B")
print(cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads)
```

## Custom device profile

Pass your own device spec directly or as a YAML file:

```python
from dhurandhar.config import DeploymentProfile, DEVICE_PROFILES
DEVICE_PROFILES["my_device"] = DeploymentProfile(
    name="My Target Device",
    ram_budget_mb=2048,
    flash_read_gbps=3.5,
    supports_npu=True,
)
```

---

## Project structure

```
src/dhurandhar/
├── config.py          # Gemma 4 E2B constants + device profiles
├── ple_analysis.py    # PLE memory math + device feasibility (analytical)
├── mmap_profiler.py   # Real mmap throughput + peak RSS probe (empirical)
├── turboquant.py      # TurboQuant codec (Hadamard + sign + residual)
├── rotorquant.py      # RotorQuant codec (blockwise 3D Clifford rotors)
├── finetune.py        # QLoRA training pipeline + audio-encoder strip
├── dashboard.py       # Gradio 5-tab dashboard
└── cli.py             # Click-based CLI entry points
```

---

## Testing

```bash
uv run pytest                    # all tests, ~15s
uv run pytest tests/test_turboquant.py -v
uv run pytest tests/test_rotorquant.py -v
uv run pytest tests/test_ple_analysis.py -v
uv run pytest tests/test_mmap_profiler.py -v
uv run pytest tests/test_strip_audio.py -v
```

---

## License

Apache 2.0
