# dhurandhar — धुरंधर

> *dhura* (धुर, burden) + *dhara* (धर, one who bears)
>
> **"Bearer of burdens"** — a framework for deploying large multimodal models
> on memory-constrained edge devices where they have no right to survive.

[![PyPI version](https://badge.fury.io/py/dhurandhar.svg)](https://pypi.org/project/dhurandhar/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## What it does

`dhurandhar` is a **model-agnostic edge deployment analysis framework**.
Given a model architecture and a target device, it answers the questions
that matter before you ship:

| Module | Question answered |
|---|---|
| **PLE Analysis** | What is the peak live memory footprint at context length N? |
| **Device Feasibility** | Can this model run resident, mmap, or not at all on this device? |
| **TurboQuant Sweep** | What is the quality / memory tradeoff at 2/3/4/6/8-bit KV compression? |
| **RotorQuant Comparison** | TurboQuant vs RotorQuant — quality vs arithmetic cost? |
| **Mmap Profiler** | What is the real mmap throughput and peak RSS on this host? |

All five analyses are exposed as a **CLI**, a **Python API**, and a
**5-tab Gradio dashboard**.

---

## Supported models (built-in)

| Slug | Architecture | Params |
|---|---|---|
| `gemma4-e2b` | Gemma4 (sliding-window + global attention) | 2B |
| `qwen2.5-0.5b` | Qwen2.5 GQA | 0.5B |
| `qwen2.5-1.5b` | Qwen2.5 GQA | 1.5B |
| `granite-3.3-2b` | IBM Granite MHA | 2B |
| `llama-3.2-1b` | Llama3 GQA | 1B |

Any model can be added via a simple YAML profile — no code required.

---

## Install

```bash
# Core (analysis + CLI)
uv add dhurandhar

# With interactive dashboard
uv add "dhurandhar[dashboard]"

# With HuggingFace Hub auto-profile derivation
uv add "dhurandhar[hf]"

# Everything
uv add "dhurandhar[all]"
```

---

## Quickstart

```bash
# PLE memory breakdown for Gemma4 E2B at 4096 context
dhurandhar-ple-analyze --model gemma4-e2b --context 4096

# Device feasibility across all registered devices
dhurandhar-device-check --model gemma4-e2b

# TurboQuant quality sweep
dhurandhar-turbo-sweep --model gemma4-e2b --residual-bits 2,3,4,6,8

# Codec comparison
dhurandhar-compare-codecs --model gemma4-e2b --head-dim 256 --seq-len 2048

# Launch 5-tab dashboard
dhurandhar-dashboard
```

### Python API

```python
from dhurandhar.models import get_profile
from dhurandhar.devices import get_device

model  = get_profile("gemma4-e2b")
device = get_device("pixel-8")

print(f"KV cache at 4096 ctx: {model.kv_cache_bytes(4096) / 1e6:.1f} MB")
print(f"Device available RAM: {device.available_ram_gb:.1f} GB")
```

### Custom model via YAML

```yaml
# my_model.yaml
name: my-custom-2b
param_count_b: 2.0
weight_bytes: 4000000000
num_layers: 32
num_attention_layers: 32
num_kv_heads: 8
head_dim: 128
architecture_family: llama
```

```bash
dhurandhar-ple-analyze --model my_model.yaml --context 2048
```

---

## Status

`v0.1.0` — model/device registries stable, analysis modules and dashboard
landing in `v0.1.x` point releases.

---

## License

Apache 2.0
