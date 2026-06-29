"""
dhurandhar — धुरंधर

Sanskrit: dhura (धुर, burden) + dhara (धर, one who bears)
"Bearer of burdens" — a model-agnostic framework for deploying large
multimodal models on memory-constrained edge devices.

Public API
----------
    from dhurandhar.models import get_model, list_models
    from dhurandhar.config import get_device, list_devices, DeploymentProfile
    from dhurandhar.ple_analysis import PLEFootprintAnalyzer
    from dhurandhar.turboquant import TurboQuantCodec, KVCacheCompressor
    from dhurandhar.rotorquant import RotorQuantCodec
    from dhurandhar.mmap_profiler import MmapDecodeProfiler

Quickstart
----------
    arch     = get_model("gemma4-e2b")   # or "llama-3.2-1b", "qwen2.5-1.5b", ...
    analyzer = PLEFootprintAnalyzer(arch)
    bd       = analyzer.compute_breakdown(context_tokens=32768, quant_bits=4)
    print(analyzer.format_breakdown(bd))
"""

from .config import (
    DEFAULT_QUANT,
    DEVICE_PROFILES,
    QUANT_PRESETS,
    DeploymentProfile,
    QuantizationProfile,
    get_device,
    list_devices,
)
from .finetune import (
    FinetuneJobConfig,
    LoRAConfig,
    QuantConfig,
    TrainingConfig,
    strip_audio_encoder,
)
from .mmap_profiler import MEMORY_BUDGETS_MB, MemoryProbeResult, MmapDecodeProfiler, ProfileResult
from .models import REGISTRY as MODEL_REGISTRY
from .models import get_model, list_models
from .models._base import ModelArchitecture
from .oscarquant import OScaRCodec, OScaRConfig, OScaRQuantizedVector
from .oscarquant import fma_cost_comparison as oscar_fma_cost_comparison
from .ple_analysis import DeviceFeasibility, MemoryBreakdown, PLEFootprintAnalyzer
from .rotorquant import RotorQuantCodec, RotorQuantConfig, RotorQuantizedVector, fma_cost_comparison
from .spectralquant import SpectralQuantCodec, SpectralQuantConfig
from .spectralquant import fma_cost_comparison as spectral_fma_cost_comparison
from .turboquant import KVCacheCompressor, QuantizedVector, TurboQuantCodec, TurboQuantConfig

__version__ = "0.1.4"

__all__ = [
    # Models
    "ModelArchitecture",
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
    # Config
    "DEVICE_PROFILES",
    "DeploymentProfile",
    "get_device",
    "list_devices",
    "DEFAULT_QUANT",
    "QuantizationProfile",
    "QUANT_PRESETS",
    # Analysis
    "PLEFootprintAnalyzer",
    "MemoryBreakdown",
    "DeviceFeasibility",
    # Profiler
    "MmapDecodeProfiler",
    "ProfileResult",
    "MemoryProbeResult",
    "MEMORY_BUDGETS_MB",
    # Codecs
    "TurboQuantCodec",
    "TurboQuantConfig",
    "KVCacheCompressor",
    "QuantizedVector",
    "RotorQuantCodec",
    "RotorQuantConfig",
    "RotorQuantizedVector",
    "fma_cost_comparison",
    "SpectralQuantCodec",
    "SpectralQuantConfig",
    "spectral_fma_cost_comparison",
    "OScaRCodec",
    "OScaRConfig",
    "OScaRQuantizedVector",
    "oscar_fma_cost_comparison",
    # Fine-tuning
    "FinetuneJobConfig",
    "LoRAConfig",
    "TrainingConfig",
    "QuantConfig",
    "strip_audio_encoder",
]
