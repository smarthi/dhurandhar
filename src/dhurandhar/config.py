"""dhurandhar configuration — device profiles and quantization settings.

Model architecture specs live in dhurandhar.models.
This module owns only deployment-side configuration that is model-independent:
target device profiles and quantization presets.

    from dhurandhar.config import DEVICE_PROFILES, DeploymentProfile
    from dhurandhar.models import get_model

    arch   = get_model("gemma4-e2b")
    device = DEVICE_PROFILES["high_end_mobile_ufs4"]
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DeploymentProfile(BaseModel):
    """Hardware envelope for a target edge device."""

    model_config = ConfigDict(frozen=True)

    name:            str
    ram_budget_mb:   float
    flash_read_gbps: float
    supports_npu:    bool  = False
    notes:           str   = ""


DEVICE_PROFILES: dict[str, DeploymentProfile] = {
    "high_end_mobile_ufs4": DeploymentProfile(
        name="High-end Mobile (UFS 4.0)",
        ram_budget_mb=2048,
        flash_read_gbps=4.2,
        supports_npu=True,
        notes="Flagship mobile SKU; PLE mmap latency-viable",
    ),
    "mid_tier_mobile_ufs3": DeploymentProfile(
        name="Mid-tier Mobile (UFS 3.1)",
        ram_budget_mb=1536,
        flash_read_gbps=2.1,
        supports_npu=True,
        notes="PLE mmap plausible; measure on device",
    ),
    "low_tier_mobile_emmc": DeploymentProfile(
        name="Low-tier Mobile (eMMC 5.1)",
        ram_budget_mb=1024,
        flash_read_gbps=0.4,
        supports_npu=False,
        notes="PLE mmap HIGH RISK — likely needs resident PLE",
    ),
    "tablet_ufs3": DeploymentProfile(
        name="Tablet (UFS 3.1)",
        ram_budget_mb=3072,
        flash_read_gbps=2.0,
        supports_npu=False,
        notes="More RAM headroom than mobile; mid-range flash",
    ),
    "laptop_nvme": DeploymentProfile(
        name="Laptop (NVMe PCIe 4.0)",
        ram_budget_mb=8192,
        flash_read_gbps=7.0,
        supports_npu=True,
        notes="Ample RAM; PLE resident feasible",
    ),
}


class QuantizationProfile(BaseModel):
    """Quantization settings for a deployment target."""

    weight_bits:       int   = 4
    kv_bits_effective: float = 3.5
    activation_bits:   int   = 16
    use_turboquant_kv: bool  = True
    label:             str   = "Q4 weights + TurboQuant KV"


DEFAULT_QUANT = QuantizationProfile()

MEMORY_BUDGET_PRESETS: dict[str, float] = {
    "int4_aggressive":  1536,
    "int8_deployment":  2048,
    "bf16_development": 4096,
}
