"""dhurandhar configuration — device profiles and quantization settings.

Model architecture specs live in dhurandhar.models.
This module owns only deployment-side configuration that is model-independent:
target device profiles and quantization presets.

    from dhurandhar.config import get_device, DEVICE_PROFILES, DeploymentProfile
    from dhurandhar.models import get_model

    arch   = get_model("gemma4-e2b")
    device = get_device("high_end_mobile_ufs4")          # built-in slug
    device = get_device("configs/my_phone.yaml")         # custom YAML
"""

from __future__ import annotations

from pathlib import Path

import yaml
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

QUANT_PRESETS: dict[str, QuantizationProfile] = {
    "q4_turboquant": DEFAULT_QUANT,
    "q2_extreme": QuantizationProfile(
        weight_bits=2,
        kv_bits_effective=2.0,
        activation_bits=16,
        use_turboquant_kv=False,
        label="Q2 extreme — severe quality loss, no optimized kernels",
    ),
    "q4_rotorquant": QuantizationProfile(
        weight_bits=4,
        kv_bits_effective=3.5,
        activation_bits=16,
        use_turboquant_kv=False,
        label="Q4 weights + RotorQuant KV",
    ),
}

MEMORY_BUDGET_PRESETS: dict[str, float] = {
    "int2_extreme":     1024,
    "int4_aggressive":  1536,
    "int8_deployment":  2048,
    "bf16_development": 4096,
}


# ------------------------------------------------------------------ #
# Device profile loader                                                #
# ------------------------------------------------------------------ #


def list_devices() -> list[str]:
    """Return sorted list of built-in device profile slugs."""
    return sorted(DEVICE_PROFILES.keys())


def get_device(name_or_path: str) -> DeploymentProfile:
    """Return a DeploymentProfile by registry slug or YAML path.

    Mirrors :func:`dhurandhar.models.get_model` — accepts either a built-in
    slug (e.g. ``"high_end_mobile_ufs4"``) or the path to a YAML file whose
    keys match the :class:`DeploymentProfile` fields.

    Parameters
    ----------
    name_or_path
        Either a key in :data:`DEVICE_PROFILES` or a path to a ``.yaml`` /
        ``.yml`` file.

    Raises
    ------
    KeyError
        If the slug is not in the registry and the path does not exist.

    Examples
    --------
    >>> dev = get_device("high_end_mobile_ufs4")
    >>> dev = get_device("configs/pixel_10_pro.yaml")
    """
    if name_or_path in DEVICE_PROFILES:
        return DEVICE_PROFILES[name_or_path]
    path = Path(name_or_path)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        with path.open() as f:
            data = yaml.safe_load(f)
        return DeploymentProfile(**data)
    raise KeyError(
        f"Unknown device {name_or_path!r}. "
        f"Built-ins: {list_devices()}. "
        f"Or pass a path to a YAML file with DeploymentProfile fields "
        f"(name, ram_budget_mb, flash_read_gbps, supports_npu, notes)."
    )
