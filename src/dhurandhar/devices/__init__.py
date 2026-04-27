"""
Device profile registry.

Usage
-----
    from dhurandhar.devices import get_device, list_devices

    device = get_device("pixel-8")
    device = get_device("path/to/my_device.yaml")
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dhurandhar.devices._base import DeviceProfile

# ------------------------------------------------------------------ #
# Reference device profiles                                            #
# ------------------------------------------------------------------ #

REGISTRY: dict[str, DeviceProfile] = {
    d.name: d for d in [
        DeviceProfile(
            name           = "pixel-8",
            total_ram_gb   = 8.0,
            os_overhead_gb = 3.5,
            compute_class  = "high",
            notes          = "Google Pixel 8 — Tensor G3, 8 GB LPDDR5.",
        ),
        DeviceProfile(
            name           = "moto-g84",
            total_ram_gb   = 12.0,
            os_overhead_gb = 4.0,
            compute_class  = "mid",
            notes          = "Motorola Moto G84 — Snapdragon 695.",
        ),
        DeviceProfile(
            name           = "snapdragon-x-laptop",
            total_ram_gb   = 16.0,
            os_overhead_gb = 4.5,
            has_gpu        = True,
            gpu_vram_gb    = 0.0,   # unified memory
            compute_class  = "high",
            notes          = "Snapdragon X Elite laptop — unified 16 GB.",
        ),
        DeviceProfile(
            name           = "lenovo-tab-p12",
            total_ram_gb   = 8.0,
            os_overhead_gb = 3.0,
            compute_class  = "mid",
            notes          = "Lenovo Tab P12 — Dimensity 7050.",
        ),
        DeviceProfile(
            name           = "m3-macbook",
            total_ram_gb   = 16.0,
            os_overhead_gb = 4.0,
            has_gpu        = True,
            gpu_vram_gb    = 0.0,   # unified memory
            compute_class  = "high",
            notes          = "Apple M3 MacBook — unified 16 GB. Reference dev machine.",
        ),
        DeviceProfile(
            name           = "raspberry-pi-5",
            total_ram_gb   = 8.0,
            os_overhead_gb = 1.5,
            compute_class  = "low",
            notes          = "Raspberry Pi 5 — 8 GB variant, mmap critical.",
        ),
    ]
}


def list_devices() -> list[str]:
    return sorted(REGISTRY.keys())


def get_device(name_or_path: str) -> DeviceProfile:
    if name_or_path in REGISTRY:
        return REGISTRY[name_or_path]
    path = Path(name_or_path)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        with path.open() as f:
            return DeviceProfile(**yaml.safe_load(f))
    raise KeyError(
        f"Unknown device {name_or_path!r}. "
        f"Built-ins: {list_devices()}. "
        f"Or pass a path to a .yaml profile."
    )


__all__ = ["DeviceProfile", "REGISTRY", "get_device", "list_devices"]
