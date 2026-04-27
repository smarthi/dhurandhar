"""Base DeviceProfile dataclass."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DeviceProfile:
    """
    Hardware facts for a target deployment device.

    Parameters
    ----------
    name
        Canonical slug, e.g. "pixel-8", "moto-g84", "snapdragon-x-laptop".
    total_ram_gb
        Physical RAM in GB.
    os_overhead_gb
        Typical OS + runtime overhead leaving this much unavailable.
    has_gpu
        Whether a discrete / integrated GPU is available for inference.
    gpu_vram_gb
        GPU VRAM in GB.  Relevant only when has_gpu=True.
    mmap_capable
        Whether the OS supports mmap for weight streaming.
    compute_class
        Rough compute tier: "high" | "mid" | "low".
    notes
        Free-text provenance.
    """

    name:            str
    total_ram_gb:    float
    os_overhead_gb:  float
    has_gpu:         bool         = False
    gpu_vram_gb:     float | None = None
    mmap_capable:    bool         = True
    compute_class:   str          = "mid"
    notes:           str          = ""

    @property
    def available_ram_gb(self) -> float:
        return max(0.0, self.total_ram_gb - self.os_overhead_gb)

    @property
    def available_ram_bytes(self) -> int:
        return int(self.available_ram_gb * (1024 ** 3))

    def __repr__(self) -> str:
        return (
            f"DeviceProfile(name={self.name!r}, "
            f"available={self.available_ram_gb:.1f}GB, "
            f"gpu={self.has_gpu})"
        )
