"""Tests for device profile loader (config.get_device / list_devices)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dhurandhar.config import (
    DEVICE_PROFILES,
    DeploymentProfile,
    get_device,
    list_devices,
)
from dhurandhar.models import GEMMA4_E2B
from dhurandhar.ple_analysis import PLEFootprintAnalyzer


def test_list_devices_returns_sorted_builtins() -> None:
    devices = list_devices()
    assert devices == sorted(devices)
    assert set(devices) == set(DEVICE_PROFILES.keys())
    # Sanity-check a couple of known built-ins
    assert "high_end_mobile_ufs4" in devices
    assert "low_tier_mobile_emmc" in devices


def test_get_device_builtin_slug() -> None:
    d = get_device("high_end_mobile_ufs4")
    assert isinstance(d, DeploymentProfile)
    assert d.ram_budget_mb == 2048
    assert d.flash_read_gbps == pytest.approx(4.2)
    assert d.supports_npu is True


def test_get_device_unknown_slug_raises() -> None:
    with pytest.raises(KeyError, match="Unknown device"):
        get_device("not_a_real_device")


def test_get_device_loads_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "custom_phone.yaml"
    yaml_path.write_text(
        """
name: My Custom Phone (UFS 4.0)
ram_budget_mb: 4096
flash_read_gbps: 4.5
supports_npu: true
notes: synthetic test profile
"""
    )
    d = get_device(str(yaml_path))
    assert isinstance(d, DeploymentProfile)
    assert d.name == "My Custom Phone (UFS 4.0)"
    assert d.ram_budget_mb == 4096
    assert d.flash_read_gbps == pytest.approx(4.5)
    assert d.notes == "synthetic test profile"


def test_get_device_yaml_uses_defaults_when_omitted(tmp_path: Path) -> None:
    """Optional fields (supports_npu, notes) default when absent from YAML."""
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text(
        """
name: Minimal Device
ram_budget_mb: 1024
flash_read_gbps: 0.4
"""
    )
    d = get_device(str(yaml_path))
    assert d.supports_npu is False
    assert d.notes == ""


def test_repo_sample_pixel_yaml_loads() -> None:
    """The shipped configs/pixel_10_pro.yaml must round-trip through the loader."""
    repo_root = Path(__file__).resolve().parents[1]
    sample = repo_root / "configs" / "pixel_10_pro.yaml"
    assert sample.exists(), f"Expected sample YAML at {sample}"

    d = get_device(str(sample))
    assert "Pixel 10 Pro" in d.name
    assert d.ram_budget_mb > 0
    assert d.flash_read_gbps > 0


def test_assess_device_accepts_yaml_path(tmp_path: Path) -> None:
    """End-to-end: assess_device should resolve a YAML path through get_device."""
    yaml_path = tmp_path / "phone.yaml"
    yaml_path.write_text(
        """
name: Phone From YAML
ram_budget_mb: 2048
flash_read_gbps: 4.2
supports_npu: true
"""
    )
    analyzer = PLEFootprintAnalyzer(GEMMA4_E2B)
    f = analyzer.assess_device(str(yaml_path))
    assert f.device.name == "Phone From YAML"
    assert f.mode in {"resident", "mmap", "infeasible"}
