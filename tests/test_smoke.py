"""Smoke tests — just enough to confirm the package installs and imports correctly."""

import pytest

from dhurandhar import DeviceProfile, ModelProfile, __version__
from dhurandhar.devices import get_device, list_devices
from dhurandhar.models import get_profile, list_profiles


def test_version():
    assert __version__ == "0.1.0"


def test_model_registry_not_empty():
    profiles = list_profiles()
    assert len(profiles) >= 5
    assert "gemma4-e2b" in profiles


def test_device_registry_not_empty():
    devices = list_devices()
    assert len(devices) >= 4


def test_gemma4_profile():
    p = get_profile("gemma4-e2b")
    assert isinstance(p, ModelProfile)
    assert p.param_count_b == 2.0
    assert p.local_attn_window == 512
    assert p.global_attn_freq == 6
    assert p.num_attention_layers == 26
    assert p.weight_gb > 0


def test_kv_cache_bytes_gemma4():
    p = get_profile("gemma4-e2b")
    kv = p.kv_cache_bytes(context_len=2048)
    assert kv > 0
    # Local attn window means KV should be less than full-context equivalent
    kv_full = ModelProfile(
        name="test", param_count_b=2.0, weight_bytes=int(2e9*2),
        num_layers=26, num_attention_layers=26,
        num_kv_heads=4, head_dim=256,
    ).kv_cache_bytes(context_len=2048)
    assert kv < kv_full


def test_device_profile():
    d = get_device("pixel-8")
    assert isinstance(d, DeviceProfile)
    assert d.available_ram_gb == pytest.approx(4.5)
    assert d.available_ram_bytes > 0


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_profile("nonexistent-model-xyz")


def test_unknown_device_raises():
    with pytest.raises(KeyError):
        get_device("nonexistent-device-xyz")
