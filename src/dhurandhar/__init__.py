"""
dhurandhar — धुरंधर

Sanskrit: dhura (धुर, burden) + dhara (धर, one who bears)
"Bearer of burdens" — a framework for deploying large multimodal models
on memory-constrained edge devices where they have no right to survive.

Modules
-------
models      Model profile registry (Gemma4, Qwen2.5, Granite, Llama, ...)
devices     Device profile registry (mobile, tablet, laptop edge targets)
analysis    PLE memory analysis, device feasibility, TurboQuant, RotorQuant, mmap profiler
dashboard   5-tab Gradio interactive dashboard (optional extra)
cli         dhurandhar-* CLI entry points
"""

__version__ = "0.1.0"
__author__  = "Suneel Marthi"
__license__ = "Apache-2.0"

from dhurandhar.devices._base import DeviceProfile
from dhurandhar.models._base import ModelProfile

__all__ = [
    "__version__",
    "ModelProfile",
    "DeviceProfile",
]
