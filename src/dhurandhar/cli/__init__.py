"""
dhurandhar CLI entry points.

All commands are thin typer wrappers over dhurandhar.analysis.*
Full implementations land in v0.1.x.
"""

import typer
from rich.console import Console

console = Console()
_WIP = "[bold yellow]Coming in v0.1.x[/bold yellow] — namespace reserved."


def ple_analyze():
    """Peak Live Estimate memory analysis."""
    console.print(_WIP)


def device_check():
    """Device feasibility verdicts across registered device profiles."""
    console.print(_WIP)


def turbo_sweep():
    """TurboQuant KV cache compression quality sweep."""
    console.print(_WIP)


def mmap_profile():
    """Real mmap throughput + peak RSS benchmark."""
    console.print(_WIP)


def compare_codecs():
    """TurboQuant vs RotorQuant codec comparison."""
    console.print(_WIP)


def dashboard():
    """Launch 5-tab Gradio dashboard (requires dhurandhar[dashboard])."""
    console.print(_WIP)
