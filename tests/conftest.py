"""Pytest configuration — ensures src layout resolves correctly."""

import sys
from pathlib import Path

# Add src/ to sys.path so tests can import dhurandhar without install.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
