"""
tdpy

A console-first, EES-like thermodynamics and equation-solving toolkit.

Current direction:
- thermo_props: CoolProp-backed property calculations (with a clean, testable API)
- equations:    GEKKO/SciPy-based nonlinear equation solving
- units:        lightweight unit parsing + conversion utilities

Design goals:
- CLI-first workflows (JSON/YAML/TXT) with reproducible outputs
- Later: PySide6 GUI that calls the same APIs (no GUI-only logic)
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0"
