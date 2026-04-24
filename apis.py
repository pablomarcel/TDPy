from __future__ import annotations

"""
apis

Central, lightweight API contracts for the top-level application layer.

Design rules:
- RunRequest / RunResult are the stable envelopes used by CLI/GUI/app.py.
- Do NOT duplicate equation-system dataclasses here:
  canonical EES-ish equation specs live in `equations.spec`.
- Keep this module dependency-light: no SciPy / GEKKO / CoolProp imports.

Latest facts / upgrades (Feb 2026):
- The project now supports an optimization problem_type ("optimize") in addition to:
    - "equations"
    - "nozzle_ideal"
    - "thermo_props"
- CLI / GUI may pass runtime overrides as either:
    - RunRequest.opts          (newer, used by CLI)
    - RunRequest.overrides     (older, kept for back-compat)
  App code should treat them as additive, with opts taking precedence if both are provided.
- thermo_props backend strings are no longer limited to CoolProp. They may include
  Cantera and native mixture backends (LiBr–H2O, NH3–H2O) depending on what's installed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

__all__ = [
    # runner envelopes
    "UnitSystem",
    "RunRequest",
    "RunResult",
    "ProblemSpec",
    # thermo_props contracts (app-facing)
    "PropsBackend",
    "PropsBasis",
    "ThermoPropsRequest",
    "ThermoPropsResult",
    # equations canonical re-exports (single source of truth)
    "EquationSpec",
    "EquationSystemSpec",
    "VarSpec",
    "ParamSpec",
    "SpecError",
    # back-compat aliases
    "EquationsProblemSpec",
    "EquationsSpecError",
    "VariableSpec",
]


# ---------------------------------------------------------------------
# Core runner request/response
# ---------------------------------------------------------------------

UnitSystem = Literal["SI", "English", "Mixed"]  # Mixed = accept per-field units (e.g., "300 K")


@dataclass(frozen=True)
class RunRequest:
    """
    Top-level run request (app/CLI/GUI envelope).

    Backward compatible:
      - in_path is required (file-driven)
      - out_path optional
      - make_plots optional

    Runtime override channels:
      - opts:      primary channel for CLI overrides (preferred)
      - overrides: legacy channel kept for compatibility

    Notes:
      - This module does not enforce how overrides are applied; app.py decides.
      - For minimal surprise, app.py should merge overrides then opts, so opts wins.
    """

    in_path: Path
    out_path: Optional[Path] = None
    make_plots: bool = False

    unit_system: UnitSystem = "SI"
    prefer_solver: Optional[str] = None

    # Newer name (used by CLI tooling)
    opts: Mapping[str, Any] = field(default_factory=dict)

    # Older name (kept for existing callers)
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """
    Standardized result envelope.

    payload: arbitrary JSON-serializable dict returned by a solver.
    plots:   name -> file path (usually HTML for Plotly)
    meta:    solver + versioning + timing info, etc.
    """

    ok: bool
    solver: str
    in_path: Path
    out_path: Optional[Path]
    payload: Dict[str, Any]

    plots: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemSpec:
    """
    Generic problem specification produced by the design/build layer.

    problem_type is a string to keep the registry extensible.
    Common values in this project:
      - "nozzle_ideal"
      - "thermo_props"
      - "equations"
      - "optimize"
    """

    problem_type: str
    data: Dict[str, Any]
    schema_version: str = "1.0"


# ---------------------------------------------------------------------
# thermo_props API contracts (app-facing)
# ---------------------------------------------------------------------

# Keep backend strings lowercase for consistency across the project.
# NOTE: The runtime may support additional backends depending on what is installed.
PropsBackend = Literal["coolprop", "cantera", "librh2o", "nh3h2o", "auto"]
PropsBasis = Literal["mass", "molar"]


@dataclass(frozen=True)
class ThermoPropsRequest:
    """
    Request a set of thermodynamic_properties from a backend (e.g., CoolProp).

    Example:
      fluid="R134a"
      inputs={"T": 300.0, "P": 101325.0}
      outputs=["H","S","D"]
      basis="mass"
      units={"T":"K","P":"Pa","H":"J/kg","S":"J/kg-K","D":"kg/m^3"}  # optional
    """

    fluid: str
    inputs: Mapping[str, Any]
    outputs: Sequence[str]

    backend: PropsBackend = "coolprop"
    basis: PropsBasis = "mass"
    units: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThermoPropsResult:
    """Result of a thermo_props evaluation."""

    fluid: str
    backend: str
    basis: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

    units: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# equations (canonical contracts live in equations.spec)
# ---------------------------------------------------------------------
# IMPORTANT:
# - Do NOT duplicate equation dataclasses here.
# - Re-export the canonical spec types so the rest of the app has ONE truth.
# ---------------------------------------------------------------------

from equations.spec import (  # noqa: E402
    EquationSpec,
    EquationSystemSpec,
    ParamSpec,
    SpecError,
    VarSpec,
)

# Back-compat aliases (older code may import these names from apis)
EquationsProblemSpec = EquationSystemSpec
EquationsSpecError = SpecError
VariableSpec = VarSpec
