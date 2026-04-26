from __future__ import annotations

"""Application API contracts for TDPy.

This module contains the lightweight request, result, and problem-specification
objects used by the top-level application layer.

Design rules
------------
The module intentionally stays dependency-light:

* ``RunRequest`` and ``RunResult`` are the stable envelopes used by the CLI,
  GUI, and ``app.py``.
* Equation-system dataclasses are not duplicated here. The canonical EES-style
  equation specs live in ``equations.spec``.
* Heavy optional backends such as SciPy, GEKKO, CoolProp, and Cantera are not
  imported here.

Runtime notes
-------------
TDPy supports ``"optimize"`` problem inputs in addition to ``"equations"``,
``"nozzle_ideal"``, and ``"thermo_props"``.

CLI and GUI callers may pass runtime overrides through ``RunRequest.opts`` or
``RunRequest.overrides``. Newer code should prefer ``opts``. Application code
should treat both mappings as additive and let ``opts`` take precedence.

Thermodynamic-property backend names are intentionally extensible. Backends may
include CoolProp, Cantera, and native mixture implementations depending on the
installed optional dependencies.
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

UnitSystem = Literal["SI", "English", "Mixed"]


@dataclass(frozen=True)
class RunRequest:
    """Top-level run request used by the app, CLI, and GUI.

    Parameters
    ----------
    in_path:
        Input file path. TDPy is intentionally file-driven, so this is the
        only required path.
    out_path:
        Optional output file path. When omitted, the application layer chooses
        a default path.
    make_plots:
        Request default plots when the selected solver supports them.
    unit_system:
        Unit-system hint. ``"Mixed"`` means individual fields may include
        explicit units such as ``"300 K"``.
    prefer_solver:
        Optional solver preference kept for compatibility with older callers.
    opts:
        Preferred runtime override mapping used by newer CLI tooling.
    overrides:
        Legacy runtime override mapping kept for compatibility.

    Notes
    -----
    This module does not apply overrides directly. The application layer decides
    how to merge and use them. For minimal surprise, application code should
    merge ``overrides`` first and then ``opts`` so that ``opts`` wins.
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
    """Standardized result envelope returned by the application layer.

    Parameters
    ----------
    ok:
        Whether the pipeline completed successfully.
    solver:
        Solver or backend label used for the run.
    in_path:
        Resolved input path.
    out_path:
        Resolved output path, if output was written.
    payload:
        JSON-serializable result payload returned by the solver.
    plots:
        Mapping from plot names to generated file paths.
    warnings:
        Non-fatal warnings produced during the run.
    meta:
        Additional metadata such as backend details, timing, or version tags.
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
    """Generic problem specification produced by the design layer.

    Parameters
    ----------
    problem_type:
        Extensible problem type string. Common values include
        ``"nozzle_ideal"``, ``"thermo_props"``, ``"equations"``, and
        ``"optimize"``.
    data:
        Problem-specific input mapping after the top-level ``problem_type`` has
        been separated.
    schema_version:
        Schema version for the generic envelope.
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
    """Request thermodynamic properties from a selected backend.

    Parameters
    ----------
    fluid:
        Fluid or working-pair identifier.
    inputs:
        Mapping of input property names to values.
    outputs:
        Requested output property names.
    backend:
        Backend selector such as ``"coolprop"``, ``"cantera"``, or ``"auto"``.
    basis:
        Property basis, usually ``"mass"`` or ``"molar"``.
    units:
        Optional mapping of property names to unit strings.
    options:
        Backend-specific options.

    Example
    -------
    A typical request may use ``fluid="R134a"``, inputs such as temperature and
    pressure, and outputs such as enthalpy, entropy, and density.
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
    """Result of a thermodynamic-property evaluation."""

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
