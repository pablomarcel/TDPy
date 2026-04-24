# interpreter/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------ config / results ------------------------------


@dataclass(frozen=True)
class InterpretConfig:
    """
    Controls interpreter heuristics and the emitted 'solve' block defaults.

    Notes
    -----
    - keep_assignments_as_equations=False means we try to pull trivial constant
      assignments like "x = 3.0" into constants when RHS is numeric-only.
    - allow_residual_lines=True means a line like "x+y-3" is treated as "x+y-3 = 0"
      (primarily in equations/default mode).
    - infer_report governs the auto-report selection when the user doesn't provide
      an explicit report list.
    """
    # Output naming
    title: Optional[str] = None

    # Solve defaults to emit
    backend: str = "auto"          # auto|scipy|gekko
    method: str = "hybr"           # scipy.root method, for GEKKO it may be ignored
    tol: float = 1e-6
    max_iter: int = 50
    max_restarts: int = 2

    # Heuristics
    keep_assignments_as_equations: bool = False
    allow_residual_lines: bool = True
    infer_report: str = "unknowns"  # "unknowns"|"all"|"none"
    default_guess: float = 1.0

    # Units support in text (e.g., "T = 300 K")
    enable_units: bool = True

    # Optional: if True, treat any interpreter warning as an error
    strict_warnings: bool = False


@dataclass(frozen=True)
class InterpretResult:
    """
    Result of interpretation (text -> JSON-ready spec).

    - ok: True when no fatal errors were found
    - spec: JSON-ready dictionary suitable for cli run
    - warnings/errors: human-readable diagnostics
    - meta: stable machine-readable summary (counts, unresolved constants, etc.)
    """
    ok: bool
    spec: Dict[str, Any]  # JSON-ready
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ------------------------------ parser output ------------------------------


@dataclass
class ParsedInput:
    """
    Output of the text parser stage.

    This intentionally stays close to the user's raw intent:
    - equation_lines: raw "equation-ish" lines (including residual lines)
    - given_lines:    raw constant assignments (or candidate givens)
    - guess_lines:    raw guess statements
    - report_names:   raw report variable names extracted by parser
    - solve_overrides: raw solve overrides extracted by parser
    - ignored_lines:  anything we didn't classify; kept for diagnostics/telemetry
    """
    title: Optional[str] = None

    equation_lines: List[str] = field(default_factory=list)
    given_lines: List[str] = field(default_factory=list)
    guess_lines: List[str] = field(default_factory=list)
    report_names: List[str] = field(default_factory=list)
    solve_overrides: Dict[str, Any] = field(default_factory=dict)

    ignored_lines: List[str] = field(default_factory=list)

    def add_ignored(self, line: str) -> None:
        s = (line or "").strip()
        if s:
            self.ignored_lines.append(s)


__all__ = [
    "InterpretConfig",
    "InterpretResult",
    "ParsedInput",
]
