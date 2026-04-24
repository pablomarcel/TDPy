# equations/__init__.py
from __future__ import annotations

"""
equations

Nonlinear equation solving layer (EES-ish).

Design goals:
- Keep the rest of TDPy independent from the solving backend.
- Support GEKKO and SciPy backends (selected via API).
- Maintain a stable, GUI/JSON-friendly API surface.

New (warm-start + methods):
- Warm-start ("guess prepass") lives in the equations solver stack and is exposed
  via solve config / CLI opts. The public surface here stays stable; this module
  simply re-exports solve_system and spec helpers.

Import strategy:
- Keep `import equations` lightweight.
- Only import heavier pieces when you access solve_system / specs.
- Provide TYPE_CHECKING imports for IDEs/mypy.

Note:
- We intentionally keep availability checks lightweight and non-invasive.
"""

from typing import Any, TYPE_CHECKING

__all__ = [
    # Facade API (lazy)
    "solve_system",
    "solve",
    "solve_optimize",
    "EquationSystem",
    "Var",
    "Param",
    "Solution",
    "BackendKind",
    "BackendUnavailableError",
    # Availability helpers (lightweight)
    "has_gekko",
    "has_scipy",
    # Specs (often useful to import directly)
    "EquationSpec",
    "EquationSystemSpec",
    "VarSpec",
    "ParamSpec",
    "system_from_mapping",
]

# --- type-checker friendliness (does not run at runtime) --------------------
if TYPE_CHECKING:  # pragma: no cover
    from .api import (  # noqa: F401
        BackendKind,
        BackendUnavailableError,
        EquationSystem,
        Param,
        Solution,
        Var,
        solve,
        solve_optimize,
        solve_system,
    )
    from .spec import (  # noqa: F401
        EquationSpec,
        EquationSystemSpec,
        ParamSpec,
        VarSpec,
        system_from_mapping,
    )


def has_gekko() -> bool:
    """
    Return True if GEKKO is importable in the current environment.

    Important:
    - This is a shallow import check only.
    - If you have a local module named 'gekko' shadowing the pip package,
      this may return True but the solver will later raise a more explicit error.
    """
    try:
        import gekko  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def has_scipy() -> bool:
    """
    Return True if SciPy is importable in the current environment.

    This is a shallow import check only.
    """
    try:
        import scipy  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def __getattr__(name: str) -> Any:  # PEP 562: module-level lazy exports
    """
    Lazy export resolver.

    This avoids importing optional heavy deps at import time and lets you do:
      from equations import solve_system, EquationSystemSpec
    """
    if name in {
        "solve_system",
        "solve",
        "solve_optimize",
        "EquationSystem",
        "Var",
        "Param",
        "Solution",
        "BackendKind",
        "BackendUnavailableError",
    }:
        from .api import (  # type: ignore
            BackendKind,
            BackendUnavailableError,
            EquationSystem,
            Param,
            Solution,
            Var,
            solve,
            solve_optimize,
            solve_system,
        )

        return {
            "solve_system": solve_system,
            "solve": solve,
            "solve_optimize": solve_optimize,
            "EquationSystem": EquationSystem,
            "Var": Var,
            "Param": Param,
            "Solution": Solution,
            "BackendKind": BackendKind,
            "BackendUnavailableError": BackendUnavailableError,
        }[name]

    if name in {
        "EquationSpec",
        "EquationSystemSpec",
        "VarSpec",
        "ParamSpec",
        "system_from_mapping",
    }:
        from .spec import (  # type: ignore
            EquationSpec,
            EquationSystemSpec,
            ParamSpec,
            VarSpec,
            system_from_mapping,
        )

        return {
            "EquationSpec": EquationSpec,
            "EquationSystemSpec": EquationSystemSpec,
            "VarSpec": VarSpec,
            "ParamSpec": ParamSpec,
            "system_from_mapping": system_from_mapping,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Helps tab-complete / introspection show the lazy exports.
    return sorted(set(list(globals().keys()) + __all__))
