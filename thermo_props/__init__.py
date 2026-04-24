# thermo_props/__init__.py
from __future__ import annotations

"""
thermo_props

Thermodynamic property services for TDPy.

Design goals:
- Keep imports lightweight: optional backends (CoolProp, Cantera, native mixtures)
  should not be imported at package import time.
- Provide a small, stable facade API for app/CLI/GUI use.
- Re-export the public surface lazily from thermo_props.api, thermo_props.state,
  and thermo_props.coolprop_backend.

Preferred public entry points:
- run(spec) / eval_states(spec)   -> app-facing facade for JSON-driven evaluation
- state(...)                      -> build ThermoState from two independent properties
- props(...), prop(...)           -> convenience wrappers
- ThermoState, DEFAULT_OUTPUTS    -> canonical thermo state model + default outputs
- coolprop_available()            -> availability probe
- CoolPropNotInstalled / CoolPropCallError -> clean backend errors

Notes:
- Keep this module dependency-light and backend-agnostic.
- Native mixture hooks (LiBr–H2O, NH3–H2O) are exposed via the thermo contract
  in coolprop_backend.py and api.py, not imported eagerly here.
"""

from typing import Any, TYPE_CHECKING

__all__ = [
    # High-level app-facing facade
    "run",
    "eval_states",
    # Primary API
    "state",
    "props",
    "prop",
    # Canonical state model + defaults
    "ThermoState",
    "DEFAULT_OUTPUTS",
    # Availability + errors
    "coolprop_available",
    "CoolPropNotInstalled",
    "CoolPropCallError",
]

# --- type-checker friendliness (does not run at runtime) --------------------
if TYPE_CHECKING:  # pragma: no cover
    from .api import (  # noqa: F401
        eval_states,
        prop,
        props,
        run,
        state,
    )
    from .coolprop_backend import (  # noqa: F401
        CoolPropCallError,
        CoolPropNotInstalled,
        coolprop_available,
    )
    from .state import (  # noqa: F401
        DEFAULT_OUTPUTS,
        ThermoState,
    )


def __getattr__(name: str) -> Any:
    """
    Lazy export resolver.

    This avoids importing optional heavy dependencies until they are actually needed.
    """
    if name in {"run", "eval_states", "state", "props", "prop"}:
        from .api import eval_states, prop, props, run, state  # type: ignore
        return {
            "run": run,
            "eval_states": eval_states,
            "state": state,
            "props": props,
            "prop": prop,
        }[name]

    if name in {"ThermoState", "DEFAULT_OUTPUTS"}:
        from .state import DEFAULT_OUTPUTS, ThermoState  # type: ignore
        return {
            "ThermoState": ThermoState,
            "DEFAULT_OUTPUTS": DEFAULT_OUTPUTS,
        }[name]

    if name in {"coolprop_available", "CoolPropNotInstalled", "CoolPropCallError"}:
        from .coolprop_backend import (  # type: ignore
            CoolPropCallError,
            CoolPropNotInstalled,
            coolprop_available,
        )
        return {
            "coolprop_available": coolprop_available,
            "CoolPropNotInstalled": CoolPropNotInstalled,
            "CoolPropCallError": CoolPropCallError,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Help tab-complete / introspection show lazy exports.
    """
    return sorted(set(list(globals().keys()) + __all__))
