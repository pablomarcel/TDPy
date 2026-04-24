# thermo_props/coolprop_backend.py
from __future__ import annotations

"""
thermo_props.coolprop_backend

CoolProp backend (low-level) + thermo registration hub.

This module is intentionally the "contract surface" consumed by:
- equations.safe_eval / equations.api eval context
- solver warm-start / property detection
- interpreter constant folding (via build_spec injected funcs)

Backends exposed here (all optional / lazy):
- CoolProp: PropsSI, PhaseSI
- CoolProp humid air: HAPropsSI
- CoolProp AbstractState wrappers: ASPropsSI + fugacity helpers
- Cantera backend (independent): CTPropsSI family (delegates to thermo_props.cantera_backend)
- Native NH3–H2O hook (explicit aliases only) inside PropsSI/PhaseSI

IMPORTANT DESIGN RULES
----------------------
- Import-time light: do not import CoolProp or Cantera at module import time.
- Robust optional dependencies: missing backends should fail with clear, wrapped errors.
- Stable outputs: always return Python floats/strings/dicts, never NumPy scalars.

Cantera integration note (Feb 2026)
-----------------------------------
Cantera is **not** part of CoolProp. We keep Cantera implementation in
`thermo_props.cantera_backend` and *delegate* from this module.
This keeps the CoolProp backend thin while still allowing the equations layer to import a single "thermo contract" module.

If you upgrade `cantera_backend.py` (e.g., memoization), you should NOT need
to change this module unless you add new public Cantera helpers to re-export.
This file re-exports the Cantera cache helpers when available:
  - ctprops_cache_info()
  - clear_ctprops_caches()
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence
import math
import numbers
import re
import warnings

__all__ = [
    # errors
    "CoolPropNotInstalled",
    "CoolPropCallError",
    "CanteraNotInstalled",
    "CanteraCallError",
    # availability
    "coolprop_available",
    "haprops_available",
    "ha_props_available",
    "abstractstate_available",
    "cantera_available",
    # version/params
    "coolprop_version",
    "global_param_string",
    # Cantera CTPropsSI wrappers (delegated)
    "ctprops_si",
    "ctprops_multi",
    "batch_ctprops",
    "ctprops_cache_info",
    "clear_ctprops_caches",
    # PropsSI/PhaseSI wrappers
    "props_si",
    "props_multi",
    "phase_si",
    "batch_props",
    # AbstractState wrappers
    "as_props_si",
    "as_props_multi",
    "batch_as_props",
    "FugacitySI",
    "FugacityCoeffSI",
    "LnFugacityCoeffSI",
    "ChemicalPotentialSI",
    # HAPropsSI wrappers
    "haprops_si",
    "haprops_multi",
    "batch_haprops",
    # back-compat HA aliases
    "ha_props_si",
    "ha_props_multi",
    "batch_ha_props",
    # CoolProp-like shims
    "PropsSI",
    "PhaseSI",
    "HAPropsSI",
    "ASPropsSI",
    "CTPropsSI",
    # dataclasses
    "CPCall",
    "HACall",
    "ASCall",
    "CTCall",
]

# ------------------------------ errors ------------------------------

class CoolPropNotInstalled(ImportError):
    """Raised when CoolProp cannot be imported."""


class CoolPropCallError(RuntimeError):
    """Raised when a CoolProp (or compatible shim) call fails; message includes full call context."""


class CanteraNotInstalled(ImportError):
    """Raised when Cantera backend cannot be imported."""


class CanteraCallError(RuntimeError):
    """Raised when a Cantera (or compatible shim) call fails; message includes full call context."""


# ------------------------------ internal import caching ------------------------------

_PropsSI: Callable[..., float] | None = None
_PhaseSI: Callable[..., str] | None = None
_HAPropsSI: Callable[..., float] | None = None
_get_global_param_string: Callable[..., str] | None = None

# AbstractState helpers (lazily imported)
_AbstractState: Callable[..., Any] | None = None
_get_parameter_index: Callable[..., int] | None = None
_generate_update_pair: Callable[..., tuple] | None = None

# Cantera backend module (optional) lazy import
_ct_backend_mod: Any | None = None
_ct_backend_checked: bool = False


def _try_import_cantera_backend() -> Any | None:
    """Try to import thermo_props.cantera_backend; return module or None."""
    global _ct_backend_mod, _ct_backend_checked
    if _ct_backend_checked:
        return _ct_backend_mod
    _ct_backend_checked = True
    try:
        from . import cantera_backend as cb  # type: ignore
    except Exception:
        _ct_backend_mod = None
        return None
    _ct_backend_mod = cb
    return _ct_backend_mod


def _import_coolprop() -> tuple[Callable[..., float], Callable[..., str]]:
    """Import CoolProp lazily and cache callables: (PropsSI, PhaseSI)."""
    global _PropsSI, _PhaseSI
    if _PropsSI is not None and _PhaseSI is not None:
        return _PropsSI, _PhaseSI

    try:
        from CoolProp.CoolProp import PropsSI, PhaseSI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise CoolPropNotInstalled(
            "CoolProp is not installed or could not be imported. Install with: pip install CoolProp"
        ) from e

    _PropsSI = PropsSI
    _PhaseSI = PhaseSI
    return _PropsSI, _PhaseSI


def _import_coolprop_ha() -> Callable[..., float]:
    """Import CoolProp HAPropsSI lazily and cache callable."""
    global _HAPropsSI
    if _HAPropsSI is not None:
        return _HAPropsSI
    try:
        from CoolProp.CoolProp import HAPropsSI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise CoolPropNotInstalled(
            "CoolProp is not installed or could not be imported (HAPropsSI unavailable). "
            "Install with: pip install CoolProp"
        ) from e
    _HAPropsSI = HAPropsSI
    return _HAPropsSI


def _import_coolprop_params() -> Callable[..., str]:
    """Import CoolProp get_global_param_string lazily and cache callable."""
    global _get_global_param_string
    if _get_global_param_string is not None:
        return _get_global_param_string
    try:
        from CoolProp.CoolProp import get_global_param_string  # type: ignore
    except Exception as e:  # pragma: no cover
        raise CoolPropNotInstalled(
            "CoolProp is not installed or could not be imported (get_global_param_string unavailable). "
            "Install with: pip install CoolProp"
        ) from e
    _get_global_param_string = get_global_param_string
    return _get_global_param_string


def _import_coolprop_state() -> tuple[Callable[..., Any], Callable[..., int], Callable[..., tuple]]:
    """Import CoolProp AbstractState helpers lazily and cache callables."""
    global _AbstractState, _get_parameter_index, _generate_update_pair
    if _AbstractState is not None and _get_parameter_index is not None and _generate_update_pair is not None:
        return _AbstractState, _get_parameter_index, _generate_update_pair

    try:
        from CoolProp.CoolProp import AbstractState, get_parameter_index, generate_update_pair  # type: ignore
    except Exception as e:  # pragma: no cover
        raise CoolPropNotInstalled(
            "CoolProp is not installed or could not be imported (AbstractState unavailable). "
            "Install with: pip install CoolProp"
        ) from e

    _AbstractState = AbstractState
    _get_parameter_index = get_parameter_index
    _generate_update_pair = generate_update_pair
    return _AbstractState, _get_parameter_index, _generate_update_pair


# ------------------------------ availability ------------------------------

def coolprop_available() -> bool:
    """Return True if PropsSI/PhaseSI are importable."""
    try:
        _import_coolprop()
        return True
    except Exception:
        return False


def haprops_available() -> bool:
    """Return True if HAPropsSI is importable."""
    try:
        _import_coolprop_ha()
        return True
    except Exception:
        return False


def ha_props_available() -> bool:
    """Back-compat alias."""
    return haprops_available()


def abstractstate_available() -> bool:
    """Return True if CoolProp AbstractState helpers are importable."""
    try:
        _import_coolprop_state()
        return True
    except Exception:
        return False


def cantera_available() -> bool:
    """Return True if tdpy Cantera backend is importable and reports availability."""
    cb = _try_import_cantera_backend()
    if cb is None:
        return False
    f = getattr(cb, "cantera_available", None)
    try:
        return bool(f()) if callable(f) else True
    except Exception:
        return False


# ------------------------------ version/params ------------------------------

def coolprop_version() -> str | None:
    """Return CoolProp version string if available, else None."""
    try:
        import CoolProp  # type: ignore
        v = getattr(CoolProp, "__version__", None)
        return str(v) if v is not None else None
    except Exception:
        return None


def global_param_string(name: str) -> str:
    """Wrapper for CoolProp.get_global_param_string."""
    if not str(name).strip():
        raise ValueError("global param name must be a non-empty string.")
    fn = _import_coolprop_params()
    try:
        return str(fn(str(name)))
    except Exception as e:
        raise CoolPropCallError(f"CoolProp get_global_param_string failed for {name!r}.") from e


# ------------------------------ dataclasses ------------------------------

@dataclass(frozen=True)
class CPCall:
    out: str
    in1: str
    v1: float
    in2: str
    v2: float
    fluid: str


@dataclass(frozen=True)
class HACall:
    out: str
    in1: str
    v1: float
    in2: str
    v2: float
    in3: str
    v3: float


@dataclass(frozen=True)
class ASCall:
    out: str
    in1: str
    v1: float
    in2: str
    v2: float
    fluid: str


@dataclass(frozen=True)
class CTCall:
    out: str
    in1: str
    v1: float
    in2: str
    v2: float
    fluid: str


# ------------------------------ small helpers ------------------------------

def _finite(x: float) -> bool:
    try:
        return isinstance(x, numbers.Real) and math.isfinite(float(x))
    except Exception:
        return False


def _to_float(name: str, x: Any) -> float:
    try:
        y = float(x)
    except Exception as e:
        raise ValueError(f"{name} must be a real scalar convertible to float. Got {x!r}") from e
    if not _finite(y):
        raise ValueError(f"{name} must be finite. Got {x!r}")
    return y


def _ensure_fluid(fluid: str) -> str:
    f = str(fluid).strip()
    if not f:
        raise ValueError("fluid must be a non-empty string.")
    return f


def _wrap_call_error(
    *,
    what: str,
    out: str | None,
    out_raw: str | None,
    in1: str,
    v1: float,
    in2: str,
    v2: float,
    fluid: str,
    exc: BaseException | None = None,
) -> CoolPropCallError:
    lines: list[str] = [f"CoolProp {what} call failed."]
    if out is not None:
        if out_raw is not None and out_raw != out:
            lines.append(f"  out={out!r} (from {out_raw!r})")
        else:
            lines.append(f"  out={out!r}")
    lines.append(f"  in1={in1!r} v1={v1}")
    lines.append(f"  in2={in2!r} v2={v2}")
    lines.append(f"  fluid={fluid!r}")
    if exc is not None:
        lines.append(f"  cause={type(exc).__name__}: {exc}")
    return CoolPropCallError("\n".join(lines))


def _wrap_ha_call_error(
    *,
    what: str,
    out: str | None,
    out_raw: str | None,
    in1: str,
    v1: float,
    in2: str,
    v2: float,
    in3: str,
    v3: float,
    exc: BaseException | None = None,
) -> CoolPropCallError:
    lines: list[str] = [f"CoolProp {what} call failed (humid air)."]
    if out is not None:
        if out_raw is not None and out_raw != out:
            lines.append(f"  out={out!r} (from {out_raw!r})")
        else:
            lines.append(f"  out={out!r}")
    lines.append(f"  in1={in1!r} v1={v1}")
    lines.append(f"  in2={in2!r} v2={v2}")
    lines.append(f"  in3={in3!r} v3={v3}")
    if exc is not None:
        lines.append(f"  cause={type(exc).__name__}: {exc}")
    return CoolPropCallError("\n".join(lines))


def _iter_calls_generic(calls: Iterable[Any], cls: Any, n: int) -> Iterable[Any]:
    """Accept dataclasses, dicts, and tuples for batch calls."""
    for c in calls:
        if isinstance(c, cls):
            yield c
            continue
        if isinstance(c, Mapping):
            try:
                yield cls(**{k: c[k] for k in cls.__dataclass_fields__.keys()})  # type: ignore
                continue
            except Exception as e:
                raise ValueError(f"Invalid {cls.__name__} mapping: {c!r}") from e
        if isinstance(c, (tuple, list)) and len(c) == n:
            try:
                yield cls(*c)  # type: ignore
                continue
            except Exception as e:
                raise ValueError(f"Invalid {cls.__name__} tuple/list: {c!r}") from e
        raise ValueError(f"Invalid {cls.__name__} item: {c!r}")


# ------------------------------ normalization (PropsSI) ------------------------------

_INPUT_ALIASES: Mapping[str, str] = {
    "t": "T",
    "temp": "T",
    "temperature": "T",
    "p": "P",
    "press": "P",
    "pressure": "P",
    "q": "Q",
    "x": "Q",
    "h": "Hmass",
    "hmass": "Hmass",
    "s": "Smass",
    "smass": "Smass",
    "u": "Umass",
    "umass": "Umass",
    "rho": "Dmass",
    "d": "Dmass",
    "dmass": "Dmass",
}

_OUTPUT_ALIASES: Mapping[str, str] = {
    "t": "T",
    "p": "P",
    "q": "Q",
    "x": "Q",
    "h": "Hmass",
    "s": "Smass",
    "u": "Umass",
    "rho": "Dmass",
    "d": "Dmass",
    "cp": "Cpmass",
    "cv": "Cvmass",
    "a": "A",
    "mu": "V",
    "k": "L",
    "pr": "Prandtl",
    "viscosity": "V",
    "conductivity": "L",
    "surface_tension": "surface_tension",
    "surfacetension": "surface_tension",
    "sigma": "surface_tension",
}

@lru_cache(maxsize=256)
def _norm_in(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("CoolProp input key is empty.")
    return _INPUT_ALIASES.get(k.lower(), k)

@lru_cache(maxsize=256)
def _norm_out(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("CoolProp output key is empty.")
    return _OUTPUT_ALIASES.get(k.lower(), k)

# ------------------------------ normalization (HAPropsSI) ------------------------------

_HA_INPUT_ALIASES: Mapping[str, str] = {
    "t": "T",
    "temp": "T",
    "temperature": "T",
    "p": "P",
    "press": "P",
    "pressure": "P",
    "r": "R",
    "rh": "R",
    "w": "W",
    "omega": "W",
    "h": "H",
    "s": "S",
    "v": "V",
    "rho": "D",
    "d": "D",
    "tdp": "D",
    "dewpoint": "D",
    "twb": "B",
    "wetbulb": "B",
}
_HA_OUTPUT_ALIASES: Mapping[str, str] = {
    **_HA_INPUT_ALIASES,
    "cp": "C",
    "mu": "M",
    "visc": "M",
    "k": "K",
    "cond": "K",
    "conductivity": "K",
    "hda": "Hda",
    "sda": "Sda",
}

@lru_cache(maxsize=256)
def _ha_norm_in(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("CoolProp HAPropsSI input key is empty.")
    return _HA_INPUT_ALIASES.get(k.lower(), k)

@lru_cache(maxsize=256)
def _ha_norm_out(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("CoolProp HAPropsSI output key is empty.")
    return _HA_OUTPUT_ALIASES.get(k.lower(), k)

# ------------------------------ native mixture hook (NH3–H2O) ------------------------------

_FLUID_BRACKET_RE = re.compile(r"^(?P<head>[^\[]+)\[(?P<tail>.+)\]\s*$")

def _norm_fluid_token(s: str) -> str:
    return re.sub(r"[\s_\-\/]", "", str(s).strip()).upper()

_NH3H2O_ALIASES_NORM: set[str] = {
    "NH3H2O",
    "AMMONIAWATER",
    "AMMONIAH2O",
    "AMMONIAWATERSOLUTION",
    "AMMONIAWATERSOLN",
    "NH3WATER",
}

def _is_nh3h2o_fluid(base: str) -> bool:
    return _norm_fluid_token(base) in _NH3H2O_ALIASES_NORM

@dataclass(frozen=True)
class _ParsedFluid:
    raw: str
    base: str
    x: float | None
    strict: bool | None

def _parse_bool_token(v: str) -> bool | None:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None

def _parse_special_fluid(fluid: str) -> _ParsedFluid:
    raw = str(fluid).strip()
    if not raw:
        raise ValueError("fluid must be a non-empty string.")
    head = raw
    tail = ""
    m = _FLUID_BRACKET_RE.match(raw)
    if m:
        head = m.group("head").strip()
        tail = m.group("tail").strip()
    else:
        if "|" in raw:
            head, tail = raw.split("|", 1)
            head = head.strip()
            tail = tail.strip()
        elif ";" in raw:
            head, tail = raw.split(";", 1)
            head = head.strip()
            tail = tail.strip()
    if not tail and "@" in head:
        h2, t2 = head.split("@", 1)
        head = h2.strip()
        tail = t2.strip()

    x: float | None = None
    strict: bool | None = None
    if tail:
        toks = [t for t in re.split(r"[,\s|;]+", tail) if t.strip()]
        for tok in toks:
            t = tok.strip()
            if not t:
                continue
            if "=" in t:
                k, v = t.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in {"x", "x_nh3", "w", "w_nh3", "massfrac", "massfraction", "xmass", "x_mass"}:
                    x = float(v)
                    continue
                if k == "strict":
                    b = _parse_bool_token(v)
                    if b is None:
                        raise ValueError(f"Invalid strict flag in fluid spec: {tok!r}")
                    strict = b
                    continue
                continue
            # bare numeric token
            try:
                x = float(t)
            except Exception:
                continue

    return _ParsedFluid(raw=raw, base=head, x=x, strict=strict)

def _nh3h2o_out_from_props_token(out_raw: str) -> str:
    o = str(out_raw).strip()
    if not o:
        raise ValueError("output key is empty")
    if o == "Hmass":
        return "h"
    if o == "Smass":
        return "s"
    if o == "Umass":
        return "u"
    if o == "Dmass":
        return "rho"
    if o == "Q":
        return "q"
    if o == "T":
        return "T_K"
    if o == "P":
        return "P_Pa"
    if o == "V":
        return "mu"
    if o == "L":
        return "k"

    s = o.lower()
    if s in {"h", "hmass", "enthalpy"}:
        return "h"
    if s in {"s", "smass", "entropy"}:
        return "s"
    if s in {"u", "umass", "internal_energy", "internalenergy"}:
        return "u"
    if s in {"v", "specificvolume", "vol"}:
        return "v"
    if s in {"rho", "d", "density", "dmass"}:
        return "rho"
    if s in {"q", "quality", "x"}:
        return "q"
    if s in {"t", "temp", "temperature"}:
        return "T_K"
    if s in {"p", "press", "pressure"}:
        return "P_Pa"
    if s in {"x_nh3", "xmass", "x_mass", "w", "w_nh3", "massfrac", "massfraction"}:
        return "X"
    if s in {"mu", "viscosity"}:
        return "mu"
    if s in {"k", "cond", "conductivity"}:
        return "k"
    if s in {"sigma", "surface_tension", "surfacetension"}:
        return "sigma"

    raise ValueError(f"NH3H2O backend does not support output {out_raw!r} via PropsSI.")

def _nh3h2o_norm_in_TP(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("input key is empty")
    s = k.lower()
    if s in {"t", "temp", "temperature", "t_k"}:
        return "T"
    if s in {"p", "press", "pressure", "p_pa"}:
        return "P"
    return k

def _nh3h2o_require_t_p(in1: str, v1: float, in2: str, v2: float) -> tuple[float, float]:
    k1 = _nh3h2o_norm_in_TP(in1)
    k2 = _nh3h2o_norm_in_TP(in2)
    vals: dict[str, float] = {k1: _to_float("v1", v1), k2: _to_float("v2", v2)}
    if "T" not in vals or "P" not in vals:
        raise ValueError("NH3H2O PropsSI/PhaseSI requires inputs T and P (order-agnostic).")
    return float(vals["T"]), float(vals["P"])

def _nh3h2o_prop_si(out: str, in1: str, v1: float, in2: str, v2: float, fluid_spec: _ParsedFluid) -> float:
    if fluid_spec.x is None:
        raise ValueError("NH3H2O requires NH3 mass fraction X in the fluid string, e.g. 'NH3H2O|X=0.30'.")
    T, P = _nh3h2o_require_t_p(in1, v1, in2, v2)
    X = float(fluid_spec.x)
    strict = True if fluid_spec.strict is None else bool(fluid_spec.strict)
    out_key = _nh3h2o_out_from_props_token(out)
    try:
        from . import nh3h2o_backend as nb  # type: ignore
    except Exception as e:
        raise CoolPropCallError("NH3H2O requested but nh3h2o_backend is not importable.") from e
    try:
        y = float(nb.NH3H2O_TPX(out_key, T, P, X, strict=strict))
    except Exception as e:
        raise CoolPropCallError(
            "NH3H2O native backend call failed.\n"
            f"  out={out!r} -> {out_key!r}\n"
            f"  T={T} K, P={P} Pa, X={X}, strict={strict}"
        ) from e
    if strict and not _finite(y):
        raise CoolPropCallError(f"NH3H2O returned non-finite result for out={out_key!r}, T={T}, P={P}, X={X}.")
    return float(y)

def _nh3h2o_phase_si(in1: str, v1: float, in2: str, v2: float, fluid_spec: _ParsedFluid) -> str:
    if fluid_spec.x is None:
        raise ValueError("NH3H2O requires NH3 mass fraction X in the fluid string.")
    T, P = _nh3h2o_require_t_p(in1, v1, in2, v2)
    X = float(fluid_spec.x)
    strict = True if fluid_spec.strict is None else bool(fluid_spec.strict)
    try:
        from . import nh3h2o_backend as nb  # type: ignore
    except Exception as e:
        raise CoolPropCallError("NH3H2O requested but nh3h2o_backend is not importable.") from e
    try:
        st = nb.state_tpx(T, P, X, strict=strict)
    except Exception as e:
        raise CoolPropCallError("NH3H2O native backend phase/state call failed.") from e

    ph = st.get("phase", None)
    if isinstance(ph, str) and ph.strip():
        p = ph.strip().lower()
        if p in {"l", "liq", "liquid"}:
            return "liquid"
        if p in {"g", "v", "vap", "vapor", "gas"}:
            return "gas"
        if p in {"2ph", "2phase", "two-phase", "twophase"}:
            return "twophase"
        return ph.strip()

    q = st.get("q", None)
    try:
        qf = float(q)  # type: ignore[arg-type]
        if math.isfinite(qf):
            if 0.0 < qf < 1.0:
                return "twophase"
            if qf <= 0.0:
                return "liquid"
            if qf >= 1.0:
                return "gas"
    except Exception:
        pass
    return "unknown"

# ------------------------------ Cantera wrappers (delegated) ------------------------------

def ctprops_si(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    cb = _try_import_cantera_backend()
    if cb is None:
        raise CanteraNotInstalled("thermo_props.cantera_backend is not importable.")
    f = getattr(cb, "ctprops_si", None) or getattr(cb, "CTPropsSI", None)
    if not callable(f):
        raise CanteraNotInstalled("cantera_backend does not expose ctprops_si/CTPropsSI.")
    try:
        return float(f(out, in1, v1, in2, v2, fluid))
    except Exception as e:
        raise CanteraCallError(
            "CTPropsSI call failed.\n"
            f"  out={out!r}\n"
            f"  in1={in1!r} v1={v1!r}\n"
            f"  in2={in2!r} v2={v2!r}\n"
            f"  fluid={str(fluid)!r}\n"
            f"  cause={type(e).__name__}: {e}"
        ) from e

def ctprops_multi(outputs: Sequence[str], in1: str, v1: float, in2: str, v2: float, fluid: str) -> dict[str, float]:
    cb = _try_import_cantera_backend()
    if cb is None:
        raise CanteraNotInstalled("thermo_props.cantera_backend is not importable.")
    f = getattr(cb, "ctprops_multi", None)
    if callable(f):
        out = f(outputs, in1, v1, in2, v2, fluid)
        return {str(k): float(v) for k, v in dict(out).items()}
    # fallback: call ctprops_si repeatedly
    return {str(k): ctprops_si(str(k), in1, v1, in2, v2, fluid) for k in outputs}

def batch_ctprops(calls: Iterable[CTCall]) -> list[float]:
    ys: list[float] = []
    for c in _iter_calls_generic(calls, CTCall, 6):
        ys.append(ctprops_si(c.out, c.in1, float(c.v1), c.in2, float(c.v2), c.fluid))
    return ys

def ctprops_cache_info() -> dict[str, Any]:
    cb = _try_import_cantera_backend()
    f = getattr(cb, "ctprops_cache_info", None) if cb is not None else None
    if callable(f):
        return dict(f())
    return {"available": cantera_available(), "note": "ctprops_cache_info unavailable (old cantera_backend?)"}

def clear_ctprops_caches() -> None:
    cb = _try_import_cantera_backend()
    f = getattr(cb, "clear_ctprops_caches", None) if cb is not None else None
    if callable(f):
        f()
        return

# ------------------------------ PropsSI / PhaseSI ------------------------------

def props_si(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    """
    CoolProp PropsSI wrapper with:
    - alias normalization
    - strict float conversion
    - optional NH3H2O native hook (explicit aliases only)
    """
    f = _ensure_fluid(fluid)

    # NH3-H2O intercept
    pf = _parse_special_fluid(f)
    if _is_nh3h2o_fluid(pf.base):
        return _nh3h2o_prop_si(out, in1, _to_float("v1", v1), in2, _to_float("v2", v2), pf)

    v1f = _to_float("v1", v1)
    v2f = _to_float("v2", v2)

    out_raw = str(out)
    out_n = _norm_out(out_raw)
    in1_n = _norm_in(in1)
    in2_n = _norm_in(in2)

    PropsSI_fn, _ = _import_coolprop()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y = float(PropsSI_fn(out_n, in1_n, v1f, in2_n, v2f, f))
        except Exception as e:
            raise _wrap_call_error(
                what="PropsSI",
                out=out_n,
                out_raw=out_raw,
                in1=in1_n,
                v1=v1f,
                in2=in2_n,
                v2=v2f,
                fluid=f,
                exc=e,
            ) from e

    if not _finite(y):
        raise CoolPropCallError(
            "CoolProp returned a non-finite result.\n"
            f"  out={out_n!r}, in1={in1_n!r}, v1={v1f}, in2={in2_n!r}, v2={v2f}, fluid={f!r}\n"
            f"  y={y!r}"
        )
    return float(y)

def phase_si(in1: str, v1: float, in2: str, v2: float, fluid: str) -> str:
    """CoolProp PhaseSI wrapper with optional NH3H2O hook."""
    f = _ensure_fluid(fluid)

    pf = _parse_special_fluid(f)
    if _is_nh3h2o_fluid(pf.base):
        return _nh3h2o_phase_si(in1, _to_float("v1", v1), in2, _to_float("v2", v2), pf)

    v1f = _to_float("v1", v1)
    v2f = _to_float("v2", v2)
    in1_n = _norm_in(in1)
    in2_n = _norm_in(in2)

    _, PhaseSI_fn = _import_coolprop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return str(PhaseSI_fn(in1_n, v1f, in2_n, v2f, f))
        except Exception as e:
            raise _wrap_call_error(
                what="PhaseSI",
                out=None,
                out_raw=None,
                in1=in1_n,
                v1=v1f,
                in2=in2_n,
                v2=v2f,
                fluid=f,
                exc=e,
            ) from e

def props_multi(outputs: Sequence[str], in1: str, v1: float, in2: str, v2: float, fluid: str) -> dict[str, float]:
    return {str(k): props_si(str(k), in1, v1, in2, v2, fluid) for k in outputs}

def batch_props(calls: Iterable[CPCall]) -> list[float]:
    ys: list[float] = []
    for c in _iter_calls_generic(calls, CPCall, 6):
        ys.append(props_si(c.out, c.in1, float(c.v1), c.in2, float(c.v2), c.fluid))
    return ys

# ------------------------------ AbstractState wrappers ------------------------------

_BACKEND_PREFIX_RE = re.compile(r"^\s*(?P<backend>[^:]+)::\s*(?P<fluid>.+?)\s*$")

def _split_backend_and_fluid(fluid: str, default_backend: str = "HEOS") -> tuple[str, str]:
    raw = str(fluid).strip()
    m = _BACKEND_PREFIX_RE.match(raw)
    if m:
        return m.group("backend").strip(), m.group("fluid").strip()
    return default_backend, raw

_AS_SPECIAL_OUT: Mapping[str, str] = {
    "f": "FUGACITY",
    "fugacity": "FUGACITY",
    "phi": "FUGACITY_COEFFICIENT",
    "fugacity_coefficient": "FUGACITY_COEFFICIENT",
    "fugacitycoefficient": "FUGACITY_COEFFICIENT",
    "fugacity_coeff": "FUGACITY_COEFFICIENT",
    "fugacitycoeff": "FUGACITY_COEFFICIENT",
    "ln_phi": "LN_FUGACITY_COEFFICIENT",
    "lnphi": "LN_FUGACITY_COEFFICIENT",
    "ln_fugacity_coefficient": "LN_FUGACITY_COEFFICIENT",
    "lnfugacitycoefficient": "LN_FUGACITY_COEFFICIENT",
    "chemical_potential": "CHEMICAL_POTENTIAL",
    "chemicalpotential": "CHEMICAL_POTENTIAL",
    "chempot": "CHEMICAL_POTENTIAL",
}
_AS_SPECIAL_KEYS = {"FUGACITY", "FUGACITY_COEFFICIENT", "LN_FUGACITY_COEFFICIENT", "CHEMICAL_POTENTIAL"}

def _parse_component_suffix(out_key: str) -> tuple[str, int]:
    s = str(out_key).strip()
    m = re.match(r"^(?P<base>.+?)(?:\[(?P<i>\d+)\]|:(?P<i2>\d+))\s*$", s)
    if not m:
        return s, 0
    base = m.group("base").strip()
    idx = m.group("i") or m.group("i2") or "0"
    i = int(idx)
    if i < 0:
        raise ValueError(f"Component index must be >= 0 in output key: {out_key!r}")
    return base, i

@lru_cache(maxsize=128)
def _as_cached_state(backend: str, fluid: str) -> Any:
    AbstractState, _, _ = _import_coolprop_state()
    return AbstractState(str(backend), str(fluid))

def as_props_si(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    f_full = _ensure_fluid(fluid)
    pf = _parse_special_fluid(f_full)
    if _is_nh3h2o_fluid(pf.base):
        raise CoolPropCallError("ASPropsSI/as_props_si is CoolProp-only; NH3H2O must use native backend.")
    v1f = _to_float("v1", v1)
    v2f = _to_float("v2", v2)
    in1_n = _norm_in(in1)
    in2_n = _norm_in(in2)

    out_raw = str(out)
    out_base, icomp = _parse_component_suffix(out_raw)
    out_n = _norm_out(out_base)
    key = _AS_SPECIAL_OUT.get(out_n.lower(), out_n)

    backend, f = _split_backend_and_fluid(f_full)

    _, get_parameter_index, generate_update_pair = _import_coolprop_state()
    AS = _as_cached_state(backend, f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            k1 = get_parameter_index(in1_n)
            k2 = get_parameter_index(in2_n)
            pair, a1, a2 = generate_update_pair(k1, v1f, k2, v2f)
            AS.update(pair, a1, a2)
        except Exception as e:
            raise _wrap_call_error(
                what="AbstractState.update",
                out=None,
                out_raw=None,
                in1=in1_n,
                v1=v1f,
                in2=in2_n,
                v2=v2f,
                fluid=f_full,
                exc=e,
            ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if key in _AS_SPECIAL_KEYS:
                if key == "FUGACITY":
                    y = AS.fugacity(icomp)
                elif key == "FUGACITY_COEFFICIENT":
                    y = AS.fugacity_coefficient(icomp)
                elif key == "LN_FUGACITY_COEFFICIENT":
                    y = math.log(float(AS.fugacity_coefficient(icomp)))
                elif key == "CHEMICAL_POTENTIAL":
                    y = AS.chemical_potential(icomp)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported AbstractState special output: {key!r}")
            else:
                kout = get_parameter_index(key)
                y = AS.keyed_output(kout)
            y = float(y)
        except Exception as e:
            raise _wrap_call_error(
                what="AbstractState.output",
                out=str(key),
                out_raw=out_raw,
                in1=in1_n,
                v1=v1f,
                in2=in2_n,
                v2=v2f,
                fluid=f_full,
                exc=e,
            ) from e

    if not _finite(y):
        raise CoolPropCallError(
            "CoolProp returned a non-finite result (AbstractState).\n"
            f"  out={key!r} (from {out_raw!r}), in1={in1_n!r}, v1={v1f}, in2={in2_n!r}, v2={v2f}, fluid={f_full!r}\n"
            f"  y={y!r}"
        )
    return float(y)

def as_props_multi(outputs: Sequence[str], in1: str, v1: float, in2: str, v2: float, fluid: str) -> dict[str, float]:
    return {str(k): as_props_si(str(k), in1, v1, in2, v2, fluid) for k in outputs}

def batch_as_props(calls: Iterable[ASCall]) -> list[float]:
    ys: list[float] = []
    for c in _iter_calls_generic(calls, ASCall, 6):
        ys.append(as_props_si(c.out, c.in1, float(c.v1), c.in2, float(c.v2), c.fluid))
    return ys

def FugacitySI(in1: str, v1: float, in2: str, v2: float, fluid: str, i: int = 0) -> float:
    return as_props_si(f"fugacity[{int(i)}]" if int(i) != 0 else "fugacity", in1, v1, in2, v2, fluid)

def FugacityCoeffSI(in1: str, v1: float, in2: str, v2: float, fluid: str, i: int = 0) -> float:
    return as_props_si(f"phi[{int(i)}]" if int(i) != 0 else "phi", in1, v1, in2, v2, fluid)

def LnFugacityCoeffSI(in1: str, v1: float, in2: str, v2: float, fluid: str, i: int = 0) -> float:
    return as_props_si(f"ln_phi[{int(i)}]" if int(i) != 0 else "ln_phi", in1, v1, in2, v2, fluid)

def ChemicalPotentialSI(in1: str, v1: float, in2: str, v2: float, fluid: str, i: int = 0) -> float:
    return as_props_si(
        f"chemical_potential[{int(i)}]" if int(i) != 0 else "chemical_potential",
        in1, v1, in2, v2, fluid
    )

# ------------------------------ HAPropsSI humid air ------------------------------

def haprops_si(out: str, in1: str, v1: float, in2: str, v2: float, in3: str, v3: float) -> float:
    v1f = _to_float("v1", v1)
    v2f = _to_float("v2", v2)
    v3f = _to_float("v3", v3)

    out_raw = str(out)
    out_n = _ha_norm_out(out_raw)
    in1_n = _ha_norm_in(in1)
    in2_n = _ha_norm_in(in2)
    in3_n = _ha_norm_in(in3)

    HAPropsSI_fn = _import_coolprop_ha()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y = float(HAPropsSI_fn(out_n, in1_n, v1f, in2_n, v2f, in3_n, v3f))
        except Exception as e:
            raise _wrap_ha_call_error(
                what="HAPropsSI",
                out=out_n,
                out_raw=out_raw,
                in1=in1_n,
                v1=v1f,
                in2=in2_n,
                v2=v2f,
                in3=in3_n,
                v3=v3f,
                exc=e,
            ) from e

    if not _finite(y):
        raise CoolPropCallError(
            "CoolProp returned a non-finite result (HAPropsSI).\n"
            f"  out={out_n!r}, in1={in1_n!r}, v1={v1f}, in2={in2_n!r}, v2={v2f}, in3={in3_n!r}, v3={v3f}\n"
            f"  y={y!r}"
        )
    return float(y)

def haprops_multi(outputs: Sequence[str], in1: str, v1: float, in2: str, v2: float, in3: str, v3: float) -> dict[str, float]:
    return {str(k): haprops_si(str(k), in1, v1, in2, v2, in3, v3) for k in outputs}

def batch_haprops(calls: Iterable[HACall]) -> list[float]:
    ys: list[float] = []
    for c in _iter_calls_generic(calls, HACall, 7):
        ys.append(haprops_si(c.out, c.in1, float(c.v1), c.in2, float(c.v2), c.in3, float(c.v3)))
    return ys

# back-compat aliases
def ha_props_si(out: str, in1: str, v1: float, in2: str, v2: float, in3: str, v3: float) -> float:
    return haprops_si(out, in1, v1, in2, v2, in3, v3)

def ha_props_multi(outputs: Sequence[str], in1: str, v1: float, in2: str, v2: float, in3: str, v3: float) -> dict[str, float]:
    return haprops_multi(outputs, in1, v1, in2, v2, in3, v3)

def batch_ha_props(calls: Iterable[HACall]) -> list[float]:
    return batch_haprops(calls)

# ------------------------------ CoolProp-like shims ------------------------------

def PropsSI(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    return props_si(out, in1, v1, in2, v2, fluid)

def PhaseSI(in1: str, v1: float, in2: str, v2: float, fluid: str) -> str:
    return phase_si(in1, v1, in2, v2, fluid)

def HAPropsSI(out: str, in1: str, v1: float, in2: str, v2: float, in3: str, v3: float) -> float:
    return haprops_si(out, in1, v1, in2, v2, in3, v3)

def ASPropsSI(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    return as_props_si(out, in1, v1, in2, v2, fluid)

def CTPropsSI(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    return ctprops_si(out, in1, v1, in2, v2, fluid)
