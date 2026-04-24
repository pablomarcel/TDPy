# equations/solver.py
from __future__ import annotations

"""
equations.solver

Backends:
- SciPy: robust general-purpose nonlinear solvers (scipy.optimize.root)
- GEKKO: EES-like variable handling / guessing + nonlinear solve (if installed)

Design goals:
- CLI-friendly, JSON-driven, deterministic
- "EES-ish" square rule: (#equations) == (#unknowns)
- Strong validation + good error messages
- Safe parsing of equation strings (via safe_eval)

Key security note:
- GEKKO backend uses Python eval() on *validated* expressions with __builtins__={}
  and a strict AST whitelist. We MUST keep a whitelist — do not allow arbitrary
  Python functions / attribute access / subscripts. Add only math-like functions.

Upgrades / behavior:
- Warm-start prepass: evaluates simple assignments lhs = rhs to improve initial guesses
  (including PropsSI/HAPropsSI/LiBrPropsSI/NH3H2O* for SciPy systems)
- Adds SciPy root() methods: krylov, anderson, broyden1 (plus existing)
- Robustly reads solve: {...} even when spec is a dict (JSON-loaded mapping)
- Adds HAPropsSI(...) support (humid-air thermodynamic_properties) in SciPy safe_eval scope
- Adds LiBr–H2O support in SciPy safe_eval scope (native backend)
- Adds NH3–H2O (ammonia-water) support in SciPy safe_eval scope (native backend)
- AUTO backend routing:
    any thermo call usage forces SciPy (GEKKO cannot support thermo calls)

Critical bugfix vs older versions:
- Do NOT swallow HAPropsSI evaluation errors and re-raise them as "no provider available".
  Provider-not-available is ONLY for missing imports. Domain/range errors must propagate
  (or be penalized if thermo_penalty is enabled).
- Adds a light “auto-guess” heuristic when an unknown has no explicit guess (default=1.0)
  so thermo calls don’t immediately blow up at the initial point (e.g., T* -> 300 K).
- Optional thermo-domain penalty handling inside SciPy residual evaluation:
    if a thermo call fails at some iterate, return a large finite residual for that equation
    so SciPy can move away from the invalid region (instead of hard-crashing).

Notes:
- SciPy root() does not support bounds; bounds are accepted in the spec for future solvers
  and are ignored here (with meta flag bounds_ignored=True).
"""

from dataclasses import dataclass
import importlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .safe_eval import (
    ParseError,
    UnsafeExpressionError,
    compile_expression,
    compile_residual,
    eval_compiled,
    eval_expression,
    normalize_equation_to_residual,
    preprocess_expr,
    split_assignment,
)

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise ImportError("equations.solver requires numpy.") from e

# Optional deps (kept optional at import time)
try:  # SciPy backend
    from scipy.optimize import root as scipy_root  # type: ignore
except Exception:  # pragma: no cover
    scipy_root = None


# ------------------------------ results ------------------------------

@dataclass(frozen=True)
class SolveResult:
    ok: bool
    backend: str
    method: str
    message: str
    nfev: int
    variables: Dict[str, Any]
    residuals: List[float]
    residual_norm: float
    meta: Dict[str, Any]


# ------------------------------ tiny internal adapter (optional) ------------------------------

@dataclass
class _SimpleVar:
    """Minimal var-like for fallback adaptation when caller passes JSON-ish variable mappings."""
    name: str
    kind: str  # "unknown"|"fixed"
    value: float | None
    guess: float | None = None
    lower: float | None = None
    upper: float | None = None


# ------------------------------ spec helpers (dict OR object) ------------------------------

def _spec_get(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _spec_has(spec: Any, key: str) -> bool:
    if isinstance(spec, Mapping):
        return key in spec
    return hasattr(spec, key)


# ------------------------------ small config helpers ------------------------------

def _get_solve_block(spec: Any) -> Dict[str, Any]:
    """
    Pull `solve: {...}` mapping if present on the spec object.

    Robust to multiple shapes:
      - mapping spec: {"solve": {...}}
      - mapping solve: {"backend":"scipy","tol":1e-6,...}
      - dataclass/object with attributes: spec.solve.backend, spec.solve.tol, ...
      - object with to_dict()/dict() method
    """
    s = _spec_get(spec, "solve", None)
    if s is None:
        return {}

    if isinstance(s, Mapping):
        return dict(s)

    # Has to_dict()
    to_dict = getattr(s, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            if isinstance(d, Mapping):
                return dict(d)
        except Exception:
            pass

    # Try dict(s) if it is dict-like
    try:
        d2 = dict(s)  # type: ignore[arg-type]
        if isinstance(d2, Mapping) and d2:
            return dict(d2)
    except Exception:
        pass

    # Attribute-based extraction
    out: Dict[str, Any] = {}
    for key in (
        "backend",
        "solver",
        "method",
        "tol",
        "rtol",
        "atol",
        "max_iter",
        "maxiter",
        "max_restarts",
        "restarts",
        "warm_start",
        "warm_start_mode",
        "warm_start_passes",
        "options",
        "scipy_options",
        "gekko_solver",
        "thermo_penalty",
        "auto_guess",
    ):
        if hasattr(s, key):
            try:
                out[key] = getattr(s, key)
            except Exception:
                pass
    return out


def _pick_first(*vals: Any) -> Any:
    """Return the first value that is not None."""
    for v in vals:
        if v is not None:
            return v
    return None


_BACKEND_ALIASES: Dict[str, str] = {
    "": "",
    "none": "",
    "auto": "auto",
    "scipy": "scipy",
    "root": "scipy",
    "scipy-root": "scipy",
    "optimize": "scipy",
    "gekko": "gekko",
    "ipopt": "gekko",
    "apopt": "gekko",
}


def _normalize_backend_name(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = _BACKEND_ALIASES.get(s, s)
    if s in {"", "none"}:
        return ""
    return s


def _normalize_method_name(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower().replace("_", "-")


# Map “EES-ish / user-ish” names to SciPy root methods.
_SCIPY_METHOD_ALIASES: Dict[str, str] = {
    # classic
    "newton": "hybr",
    "hybrid": "hybr",
    "hybrid-newton": "hybr",
    "minpack": "hybr",

    # LM
    "levenberg-marquardt": "lm",
    "levenberg": "lm",
    "marquardt": "lm",

    # Broyden
    "broyden": "broyden1",
    "broyden-1": "broyden1",
    "broyden1": "broyden1",
    "broyden-2": "broyden2",
    "broyden2": "broyden2",

    # New additions
    "krylov": "krylov",
    "anderson": "anderson",

    # DF-SANE
    "df-sane": "df-sane",
    "dfsane": "df-sane",

    # misc
    "linearmixing": "linearmixing",
    "linear-mixing": "linearmixing",
    "diagbroyden": "diagbroyden",
    "diag-broyden": "diagbroyden",
    "excitingmixing": "excitingmixing",
    "exciting-mixing": "excitingmixing",
}


def _normalize_scipy_method(method: str) -> str:
    m = _normalize_method_name(method)
    if not m:
        return "hybr"
    return _SCIPY_METHOD_ALIASES.get(m, m)


_VALID_SCIPY_METHODS: set[str] = {
    "hybr",
    "lm",
    "broyden1",
    "broyden2",
    "anderson",
    "krylov",
    "df-sane",
    "linearmixing",
    "diagbroyden",
    "excitingmixing",
}


def _scipy_root_options(
    method: str,
    max_iter: int,
    user_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build SciPy root() options without triggering OptimizeWarning.

    Important method-specific behavior:
      - 'hybr' (MINPACK) uses 'maxfev' not 'maxiter'
      - 'df-sane' uses 'maxfev' not 'maxiter'
      - 'lm' uses 'maxiter'
      - most other methods use 'maxiter'
    """
    m = _normalize_scipy_method(method)

    if m == "hybr":
        base: Dict[str, Any] = {"maxfev": int(max_iter)}
    elif m in {"df-sane"}:
        base = {"maxfev": int(max_iter)}
    elif m == "lm":
        base = {"maxiter": int(max_iter)}
    else:
        base = {"maxiter": int(max_iter)}

    if user_options and isinstance(user_options, Mapping):
        for k, v in user_options.items():
            if v is None:
                continue
            base[str(k)] = v

    return base


# ------------------------------ equation + IO helpers ------------------------------

def _coerce_equation_strings(eqs: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for e in eqs:
        if isinstance(e, str):
            out.append(e)
            continue
        # allow EquationSpec-like objects (kind/expr)
        kind = getattr(e, "kind", None)
        if kind is not None:
            k = str(kind)
            if k == "expr":
                expr = getattr(e, "expr", None)
                if expr is None:
                    raise ValueError("EquationSpec(kind='expr') has no expr.")
                out.append(str(expr))
                continue
            if k == "residual":
                raise NotImplementedError(
                    "EquationSpec(kind='residual') is not supported in solver.py. "
                    "Use kind='expr' with an explicit equation string."
                )
        out.append(str(e))
    return out


def _is_primitive_param(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool))


def _extract_params(spec: Any) -> Dict[str, Any]:
    """
    Pull constants/params from multiple spec shapes:

    - spec.params (adapter-preferred)
    - spec.constants (JSON)
    - spec.params mapping may contain ParamSpec-like objects with .value/.name
    """
    params_any = _spec_get(spec, "params", None)
    const_any = _spec_get(spec, "constants", None)

    merged: Dict[str, Any] = {}

    def ingest(mapping: Any) -> None:
        if not isinstance(mapping, Mapping):
            return
        for k, v in mapping.items():
            key = str(k)
            if _is_primitive_param(v):
                merged[key] = v
                continue
            nm = getattr(v, "name", key)
            vv = getattr(v, "value", None)
            if _is_primitive_param(vv):
                merged[str(nm)] = vv
                continue

    ingest(params_any or {})
    ingest(const_any or {})
    return merged


def _extract_variables(spec: Any) -> List[Any]:
    """
    Preferred (adapter):
      spec.variables : list[var-like objects]

    Raw JSON-friendly support:
      spec.variables : mapping of name -> dict {guess,bounds,unit,value,...}
      spec.variables : list[dict] with keys {name, guess, value, lower, upper, ...}
      spec.vars      : mapping (older style)
    """
    vars_any = _spec_get(spec, "variables", None)

    if isinstance(vars_any, (list, tuple)) and list(vars_any):
        return list(vars_any)

    if isinstance(vars_any, Mapping) and vars_any:
        return _vars_mapping_to_list(vars_any)

    vars_map = _spec_get(spec, "vars", None)
    if isinstance(vars_map, Mapping) and vars_map:
        tmp: List[_SimpleVar] = []
        for name, v in vars_map.items():
            nm = str(getattr(v, "name", name))
            fixed = bool(getattr(v, "fixed", False))
            value = getattr(v, "value", None)
            lower = getattr(v, "lower", None)
            upper = getattr(v, "upper", None)
            if fixed:
                if value is None:
                    raise ValueError(f"Fixed variable {nm!r} must have a value.")
                tmp.append(_SimpleVar(name=nm, kind="fixed", value=float(value), guess=None, lower=lower, upper=upper))
            else:
                guess = 1.0 if value is None else float(value)
                tmp.append(_SimpleVar(name=nm, kind="unknown", value=None, guess=guess, lower=lower, upper=upper))
        return tmp

    return []


def _extract_system(spec: Any) -> Tuple[List[str], List[Any], Dict[str, Any]]:
    eqs_raw = list(_spec_get(spec, "equations", []) or [])
    if not eqs_raw:
        raise ValueError("Equation system has no equations.")
    eqs = _coerce_equation_strings(eqs_raw)

    vars_list = _extract_variables(spec)
    if not vars_list:
        raise ValueError("Equation system has no variables.")

    params = _extract_params(spec)
    return eqs, vars_list, params


# ------------------------------ units helpers (optional but very useful) ------------------------------

try:
    from units import DEFAULT_REGISTRY, UnitError, parse_quantity  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_REGISTRY = None
    UnitError = Exception  # type: ignore
    parse_quantity = None  # type: ignore


def _convert_value_with_unit(v: Any, unit: str | None) -> Any:
    """
    Convert numeric-like values into registry base units when a unit token is provided.
    Leaves non-numeric values untouched.
    """
    if unit is None or DEFAULT_REGISTRY is None:
        return v

    u = str(unit).strip()
    if not u:
        return v

    if isinstance(v, str):
        s = v.strip()
        if parse_quantity is not None:
            try:
                q = parse_quantity(s, DEFAULT_REGISTRY)
                return q.base_value()
            except Exception:
                pass
        try:
            x = float(s)
            return float(DEFAULT_REGISTRY.to_base(x, u))
        except Exception:
            return v

    if isinstance(v, (int, float)):
        try:
            return float(DEFAULT_REGISTRY.to_base(float(v), u))
        except Exception:
            return v

    return v


def _vars_mapping_to_list(m: Mapping[str, Any]) -> List[_SimpleVar]:
    out: List[_SimpleVar] = []
    for name, spec in m.items():
        nm = str(name)
        if isinstance(spec, Mapping):
            unit = spec.get("unit", None)

            if "value" in spec and spec.get("value", None) is not None:
                value = _convert_value_with_unit(spec.get("value"), unit)
                try:
                    fv = float(value)
                except Exception:
                    raise ValueError(f"Fixed variable {nm!r} value must be numeric; got {value!r}")
                out.append(_SimpleVar(name=nm, kind="fixed", value=fv))
                continue

            guess_raw = spec.get("guess", None)
            if guess_raw is None:
                guess_raw = 1.0

            guess_conv = _convert_value_with_unit(guess_raw, unit)
            try:
                guess = float(guess_conv)
            except Exception:
                raise ValueError(f"Unknown variable {nm!r} guess must be numeric or 'value unit'; got {guess_raw!r}")

            lower = None
            upper = None
            b = spec.get("bounds", None)
            if isinstance(b, (list, tuple)) and len(b) == 2:
                lower = _convert_value_with_unit(b[0], unit)
                upper = _convert_value_with_unit(b[1], unit)
                lower = float(lower) if lower is not None else None
                upper = float(upper) if upper is not None else None

            out.append(_SimpleVar(name=nm, kind="unknown", value=None, guess=guess, lower=lower, upper=upper))
        else:
            try:
                guess = float(spec)
            except Exception:
                raise ValueError(f"Variable {nm!r} spec must be mapping or numeric; got {spec!r}")
            out.append(_SimpleVar(name=nm, kind="unknown", value=None, guess=guess))
    return out


# ------------------------------ import helper ------------------------------

def _import_first(modnames: Sequence[str]) -> tuple[Any | None, List[str]]:
    """
    Try to import modules in order; return (module, errors).
    Errors are strings with modname + exception summary.
    """
    errs: List[str] = []
    for mn in modnames:
        try:
            return importlib.import_module(mn), errs
        except Exception as e:
            errs.append(f"{mn}: {type(e).__name__}: {e}")
    return None, errs


# ------------------------------ thermo injection (SciPy only) ------------------------------

def _safe_propssi(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any) -> float:
    """
    Safe PropsSI wrapper used ONLY inside safe_eval scope for SciPy residual evaluation / warm-start.
    """
    try:
        from thermo_props.coolprop_backend import props_si  # local import
    except Exception as e:
        raise ImportError(
            "thermo PropsSI(...) requested but thermo_props is not available. "
            "Ensure thermo_props is available in your environment."
        ) from e

    try:
        fv1 = float(v1)
        fv2 = float(v2)
    except Exception as e:
        raise ValueError(f"PropsSI inputs must be numeric for v1/v2; got v1={v1!r}, v2={v2!r}") from e

    return float(props_si(str(out), str(in1), fv1, str(in2), fv2, str(fluid)))


def _safe_ctpropssi(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any) -> float:
    """
    Safe CTPropsSI wrapper used ONLY inside safe_eval scope for SciPy residual evaluation / warm-start.

    Signature mirrors CoolProp PropsSI-style calls:
        CTPropsSI(out, in1, v1, in2, v2, fluid)

    This is independent from CoolProp and is provided by thermo_props.cantera_backend.
    """
    try:
        from thermo_props.cantera_backend import ctprops_si  # local import
    except Exception as e:
        raise ImportError(
            "thermo CTPropsSI(...) requested but thermo_props.cantera_backend is not available. "
            "Install Cantera and ensure cantera_backend.py is present."
        ) from e

    try:
        fv1 = float(v1)
        fv2 = float(v2)
    except Exception as e:
        raise ValueError(f"CTPropsSI inputs must be numeric for v1/v2; got v1={v1!r}, v2={v2!r}") from e

    return float(ctprops_si(str(out), str(in1), fv1, str(in2), fv2, str(fluid)))


def _safe_ctprops_multi(*args: Any, **kwargs: Any) -> Any:
    # If users try to use a batch/multi-output call inside a scalar equation, fail loudly.
    raise ValueError(
        "CTPropsMulti/CTBatchProps are not valid inside a scalar equation expression. "
        "Use CTPropsSI(...) to return a single numeric value per call."
    )


def _safe_cantera_available() -> float:
    """
    Numeric-friendly wrapper for cantera_available().

    Returns:
      1.0 if Cantera backend is importable/usable, else 0.0
    """
    try:
        from thermo_props.cantera_backend import cantera_available  # local import
        return 1.0 if bool(cantera_available()) else 0.0
    except Exception:
        return 0.0


def _maybe_add_ct_cache_meta(meta: Dict[str, Any]) -> None:
    """Best-effort attach CTPropsSI cache stats into solver meta.

    This is purely diagnostic and should never affect solve behavior.
    """
    try:
        from thermo_props import cantera_backend as cb  # local import

        info = getattr(cb, "ctprops_cache_info", None)
        if callable(info):
            # Example shape: {"ctprops_si": {"hits":..., "misses":..., "size":..., ...}, ...}
            meta.setdefault("ctprops_cache", info())
    except Exception:
        return



def _safe_aspropssi(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any) -> float:
    """
    Safe AbstractState-backed ASPropsSI wrapper used ONLY inside safe_eval scope for SciPy evaluation.

    Signature mirrors PropsSI-style calls:
        ASPropsSI(out, in1, v1, in2, v2, fluid)
    """
    try:
        from thermo_props.coolprop_backend import as_props_si  # local import
    except Exception as e:
        raise ImportError(
            "thermo ASPropsSI(...) requested but thermo_props is not available. "
            "Ensure thermo_props is available in your environment."
        ) from e

    try:
        fv1 = float(v1)
        fv2 = float(v2)
    except Exception as e:
        raise ValueError(f"ASPropsSI inputs must be numeric for v1/v2; got v1={v1!r}, v2={v2!r}") from e

    return float(as_props_si(str(out), str(in1), fv1, str(in2), fv2, str(fluid)))


def _safe_fugacitycoeffsi(in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any, i: Any = 0) -> float:
    try:
        ii = int(i)
    except Exception as e:
        raise ValueError(f"FugacityCoeffSI component index must be an int; got {i!r}") from e
    out = "phi" if ii == 0 else f"phi[{ii}]"
    return _safe_aspropssi(out, in1, v1, in2, v2, fluid)


def _safe_lnfugacitycoeffsi(in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any, i: Any = 0) -> float:
    try:
        ii = int(i)
    except Exception as e:
        raise ValueError(f"LnFugacityCoeffSI component index must be an int; got {i!r}") from e
    out = "ln_phi" if ii == 0 else f"ln_phi[{ii}]"
    return _safe_aspropssi(out, in1, v1, in2, v2, fluid)


def _safe_fugacitysi(in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any, i: Any = 0) -> float:
    try:
        ii = int(i)
    except Exception as e:
        raise ValueError(f"FugacitySI component index must be an int; got {i!r}") from e
    out = "fugacity" if ii == 0 else f"fugacity[{ii}]"
    return _safe_aspropssi(out, in1, v1, in2, v2, fluid)


def _safe_chemicalpotentialsi(in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any, i: Any = 0) -> float:
    try:
        ii = int(i)
    except Exception as e:
        raise ValueError(f"ChemicalPotentialSI component index must be an int; got {i!r}") from e
    out = "chemical_potential" if ii == 0 else f"chemical_potential[{ii}]"
    return _safe_aspropssi(out, in1, v1, in2, v2, fluid)

def _safe_phasesi(in1: Any, v1: Any, in2: Any, v2: Any, fluid: Any) -> float:
    """
    Safe PhaseSI wrapper.

    CoolProp PhaseSI returns a phase STRING. Since safe_eval requires numeric outputs
    (float coercion), we map common phases to numeric codes:

      - gas/vapor/supercritical_gas : +2
      - liquid/subcooled_liquid     : +1
      - two_phase/twophase          :  0
      - supercritical               : +3
      - unknown/other               : -1

    This is primarily intended for diagnostics / guard equations.
    """
    try:
        # Prefer our wrapper if present
        from thermo_props import coolprop_backend as cb  # local import
        f = getattr(cb, "phase_si", None)
        if callable(f):
            ph = str(f(str(in1), float(v1), str(in2), float(v2), str(fluid))).strip().lower()
        else:
            from CoolProp.CoolProp import PhaseSI as _PhaseSI  # type: ignore
            ph = str(_PhaseSI(str(in1), float(v1), str(in2), float(v2), str(fluid))).strip().lower()
    except ImportError as e:
        raise ImportError(
            "thermo PhaseSI(...) requested but no provider is available. "
            "Install CoolProp or provide thermo_props.coolprop_backend.phase_si."
        ) from e
    except Exception as e:
        raise ValueError(f"PhaseSI call failed: {e}") from e

    # normalize
    ph = ph.replace(" ", "_")
    if "supercritical" in ph and "gas" in ph:
        return 2.0
    if ph in {"gas", "vapor", "vapour", "superheated", "superheated_gas"} or "gas" in ph:
        return 2.0
    if ph in {"liquid", "subcooled_liquid", "compressed_liquid"} or "liquid" in ph:
        return 1.0
    if "two" in ph and "phase" in ph:
        return 0.0
    if "supercritical" in ph:
        return 3.0
    return -1.0


def _format_ha_ctx(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, in3: Any, v3: Any) -> str:
    return f"HAPropsSI({out!r},{in1!r},{v1!r},{in2!r},{v2!r},{in3!r},{v3!r})"


def _safe_hapropssi(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, in3: Any, v3: Any) -> float:
    """
    Safe HAPropsSI wrapper (humid-air thermodynamic_properties) used ONLY inside safe_eval scope for SciPy residual
    evaluation / warm-start.

    IMPORTANT:
    - Only raise ImportError when no provider can be imported.
    - If provider is installed and call fails due to invalid inputs (domain/range),
      propagate that error with context.
    """
    try:
        fv1 = float(v1)
        fv2 = float(v2)
        fv3 = float(v3)
    except Exception as e:
        raise ValueError(
            "HAPropsSI inputs must be numeric for v1/v2/v3; "
            f"got v1={v1!r}, v2={v2!r}, v3={v3!r}"
        ) from e

    # Preferred: our backend wrapper
    try:
        from thermo_props import coolprop_backend as cb  # local import

        f = getattr(cb, "ha_props_si", None) or getattr(cb, "haprops_si", None)
        if callable(f):
            return float(f(str(out), str(in1), fv1, str(in2), fv2, str(in3), fv3))
    except ImportError:
        pass

    # Fallback: direct CoolProp import.
    try:
        from CoolProp.CoolProp import HAPropsSI as _HAPropsSI  # type: ignore
    except Exception as e:
        raise ImportError(
            "thermo HAPropsSI(...) requested but no provider is available. "
            "Install CoolProp, or ensure thermo_props.coolprop_backend provides ha_props_si()/haprops_si()."
        ) from e

    try:
        return float(_HAPropsSI(str(out), str(in1), fv1, str(in2), fv2, str(in3), fv3))
    except Exception as e:
        raise ValueError(f"HAPropsSI call failed for {_format_ha_ctx(out, in1, fv1, in2, fv2, in3, fv3)}: {e}") from e


# ------------------------------ LiBr–H2O injection (SciPy only) ------------------------------

def _format_libr_ctx(out: Any, pairs: List[Tuple[str, float]]) -> str:
    p = ",".join([f"{k!r},{v!r}" for k, v in pairs])
    return f"LiBrPropsSI({out!r},{p})"


def _import_libr_provider() -> Any:
    """
    Import the LiBr–H2O provider module with a few fallback names.
    """
    mod, errs = _import_first(
        [
            "thermo_props.librh2o_backend",
            "thermo_props.librh2o_ashrae_backend",
            "thermo_props.librh2o",
        ]
    )
    if mod is None:
        raise ImportError(
            "thermo LiBrPropsSI(...) requested but no LiBr–H2O backend is importable. "
            "Expected a module like thermo_props.librh2o_backend (or librh2o_ashrae_backend). "
            f"Import attempts: {errs}"
        )
    return mod


def _safe_librpropssi(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, *rest: Any) -> float:
    """
    Safe LiBrPropsSI wrapper (LiBr–H2O property engine) used ONLY inside safe_eval scope for SciPy.

    Supported calling shapes (flexible):
      - LiBrPropsSI(out, in1, v1, in2, v2)
      - LiBrPropsSI(out, in1, v1, in2, v2, in3, v3, ...)
    """
    # Parse into (key,value) pairs
    pairs: List[Tuple[str, float]] = []
    try:
        pairs.append((str(in1), float(v1)))
        pairs.append((str(in2), float(v2)))
    except Exception as e:
        raise ValueError(f"LiBrPropsSI inputs must be numeric for values; got v1={v1!r}, v2={v2!r}") from e

    if rest:
        if len(rest) % 2 != 0:
            raise ValueError(
                "LiBrPropsSI extra args must be in (key,value) pairs. "
                f"Got odd-length tail: {rest!r}"
            )
        it = iter(rest)
        for k, v in zip(it, it):
            try:
                pairs.append((str(k), float(v)))
            except Exception as e:
                raise ValueError(f"LiBrPropsSI extra pair value must be numeric: {k!r},{v!r}") from e

    lb = _import_libr_provider()

    # Try a few plausible provider names for robustness across refactors.
    f = (
        getattr(lb, "librh2o_props_si", None)
        or getattr(lb, "props_si", None)
        or getattr(lb, "LiBrPropsSI", None)
        or getattr(lb, "libr_props_si", None)
    )
    if not callable(f):
        raise ImportError(
            "LiBr–H2O backend imported, but no callable LiBr props function was found. "
            "Expected one of: librh2o_props_si, props_si, LiBrPropsSI, libr_props_si."
        )

    # Preferred call signature: (out, in1, v1, in2, v2, ...)
    try:
        flat: List[Any] = []
        for k, v in pairs:
            flat.extend([k, v])
        return float(f(str(out), *flat))
    except TypeError:
        # Alternate signature: (out, flat_pairs_list)
        try:
            flat2: List[Any] = []
            for k, v in pairs:
                flat2.extend([k, v])
            return float(f(str(out), flat2))
        except Exception as ex2:
            raise ValueError(f"LiBrPropsSI call failed for {_format_libr_ctx(out, pairs)}: {ex2}") from ex2
    except Exception as ex:
        raise ValueError(f"LiBrPropsSI call failed for {_format_libr_ctx(out, pairs)}: {ex}") from ex


def _safe_librprops_multi(*args: Any, **kwargs: Any) -> Any:
    # If users try to use a batch/multi-output call inside a scalar equation, fail loudly.
    raise ValueError(
        "LiBrPropsMulti/LiBrBatchProps are not valid inside a scalar equation expression. "
        "Use LiBrPropsSI(...) to return a single numeric value per call."
    )


def _safe_libr_helper(fn_name: str, *args: Any) -> float:
    """
    Call a LiBr helper function from the provider module.

    Expected helper names:
      - LiBrX_TP(T, P)          -> X
      - LiBrH_TX(T, X)          -> H
      - LiBrRho_TXP(T, X, P)    -> rho
      - LiBrT_HX(H, X)          -> T
    """
    lb = _import_libr_provider()
    f = getattr(lb, fn_name, None)
    if not callable(f):
        raise ImportError(
            f"LiBr helper {fn_name}(...) was requested but is not implemented in the LiBr backend. "
            f"Backend module: {getattr(lb, '__name__', '<??>')}. "
            f"Expected a callable named {fn_name}."
        )
    try:
        return float(f(*args))
    except Exception as e:
        raise ValueError(f"{fn_name}(...) call failed: {e}") from e


def _safe_LiBrX_TP(T: Any, P: Any) -> float:
    return _safe_libr_helper("LiBrX_TP", float(T), float(P))


def _safe_LiBrH_TX(T: Any, X: Any) -> float:
    return _safe_libr_helper("LiBrH_TX", float(T), float(X))


def _safe_LiBrRho_TXP(T: Any, X: Any, P: Any) -> float:
    return _safe_libr_helper("LiBrRho_TXP", float(T), float(X), float(P))


def _safe_LiBrT_HX(H: Any, X: Any) -> float:
    return _safe_libr_helper("LiBrT_HX", float(H), float(X))


# ------------------------------ NH3–H2O (ammonia-water) injection (SciPy only) ------------------------------

def _format_nh3_ctx(name: str, out: Any, pairs: List[Tuple[str, float]]) -> str:
    p = ",".join([f"{k!r},{v!r}" for k, v in pairs])
    return f"{name}({out!r},{p})"


def _import_nh3_provider() -> Any:
    """
    Import NH3–H2O provider module with a few fallback names.
    """
    mod, errs = _import_first(
        [
            "thermo_props.nh3h2o_backend",
            "thermo_props.ammonia_water_backend",
            "thermo_props.ammonia_water",
        ]
    )
    if mod is None:
        raise ImportError(
            "thermo NH3–H2O call requested but no NH3H2O backend is importable. "
            "Expected a module like thermo_props.nh3h2o_backend (or ammonia_water_backend). "
            f"Import attempts: {errs}"
        )
    return mod


def _safe_nh3h2o_propssi(out: Any, in1: Any, v1: Any, in2: Any, v2: Any, *rest: Any) -> float:
    """
    Safe NH3–H2O property wrapper used ONLY inside safe_eval scope for SciPy.

    Supported call shapes (flexible):
      - NH3H2OPropsSI(out, in1, v1, in2, v2)
      - NH3H2OPropsSI(out, in1, v1, in2, v2, in3, v3, ...)
      - NH3H2O_TPX(out, "T", T, "P", P, "X", X)  (alias)
      - prop_tpx / state_tpx aliases map here (single-scalar returns only)

    The backend is expected to provide ONE of these callables:
      - nh3h2o_props_si(out, in1, v1, in2, v2, ...)
      - props_si(out, in1, v1, in2, v2, ...)
      - NH3H2OPropsSI(out, in1, v1, in2, v2, ...)
      - ammonia_water_props_si(...)
    """
    # Parse into (key,value) pairs
    pairs: List[Tuple[str, float]] = []
    try:
        pairs.append((str(in1), float(v1)))
        pairs.append((str(in2), float(v2)))
    except Exception as e:
        raise ValueError(f"NH3H2O props inputs must be numeric for values; got v1={v1!r}, v2={v2!r}") from e

    if rest:
        if len(rest) % 2 != 0:
            raise ValueError(
                "NH3H2O props extra args must be in (key,value) pairs. "
                f"Got odd-length tail: {rest!r}"
            )
        it = iter(rest)
        for k, v in zip(it, it):
            try:
                pairs.append((str(k), float(v)))
            except Exception as e:
                raise ValueError(f"NH3H2O extra pair value must be numeric: {k!r},{v!r}") from e

    provider = _import_nh3_provider()

    f = (
        getattr(provider, "nh3h2o_props_si", None)
        or getattr(provider, "props_si", None)
        or getattr(provider, "NH3H2OPropsSI", None)
        or getattr(provider, "ammonia_water_props_si", None)
    )
    if not callable(f):
        raise ImportError(
            "NH3H2O backend imported, but no callable props function was found. "
            "Expected one of: nh3h2o_props_si, props_si, NH3H2OPropsSI, ammonia_water_props_si."
        )

    try:
        flat: List[Any] = []
        for k, v in pairs:
            flat.extend([k, v])
        return float(f(str(out), *flat))
    except TypeError:
        # Alternate signature: (out, flat_pairs_list)
        try:
            flat2: List[Any] = []
            for k, v in pairs:
                flat2.extend([k, v])
            return float(f(str(out), flat2))
        except Exception as ex2:
            raise ValueError(f"NH3H2O props call failed for {_format_nh3_ctx('NH3H2OPropsSI', out, pairs)}: {ex2}") from ex2
    except Exception as ex:
        raise ValueError(f"NH3H2O props call failed for {_format_nh3_ctx('NH3H2OPropsSI', out, pairs)}: {ex}") from ex


def _safe_nh3h2o_multi(*args: Any, **kwargs: Any) -> Any:
    raise ValueError(
        "NH3H2O multi/batch calls are not valid inside a scalar equation expression. "
        "Use NH3H2OPropsSI/NH3H2O_TPX to return a single numeric value per call."
    )


# ------------------------------ numeric function table (SciPy scope) ------------------------------

def _common_numeric_math_funcs() -> Dict[str, Any]:
    """
    Numeric math functions for safe_eval evaluation (SciPy residual evaluation / warm-start).
    Pure numeric (math module) so they are safe in the whitelist model.
    """
    import math

    def _cbrt(x: float) -> float:
        try:
            f = getattr(math, "cbrt", None)
            if callable(f):
                return float(f(x))
        except Exception:
            pass
        return float(np.sign(x) * (abs(x) ** (1.0 / 3.0)))

    def _sign(x: float) -> float:
        return float(np.sign(x))

    def _step(x: float) -> float:
        return 0.0 if x < 0 else 1.0

    return {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
        "sign": _sign,
        "step": _step,

        "sqrt": math.sqrt,
        "cbrt": _cbrt,
        "exp": math.exp,
        "expm1": getattr(math, "expm1", None),
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "log2": getattr(math, "log2", None),
        "log1p": getattr(math, "log1p", None),

        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,

        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,

        "floor": math.floor,
        "ceil": math.ceil,

        "hypot": math.hypot,
        "degrees": math.degrees,
        "radians": math.radians,
        "erf": getattr(math, "erf", None),
        "erfc": getattr(math, "erfc", None),
        "gamma": getattr(math, "gamma", None),
        "lgamma": getattr(math, "lgamma", None),
    }


def _scipy_extra_funcs() -> Dict[str, Any]:
    out = dict(_common_numeric_math_funcs())
    out = {k: v for k, v in out.items() if v is not None}

    # CoolProp + humid air
    out["PropsSI"] = _safe_propssi
    out["HAPropsSI"] = _safe_hapropssi
    # Cantera property engine (independent of CoolProp)
    out["CTPropsSI"] = _safe_ctpropssi
    out["ctprops_si"] = _safe_ctpropssi
    out["CTPropsMulti"] = _safe_ctprops_multi
    out["ctprops_multi"] = _safe_ctprops_multi
    out["CTBatchProps"] = _safe_ctprops_multi
    out["batch_ctprops"] = _safe_ctprops_multi
    out["cantera_available"] = _safe_cantera_available

    out["PhaseSI"] = _safe_phasesi

    # CoolProp AbstractState (fugacity, chemical potential, etc.)
    out["ASPropsSI"] = _safe_aspropssi
    out["as_props_si"] = _safe_aspropssi  # convenience alias
    out["FugacityCoeffSI"] = _safe_fugacitycoeffsi
    out["LnFugacityCoeffSI"] = _safe_lnfugacitycoeffsi
    out["FugacitySI"] = _safe_fugacitysi
    out["ChemicalPotentialSI"] = _safe_chemicalpotentialsi

    # LiBr–H2O property engine + aliases
    out["LiBrPropsSI"] = _safe_librpropssi
    out["LiBrH2OPropsSI"] = _safe_librpropssi
    out["librh2o_props_si"] = _safe_librpropssi  # convenience alias
    out["LiBrPropsMulti"] = _safe_librprops_multi
    out["LiBrBatchProps"] = _safe_librprops_multi
    out["librh2o_props_multi"] = _safe_librprops_multi
    out["batch_librh2o_props"] = _safe_librprops_multi

    # LiBr helper calls frequently used in equation sets
    out["LiBrX_TP"] = _safe_LiBrX_TP
    out["LiBrH_TX"] = _safe_LiBrH_TX
    out["LiBrRho_TXP"] = _safe_LiBrRho_TXP
    out["LiBrT_HX"] = _safe_LiBrT_HX

    # NH3–H2O property engine (native backend) + aliases
    out["NH3H2OPropsSI"] = _safe_nh3h2o_propssi
    out["nh3h2o_props_si"] = _safe_nh3h2o_propssi

    # EES-ish helper names (single-scalar return)
    out["NH3H2O_TPX"] = _safe_nh3h2o_propssi
    out["NH3H2O_STATE_TPX"] = _safe_nh3h2o_propssi
    out["NH3H2O"] = _safe_nh3h2o_propssi
    out["NH3H2O_STATE"] = _safe_nh3h2o_propssi
    out["prop_tpx"] = _safe_nh3h2o_propssi
    out["state_tpx"] = _safe_nh3h2o_propssi

    # Optional multi/batch names (explicitly rejected in scalar expressions)
    out["NH3H2OPropsMulti"] = _safe_nh3h2o_multi
    out["NH3H2OBatchProps"] = _safe_nh3h2o_multi
    out["props_multi_tpx"] = _safe_nh3h2o_multi
    out["batch_prop_tpx"] = _safe_nh3h2o_multi

    return out


# ------------------------------ safe_eval compatibility wrappers ------------------------------

_LBR_FUNC_HINT = (
    "Your equation contains LiBrPropsSI(...), but safe_eval rejected the function name. "
    "Ensure equations.safe_eval is upgraded to allow LiBrPropsSI (or supports extra_funcs)."
)

_NH3_FUNC_HINT = (
    "Your equation contains an NH3–H2O helper call (e.g., NH3H2OPropsSI/NH3H2O_TPX), "
    "but safe_eval rejected the function name. Ensure equations.safe_eval is upgraded "
    "to allow NH3H2O* calls (or supports extra_funcs)."
)

_PHASE_FUNC_HINT = (
    "Your equation contains PhaseSI(...), but safe_eval rejected the function name. "
    "Ensure equations.safe_eval is upgraded to allow PhaseSI (or supports extra_funcs)."
)

_CT_FUNC_HINT = (
    "Your equation contains Cantera CTPropsSI(...) (or CTPropsMulti/CTBatchProps), "
    "but safe_eval rejected the function name. Ensure equations.safe_eval is upgraded "
    "to allow CTProps* calls (or supports extra_funcs)."
)


def _compile_residual_with_extras(expr: str, *, extra_funcs: Mapping[str, Any]) -> Any:
    try:
        return compile_residual(expr, extra_funcs=extra_funcs)  # type: ignore[call-arg]
    except TypeError:
        # older safe_eval without extra_funcs support
        return compile_residual(expr)
    except UnsafeExpressionError as e:
        s = str(e).lower()
        if "librpropssi" in s or "librh2opropssi" in s:
            raise UnsafeExpressionError(f"{e}\n{_LBR_FUNC_HINT}") from e
        if "nh3h2o" in s or "ammonia" in s:
            raise UnsafeExpressionError(f"{e}\n{_NH3_FUNC_HINT}") from e
        if "phasesi" in s:
            raise UnsafeExpressionError(f"{e}\n{_PHASE_FUNC_HINT}") from e
        if "ctprops" in s or "cantera" in s:
            raise UnsafeExpressionError(f"{e}\n{_CT_FUNC_HINT}") from e
        raise


def _eval_compiled_with_extras(
    c: Any,
    *,
    values: Mapping[str, Any],
    params: Mapping[str, Any],
    extra_funcs: Mapping[str, Any],
) -> float:
    try:
        return float(eval_compiled(c, values=values, params=params, extra_funcs=extra_funcs))  # type: ignore[call-arg]
    except TypeError:
        return float(eval_compiled(c, values=values, params=params))


def _compile_expression_with_extras(expr: str, *, extra_funcs: Mapping[str, Any]) -> Any:
    try:
        return compile_expression(expr, extra_funcs=extra_funcs)  # type: ignore[call-arg]
    except TypeError:
        return compile_expression(expr)
    except UnsafeExpressionError as e:
        s = str(e).lower()
        if "librpropssi" in s or "librh2opropssi" in s:
            raise UnsafeExpressionError(f"{e}\n{_LBR_FUNC_HINT}") from e
        if "nh3h2o" in s or "ammonia" in s:
            raise UnsafeExpressionError(f"{e}\n{_NH3_FUNC_HINT}") from e
        if "phasesi" in s:
            raise UnsafeExpressionError(f"{e}\n{_PHASE_FUNC_HINT}") from e
        if "ctprops" in s or "cantera" in s:
            raise UnsafeExpressionError(f"{e}\n{_CT_FUNC_HINT}") from e
        raise


def _eval_expression_with_extras(
    c_or_expr: Any,
    *,
    values: Mapping[str, Any],
    params: Mapping[str, Any],
    extra_funcs: Mapping[str, Any],
) -> Any:
    try:
        return eval_expression(c_or_expr, values=values, params=params, extra_funcs=extra_funcs)  # type: ignore[call-arg]
    except TypeError:
        return eval_expression(c_or_expr, values=values, params=params)


# ------------------------------ thermo detection (AUTO backend routing fix) ------------------------------

_PROPS_CALL_RE = re.compile(r"\bpropssi\s*\(", re.IGNORECASE)
_HAPROPS_CALL_RE = re.compile(r"\bhapropssi\s*\(", re.IGNORECASE)
_PHASE_CALL_RE = re.compile(r"\bphasesi\s*\(", re.IGNORECASE)
_ASPROPS_CALL_RE = re.compile(
    r"\b(aspropssi|as_props_si|fugacitycoeffsi|lnfugacitycoeffsi|fugacitysi|chemicalpotentialsi)\s*\(",
    re.IGNORECASE,
)
_LIBRPROPS_CALL_RE = re.compile(r"\b(libr\w*propssi|librh2o_props_si|librx_tp|librh_tx|librt_hx|librrho_txp)\s*\(", re.IGNORECASE)

# Cantera (CTPropsSI family) call detection.
_CTPROPS_CALL_RE = re.compile(
    r"\b(ctpropssi|ctpropsmulti|ctbatchprops|ctprops_si|ctprops_multi|batch_ctprops)\s*\(",
    re.IGNORECASE,
)


# NH3–H2O call detection: keep broad so user equation styles don't break routing.
_NH3H2O_CALL_RE = re.compile(
    r"\b(nh3h2o\w*|ammonia\w*|prop_tpx|state_tpx)\s*\(",
    re.IGNORECASE,
)


def _equations_require_thermo(eqs: Sequence[str]) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Detect thermo property calls which require numeric callbacks and therefore MUST use SciPy backend
    (GEKKO can't support these).

    Returns: (needs_propssi, needs_hapropssi, needs_phasesi, needs_librpropssi, needs_nh3h2o)
    """
    needs_props = False
    needs_ha = False
    needs_phase = False
    needs_libr = False
    needs_nh3 = False
    for e in eqs:
        s = preprocess_expr(str(e))
        if (not needs_props) and _PROPS_CALL_RE.search(s):
            needs_props = True
        if (not needs_props) and _ASPROPS_CALL_RE.search(s):
            needs_props = True
        if (not needs_props) and _CTPROPS_CALL_RE.search(s):
            needs_props = True
        if (not needs_ha) and _HAPROPS_CALL_RE.search(s):
            needs_ha = True
        if (not needs_phase) and _PHASE_CALL_RE.search(s):
            needs_phase = True
        if (not needs_libr) and _LIBRPROPS_CALL_RE.search(s):
            needs_libr = True
        if (not needs_nh3) and _NH3H2O_CALL_RE.search(s):
            needs_nh3 = True
        if needs_props and needs_ha and needs_phase and needs_libr and needs_nh3:
            break
    return needs_props, needs_ha, needs_phase, needs_libr, needs_nh3


# ------------------------------ GEKKO lazy import (no masking) ------------------------------

_GEKKO_LOADED = False
_GEKKO_CLASS: Any = None
_GEKKO_IMPORT_ERROR: BaseException | None = None
_GEKKO_ORIGIN: str | None = None


def _load_gekko() -> None:
    """Try to import gekko once; cache GEKKO class or the original import error."""
    global _GEKKO_LOADED, _GEKKO_CLASS, _GEKKO_IMPORT_ERROR, _GEKKO_ORIGIN
    if _GEKKO_LOADED:
        return
    _GEKKO_LOADED = True

    try:
        import gekko as g  # type: ignore

        _GEKKO_ORIGIN = getattr(g, "__file__", None)
        GEKKO = getattr(g, "GEKKO", None)
        if GEKKO is None:
            _GEKKO_CLASS = None
            _GEKKO_IMPORT_ERROR = ImportError(
                "Imported a module named 'gekko' but it does not provide 'GEKKO'. "
                f"Imported from: {_GEKKO_ORIGIN or '<?>'} "
                "This usually means a local file/folder named 'gekko' is shadowing the pip package."
            )
            return

        _GEKKO_CLASS = GEKKO
        _GEKKO_IMPORT_ERROR = None

    except BaseException as e:  # pragma: no cover
        try:
            import gekko as g2  # type: ignore
            _GEKKO_ORIGIN = getattr(g2, "__file__", None)
        except Exception:
            _GEKKO_ORIGIN = None
        _GEKKO_CLASS = None
        _GEKKO_IMPORT_ERROR = e


def _gekko_available() -> bool:
    _load_gekko()
    return _GEKKO_CLASS is not None


def _require_gekko() -> Any:
    _load_gekko()
    if _GEKKO_CLASS is not None:
        return _GEKKO_CLASS

    import sys
    err = _GEKKO_IMPORT_ERROR
    if err is None:
        raise ImportError(
            "GEKKO is not available for an unknown reason.\n"
            f"sys.executable: {sys.executable}\n"
            f"sys.version: {sys.version}"
        )

    raise ImportError(
        "Failed to import GEKKO backend.\n"
        f"sys.executable: {sys.executable}\n"
        f"sys.version: {sys.version}\n"
        f"gekko origin: {_GEKKO_ORIGIN or '<?>'}\n"
        f"Original error: {type(err).__name__}: {err}"
    ) from err


# ------------------------------ warm-start prepass ------------------------------

def _parse_warm_start_config(spec: Any, solve_cfg: Mapping[str, Any]) -> Tuple[bool, int, str]:
    """
    Returns:
      enabled, max_passes, mode

    mode:
      - "override": assignment evaluations overwrite current guesses
      - "conservative": currently same overwrite behavior, hook reserved
    """
    ws_in = _pick_first(_spec_get(spec, "warm_start", None), solve_cfg.get("warm_start"), None)
    mode_in = _pick_first(_spec_get(spec, "warm_start_mode", None), solve_cfg.get("warm_start_mode"), None)
    passes_in = _pick_first(_spec_get(spec, "warm_start_passes", None), solve_cfg.get("warm_start_passes"), None)

    enabled = True if ws_in is None else bool(ws_in)
    max_passes = int(passes_in) if passes_in is not None else 6
    max_passes = max(1, min(50, max_passes))

    mode = str(mode_in).strip().lower() if mode_in is not None else "override"
    if mode not in {"override", "conservative"}:
        mode = "override"

    return enabled, max_passes, mode


def _warm_start_guesses(
    *,
    equations_raw: Sequence[str],
    unknowns: List[Dict[str, Any]],
    fixed: Mapping[str, Any],
    params: Mapping[str, Any],
    extra_funcs: Mapping[str, Any],
    max_passes: int,
    mode: str,
) -> Dict[str, Any]:
    """
    Attempt to generate better initial guesses by evaluating simple assignments:
      lhs = rhs
    """
    unknown_names = {u["name"] for u in unknowns}
    guess_map: Dict[str, float] = {u["name"]: float(u.get("guess", 1.0)) for u in unknowns}

    scope_vals: Dict[str, Any] = {}
    scope_vals.update(dict(params))
    scope_vals.update(dict(fixed))
    scope_vals.update(dict(guess_map))
    scope_vals.update({"pi": float(np.pi), "e": float(np.e)})

    updates: List[Dict[str, Any]] = []
    eval_failures = 0

    rhs_cache: Dict[int, Any] = {}

    for p in range(1, max_passes + 1):
        changed = 0

        for i, raw in enumerate(equations_raw):
            lhs, rhs = split_assignment(raw)
            if lhs is None or rhs is None:
                continue
            if lhs not in unknown_names:
                continue

            if i not in rhs_cache:
                try:
                    rhs_cache[i] = _compile_expression_with_extras(rhs, extra_funcs=extra_funcs)
                except Exception:
                    rhs_cache[i] = None

            c_rhs = rhs_cache.get(i, None)
            if c_rhs is None:
                continue

            try:
                val = float(
                    _eval_expression_with_extras(
                        c_rhs,
                        values=scope_vals,
                        params=params,
                        extra_funcs=extra_funcs,
                    )
                )
            except Exception:
                eval_failures += 1
                continue

            if not np.isfinite(val):
                eval_failures += 1
                continue

            prev = float(guess_map.get(lhs, float("nan")))
            if mode in {"override", "conservative"}:
                guess_map[lhs] = float(val)
                scope_vals[lhs] = float(val)

                if (not np.isfinite(prev)) or abs(float(val) - float(prev)) > 0.0:
                    changed += 1
                    updates.append({"pass": p, "equation_index": i + 1, "lhs": lhs, "rhs": rhs, "value": float(val)})

        if changed == 0:
            break

    for u in unknowns:
        nm = u["name"]
        if nm in guess_map and np.isfinite(guess_map[nm]):
            u["guess"] = float(guess_map[nm])

    return {
        "enabled": True,
        "mode": mode,
        "max_passes": max_passes,
        "updates": updates,
        "n_updates": len(updates),
        "eval_failures": int(eval_failures),
    }


# ------------------------------ public entrypoint ------------------------------

_BUILTIN_CONSTS = {"pi", "e"}


def solve_system(
    spec: Any,
    *,
    backend: str = "auto",
    method: str = "hybr",
    tol: float = 1e-9,
    max_iter: int = 200,
    max_restarts: int = 2,
) -> SolveResult:
    """
    Solve a nonlinear equation system.

    Notes:
    - Enforces EES-style "square system": n_equations == n_unknowns.
    - Uses safe_eval for equation parsing and evaluation.
    """
    solve_cfg = _get_solve_block(spec)

    backend = _normalize_backend_name(
        _pick_first(
            _spec_get(spec, "backend", None),
            _spec_get(spec, "solver", None),
            solve_cfg.get("backend"),
            solve_cfg.get("solver"),
            backend,
        )
    ) or "auto"
    backend_requested = backend

    method_in = _pick_first(solve_cfg.get("method"), _spec_get(spec, "method", None), method)
    method = str(method_in) if method_in is not None else "hybr"

    tol_in = _pick_first(solve_cfg.get("tol"), _spec_get(spec, "tol", None), tol)
    tol = float(tol_in if tol_in is not None else tol)

    mi_in = _pick_first(solve_cfg.get("max_iter"), solve_cfg.get("maxiter"), _spec_get(spec, "max_iter", None), max_iter)
    max_iter = int(mi_in if mi_in is not None else max_iter)

    mr_in = _pick_first(solve_cfg.get("max_restarts"), _spec_get(spec, "max_restarts", None), max_restarts)
    max_restarts = int(mr_in if mr_in is not None else max_restarts)

    eqs, vars_list, params = _extract_system(spec)

    auto_guess = bool(_pick_first(solve_cfg.get("auto_guess"), _spec_get(spec, "auto_guess", None), True))

    unknowns, fixed, bounds_used = _split_variables(vars_list, auto_guess=auto_guess)

    if len(eqs) != len(unknowns):
        raise ValueError(
            "EES rule violated: number of equations must equal number of unknowns "
            f"(got equations={len(eqs)}, unknowns={len(unknowns)})."
        )

    needs_props, needs_ha, needs_phase, needs_libr, needs_nh3 = _equations_require_thermo(eqs)
    # Track Cantera usage separately for meta/reporting (CTPropsSI family).
    needs_ct = any(_CTPROPS_CALL_RE.search(preprocess_expr(str(e))) for e in eqs)
    # NOTE: needs_props already becomes True for CTPropsSI (and AbstractState) calls, but we keep
    # needs_ct explicit so meta/reporting remains accurate across future refactors.
    needs_thermo = needs_props or needs_ha or needs_phase or needs_libr or needs_nh3 or needs_ct

    if needs_thermo:
        if backend in {"", "auto"}:
            backend = "scipy"
        elif backend == "gekko":
            raise ValueError(
                "GEKKO backend does not support thermo calls "
                "(PropsSI/CTPropsSI/PhaseSI/HAPropsSI/LiBrPropsSI/NH3H2O...). "
                "Your equations contain thermo property calls, so you must use backend='scipy' "
                "(or backend='auto', which will automatically select SciPy for thermo systems)."
            )

    if backend == "auto":
        backend = "gekko" if _gekko_available() else "scipy"

    method_requested = method
    if backend == "scipy":
        method = _normalize_scipy_method(method)
        if method not in _VALID_SCIPY_METHODS:
            raise ValueError(
                f"Unknown/unsupported SciPy root method {method_requested!r} -> {method!r}. "
                f"Supported: {sorted(_VALID_SCIPY_METHODS)}"
            )

    if backend == "gekko":
        for e in eqs:
            s = preprocess_expr(str(e))
            if (
                _PROPS_CALL_RE.search(s)
                or _HAPROPS_CALL_RE.search(s)
                or _PHASE_CALL_RE.search(s)
                or _ASPROPS_CALL_RE.search(s)
                or _LIBRPROPS_CALL_RE.search(s)
                or _NH3H2O_CALL_RE.search(s)
                or _CTPROPS_CALL_RE.search(s)
            ):
                raise ValueError(
                    "GEKKO backend does not support thermo calls "
                    "(PropsSI/CTPropsSI/PhaseSI/HAPropsSI/LiBrPropsSI/NH3H2O...). "
                    "Use backend='scipy' for thermo props equations."
                )

    if backend == "scipy":
        extra_funcs = _scipy_extra_funcs()
    else:
        extra_funcs = {k: v for k, v in _common_numeric_math_funcs().items() if v is not None}

    warm_enabled, warm_passes, warm_mode = _parse_warm_start_config(spec, solve_cfg)
    warm_meta: Dict[str, Any] = {"enabled": False}
    if warm_enabled:
        try:
            warm_meta = _warm_start_guesses(
                equations_raw=eqs,
                unknowns=unknowns,
                fixed=fixed,
                params=params,
                extra_funcs=extra_funcs,
                max_passes=warm_passes,
                mode=warm_mode,
            )
        except Exception as e:
            warm_meta = {"enabled": False, "error": f"{type(e).__name__}: {e}"}

    compiled = [_compile_residual_with_extras(e, extra_funcs=extra_funcs) for e in eqs]

    known_var_names = set(fixed.keys()) | {u["name"] for u in unknowns}
    known_names = known_var_names | set(params.keys()) | set(_BUILTIN_CONSTS)

    for i, c in enumerate(compiled, start=1):
        missing = [n for n in getattr(c, "names", []) if n not in known_names]
        if missing:
            raise ValueError(
                f"Equation {i} references unknown names {missing}. "
                f"Known vars={sorted(known_var_names)}, "
                f"params/constants={sorted(params.keys())}, "
                f"builtins={sorted(_BUILTIN_CONSTS)}"
            )

    if backend == "scipy":
        user_opts = _pick_first(
            solve_cfg.get("options"),
            solve_cfg.get("scipy_options"),
            _spec_get(spec, "options", None),
            _spec_get(spec, "scipy_options", None),
        )
        if user_opts is not None and not isinstance(user_opts, Mapping):
            user_opts = None

        thermo_penalty = float(_pick_first(solve_cfg.get("thermo_penalty"), _spec_get(spec, "thermo_penalty", None), 1e9))

        res = _solve_scipy(
            equations=compiled,
            unknowns=unknowns,
            fixed=fixed,
            params=params,
            method=method,
            tol=tol,
            max_iter=max_iter,
            max_restarts=max_restarts,
            bounds_ignored=bounds_used,
            extra_funcs=extra_funcs,
            user_options=user_opts,
            warm_start_meta=warm_meta,
            thermo_penalty=thermo_penalty,
        )
        res.meta.setdefault("method_requested", str(method_requested))
        res.meta.setdefault("method_effective", str(method))
        res.meta.setdefault("backend_requested", str(backend_requested))
        res.meta.setdefault("backend_effective", "scipy")
        res.meta.setdefault("propssi_detected", bool(needs_props))
        res.meta.setdefault("ctprops_detected", bool(needs_ct))
        res.meta.setdefault("phasesi_detected", bool(needs_phase))
        res.meta.setdefault("hapropssi_detected", bool(needs_ha))
        res.meta.setdefault("librpropssi_detected", bool(needs_libr))
        res.meta.setdefault("nh3h2o_detected", bool(needs_nh3))
        res.meta.setdefault("thermo_detected", bool(needs_thermo))
        res.meta.setdefault("auto_guess", bool(auto_guess))

        # Optional: expose CTPropsSI cache stats for debugging/perf analysis.
        if needs_ct:
            _maybe_add_ct_cache_meta(res.meta)
        return res

    if backend == "gekko":
        res = _solve_gekko(
            equations_raw=eqs,
            unknowns=unknowns,
            fixed=fixed,
            params=params,
            solve_cfg=solve_cfg,
            tol=tol,
            max_iter=max_iter,
            method=str(method_requested),
            warm_start_meta=warm_meta,
        )
        res.meta.setdefault("method_requested", str(method_requested))
        res.meta.setdefault("backend_requested", str(backend_requested))
        res.meta.setdefault("backend_effective", "gekko")
        res.meta.setdefault("propssi_detected", bool(needs_props))
        res.meta.setdefault("phasesi_detected", bool(needs_phase))
        res.meta.setdefault("hapropssi_detected", bool(needs_ha))
        res.meta.setdefault("librpropssi_detected", bool(needs_libr))
        res.meta.setdefault("nh3h2o_detected", bool(needs_nh3))
        res.meta.setdefault("thermo_detected", bool(needs_thermo))
        res.meta.setdefault("auto_guess", bool(auto_guess))
        return res

    raise ValueError(f"Unknown backend: {backend!r} (expected auto|scipy|gekko).")


# ------------------------------ variable handling ------------------------------

def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        if DEFAULT_REGISTRY is not None and parse_quantity is not None:
            try:
                q = parse_quantity(s, DEFAULT_REGISTRY)
                return float(q.base_value())
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            return default
    try:
        return float(x)
    except Exception:
        return default


def _var_get(v: Any, key: str, default: Any = None) -> Any:
    if isinstance(v, Mapping):
        return v.get(key, default)
    return getattr(v, key, default)


def _is_unknown(v: Any) -> bool:
    if isinstance(v, Mapping):
        k = v.get("kind", None)
        if k is not None:
            return str(k).lower() == "unknown"
        if "unknown" in v:
            try:
                return bool(v.get("unknown"))
            except Exception:
                pass
        if "value" in v and v.get("value", None) is not None:
            return False
        return True

    if hasattr(v, "unknown"):
        try:
            return bool(getattr(v, "unknown"))
        except Exception:
            pass
    if hasattr(v, "kind"):
        try:
            return str(getattr(v, "kind")).lower() == "unknown"
        except Exception:
            pass
    if hasattr(v, "value"):
        return getattr(v, "value") is None
    return False


def _var_name(v: Any) -> str:
    if isinstance(v, Mapping):
        n = v.get("name", None)
        if not n or not isinstance(n, str):
            raise ValueError(f"Variable missing valid name: {v!r}")
        return n
    n = getattr(v, "name", None)
    if not n or not isinstance(n, str):
        raise ValueError(f"Variable missing valid name: {v!r}")
    return n


def _auto_guess_from_name(name: str) -> float | None:
    """
    Heuristic guess only used when the user did NOT provide a guess/value.

    Returns:
      a float guess, or None if no heuristic applies.
    """
    n = name.strip()
    if not n:
        return None
    s = n.lower()

    # Temperature-like
    if re.match(r"^t\d*$", s) or s.startswith("t_") or s.endswith("_t") or "temp" in s:
        return 300.0  # K

    # Pressure-like
    if re.match(r"^p\d*$", s) or s.startswith("p_") or s.endswith("_p") or "press" in s:
        return 101325.0  # Pa

    # Relative humidity / phi
    if s in {"rh", "r", "phi"} or s.startswith("phi") or "relhum" in s:
        return 0.5

    # Humidity ratio
    if re.match(r"^w\d*$", s) or s.startswith("w_") or s in {"w", "omega"}:
        return 0.01

    # Concentration-like (LiBr mass fraction X)
    if s in {"x", "xb", "x_libr", "x_libr_mass"} or s.startswith("x_") or "conc" in s:
        return 0.55

    # Ammonia mass fraction / concentration (common names)
    if s in {"x_nh3", "x_nh3_mass", "x_ammonia"} or ("nh3" in s and s.startswith("x")):
        return 0.4

    return None


def _split_variables(vars_list: Sequence[Any], *, auto_guess: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """
    Returns:
      unknowns: list of dicts with fields {name, guess, lower, upper}
      fixed:   dict name->value
      bounds_used: True if any unknown had bounds (SciPy root ignores them)
    """
    seen: set[str] = set()
    unknowns: List[Dict[str, Any]] = []
    fixed: Dict[str, Any] = {}
    bounds_used = False

    for v in vars_list:
        name = _var_name(v)
        if name in seen:
            raise ValueError(f"Duplicate variable name: {name!r}")
        seen.add(name)

        if _is_unknown(v):
            user_guess_raw = _var_get(v, "guess", None)
            user_value_raw = _var_get(v, "value", None)

            guess_is_default = False
            guess: float | None

            guess = _as_float(user_guess_raw, default=None)
            if guess is None:
                guess = _as_float(user_value_raw, default=None)

            if guess is None:
                guess = 1.0
                guess_is_default = True

            if auto_guess and guess_is_default:
                g2 = _auto_guess_from_name(name)
                if g2 is not None:
                    guess = float(g2)

            lower = _as_float(_var_get(v, "lower", None), default=None)
            upper = _as_float(_var_get(v, "upper", None), default=None)
            if lower is not None or upper is not None:
                bounds_used = True

            unknowns.append(
                {
                    "name": name,
                    "guess": float(guess if guess is not None else 1.0),
                    "lower": lower,
                    "upper": upper,
                }
            )
        else:
            val = _var_get(v, "value", None)
            fv = _as_float(val, default=None)
            if fv is None:
                raise ValueError(f"Fixed variable {name!r} must have a numeric value.")
            fixed[name] = float(fv)

    return unknowns, fixed, bounds_used


# ------------------------------ SciPy backend ------------------------------

def _looks_like_thermo_error(e: BaseException) -> bool:
    # Missing providers should be loud, not masked by thermo penalties.
    if isinstance(e, ImportError):
        return False

    msg = str(e)
    s = msg.lower()
    return (
        ("propssi" in s)
        or ("hapropssi" in s)
        or ("phasesi" in s)
        or ("ctprops" in s)
        or ("cantera" in s)
        or ("librpropssi" in s)
        or ("librh2o" in s)
        or ("ashrae" in s)
        or ("coolprop" in s)
        or ("outside the range of validity" in s)
        or ("humidairprop" in s)
        or ("nh3h2o" in s)
        or ("ammonia" in s)
    )


def _solve_scipy(
    *,
    equations: Sequence[Any],
    unknowns: Sequence[Dict[str, Any]],
    fixed: Mapping[str, Any],
    params: Mapping[str, Any],
    method: str,
    tol: float,
    max_iter: int,
    max_restarts: int,
    bounds_ignored: bool,
    extra_funcs: Mapping[str, Any],
    user_options: Optional[Mapping[str, Any]],
    warm_start_meta: Mapping[str, Any],
    thermo_penalty: float,
) -> SolveResult:
    if scipy_root is None:
        raise ImportError("SciPy is not installed. Install `scipy` or use backend='gekko' (non-thermo only).")

    method_eff = _normalize_scipy_method(method)
    if method_eff not in _VALID_SCIPY_METHODS:
        raise ValueError(
            f"Unknown/unsupported SciPy root method: {method!r} -> {method_eff!r}. "
            f"Supported: {sorted(_VALID_SCIPY_METHODS)}"
        )

    names = [u["name"] for u in unknowns]
    x0 = np.array([float(u["guess"]) for u in unknowns], dtype=float)

    thermo_eval_errors: List[str] = []

    def residual_vec(x: "np.ndarray") -> "np.ndarray":
        vals: Dict[str, Any] = dict(fixed)
        vals.update({n: float(xi) for n, xi in zip(names, x)})
        vals.update({"pi": float(np.pi), "e": float(np.e)})

        r: List[float] = []
        for j, c in enumerate(equations, start=1):
            try:
                rj = _eval_compiled_with_extras(c, values=vals, params=params, extra_funcs=extra_funcs)
                r.append(float(rj))
            except Exception as ex:
                if _looks_like_thermo_error(ex):
                    if len(thermo_eval_errors) < 8:
                        thermo_eval_errors.append(f"eq{j}: {type(ex).__name__}: {ex}")
                    r.append(float(thermo_penalty))
                    continue
                raise

        return np.array(r, dtype=float)

    # initial evaluation (fail fast with a useful message)
    try:
        _ = residual_vec(x0)
    except Exception as e:
        raise ValueError(f"Initial residual evaluation failed at initial guess. Details: {e}") from e

    guesses = [x0]
    if max_restarts > 0:
        guesses.append(_safe_scale_guess(x0, 1.1))
    if max_restarts > 1:
        guesses.append(_safe_scale_guess(x0, 0.9))
    guesses = guesses[: 1 + max_restarts]

    options = _scipy_root_options(method_eff, max_iter, user_options=user_options)

    best_x = None
    best_norm = float("inf")
    best_msg = "no attempt"
    best_status = 0
    total_nfev = 0
    attempts = 0

    for attempt, g in enumerate(guesses, start=1):
        attempts = attempt
        try:
            sol = scipy_root(
                fun=residual_vec,
                x0=g,
                method=method_eff,
                tol=tol,
                options=dict(options),
            )
        except Exception as e:
            best_msg = f"scipy.root threw: {e}"
            continue

        total_nfev += int(getattr(sol, "nfev", 0) or 0)

        x = np.array(getattr(sol, "x", g), dtype=float)
        r = residual_vec(x).tolist()
        rn = float(np.linalg.norm(np.array(r, dtype=float)))
        msg = str(getattr(sol, "message", "")) or "scipy.root finished"
        status = int(getattr(sol, "status", 0) or 0)

        if rn < best_norm:
            best_x = x
            best_norm = rn
            best_msg = msg
            best_status = status

        if bool(getattr(sol, "success", False)):
            vals: Dict[str, Any] = dict(fixed)
            vals.update({n: float(xi) for n, xi in zip(names, x)})
            vals.update({"pi": float(np.pi), "e": float(np.e)})
            return SolveResult(
                ok=True,
                backend="scipy",
                method=str(method_eff),
                message=msg,
                nfev=int(getattr(sol, "nfev", 0) or 0),
                variables=vals,
                residuals=[float(v) for v in r],
                residual_norm=float(rn),
                meta={
                    "attempt": attempt,
                    "attempts": len(guesses),
                    "tol": float(tol),
                    "max_iter": int(max_iter),
                    "options_used": dict(options),
                    "scipy_status": status,
                    "bounds_ignored": bool(bounds_ignored),
                    "warm_start": dict(warm_start_meta),
                    "thermo_penalty": float(thermo_penalty),
                    "thermo_eval_errors": list(thermo_eval_errors),
                },
            )

    # If no success, return best attempt by residual norm (useful for debugging)
    if best_x is not None:
        vals2: Dict[str, Any] = dict(fixed)
        vals2.update({n: float(xi) for n, xi in zip(names, best_x)})
        vals2.update({"pi": float(np.pi), "e": float(np.e)})
        r2 = residual_vec(np.array(best_x, dtype=float)).tolist()
        rn2 = float(np.linalg.norm(np.array(r2, dtype=float)))
        return SolveResult(
            ok=False,
            backend="scipy",
            method=str(method_eff),
            message=best_msg,
            nfev=int(total_nfev),
            variables=vals2,
            residuals=[float(v) for v in r2],
            residual_norm=float(rn2),
            meta={
                "attempts": attempts,
                "tol": float(tol),
                "max_iter": int(max_iter),
                "options_used": dict(options),
                "scipy_status": best_status,
                "bounds_ignored": bool(bounds_ignored),
                "note": "No SciPy attempt returned success=True; returning best residual-norm attempt.",
                "warm_start": dict(warm_start_meta),
                "thermo_penalty": float(thermo_penalty),
                "thermo_eval_errors": list(thermo_eval_errors),
            },
        )

    return SolveResult(
        ok=False,
        backend="scipy",
        method=str(method_eff),
        message=best_msg,
        nfev=int(total_nfev),
        variables=dict(fixed),
        residuals=[],
        residual_norm=float("inf"),
        meta={
            "attempts": attempts,
            "tol": float(tol),
            "max_iter": int(max_iter),
            "options_used": dict(options),
            "bounds_ignored": bool(bounds_ignored),
            "warm_start": dict(warm_start_meta),
            "thermo_penalty": float(thermo_penalty),
            "thermo_eval_errors": list(thermo_eval_errors),
        },
    )


def _safe_scale_guess(x: "np.ndarray", scale: float) -> "np.ndarray":
    x2 = np.array(x, dtype=float) * float(scale)
    x2 = np.where(np.abs(x2) < 1e-12, np.sign(x + 1e-12) * 1e-6, x2)
    return x2


# ------------------------------ GEKKO backend ------------------------------

_GEKKO_ALLOWED_FUNC_NAMES: Tuple[str, ...] = (
    "abs", "pow", "min", "max", "clamp",
    "sign", "step",
    "sqrt", "exp", "log", "ln", "log10", "log2", "log1p", "expm1",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "floor", "ceil",
    "hypot", "degrees", "radians",
)

_GEKKO_SAFE_FUNCS: Dict[str, Any] | None = None


def _missing_func(name: str) -> Any:
    def _f(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(
            f"Function {name!r} is allowed by the expression whitelist, "
            "but is not implemented in the current GEKKO scope (model method missing)."
        )
    _f.__name__ = name
    return _f


def _gekko_safe_funcs() -> Dict[str, Any]:
    global _GEKKO_SAFE_FUNCS
    if _GEKKO_SAFE_FUNCS is not None:
        return _GEKKO_SAFE_FUNCS

    _ = _require_gekko()

    funcs: Dict[str, Any] = {
        "abs": abs,
        "pow": pow,
        "min": _missing_func("min"),
        "max": _missing_func("max"),
        "clamp": _missing_func("clamp"),
        "sign": _missing_func("sign"),
        "step": _missing_func("step"),
    }

    for nm in _GEKKO_ALLOWED_FUNC_NAMES:
        funcs.setdefault(nm, _missing_func(nm))

    _GEKKO_SAFE_FUNCS = funcs
    return _GEKKO_SAFE_FUNCS


def _gekko_symbolic_funcs(m: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    direct = [
        "sin", "cos", "tan",
        "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "exp", "log", "sqrt",
        "floor", "ceil",
        "atan2",
        "hypot",
    ]
    for nm in direct:
        f = getattr(m, nm, None)
        if callable(f):
            out[nm] = f

    if "log" in out:
        out["ln"] = out["log"]
        out["log10"] = (lambda x, _log=out["log"]: _log(x) / _log(10.0))
        out["log2"] = (lambda x, _log=out["log"]: _log(x) / _log(2.0))
        out["log1p"] = (lambda x, _log=out["log"]: _log(1.0 + x))
    if "exp" in out:
        out["expm1"] = (lambda x, _exp=out["exp"]: _exp(x) - 1.0)

    max2 = getattr(m, "max2", None)
    min2 = getattr(m, "min2", None)
    if callable(max2):
        out["max"] = max2
    if callable(min2):
        out["min"] = min2

    if "min" in out and "max" in out:
        out["clamp"] = lambda x, lo, hi: out["max"](lo, out["min"](hi, x))

    out["radians"] = lambda x: x * (np.pi / 180.0)
    out["degrees"] = lambda x: x * (180.0 / np.pi)

    sign2 = getattr(m, "sign2", None)
    if callable(sign2):
        out["sign"] = sign2

    return out


def _gekko_solver_id(method: str, solve_cfg: Mapping[str, Any]) -> int:
    override = solve_cfg.get("gekko_solver", None)
    if override is not None:
        try:
            return int(override)
        except Exception:
            pass

    m = _normalize_method_name(method)
    if m in {"apopt", "apopt-solver", "solver-1"}:
        return 1
    if m in {"ipopt", "ipopt-solver", "solver-3", "", "default"}:
        return 3
    return 3


def _solve_gekko(
    *,
    equations_raw: Sequence[str],
    unknowns: Sequence[Dict[str, Any]],
    fixed: Mapping[str, Any],
    params: Mapping[str, Any],
    solve_cfg: Mapping[str, Any],
    tol: float,
    max_iter: int,
    method: str,
    warm_start_meta: Mapping[str, Any],
) -> SolveResult:
    GEKKO = _require_gekko()

    for k, v in params.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"GEKKO backend only supports numeric constants/params. Got {k!r}={v!r}")

    m = GEKKO(remote=False)

    solve_cfg_ = dict(solve_cfg) if isinstance(solve_cfg, Mapping) else {}
    try:
        m.options.SOLVER = _gekko_solver_id(method, solve_cfg_)
    except Exception:
        pass
    try:
        m.options.MAX_ITER = int(max_iter)
    except Exception:
        pass
    try:
        m.options.OTOL = float(tol)
        m.options.RTOL = float(tol)
    except Exception:
        pass

    scope: Dict[str, Any] = {}
    scope.update(_gekko_safe_funcs())
    scope.update(_gekko_symbolic_funcs(m))
    scope.update({"pi": float(np.pi), "e": float(np.e)})

    for k, v in params.items():
        scope[str(k)] = m.Param(value=float(v))
    for k, v in fixed.items():
        scope[str(k)] = m.Param(value=float(v))

    for u in unknowns:
        name = str(u["name"])
        guess = float(u.get("guess", 1.0))
        lb = u.get("lower", None)
        ub = u.get("upper", None)
        if lb is not None or ub is not None:
            vv = m.Var(value=guess, lb=lb, ub=ub)
        else:
            vv = m.Var(value=guess)
        scope[name] = vv

    for raw in equations_raw:
        s = preprocess_expr(raw)

        if "==" in s and "=" not in s.replace("==", ""):
            s = s.replace("==", "=")

        if "=" in s:
            left, right = s.split("=", 1)
            left = left.strip()
            right = right.strip()
            if not left or not right:
                raise ParseError(f"Malformed equation (empty side): {raw!r}")
            expr_left = _eval_gekko_expr(left, scope)
            expr_right = _eval_gekko_expr(right, scope)
            m.Equation(expr_left == expr_right)
        else:
            resid = normalize_equation_to_residual(s)
            expr = _eval_gekko_expr(resid, scope)
            m.Equation(expr == 0)

    try:
        m.solve(disp=False)
        solved = True
        msg = "gekko solve ok"
    except Exception as e:
        solved = False
        msg = f"gekko solve failed: {e}"

    out_vars: Dict[str, Any] = dict(fixed)
    out_vars.update({"pi": float(np.pi), "e": float(np.e)})

    for u in unknowns:
        n = str(u["name"])
        vv = scope[n]
        try:
            out_vars[n] = float(vv.value[0])
        except Exception:
            try:
                out_vars[n] = float(vv.value)
            except Exception:
                out_vars[n] = float("nan")

    extra_funcs_num = {k: v for k, v in _common_numeric_math_funcs().items() if v is not None}
    compiled = [_compile_residual_with_extras(e, extra_funcs=extra_funcs_num) for e in equations_raw]
    residuals = [
        float(_eval_compiled_with_extras(c, values=out_vars, params=params, extra_funcs=extra_funcs_num))
        for c in compiled
    ]
    rn_raw = float(np.linalg.norm(np.array(residuals, dtype=float)))

    # Scale-aware residuals for ok-check (dimensionless / relative to equation magnitude).
    scaled_residuals: List[float] = []
    for raw, r in zip(equations_raw, residuals):
        lhs, rhs = split_assignment(raw)
        denom = 1.0
        if lhs is not None and rhs is not None:
            lhs_val = out_vars.get(lhs, float("nan"))
            try:
                c_rhs = _compile_expression_with_extras(rhs, extra_funcs=extra_funcs_num)
                rhs_val = float(_eval_expression_with_extras(c_rhs, values=out_vars, params=params, extra_funcs=extra_funcs_num))
            except Exception:
                rhs_val = float("nan")

            a = abs(float(lhs_val)) if np.isfinite(lhs_val) else 0.0
            b = abs(float(rhs_val)) if np.isfinite(rhs_val) else 0.0
            denom = max(1.0, a, b)
        else:
            denom = max(1.0, abs(float(r)))

        scaled_residuals.append(float(r) / float(denom))

    rn_scaled = float(np.linalg.norm(np.array(scaled_residuals, dtype=float)))
    maxabs_scaled = float(np.max(np.abs(np.array(scaled_residuals, dtype=float)))) if scaled_residuals else float("inf")

    finite = bool(np.isfinite(rn_raw)) and bool(np.all(np.isfinite(np.array(residuals, dtype=float))))
    ok_factor = float(solve_cfg.get("gekko_ok_factor", 10000.0))
    ok = bool(solved) and finite and (maxabs_scaled <= max(ok_factor * tol, tol))

    return SolveResult(
        ok=ok,
        backend="gekko",
        method="ipopt" if solved else "ipopt(failed)",
        message=msg,
        nfev=0,
        variables=out_vars,
        residuals=[float(r) for r in residuals],
        residual_norm=float(rn_raw),
        meta={
            "tol": float(tol),
            "max_iter": int(max_iter),
            "warm_start": dict(warm_start_meta),
            "residual_norm_raw": float(rn_raw),
            "residual_norm_scaled": float(rn_scaled),
            "residual_maxabs_scaled": float(maxabs_scaled),
            "ok_rule": "gekko: maxabs(scaled_residual) <= max(1000*tol,tol)",
        },
    )


def _eval_gekko_expr(expr: str, scope: Mapping[str, Any]) -> Any:
    s = preprocess_expr(expr)
    _validate_expr_only(s)

    try:
        import ast

        node = ast.parse(s, mode="eval")
        code = compile(node, filename="<gekko_expr>", mode="eval")
        return eval(code, {"__builtins__": {}}, dict(scope))  # noqa: S307
    except Exception as e:
        raise ParseError(f"GEKKO expression eval failed for {expr!r}: {e}") from e


def _validate_expr_only(expr: str) -> None:
    import ast

    s = preprocess_expr(expr)
    if "=" in s:
        raise UnsafeExpressionError("Expressions passed to GEKKO evaluator must not contain '='.")

    try:
        node = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise ParseError(f"Invalid syntax: {expr!r}") from e

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    allowed_unaryops = (ast.UAdd, ast.USub)
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant)

    safe_funcs = set(_gekko_safe_funcs().keys())
    safe_consts = {"pi", "e"}

    class V(ast.NodeVisitor):
        def generic_visit(self, n: ast.AST) -> Any:
            if not isinstance(n, allowed_nodes):
                raise UnsafeExpressionError(f"Unsupported/unsafe syntax: {type(n).__name__}")
            return super().generic_visit(n)

        def visit_Name(self, n: ast.Name) -> Any:
            if n.id.startswith("__"):
                raise UnsafeExpressionError("Dunder names are not allowed.")
            return self.generic_visit(n)

        def visit_BinOp(self, n: ast.BinOp) -> Any:
            if not isinstance(n.op, allowed_binops):
                raise UnsafeExpressionError(f"Operator not allowed: {type(n.op).__name__}")
            self.visit(n.left)
            self.visit(n.right)

        def visit_UnaryOp(self, n: ast.UnaryOp) -> Any:
            if not isinstance(n.op, allowed_unaryops):
                raise UnsafeExpressionError(f"Unary operator not allowed: {type(n.op).__name__}")
            self.visit(n.operand)

        def visit_Call(self, n: ast.Call) -> Any:
            if not isinstance(n.func, ast.Name):
                raise UnsafeExpressionError("Only direct function calls like f(x) are allowed.")
            fn = n.func.id
            if fn not in safe_funcs:
                raise UnsafeExpressionError(f"Function not allowed: {fn!r}")

            for kw in n.keywords:
                if kw.arg is None:
                    raise UnsafeExpressionError("**kwargs are not allowed.")

            for a in n.args:
                self.visit(a)
            for kw in n.keywords:
                self.visit(kw.value)

        def visit_Attribute(self, n: ast.Attribute) -> Any:  # pragma: no cover
            raise UnsafeExpressionError("Attribute access is not allowed (e.g., obj.x).")

        def visit_Subscript(self, n: ast.Subscript) -> Any:  # pragma: no cover
            raise UnsafeExpressionError("Indexing/subscripts are not allowed (e.g., a[0]).")

    V().visit(node)

    for nm in _iter_names(node):
        if nm.startswith("__"):
            raise UnsafeExpressionError("Dunder names are not allowed.")
        if nm in safe_funcs or nm in safe_consts:
            continue


def _iter_names(node: Any) -> List[str]:
    import ast

    out: List[str] = []

    class N(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> Any:
            out.append(n.id)

    N().visit(node)
    return out
