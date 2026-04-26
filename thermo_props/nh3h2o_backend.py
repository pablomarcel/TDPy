"""NH3-H2O property backend for TDPy.

This module wraps the native ammonia-water implementation used for Ibrahim and
Klein style NH3-H2O property work. It provides TPX state evaluation, scalar
property calls, batch helpers, EES-style shims, and CoolProp-style shims.

Inputs are temperature in kelvin, pressure in pascals, NH3 mass fraction, and
an optional EES quality flag for supported TXQ calls. Outputs are plain Python
floats or dictionaries suitable for JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence
import math
import numbers


__all__ = [
    # fluid support
    "ALIASES",
    "supports",
    # errors
    "NH3H2ONotInstalled",
    "NH3H2OCallError",
    # availability
    "nh3h2o_available",
    # primary API (TPX)
    "state_tpx",
    "prop_tpx",
    "props_multi_tpx",
    "phase_tpx",
    "ok_tpx",
    "batch_prop_tpx",
    "batch_state_tpx",
    # convenience wrappers (TPX)
    "h_tpx",
    "s_tpx",
    "u_tpx",
    "v_tpx",
    "rho_tpx",
    "q_tpx",
    # EES-PropsSI style scalar shim (what the solver is trying to inject)
    "NH3H2OPropsSI",
    "nh3h2o_props_si",
    # CoolProp-like shims
    "NH3H2O",
    "NH3H2O_STATE",
    "NH3H2O_TPX",
    "NH3H2O_STATE_TPX",
    # dataclasses
    "NH3H2OCall",
]


# ------------------------------ fluid aliases ------------------------------

ALIASES = {
    "NH3H2O",
    "NH3-H2O",
    "NH3_H2O",
    "AMMONIAWATER",
    "AMMONIA_WATER",
    "AMMONIA-WATER",
    "AMMONIA/H2O",
}


def supports(fluid: str) -> bool:
    """Return whether a fluid string should dispatch to this backend."""
    f = str(fluid).strip().upper()
    return bool(f) and (f in ALIASES)


# ------------------------------ errors ------------------------------

class NH3H2ONotInstalled(ImportError):
    """Raised when the native ammonia-water implementation cannot be imported."""


class NH3H2OCallError(RuntimeError):
    """Raised when an NH3-H2O property call fails with context."""


# ------------------------------ internal import caching ------------------------------

_props_tpx_impl: Callable[..., Mapping[str, Any]] | None = None


def _import_ammonia_water_props_tpx() -> Callable[..., Mapping[str, Any]]:
    """Import ``ammonia_water.props_tpx`` lazily and cache the callable.

The preferred import path is the sibling module inside ``thermo_props``. Fallback
paths are retained for package moves and sandbox-style execution."""
    global _props_tpx_impl
    if _props_tpx_impl is not None:
        return _props_tpx_impl

    last: Exception | None = None

    try:
        from .ammonia_water import props_tpx as fn  # type: ignore
        _props_tpx_impl = fn
        return _props_tpx_impl
    except Exception as e:  # pragma: no cover
        last = e

    try:
        from thermo_props.ammonia_water import props_tpx as fn  # type: ignore
        _props_tpx_impl = fn
        return _props_tpx_impl
    except Exception as e:  # pragma: no cover
        last = e

    try:
        import ammonia_water  # type: ignore
        fn = getattr(ammonia_water, "props_tpx", None)
        if not callable(fn):
            raise AttributeError("ammonia_water.props_tpx not found or not callable")
        _props_tpx_impl = fn
        return _props_tpx_impl
    except Exception as e:  # pragma: no cover
        last = e

    raise NH3H2ONotInstalled(
        "ammonia_water.py could not be imported. Expected it to be available as "
        "thermo_props.ammonia_water (preferred) or as a top-level module "
        "named ammonia_water. Ensure ammonia_water.py is on the import path."
    ) from last


# ------------------------------ availability ------------------------------

def nh3h2o_available() -> bool:
    """Return whether ``ammonia_water.props_tpx`` is importable."""
    try:
        _import_ammonia_water_props_tpx()
        return True
    except Exception:
        return False


# ------------------------------ normalization ------------------------------

# Inputs for the CoolProp-like signature (order-agnostic)
_INPUT_ALIASES: Mapping[str, str] = {
    # temperature
    "t": "T",
    "temp": "T",
    "temperature": "T",
    "t_k": "T",

    # pressure
    "p": "P",
    "press": "P",
    "pressure": "P",
    "p_pa": "P",

    # NH3 mass fraction (composition)
    "x": "X",
    "x_nh3": "X",
    "xmass": "X",
    "x_mass": "X",
    "w": "X",
    "w_nh3": "X",
    "massfrac": "X",
    "massfraction": "X",

    # EES-ish quality flag (used as an input in the EES docs for surface tension)
    "q": "Q",
    "quality": "Q",
}

# Outputs (numeric) from ammonia_water.props_tpx result dict
# Map common EES-ish names to result keys.
_OUTPUT_ALIASES: Mapping[str, str] = {
    # primary thermo
    "h": "h_J_per_kg",
    "hmass": "h_J_per_kg",
    "enthalpy": "h_J_per_kg",
    "s": "s_J_per_kgK",
    "smass": "s_J_per_kgK",
    "entropy": "s_J_per_kgK",
    "u": "u_J_per_kg",
    "umass": "u_J_per_kg",
    "internalenergy": "u_J_per_kg",
    "v": "v_m3_per_kg",
    "vol": "v_m3_per_kg",
    "specificvolume": "v_m3_per_kg",
    "rho": "rho_kg_per_m3",
    "d": "rho_kg_per_m3",
    "density": "rho_kg_per_m3",

    # heat capacities (if your ammonia_water exposes them)
    "cp": "cp_J_per_kgK",
    "cp_mass": "cp_J_per_kgK",
    "cpmass": "cp_J_per_kgK",
    "cv": "cv_J_per_kgK",
    "cv_mass": "cv_J_per_kgK",
    "cvmass": "cv_J_per_kgK",

    # EES-style quality flag (output)
    "q": "q",
    "quality": "q",

    # ---- transport props ----
    "k": "k",          # conductivity request
    "mu": "mu",        # viscosity request
    "sigma": "sigma",  # surface tension request
    "conductivity": "k",
    "viscosity": "mu",
    "surfacetension": "sigma",
    "surface_tension": "sigma",

    # inputs echoed back / convenience
    "t_k": "T_K",
    "t_c": "T_C",
    "p_pa": "P_Pa",
    "p_kpa": "P_kPa",
    "p_bar": "P_bar",
    "x_mass": "X",
    "z_mole": "z_mole",

    # VLE outputs (if present)
    "xl": "xL",
    "xL": "xL",
    "yv": "yV",
    "yV": "yV",
    "wl": "wL",
    "wL": "wL",
    "wv": "wV",
    "wV": "wV",
}

# If the underlying ammonia_water dict uses different names for certain keys,
# try these (in order) before raising a KeyError.
_OUTPUT_FALLBACKS: Mapping[str, Sequence[str]] = {
    # conductivity
    "k": (
        "k_W_per_mK",
        "k_W_per_m_K",
        "k_WmK",
        "k",
        "k_liq_W_per_mK",
        "k_liq_W_per_m_K",
        "conductivity_W_per_mK",
        "conductivity",
    ),
    # viscosity
    "mu": (
        "mu_Pa_s",
        "mu_Pas",
        "mu",
        "mu_liq_Pa_s",
        "mu_liq_Pas",
        "viscosity_Pa_s",
        "viscosity",
    ),
    # surface tension
    "sigma": (
        "sigma_N_per_m",
        "sigma_Nm",
        "sigma",
        "surface_tension_N_per_m",
        "surface_tension",
        "surfacetension_N_per_m",
        "surfacetension",
    ),
    # cp / cv (optional)
    "cp_J_per_kgK": (
        "cp_J_per_kgK",
        "cp_J_per_kg_K",
        "cp_JkgK",
        "cp",
        "cp_liq_J_per_kgK",
        "cpV_J_per_kgK",
        "cpL_J_per_kgK",
    ),
    "cv_J_per_kgK": (
        "cv_J_per_kgK",
        "cv_J_per_kg_K",
        "cv_JkgK",
        "cv",
        "cv_liq_J_per_kgK",
        "cvV_J_per_kgK",
        "cvL_J_per_kgK",
    ),
    # enthalpy etc (tolerate alternate spellings if present)
    "h_J_per_kg": ("h_J_per_kg", "h", "hmass_J_per_kg", "h_mass_J_per_kg"),
    "s_J_per_kgK": ("s_J_per_kgK", "s", "smass_J_per_kgK", "s_mass_J_per_kgK"),
    "u_J_per_kg": ("u_J_per_kg", "u", "umass_J_per_kg", "u_mass_J_per_kg"),
    "v_m3_per_kg": ("v_m3_per_kg", "v", "vmass_m3_per_kg", "v_mass_m3_per_kg"),
    "rho_kg_per_m3": ("rho_kg_per_m3", "rho", "D", "density_kg_per_m3", "density"),
}


@lru_cache(maxsize=256)
def _norm_in(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("NH3H2O input key is empty.")
    return _INPUT_ALIASES.get(k.lower(), k)


@lru_cache(maxsize=256)
def _norm_out(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("NH3H2O output key is empty.")
    return _OUTPUT_ALIASES.get(k.lower(), k)


# ------------------------------ dataclasses ------------------------------

@dataclass(frozen=True)
class NH3H2OCall:
    """Container for one NH3-H2O CoolProp-style property call.

The object stores one output key, three input key-value pairs, and a strict
error-handling flag. The input keys are expected to resolve to T, P, and X."""
    out: str
    in1: str
    v1: float
    in2: str
    v2: float
    in3: str
    v3: float
    strict: bool = True


# ------------------------------ helpers ------------------------------

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


def _to_massfrac(name: str, x: Any) -> float:
    """Validate and return a mass fraction in the closed interval from zero to one."""
    y = _to_float(name, x)
    if y < 0.0 or y > 1.0:
        raise ValueError(f"{name} must be a mass fraction in [0, 1]. Got {y!r}")
    return y


def _wrap_call_error(
    *,
    what: str,
    out: str | None,
    out_raw: str | None,
    T: float | None,
    P: float | None,
    X: float | None,
    Q: float | None = None,
    in1: str | None = None,
    v1: float | None = None,
    in2: str | None = None,
    v2: float | None = None,
    in3: str | None = None,
    v3: float | None = None,
    exc: BaseException | None = None,
) -> NH3H2OCallError:
    lines: list[str] = [f"NH3H2O {what} call failed."]
    if out is not None:
        if out_raw is not None and out_raw != out:
            lines.append(f"  out={out!r} (from {out_raw!r})")
        else:
            lines.append(f"  out={out!r}")
    if T is not None:
        lines.append(f"  T={T} K")
    if P is not None:
        lines.append(f"  P={P} Pa")
    if X is not None:
        lines.append(f"  X={X} (NH3 mass fraction)")
    if Q is not None:
        lines.append(f"  Q={Q} (EES quality flag)")
    if in1 is not None:
        lines.append(f"  in1={in1!r} v1={v1}")
    if in2 is not None:
        lines.append(f"  in2={in2!r} v2={v2}")
    if in3 is not None:
        lines.append(f"  in3={in3!r} v3={v3}")
    if exc is not None:
        lines.append(f"  cause={type(exc).__name__}: {exc}")
    return NH3H2OCallError("\n".join(lines))


def _normalize_state_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """Return a state dictionary with numeric-looking values converted to floats.

Boolean values, strings, and metadata are preserved."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        kk = str(k)
        if isinstance(v, bool):
            out[kk] = bool(v)
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[kk] = float(v)
            continue
        if isinstance(v, numbers.Real):
            try:
                out[kk] = float(v)
            except Exception:
                out[kk] = v
            continue
        out[kk] = v
    return out


def _state_from_tpx(T: float, P: float, X: float, *, strict: bool) -> dict[str, Any]:
    fn = _import_ammonia_water_props_tpx()
    try:
        # Prefer keyword arguments (more robust to signature drift)
        res = fn(T_K=T, P_Pa=P, X=X, strict=bool(strict))  # type: ignore[misc]
        if not isinstance(res, Mapping):
            raise TypeError(f"ammonia_water.props_tpx returned {type(res).__name__}, expected Mapping")
        res2 = _normalize_state_dict(res)
    except Exception as e:
        raise _wrap_call_error(
            what="props_tpx(T,P,X)",
            out=None,
            out_raw=None,
            T=T,
            P=P,
            X=X,
            exc=e,
        ) from e

    # Even if strict=True, keep a consistent check for ok=0
    ok = res2.get("ok", 1)
    if bool(strict) and isinstance(ok, numbers.Real) and float(ok) == 0.0:
        msg = str(res2.get("error", "ammonia_water returned ok=0"))
        raise _wrap_call_error(
            what="props_tpx(T,P,X)",
            out=None,
            out_raw=None,
            T=T,
            P=P,
            X=X,
            exc=RuntimeError(msg),
        )

    return res2


def _state_from_txq(T: float, X: float, Q: float, *, strict: bool) -> dict[str, Any]:
    """Return a state dictionary from T, X, and Q inputs.

This helper supports the EES-style TXQ call shape used by some surface-tension
examples. Several keyword spellings are attempted for compatibility with the
underlying ammonia-water implementation."""
    fn = _import_ammonia_water_props_tpx()
    last: Exception | None = None

    # Try common kw variants to tolerate drift:
    # - Q vs q
    # - X vs x
    for kwargs in (
        {"T_K": T, "X": X, "Q": Q, "strict": bool(strict)},
        {"T_K": T, "X": X, "q": Q, "strict": bool(strict)},
        {"T_K": T, "x": X, "Q": Q, "strict": bool(strict)},
        {"T_K": T, "x": X, "q": Q, "strict": bool(strict)},
    ):
        try:
            res = fn(**kwargs)  # type: ignore[misc]
            if not isinstance(res, Mapping):
                raise TypeError(f"ammonia_water.props_tpx returned {type(res).__name__}, expected Mapping")
            res2 = _normalize_state_dict(res)

            ok = res2.get("ok", 1)
            if bool(strict) and isinstance(ok, numbers.Real) and float(ok) == 0.0:
                msg = str(res2.get("error", "ammonia_water returned ok=0"))
                raise RuntimeError(msg)

            return res2
        except TypeError as e:
            # signature mismatch → try next variant
            last = e
            continue
        except Exception as e:
            # real runtime failure
            raise _wrap_call_error(
                what="props_tpx(T,X,Q)",
                out=None,
                out_raw=None,
                T=T,
                P=None,
                X=X,
                Q=Q,
                exc=e,
            ) from e

    # If we got here: all attempts were signature mismatches.
    raise _wrap_call_error(
        what="props_tpx(T,X,Q)",
        out=None,
        out_raw=None,
        T=T,
        P=None,
        X=X,
        Q=Q,
        exc=last or TypeError("ammonia_water.props_tpx does not accept TXQ-style inputs"),
    )


def _extract_float(res: Mapping[str, Any], key: str) -> float:
    if key not in res:
        raise KeyError(f"NH3H2O output key {key!r} not present in result.")
    v = res[key]
    try:
        return float(v)
    except Exception as e:
        raise TypeError(f"NH3H2O output {key!r} is not float-convertible: {v!r}") from e


def _extract_float_any(res: Mapping[str, Any], key: str) -> float:
    """Extract a float output from a state dictionary.

Known alternate key names are tried before raising ``KeyError``."""
    if key in res:
        return _extract_float(res, key)

    # If key itself is an alias target with fallbacks, try those
    cands = _OUTPUT_FALLBACKS.get(key, ())
    for k in cands:
        if k in res:
            return _extract_float(res, k)

    raise KeyError(
        f"NH3H2O output key {key!r} not present in result "
        f"(and no fallbacks matched). Available keys: {sorted(map(str, res.keys()))}"
    )


def _extract_str(res: Mapping[str, Any], key: str) -> str:
    if key not in res:
        raise KeyError(f"NH3H2O string key {key!r} not present in result.")
    return str(res[key])


def _extract_str_any(res: Mapping[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        if k in res:
            return str(res[k])
    raise KeyError(f"NH3H2O string keys {list(keys)!r} not present. Available keys: {sorted(map(str, res.keys()))}")


# ------------------------------ primary API (TPX) ------------------------------

def state_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> dict[str, Any]:
    """Return the full NH3-H2O state dictionary for T, P, and X.

When ``strict`` is true, failures or an ``ok`` value of zero raise
``NH3H2OCallError``. When ``strict`` is false, the underlying result dictionary
is returned for caller-side inspection."""
    T = _to_float("T_K", T_K)
    P = _to_float("P_Pa", P_Pa)
    Xv = _to_massfrac("X", X)
    return _state_from_tpx(T, P, Xv, strict=bool(strict))


def ok_tpx(T_K: float, P_Pa: float, X: float) -> bool:
    """Return a quick success flag using ``strict=False`` internally."""
    st = state_tpx(T_K, P_Pa, X, strict=False)
    ok = st.get("ok", 0)
    try:
        return bool(int(ok))  # type: ignore[arg-type]
    except Exception:
        return False


def phase_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> str:
    """Return the phase string reported by the NH3-H2O model.

The value is intended for reporting and diagnostics, not as a scalar equation
solver residual."""
    st = state_tpx(T_K, P_Pa, X, strict=bool(strict))
    try:
        # tolerate minor key drift
        return _extract_str_any(st, ("phase", "Phase", "region", "Region"))
    except Exception as e:
        raise _wrap_call_error(
            what="phase_tpx",
            out="phase",
            out_raw="phase",
            T=float(st.get("T_K")) if isinstance(st.get("T_K", None), numbers.Real) else None,
            P=float(st.get("P_Pa")) if isinstance(st.get("P_Pa", None), numbers.Real) else None,
            X=float(st.get("X")) if isinstance(st.get("X", None), numbers.Real) else None,
            exc=e,
        ) from e


def prop_tpx(out: str, T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    """Return one float-valued NH3-H2O property at T, P, and X."""
    out_raw = str(out)
    out_key = _norm_out(out_raw)

    T = _to_float("T_K", T_K)
    P = _to_float("P_Pa", P_Pa)
    Xv = _to_massfrac("X", X)

    res = _state_from_tpx(T, P, Xv, strict=bool(strict))
    try:
        y = _extract_float_any(res, out_key)
    except Exception as e:
        raise _wrap_call_error(
            what="prop_tpx",
            out=out_key,
            out_raw=out_raw,
            T=T,
            P=P,
            X=Xv,
            exc=e,
        ) from e

    if bool(strict) and not _finite(y):
        raise _wrap_call_error(
            what="prop_tpx",
            out=out_key,
            out_raw=out_raw,
            T=T,
            P=P,
            X=Xv,
            exc=RuntimeError(f"Non-finite result for {out_key!r}: {y!r}"),
        )

    return float(y)


def props_multi_tpx(
    outputs: Sequence[str],
    T_K: float,
    P_Pa: float,
    X: float,
    *,
    strict: bool = True,
) -> dict[str, float]:
    """Return several float-valued NH3-H2O properties for one TPX state.

The returned dictionary keeps the output keys exactly as requested by the caller."""
    st = state_tpx(T_K, P_Pa, X, strict=bool(strict))
    out: dict[str, float] = {}
    for k in outputs:
        k_str = str(k)
        key = _norm_out(k_str)
        try:
            out[k_str] = float(_extract_float_any(st, key))
        except Exception as e:
            raise _wrap_call_error(
                what="props_multi_tpx",
                out=key,
                out_raw=k_str,
                T=float(st.get("T_K")) if isinstance(st.get("T_K", None), numbers.Real) else None,
                P=float(st.get("P_Pa")) if isinstance(st.get("P_Pa", None), numbers.Real) else None,
                X=float(st.get("X")) if isinstance(st.get("X", None), numbers.Real) else None,
                exc=e,
            ) from e
    return out


def _iter_calls(calls: Iterable[Any]) -> Iterable[NH3H2OCall]:
    """Yield ``NH3H2OCall`` objects from dataclasses, mappings, or tuples.

The conversion is forgiving so batch helpers can accept common user-facing
shapes without changing the public function signature."""
    for c in calls:
        if isinstance(c, NH3H2OCall):
            yield c
            continue
        if isinstance(c, Mapping):
            try:
                yield NH3H2OCall(
                    out=str(c["out"]),
                    in1=str(c["in1"]),
                    v1=float(c["v1"]),
                    in2=str(c["in2"]),
                    v2=float(c["v2"]),
                    in3=str(c["in3"]),
                    v3=float(c["v3"]),
                    strict=bool(c.get("strict", True)),
                )
                continue
            except Exception as e:
                raise ValueError(f"Invalid NH3H2OCall mapping: {c!r}") from e
        if isinstance(c, (tuple, list)) and (len(c) == 7 or len(c) == 8):
            try:
                if len(c) == 7:
                    out, in1, v1, in2, v2, in3, v3 = c
                    strict = True
                else:
                    out, in1, v1, in2, v2, in3, v3, strict = c
                yield NH3H2OCall(
                    out=str(out),
                    in1=str(in1),
                    v1=float(v1),
                    in2=str(in2),
                    v2=float(v2),
                    in3=str(in3),
                    v3=float(v3),
                    strict=bool(strict),
                )
                continue
            except Exception as e:
                raise ValueError(f"Invalid NH3H2OCall tuple/list: {c!r}") from e
        raise ValueError(f"Invalid NH3H2OCall item: {c!r}")


def batch_prop_tpx(calls: Iterable[NH3H2OCall]) -> list[float]:
    """Execute a batch of NH3-H2O property calls and return floats."""
    ys: list[float] = []
    for c in _iter_calls(calls):
        ys.append(
            NH3H2O(
                c.out,
                c.in1, c.v1,
                c.in2, c.v2,
                c.in3, c.v3,
                strict=bool(c.strict),
            )
        )
    return ys


def batch_state_tpx(states: Iterable[tuple[float, float, float] | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return full state dictionaries for several TPX states.

Each item may be a ``(T_K, P_Pa, X)`` tuple or a mapping with equivalent keys."""
    out: list[dict[str, Any]] = []
    for item in states:
        if isinstance(item, Mapping):
            # tolerate common keys
            T = item.get("T_K", item.get("T", None))
            P = item.get("P_Pa", item.get("P", None))
            X = item.get("X", item.get("x", item.get("w", None)))
            if T is None or P is None or X is None:
                raise ValueError(f"Invalid state mapping (need T_K/T, P_Pa/P, X): {item!r}")
            out.append(state_tpx(float(T), float(P), float(X), strict=True))
            continue
        if isinstance(item, (tuple, list)) and len(item) == 3:
            T, P, X = item
            out.append(state_tpx(float(T), float(P), float(X), strict=True))
            continue
        raise ValueError(f"Invalid state item: {item!r}")
    return out


# ------------------------------ convenience scalar wrappers ------------------------------

def h_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    return prop_tpx("h", T_K, P_Pa, X, strict=bool(strict))


def s_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    return prop_tpx("s", T_K, P_Pa, X, strict=bool(strict))


def u_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    return prop_tpx("u", T_K, P_Pa, X, strict=bool(strict))


def v_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    return prop_tpx("v", T_K, P_Pa, X, strict=bool(strict))


def rho_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    return prop_tpx("rho", T_K, P_Pa, X, strict=bool(strict))


def q_tpx(T_K: float, P_Pa: float, X: float, *, strict: bool = True) -> float:
    return prop_tpx("q", T_K, P_Pa, X, strict=bool(strict))


# ------------------------------ EES PropsSI-like shim ------------------------------

def _pairs_to_dict(args: Sequence[Any]) -> dict[str, float]:
    if len(args) % 2 != 0:
        raise ValueError(f"NH3H2OPropsSI expects (key,value) pairs. Got odd args: {args!r}")

    out: dict[str, float] = {}
    it = iter(args)
    for k, v in zip(it, it):
        kk = _norm_in(str(k))
        # mass fraction sanity for X only; other inputs just float-check
        if kk == "X":
            out[kk] = _to_massfrac(kk, v)
        else:
            out[kk] = _to_float(kk, v)
    return out


# EES doc output designators for NH3H2O (plus transport shorthands)
_EES_OUT_MAP: Mapping[str, str] = {
    # thermo
    "H": "h",
    "S": "s",
    "U": "u",
    "V": "v",
    "D": "rho",
    "RHO": "rho",
    "T": "T_K",
    "P": "P_Pa",
    "Q": "q",
    "X": "X",
    # heat capacities (optional)
    "CP": "cp",
    "CV": "cv",
    # transport
    "K": "k",
    "MU": "mu",
    "SIGMA": "sigma",
}


def NH3H2OPropsSI(out: Any, *args: Any, strict: bool = True) -> float:
    """Return one NH3-H2O property using an EES-style scalar signature.

The function accepts key-value input pairs. It supports TPX inputs for the usual
state calls and TXQ inputs for EES-style surface-tension calls."""
    out_raw = str(out).strip()
    out_u = out_raw.upper()
    # If user passed full keys ("h","k","sigma") accept them too.
    canonical = _EES_OUT_MAP.get(out_u, out_raw)
    out_key = _norm_out(str(canonical))

    kv = _pairs_to_dict(list(args))

    T = kv.get("T", None)
    P = kv.get("P", None)
    X = kv.get("X", None)
    Q = kv.get("Q", None)

    # trivial echo / passthrough for requested inputs
    if out_u == "X" and X is not None:
        return float(X)
    if out_u == "T" and T is not None:
        return float(T)
    if out_u == "P" and P is not None:
        return float(P)

    try:
        # Dispatch by which 3 independent props we have:
        if (T is not None) and (P is not None) and (X is not None):
            # TPX path
            st = _state_from_tpx(float(T), float(P), float(X), strict=bool(strict))
            return float(_extract_float_any(st, out_key))

        if (T is not None) and (X is not None) and (Q is not None):
            # TXQ path (used in EES docs for surface tension)
            st = _state_from_txq(float(T), float(X), float(Q), strict=bool(strict))
            return float(_extract_float_any(st, out_key))

        raise ValueError(
            "NH3H2OPropsSI requires 3 independent thermodynamic_properties. Supported triplets:\n"
            "  - (T, P, X)\n"
            "  - (T, X, Q)\n"
            f"Got keys: {sorted(kv.keys())}"
        )

    except Exception as e:
        if isinstance(e, NH3H2OCallError):
            raise
        raise _wrap_call_error(
            what="NH3H2OPropsSI",
            out=out_key,
            out_raw=out_raw,
            T=float(T) if T is not None else None,
            P=float(P) if P is not None else None,
            X=float(X) if X is not None else None,
            Q=float(Q) if Q is not None else None,
            exc=e,
        ) from e


# alias (nice for injection / backwards compatibility)
nh3h2o_props_si = NH3H2OPropsSI


# ------------------------------ CoolProp-like shims ------------------------------

def _parse_tpx_inputs(
    in1: str, v1: float,
    in2: str, v2: float,
    in3: str, v3: float,
) -> tuple[float, float, float]:
    k1 = _norm_in(in1)
    k2 = _norm_in(in2)
    k3 = _norm_in(in3)

    keys = (k1, k2, k3)
    if len(set(keys)) != 3:
        raise ValueError(f"NH3H2O requires distinct inputs T,P,X. Got duplicate keys: {keys}")

    vals = {
        k1: _to_float("v1", v1),
        k2: _to_float("v2", v2),
        k3: _to_float("v3", v3),
    }

    missing = [k for k in ("T", "P", "X") if k not in vals]
    if missing:
        raise ValueError(
            f"NH3H2O requires inputs T, P, X. Missing: {missing}. Got keys: {sorted(vals.keys())}"
        )

    # X mass fraction sanity here too
    X = _to_massfrac("X", vals["X"])
    return float(vals["T"]), float(vals["P"]), float(X)


def NH3H2O(
    out: str,
    in1: str, v1: float,
    in2: str, v2: float,
    in3: str, v3: float,
    *,
    strict: bool = True,
) -> float:
    """Return one NH3-H2O property using a CoolProp-like TPX signature.

The three input key-value pairs are order agnostic but must resolve to T, P, and
X. TXQ calls are intentionally handled by ``NH3H2OPropsSI``."""
    out_raw = str(out)
    out_key = _norm_out(out_raw)
    try:
        T, P, X = _parse_tpx_inputs(in1, v1, in2, v2, in3, v3)
        return prop_tpx(out_key, T, P, X, strict=bool(strict))
    except Exception as e:
        if isinstance(e, NH3H2OCallError):
            raise
        raise _wrap_call_error(
            what="NH3H2O",
            out=out_key,
            out_raw=out_raw,
            T=None,
            P=None,
            X=None,
            in1=_norm_in(in1),
            v1=float(v1),
            in2=_norm_in(in2),
            v2=float(v2),
            in3=_norm_in(in3),
            v3=float(v3),
            exc=e,
        ) from e


def NH3H2O_STATE(
    in1: str, v1: float,
    in2: str, v2: float,
    in3: str, v3: float,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Return the full state dictionary using a CoolProp-like TPX signature."""
    try:
        T, P, X = _parse_tpx_inputs(in1, v1, in2, v2, in3, v3)
        return state_tpx(T, P, X, strict=bool(strict))
    except Exception as e:
        if isinstance(e, NH3H2OCallError):
            raise
        raise _wrap_call_error(
            what="NH3H2O_STATE",
            out=None,
            out_raw=None,
            T=None,
            P=None,
            X=None,
            in1=_norm_in(in1),
            v1=float(v1),
            in2=_norm_in(in2),
            v2=float(v2),
            in3=_norm_in(in3),
            v3=float(v3),
            exc=e,
        ) from e


def NH3H2O_TPX(out: str, T: float, P: float, X: float, *, strict: bool = True) -> float:
    """Return one property from direct T, P, and X inputs."""
    return prop_tpx(out, T, P, X, strict=bool(strict))


def NH3H2O_STATE_TPX(T: float, P: float, X: float, *, strict: bool = True) -> dict[str, Any]:
    """Return the full state dictionary from direct T, P, and X inputs."""
    return state_tpx(T, P, X, strict=bool(strict))
