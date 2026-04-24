# thermo_props/librh2o_ashrae_backend.py
from __future__ import annotations

"""
thermo_props.librh2o_ashrae_backend

Pure-Python property correlations for **LiBr–H2O (aqueous lithium bromide) solutions**
in the ASHRAE / McNeely style.

Primary scope (what we need for K&N absorption-cycle problems)
-------------------------------------------------------------
1) Equilibrium vapor pressure of water over LiBr solution:
       P_eq(T, x)

2) Solution enthalpy:
       h(T, x)

3) Concentration solved from equilibrium:
       x(T, P)  such that P_eq(T, x) = P

Optional helpers
----------------
- T solved from equilibrium:  T(P, x)
- T solved from enthalpy:     T(h, x)
- Density / Cp:
    * If CoolProp INCOMP is available, we can optionally call it for rho/cp
      (still "liquid-only", but we keep it in a safe region).
    * If CoolProp is not available, we provide a simple, documented approximation
      intended mainly for pump-work estimates, which are tiny in LiBr chillers.

Units and conventions
---------------------
Inputs:
- T : Kelvin [K]
- P : Pascal [Pa]
- x : LiBr mass fraction in solution [kg_LiBr / kg_solution], 0..1

Outputs:
- P_eq : Pascal [Pa]
- h    : J/kg_solution
- rho  : kg/m^3
- cp   : J/kg/K

Correlation details (from your shared reference PDF)
----------------------------------------------------
Enthalpy (kJ/kg):
    H = Σ A_n X^n + t Σ B_n X^n + t^2 Σ C_n X^n
where:
    X = 100*x  (mass percent)
    t = T[K] - 273.15  (°C)

Vapor pressure (kPa):
    log10(P_kPa) = C + D/rT + E/rT^2
    rT[K] = (t - Σ B_n X^n) / (Σ A_n X^n) + 273.16
where X,t are as above.

Recommended validity ranges
---------------------------
- x ≈ 0.40 .. 0.70 (40–70% LiBr by mass)
- T ≈ 288.15 .. 438.15 K (15–165°C)

Outside these bounds, the vapor-pressure model can become ill-conditioned
(e.g., ΣA_n X^n crossing zero), so the default solvers restrict to a safe x-range.

Design goal
-----------
Dependency-light at import time, designed to be called from the equation solver
with predictable domain errors (so the solver can apply penalties instead of crashing).
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import math
import warnings


# ------------------------------ errors ------------------------------

class LiBrH2OCallError(RuntimeError):
    """Raised when an ASHRAE LiBr–H2O property call fails; message includes full call context."""


class LiBrH2ODomainError(ValueError):
    """Raised when inputs are outside the intended correlation domain."""


# ------------------------------ recommended domains ------------------------------

X_MIN_RECOMMENDED = 0.40   # mass fraction
X_MAX_RECOMMENDED = 0.70

T_MIN_RECOMMENDED = 273.15 + 15.0   # 288.15 K
T_MAX_RECOMMENDED = 273.15 + 165.0  # 438.15 K

# Hard bounds (still conservative; prevents obvious nonsense)
X_MIN_HARD = 0.0
X_MAX_HARD = 0.75

T_MIN_HARD = 173.15  # -100C (keeps solvers from probing negative K)
T_MAX_HARD = 600.0


# ------------------------------ internal helpers ------------------------------

def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _check_T(T: float) -> None:
    if not _finite(T):
        raise LiBrH2ODomainError(f"T must be finite [K]. Got {T!r}")
    if T <= 0.0:
        raise LiBrH2ODomainError(f"T must be > 0 K. Got {T!r}")
    if T < T_MIN_HARD or T > T_MAX_HARD:
        raise LiBrH2ODomainError(
            f"T={T:.6g} K is outside hard bounds [{T_MIN_HARD}, {T_MAX_HARD}] K."
        )


def _check_x(x: float) -> None:
    if not _finite(x):
        raise LiBrH2ODomainError(f"x must be finite mass fraction. Got {x!r}")
    if x < X_MIN_HARD or x > X_MAX_HARD:
        raise LiBrH2ODomainError(
            f"x={x:.6g} is outside hard bounds [{X_MIN_HARD}, {X_MAX_HARD}]."
        )


def _warn_if_extrapolating(T: float, x: float) -> None:
    if (T < T_MIN_RECOMMENDED or T > T_MAX_RECOMMENDED or x < X_MIN_RECOMMENDED or x > X_MAX_RECOMMENDED):
        warnings.warn(
            "LiBr–H2O ASHRAE correlations called outside recommended domain "
            f"(T={T:.3f} K, x={x:.4f}). Results may be less reliable.",
            RuntimeWarning,
            stacklevel=2,
        )


def _poly_sum(coeffs: Sequence[float], X: float) -> float:
    """Return Σ coeffs[n] * X^n (n=0..len-1) using Horner for stability."""
    y = 0.0
    for c in reversed(coeffs):
        y = y * X + float(c)
    return y


def _bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise LiBrH2ODomainError(
            "Root not bracketed:\n"
            f"  a={a}, f(a)={fa}\n"
            f"  b={b}, f(b)={fb}"
        )

    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def _bracket_scan(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    n: int = 120,
) -> tuple[float, float]:
    if n < 2:
        raise ValueError("n must be >= 2")
    x_prev = lo
    f_prev = f(x_prev)
    step = (hi - lo) / (n - 1)
    for i in range(1, n):
        x = lo + i * step
        fx = f(x)
        if f_prev == 0.0:
            return x_prev, x_prev
        if fx == 0.0:
            return x, x
        if f_prev * fx < 0.0:
            return x_prev, x
        x_prev, f_prev = x, fx
    raise LiBrH2ODomainError(
        "Could not bracket root in scan:\n"
        f"  lo={lo}, hi={hi}, n={n}\n"
        f"  f(lo)={f(lo)}, f(hi)={f(hi)}"
    )


# ------------------------------ ASHRAE / McNeely coefficients ------------------------------

_H_A = (-2024.33, 163.309, -4.88161, 6.302948e-2, -2.913705e-4)
_H_B = (18.2829, -1.1691757, 3.248041e-2, -4.034184e-4, 1.8520569e-6)
_H_C = (-3.7008214e-2, 2.8877666e-3, -8.1313015e-5, 9.9116628e-7, -4.4441207e-9)

_RT_A = (-2.00755, 0.16976, -3.133362e-3, 1.97668e-5)
_RT_B = (124.937, -7.71649, 0.152286, -7.95090e-4)

_LOGP_C = 7.05
_LOGP_D = -1596.49
_LOGP_E = -104095.5


# ------------------------------ primary correlation functions ------------------------------

def h_LiBrH2O(T: float, x: float, *, warn_extrapolation: bool = False) -> float:
    """
    Solution enthalpy h(T,x) [J/kg_solution] using ASHRAE-style polynomial.
    """
    _check_T(T)
    _check_x(x)
    if warn_extrapolation:
        _warn_if_extrapolating(T, x)

    X = 100.0 * float(x)
    t = float(T) - 273.15

    A = _poly_sum(_H_A, X)
    B = _poly_sum(_H_B, X)
    C = _poly_sum(_H_C, X)

    H_kJkg = A + t * B + (t * t) * C
    h = 1000.0 * H_kJkg
    if not _finite(h):
        raise LiBrH2OCallError(f"Non-finite enthalpy computed for T={T}, x={x}: h={h!r}")
    return float(h)


def _rT_from_Tx(T: float, x: float, *, warn_extrapolation: bool = False) -> float:
    """
    Refrigerant temperature rT [K] used internally by the vapor pressure correlation.
    """
    _check_T(T)
    _check_x(x)
    if warn_extrapolation:
        _warn_if_extrapolating(T, x)

    X = 100.0 * float(x)
    t = float(T) - 273.15

    denom = _poly_sum(_RT_A, X)
    numer = t - _poly_sum(_RT_B, X)

    if abs(denom) < 1e-12:
        raise LiBrH2ODomainError(
            "Vapor-pressure correlation is ill-conditioned: ΣA_n X^n ≈ 0.\n"
            f"  T={T} K, x={x}, X%={X}\n"
            f"  denom={denom}"
        )

    rT = numer / denom + 273.16
    if rT <= 0.0 or not _finite(rT):
        raise LiBrH2OCallError(f"Computed invalid rT={rT!r} for T={T}, x={x}")
    return float(rT)


def P_eq_LiBrH2O(T: float, x: float, *, warn_extrapolation: bool = False) -> float:
    """
    Equilibrium vapor pressure over LiBr solution P_eq(T,x) [Pa].
    """
    rT = _rT_from_Tx(T, x, warn_extrapolation=warn_extrapolation)

    log10_P_kPa = _LOGP_C + _LOGP_D / rT + _LOGP_E / (rT * rT)

    if log10_P_kPa > 20.0 or log10_P_kPa < -50.0:
        raise LiBrH2ODomainError(
            "Vapor-pressure correlation produced an extreme log10(P_kPa).\n"
            f"  T={T} K, x={x}, rT={rT} K, log10(P_kPa)={log10_P_kPa}"
        )

    P_kPa = 10.0 ** log10_P_kPa
    P_Pa = 1000.0 * P_kPa
    if P_Pa <= 0.0 or not _finite(P_Pa):
        raise LiBrH2OCallError(f"Computed invalid vapor pressure P={P_Pa!r} for T={T}, x={x}")
    return float(P_Pa)


def x_LiBrH2O(
    T: float,
    P: float,
    *,
    x_lo: float = X_MIN_RECOMMENDED,
    x_hi: float = X_MAX_HARD,
    tol: float = 1e-10,
    warn_extrapolation: bool = False,
) -> float:
    """
    Solve x such that P_eq(T,x) = P.
    """
    _check_T(T)
    if not _finite(P) or P <= 0.0:
        raise LiBrH2ODomainError(f"P must be positive finite [Pa]. Got {P!r}")
    _check_x(x_lo)
    _check_x(x_hi)
    if x_hi <= x_lo:
        raise ValueError("x_hi must be > x_lo")

    f = lambda xx: P_eq_LiBrH2O(T, xx, warn_extrapolation=warn_extrapolation) - float(P)

    try:
        return _bisect(f, float(x_lo), float(x_hi), tol=tol)
    except LiBrH2ODomainError:
        a, b = _bracket_scan(f, float(x_lo), float(x_hi), n=160)
        if a == b:
            return a
        return _bisect(f, a, b, tol=tol)


def T_from_Px(
    P: float,
    x: float,
    *,
    T_lo: float = T_MIN_RECOMMENDED,
    T_hi: float = T_MAX_RECOMMENDED,
    tol: float = 1e-8,
    warn_extrapolation: bool = False,
) -> float:
    """Solve T such that P_eq(T,x)=P."""
    _check_x(x)
    if not _finite(P) or P <= 0.0:
        raise LiBrH2ODomainError(f"P must be positive finite [Pa]. Got {P!r}")
    if T_hi <= T_lo:
        raise ValueError("T_hi must be > T_lo")
    _check_T(T_lo)
    _check_T(T_hi)

    f = lambda TT: P_eq_LiBrH2O(TT, x, warn_extrapolation=warn_extrapolation) - float(P)

    try:
        return _bisect(f, float(T_lo), float(T_hi), tol=tol)
    except LiBrH2ODomainError:
        a, b = _bracket_scan(f, float(T_lo), float(T_hi), n=160)
        if a == b:
            return a
        return _bisect(f, a, b, tol=tol)


def T_from_hx(
    h_target: float,
    x: float,
    *,
    T_lo: float = T_MIN_RECOMMENDED,
    T_hi: float = T_MAX_RECOMMENDED,
    tol: float = 1e-8,
    warn_extrapolation: bool = False,
) -> float:
    """Solve T such that h(T,x)=h_target."""
    _check_x(x)
    if not _finite(h_target):
        raise LiBrH2ODomainError(f"h_target must be finite [J/kg]. Got {h_target!r}")
    if T_hi <= T_lo:
        raise ValueError("T_hi must be > T_lo")
    _check_T(T_lo)
    _check_T(T_hi)

    f = lambda TT: h_LiBrH2O(TT, x, warn_extrapolation=warn_extrapolation) - float(h_target)

    try:
        return _bisect(f, float(T_lo), float(T_hi), tol=tol)
    except LiBrH2ODomainError:
        a, b = _bracket_scan(f, float(T_lo), float(T_hi), n=160)
        if a == b:
            return a
        return _bisect(f, a, b, tol=tol)


# ------------------------------ optional: density/cp ------------------------------

def _coolprop_incomp_density_cp(T: float, x: float) -> tuple[float | None, float | None]:
    """
    Attempt to fetch rho and cp from CoolProp INCOMP::LiBr[x] if CoolProp is installed.
    """
    try:
        from CoolProp.CoolProp import PropsSI  # type: ignore
    except Exception:
        return None, None

    try:
        P_sat = P_eq_LiBrH2O(T, x)
    except Exception:
        P_sat = 0.0
    P_use = max(101325.0, float(P_sat) + 2000.0)

    fluid = f"INCOMP::LiBr[{float(x)}]"
    try:
        rho = float(PropsSI("D", "T", float(T), "P", P_use, fluid))
    except Exception:
        rho = None
    try:
        cp = float(PropsSI("C", "T", float(T), "P", P_use, fluid))
    except Exception:
        cp = None
    return rho, cp


def rho_LiBrH2O(T: float, x: float, *, prefer_coolprop: bool = True) -> float:
    """
    Density ρ(T,x) [kg/m^3].
    """
    _check_T(T)
    _check_x(x)

    if prefer_coolprop:
        rho, _ = _coolprop_incomp_density_cp(T, x)
        if rho is not None and _finite(rho) and rho > 0.0:
            return float(rho)

    t_C = float(T) - 273.15
    rho0 = 1000.0 + 1100.0 * float(x)
    rhoT = rho0 - 0.35 * (t_C - 25.0)
    if rhoT <= 0.0 or not _finite(rhoT):
        raise LiBrH2OCallError(f"Computed invalid density rho={rhoT!r} for T={T}, x={x}")
    return float(rhoT)


def cp_LiBrH2O(T: float, x: float, *, prefer_coolprop: bool = True) -> float:
    """
    Specific heat cp(T,x) [J/kg/K].
    """
    _check_T(T)
    _check_x(x)

    if prefer_coolprop:
        _, cp = _coolprop_incomp_density_cp(T, x)
        if cp is not None and _finite(cp) and cp > 0.0:
            return float(cp)

    cp0 = 4180.0
    cp = cp0 * (1.0 - 0.55 * float(x))
    if cp <= 0.0 or not _finite(cp):
        raise LiBrH2OCallError(f"Computed invalid cp={cp!r} for T={T}, x={x}")
    return float(cp)


# ------------------------------ PropsSI-like front door ------------------------------

_INPUT_ALIASES: Mapping[str, str] = {
    "t": "T",
    "temp": "T",
    "temperature": "T",
    "p": "P",
    "press": "P",
    "pressure": "P",
    "x": "X",
    "w": "X",
    "massfraction": "X",
    "h": "H",
    "enthalpy": "H",
    "rho": "D",
    "d": "D",
    "density": "D",
}

_OUTPUT_ALIASES: Mapping[str, str] = {
    **_INPUT_ALIASES,
    "peq": "P",
    "cp": "C",
    "c": "C",
}


def _norm_in(k: str) -> str:
    s = str(k).strip()
    if not s:
        raise ValueError("Input key is empty")
    return _INPUT_ALIASES.get(s.lower(), s)


def _norm_out(k: str) -> str:
    s = str(k).strip()
    if not s:
        raise ValueError("Output key is empty")
    return _OUTPUT_ALIASES.get(s.lower(), s)


@dataclass(frozen=True)
class LiBrCall:
    out: str
    in1: str
    v1: float
    in2: str
    v2: float


def librh2o_available() -> bool:
    """This backend is pure-python; always available."""
    return True


def librh2o_props_si(out: str, in1: str, v1: float, in2: str, v2: float) -> float:
    """
    PropsSI-like wrapper for LiBr–H2O properties.

    Supported pairs:
    - (T, X) -> outputs: P, H, D, C
    - (T, P) -> output: X
    - (P, X) -> output: T
    - (H, X) -> output: T
    """
    o = _norm_out(out)
    i1 = _norm_in(in1)
    i2 = _norm_in(in2)
    a = float(v1)
    b = float(v2)

    key = tuple(sorted([i1, i2]))

    try:
        if key == ("T", "X"):
            T = a if i1 == "T" else b
            x = a if i1 == "X" else b
            if o == "P":
                return P_eq_LiBrH2O(T, x)
            if o == "H":
                return h_LiBrH2O(T, x)
            if o == "D":
                return rho_LiBrH2O(T, x)
            if o == "C":
                return cp_LiBrH2O(T, x)
            raise ValueError(f"Unsupported output {out!r} for inputs (T,X).")

        if key == ("P", "T"):
            T = a if i1 == "T" else b
            P = a if i1 == "P" else b
            if o != "X":
                raise ValueError("For inputs (T,P) the only supported output is X.")
            return x_LiBrH2O(T, P)

        if key == ("P", "X"):
            P = a if i1 == "P" else b
            x = a if i1 == "X" else b
            if o != "T":
                raise ValueError("For inputs (P,X) the only supported output is T.")
            return T_from_Px(P, x)

        if key == ("H", "X"):
            h = a if i1 == "H" else b
            x = a if i1 == "X" else b
            if o != "T":
                raise ValueError("For inputs (H,X) the only supported output is T.")
            return T_from_hx(h, x)

        raise ValueError(f"Unsupported input pair ({in1!r},{in2!r}).")

    except (LiBrH2ODomainError, LiBrH2OCallError) as e:
        raise LiBrH2OCallError(
            "LiBrH2O property call failed.\n"
            f"  out={out!r}\n"
            f"  in1={in1!r} v1={v1}\n"
            f"  in2={in2!r} v2={v2}\n"
            f"  normalized: out={o!r}, in1={i1!r}, in2={i2!r}"
        ) from e


def librh2o_props_multi(
    outputs: Sequence[str],
    in1: str,
    v1: float,
    in2: str,
    v2: float,
) -> dict[str, float]:
    """Compute multiple outputs for the same LiBr–H2O state."""
    out: dict[str, float] = {}
    for k in outputs:
        out[str(k)] = librh2o_props_si(str(k), in1, v1, in2, v2)
    return out


def batch_librh2o_props(calls: Iterable[LiBrCall]) -> list[float]:
    """Execute a batch of LiBr–H2O PropsSI-like calls."""
    ys: list[float] = []
    for c in calls:
        ys.append(librh2o_props_si(c.out, c.in1, c.v1, c.in2, c.v2))
    return ys


# ------------------------------ back-compat aliases ------------------------------

def x_LiBrH2O_from_TP(T: float, P: float, **kwargs: float) -> float:
    """Back-compat alias for x_LiBrH2O(T, P, ...)."""
    return x_LiBrH2O(T, P, **kwargs)


def T_LiBrH2O_from_Px(P: float, x: float, **kwargs: float) -> float:
    """Back-compat alias for T_from_Px(P, x, ...)."""
    return T_from_Px(P, x, **kwargs)


def T_LiBrH2O_from_hx(h_target: float, x: float, **kwargs: float) -> float:
    """Back-compat alias for T_from_hx(h_target, x, ...)."""
    return T_from_hx(h_target, x, **kwargs)


def LiBrPropsSI(out: str, in1: str, v1: float, in2: str, v2: float) -> float:
    """Back-compat CoolProp-style alias."""
    return librh2o_props_si(out, in1, v1, in2, v2)


def LiBrH2OPropsSI(out: str, in1: str, v1: float, in2: str, v2: float) -> float:
    """Back-compat CoolProp-style alias."""
    return librh2o_props_si(out, in1, v1, in2, v2)


__all__ = [
    "LiBrH2OCallError",
    "LiBrH2ODomainError",
    "X_MIN_RECOMMENDED",
    "X_MAX_RECOMMENDED",
    "T_MIN_RECOMMENDED",
    "T_MAX_RECOMMENDED",
    "X_MIN_HARD",
    "X_MAX_HARD",
    "T_MIN_HARD",
    "T_MAX_HARD",
    "h_LiBrH2O",
    "P_eq_LiBrH2O",
    "x_LiBrH2O",
    "T_from_Px",
    "T_from_hx",
    "rho_LiBrH2O",
    "cp_LiBrH2O",
    "librh2o_available",
    "librh2o_props_si",
    "librh2o_props_multi",
    "batch_librh2o_props",
    "LiBrCall",
    # back-compat aliases
    "x_LiBrH2O_from_TP",
    "T_LiBrH2O_from_Px",
    "T_LiBrH2O_from_hx",
    "LiBrPropsSI",
    "LiBrH2OPropsSI",
]
