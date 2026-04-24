# librh2o/absorption_cooling.py
"""
Sandbox prototype for Klein & Nellis Example 9.4-1 (Waste-Heat Driven Absorption Cooling)
using CoolProp:

- Refrigerant side: Water via HEOS (PropsSI with T/Q or H/P pairs)
- Solution side: LiBr-H2O via INCOMP::LiBr[x]  (x = LiBr mass fraction)

Validated fact:
- VLE hook for LiBr solution exists in CoolProp INCOMP:
    P_eq(T, x) = PropsSI("P", "T", T, "Q", 0, f"INCOMP::LiBr[{x}]")

Important implementation detail:
- INCOMP correlations are "liquid-only". When requesting liquid thermodynamic properties (H, D, C),
  CoolProp errors if supplied P < P_eq(T,x). Pressure influence is weak, so we select a
  "safe liquid pressure" very close to equilibrium:
    P_use = max(P_min, P_eq(T,x) + margin)

We intentionally keep this as a sandbox script (no integration into main tdpy yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

from CoolProp.CoolProp import PropsSI

ATM = 101325.0


# ----------------------------- utils -----------------------------

def K(C: float) -> float:
    return C + 273.15


def C(Kelvin: float) -> float:
    return Kelvin - 273.15


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Robust bisection with clear error if the root is not bracketed."""
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(
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


def bracket_scan(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    n: int = 120,
) -> tuple[float, float]:
    """Find a bracket [a,b] with sign change by scanning a grid."""
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

    raise ValueError(
        "Could not bracket root in scan:\n"
        f"  lo={lo}, hi={hi}, n={n}\n"
        f"  f(lo)={f(lo)}, f(hi)={f(hi)}"
    )


# ----------------------------- water (refrigerant) -----------------------------

def water_Psat(T: float) -> float:
    return PropsSI("P", "T", T, "Q", 0, "Water")


def water_h_TQ(T: float, Q: float) -> float:
    return PropsSI("H", "T", T, "Q", Q, "Water")


def water_h_TP(T: float, P: float) -> float:
    return PropsSI("H", "T", T, "P", P, "Water")


def water_T_from_HP(H: float, P: float) -> float:
    return PropsSI("T", "H", H, "P", P, "Water")


def water_Q_from_HP(H: float, P: float) -> float:
    return PropsSI("Q", "H", H, "P", P, "Water")


# ----------------------------- LiBr solution (INCOMP) -----------------------------

def libr_fluid(x: float) -> str:
    return f"INCOMP::LiBr[{x}]"


def libr_Peq(T: float, x: float) -> float:
    """Equilibrium water vapor pressure over LiBr solution at (T,x)."""
    return PropsSI("P", "T", T, "Q", 0, libr_fluid(x))


def libr_liquid_pressure(T: float, x: float, P_min: float, margin_Pa: float = 1.0) -> float:
    """
    Choose a pressure valid for liquid-only correlations.

    Use the smallest possible margin to stay close to equilibrium while avoiding
    floating-point "P < psat" errors in CoolProp.
    """
    P_sat = libr_Peq(T, x)
    return max(P_min, P_sat + margin_Pa)


def libr_h(T: float, x: float, P_min: float = ATM) -> float:
    P_use = libr_liquid_pressure(T, x, P_min=P_min)
    return PropsSI("H", "T", T, "P", P_use, libr_fluid(x))


def libr_rho(T: float, x: float, P_min: float = ATM) -> float:
    P_use = libr_liquid_pressure(T, x, P_min=P_min)
    return PropsSI("D", "T", T, "P", P_use, libr_fluid(x))


def libr_cp(T: float, x: float, P_min: float = ATM) -> float:
    P_use = libr_liquid_pressure(T, x, P_min=P_min)
    return PropsSI("C", "T", T, "P", P_use, libr_fluid(x))


def x_LiBr_from_TP(T: float, P: float, x_lo: float = 0.0, x_hi: float = 0.75) -> float:
    """Solve Peq(T,x) = P for x."""
    f = lambda x: libr_Peq(T, x) - P
    try:
        return bisect(f, x_lo, x_hi, tol=1e-10)
    except ValueError:
        a, b = bracket_scan(f, x_lo, x_hi, n=160)
        if a == b:
            return a
        return bisect(f, a, b, tol=1e-10)


def T_from_hx(
    h_target: float,
    x: float,
    P_min: float = ATM,
    T_guess: float | None = None,
    span: float = 25.0,
) -> float:
    """
    Invert T from (h_target, x) using bisection.
    Bracket locally around a guess first; fall back to global scan.
    """
    def f(T: float) -> float:
        return libr_h(T, x, P_min=P_min) - h_target

    if T_guess is not None:
        T_lo = max(273.0, T_guess - span)
        T_hi = min(500.0, T_guess + span)
        try:
            return bisect(f, T_lo, T_hi, tol=1e-8)
        except ValueError:
            pass

    a, b = bracket_scan(f, 273.0, 500.0, n=160)
    if a == b:
        return a
    return bisect(f, a, b, tol=1e-8)


# ----------------------------- case + solver -----------------------------

@dataclass
class Case941:
    # Chilled-water side (evaporator load)
    m_dot_w: float = 0.5       # kg/s
    T_w_in_C: float = 12.0
    T_w_out_C: float = 8.0
    DT_evap: float = 3.5       # K approach

    # Condenser/ambient
    T_amb_C: float = 25.0
    DT_cond: float = 5.0       # K approach

    # Absorber/generator temperatures
    T_gen_C: float = 95.0
    T_abs_C: float = 32.0

    # Pump efficiency
    eta_p: float = 0.5

    # Logging
    verbose: bool = True
    print_states: bool = True
    print_balance_terms: bool = True


def run_case(cfg: Case941) -> Dict[str, Any]:
    # ---- chilled-water load ----
    T_w_in = K(cfg.T_w_in_C)
    T_w_out = K(cfg.T_w_out_C)
    h_w_in = water_h_TP(T_w_in, ATM)
    h_w_out = water_h_TP(T_w_out, ATM)
    Q_evap = cfg.m_dot_w * (h_w_in - h_w_out)  # W

    # ---- refrigerant (water) states ----
    # State 2: evaporator exit (sat vapor at T2)
    T2 = T_w_out - cfg.DT_evap
    h2 = water_h_TQ(T2, 1.0)
    P2 = water_Psat(T2)

    # State 4: condenser exit (sat liquid at T4)
    T4 = K(cfg.T_amb_C) + cfg.DT_cond
    h4 = water_h_TQ(T4, 0.0)
    P4 = water_Psat(T4)

    # State 1: throttle (h1=h4), at evaporator pressure
    h1 = h4
    P1 = P2
    T1 = water_T_from_HP(h1, P1)
    Q1 = water_Q_from_HP(h1, P1)

    # State 3: generator vapor to condenser at condenser pressure, generator temperature
    T3 = K(cfg.T_gen_C)
    P3 = P4
    h3 = water_h_TP(T3, P3)

    # Refrigerant mass flow
    m_dot_r = Q_evap / (h2 - h1)

    # ---- solution (LiBr) states ----
    # State 5: absorber outlet (weak solution), equilibrium at (T_abs, P_evap)
    T5 = K(cfg.T_abs_C)
    P5 = P2
    x5 = x_LiBr_from_TP(T5, P5)
    h5 = libr_h(T5, x5, P_min=P5)
    rho5 = libr_rho(T5, x5, P_min=P5)

    # State 6: pump outlet (high pressure)
    P6 = P3
    w_p = (P6 - P5) / (rho5 * cfg.eta_p)  # J/kg
    h6 = h5 + w_p
    x6 = x5
    T6 = T_from_hx(h6, x6, P_min=P6, T_guess=T5, span=10.0)

    # State 7: generator solution outlet (strong solution), equilibrium at (T_gen, P_cond)
    T7 = T3
    P7 = P3
    x7 = x_LiBr_from_TP(T7, P7)
    h7 = libr_h(T7, x7, P_min=P7)

    # State 8: throttled strong solution to absorber pressure (h8=h7, x8=x7)
    P8 = P2
    x8 = x7
    h8 = h7

    if x8 <= x5:
        raise ValueError(
            f"Expected strong solution concentration x8 > x5, got x5={x5}, x8={x8}."
        )

    # ---- absorber mass balances ----
    # m_r + m_gen = m_abs
    # m_gen*x8 = m_abs*x5
    m_dot_abs = m_dot_r / (1.0 - x5 / x8)
    m_dot_gen = m_dot_abs * x5 / x8

    # ---- energy balances ----
    Q_abs = m_dot_r * h2 + m_dot_gen * h8 - m_dot_abs * h5
    Q_gen = m_dot_gen * h7 + m_dot_r * h3 - m_dot_abs * h6
    Q_cond = m_dot_r * (h3 - h4)
    W_p_total = m_dot_abs * w_p

    COP = Q_evap / Q_gen

    states = {
        1: dict(kind="water", T=T1, P=P1, h=h1, Q=Q1),
        2: dict(kind="water", T=T2, P=P2, h=h2, Q=1.0),
        3: dict(kind="water", T=T3, P=P3, h=h3),
        4: dict(kind="water", T=T4, P=P4, h=h4, Q=0.0),
        5: dict(kind="libr",  T=T5, P=P5, h=h5, x=x5, rho=rho5),
        6: dict(kind="libr",  T=T6, P=P6, h=h6, x=x6),
        7: dict(kind="libr",  T=T7, P=P7, h=h7, x=x7),
        8: dict(kind="libr",  P=P8, h=h8, x=x8),
    }

    # Balance residuals (should be ~0; helps debugging)
    evap_resid = Q_evap - m_dot_r * (h2 - h1)
    cond_resid = Q_cond - m_dot_r * (h3 - h4)
    abs_resid = (m_dot_r * h2 + m_dot_gen * h8) - (Q_abs + m_dot_abs * h5)
    gen_resid = (m_dot_abs * h6 + Q_gen) - (m_dot_gen * h7 + m_dot_r * h3)

    return {
        "cfg": cfg,
        "Q_evap_W": Q_evap,
        "Q_abs_W": Q_abs,
        "Q_gen_W": Q_gen,
        "Q_cond_W": Q_cond,
        "COP": COP,
        "W_p_W": W_p_total,
        "w_p_Jkg": w_p,
        "m_dot_r": m_dot_r,
        "m_dot_abs": m_dot_abs,
        "m_dot_gen": m_dot_gen,
        "states": states,
        "residuals": {
            "evap_W": evap_resid,
            "cond_W": cond_resid,
            "abs_W": abs_resid,
            "gen_W": gen_resid,
        },
        "terms": {
            "gen": {
                "m_abs_h6_W": m_dot_abs * h6,
                "m_gen_h7_W": m_dot_gen * h7,
                "m_r_h3_W": m_dot_r * h3,
            },
            "abs": {
                "m_r_h2_W": m_dot_r * h2,
                "m_gen_h8_W": m_dot_gen * h8,
                "m_abs_h5_W": m_dot_abs * h5,
            }
        }
    }


# ----------------------------- reporting -----------------------------

def _fmt(x: float | None, w: int = 12, p: int = 6) -> str:
    if x is None:
        return " " * w
    return f"{x:{w}.{p}f}"


def print_state_table(states: Dict[int, Dict[str, Any]]) -> None:
    print("\n--- State table ---")
    print(" st | kind  |   T [C]   |    P [Pa]    |   h [kJ/kg] |   Q or x    |  rho [kg/m3]")
    print("-" * 88)
    for st in sorted(states.keys()):
        s = states[st]
        kind = s.get("kind", "?")
        T = s.get("T", None)
        P = s.get("P", None)
        h = s.get("h", None)
        Q = s.get("Q", None)
        x = s.get("x", None)
        rho = s.get("rho", None)

        T_C = C(T) if T is not None else None
        h_kJkg = (h / 1000.0) if h is not None else None

        q_or_x = Q if Q is not None else x
        print(
            f"{st:>3d} | {kind:<5s} |"
            f"{_fmt(T_C, w=9, p=3)} |"
            f"{_fmt(P,   w=12, p=3)} |"
            f"{_fmt(h_kJkg, w=11, p=3)} |"
            f"{_fmt(q_or_x, w=11, p=6)} |"
            f"{_fmt(rho, w=11, p=3)}"
        )


def print_balance_terms(res: Dict[str, Any]) -> None:
    Q_evap = res["Q_evap_W"]
    Q_abs = res["Q_abs_W"]
    Q_gen = res["Q_gen_W"]
    Q_cond = res["Q_cond_W"]
    Wp = res["W_p_W"]

    print("\n--- Heat rates ---")
    print(f"Q_evap [kW] = {Q_evap/1000:.6f}")
    print(f"Q_abs  [kW] = {Q_abs/1000:.6f}")
    print(f"Q_gen  [kW] = {Q_gen/1000:.6f}")
    print(f"Q_cond [kW] = {Q_cond/1000:.6f}")
    print(f"W_p    [W]  = {Wp:.6f}")
    print(f"COP         = {res['COP']:.6f}")

    r = res["residuals"]
    print("\n--- Balance residuals (should be ~0) ---")
    print(f"evap resid [W] = {r['evap_W']:.6e}")
    print(f"cond resid [W] = {r['cond_W']:.6e}")
    print(f"abs  resid [W] = {r['abs_W']:.6e}")
    print(f"gen  resid [W] = {r['gen_W']:.6e}")

    print("\n--- Generator term breakdown (W) ---")
    t = res["terms"]["gen"]
    print(f"m_abs*h6 = {t['m_abs_h6_W']:.6f}")
    print(f"m_gen*h7 = {t['m_gen_h7_W']:.6f}")
    print(f"m_r*h3   = {t['m_r_h3_W']:.6f}")
    print(f"Q_gen    = {Q_gen:.6f}")

    print("\n--- Absorber term breakdown (W) ---")
    ta = res["terms"]["abs"]
    print(f"m_r*h2   = {ta['m_r_h2_W']:.6f}")
    print(f"m_gen*h8 = {ta['m_gen_h8_W']:.6f}")
    print(f"m_abs*h5 = {ta['m_abs_h5_W']:.6f}")
    print(f"Q_abs    = {Q_abs:.6f}")


def print_vle_sanity(states: Dict[int, Dict[str, Any]]) -> None:
    T5 = states[5]["T"]; x5 = states[5]["x"]; P2 = states[2]["P"]
    T7 = states[7]["T"]; x7 = states[7]["x"]; P4 = states[4]["P"]

    print("\n--- VLE sanity ---")
    print(f"Absorber:  T5={C(T5):.3f}C, P2={P2:.3f} Pa, Peq(T5,x5)={libr_Peq(T5,x5):.3f} Pa")
    print(f"Generator: T7={C(T7):.3f}C, P4={P4:.3f} Pa, Peq(T7,x7)={libr_Peq(T7,x7):.3f} Pa")


# ----------------------------- run -----------------------------

if __name__ == "__main__":
    cfg = Case941()
    res = run_case(cfg)

    print("=== Waste-Heat Absorption Cooling (LiBr-H2O / Water) ===")
    print(f"Q_evap [kW] = {res['Q_evap_W']/1000:.6f}")
    print(f"Q_gen  [kW] = {res['Q_gen_W']/1000:.6f}")
    print(f"COP         = {res['COP']:.6f}")
    print(f"Pump W [W]   = {res['W_p_W']:.6f}")
    print(f"m_dot_r [kg/s]   = {res['m_dot_r']:.9f}")
    print(f"m_dot_abs [kg/s] = {res['m_dot_abs']:.9f}")
    print(f"m_dot_gen [kg/s] = {res['m_dot_gen']:.9f}")
    print(f"w_p [J/kg]       = {res['w_p_Jkg']:.9f}")

    x5 = res["states"][5]["x"]
    x7 = res["states"][7]["x"]
    print(f"x5 (weak)   = {x5:.12f}")
    print(f"x7 (strong) = {x7:.12f}")

    if cfg.print_states:
        print_state_table(res["states"])

    if cfg.print_balance_terms:
        print_balance_terms(res)

    print_vle_sanity(res["states"])

__all__ = [
    "ATM",
    "Case941",
    "K",
    "C",
    "bisect",
    "bracket_scan",
    "water_Psat",
    "water_h_TQ",
    "water_h_TP",
    "water_T_from_HP",
    "water_Q_from_HP",
    "libr_fluid",
    "libr_Peq",
    "libr_liquid_pressure",
    "libr_h",
    "libr_rho",
    "libr_cp",
    "x_LiBr_from_TP",
    "T_from_hx",
    "run_case",
    "print_state_table",
    "print_balance_terms",
    "print_vle_sanity",
]
