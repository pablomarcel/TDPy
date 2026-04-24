# librh2o/absorption_cooling_ashrae.py
"""
Klein & Nellis Example 9.4-1 (Waste-Heat Driven Absorption Cooling),
BUT with LiBr-H2O solution properties computed using ASHRAE-style correlations
(as shown in: "Thermodynamic thermodynamic_properties of lithium bromide-water solution using Python.pdf"):

Solution-side (LiBr-H2O):
  - Enthalpy: H = sum A(n) X^n + t * sum B(n) X^n + t^2 * sum C(n) X^n  [kJ/kg]
    where t is solution temperature in °C, X is concentration in %.
    Coefficients from Table 01 + code snippet in the PDF.
  - Equilibrium vapor pressure:
        log10(P[kPa]) = C + D/rT + E/rT^2
        rT[K] = (t - sum B(n)X^n) / (sum A(n)X^n) + 273.16
    Coefficients from Table 02 + Eqns (2)-(3) in the PDF.

Refrigerant-side (Water):
  - Use CoolProp Water for h, P_sat, quality, etc. (same as your current script)

NOTE:
  - Pump work needs rho. The reference-PDF excerpts we found don't give rho,
    so we use CoolProp INCOMP density *only* for pump work.
  - Correlation validity (per the PDF text): ~40–70% and ~15–165°C for the ASHRAE-based code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any
import math

from CoolProp.CoolProp import PropsSI

ATM = 101325.0


# ----------------------------- utils -----------------------------

def K(C: float) -> float:
    return C + 273.15


def C_(K_: float) -> float:
    return K_ - 273.15


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
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


def bracket_scan(f: Callable[[float], float], lo: float, hi: float, n: int = 200) -> tuple[float, float]:
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

def water_Psat(TK: float) -> float:
    return PropsSI("P", "T", TK, "Q", 0, "Water")


def water_h_TQ(TK: float, Q: float) -> float:
    return PropsSI("H", "T", TK, "Q", Q, "Water")


def water_h_TP(TK: float, P: float) -> float:
    return PropsSI("H", "T", TK, "P", P, "Water")


def water_T_from_HP(H: float, P: float) -> float:
    return PropsSI("T", "H", H, "P", P, "Water")


def water_Q_from_HP(H: float, P: float) -> float:
    return PropsSI("Q", "H", H, "P", P, "Water")


# ----------------------------- LiBr-H2O (ASHRAE correlations from reference PDF) -----------------------------
# The PDF uses X as concentration in percent (%), t in Celsius (°C), enthalpy in kJ/kg.

# Enthalpy coefficients (Table 01 + provided code snippet)
# NOTE: Table line for C(0) looks like a formatting glitch in the PDF; the code snippet uses 1e-2.
H_A = [-2024.33, 163.309, -4.88161, 6.302948e-2, -2.913705e-4]
H_B = [18.2829, -1.1691757, 3.248041e-2, -4.034184e-4, 1.8520569e-6]
H_C = [-3.7008214e-2, 2.8877666e-3, -8.1313015e-5, 9.9116628e-7, -4.4441207e-9]

# Vapor pressure coefficients (Eqn 2-3 + Table 02)
P_A = [-2.00755, 0.16976, -3.133362e-3, 1.97668e-5]
P_B = [124.937, -7.71649, 0.152286, -7.95090e-4]
P_C = 7.05
P_D = -1596.49
P_E = -104095.5


def _poly(coeffs: list[float], X_pct: float) -> float:
    s = 0.0
    xpow = 1.0
    for c in coeffs:
        s += c * xpow
        xpow *= X_pct
    return s


def libr_h_ashrae(TK: float, x_mass: float) -> float:
    """
    Solution enthalpy [J/kg] using ASHRAE polynomial form (kJ/kg in ref).
    """
    tC = C_(TK)
    X = 100.0 * x_mass

    A = _poly(H_A, X)
    B = _poly(H_B, X)
    Cc = _poly(H_C, X)
    H_kJkg = A + tC * B + (tC * tC) * Cc
    return 1000.0 * H_kJkg


def libr_tC_from_hx_ashrae(H_Jkg: float, x_mass: float) -> float:
    """
    Invert the ASHRAE enthalpy equation for tC given (H, X).
    Since H = A + t B + t^2 C, solve quadratic (preferred).
    Returns tC [°C].
    """
    X = 100.0 * x_mass
    A = _poly(H_A, X)               # kJ/kg
    B = _poly(H_B, X)               # kJ/kg/°C
    Cc = _poly(H_C, X)              # kJ/kg/°C^2

    H_kJkg = H_Jkg / 1000.0
    # Cc*t^2 + B*t + (A - H) = 0
    aa = Cc
    bb = B
    cc = (A - H_kJkg)

    # If nearly linear
    if abs(aa) < 1e-12:
        if abs(bb) < 1e-12:
            raise ValueError("Enthalpy inversion failed: both quadratic and linear terms ~0.")
        return -cc / bb

    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0:
        # Numerical guard: fallback to bisection in plausible range
        def f(tC: float) -> float:
            TK = tC + 273.15
            return libr_h_ashrae(TK, x_mass) - H_Jkg

        a, b = bracket_scan(f, 0.0, 200.0, n=400)
        return bisect(f, a, b, tol=1e-8)

    rdisc = math.sqrt(disc)
    t1 = (-bb + rdisc) / (2.0 * aa)
    t2 = (-bb - rdisc) / (2.0 * aa)

    # Choose the physically plausible root (typical cycle temps ~ 0–200°C)
    candidates = [t for t in (t1, t2) if -50.0 <= t <= 250.0]
    if not candidates:
        # fall back: choose the one closer to typical range
        return t1 if abs(t1 - 80.0) < abs(t2 - 80.0) else t2
    # If both plausible, pick the lower one unless you want super-high solution temps
    return min(candidates)


def libr_Peq_ashrae(TK: float, x_mass: float) -> float:
    """
    Equilibrium vapor pressure over LiBr solution [Pa] using:
      log10(P[kPa]) = C + D/rT + E/rT^2
      rT[K] = (t - sum(B X^n)) / (sum(A X^n)) + 273.16
    """
    tC = C_(TK)
    X = 100.0 * x_mass

    a = _poly(P_A, X)
    b = _poly(P_B, X)
    rT = (tC - b) / a + 273.16  # Kelvin

    log10P_kPa = P_C + (P_D / rT) + (P_E / (rT * rT))
    P_kPa = 10.0 ** log10P_kPa
    return 1000.0 * P_kPa


def x_LiBr_from_TP_ashrae(TK: float, P: float, x_lo: float = 0.40, x_hi: float = 0.75) -> float:
    """
    Solve Peq(T,x) = P for x (mass fraction).
    Default bracket aims at ~ASHRAE validity band.
    """
    f = lambda x: libr_Peq_ashrae(TK, x) - P
    try:
        return bisect(f, x_lo, x_hi, tol=1e-10)
    except ValueError:
        a, b = bracket_scan(f, x_lo, x_hi, n=300)
        if a == b:
            return a
        return bisect(f, a, b, tol=1e-10)


def libr_rho_coolprop(TK: float, x_mass: float, P: float) -> float:
    """
    Density helper from CoolProp INCOMP for pump work only.
    """
    fluid = f"INCOMP::LiBr[{x_mass}]"
    # pick a safe pressure for liquid-only correlations
    # (INCOMP may complain if P < psat at that T,x)
    P_use = max(P, PropsSI("P", "T", TK, "Q", 0, fluid) + 2000.0)
    return PropsSI("D", "T", TK, "P", P_use, fluid)


# ----------------------------- case + solver -----------------------------

@dataclass
class Case941:
    m_dot_w: float = 0.5
    T_w_in_C: float = 12.0
    T_w_out_C: float = 8.0
    DT_evap: float = 3.5
    T_amb_C: float = 25.0
    DT_cond: float = 5.0
    T_gen_C: float = 95.0
    T_abs_C: float = 32.0
    eta_p: float = 0.5


def run_case(cfg: Case941) -> Dict[str, Any]:
    # ---- evaporator load (chilled water) ----
    T_w_in = K(cfg.T_w_in_C)
    T_w_out = K(cfg.T_w_out_C)
    h_w_in = water_h_TP(T_w_in, ATM)
    h_w_out = water_h_TP(T_w_out, ATM)
    Q_evap = cfg.m_dot_w * (h_w_in - h_w_out)  # W

    # ---- refrigerant (water) ----
    T2 = T_w_out - cfg.DT_evap
    h2 = water_h_TQ(T2, 1.0)
    P2 = water_Psat(T2)

    T4 = K(cfg.T_amb_C) + cfg.DT_cond
    h4 = water_h_TQ(T4, 0.0)
    P4 = water_Psat(T4)

    h1 = h4
    P1 = P2
    T1 = water_T_from_HP(h1, P1)
    Q1 = water_Q_from_HP(h1, P1)

    T3 = K(cfg.T_gen_C)
    P3 = P4
    h3 = water_h_TP(T3, P3)

    m_dot_r = Q_evap / (h2 - h1)

    # ---- solution (LiBr-H2O) using ASHRAE correlations ----
    T5 = K(cfg.T_abs_C)
    P5 = P2
    x5 = x_LiBr_from_TP_ashrae(T5, P5)
    h5 = libr_h_ashrae(T5, x5)
    rho5 = libr_rho_coolprop(T5, x5, P5)

    P6 = P3
    w_p = (P6 - P5) / (rho5 * cfg.eta_p)  # J/kg
    h6 = h5 + w_p
    x6 = x5
    t6C = libr_tC_from_hx_ashrae(h6, x6)
    T6 = K(t6C)

    T7 = T3
    P7 = P3
    x7 = x_LiBr_from_TP_ashrae(T7, P7)
    h7 = libr_h_ashrae(T7, x7)

    P8 = P2
    x8 = x7
    h8 = h7

    if x8 <= x5:
        raise ValueError(f"Expected strong solution x8 > x5; got x5={x5}, x8={x8}")

    # absorber mass balances (same as K&N)
    m_dot_abs = m_dot_r / (1.0 - x5 / x8)
    m_dot_gen = m_dot_abs * x5 / x8

    # energy balances
    Q_abs = m_dot_r * h2 + m_dot_gen * h8 - m_dot_abs * h5
    Q_gen = m_dot_gen * h7 + m_dot_r * h3 - m_dot_abs * h6
    Q_cond = m_dot_r * (h3 - h4)
    W_p_total = m_dot_abs * w_p
    COP = Q_evap / Q_gen

    # residual checks (should be ~0 by construction)
    evap_resid = Q_evap - m_dot_r * (h2 - h1)
    cond_resid = Q_cond - m_dot_r * (h3 - h4)
    abs_resid = Q_abs - (m_dot_r * h2 + m_dot_gen * h8 - m_dot_abs * h5)
    gen_resid = Q_gen - (m_dot_gen * h7 + m_dot_r * h3 - m_dot_abs * h6)

    states = {
        1: dict(kind="water", T=T1, P=P1, h=h1, Q_or_x=Q1),
        2: dict(kind="water", T=T2, P=P2, h=h2, Q_or_x=1.0),
        3: dict(kind="water", T=T3, P=P3, h=h3, Q_or_x=None),
        4: dict(kind="water", T=T4, P=P4, h=h4, Q_or_x=0.0),
        5: dict(kind="libr",  T=T5, P=P5, h=h5, Q_or_x=x5, rho=rho5),
        6: dict(kind="libr",  T=T6, P=P6, h=h6, Q_or_x=x6),
        7: dict(kind="libr",  T=T7, P=P7, h=h7, Q_or_x=x7),
        8: dict(kind="libr",  T=None, P=P8, h=h8, Q_or_x=x8),
    }

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
        "residuals": dict(evap=evap_resid, cond=cond_resid, abs=abs_resid, gen=gen_resid),
    }


def _fmt(x, w=10, p=6):
    if x is None:
        return ""
    return f"{x:{w}.{p}f}"


if __name__ == "__main__":
    res = run_case(Case941())

    print("=== Waste-Heat Absorption Cooling (LiBr-H2O / Water) — ASHRAE correlations ===")
    print(f"Q_evap [kW]     = {res['Q_evap_W']/1000:.6f}")
    print(f"Q_gen  [kW]     = {res['Q_gen_W']/1000:.6f}")
    print(f"COP             = {res['COP']:.6f}")
    print(f"Pump W [W]      = {res['W_p_W']:.6f}")
    print(f"w_p [J/kg]      = {res['w_p_Jkg']:.9f}")
    print(f"m_dot_r [kg/s]  = {res['m_dot_r']:.9f}")
    print(f"m_dot_abs [kg/s]= {res['m_dot_abs']:.9f}")
    print(f"m_dot_gen [kg/s]= {res['m_dot_gen']:.9f}")

    x5 = res["states"][5]["Q_or_x"]
    x7 = res["states"][7]["Q_or_x"]
    print(f"x5 (weak)       = {x5:.12f}   ({x5*100:.3f} %)")
    print(f"x7 (strong)     = {x7:.12f}   ({x7*100:.3f} %)")

    # ---- state table ----
    print("\n--- State table ---")
    print(" st | kind  |   T [C]   |    P [Pa]    |   h [kJ/kg] |   Q or x    |  rho [kg/m3]")
    print("-" * 88)
    for st in range(1, 9):
        s = res["states"][st]
        kind = s["kind"]
        TK = s["T"]
        PC = (C_(TK) if TK is not None else None)
        P = s["P"]
        h = s["h"] / 1000.0
        qx = s["Q_or_x"]
        rho = s.get("rho", None)
        print(
            f"{st:>3d} | {kind:<5s} |"
            f"{_fmt(PC, w=9, p=3)} |"
            f"{_fmt(P,  w=12, p=3)} |"
            f"{_fmt(h,  w=11, p=3)} |"
            f"{_fmt(qx, w=11, p=6)} |"
            f"{_fmt(rho, w=11, p=3)}"
        )

    # ---- heat rates ----
    print("\n--- Heat rates ---")
    print(f"Q_evap [kW] = {res['Q_evap_W']/1000:.6f}")
    print(f"Q_abs  [kW] = {res['Q_abs_W']/1000:.6f}")
    print(f"Q_gen  [kW] = {res['Q_gen_W']/1000:.6f}")
    print(f"Q_cond [kW] = {res['Q_cond_W']/1000:.6f}")
    print(f"W_p    [W]  = {res['W_p_W']:.6f}")
    print(f"COP         = {res['COP']:.6f}")

    # ---- residuals ----
    r = res["residuals"]
    print("\n--- Balance residuals (should be ~0) ---")
    print(f"evap resid [W] = {r['evap']:.6e}")
    print(f"cond resid [W] = {r['cond']:.6e}")
    print(f"abs  resid [W] = {r['abs']:.6e}")
    print(f"gen  resid [W] = {r['gen']:.6e}")

    # ---- VLE sanity ----
    T5 = res["states"][5]["T"]
    T7 = res["states"][7]["T"]
    P2 = res["states"][2]["P"]
    P4 = res["states"][4]["P"]
    print("\n--- VLE sanity (ASHRAE Peq) ---")
    print(f"Absorber:  T5={C_(T5):.3f}C, P2={P2:.3f} Pa, Peq(T5,x5)={libr_Peq_ashrae(T5,x5):.3f} Pa")
    print(f"Generator: T7={C_(T7):.3f}C, P4={P4:.3f} Pa, Peq(T7,x7)={libr_Peq_ashrae(T7,x7):.3f} Pa")

__all__ = [
    "ATM",
    "Case941",
    "K",
    "C_",
    "bisect",
    "bracket_scan",
    "water_Psat",
    "water_h_TQ",
    "water_h_TP",
    "water_T_from_HP",
    "water_Q_from_HP",
    "libr_h_ashrae",
    "libr_tC_from_hx_ashrae",
    "libr_Peq_ashrae",
    "x_LiBr_from_TP_ashrae",
    "libr_rho_coolprop",
    "run_case",
]
