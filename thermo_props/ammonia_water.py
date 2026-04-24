"""
ammonia_water.py  (Ibrahim & Klein 1993 core thermo)

Implements thermodynamic properties for NH3-H2O using the reduced Gibbs formulation
and excess Gibbs model from:

Ibrahim, O.M., Klein, S.A.,
"Thermodynamic Properties of Ammonia-Water Mixtures,"
ASHRAE Trans.: Symposia, 21(2), 1495 (1993).

Primary goal:
- EES-like state call: props_tpx(T, P, X_mass) -> h,s,u,v,rho and EES q flag
- Robust outputs: never return negative/NaN/inf v or rho in “ok” states
- Avoid false vapor / 2-phase classification in regions where the IK vapor EOS is not applicable

Units:
- Inputs:  T [K], P [Pa], X = NH3 mass fraction [-]
- Internals: reduced Tr=T/TB, Pr=P/PB
- Outputs: J/kg, J/kg-K, m^3/kg, kg/m^3; plus kJ/kg helpers

VLE approach (Gibbs-consistent K-form):
- Ideal vapor mixing + liquid activity coefficients from excess Gibbs:
    y_i = x_i * gamma_i(x,T,P) * K0_i(T,P)
    ln K0_i = (g_i^L(T,P) - g_i^g(T,P)) / (R T)
  plus y1 + y2 = 1 => solve for x (at given T,P)

Critical guardrail:
- The IK vapor EOS is intended for vapor states. At some (T,P) the vapor EOS yields
  non-physical molar volumes (V_mol <= 0). When that happens, we:
    - skip VLE
    - force single-phase liquid (subcooled) behavior
  This prevents negative densities/volumes and bogus "superheated/2ph" flags.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple, Optional
import math


# --------------------------- constants / scaling ---------------------------

R = 8.31446261815324  # J/mol-K

TB = 100.0            # K   reducing temperature (IK)
PB = 1.0e6            # Pa  reducing pressure (IK) = 10 bar

# Reported IK correlation envelope (paper): 230 K <= T <= 600 K ; 0.2 bar <= P <= 110 bar
IK_T_MIN = 230.0
IK_T_MAX = 600.0
IK_P_MIN = 0.2e5      # Pa
IK_P_MAX = 110.0e5    # Pa

Phase = Literal["L", "g"]
Comp = Literal["NH3", "H2O"]

_EPS_X = 1e-18
_LOG_MIN = -745.0  # exp(-745) ~ 5e-324 (near underflow)
_LOG_MAX =  709.0  # exp(709) ~ 8e307 (near overflow)


# ------------------------------ coefficients ------------------------------

@dataclass(frozen=True)
class PureCoeffs:
    # liquid reduced volume (via dGr/dPr):
    # Vr^L = A1 + A2*Pr + A3*Tr + A4*Tr^2
    A1: float
    A2: float
    A3: float
    A4: float

    # liquid cp terms embedded in Gr^L (Eq 6)
    B1: float
    B2: float
    B3: float

    # vapor reduced volume:
    # Vr^g = Tr/Pr + C1 + C2/Tr^3 + C3/Tr^11 + C4*Pr^2/Tr^11
    C1: float
    C2: float
    C3: float
    C4: float

    # vapor cp terms embedded in Gr^g (Eq 7)
    D1: float
    D2: float
    D3: float

    # reference reduced h,s at (Tr0,Pr0)
    hr0_L: float
    hr0_g: float
    sr0_L: float
    sr0_g: float
    Tr0: float
    Pr0: float

    M: float  # kg/mol


# Table 1 coefficients (IK / common EES-compatible transcriptions)
PURE: dict[Comp, PureCoeffs] = {
    "NH3": PureCoeffs(
        A1=3.971423e-2, A2=-1.790557e-5, A3=-1.308905e-2, A4=3.752836e-3,
        B1=1.634519e1,  B2=-6.508119,    B3=1.448937,
        C1=-1.049377e-2, C2=-8.288224,   C3=-6.647257e2, C4=-3.045352e3,
        D1=3.673647,     D2=9.989629e-2, D3=3.617622e-2,
        hr0_L=4.878573, hr0_g=26.468879,
        sr0_L=1.644773, sr0_g=8.339026,
        Tr0=3.2252, Pr0=2.0,
        M=0.01703052
    ),
    "H2O": PureCoeffs(
        A1=4.093015e-2, A2=-4.597594e-5, A3=-3.719632e-3, A4=8.389246e-4,
        B1=1.214557e1,  B2=-1.898065,    B3=2.911966e-1,
        C1=2.136131e-2, C2=-3.169291e1,  C3=-4.634611e4, C4=0.0,
        D1=4.019170,    D2=-5.175550e-2, D3=1.951939e-2,
        hr0_L=21.821141, hr0_g=60.965058,
        sr0_L=5.733498,  sr0_g=13.453430,
        Tr0=5.0705, Pr0=3.0,
        M=0.01801528
    ),
}

# Table 2 excess Gibbs coefficients
# Important: E10 is negative in IK Table 2 (a common transcription error is flipping its sign).
E = {
    "E1":  -41.733398,
    "E2":   0.02414,
    "E3":   6.702285,
    "E4":  -0.011475,
    "E5":  63.608967,
    "E6": -62.490768,
    "E7":   1.761064,
    "E8":   0.008626,
    "E9":   0.387983,
    "E10": -0.004772,
    "E11": -4.648107,
    "E12":  0.836376,
    "E13": -3.553627,
    "E14":  0.000904,
    "E15": 24.361723,
    "E16": -20.7365477,
}


# ------------------------ small helpers ------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

def _safe_log(x: float) -> float:
    return math.log(x if x > _EPS_X else _EPS_X)

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def _safe_exp(lnx: float) -> float:
    return math.exp(_clip(lnx, _LOG_MIN, _LOG_MAX))

def _logsumexp(a: float, b: float) -> float:
    m = a if a >= b else b
    return m + _safe_log(_safe_exp(a - m) + _safe_exp(b - m))

def _finite(x: float) -> bool:
    return math.isfinite(x)

def _in_envelope(T: float, P: float) -> bool:
    return (IK_T_MIN <= T <= IK_T_MAX) and (IK_P_MIN <= P <= IK_P_MAX)

def _envelope_note(T: float, P: float) -> str:
    if _in_envelope(T, P):
        return ""
    return f"Outside IK envelope: T={T:.3f} K (230..600), P={P/1e5:.3f} bar (0.2..110)."


# ------------------------ fraction conversion ------------------------

def x_from_w(w_nh3: float) -> float:
    """NH3 mass fraction -> mole fraction."""
    w = _clamp01(float(w_nh3))
    M1 = PURE["NH3"].M
    M2 = PURE["H2O"].M
    n1 = w / M1
    n2 = (1.0 - w) / M2
    return n1 / (n1 + n2)

def w_from_x(x_nh3: float) -> float:
    """NH3 mole fraction -> mass fraction."""
    x = _clamp01(float(x_nh3))
    M1 = PURE["NH3"].M
    M2 = PURE["H2O"].M
    m1 = x * M1
    m2 = (1.0 - x) * M2
    return m1 / (m1 + m2)

def M_mix_from_x(x_nh3: float) -> float:
    """Mixture molar mass [kg/mol] from mole fraction."""
    x = _clamp01(float(x_nh3))
    return x * PURE["NH3"].M + (1.0 - x) * PURE["H2O"].M


# ------------------------ pure reduced volume (guardrail) ------------------------

def _Vr_pure(comp: Comp, phase: Phase, Tr: float, Pr: float) -> float:
    c = PURE[comp]
    if phase == "L":
        return (c.A1 + c.A3 * Tr + c.A4 * Tr**2) + c.A2 * Pr
    if phase == "g":
        return (Tr / Pr) + c.C1 + (c.C2 / Tr**3) + (c.C3 / Tr**11) + (c.C4 * Pr**2 / Tr**11)
    raise ValueError("phase must be 'L' or 'g'")

def _pure_vapor_applicable(T: float, P: float, vr_tol: float = 1e-12) -> bool:
    """
    Vapor correlation applicability screen:
    require positive reduced molar volume for both pure vapors at (T,P).
    """
    Tr = T / TB
    Pr = P / PB
    if Tr <= 0.0 or Pr <= 0.0:
        return False
    vr1 = _Vr_pure("NH3", "g", Tr, Pr)
    vr2 = _Vr_pure("H2O", "g", Tr, Pr)
    return (_finite(vr1) and _finite(vr2) and vr1 > vr_tol and vr2 > vr_tol)


# ------------------------ pure reduced Gibbs: liquid (Eq 6) ------------------------

def _G_r_L(comp: Comp, Tr: float, Pr: float) -> Tuple[float, float, float]:
    """
    IK Eq (6): pure liquid reduced Gibbs and derivatives.
    Returns (Gr, dGr/dTr, dGr/dPr).
    """
    c = PURE[comp]
    Tr0, Pr0 = c.Tr0, c.Pr0

    d1 = (Tr - Tr0)
    d2 = (Tr**2 - Tr0**2)
    d3 = (Tr**3 - Tr0**3)

    Gr = (
        c.hr0_L - Tr * c.sr0_L
        + c.B1 * d1 + 0.5 * c.B2 * d2 + (1.0 / 3.0) * c.B3 * d3
        - c.B1 * Tr * _safe_log(Tr / Tr0)
        - c.B2 * Tr * (Tr - Tr0)
        - 0.5 * c.B3 * Tr * (Tr**2 - Tr0**2)
        + (c.A1 + c.A3 * Tr + c.A4 * Tr**2) * (Pr - Pr0)
        + 0.5 * c.A2 * (Pr**2 - Pr0**2)
    )

    dGr_dTr = (
        -c.sr0_L
        + c.B1 + c.B2 * Tr + c.B3 * Tr**2
        - c.B1 * (_safe_log(Tr / Tr0) + 1.0)
        - c.B2 * (2.0 * Tr - Tr0)
        - 0.5 * c.B3 * (3.0 * Tr**2 - Tr0**2)
        + (c.A3 + 2.0 * c.A4 * Tr) * (Pr - Pr0)
    )

    # dGr/dPr = Vr^L
    dGr_dPr = (c.A1 + c.A3 * Tr + c.A4 * Tr**2) + c.A2 * Pr

    return Gr, dGr_dTr, dGr_dPr


# ------------------------ pure reduced Gibbs: vapor (Eq 7) ------------------------

def _G_r_g(comp: Comp, Tr: float, Pr: float) -> Tuple[float, float, float]:
    """
    IK Eq (7): pure vapor reduced Gibbs and derivatives.
    Returns (Gr, dGr/dTr, dGr/dPr).
    """
    c = PURE[comp]
    Tr0, Pr0 = c.Tr0, c.Pr0

    d1 = (Tr - Tr0)
    d2 = (Tr**2 - Tr0**2)
    d3 = (Tr**3 - Tr0**3)

    # helper integrals (IK Eq 7)
    F2 = (Pr / Tr**3) - (4.0 * Pr0 / Tr0**3) + (3.0 * Pr0 * Tr / Tr0**4)
    F3 = (Pr / Tr**11) - (12.0 * Pr0 / Tr0**11) + (11.0 * Pr0 * Tr / Tr0**12)
    F4 = (Pr**3 / Tr**11) - (12.0 * Pr0**3 / Tr0**11) + (11.0 * Pr0**3 * Tr / Tr0**12)

    Gr = (
        c.hr0_g - Tr * c.sr0_g
        + c.D1 * d1 + 0.5 * c.D2 * d2 + (1.0 / 3.0) * c.D3 * d3
        - c.D1 * Tr * _safe_log(Tr / Tr0)
        - c.D2 * Tr * (Tr - Tr0)
        - 0.5 * c.D3 * Tr * (Tr**2 - Tr0**2)
        + Tr * _safe_log(Pr / Pr0)
        + c.C1 * (Pr - Pr0)
        + c.C2 * F2
        + c.C3 * F3
        + (c.C4 / 3.0) * F4
    )

    # dGr/dPr = Vr^g
    dGr_dPr = (Tr / Pr) + c.C1 + (c.C2 / Tr**3) + (c.C3 / Tr**11) + (c.C4 * Pr**2 / Tr**11)

    base_dGr_dTr = (
        -c.sr0_g
        + c.D1 + c.D2 * Tr + c.D3 * Tr**2
        - c.D1 * (_safe_log(Tr / Tr0) + 1.0)
        - c.D2 * (2.0 * Tr - Tr0)
        - 0.5 * c.D3 * (3.0 * Tr**2 - Tr0**2)
    )

    dF2_dTr = (-3.0 * Pr / Tr**4) + (3.0 * Pr0 / Tr0**4)
    dF3_dTr = (-11.0 * Pr / Tr**12) + (11.0 * Pr0 / Tr0**12)
    dF4_dTr = (-11.0 * Pr**3 / Tr**12) + (11.0 * Pr0**3 / Tr0**12)

    dGr_dTr = (
        base_dGr_dTr
        + _safe_log(Pr / Pr0)
        + c.C2 * dF2_dTr
        + c.C3 * dF3_dTr
        + (c.C4 / 3.0) * dF4_dTr
    )

    return Gr, dGr_dTr, dGr_dPr


def pure_props(
    comp: Comp,
    phase: Phase,
    T: float,
    P: float,
    *,
    validate_volume: bool = True,
) -> Dict[str, float]:
    """
    Pure component thermodynamic_properties at (T,P) for phase 'L' or 'g'.

    Returns molar:
      G_mol [J/mol], H_mol [J/mol], S_mol [J/mol-K], V_mol [m^3/mol]
    And mass:
      h,s,u,v,rho with keys:
        h_J_per_kg, s_J_per_kgK, u_J_per_kg, v_m3_per_kg, rho_kg_per_m3
    """
    T = float(T)
    P = float(P)
    Tr = T / TB
    Pr = P / PB
    if Tr <= 0.0 or Pr <= 0.0:
        raise ValueError("T and P must be positive.")

    if phase == "L":
        Gr, dGr_dTr, dGr_dPr = _G_r_L(comp, Tr, Pr)
    elif phase == "g":
        Gr, dGr_dTr, dGr_dPr = _G_r_g(comp, Tr, Pr)
    else:
        raise ValueError("phase must be 'L' or 'g'")

    # IK reduced thermo property_relations:
    # s = -R * dGr/dTr
    # h = R*TB*(Gr - Tr*dGr/dTr)
    # v = (R*TB/PB) * dGr/dPr
    sr = -dGr_dTr
    hr = Gr - Tr * dGr_dTr
    vr = dGr_dPr  # reduced volume

    V_mol = (R * TB / PB) * vr
    if validate_volume and (not _finite(V_mol) or V_mol <= 0.0):
        raise ValueError(f"Non-physical pure {comp} {phase} molar volume at T={T}, P={P}: V_mol={V_mol}")

    G_mol = R * TB * Gr
    H_mol = R * TB * hr
    S_mol = R * sr

    M = PURE[comp].M
    v = V_mol / M
    if validate_volume and (not _finite(v) or v <= 0.0):
        raise ValueError(f"Non-physical pure {comp} {phase} specific volume at T={T}, P={P}: v={v}")

    rho = 1.0 / v
    h = H_mol / M
    s = S_mol / M
    u = h - P * v

    return {
        "G_mol": G_mol,
        "H_mol": H_mol,
        "S_mol": S_mol,
        "V_mol": V_mol,
        "h_J_per_kg": h,
        "s_J_per_kgK": s,
        "u_J_per_kg": u,
        "v_m3_per_kg": v,
        "rho_kg_per_m3": rho,
        # debug (useful when hunting bad states)
        "Gr": Gr,
        "Vr": vr,
    }


# ------------------------ excess Gibbs (Eq 15 + Eqs 16-18) ------------------------

def _F_terms_excess(Tr: float, Pr: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    IK Eqs (16)-(18) for excess Gibbs model:

      F1 = E1 + E2 Pr + (E3 + E4 Pr) Tr + E5/Tr + E6/Tr^2
      F2 = E7 + E8 Pr + (E9 + E10 Pr) Tr + E11/Tr + E12/Tr^2
      F3 = E13 + E14 Pr + E15/Tr + E16/Tr^2

    Returns:
      F1,F2,F3,
      dF1/dTr,dF2/dTr,dF3/dTr,
      dF1/dPr,dF2/dPr,dF3/dPr
    """
    invT = 1.0 / Tr
    invT2 = invT * invT
    invT3 = invT2 * invT

    F1 = E["E1"] + E["E2"]*Pr + (E["E3"] + E["E4"]*Pr)*Tr + E["E5"]*invT + E["E6"]*invT2
    F2 = E["E7"] + E["E8"]*Pr + (E["E9"] + E["E10"]*Pr)*Tr + E["E11"]*invT + E["E12"]*invT2
    F3 = E["E13"] + E["E14"]*Pr + E["E15"]*invT + E["E16"]*invT2

    # d/dPr
    dF1_dPr = E["E2"] + E["E4"]*Tr
    dF2_dPr = E["E8"] + E["E10"]*Tr
    dF3_dPr = E["E14"]

    # d/dTr
    dF1_dTr = (E["E3"] + E["E4"]*Pr) - E["E5"]*invT2 - 2.0*E["E6"]*invT3
    dF2_dTr = (E["E9"] + E["E10"]*Pr) - E["E11"]*invT2 - 2.0*E["E12"]*invT3
    dF3_dTr = -E["E15"]*invT2 - 2.0*E["E16"]*invT3

    return F1, F2, F3, dF1_dTr, dF2_dTr, dF3_dTr, dF1_dPr, dF2_dPr, dF3_dPr


def excess_reduced(x: float, Tr: float, Pr: float) -> Tuple[float, float, float]:
    """
    IK Eq (15):
      GEr = x(1-x) [ F1 + F2(2x-1) + F3(2x-1)^2 ]

    Returns (GEr, dGEr/dTr, dGEr/dPr).
    """
    x1 = _clamp01(float(x))
    x2 = 1.0 - x1
    d = 2.0*x1 - 1.0

    F1, F2, F3, dF1dTr, dF2dTr, dF3dTr, dF1dPr, dF2dPr, dF3dPr = _F_terms_excess(Tr, Pr)

    GEr = x1 * x2 * (F1 + F2*d + F3*d*d)
    dG_dTr = x1 * x2 * (dF1dTr + dF2dTr*d + dF3dTr*d*d)
    dG_dPr = x1 * x2 * (dF1dPr + dF2dPr*d + dF3dPr*d*d)

    return GEr, dG_dTr, dG_dPr


def excess_props(x: float, T: float, P: float) -> Dict[str, float]:
    """
    Excess liquid thermodynamic_properties from reduced excess Gibbs and derivatives:

      sE_mol = -R * dGEr/dTr
      hE_mol = R*TB*(GEr - Tr*dGEr/dTr)
      vE_mol = (R*TB/PB) * dGEr/dPr
    """
    Tr = float(T) / TB
    Pr = float(P) / PB

    GEr, dGdTr, dGdPr = excess_reduced(x, Tr, Pr)

    SE_mol = R * (-dGdTr)
    HE_mol = R * TB * (GEr - Tr*dGdTr)
    VE_mol = (R * TB / PB) * dGdPr
    GE_mol = R * TB * GEr

    Mmix = M_mix_from_x(x)
    return {
        "GE_mol": GE_mol,
        "HE_mol": HE_mol,
        "SE_mol": SE_mol,
        "VE_mol": VE_mol,
        "HE": HE_mol / Mmix,
        "SE": SE_mol / Mmix,
        "VE": VE_mol / Mmix,
    }


def activity_ln_gamma(x: float, T: float, P: float) -> Tuple[float, float]:
    """
    Activity coefficient logs from excess Gibbs (binary mixture):

      ln(gamma_i) = mu_i^E / (R T)

    With reduced scaling (GE_mol = R*TB*GEr):
      ln(gamma_i) = mu_i,r^E / Tr

    where:
      mu1_r^E = GEr + (1-x)*d(GEr)/dx
      mu2_r^E = GEr - x*d(GEr)/dx
    """
    Tr = float(T) / TB
    Pr = float(P) / PB
    x1 = _clamp01(float(x))
    x2 = 1.0 - x1
    d = 2.0*x1 - 1.0

    F1, F2, F3, *_ = _F_terms_excess(Tr, Pr)

    Q = (F1 + F2*d + F3*d*d)
    g = x1 * x2 * Q  # GEr itself (reduced)

    # dGEr/dx:
    # GEr = x(1-x)Q, Q=F1+F2 d+F3 d^2, d=2x-1
    # dQ/dx = 2F2 + 4F3 d
    dgdx = (1.0 - 2.0*x1) * Q + (x1*x2) * (2.0*F2 + 4.0*F3*d)

    mu1_r = g + (1.0 - x1) * dgdx
    mu2_r = g - x1 * dgdx

    ln_g1 = mu1_r / Tr
    ln_g2 = mu2_r / Tr
    if not (_finite(ln_g1) and _finite(ln_g2)):
        raise ValueError(f"Non-finite ln(gamma) at x={x1}, T={T}, P={P}")
    return ln_g1, ln_g2


def activity_coeffs(x: float, T: float, P: float) -> Tuple[float, float]:
    """Activity coefficients (gamma) from excess Gibbs; exp is clipped to avoid overflow."""
    ln1, ln2 = activity_ln_gamma(x, T, P)
    return _safe_exp(ln1), _safe_exp(ln2)


# ------------------------ mixture thermodynamic_properties ------------------------

def mix_liquid_props(T: float, P: float, x_nh3: float) -> Dict[str, float]:
    """
    Liquid NH3-H2O solution at (T,P,x) where x is NH3 mole fraction.
    """
    x = _clamp01(float(x_nh3))
    x2 = 1.0 - x

    nh3 = pure_props("NH3", "L", T, P, validate_volume=True)
    h2o = pure_props("H2O", "L", T, P, validate_volume=True)
    ex  = excess_props(x, T, P)

    # ideal mixing entropy (molar)
    S_mix_mol = -R * (x * _safe_log(x) + x2 * _safe_log(x2))

    H_mol = x * nh3["H_mol"] + x2 * h2o["H_mol"] + ex["HE_mol"]
    S_mol = x * nh3["S_mol"] + x2 * h2o["S_mol"] + ex["SE_mol"] + S_mix_mol
    V_mol = x * nh3["V_mol"] + x2 * h2o["V_mol"] + ex["VE_mol"]

    Mmix = M_mix_from_x(x)
    v = V_mol / Mmix
    if (not _finite(v)) or v <= 0.0:
        raise ValueError(f"Non-physical liquid v at T={T}, P={P}, x={x}: v={v}")

    rho = 1.0 / v
    h = H_mol / Mmix
    s = S_mol / Mmix
    u = h - float(P) * v

    return {
        "phase": "L",
        "x_mole": x,
        "w_mass": w_from_x(x),
        "h_J_per_kg": h,
        "s_J_per_kgK": s,
        "u_J_per_kg": u,
        "v_m3_per_kg": v,
        "rho_kg_per_m3": rho,
        "h_kJ_per_kg": h / 1000.0,
        "s_kJ_per_kgK": s / 1000.0,
        "u_kJ_per_kg": u / 1000.0,
    }


def mix_vapor_props(T: float, P: float, y_nh3: float) -> Dict[str, float]:
    """
    Vapor mixture at (T,P,y) where y is NH3 mole fraction.

    IK treatment here: ideal mixing of the pure-vapor real-gas correlations (no vapor excess term).
    """
    y = _clamp01(float(y_nh3))
    y2 = 1.0 - y

    nh3 = pure_props("NH3", "g", T, P, validate_volume=True)
    h2o = pure_props("H2O", "g", T, P, validate_volume=True)

    S_mix_mol = -R * (y * _safe_log(y) + y2 * _safe_log(y2))

    H_mol = y * nh3["H_mol"] + y2 * h2o["H_mol"]
    S_mol = y * nh3["S_mol"] + y2 * h2o["S_mol"] + S_mix_mol
    V_mol = y * nh3["V_mol"] + y2 * h2o["V_mol"]

    Mmix = M_mix_from_x(y)
    v = V_mol / Mmix
    if (not _finite(v)) or v <= 0.0:
        raise ValueError(f"Non-physical vapor v at T={T}, P={P}, y={y}: v={v}")

    rho = 1.0 / v
    h = H_mol / Mmix
    s = S_mol / Mmix
    u = h - float(P) * v

    return {
        "phase": "g",
        "y_mole": y,
        "w_mass": w_from_x(y),
        "h_J_per_kg": h,
        "s_J_per_kgK": s,
        "u_J_per_kg": u,
        "v_m3_per_kg": v,
        "rho_kg_per_m3": rho,
        "h_kJ_per_kg": h / 1000.0,
        "s_kJ_per_kgK": s / 1000.0,
        "u_kJ_per_kg": u / 1000.0,
    }


# ------------------------ VLE helper (Gibbs-consistent K-form) ------------------------

def _lnK0_pure_from_gibbs(T: float, P: float) -> Optional[Tuple[float, float]]:
    """
    Build ln(K0) for each component:

      ln K0_i(T,P) = (g_i^L - g_i^g) / (R T)

    Returns (lnK0_nh3, lnK0_h2o) or None if the vapor correlation is not applicable at (T,P).
    """
    if not _pure_vapor_applicable(T, P):
        return None

    # If a pure vapor correlation fails (non-physical volume), treat K0 as unavailable.
    gL_nh3 = pure_props("NH3", "L", T, P, validate_volume=True)["G_mol"]
    gG_nh3 = pure_props("NH3", "g", T, P, validate_volume=True)["G_mol"]
    gL_h2o = pure_props("H2O", "L", T, P, validate_volume=True)["G_mol"]
    gG_h2o = pure_props("H2O", "g", T, P, validate_volume=True)["G_mol"]

    lnK0_nh3 = (gL_nh3 - gG_nh3) / (R * T)
    lnK0_h2o = (gL_h2o - gG_h2o) / (R * T)
    if not (_finite(lnK0_nh3) and _finite(lnK0_h2o)):
        return None
    return lnK0_nh3, lnK0_h2o


def _y_sums_from_x(
    x: float,
    T: float,
    P: float,
    lnK0_1: float,
    lnK0_2: float,
) -> Tuple[float, float, float]:
    """
    Compute (y1, y2, sumy) for a given liquid composition x using log-space arithmetic.
    y1,y2 are normalized so y1+y2=1 even if sumy != 1 due to numeric drift.
    """
    x1 = _clamp01(x)
    x2 = 1.0 - x1
    ln_g1, ln_g2 = activity_ln_gamma(x1, T, P)

    a = _safe_log(x1) + ln_g1 + lnK0_1
    b = _safe_log(x2) + ln_g2 + lnK0_2

    ln_sumy = _logsumexp(a, b)
    sumy = _safe_exp(ln_sumy)

    y1 = _safe_exp(a - ln_sumy)
    y2 = _safe_exp(b - ln_sumy)
    return y1, y2, sumy


def equilibrium_xy_TP(
    T: float,
    P: float,
    *,
    nscan: int = 120,
    xtol: float = 1e-13,
    ftol: float = 1e-13,
    max_iter: int = 200,
) -> Optional[Tuple[float, float]]:
    """
    Solve equilibrium liquid x and vapor y at given (T,P) via:

      y1 = x1 * gamma1(x) * K0_1(T,P)
      y2 = x2 * gamma2(x) * K0_2(T,P)
      y1 + y2 = 1

    Returns (xL, yV) in mole fractions if a root/bracket is found, else None.
    """
    lnK0 = _lnK0_pure_from_gibbs(T, P)
    if lnK0 is None:
        return None
    lnK0_1, lnK0_2 = lnK0

    def f(x: float) -> float:
        _, _, sumy = _y_sums_from_x(x, T, P, lnK0_1, lnK0_2)
        return sumy - 1.0

    lo = 1e-12
    hi = 1.0 - 1e-12

    prev_x = lo
    prev_f = f(prev_x)
    if abs(prev_f) < ftol:
        y1, _, _ = _y_sums_from_x(prev_x, T, P, lnK0_1, lnK0_2)
        return prev_x, _clamp01(y1)

    bracket: Optional[Tuple[float, float, float, float]] = None
    for i in range(1, nscan + 1):
        x = lo + (hi - lo) * i / nscan
        fx = f(x)
        if abs(fx) < ftol:
            y1, _, _ = _y_sums_from_x(x, T, P, lnK0_1, lnK0_2)
            return x, _clamp01(y1)
        if prev_f * fx < 0.0:
            bracket = (prev_x, x, prev_f, fx)
            break
        prev_x, prev_f = x, fx

    if bracket is None:
        return None

    a, b, fa, fb = bracket

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < ftol or abs(b - a) < xtol:
            y1, _, _ = _y_sums_from_x(m, T, P, lnK0_1, lnK0_2)
            return m, _clamp01(y1)
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    m = 0.5 * (a + b)
    y1, _, _ = _y_sums_from_x(m, T, P, lnK0_1, lnK0_2)
    return m, _clamp01(y1)


# ------------------------ EES-like entry point ------------------------

def _result_error(T: float, P: float, X: float, msg: str) -> Dict[str, float]:
    nan = float("nan")
    z = x_from_w(_clamp01(X))
    return {
        "ok": 0,
        "error": msg,
        "warning": _envelope_note(T, P),
        "T_K": T,
        "T_C": T - 273.15,
        "P_Pa": P,
        "P_kPa": P / 1000.0,
        "P_bar": P / 1e5,
        "X": X,
        "z_mole": z,
        "phase": "err",
        "q": nan,
        "h_J_per_kg": nan,
        "s_J_per_kgK": nan,
        "u_J_per_kg": nan,
        "v_m3_per_kg": nan,
        "rho_kg_per_m3": nan,
        "h_kJ_per_kg": nan,
        "s_kJ_per_kgK": nan,
        "u_kJ_per_kg": nan,
        "xL": nan,
        "yV": nan,
        "wL": nan,
        "wV": nan,
        "xL_mole": nan,
        "yV_mole": nan,
        "wL_mass": nan,
        "wV_mass": nan,
    }

def _finalize(base: Dict[str, float], *, ok: int = 1, error: str = "", warning: str = "") -> Dict[str, float]:
    base["ok"] = int(ok)
    base["error"] = error
    base["warning"] = warning

    # standardized convenience fields
    T = float(base.get("T_K", float("nan")))
    P = float(base.get("P_Pa", float("nan")))
    base.setdefault("T_C", T - 273.15)
    base.setdefault("P_kPa", P / 1000.0)
    base.setdefault("P_bar", P / 1e5)

    if "X" in base:
        base.setdefault("z_mole", x_from_w(base["X"]))

    if "h_J_per_kg" in base:
        base["h_kJ_per_kg"] = base["h_J_per_kg"] / 1000.0
    if "s_J_per_kgK" in base:
        base["s_kJ_per_kgK"] = base["s_J_per_kgK"] / 1000.0
    if "u_J_per_kg" in base:
        base["u_kJ_per_kg"] = base["u_J_per_kg"] / 1000.0

    # aliases commonly used in CSV/post-processing
    if "xL" in base:
        base["xL_mole"] = base["xL"]
    if "yV" in base:
        base["yV_mole"] = base["yV"]
    if "wL" in base:
        base["wL_mass"] = base["wL"]
    if "wV" in base:
        base["wV_mass"] = base["wV"]

    return base


def props_tpx(T_K: float, P_Pa: float, X: float, strict: bool = True) -> Dict[str, float]:
    """
    EES-like state call: given T, P, and NH3 mass fraction X, return:
      h,s,u,v,rho and EES-style quality flag q.

    q convention (EES docs):
      q = -0.001  => subcooled (single-phase liquid)
      0<=q<=1     => saturated two-phase (vapor mass fraction)
      q = 1.001   => superheated (single-phase vapor)

    Robustness upgrades:
    - If the vapor EOS is not applicable at (T,P) (non-physical molar volumes),
      skip VLE and do not classify as vapor/2-phase.
    - Never return negative v or rho; if something goes wrong:
        strict=True  -> raise
        strict=False -> return ok=0 with NaNs + error message
    """
    T = float(T_K)
    P = float(P_Pa)
    X_in = float(X)

    if T <= 0.0 or P <= 0.0:
        if strict:
            raise ValueError("T and P must be positive.")
        return _result_error(T, P, X_in, "T and P must be positive.")

    warning = _envelope_note(T, P)

    if X_in < 0.0 or X_in > 1.0:
        if strict:
            raise ValueError(f"X mass fraction out of range [0,1]: {X_in}")
        Xc = _clamp01(X_in)
        warning = (warning + " " if warning else "") + f"Clamped X from {X_in} to {Xc}."
    else:
        Xc = X_in

    try:
        z = x_from_w(Xc)  # overall mole fraction

        # Attempt VLE only if pure-vapor EOS is applicable here
        xy = equilibrium_xy_TP(T, P)

        # If VLE is unavailable, choose single-phase using a bubble-sum discriminator
        if xy is None:
            # If the vapor EOS is not applicable, force liquid (subcooled)
            if not _pure_vapor_applicable(T, P):
                L = mix_liquid_props(T, P, z)
                out = {**L, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": -0.001}
                w = (warning + " " if warning else "") + "VLE skipped: vapor EOS not applicable."
                return _finalize(out, ok=1, error="", warning=w.strip())

            lnK0 = _lnK0_pure_from_gibbs(T, P)
            if lnK0 is None:
                L = mix_liquid_props(T, P, z)
                out = {**L, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": -0.001}
                w = (warning + " " if warning else "") + "VLE unavailable; returned liquid."
                return _finalize(out, ok=1, error="", warning=w.strip())

            lnK0_1, lnK0_2 = lnK0
            _, _, bubble_sum = _y_sums_from_x(z, T, P, lnK0_1, lnK0_2)

            if bubble_sum <= 1.0:
                L = mix_liquid_props(T, P, z)
                out = {**L, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": -0.001}
                return _finalize(out, ok=1, error="", warning=warning)
            else:
                V = mix_vapor_props(T, P, z)
                out = {**V, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": 1.001}
                return _finalize(out, ok=1, error="", warning=warning)

        # Equilibrium exists
        xL, yV = xy
        wL = w_from_x(xL)
        wV = w_from_x(yV)

        # Degenerate equilibrium -> treat as liquid
        if (not _finite(wL)) or (not _finite(wV)) or (wV <= wL + 1e-15):
            L = mix_liquid_props(T, P, z)
            out = {**L, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": -0.001, "xL": xL, "yV": yV}
            w = (warning + " " if warning else "") + "Degenerate VLE; returned liquid."
            return _finalize(out, ok=1, error="", warning=w.strip())

        # Regions based on overall X (mass) relative to sat bounds
        if Xc <= wL:
            L = mix_liquid_props(T, P, z)
            out = {**L, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": -0.001, "xL": xL, "yV": yV, "wL": wL, "wV": wV}
            return _finalize(out, ok=1, error="", warning=warning)

        if Xc >= wV:
            V = mix_vapor_props(T, P, z)
            out = {**V, "T_K": T, "P_Pa": P, "X": Xc, "z_mole": z, "q": 1.001, "xL": xL, "yV": yV, "wL": wL, "wV": wV}
            return _finalize(out, ok=1, error="", warning=warning)

        # Two-phase: vapor mass fraction (quality) on a mass basis (lever rule)
        q = (Xc - wL) / (wV - wL)
        q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)

        Lsat = mix_liquid_props(T, P, xL)
        Vsat = mix_vapor_props(T, P, yV)

        # Mass-weighted two-phase mixture thermodynamic_properties
        h = (1.0 - q) * Lsat["h_J_per_kg"] + q * Vsat["h_J_per_kg"]
        s = (1.0 - q) * Lsat["s_J_per_kgK"] + q * Vsat["s_J_per_kgK"]
        v = (1.0 - q) * Lsat["v_m3_per_kg"] + q * Vsat["v_m3_per_kg"]
        if (not _finite(v)) or v <= 0.0:
            raise ValueError(f"Non-physical two-phase v at T={T}, P={P}, X={Xc}: v={v}")
        rho = 1.0 / v
        u = h - P * v

        out = {
            "phase": "2ph",
            "T_K": T,
            "P_Pa": P,
            "X": Xc,
            "z_mole": z,
            "q": q,
            "xL": xL,
            "yV": yV,
            "wL": wL,
            "wV": wV,
            "h_J_per_kg": h,
            "s_J_per_kgK": s,
            "u_J_per_kg": u,
            "v_m3_per_kg": v,
            "rho_kg_per_m3": rho,
            "L_h_kJkg": Lsat["h_kJ_per_kg"],
            "V_h_kJkg": Vsat["h_kJ_per_kg"],
        }
        return _finalize(out, ok=1, error="", warning=warning)

    except Exception as e:
        if strict:
            raise
        return _result_error(T, P, X_in, str(e))


# ------------------------ quick self-test ------------------------

if __name__ == "__main__":
    # EES documentation example:
    #   T = 10 C ; P = 1000 kPa ; X = 0.5
    # Expected (EES):
    #   h ≈ -194.8 kJ/kg ; q = -0.001
    T = 10.0 + 273.15
    P = 1000.0e3
    X = 0.5

    out = props_tpx(T, P, X, strict=True)

    print("=== NH3H2O (IK) check: EES doc example ===")
    print(f"T = {T:.2f} K, P = {P:.3e} Pa, X(NH3 mass) = {X:.6f}")
    print(f"q = {out['q']}")
    print(f"h = {out.get('h_kJ_per_kg', out['h_J_per_kg']/1000.0):.3f} kJ/kg")
    print(f"rho = {out['rho_kg_per_m3']:.3f} kg/m^3")
    print(f"phase = {out['phase']}")
    if out.get("warning"):
        print(f"warning: {out['warning']}")

__all__ = [
    "R",
    "TB",
    "PB",
    "IK_T_MIN",
    "IK_T_MAX",
    "IK_P_MIN",
    "IK_P_MAX",
    "Phase",
    "Comp",
    "PureCoeffs",
    "x_from_w",
    "w_from_x",
    "M_mix_from_x",
    "pure_props",
    "excess_reduced",
    "excess_props",
    "activity_ln_gamma",
    "activity_coeffs",
    "mix_liquid_props",
    "mix_vapor_props",
    "vle_x_at_TP",
    "props_tpx",
    "props_si",
    "nh3h2o_props_si",
    "NH3H2OPropsSI",
    "NH3H2O_TPX",
    "NH3H2O",
    "state_tpx",
    "NH3H2O_STATE_TPX",
    "NH3H2O_STATE",
]
