#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nh3h2o/ammonia_cooling_trf.py

Pure-Python NH3–H2O absorption cycle sandbox (Klein & Nellis 9.B-6 style),
using a Tillner-Roth & Friend (1998) mixture Helmholtz formulation ("TRF").

Key points
----------
- No REFPROP dependency (no ctREFPROP, no RPPREFIX, nothing).
- Robust CLI: explicit --backend and --mode (no hidden env-var defaults).
- Provides:
    • state_tp(T,P,w)  -> single-phase state (phase hint supported)
    • flash_tp(T,P,w)  -> VLE flash at fixed T,P,overall composition
    • compute_cycle_kn_9b6(...) -> cycle metrics (T5, ε_shx, T6(q≈1), COP, ...)

Dependencies (TRF backend)
-------------------------
- CoolProp  (pure-fluid residual Helmholtz alphar for Water and Ammonia)
- SciPy     (root finding / bracketing)

Install inside your venv:
    pip install CoolProp scipy

Examples
--------
Cycle:
    runroot python nh3h2o/ammonia_cooling_trf.py --backend trf --mode cycle

Save JSON:
    runroot python nh3h2o/ammonia_cooling_trf.py --backend trf --mode cycle --out out/kn9b6_trf.json

Debug a single state:
    runroot python nh3h2o/ammonia_cooling_trf.py --mode state --T_C 27 --P_bar 13.5 --w 0.95 --phase liquid

Flash:
    runroot python nh3h2o/ammonia_cooling_trf.py --mode flash --T_C 115 --P_bar 13.5 --w 0.48
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


# ------------------------------- constants -------------------------------

# molar masses [kg/mol]
M_NH3 = 17.03052e-3
M_H2O = 18.01528e-3

_RU = 8.31446261815324  # J/(mol·K)


# ------------------------------- helpers --------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def massfrac_to_molefrac(w_nh3: float, M_nh3: float = M_NH3, M_h2o: float = M_H2O) -> float:
    """NH3 mass fraction -> NH3 mole fraction."""
    w = clamp(w_nh3, 0.0, 1.0)
    n_nh3 = w / M_nh3
    n_h2o = (1.0 - w) / M_h2o
    n_tot = n_nh3 + n_h2o
    if n_tot <= 0.0:
        return 0.0
    return n_nh3 / n_tot


def molefrac_to_massfrac(x_nh3: float, M_nh3: float = M_NH3, M_h2o: float = M_H2O) -> float:
    """NH3 mole fraction -> NH3 mass fraction."""
    x = clamp(x_nh3, 0.0, 1.0)
    m_nh3 = x * M_nh3
    m_h2o = (1.0 - x) * M_h2o
    m_tot = m_nh3 + m_h2o
    if m_tot <= 0.0:
        return 0.0
    return m_nh3 / m_tot


def molar_mass_mix(x_nh3: float, M_nh3: float = M_NH3, M_h2o: float = M_H2O) -> float:
    x = clamp(x_nh3, 0.0, 1.0)
    return x * M_nh3 + (1.0 - x) * M_h2o


def _clamp_x(x: float) -> float:
    # avoid log(0) and extreme derivatives
    return max(1e-12, min(1.0 - 1e-12, x))


# ------------------------------- errors ---------------------------------

class BackendUnavailableError(RuntimeError):
    pass


class ThermoEvalError(RuntimeError):
    pass


# ------------------------------- dataclasses ----------------------------

@dataclass
class State:
    """Single-phase (or bulk) state on a mass basis."""
    T_K: float
    P_Pa: float
    w_nh3: float
    h_J_per_kg: float
    rho_kg_per_m3: float
    s_J_per_kgK: Optional[float] = None
    note: str = ""

    @property
    def T_C(self) -> float:
        return self.T_K - 273.15

    @property
    def P_bar(self) -> float:
        return self.P_Pa / 1e5


@dataclass
class FlashTP:
    """Two-phase equilibrium at given T,P,overall composition (mass basis)."""
    T_K: float
    P_Pa: float
    w_overall: float

    # equilibrium phase compositions (mass fractions)
    wL: float
    wV: float

    # equilibrium phase compositions (mole fractions)
    xL_nh3: float
    yV_nh3: float

    # phase thermodynamic properties (mass basis)
    hL_J_per_kg: float
    hV_J_per_kg: float
    rhoL_kg_per_m3: float
    rhoV_kg_per_m3: float

    # vapor mass fraction of the bulk mixture
    q_mass: float

    note: str = ""

    @property
    def h_bulk(self) -> float:
        return (1.0 - self.q_mass) * self.hL_J_per_kg + self.q_mass * self.hV_J_per_kg


# ------------------------------- backend interface ----------------------

class NH3H2OBackend:
    name: str = "base"

    def state_tp(self, T_K: float, P_Pa: float, w_nh3: float, phase_hint: Optional[str] = None) -> State:
        raise NotImplementedError

    def flash_tp(self, T_K: float, P_Pa: float, w_overall: float) -> FlashTP:
        raise NotImplementedError


# -------------------------- TRF mixture EOS backend ---------------------

# Arbitrary reducing constants used in Φ^0 (Eq. 5)
_TRF_TN0 = 500.0        # K
_TRF_RHON0 = 15000.0    # mol/m^3  (since 1/Vn° = 15000 mol m^-3)

# Pure-fluid critical constants used in mixture reducing functions (Table 2)
_TRF_Tc01 = 647.096       # K (water)
_TRF_rhoc01_mass = 322.0  # kg/m^3
_TRF_M1 = 0.018015268     # kg/mol (water)

_TRF_Tc02 = 405.40        # K (ammonia)
_TRF_rhoc02_mass = 225.0  # kg/m^3
_TRF_M2 = 0.01703026      # kg/mol (ammonia)

# Reducing-function parameters (Table 2)
_TRF_kV = 1.2395117
_TRF_kT = 0.9648407
_TRF_alpha = 1.125455
_TRF_beta = 0.8978069

# Departure-function parameter (Table 2)
_TRF_gamma = 0.5248379

# Table 1: coefficients for Φ^0 (Eq. 5)
_TRF_a0 = {
    1: -7.720435,
    2:  8.649358,
    3: -3.00632,
    4:  0.012436,
    5:  0.97315,
    6:  1.27950,
    7:  0.96956,
    8:  0.24873,
    9: -16.444285,
    10: 4.036946,
    11: -1.0,
    12: 10.69955,
    13: -1.775436,
    14: 0.82374034,
}
_TRF_u = {4: 1.666, 5: 4.578, 6: 10.018, 7: 11.964, 8: 35.600}
_TRF_tpow = {12: 1.0/3.0, 13: 23.0/2.0, 14: 27.0/4.0}

# Table 2: coefficients for ΔΦ^r (Eq. 13)
_TRF_dep_terms = {
    1:  {"a": -1.855822e-02, "t": 3.0/2.0,  "d": 4.0,  "e": None},
    2:  {"a":  5.258010e-02, "t": 1.0/2.0,  "d": 5.0,  "e": 1.0},
    3:  {"a":  3.552874e-10, "t": 13.0/2.0, "d": 15.0, "e": 1.0},
    4:  {"a":  5.451379e-06, "t": 7.0/4.0,  "d": 12.0, "e": 1.0},
    5:  {"a": -5.998546e-13, "t": 15.0,     "d": 12.0, "e": 1.0},
    6:  {"a": -3.687808e-06, "t": 6.0,      "d": 15.0, "e": 2.0},
    7:  {"a":  2.586192e-01, "t": 2.0,      "d": 4.0,  "e": 1.0},
    8:  {"a": -1.368072e-08, "t": 4.0,      "d": 15.0, "e": 1.0},
    9:  {"a":  1.226146e-02, "t": 7.0/2.0,  "d": 4.0,  "e": 1.0},
    10: {"a": -7.181443e-02, "t": 0.0,      "d": 5.0,  "e": 1.0},
    11: {"a":  9.970849e-02, "t": 2.0,      "d": 6.0,  "e": 2.0},
    12: {"a":  1.0584086e-03,"t": 8.0,      "d": 10.0, "e": 2.0},
    13: {"a": -1.963687e-01, "t": 15.0/2.0, "d": 6.0,  "e": 2.0},
    14: {"a": -7.777897e-01, "t": 4.0,      "d": 2.0,  "e": 2.0},
}


def _phi0_trf(T_K: float, rho_molar: float, x_nh3: float) -> float:
    """Dimensionless ideal-gas mixture Helmholtz Φ^0 from Eq. (5)."""
    x = _clamp_x(x_nh3)
    t0 = _TRF_TN0 / T_K
    d0 = rho_molar / _TRF_RHON0

    val = math.log(d0)  # ln(d°)

    # Water bracket
    water = (_TRF_a0[1] + _TRF_a0[2] * t0 + _TRF_a0[3] * math.log(t0) + math.log(1.0 - x))
    for i in (4, 5, 6, 7, 8):
        ui = _TRF_u[i]
        water += _TRF_a0[i] * math.log(1.0 - math.exp(-ui * t0))

    # Ammonia bracket
    nh3 = (_TRF_a0[9] + _TRF_a0[10] * t0 + _TRF_a0[11] * math.log(t0) + math.log(x))
    for i in (12, 13, 14):
        nh3 += _TRF_a0[i] * (t0 ** _TRF_tpow[i])

    return val + (1.0 - x) * water + x * nh3


def _Tc12() -> float:
    return (1.0 / _TRF_kT) * math.sqrt(_TRF_Tc01 * _TRF_Tc02)


def _Vc01() -> float:
    return _TRF_M1 / _TRF_rhoc01_mass  # m^3/mol


def _Vc02() -> float:
    return _TRF_M2 / _TRF_rhoc02_mass  # m^3/mol


def _Vc12() -> float:
    return (1.0 / _TRF_kV) * (_Vc01() + _Vc02())


def _Tn(x_nh3: float) -> float:
    x = _clamp_x(x_nh3)
    Tc12 = _Tc12()
    return (1.0 - x) ** 2 * _TRF_Tc01 + x ** 2 * _TRF_Tc02 + 2.0 * x * (1.0 - x ** _TRF_alpha) * Tc12


def _Vn(x_nh3: float) -> float:
    x = _clamp_x(x_nh3)
    Vc01 = _Vc01()
    Vc02 = _Vc02()
    Vc12 = _Vc12()
    return (1.0 - x) ** 2 * Vc01 + x ** 2 * Vc02 + 2.0 * x * (1.0 - x ** _TRF_beta) * Vc12


def _delta_phir_trf(T_K: float, rho_molar: float, x_nh3: float) -> float:
    x = _clamp_x(x_nh3)
    Tn = _Tn(x)
    Vn = _Vn(x)
    tau = Tn / T_K
    delta = rho_molar * Vn

    pref = x * (1.0 - x ** _TRF_gamma)

    # i=1 term (no exponential damping)
    t1 = _TRF_dep_terms[1]
    val = t1["a"] * (tau ** t1["t"]) * (delta ** t1["d"])

    # i=2..6
    for i in (2, 3, 4, 5, 6):
        ti = _TRF_dep_terms[i]
        val += ti["a"] * math.exp(-(delta ** ti["e"])) * (tau ** ti["t"]) * (delta ** ti["d"])

    # + x * (i=7..13)
    s = 0.0
    for i in (7, 8, 9, 10, 11, 12, 13):
        ti = _TRF_dep_terms[i]
        s += ti["a"] * math.exp(-(delta ** ti["e"])) * (tau ** ti["t"]) * (delta ** ti["d"])
    val += x * s

    # i=14
    t14 = _TRF_dep_terms[14]
    val += (x ** 2) * t14["a"] * math.exp(-(delta ** t14["e"])) * (tau ** t14["t"]) * (delta ** t14["d"])

    return pref * val


class _PureCoolProp:
    """Wrap a CoolProp AbstractState for repeated residual Helmholtz calls."""
    def __init__(self, fluid: str, molar_mass: float):
        try:
            import CoolProp.CoolProp as CP  # type: ignore
            from CoolProp import AbstractState  # type: ignore
        except Exception as e:
            raise BackendUnavailableError("CoolProp not available. Install with: pip install CoolProp") from e
        self._CP = CP
        self._AS = AbstractState("HEOS", fluid)
        self._M = molar_mass

    def alphar(self, T_K: float, rho_molar: float) -> float:
        rho_mass = rho_molar * self._M
        self._AS.update(self._CP.DmassT_INPUTS, rho_mass, T_K)
        if hasattr(self._AS, "alphar"):
            return float(self._AS.alphar())
        for key in ("iAlphaR", "iALPHAR", "iAlpha_r"):
            if hasattr(self._CP, key):
                return float(self._AS.keyed_output(getattr(self._CP, key)))
        raise BackendUnavailableError("CoolProp build lacks alphar() / iAlphaR. Upgrade CoolProp.")


class NH3H2OTRFBackend(NH3H2OBackend):
    """
    NH3–H2O backend based on a TRF-style mixture Helmholtz EOS.

    a(T,ρ,x) = R T [ Φ^0(T,ρ,x) + (1-x) α^r_water(T,ρ) + x α^r_nh3(T,ρ) + ΔΦ^r(T,ρ,x) ]

    p, s, h obtained via numerical derivatives of a(T,ρ,x).
    Flash solves: p_L=p_V=P and μ_i^L=μ_i^V (i in {NH3,H2O}).
    """
    name = "trf"

    def __init__(self):
        try:
            from scipy import optimize  # type: ignore
        except Exception as e:
            raise BackendUnavailableError("SciPy not available. Install with: pip install scipy") from e
        self._opt = optimize
        self._water = _PureCoolProp("Water", _TRF_M1)
        self._nh3 = _PureCoolProp("Ammonia", _TRF_M2)

    # ---------- Helmholtz kernel ----------

    def _phi_total(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        x = _clamp_x(x_nh3)
        phi0 = _phi0_trf(T_K, rho_molar, x)
        dep = _delta_phir_trf(T_K, rho_molar, x)
        ar_w = self._water.alphar(T_K, rho_molar)
        ar_a = self._nh3.alphar(T_K, rho_molar)
        phir = (1.0 - x) * ar_w + x * ar_a + dep
        return phi0 + phir

    def _a_molar(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        return _RU * T_K * self._phi_total(T_K, rho_molar, x_nh3)

    # ---------- numerical derivatives ----------

    def _dadrho(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        r = max(1e-12, rho_molar)
        h = max(1e-6, abs(r) * 1e-6)
        if r - h <= 0:
            return (self._a_molar(T_K, r + h, x_nh3) - self._a_molar(T_K, r, x_nh3)) / h
        return (self._a_molar(T_K, r + h, x_nh3) - self._a_molar(T_K, r - h, x_nh3)) / (2.0 * h)

    def _dadT(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        T = max(1e-6, T_K)
        h = max(1e-3, abs(T) * 1e-6)
        if T - h <= 1e-6:
            return (self._a_molar(T + h, rho_molar, x_nh3) - self._a_molar(T, rho_molar, x_nh3)) / h
        return (self._a_molar(T + h, rho_molar, x_nh3) - self._a_molar(T - h, rho_molar, x_nh3)) / (2.0 * h)

    def _dadx(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        x = _clamp_x(x_nh3)
        h = min(1e-6, 0.05 * min(x, 1.0 - x))
        h = max(h, 1e-10)
        x1 = _clamp_x(x - h)
        x2 = _clamp_x(x + h)
        return (self._a_molar(T_K, rho_molar, x2) - self._a_molar(T_K, rho_molar, x1)) / (x2 - x1)

    # ---------- thermodynamic thermodynamic properties (molar) ----------

    def _p(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        rho = max(1e-12, rho_molar)
        return rho * rho * self._dadrho(T_K, rho, x_nh3)

    def _s_molar(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        return -self._dadT(T_K, rho_molar, x_nh3)

    def _u_molar(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        a = self._a_molar(T_K, rho_molar, x_nh3)
        s = self._s_molar(T_K, rho_molar, x_nh3)
        return a + T_K * s

    def _h_molar(self, T_K: float, rho_molar: float, x_nh3: float) -> float:
        rho = max(1e-12, rho_molar)
        u = self._u_molar(T_K, rho, x_nh3)
        p = self._p(T_K, rho, x_nh3)
        return u + p / rho

    def _chemical_potentials(self, T_K: float, rho_molar: float, x_nh3: float) -> tuple[float, float]:
        x = _clamp_x(x_nh3)
        rho = max(1e-12, rho_molar)
        a = self._a_molar(T_K, rho, x)
        p = self._p(T_K, rho, x)
        ax = self._dadx(T_K, rho, x)

        common = a + p / rho
        mu_nh3 = common + (1.0 - x) * ax
        mu_h2o = common - x * ax
        return mu_nh3, mu_h2o

    # ---------- density solver: rho(T,P,x) ----------

    def _rho_roots(self, T_K: float, P_Pa: float, x_nh3: float) -> list[float]:
        def f(rho: float) -> float:
            return self._p(T_K, rho, x_nh3) - P_Pa

        rho_min = 1e-6
        rho_max = 8e4
        N = 220
        rhos = [rho_min * (rho_max / rho_min) ** (i / (N - 1)) for i in range(N)]

        vals: list[float] = []
        for r in rhos:
            try:
                vals.append(f(r))
            except Exception:
                vals.append(float("nan"))

        brackets: list[tuple[float, float]] = []
        for i in range(N - 1):
            v1, v2 = vals[i], vals[i + 1]
            if not (math.isfinite(v1) and math.isfinite(v2)):
                continue
            if v1 == 0.0:
                brackets.append((rhos[i], rhos[i]))
            if v1 * v2 < 0.0:
                brackets.append((rhos[i], rhos[i + 1]))

        roots: list[float] = []
        for a, b in brackets:
            if a == b:
                roots.append(a)
                continue
            try:
                root = float(self._opt.brentq(f, a, b, maxiter=120))
                roots.append(root)
            except Exception:
                continue

        return sorted(set(round(r, 10) for r in roots))

    def _rho_tp(self, T_K: float, P_Pa: float, x_nh3: float, phase_hint: Optional[str]) -> float:
        roots = self._rho_roots(T_K, P_Pa, x_nh3)
        if roots:
            hint = (phase_hint or "").lower()
            if hint.startswith("v"):
                return roots[0]
            if hint.startswith("l"):
                return roots[-1]
            rho_ig = P_Pa / (_RU * T_K)
            return min(roots, key=lambda r: abs(r - rho_ig))

        # fallback: solve on log(rho)
        rho0 = max(1e-3, P_Pa / (_RU * T_K))
        if (phase_hint or "").lower().startswith("l"):
            rho0 = 3e4

        def g(log_r: float) -> float:
            r = math.exp(log_r)
            return self._p(T_K, r, x_nh3) - P_Pa

        sol = self._opt.root(lambda y: [g(y[0])], x0=[math.log(rho0)], method="hybr")
        if not sol.success:
            raise ThermoEvalError(f"rho(T,P,x) solve failed: {sol.message}")
        return float(math.exp(sol.x[0]))

    # ---------- mass-basis state builder ----------

    def _state_from_rho(self, T_K: float, P_target: float, x_nh3: float, rho_molar: float,
                        w_nh3: float, note: str) -> State:
        p = self._p(T_K, rho_molar, x_nh3)
        mismatch = abs(p - P_target) / max(P_target, 1.0)
        extra = f", p_err={mismatch:.2e}" if mismatch > 2e-3 else ""

        h_m = self._h_molar(T_K, rho_molar, x_nh3)
        s_m = self._s_molar(T_K, rho_molar, x_nh3)

        Mmix = molar_mass_mix(x_nh3, M_nh3=_TRF_M2, M_h2o=_TRF_M1)
        rho_mass = rho_molar * Mmix
        h_mass = h_m / Mmix
        s_mass = s_m / Mmix

        return State(
            T_K=T_K, P_Pa=P_target, w_nh3=w_nh3,
            h_J_per_kg=h_mass, rho_kg_per_m3=rho_mass, s_J_per_kgK=s_mass,
            note=f"{note}{extra}",
        )

    # ---------- NH3H2OBackend API ----------

    def state_tp(self, T_K: float, P_Pa: float, w_nh3: float, phase_hint: Optional[str] = None) -> State:
        x = massfrac_to_molefrac(w_nh3, M_nh3=_TRF_M2, M_h2o=_TRF_M1)
        rho = self._rho_tp(T_K, P_Pa, x, phase_hint)
        return self._state_from_rho(
            T_K=T_K, P_target=P_Pa, x_nh3=x, rho_molar=rho, w_nh3=w_nh3,
            note=f"TRF single-phase ({phase_hint or 'auto'})",
        )

    def flash_tp(self, T_K: float, P_Pa: float, w_overall: float) -> FlashTP:
        z = massfrac_to_molefrac(w_overall, M_nh3=_TRF_M2, M_h2o=_TRF_M1)

        # Unknowns: log(rhoL), log(rhoV), xL, yV
        def F(u):
            log_rL, log_rV, xL, yV = u
            if not (1e-10 < xL < 1.0 - 1e-10 and 1e-10 < yV < 1.0 - 1e-10):
                return [1e6, 1e6, 1e6, 1e6]
            rL = math.exp(log_rL)
            rV = math.exp(log_rV)
            if rL <= rV:
                return [1e6, 1e6, 1e6, 1e6]
            if yV <= xL:
                return [1e5, 1e5, 1e5, 1e5]

            pL = self._p(T_K, rL, xL) - P_Pa
            pV = self._p(T_K, rV, yV) - P_Pa
            muN_L, muW_L = self._chemical_potentials(T_K, rL, xL)
            muN_V, muW_V = self._chemical_potentials(T_K, rV, yV)

            return [muN_L - muN_V, muW_L - muW_V, pL, pV]

        rhoV_ig = max(1e-6, P_Pa / (_RU * T_K))
        guesses = [
            (3e4, rhoV_ig, clamp(z - 0.25, 1e-6, 0.999999), clamp(z + 0.25, 1e-6, 0.999999)),
            (2e4, rhoV_ig, clamp(z - 0.15, 1e-6, 0.999999), clamp(z + 0.15, 1e-6, 0.999999)),
            (4e4, rhoV_ig, clamp(z - 0.35, 1e-6, 0.999999), clamp(z + 0.35, 1e-6, 0.999999)),
        ]

        last_err: Optional[str] = None
        sol = None
        for rhoL0, rhoV0, xL0, yV0 in guesses:
            try:
                sol_try = self._opt.root(
                    F,
                    x0=[math.log(rhoL0), math.log(rhoV0), xL0, yV0],
                    method="hybr",
                    options={"maxfev": 4000},
                )
                if sol_try.success:
                    sol = sol_try
                    break
                last_err = str(sol_try.message)
            except Exception as e:
                last_err = str(e)

        if sol is None:
            # Single-phase fallback (still returns FlashTP so upstream code doesn't explode)
            st_auto = self.state_tp(T_K, P_Pa, w_overall, phase_hint=None)

            # crude region decision from density
            # (vapor densities are typically << 50 kg/m3 here)
            q_mass = 1.0 if st_auto.rho_kg_per_m3 < 50.0 else 0.0

            return FlashTP(
                T_K=T_K, P_Pa=P_Pa, w_overall=w_overall,
                wL=w_overall, wV=w_overall, xL_nh3=z, yV_nh3=z,
                hL_J_per_kg=st_auto.h_J_per_kg, hV_J_per_kg=st_auto.h_J_per_kg,
                rhoL_kg_per_m3=st_auto.rho_kg_per_m3, rhoV_kg_per_m3=st_auto.rho_kg_per_m3,
                q_mass=q_mass,
                note=f"TRF flash fallback: treated as single-phase (reason: {last_err})",
            )

        log_rL, log_rV, xL, yV = [float(v) for v in sol.x]
        rL = float(math.exp(log_rL))
        rV = float(math.exp(log_rV))

        denom = (yV - xL)
        beta = 0.0 if abs(denom) < 1e-14 else (z - xL) / denom
        beta = clamp(float(beta), 0.0, 1.0)

        Ml = molar_mass_mix(xL, M_nh3=_TRF_M2, M_h2o=_TRF_M1)
        Mv = molar_mass_mix(yV, M_nh3=_TRF_M2, M_h2o=_TRF_M1)
        q_mass = 0.0 if (beta * Mv + (1.0 - beta) * Ml) <= 0 else (beta * Mv) / (beta * Mv + (1.0 - beta) * Ml)

        wL = molefrac_to_massfrac(xL, M_nh3=_TRF_M2, M_h2o=_TRF_M1)
        wV = molefrac_to_massfrac(yV, M_nh3=_TRF_M2, M_h2o=_TRF_M1)

        stL = self._state_from_rho(T_K, P_Pa, xL, rL, wL, note="TRF sat liquid")
        stV = self._state_from_rho(T_K, P_Pa, yV, rV, wV, note="TRF sat vapor")

        return FlashTP(
            T_K=T_K, P_Pa=P_Pa, w_overall=w_overall,
            wL=wL, wV=wV, xL_nh3=xL, yV_nh3=yV,
            hL_J_per_kg=stL.h_J_per_kg, hV_J_per_kg=stV.h_J_per_kg,
            rhoL_kg_per_m3=stL.rho_kg_per_m3, rhoV_kg_per_m3=stV.rho_kg_per_m3,
            q_mass=q_mass,
            note="TRF flash (μ-equality + p-equality)",
        )


# ------------------------------- cycle model ----------------------------

@dataclass
class KN9B6Config:
    w1_NH3: float = 0.48
    m1: float = 0.05                  # kg/s strong solution
    T1_C: float = 80.0                # generator inlet (after SHX), point 1
    P_high_bar: float = 13.5
    T_gen_C: float = 115.0            # generator flash temperature
    T_cond_C: float = 27.0            # condenser outlet temperature
    T_abs_C: float = 27.0             # absorber outlet temperature (strong solution)
    P_low_bar: float = 3.0
    T_evap_out_C: float = 5.0         # evaporator outlet bulk temperature (state 6)
    pump_ideal: bool = True


def solve_T_for_ph(
    backend: NH3H2OBackend,
    P_Pa: float,
    h_target: float,
    w_nh3: float,
    T_lo: float = 200.0,
    T_hi: float = 450.0,
    tol: float = 1e-5,
    max_iter: int = 160,
) -> float:
    """Solve T such that bulk enthalpy at (T,P,w) equals h_target."""
    def h_bulk(T: float) -> float:
        try:
            f = backend.flash_tp(T, P_Pa, w_nh3)
            return f.h_bulk
        except Exception:
            return backend.state_tp(T, P_Pa, w_nh3).h_J_per_kg

    f_lo = h_bulk(T_lo) - h_target
    f_hi = h_bulk(T_hi) - h_target

    expand = 0
    while f_lo * f_hi > 0 and expand < 14:
        T_lo = max(120.0, T_lo - 25.0)
        T_hi = min(700.0, T_hi + 25.0)
        f_lo = h_bulk(T_lo) - h_target
        f_hi = h_bulk(T_hi) - h_target
        expand += 1

    if f_lo * f_hi > 0:
        raise ThermoEvalError(
            f"solve_T_for_ph could not bracket root. f(T_lo)={f_lo:.3e}, f(T_hi)={f_hi:.3e}"
        )

    for _ in range(max_iter):
        T_mid = 0.5 * (T_lo + T_hi)
        f_mid = h_bulk(T_mid) - h_target
        if abs(f_mid) < tol:
            return T_mid
        if f_lo * f_mid <= 0:
            T_hi, f_hi = T_mid, f_mid
        else:
            T_lo, f_lo = T_mid, f_mid

    return 0.5 * (T_lo + T_hi)


def solve_T_for_q1(
    backend: NH3H2OBackend,
    P_Pa: float,
    w_nh3: float,
    T_start: float,
    T_max: float = 520.0,
    q_target: float = 0.999,
    tolT: float = 1e-3,
) -> float:
    """Find T where q_mass ≈ 1 at given (P,w), via bisection on T."""
    T_lo = T_start
    q_lo = backend.flash_tp(T_lo, P_Pa, w_nh3).q_mass

    T_hi = min(T_max, T_lo + 5.0)
    q_hi = backend.flash_tp(T_hi, P_Pa, w_nh3).q_mass
    while q_hi < q_target and T_hi < T_max:
        T_hi = min(T_max, T_hi + 5.0)
        q_hi = backend.flash_tp(T_hi, P_Pa, w_nh3).q_mass

    if q_hi < q_target:
        raise ThermoEvalError("Could not reach q≈1 within search range for solve_T_for_q1().")

    for _ in range(160):
        T_mid = 0.5 * (T_lo + T_hi)
        q_mid = backend.flash_tp(T_mid, P_Pa, w_nh3).q_mass
        if abs(T_hi - T_lo) < tolT:
            return T_mid
        if q_mid >= q_target:
            T_hi = T_mid
        else:
            T_lo = T_mid
    return 0.5 * (T_lo + T_hi)


def compute_cycle_kn_9b6(cfg: KN9B6Config, backend: NH3H2OBackend) -> Dict[str, Any]:
    P_high = cfg.P_high_bar * 1e5
    P_low = cfg.P_low_bar * 1e5

    T1 = cfg.T1_C + 273.15
    Tgen = cfg.T_gen_C + 273.15
    Tcond = cfg.T_cond_C + 273.15
    Tabs = cfg.T_abs_C + 273.15
    Tevap_out = cfg.T_evap_out_C + 273.15

    w1 = cfg.w1_NH3
    m1 = cfg.m1

    # --- 7 (absorber outlet strong solution), 8 (after pump) ---
    st7 = backend.state_tp(Tabs, P_low, w1, phase_hint="liquid")
    dp = P_high - P_low
    w_p = dp / max(st7.rho_kg_per_m3, 1e-9)
    W_dot_p = m1 * w_p

    h8 = st7.h_J_per_kg + w_p
    T8 = solve_T_for_ph(backend, P_high, h8, w1, T_lo=Tabs - 20, T_hi=Tabs + 20)
    st8 = backend.state_tp(T8, P_high, w1, phase_hint="liquid")

    # --- 1 (after SHX, to generator) ---
    st1 = backend.state_tp(T1, P_high, w1, phase_hint="liquid")

    # --- generator flash at (Tgen, P_high, w1) -> 2 vapor, 3 liquid ---
    gen = backend.flash_tp(Tgen, P_high, w1)
    m2 = m1 * gen.q_mass
    m3 = m1 - m2

    h2 = gen.hV_J_per_kg
    h3 = gen.hL_J_per_kg
    w2 = gen.wV
    w3 = gen.wL

    Q_dot_gen = m2 * h2 + m3 * h3 - m1 * st1.h_J_per_kg

    # --- solution heat exchanger (3 hot, 8 cold) ---
    Q_dot_shx = m1 * (st1.h_J_per_kg - st8.h_J_per_kg)
    h9 = h3 - Q_dot_shx / max(m3, 1e-12)
    T9 = solve_T_for_ph(backend, P_high, h9, w3, T_lo=Tabs, T_hi=Tgen + 30)
    st9 = backend.state_tp(T9, P_high, w3, phase_hint="liquid")

    # valve 9 -> 10 (solution throttle): h10 = h9 at P_low
    T10 = solve_T_for_ph(backend, P_low, h9, w3, T_lo=160.0, T_hi=T9 + 20)
    st10 = backend.state_tp(T10, P_low, w3)

    # SHX effectiveness ε = (T1 - T8)/(T3 - T8) with T3 ~ generator temperature
    eps_shx = (st1.T_K - st8.T_K) / (Tgen - st8.T_K) if abs(Tgen - st8.T_K) > 1e-12 else float("nan")

    # --- condenser: 2 -> 4 at P_high, Tcond (assume liquid outlet) ---
    st4 = backend.state_tp(Tcond, P_high, w2, phase_hint="liquid")
    Q_dot_cond = m2 * (h2 - st4.h_J_per_kg)

    # valve 4 -> 5 (refrigerant throttle): h5 = h4 at P_low
    h5 = st4.h_J_per_kg
    T5 = solve_T_for_ph(backend, P_low, h5, w2, T_lo=160.0, T_hi=Tcond + 10)
    st5_bulk = backend.flash_tp(T5, P_low, w2)

    # --- evaporator: 5 -> 6 ---
    evap_out = backend.flash_tp(Tevap_out, P_low, w2)
    h6 = evap_out.h_bulk
    Q_dot_evap = m2 * (h6 - h5)

    # temperature required for full evaporation at P_low, w2
    T6_req = solve_T_for_q1(backend, P_low, w2, T_start=Tevap_out, T_max=520.0)

    # --- absorber heat rejected ---
    Q_dot_abs = (m2 * h6 + m3 * st10.h_J_per_kg) - (m1 * st7.h_J_per_kg)

    COP_gen_only = Q_dot_evap / Q_dot_gen
    COP_with_pump = Q_dot_evap / (Q_dot_gen + W_dot_p)

    return {
        "cfg": asdict(cfg),
        "backend": getattr(backend, "name", type(backend).__name__),
        "states": {
            "1": asdict(st1),
            "2_vapor": {"T_K": Tgen, "P_Pa": P_high, "w_nh3": w2, "h_J_per_kg": h2},
            "3_liquid": {"T_K": Tgen, "P_Pa": P_high, "w_nh3": w3, "h_J_per_kg": h3},
            "4": asdict(st4),
            "5": {
                "T_K": T5, "T_C": T5 - 273.15, "P_Pa": P_low, "w_nh3": w2,
                "h_J_per_kg": h5, "q_mass": st5_bulk.q_mass, "wL": st5_bulk.wL, "wV": st5_bulk.wV,
                "note": st5_bulk.note,
            },
            "6": {
                "T_K": Tevap_out, "T_C": cfg.T_evap_out_C, "P_Pa": P_low, "w_nh3": w2,
                "h_J_per_kg": h6, "q_mass": evap_out.q_mass, "wL": evap_out.wL, "wV": evap_out.wV,
                "note": evap_out.note,
            },
            "7": asdict(st7),
            "8": asdict(st8),
            "9": asdict(st9),
            "10": asdict(st10),
        },
        "mass_flows": {"m1": m1, "m2_refrig": m2, "m3_solution": m3},
        "compositions": {"w1_strong": w1, "w2_refrig": w2, "w3_weak": w3},
        "heat_rates_W": {
            "Q_dot_gen": Q_dot_gen,
            "Q_dot_cond": Q_dot_cond,
            "Q_dot_evap": Q_dot_evap,
            "Q_dot_abs": Q_dot_abs,
            "Q_dot_shx": Q_dot_shx,
        },
        "pump": {"w_p_J_per_kg": w_p, "W_dot_p_W": W_dot_p},
        "metrics": {
            "T5_inlet_evap_C": T5 - 273.15,
            "eps_solution_hx": eps_shx,
            "T6_req_full_evap_C": T6_req - 273.15,
            "COP_gen_only": COP_gen_only,
            "COP_with_pump": COP_with_pump,
        },
    }


# ------------------------------- CLI / main -----------------------------

def _load_cfg_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_cfg_overrides(cfg: KN9B6Config, d: dict) -> KN9B6Config:
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build_backend(name: str) -> NH3H2OBackend:
    name = (name or "trf").lower().strip()
    if name in ("trf", "tillner-roth", "tillnerroth"):
        return NH3H2OTRFBackend()
    raise BackendUnavailableError(f"Unknown backend: {name}. Only 'trf' is supported in this script.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ammonia_cooling_trf.py", add_help=True)
    p.add_argument("--backend", default="trf", help="Backend: trf (only supported)")
    p.add_argument("--mode", default="cycle", choices=("cycle", "state", "flash"), help="What to run")
    p.add_argument("--infile", default="", help="Optional JSON file with KN9B6Config overrides (mode=cycle)")
    p.add_argument("--out", default="", help="Write full JSON result to this path (cycle/flash/state)")

    # state/flash args
    p.add_argument("--T_C", type=float, default=27.0, help="Temperature [°C] for state/flash modes")
    p.add_argument("--P_bar", type=float, default=1.0, help="Pressure [bar] for state/flash modes")
    p.add_argument("--w", type=float, default=0.5, help="NH3 mass fraction for state/flash modes")
    p.add_argument("--phase", default="auto", choices=("auto", "liquid", "vapor"), help="Phase hint for state()")

    p.add_argument("--pretty", action="store_true", help="Pretty-print key results to stdout (default)")
    p.add_argument("--no-pretty", action="store_true", help="Disable pretty printing")
    p.add_argument("--debug", action="store_true", help="Show full stack traces on failure")

    args = p.parse_args(argv)

    pretty = True
    if args.no_pretty:
        pretty = False
    if args.pretty:
        pretty = True

    try:
        backend = build_backend(args.backend)

        if args.mode == "cycle":
            cfg = KN9B6Config()
            if args.infile:
                overrides = _load_cfg_json(args.infile)
                cfg = _apply_cfg_overrides(cfg, overrides)

            res = compute_cycle_kn_9b6(cfg, backend)

            if pretty:
                m = res["metrics"]
                print(f"[ammonia_cooling_trf] backend={backend.name}")
                print("[ammonia_cooling_trf] K&N 9.B-6 metrics:")
                print(f"  (a) T5 (into evaporator) = {m['T5_inlet_evap_C']:.3f} °C")
                print(f"  (b) ε_solution_HX        = {m['eps_solution_hx']:.4f}")
                print(f"  (c) T6 for q≈1           = {m['T6_req_full_evap_C']:.3f} °C")
                print(f"  (d) COP (Qe/Qg)          = {m['COP_gen_only']:.4f}")
                print(f"      COP (incl pump)      = {m['COP_with_pump']:.4f}")

        elif args.mode == "state":
            T_K = args.T_C + 273.15
            P_Pa = args.P_bar * 1e5
            ph = None if args.phase == "auto" else args.phase
            st = backend.state_tp(T_K, P_Pa, args.w, phase_hint=ph)
            res = {"backend": backend.name, "state": asdict(st)}

            if pretty:
                print(f"[ammonia_cooling_trf] backend={backend.name}")
                print(json.dumps(res["state"], indent=2))

        elif args.mode == "flash":
            T_K = args.T_C + 273.15
            P_Pa = args.P_bar * 1e5
            fl = backend.flash_tp(T_K, P_Pa, args.w)
            res = {"backend": backend.name, "flash": asdict(fl), "h_bulk_J_per_kg": fl.h_bulk}

            if pretty:
                print(f"[ammonia_cooling_trf] backend={backend.name}")
                print(json.dumps(res, indent=2))

        else:
            raise BackendUnavailableError(f"Unknown mode: {args.mode}")

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            if pretty:
                print(f"[ammonia_cooling_trf] wrote: {args.out}")

        return 0

    except Exception as e:
        if args.debug:
            raise
        print(f"[ammonia_cooling_trf] ERROR: {e}", file=sys.stderr)
        print("Use --debug for full stack trace.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "M_NH3",
    "M_H2O",
    "BackendUnavailableError",
    "ThermoEvalError",
    "State",
    "FlashTP",
    "NH3H2OBackend",
    "NH3H2OTRFBackend",
    "KN9B6Config",
    "clamp",
    "massfrac_to_molefrac",
    "molefrac_to_massfrac",
    "molar_mass_mix",
    "solve_T_for_ph",
    "solve_T_for_q1",
    "compute_cycle_kn_9b6",
    "main",
]
