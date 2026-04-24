#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nh3h2o/ammonia_cooling.py

Sandbox script for Klein & Nellis exercise 9.B-6 (simplified NH3–H2O absorption cycle).

This file is intentionally *self-contained* so you can iterate fast before wiring an NH3H2O backend
into the main tdpy equation-solver pipeline (similar to LiBrPropsSI).

Backends
--------
1) REFPROP (recommended for now): matches EES NH3H2O reasonably well.
   Requires:
     - NIST REFPROP installed + licensed
     - pip install ctREFPROP
     - env var RPPREFIX pointing to your REFPROP directory

2) Pure Python Tillner–Roth–Friend (TRF) Helmholtz EOS:
   Not implemented here yet. See notes at bottom for what you’ll need.

What this script computes
------------------------------------------------------------
For K&N 9.B-6 givens, it computes:
- (a) T5: temperature entering evaporator after throttling (state 5)
- (b) ε_shx: effectiveness of solution heat exchanger
- (c) T6_req: evaporator outlet temperature required for full evaporation (q6=1)
- (d) COP (with and without pump work)

States follow the figure in the problem statement.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


# ------------------------------- constants -------------------------------

M_NH3 = 17.03052e-3   # kg/mol
M_H2O = 18.01528e-3   # kg/mol


# ------------------------------- helpers --------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def massfrac_to_molefrac(w_nh3: float) -> Tuple[float, float]:
    """Convert NH3 mass fraction to (x_NH3, x_H2O) mole fractions."""
    w = clamp(w_nh3, 0.0, 1.0)
    n_nh3 = w / M_NH3
    n_h2o = (1.0 - w) / M_H2O
    n_tot = n_nh3 + n_h2o
    if n_tot <= 0:
        return 0.0, 1.0
    x_nh3 = n_nh3 / n_tot
    return x_nh3, 1.0 - x_nh3


def molefrac_to_massfrac(x_nh3: float) -> float:
    """Convert NH3 mole fraction to NH3 mass fraction."""
    x = clamp(x_nh3, 0.0, 1.0)
    m_nh3 = x * M_NH3
    m_h2o = (1.0 - x) * M_H2O
    m_tot = m_nh3 + m_h2o
    if m_tot <= 0:
        return 0.0
    return m_nh3 / m_tot


def molar_mass_from_molefrac(x_nh3: float) -> float:
    x = clamp(x_nh3, 0.0, 1.0)
    return x * M_NH3 + (1.0 - x) * M_H2O


# ------------------------------- errors ---------------------------------

class BackendUnavailableError(RuntimeError):
    pass


class ThermoEvalError(RuntimeError):
    pass


# ------------------------------- dataclasses ----------------------------

@dataclass
class State:
    """Single-phase (or bulk) state."""
    T_K: float
    P_Pa: float
    w_nh3: float
    h_J_per_kg: float
    rho_kg_per_m3: float
    s_J_per_kgK: Optional[float] = None

    @property
    def T_C(self) -> float:
        return self.T_K - 273.15

    @property
    def P_bar(self) -> float:
        return self.P_Pa / 1e5


@dataclass
class FlashTP:
    """Two-phase equilibrium at given T,P,overall composition."""
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

    # vapor mass fraction of the *bulk* mixture
    q_mass: float

    @property
    def h_bulk(self) -> float:
        return (1.0 - self.q_mass) * self.hL_J_per_kg + self.q_mass * self.hV_J_per_kg


# ------------------------------- backend interface ----------------------

class NH3H2OBackend:
    name: str = "base"

    def state_tp(self, T_K: float, P_Pa: float, w_nh3: float) -> State:
        raise NotImplementedError

    def flash_tp(self, T_K: float, P_Pa: float, w_overall: float) -> FlashTP:
        raise NotImplementedError


# ------------------------------- REFPROP backend ------------------------

class NH3H2ORefpropBackend(NH3H2OBackend):
    """
    REFPROP-based backend via ctREFPROP.

    We use REFPROPdll for single-phase thermodynamic properties (mass base SI),
    and TPFLSHdll + THERMdll for VLE (returns x/y and phase densities, then we compute hL/hV).
    """
    name = "refprop"

    def __init__(self, fluids: str = "AMMONIA;WATER", rp_prefix: Optional[str] = None):
        try:
            from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary  # type: ignore
        except Exception as e:  # pragma: no cover
            raise BackendUnavailableError(
                "ctREFPROP not importable. Install with: pip install ctREFPROP"
            ) from e

        rp_prefix = rp_prefix or os.environ.get("RPPREFIX") or os.environ.get("REFPROP_PATH")
        if not rp_prefix:
            raise BackendUnavailableError(
                "REFPROP path not configured. Set RPPREFIX to your REFPROP install directory."
            )

        self._rp_prefix = rp_prefix
        self._fluids = fluids

        RP = REFPROPFunctionLibrary(rp_prefix)
        RP.SETPATHdll(rp_prefix)

        # Prefer setting fluids once, then pass "" as hfld.
        # For mixtures, REFPROP accepts component names separated by ';' (e.g., "AMMONIA;WATER").
        RP.SETFLUIDSdll(fluids)

        # Unit enums (base SI)
        self.RP = RP
        self.MASS_BASE_SI = RP.GETENUMdll(0, "MASS BASE SI").iEnum

    # ---- low-level calls ----

    def _refprop_tp(self, T_K: float, P_Pa: float, w_nh3: float, out: str) -> Tuple[float, ...]:
        x_nh3, x_h2o = massfrac_to_molefrac(w_nh3)
        z = [x_nh3, x_h2o] + [0.0] * 18  # REFPROP expects len 20
        try:
            r = self.RP.REFPROPdll(
                "", "TP", out, self.MASS_BASE_SI, 0, 0, T_K, P_Pa, z
            )
        except Exception as e:
            raise ThermoEvalError(f"REFPROPdll TP failed: {e}") from e
        return tuple(float(v) for v in r.Output)

    def state_tp(self, T_K: float, P_Pa: float, w_nh3: float) -> State:
        # H [J/kg], S [J/kg-K], D [kg/m^3]
        H, S, D = self._refprop_tp(T_K, P_Pa, w_nh3, "H;S;D")
        return State(T_K=T_K, P_Pa=P_Pa, w_nh3=w_nh3, h_J_per_kg=H, s_J_per_kgK=S, rho_kg_per_m3=D)

    def _try_tpflsh(self, T_K: float, P_Pa: float, z_mole_2: Tuple[float, float]):
        """
        TPFLSHdll historically uses kPa for pressure.
        Newer wrappers can be ambiguous depending on build.
        We try Pa first, then kPa.
        """
        z = [z_mole_2[0], z_mole_2[1]] + [0.0] * 18
        last_err = None
        for P_try in (P_Pa, P_Pa / 1000.0):
            try:
                out = self.RP.TPFLSHdll(T_K, P_try, z)
                return out, P_try
            except Exception as e:
                last_err = e
        raise ThermoEvalError(f"TPFLSHdll failed (Pa and kPa attempts). Last error: {last_err}")

    def _try_therm(self, T_K: float, D_mol_per_L: float, z_mole_2: Tuple[float, float]):
        """
        THERMdll uses T [K], D [mol/L], z [mole fractions] and returns thermodynamic properties in molar units.
        """
        z = [z_mole_2[0], z_mole_2[1]] + [0.0] * 18
        try:
            out = self.RP.THERMdll(T_K, D_mol_per_L, z)
            return out
        except Exception as e:
            raise ThermoEvalError(f"THERMdll failed: {e}") from e

    def flash_tp(self, T_K: float, P_Pa: float, w_overall: float) -> FlashTP:
        # Convert overall mass fraction to mole fractions for REFPROP flash routines
        z_nh3, z_h2o = massfrac_to_molefrac(w_overall)
        flsh, P_used = self._try_tpflsh(T_K, P_Pa, (z_nh3, z_h2o))

        # Pull equilibrium compositions (mole fractions)
        try:
            xL = float(flsh.x[0])
            yV = float(flsh.y[0])
            Dl = float(flsh.Dl)  # mol/L
            Dv = float(flsh.Dv)  # mol/L
        except Exception as e:
            raise ThermoEvalError(f"Unexpected TPFLSH output structure: {e}") from e

        # Leverage lever rule to compute vapor molar fraction (beta)
        denom = (yV - xL)
        if abs(denom) < 1e-14:
            # single-phase or near-critical; fall back to single-phase
            st = self.state_tp(T_K, P_Pa, w_overall)
            # treat as all-liquid for bookkeeping
            return FlashTP(
                T_K=T_K, P_Pa=P_Pa, w_overall=w_overall,
                wL=w_overall, wV=w_overall, xL_nh3=z_nh3, yV_nh3=z_nh3,
                hL_J_per_kg=st.h_J_per_kg, hV_J_per_kg=st.h_J_per_kg,
                rhoL_kg_per_m3=st.rho_kg_per_m3, rhoV_kg_per_m3=st.rho_kg_per_m3,
                q_mass=0.0
            )

        beta_molar = clamp((z_nh3 - xL) / denom, 0.0, 1.0)

        # Convert equilibrium compositions to mass fractions
        wL = molefrac_to_massfrac(xL)
        wV = molefrac_to_massfrac(yV)

        # Compute phase molar masses
        M_l = molar_mass_from_molefrac(xL)  # kg/mol
        M_v = molar_mass_from_molefrac(yV)  # kg/mol

        # Convert molar vapor fraction to mass vapor fraction
        m_v = beta_molar * M_v
        m_l = (1.0 - beta_molar) * M_l
        q_mass = 0.0 if (m_v + m_l) <= 0 else m_v / (m_v + m_l)

        # Phase thermodynamic properties via THERMdll at (T, D_phase, z_phase_mole)
        therm_l = self._try_therm(T_K, Dl, (xL, 1.0 - xL))
        therm_v = self._try_therm(T_K, Dv, (yV, 1.0 - yV))

        # THERMdll returns molar h [J/mol]; convert to J/kg
        hL = float(therm_l.h) / M_l
        hV = float(therm_v.h) / M_v

        # Convert molar density (mol/L) -> kg/m^3
        rhoL = Dl * 1000.0 * M_l
        rhoV = Dv * 1000.0 * M_v

        return FlashTP(
            T_K=T_K, P_Pa=P_Pa, w_overall=w_overall,
            wL=wL, wV=wV, xL_nh3=xL, yV_nh3=yV,
            hL_J_per_kg=hL, hV_J_per_kg=hV,
            rhoL_kg_per_m3=rhoL, rhoV_kg_per_m3=rhoV,
            q_mass=q_mass
        )


# ------------------------------- cycle model (K&N 9.B-6) ----------------

@dataclass
class KN9B6Config:
    w1_NH3: float = 0.48
    m1: float = 0.05                  # kg/s
    T1_C: float = 80.0                # generator inlet (after SHX), point 1
    P_high_bar: float = 13.5
    T_gen_C: float = 115.0            # flash generator temperature
    T_cond_C: float = 27.0            # point 4
    T_abs_C: float = 27.0             # point 7
    P_low_bar: float = 3.0
    T_evap_out_C: float = 5.0         # point 6
    pump_ideal: bool = True
    dp_negligible_except_valves: bool = True


def solve_T_for_ph(backend: NH3H2OBackend, P_Pa: float, h_target: float, w_nh3: float,
                   T_lo: float = 200.0, T_hi: float = 450.0, tol: float = 1e-6, max_iter: int = 120) -> float:
    """
    Solve T such that bulk enthalpy at (T,P,w) equals h_target.

    For robustness:
    - uses backend.flash_tp to compute bulk enthalpy in 2-phase region
    - falls back to backend.state_tp if flash is not possible

    Bisection assumes h(T) is monotonic over the bracket (usually true for these paths).
    """
    def h_bulk(T):
        try:
            f = backend.flash_tp(T, P_Pa, w_nh3)
            # If flash succeeded and indicates two-phase, use bulk from q/hL/hV.
            # If it returned "single-phase fallback", q_mass will be 0 and h_bulk ~= hL.
            return f.h_bulk
        except Exception:
            return backend.state_tp(T, P_Pa, w_nh3).h_J_per_kg

    f_lo = h_bulk(T_lo) - h_target
    f_hi = h_bulk(T_hi) - h_target

    # Expand bracket if needed
    expand = 0
    while f_lo * f_hi > 0 and expand < 12:
        # try widening the temperature window
        T_lo = max(120.0, T_lo - 20.0)
        T_hi = min(650.0, T_hi + 20.0)
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


def solve_T_for_q1(backend: NH3H2OBackend, P_Pa: float, w_nh3: float,
                   T_start: float, T_max: float = 450.0, q_target: float = 0.999, tolT: float = 1e-3) -> float:
    """
    Find the temperature at which the mixture becomes essentially all vapor (q ≈ 1) at given P,w.
    Uses bisection over T.

    If backend.flash_tp cannot return two-phase near dew line, this may fail; that’s OK for sandboxing.
    """
    T_lo = T_start
    # ensure T_lo is in two-phase or below dew line
    q_lo = backend.flash_tp(T_lo, P_Pa, w_nh3).q_mass

    # find upper bound where q >= target or flash indicates single vapor (q close to 1)
    T_hi = min(T_max, T_lo + 5.0)
    q_hi = backend.flash_tp(T_hi, P_Pa, w_nh3).q_mass
    while q_hi < q_target and T_hi < T_max:
        T_hi = min(T_max, T_hi + 5.0)
        q_hi = backend.flash_tp(T_hi, P_Pa, w_nh3).q_mass

    if q_hi < q_target:
        raise ThermoEvalError("Could not reach q≈1 within search range for solve_T_for_q1().")

    # bisection on temperature for q(T) = q_target (monotonic increasing near dew line)
    for _ in range(120):
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
    # pressures
    P_high = cfg.P_high_bar * 1e5
    P_low = cfg.P_low_bar * 1e5

    # temperatures
    T1 = cfg.T1_C + 273.15
    Tgen = cfg.T_gen_C + 273.15
    Tcond = cfg.T_cond_C + 273.15
    Tabs = cfg.T_abs_C + 273.15
    Tevap_out = cfg.T_evap_out_C + 273.15

    w1 = cfg.w1_NH3
    m1 = cfg.m1

    # --- state 7 (absorber outlet strong solution), state 8 (after pump) ---
    st7 = backend.state_tp(Tabs, P_low, w1)
    dp = P_high - P_low
    w_p = dp / st7.rho_kg_per_m3  # J/kg (incompressible)
    W_dot_p = m1 * w_p            # W

    h8 = st7.h_J_per_kg + w_p
    T8 = solve_T_for_ph(backend, P_high, h8, w1, T_lo=Tabs - 10, T_hi=Tabs + 10)
    st8 = backend.state_tp(T8, P_high, w1)

    # --- state 1 (after SHX, to generator) ---
    st1 = backend.state_tp(T1, P_high, w1)

    # --- generator flash at (Tgen, P_high, w1) -> states 2 (vapor), 3 (liquid) ---
    gen = backend.flash_tp(Tgen, P_high, w1)
    m2 = m1 * gen.q_mass
    m3 = m1 - m2

    # Treat state 2/3 as phase states at generator conditions
    h2 = gen.hV_J_per_kg
    h3 = gen.hL_J_per_kg
    w2 = gen.wV
    w3 = gen.wL

    # Generator heat input (steady state, neglect KE/PE)
    Q_dot_gen = m2 * h2 + m3 * h3 - m1 * st1.h_J_per_kg  # W

    # --- solution heat exchanger (between 3 hot and 8 cold) ---
    # cold side (8 -> 1) heat gain:
    Q_dot_shx = m1 * (st1.h_J_per_kg - st8.h_J_per_kg)  # W
    # hot side (3 -> 9) enthalpy drop:
    h9 = h3 - Q_dot_shx / max(m3, 1e-12)
    T9 = solve_T_for_ph(backend, P_high, h9, w3, T_lo=Tabs, T_hi=Tgen)
    st9 = backend.state_tp(T9, P_high, w3)

    # valve 9 -> 10 (solution throttle): h10 = h9, P_low
    T10 = solve_T_for_ph(backend, P_low, h9, w3, T_lo=180.0, T_hi=T9)
    st10 = backend.state_tp(T10, P_low, w3)

    # SHX effectiveness (common absorption-cycle definition)
    # epsilon = (T1 - T8) / (T3 - T8)
    eps_shx = (cfg.T1_C - (st8.T_C)) / (cfg.T_gen_C - (st8.T_C)) if (cfg.T_gen_C - st8.T_C) != 0 else float("nan")

    # --- condenser: 2 -> 4 at P_high, Tcond ---
    st4 = backend.state_tp(Tcond, P_high, w2)
    Q_dot_cond = m2 * (h2 - st4.h_J_per_kg)  # W rejected

    # valve 4 -> 5 (refrigerant throttle): h5 = h4, P_low
    h5 = st4.h_J_per_kg
    T5 = solve_T_for_ph(backend, P_low, h5, w2, T_lo=180.0, T_hi=Tcond)
    st5_bulk = backend.flash_tp(T5, P_low, w2)  # for quality etc

    # --- evaporator: state 5 -> state 6 at P_low, Tevap_out ---
    evap_out = backend.flash_tp(Tevap_out, P_low, w2)
    h6 = evap_out.h_bulk
    Q_dot_evap = m2 * (h6 - h5)  # W

    # temperature required for full evaporation at P_low, w2 (q -> 1)
    T6_req = solve_T_for_q1(backend, P_low, w2, T_start=Tevap_out, T_max=500.0)

    # --- absorber energy balance to get Q_abs ---
    # absorber in: refrigerant stream (bulk) at 6, solution stream at 10; out: solution at 7
    Q_dot_abs = (m2 * h6 + m3 * st10.h_J_per_kg) - (m1 * st7.h_J_per_kg)  # W rejected

    # COP definitions
    COP_gen_only = Q_dot_evap / Q_dot_gen
    COP_with_pump = Q_dot_evap / (Q_dot_gen + W_dot_p)

    out = {
        "cfg": asdict(cfg),
        "backend": getattr(backend, "name", type(backend).__name__),
        "states": {
            "1": asdict(st1),
            "2_vapor": {"T_K": Tgen, "P_Pa": P_high, "w_nh3": w2, "h_J_per_kg": h2},
            "3_liquid": {"T_K": Tgen, "P_Pa": P_high, "w_nh3": w3, "h_J_per_kg": h3},
            "4": asdict(st4),
            "5": {
                "T_K": T5, "T_C": T5 - 273.15, "P_Pa": P_low, "w_nh3": w2,
                "h_J_per_kg": h5, "q_mass": st5_bulk.q_mass, "wL": st5_bulk.wL, "wV": st5_bulk.wV
            },
            "6": {
                "T_K": Tevap_out, "T_C": cfg.T_evap_out_C, "P_Pa": P_low, "w_nh3": w2,
                "h_J_per_kg": h6, "q_mass": evap_out.q_mass, "wL": evap_out.wL, "wV": evap_out.wV
            },
            "7": asdict(st7),
            "8": asdict(st8),
            "9": asdict(st9),
            "10": asdict(st10),
        },
        "mass_flows": {"m1": m1, "m2_refrig": m2, "m3_solution": m3},
        "compositions": {"w1": w1, "w2_refrig": w2, "w3_solution": w3},
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
    return out


# ------------------------------- CLI / main -----------------------------

def main() -> None:
    cfg = KN9B6Config()

    # backend selection: default to REFPROP if available
    backend_name = os.environ.get("NH3H2O_BACKEND", "refprop").lower()
    if backend_name in ("refprop", "rp"):
        backend = NH3H2ORefpropBackend()
    else:
        raise BackendUnavailableError(
            "NH3–H2O backend not implemented. Set NH3H2O_BACKEND=refprop and configure RPPREFIX."
        )

    print(f"[ammonia_cooling] backend={backend.name}")
    print("[ammonia_cooling] running K&N 9.B-6 with givens:")
    print(json.dumps(asdict(cfg), indent=2))

    res = compute_cycle_kn_9b6(cfg, backend)

    # pretty print key answers (a)-(d)
    m = res["metrics"]
    print("\n[ammonia_cooling] results:")
    print(f"  (a) T5 (into evaporator) = {m['T5_inlet_evap_C']:.3f} °C")
    print(f"  (b) ε_solution_HX        = {m['eps_solution_hx']:.4f}")
    print(f"  (c) T6 for q=1           = {m['T6_req_full_evap_C']:.3f} °C")
    print(f"  (d) COP (Qe/Qg)          = {m['COP_gen_only']:.4f}")
    print(f"      COP (incl pump)      = {m['COP_with_pump']:.4f}")

    # also write a JSON artifact next to script unless disabled
    out_path = os.environ.get("NH3H2O_OUT_JSON", "")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\n[ammonia_cooling] wrote: {out_path}")


if __name__ == "__main__":
    main()

__all__ = [
    "M_NH3",
    "M_H2O",
    "BackendUnavailableError",
    "ThermoEvalError",
    "State",
    "FlashTP",
    "NH3H2OBackend",
    "NH3H2ORefpropBackend",
    "KN9B6Config",
    "clamp",
    "massfrac_to_molefrac",
    "molefrac_to_massfrac",
    "molar_mass_from_molefrac",
    "solve_T_for_ph",
    "solve_T_for_q1",
    "compute_cycle_kn_9b6",
    "main",
]
