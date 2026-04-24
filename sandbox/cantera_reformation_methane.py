#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Klein & Nellis Example 14.3-1 — Reformation of Methane (Cantera NASA thermo)

This script does **not** use CoolProp.
Instead it uses Cantera's NASA polynomial thermo to compute standard chemical
potentials and equilibrium constants, then solves the two mass-action equations
for the two reaction coordinates (eps1, eps2).

Goal: reproduce the EES workflow:
  1) compute g_bar_o[i] = h_i(T) - T*s_i(T,P_ref)
  2) compute ΔG° for each reaction
  3) compute K from ΔG°
  4) enforce mass action using reaction coordinates

It also optionally runs Cantera's built-in Gibbs minimization on the restricted
species set as a cross-check.

Run:
  runroot python sandbox/cantera_reformation_methane.py --T 1100 --Pbar 10 --mech gri30.yaml

Notes:
- Species set is restricted to: CH4, H2O, CO, H2, CO2.
- Pressure standard state is Cantera's reference pressure (typically 1 atm).
- Units:
    Cantera uses kmol-based SI: ct.gas_constant is J/(kmol*K)
    mu0 returned by Cantera is J/kmol.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import cantera as ct
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Cantera is required for this sandbox. Install with: pip install cantera\n"
        f"Import error: {e}"
    )

try:
    from scipy.optimize import root
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "SciPy is required for this sandbox. Install with: pip install scipy\n"
        f"Import error: {e}"
    )


WANTED = ["CH4", "H2O", "CO", "H2", "CO2"]


@dataclass
class Case:
    T: float
    P: float
    P_ref: float
    n0: Dict[str, float]


def _sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _smooth_max(a: float, b: float, delta: float) -> float:
    return 0.5 * (a + b + math.sqrt((a - b) ** 2 + delta))


def _smooth_min(a: float, b: float, delta: float) -> float:
    return 0.5 * (a + b - math.sqrt((a - b) ** 2 + delta))


def _safe_ln(x: float, eps: float) -> float:
    return math.log(x + eps)


def build_restricted_gas(mech: str) -> ct.Solution:
    """Create an IdealGas phase with only the 5 desired species.

    We *only* need thermo for equilibrium constants. Reactions are not required.
    """
    # Load all species from mechanism file and keep only WANTED
    all_species = ct.Species.list_from_file(mech)
    wanted_species = [sp for sp in all_species if sp.name in set(WANTED)]

    if len(wanted_species) != len(WANTED):
        missing = sorted(set(WANTED) - {sp.name for sp in wanted_species})
        raise ValueError(f"Mechanism '{mech}' is missing species: {missing}")

    # Try a few constructor signatures across Cantera versions
    last_err = None
    for kwargs in (
        dict(thermo="IdealGas", kinetics="GasKinetics", species=wanted_species, reactions=[]),
        dict(thermo="IdealGas", kinetics="GasKinetics", species=wanted_species),
        dict(thermo="IdealGas", species=wanted_species),
    ):
        try:
            return ct.Solution(**kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not build restricted IdealGas phase: {last_err}")


def get_ref_pressure(gas: ct.Solution) -> float:
    """Best-effort retrieval of Cantera's reference pressure (Pa)."""
    for attr in ("ref_pressure", "reference_pressure", "standard_state_pressure"):
        if hasattr(gas, attr):
            try:
                val = getattr(gas, attr)
                # some Cantera versions expose it as a method
                if callable(val):
                    val = val()
                return float(val)
            except Exception:
                pass
    return float(ct.one_atm)


def standard_mu0(gas: ct.Solution, T: float, P_ref: float) -> Dict[str, float]:
    """Return standard-state chemical potentials μ°_i(T, P_ref) for each wanted species."""
    gas.TP = T, P_ref

    # Prefer standard_chem_potentials if available
    if hasattr(gas, "standard_chem_potentials"):
        mu = np.array(gas.standard_chem_potentials, dtype=float)
    elif hasattr(gas, "standard_gibbs_RT"):
        mu = np.array(gas.standard_gibbs_RT, dtype=float) * ct.gas_constant * T
    else:
        raise AttributeError("Cantera Solution missing standard chemical potential accessors")

    # Map to names
    return {nm: float(mu[gas.species_index(nm)]) for nm in WANTED}


def equilibrium_constants(mu0: Dict[str, float], T: float) -> Tuple[float, float]:
    """Compute ln K1, ln K2 from ΔG° = Σ ν_i μ°_i.

    Reaction 1: CH4 + H2O <-> CO + 3 H2
    Reaction 2: CO2 + H2 <-> CO + H2O
    """
    # stoichiometric coefficients ν_i,j
    nu1 = {"CH4": -1.0, "H2O": -1.0, "CO": 1.0, "H2": 3.0, "CO2": 0.0}
    nu2 = {"CH4": 0.0, "H2O": 1.0, "CO": 1.0, "H2": -1.0, "CO2": -1.0}

    dG1 = sum(nu1[k] * mu0[k] for k in WANTED)
    dG2 = sum(nu2[k] * mu0[k] for k in WANTED)

    lnK1 = -dG1 / (ct.gas_constant * T)
    lnK2 = -dG2 / (ct.gas_constant * T)
    return float(lnK1), float(lnK2)


def solve_reaction_coordinates(case: Case, lnK1: float, lnK2: float) -> Dict[str, float]:
    """Solve for eps1, eps2 using bounded mapping to keep moles nonnegative."""

    # Bounds derived from nonnegativity for this initial mixture:
    # n_CH4 = 2 - eps1
    # n_H2O = 3 - eps1 + eps2
    # n_CO  = eps1 + eps2
    # n_H2  = 3 eps1 - eps2
    # n_CO2 = 1 - eps2
    # eps1 in [0,2]
    # eps2 in [max(-eps1, eps1-3), min(1, 3 eps1)]

    delta = 1e-12
    eps_ln = 1e-300  # for ln() safety

    pr = case.P / case.P_ref
    lnpr = math.log(pr)

    def unpack(z: np.ndarray) -> Tuple[float, float]:
        z1, z2 = float(z[0]), float(z[1])
        s1 = _sigmoid(z1)
        eps1 = 2.0 * s1

        L = _smooth_max(-eps1, eps1 - 3.0, delta)
        U = _smooth_min(1.0, 3.0 * eps1, delta)
        w = _smooth_max(U - L, 0.0, delta)

        s2 = _sigmoid(z2)
        eps2 = L + w * s2
        return eps1, eps2

    def residual(z: np.ndarray) -> np.ndarray:
        eps1, eps2 = unpack(z)

        n_CH4 = case.n0["CH4"] - eps1
        n_H2O = case.n0["H2O"] - eps1 + eps2
        n_CO = case.n0["CO"] + eps1 + eps2
        n_H2 = case.n0["H2"] + 3.0 * eps1 - eps2
        n_CO2 = case.n0["CO2"] - eps2

        n_tot = n_CH4 + n_H2O + n_CO + n_H2 + n_CO2

        # log form of mass action, written in n-space (avoids y-variables going negative)
        lnK1_calc = (
            _safe_ln(n_CO, eps_ln)
            + 3.0 * _safe_ln(n_H2, eps_ln)
            - _safe_ln(n_CH4, eps_ln)
            - _safe_ln(n_H2O, eps_ln)
            - 2.0 * _safe_ln(n_tot, eps_ln)
            + 2.0 * lnpr
        )
        lnK2_calc = (
            _safe_ln(n_CO, eps_ln)
            + _safe_ln(n_H2O, eps_ln)
            - _safe_ln(n_H2, eps_ln)
            - _safe_ln(n_CO2, eps_ln)
        )

        return np.array([lnK1_calc - lnK1, lnK2_calc - lnK2], dtype=float)

    # Starting guess: same neighborhood we used in tdpy
    z0 = np.array([7.6, -1.04], dtype=float)

    sol = root(residual, z0, method="hybr")
    if not sol.success:
        raise RuntimeError(f"SciPy root failed: {sol.message}")

    eps1, eps2 = unpack(sol.x)

    # compute final moles and mole fractions
    n_CH4 = case.n0["CH4"] - eps1
    n_H2O = case.n0["H2O"] - eps1 + eps2
    n_CO = case.n0["CO"] + eps1 + eps2
    n_H2 = case.n0["H2"] + 3.0 * eps1 - eps2
    n_CO2 = case.n0["CO2"] - eps2

    n_tot = n_CH4 + n_H2O + n_CO + n_H2 + n_CO2

    y = {
        "CH4": n_CH4 / n_tot,
        "H2O": n_H2O / n_tot,
        "CO": n_CO / n_tot,
        "H2": n_H2 / n_tot,
        "CO2": n_CO2 / n_tot,
    }

    out = {
        "eps1": float(eps1),
        "eps2": float(eps2),
        "n_CH4": float(n_CH4),
        "n_H2O": float(n_H2O),
        "n_CO": float(n_CO),
        "n_H2": float(n_H2),
        "n_CO2": float(n_CO2),
        "n_tot": float(n_tot),
        "y_CH4": float(y["CH4"]),
        "y_H2O": float(y["H2O"]),
        "y_CO": float(y["CO"]),
        "y_H2": float(y["H2"]),
        "y_CO2": float(y["CO2"]),
        "z1": float(sol.x[0]),
        "z2": float(sol.x[1]),
        "nfev": int(sol.nfev),
    }
    return out


def cantera_gibbs_equilibrium(gas: ct.Solution, case: Case) -> Dict[str, float] | None:
    """Optional cross-check: Gibbs minimization equilibrium on restricted species set."""
    try:
        gas.TP = case.T, case.P
        # Set initial composition by mole numbers (normalized internally)
        tot = sum(case.n0.values())
        x0 = {k: v / tot for k, v in case.n0.items()}
        gas.X = x0
        gas.equilibrate("TP", solver="gibbs")
        X = {nm: float(gas.X[gas.species_index(nm)]) for nm in WANTED}
        return {
            "y_CH4": X["CH4"],
            "y_H2O": X["H2O"],
            "y_CO": X["CO"],
            "y_H2": X["H2"],
            "y_CO2": X["CO2"],
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1100.0)
    ap.add_argument("--Pbar", type=float, default=10.0)
    ap.add_argument("--mech", type=str, default="gri30.yaml")
    ap.add_argument("--book_yH2", type=float, default=0.4899, help="Klein & Nellis reported y_H2 for comparison")
    args = ap.parse_args()

    T = float(args.T)
    P = float(args.Pbar) * 1e5

    gas = build_restricted_gas(args.mech)
    P_ref = get_ref_pressure(gas)

    case = Case(
        T=T,
        P=P,
        P_ref=P_ref,
        n0={"CH4": 2.0, "H2O": 3.0, "CO": 0.0, "H2": 0.0, "CO2": 1.0},
    )

    print("=== Klein & Nellis Example 14.3-1 — Cantera NASA thermo + mass action ===")
    print(f"T = {case.T:g} K")
    print(f"P = {case.P:g} Pa ({case.P/1e5:g} bar)")
    print(f"Mechanism file: {args.mech}")
    print(f"Reference pressure (standard state) = {case.P_ref:.6g} Pa")

    mu0 = standard_mu0(gas, case.T, case.P_ref)
    lnK1, lnK2 = equilibrium_constants(mu0, case.T)
    print(f"lnK1 = {lnK1:.6f}")
    print(f"lnK2 = {lnK2:.6f}")

    sol = solve_reaction_coordinates(case, lnK1, lnK2)

    print("\n--- Reaction-coordinate solution (matches EES math, using Cantera thermo) ---")
    print(f"eps1 = {sol['eps1']:.9f} kmol")
    print(f"eps2 = {sol['eps2']:.9f} kmol")
    print(f"n_tot = {sol['n_tot']:.9f} kmol")
    print("Mole fractions:")
    print(f"  y_CH4 = {sol['y_CH4']:.6f}")
    print(f"  y_H2O = {sol['y_H2O']:.6f}")
    print(f"  y_CO  = {sol['y_CO']:.6f}")
    print(f"  y_H2  = {sol['y_H2']:.6f}")
    print(f"  y_CO2 = {sol['y_CO2']:.6f}")

    y_book = float(args.book_yH2)
    y_h2 = sol["y_H2"]
    abs_err = y_h2 - y_book
    rel_err = abs_err / y_book
    print("\nComparison vs Klein & Nellis (book):")
    print(f"  book y_H2 = {y_book:.6f}")
    print(f"  this  y_H2 = {y_h2:.6f}")
    print(f"  error = {abs_err:+.6f} ({rel_err:+.2%})")

    # Optional Cantera Gibbs minimization on restricted species
    eq = cantera_gibbs_equilibrium(gas, case)
    if eq is not None:
        print("\n--- Cantera equilibrate('TP', solver='gibbs') on same restricted species ---")
        print(f"  y_CH4 = {eq['y_CH4']:.6f}")
        print(f"  y_H2O = {eq['y_H2O']:.6f}")
        print(f"  y_CO  = {eq['y_CO']:.6f}")
        print(f"  y_H2  = {eq['y_H2']:.6f}")
        print(f"  y_CO2 = {eq['y_CO2']:.6f}")


if __name__ == "__main__":
    main()
