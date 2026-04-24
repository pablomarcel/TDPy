#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoolProp reference-state shifting sandbox (v2).

Fixes vs v1:
  - Anchor checks now evaluate *saturated liquid* thermodynamic_properties using (T,Q=0) or (P,Q=0)
    so we don't hit the "T,P exactly at saturation" ambiguity error.
  - Custom signature set_reference_state(Fluid, T0, rhomolar, hmolar0, smolar0) now
    uses the *bare fluid name* (no "HEOS::" prefix), which avoids:
        "key [HEOS::Fluid] was not found in string_to_index_map in JSONFluidLibrary"
  - Logs both "bare" and "backend-prefixed" query results (when backend is HEOS) so you
    can see whether the ref-state shift propagates to prefixed calls.

What you should see:
  - Built-in modes (IIR/ASHRAE/NBP/DEF/RESET) change absolute h/s.
  - Δh and Δs between two non-identical states remain invariant (up to floating error).
  - RESET removes the offset; DEF restores the library default for the fluid.
    (Per CoolProp docs table)

Docs:
  - Reference state options (incl. RESET meaning): https://coolprop.org/_static/doxygen/html/namespace_cool_prop.html
  - High-level ref-state notes: https://coolprop.org/coolprop/HighLevelAPI.html
  - Low-level warning (call at start): https://coolprop.org/coolprop/LowLevelAPI.html
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Tuple

import CoolProp.CoolProp as CP


# --------------------------- helpers ---------------------------------

def safe_props(output: str, i1: str, v1: float, i2: str, v2: float, fluid: str) -> float:
    try:
        return float(CP.PropsSI(output, i1, v1, i2, v2, fluid))
    except Exception as e:
        raise RuntimeError(f"PropsSI({output!r},{i1}={v1},{i2}={v2},fluid={fluid!r}) failed: {e}") from e


def state_TP(T: float, P: float, fluid: str) -> Dict[str, float]:
    return {
        "T": float(T),
        "P": float(P),
        "Dmolar": safe_props("Dmolar", "T", T, "P", P, fluid),
        "HMASS": safe_props("HMASS", "T", T, "P", P, fluid),
        "SMASS": safe_props("SMASS", "T", T, "P", P, fluid),
        "HMOLAR": safe_props("HMOLAR", "T", T, "P", P, fluid),
        "SMOLAR": safe_props("SMOLAR", "T", T, "P", P, fluid),
    }


def satliq_T(T: float, fluid: str) -> Dict[str, float]:
    """Saturated liquid at temperature T using (T,Q=0) for all property calls."""
    Q = 0.0
    P = safe_props("P", "T", T, "Q", Q, fluid)
    return {
        "T": float(T),
        "P": float(P),
        "Q": Q,
        "Dmolar": safe_props("Dmolar", "T", T, "Q", Q, fluid),
        "HMASS": safe_props("HMASS", "T", T, "Q", Q, fluid),
        "SMASS": safe_props("SMASS", "T", T, "Q", Q, fluid),
        "HMOLAR": safe_props("HMOLAR", "T", T, "Q", Q, fluid),
        "SMOLAR": safe_props("SMOLAR", "T", T, "Q", Q, fluid),
    }


def satliq_P(P: float, fluid: str) -> Dict[str, float]:
    """Saturated liquid at pressure P using (P,Q=0) for all property calls."""
    Q = 0.0
    T = safe_props("T", "P", P, "Q", Q, fluid)
    return {
        "T": float(T),
        "P": float(P),
        "Q": Q,
        "Dmolar": safe_props("Dmolar", "P", P, "Q", Q, fluid),
        "HMASS": safe_props("HMASS", "P", P, "Q", Q, fluid),
        "SMASS": safe_props("SMASS", "P", P, "Q", Q, fluid),
        "HMOLAR": safe_props("HMOLAR", "P", P, "Q", Q, fluid),
        "SMOLAR": safe_props("SMOLAR", "P", P, "Q", Q, fluid),
    }


def try_set_refstate(fluid: str, ref: Any) -> Optional[str]:
    try:
        if isinstance(ref, tuple):
            FluidName, T0, rhomolar, hmolar0, smolar0 = ref
            CP.set_reference_state(FluidName, T0, rhomolar, hmolar0, smolar0)
        else:
            CP.set_reference_state(fluid, str(ref))
        return None
    except Exception as e:
        return str(e)


# --------------------------- main ---------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CoolProp reference-state shifting sandbox (v2)")
    ap.add_argument("--fluid", default="R134a", help="Fluid name, e.g. R134a, n-Propane, Ammonia")
    ap.add_argument("--backend", default="HEOS", help="Backend, e.g. HEOS or REFPROP")
    ap.add_argument("--T", type=float, default=300.0, help="Target temperature [K]")
    ap.add_argument("--P", type=float, default=101325.0, help="Target pressure [Pa]")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    # Normalize names
    bare = args.fluid.split("::")[-1]
    fluid_full = f"{args.backend}::{bare}" if args.backend else bare

    # For HEOS, the "bare" name and "HEOS::name" should yield the same thermodynamic model.
    # We'll use bare names for setting reference states and for the custom signature,
    # and optionally query both to verify propagation.
    fluid_set = bare if args.backend.upper() == "HEOS" else fluid_full
    fluid_query_primary = bare if args.backend.upper() == "HEOS" else fluid_full
    fluid_query_alt = fluid_full if args.backend.upper() == "HEOS" else None

    # Reference points used by the built-in standards (per CoolProp docs)
    T_IIR = 273.15      # 0°C
    T_ASH = 233.15      # -40°C
    P_NBP = 1.0e5       # 1 bar

    # Targets per docs (mass basis: J/kg, J/kg/K)
    targets = {
        "IIR":    {"HMASS": 200e3, "SMASS": 1e3, "anchor": ("T", T_IIR)},
        "ASHRAE": {"HMASS": 0.0,   "SMASS": 0.0, "anchor": ("T", T_ASH)},
        "NBP":    {"HMASS": 0.0,   "SMASS": 0.0, "anchor": ("P", P_NBP)},
    }

    # Two comparison states (try to be single-phase for typical refrigerants)
    A = (300.0, 2.0e5)
    B = (320.0, 8.0e5)

    report: Dict[str, Any] = {
        "fluid_in": args.fluid,
        "backend": args.backend,
        "fluid_full": fluid_full,
        "fluid_set_used": fluid_set,
        "fluid_query_primary": fluid_query_primary,
        "fluid_query_alt": fluid_query_alt,
        "target_state": {"T": args.T, "P": args.P},
        "notes": [
            "Built-in ref states targets are on mass basis (J/kg, J/kg/K).",
            "Custom signature set_reference_state(T0,rhomolar,hmolar0,smolar0) uses molar basis (J/mol, J/mol/K).",
            "Anchor checks are evaluated using saturation inputs (T,Q=0) or (P,Q=0).",
            "RESET removes the offset; DEF restores the library default for the fluid (per CoolProp docs).",
        ],
        "baseline_DEF": {},
        "modes": [],
    }

    # Baseline: ensure DEF
    report["init_DEF_error"] = try_set_refstate(fluid_set, "DEF")

    base_target = state_TP(args.T, args.P, fluid_query_primary)
    base_A = state_TP(A[0], A[1], fluid_query_primary)
    base_B = state_TP(B[0], B[1], fluid_query_primary)
    base_delta = {
        "dHMASS": base_B["HMASS"] - base_A["HMASS"],
        "dSMASS": base_B["SMASS"] - base_A["SMASS"],
        "dHMOLAR": base_B["HMOLAR"] - base_A["HMOLAR"],
        "dSMOLAR": base_B["SMOLAR"] - base_A["SMOLAR"],
    }
    report["baseline_DEF"] = {"target": base_target, "A": base_A, "B": base_B, "delta_B_minus_A": base_delta}

    if fluid_query_alt:
        # Sanity check: bare vs prefixed should match for HEOS
        alt_target = state_TP(args.T, args.P, fluid_query_alt)
        report["baseline_bare_vs_prefixed_diff"] = {
            "HMASS": alt_target["HMASS"] - base_target["HMASS"],
            "SMASS": alt_target["SMASS"] - base_target["SMASS"],
            "HMOLAR": alt_target["HMOLAR"] - base_target["HMOLAR"],
            "SMOLAR": alt_target["SMOLAR"] - base_target["SMOLAR"],
        }

    # Built-in modes
    for opt in ["IIR", "ASHRAE", "NBP", "RESET", "DEF"]:
        mode: Dict[str, Any] = {"name": f"builtin:{opt}", "set_error": None}
        mode["set_error"] = try_set_refstate(fluid_set, opt)
        if mode["set_error"]:
            report["modes"].append(mode)
            continue

        st_target = state_TP(args.T, args.P, fluid_query_primary)
        st_A = state_TP(A[0], A[1], fluid_query_primary)
        st_B = state_TP(B[0], B[1], fluid_query_primary)
        delta = {
            "dHMASS": st_B["HMASS"] - st_A["HMASS"],
            "dSMASS": st_B["SMASS"] - st_A["SMASS"],
            "dHMOLAR": st_B["HMOLAR"] - st_A["HMOLAR"],
            "dSMOLAR": st_B["SMOLAR"] - st_A["SMOLAR"],
        }
        mode["target_state"] = st_target
        mode["delta_B_minus_A"] = delta
        mode["delta_diff_vs_DEF"] = {k: delta[k] - base_delta[k] for k in delta}

        # Anchor checks against documented targets
        if opt in targets:
            t = targets[opt]
            try:
                if t["anchor"][0] == "T":
                    anch = satliq_T(float(t["anchor"][1]), fluid_query_primary)
                else:
                    anch = satliq_P(float(t["anchor"][1]), fluid_query_primary)

                mode["anchor_check"] = {
                    "anchor": {"type": t["anchor"][0], "value": float(t["anchor"][1]), "Q": 0.0},
                    "state": anch,
                    "targets_mass_basis": {"HMASS": t["HMASS"], "SMASS": t["SMASS"]},
                    "errors_mass_basis": {"dHMASS": anch["HMASS"] - t["HMASS"], "dSMASS": anch["SMASS"] - t["SMASS"]},
                }
            except Exception as e:
                mode["anchor_check_error"] = str(e)

        report["modes"].append(mode)

    # Custom mode: HMOLAR=0, SMOLAR=0 at (T0,P0)
    T0 = 300.0
    P0 = 101325.0
    custom: Dict[str, Any] = {
        "name": "custom:set_reference_state(T0,Dmolar0,HMOLAR=0,SMOLAR=0)",
        "T0": T0,
        "P0": P0,
        "set_error": None,
    }

    try:
        Dmolar0 = safe_props("Dmolar", "T", T0, "P", P0, fluid_query_primary)
        custom["Dmolar0"] = Dmolar0

        # IMPORTANT: use the *set* fluid key (bare name for HEOS)
        custom_tuple = (fluid_set, T0, Dmolar0, 0.0, 0.0)
        custom["set_error"] = try_set_refstate(fluid_set, custom_tuple)

        if not custom["set_error"]:
            at0_primary = state_TP(T0, P0, fluid_query_primary)
            custom["check_at_T0P0_primary_query"] = at0_primary
            custom["check_errors_molar_primary_query"] = {"dHMOLAR": at0_primary["HMOLAR"], "dSMOLAR": at0_primary["SMOLAR"]}

            if fluid_query_alt:
                at0_alt = state_TP(T0, P0, fluid_query_alt)
                custom["check_at_T0P0_alt_query"] = at0_alt
                custom["check_errors_molar_alt_query"] = {"dHMOLAR": at0_alt["HMOLAR"], "dSMOLAR": at0_alt["SMOLAR"]}

            # Also show effect on target and on deltas
            st_target = state_TP(args.T, args.P, fluid_query_primary)
            st_A = state_TP(A[0], A[1], fluid_query_primary)
            st_B = state_TP(B[0], B[1], fluid_query_primary)
            delta = {
                "dHMASS": st_B["HMASS"] - st_A["HMASS"],
                "dSMASS": st_B["SMASS"] - st_A["SMASS"],
                "dHMOLAR": st_B["HMOLAR"] - st_A["HMOLAR"],
                "dSMOLAR": st_B["SMOLAR"] - st_A["SMOLAR"],
            }
            custom["target_state"] = st_target
            custom["delta_B_minus_A"] = delta
            custom["delta_diff_vs_DEF"] = {k: delta[k] - base_delta[k] for k in delta}

    except Exception as e:
        custom["custom_setup_error"] = str(e)

    report["modes"].append(custom)

    # Restore DEF
    report["restore_DEF_error"] = try_set_refstate(fluid_set, "DEF")

    out_json = json.dumps(report, indent=2, sort_keys=False)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(out_json, encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
