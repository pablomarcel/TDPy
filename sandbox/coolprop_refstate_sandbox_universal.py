#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoolProp reference-state shifting sandbox (universal, CLI-driven).

Goals
-----
- No hardcoded "anchor" values: everything is configurable via CLI flags.
- Demonstrate (and log) that:
    1) Changing reference state shifts *absolute* h/s values
    2) Δh and Δs between two non-identical states remain invariant (up to float noise)
- Validate anchor points for built-in reference states using sat-liquid inputs (T,Q=0) or (P,Q=0)
  to avoid the "T,P exactly at saturation" ambiguity.

CoolProp notes (important)
--------------------------
- Built-in reference states are defined on a *mass basis* (J/kg, J/kg/K).
- Custom signature CP.set_reference_state(FluidName, T0, rhomolar0, hmolar0, smolar0)
  uses a *molar basis* (J/mol, J/mol/K).
- Reference state shifts are global (per-fluid) inside the CoolProp process. Treat as initialization:
  set once, do your work, restore.

Usage examples
--------------
# Baseline run (defaults are reasonable and overrideable)
runroot python sandbox/coolprop_refstate_sandbox_universal.py --fluid R134a

# Check NBP at both 1 bar and 1 atm in one run:
runroot python sandbox/coolprop_refstate_sandbox_universal.py --fluid R134a --nbp-P 100000 --nbp-P 101325

# Change IIR anchor temp (and targets) if you want to experiment:
runroot python sandbox/coolprop_refstate_sandbox_universal.py --fluid n-Propane --iir-T 273.15 --iir-hmass 200000 --iir-smass 1000

# Custom reference: set HMOLAR=0, SMOLAR=0 at T0,P0
runroot python sandbox/coolprop_refstate_sandbox_universal.py --fluid Ammonia --enable-custom --custom-T0 300 --custom-P0 101325 --custom-hmolar0 0 --custom-smolar0 0

# Custom reference specified on a mass basis (convert internally using M):
runroot python sandbox/coolprop_refstate_sandbox_universal.py --fluid R134a --enable-custom --custom-T0 300 --custom-P0 101325 --custom-hmass0 0 --custom-smass0 0

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import CoolProp.CoolProp as CP


# --------------------------- helpers ---------------------------------

def _float_list(values: Optional[Sequence[float]]) -> List[float]:
    return list(values) if values else []


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
    """Saturated liquid at temperature T using (T,Q=0)."""
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
    """Saturated liquid at pressure P using (P,Q=0)."""
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
    """
    - If ref is a string: CP.set_reference_state(fluid, ref)
    - If ref is a tuple: CP.set_reference_state(FluidName, T0, rhomolar0, hmolar0, smolar0)
    """
    try:
        if isinstance(ref, tuple):
            FluidName, T0, rhomolar, hmolar0, smolar0 = ref
            CP.set_reference_state(FluidName, float(T0), float(rhomolar), float(hmolar0), float(smolar0))
        else:
            CP.set_reference_state(fluid, str(ref))
        return None
    except Exception as e:
        return str(e)


def deltas(A: Dict[str, float], B: Dict[str, float]) -> Dict[str, float]:
    return {
        "dHMASS": B["HMASS"] - A["HMASS"],
        "dSMASS": B["SMASS"] - A["SMASS"],
        "dHMOLAR": B["HMOLAR"] - A["HMOLAR"],
        "dSMOLAR": B["SMOLAR"] - A["SMOLAR"],
    }


@dataclass(frozen=True)
class AnchorSpec:
    # kind = "T" or "P", value in SI units
    kind: str
    value: float
    # targets on mass basis
    hmass_target: float
    smass_target: float


def check_anchor(spec: AnchorSpec, fluid: str) -> Dict[str, Any]:
    if spec.kind.upper() == "T":
        st = satliq_T(spec.value, fluid)
        anchor = {"type": "T", "value": float(spec.value), "Q": 0.0}
    elif spec.kind.upper() == "P":
        st = satliq_P(spec.value, fluid)
        anchor = {"type": "P", "value": float(spec.value), "Q": 0.0}
    else:
        raise ValueError(f"Unknown anchor kind: {spec.kind!r} (expected 'T' or 'P')")

    return {
        "anchor": anchor,
        "state": st,
        "targets_mass_basis": {"HMASS": spec.hmass_target, "SMASS": spec.smass_target},
        "errors_mass_basis": {"dHMASS": st["HMASS"] - spec.hmass_target, "dSMASS": st["SMASS"] - spec.smass_target},
    }


# --------------------------- main ---------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CoolProp reference-state shifting sandbox (universal, CLI-driven)")

    # Fluid selection
    ap.add_argument("--fluid", default="R134a", help="Fluid name (bare or prefixed), e.g. R134a, n-Propane, HEOS::Ammonia")
    ap.add_argument("--backend", default="HEOS", help="Backend prefix for alt querying, e.g. HEOS (REFPROP optional)")
    ap.add_argument("--no-alt-query", action="store_true", help="Do not query the backend-prefixed name (e.g. HEOS::Fluid)")

    # Target state (where you want to observe absolute values)
    ap.add_argument("--T", type=float, default=300.0, help="Target temperature [K]")
    ap.add_argument("--P", type=float, default=101325.0, help="Target pressure [Pa]")

    # Comparison states for invariance proof (ΔB-ΔA)
    ap.add_argument("--A-T", type=float, default=300.0, help="State A temperature [K]")
    ap.add_argument("--A-P", type=float, default=2.0e5, help="State A pressure [Pa]")
    ap.add_argument("--B-T", type=float, default=320.0, help="State B temperature [K]")
    ap.add_argument("--B-P", type=float, default=8.0e5, help="State B pressure [Pa]")

    # Which modes to run
    ap.add_argument(
        "--modes",
        default="IIR,ASHRAE,NBP,RESET,DEF,CUSTOM",
        help="Comma-separated modes. Options: IIR,ASHRAE,NBP,RESET,DEF,CUSTOM",
    )

    # Built-in anchor specifications (all configurable)
    ap.add_argument("--iir-T", type=float, default=273.15, help="IIR anchor temperature [K] (sat liquid)")
    ap.add_argument("--iir-hmass", type=float, default=200e3, help="IIR target HMASS at anchor [J/kg]")
    ap.add_argument("--iir-smass", type=float, default=1e3, help="IIR target SMASS at anchor [J/kg/K]")

    ap.add_argument("--ashrae-T", type=float, default=233.15, help="ASHRAE anchor temperature [K] (sat liquid)")
    ap.add_argument("--ashrae-hmass", type=float, default=0.0, help="ASHRAE target HMASS at anchor [J/kg]")
    ap.add_argument("--ashrae-smass", type=float, default=0.0, help="ASHRAE target SMASS at anchor [J/kg/K]")

    # NBP: allow multiple pressures to test both 1 bar and 1 atm (or anything else)
    ap.add_argument(
        "--nbp-P",
        type=float,
        action="append",
        default=[],
        help="NBP anchor pressure [Pa] (sat liquid). Repeat to test multiple pressures. Default if omitted: 101325.",
    )
    ap.add_argument("--nbp-hmass", type=float, default=0.0, help="NBP target HMASS at anchor [J/kg]")
    ap.add_argument("--nbp-smass", type=float, default=0.0, help="NBP target SMASS at anchor [J/kg/K]")

    # Custom reference state mode
    ap.add_argument("--enable-custom", action="store_true", help="Enable custom reference-state mode (also requires CUSTOM in --modes)")
    ap.add_argument("--custom-T0", type=float, default=300.0, help="Custom anchor temperature T0 [K]")
    ap.add_argument("--custom-P0", type=float, default=101325.0, help="Custom anchor pressure P0 [Pa]")
    ap.add_argument("--custom-Dmolar0", type=float, default=float("nan"), help="Custom anchor Dmolar0 [mol/m^3]. If NaN, compute from T0,P0.")
    ap.add_argument("--custom-hmolar0", type=float, default=float("nan"), help="Custom HMOLAR0 [J/mol]. If NaN and HMASS0 provided, computed from MW.")
    ap.add_argument("--custom-smolar0", type=float, default=float("nan"), help="Custom SMOLAR0 [J/mol/K]. If NaN and SMASS0 provided, computed from MW.")
    ap.add_argument("--custom-hmass0", type=float, default=float("nan"), help="Optional: specify custom HMASS0 [J/kg] (converted to molar if hmolar0 NaN)")
    ap.add_argument("--custom-smass0", type=float, default=float("nan"), help="Optional: specify custom SMASS0 [J/kg/K] (converted to molar if smolar0 NaN)")

    # Output
    ap.add_argument("--out", default="", help="Optional output JSON path")
    ap.add_argument("--sort-keys", action="store_true", help="Sort JSON keys (nicer diffs, slightly less human-friendly)")

    # Restore behavior
    ap.add_argument("--no-restore-def", action="store_true", help="Do not restore DEF at the end (debug only)")

    args = ap.parse_args()

    # Normalize names
    bare = args.fluid.split("::")[-1]
    fluid_full = f"{args.backend}::{bare}" if args.backend else bare

    backend_upper = (args.backend or "").upper()
    # For HEOS, use bare for setting the ref state + primary queries; optionally query prefixed too.
    fluid_set = bare if backend_upper == "HEOS" else fluid_full
    fluid_query_primary = bare if backend_upper == "HEOS" else fluid_full
    fluid_query_alt = None if (args.no_alt_query or backend_upper != "HEOS") else fluid_full

    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    if "CUSTOM" in modes and not args.enable_custom:
        # Keep it explicit: user must opt-in because custom mode changes global reference state.
        # We'll still log it as "skipped".
        pass

    # Default NBP anchor pressures if user didn't provide any
    nbp_pressures = _float_list(args.nbp_P) or [101325.0]

    # Anchor specs table (configurable)
    anchor_specs: Dict[str, List[AnchorSpec]] = {
        "IIR": [AnchorSpec("T", float(args.iir_T), float(args.iir_hmass), float(args.iir_smass))],
        "ASHRAE": [AnchorSpec("T", float(args.ashrae_T), float(args.ashrae_hmass), float(args.ashrae_smass))],
        "NBP": [AnchorSpec("P", float(p), float(args.nbp_hmass), float(args.nbp_smass)) for p in nbp_pressures],
    }

    # Build report
    report: Dict[str, Any] = {
        "fluid_in": args.fluid,
        "backend": args.backend,
        "fluid_full": fluid_full,
        "fluid_set_used": fluid_set,
        "fluid_query_primary": fluid_query_primary,
        "fluid_query_alt": fluid_query_alt,
        "target_state": {"T": args.T, "P": args.P},
        "compare_states": {"A": {"T": args.A_T, "P": args.A_P}, "B": {"T": args.B_T, "P": args.B_P}},
        "modes_requested": modes,
        "notes": [
            "Built-in ref states targets are on mass basis (J/kg, J/kg/K).",
            "Custom signature set_reference_state(T0,rhomolar0,hmolar0,smolar0) uses molar basis (J/mol, J/mol/K).",
            "Anchor checks are evaluated using saturation inputs (T,Q=0) or (P,Q=0).",
            "Reference state shifts are global (per-fluid) inside the process; treat as initialization and restore DEF afterwards.",
        ],
        "baseline_DEF": {},
        "modes": [],
    }

    # Baseline: ensure DEF
    report["init_DEF_error"] = try_set_refstate(fluid_set, "DEF")

    # Baseline states + invariance deltas
    base_target = state_TP(args.T, args.P, fluid_query_primary)
    base_A = state_TP(args.A_T, args.A_P, fluid_query_primary)
    base_B = state_TP(args.B_T, args.B_P, fluid_query_primary)
    base_delta = deltas(base_A, base_B)
    report["baseline_DEF"] = {"target": base_target, "A": base_A, "B": base_B, "delta_B_minus_A": base_delta}

    if fluid_query_alt:
        alt_target = state_TP(args.T, args.P, fluid_query_alt)
        report["baseline_bare_vs_prefixed_diff"] = {
            "HMASS": alt_target["HMASS"] - base_target["HMASS"],
            "SMASS": alt_target["SMASS"] - base_target["SMASS"],
            "HMOLAR": alt_target["HMOLAR"] - base_target["HMOLAR"],
            "SMOLAR": alt_target["SMOLAR"] - base_target["SMOLAR"],
        }

    # Helper to run a built-in refstate option
    def run_builtin(opt: str) -> Dict[str, Any]:
        mode: Dict[str, Any] = {"name": f"builtin:{opt}", "set_error": None}
        mode["set_error"] = try_set_refstate(fluid_set, opt)
        if mode["set_error"]:
            return mode

        st_target = state_TP(args.T, args.P, fluid_query_primary)
        st_A = state_TP(args.A_T, args.A_P, fluid_query_primary)
        st_B = state_TP(args.B_T, args.B_P, fluid_query_primary)
        d = deltas(st_A, st_B)

        mode["target_state"] = st_target
        mode["delta_B_minus_A"] = d
        mode["delta_diff_vs_DEF"] = {k: d[k] - base_delta[k] for k in d}

        # Anchor checks (possibly multiple anchors per option; e.g., NBP at 1 bar and 1 atm)
        if opt in anchor_specs:
            checks = []
            for spec in anchor_specs[opt]:
                try:
                    checks.append(check_anchor(spec, fluid_query_primary))
                except Exception as e:
                    checks.append({"error": str(e), "spec": {"kind": spec.kind, "value": spec.value}})
            mode["anchor_checks"] = checks

        return mode

    # Run built-ins in requested order
    for opt in ["IIR", "ASHRAE", "NBP", "RESET", "DEF"]:
        if opt in modes:
            report["modes"].append(run_builtin(opt))

    # Custom mode (optional)
    if "CUSTOM" in modes:
        custom: Dict[str, Any] = {
            "name": "custom:set_reference_state(Fluid,T0,Dmolar0,HMOLAR0,SMOLAR0)",
            "enabled": bool(args.enable_custom),
            "T0": float(args.custom_T0),
            "P0": float(args.custom_P0),
            "set_error": None,
        }

        if not args.enable_custom:
            custom["skipped_reason"] = "CUSTOM requested, but --enable-custom not provided."
            report["modes"].append(custom)
        else:
            try:
                # Compute MW (kg/mol) to support mass->molar conversion if user provides HMASS0/SMASS0.
                MW = safe_props("M", "T", float(args.custom_T0), "P", float(args.custom_P0), fluid_query_primary)  # kg/mol
                custom["MW_at_T0P0_kg_per_mol"] = MW

                # Dmolar0
                if args.custom_Dmolar0 == args.custom_Dmolar0:  # not NaN
                    Dmolar0 = float(args.custom_Dmolar0)
                else:
                    Dmolar0 = safe_props("Dmolar", "T", float(args.custom_T0), "P", float(args.custom_P0), fluid_query_primary)
                custom["Dmolar0"] = Dmolar0

                # HMOLAR0, SMOLAR0
                hmolar0 = float(args.custom_hmolar0) if (args.custom_hmolar0 == args.custom_hmolar0) else float("nan")
                smolar0 = float(args.custom_smolar0) if (args.custom_smolar0 == args.custom_smolar0) else float("nan")

                if hmolar0 != hmolar0:  # NaN
                    if args.custom_hmass0 == args.custom_hmass0:  # provided
                        hmolar0 = float(args.custom_hmass0) * MW
                if smolar0 != smolar0:  # NaN
                    if args.custom_smass0 == args.custom_smass0:  # provided
                        smolar0 = float(args.custom_smass0) * MW

                # Default to 0 if still NaN (matches your typical use case)
                if hmolar0 != hmolar0:
                    hmolar0 = 0.0
                if smolar0 != smolar0:
                    smolar0 = 0.0

                custom["hmolar0_J_per_mol"] = hmolar0
                custom["smolar0_J_per_molK"] = smolar0

                # IMPORTANT: custom signature expects the "set" fluid key (bare for HEOS in our convention)
                custom_tuple = (fluid_set, float(args.custom_T0), float(Dmolar0), float(hmolar0), float(smolar0))
                custom["set_error"] = try_set_refstate(fluid_set, custom_tuple)

                if not custom["set_error"]:
                    # Check at (T0,P0): should be ~hmolar0, smolar0; if we set to 0, should be ~0.
                    at0_primary = state_TP(float(args.custom_T0), float(args.custom_P0), fluid_query_primary)
                    custom["check_at_T0P0_primary_query"] = at0_primary
                    custom["check_errors_molar_primary_query"] = {
                        "dHMOLAR": at0_primary["HMOLAR"] - hmolar0,
                        "dSMOLAR": at0_primary["SMOLAR"] - smolar0,
                    }

                    if fluid_query_alt:
                        at0_alt = state_TP(float(args.custom_T0), float(args.custom_P0), fluid_query_alt)
                        custom["check_at_T0P0_alt_query"] = at0_alt
                        custom["check_errors_molar_alt_query"] = {
                            "dHMOLAR": at0_alt["HMOLAR"] - hmolar0,
                            "dSMOLAR": at0_alt["SMOLAR"] - smolar0,
                        }

                    # Show effect at target and on invariance deltas
                    st_target = state_TP(args.T, args.P, fluid_query_primary)
                    st_A = state_TP(args.A_T, args.A_P, fluid_query_primary)
                    st_B = state_TP(args.B_T, args.B_P, fluid_query_primary)
                    d = deltas(st_A, st_B)

                    custom["target_state"] = st_target
                    custom["delta_B_minus_A"] = d
                    custom["delta_diff_vs_DEF"] = {k: d[k] - base_delta[k] for k in d}

            except Exception as e:
                custom["custom_setup_error"] = str(e)

            report["modes"].append(custom)

    # Restore DEF
    if args.no_restore_def:
        report["restore_DEF_skipped"] = True
    else:
        report["restore_DEF_error"] = try_set_refstate(fluid_set, "DEF")

    out_json = json.dumps(report, indent=2, sort_keys=bool(args.sort_keys))
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
