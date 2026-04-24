#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
nh3h2o/verify.py

Verify ammonia-water (Ibrahim & Klein) implementation against the EES NH3H2O
documentation example.

EES example:
  T = 10 C
  P = 1000 kPa
  X = 0.5   (NH3 mass fraction)

Expected:
  h = -194.8 kJ/kg
  Q = -0.001   (subcooled flag in EES)

Purpose:
- quick first-gate sanity check after refactors
- confirms imports, units, enthalpy reference handling, and q/phase flag wiring
"""


def main() -> None:
    # --- EES doc inputs ---
    T_K = 10.0 + 273.15
    P_Pa = 1000.0e3
    X = 0.5  # NH3 mass fraction (EES definition)

    # --- import your implementation ---
    # Prefer package-style import, then fall back to local-script usage.
    try:
        from nh3h2o.ammonia_water import props_tpx
    except Exception:
        from ammonia_water import props_tpx  # type: ignore

    out = props_tpx(T_K=T_K, P_Pa=P_Pa, X=X)  # should return dict-like

    h_Jkg = out["h_J_per_kg"]
    q = out["q"]  # your quality / phase flag
    v_m3kg = out.get("v_m3_per_kg", None)
    rho = out.get("rho_kg_per_m3", None)

    h_kJkg = h_Jkg / 1000.0

    print("=== EES doc example check ===")
    print(f"Inputs: T={T_K:.2f} K, P={P_Pa:.3e} Pa, X={X:.6f} (NH3 mass fraction)")
    print(f"Computed: h={h_kJkg:.4f} kJ/kg, q={q}")
    if v_m3kg is not None:
        print(f"Computed: v={v_m3kg:.6e} m^3/kg")
    if rho is not None:
        print(f"Computed: rho={rho:.6f} kg/m^3")

    # --- compare to EES expected ---
    h_expected = -194.8
    q_expected = -0.001

    dh = h_kJkg - h_expected
    print("\nExpected (EES):")
    print(f"  h={h_expected:.4f} kJ/kg")
    print(f"  q={q_expected}")
    print("\nErrors:")
    print(f"  Δh = {dh:+.4f} kJ/kg")
    print(f"  q_match = {q == q_expected}")

    # crude pass/fail threshold for first gate
    if abs(dh) < 2.0 and q == q_expected:
        print("\nPASS (first gate): matches EES doc example within ±2 kJ/kg and q flag.")
    else:
        print("\nFAIL: does not match EES doc example. Fix units/derivatives/reference handling.")


if __name__ == "__main__":
    main()


__all__ = ["main"]
