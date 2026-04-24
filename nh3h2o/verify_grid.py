#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_grid.py

Grid regression for ammonia-water (Ibrahim & Klein 1993) against EES NH3H2O.

Outputs a CSV with columns suited for:
  - quick sanity checks (v>0, rho>0, etc.)
  - copy/paste or import into EES parametric tables

Default grid stays within IK nominal validity:
  T: 273–450 K (0–177 C) by default
  P: 0.2–100 bar by default
  X: NH3 mass fraction sweep

Usage examples:

  # basic run (default grid)
  runroot python -m nh3h2o.verify_grid --out out/nh3h2o_grid.csv

  # custom ranges
  python verify_grid.py --TminC 0 --TmaxC 120 --nT 7 --Pbars 2,5,10,20,50 --X 0.1,0.3,0.5,0.7,0.9 --out out/grid.csv

  # show only failures
  python verify_grid.py --only-fail --out out/fails.csv

Notes:
  - Two-phase rows MUST have xL/yV/wL/wV populated and ordered (xL<yV, wL<wV).
  - Single-phase rows DO NOT carry saturation fields by default (hygiene).
    If you want to keep VLE endpoints for all phases, pass --keep-vle.
  - If you run awk without a filename argument, awk will wait for stdin (which can look like a hang).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_blank(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    return False


def _as_float(v: object, *, name: str) -> float:
    if _is_blank(v):
        raise ValueError(f"Missing required value: {name}")
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Could not parse {name}={v!r} as float") from e


@dataclass(frozen=True)
class Case:
    T_K: float
    P_Pa: float
    X: float  # NH3 mass fraction


def build_cases(
    TminC: float,
    TmaxC: float,
    nT: int,
    Pbars: List[float],
    Xs: List[float],
) -> List[Case]:
    Ts_C = _linspace(TminC, TmaxC, nT)
    Ts_K = [t + 273.15 for t in Ts_C]
    Ps_Pa = [pbar * 1e5 for pbar in Pbars]  # bar -> Pa
    cases: List[Case] = []
    for T in Ts_K:
        for P in Ps_Pa:
            for X in Xs:
                cases.append(Case(T_K=T, P_Pa=P, X=X))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out/nh3h2o_grid.csv", help="Output CSV path")
    ap.add_argument("--TminC", type=float, default=0.0, help="Minimum temperature in C")
    ap.add_argument("--TmaxC", type=float, default=120.0, help="Maximum temperature in C")
    ap.add_argument("--nT", type=int, default=7, help="Number of temperature points (linspace)")
    ap.add_argument("--Pbars", type=str, default="2,5,10,20,50,100", help="Comma list of pressures in bar")
    ap.add_argument("--X", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma list of NH3 mass fractions")
    ap.add_argument("--strict", action="store_true", help="Call props_tpx(strict=True) to raise on nonphysical")
    ap.add_argument("--only-fail", action="store_true", help="Write only points that fail/throw")
    ap.add_argument(
        "--keep-vle",
        action="store_true",
        help="Keep xL/yV/wL/wV populated even for single-phase rows (default: blank them unless phase==2ph).",
    )
    ap.add_argument(
        "--auto-order-vle",
        action="store_true",
        help="If phase==2ph and xL>yV (or wL>wV), swap them instead of failing.",
    )
    args = ap.parse_args()

    # --- import ammonia_water props_tpx ---
    try:
        from nh3h2o.ammonia_water import props_tpx
    except Exception:
        from ammonia_water import props_tpx  # type: ignore

    Pbars = _parse_csv_floats(args.Pbars)
    Xs = _parse_csv_floats(args.X)
    cases = build_cases(args.TminC, args.TmaxC, args.nT, Pbars, Xs)

    out_path = Path(args.out)
    ensure_parent(out_path)

    # CSV columns (EES-friendly + SI + computed)
    # NOTE: ordering matters (awk checks rely on column indices).
    fieldnames = [
        # inputs
        "T_K", "T_C",
        "P_Pa", "P_kPa", "P_bar",
        "X_mass",
        # computed / metadata
        "phase", "q",
        "z_mole",
        # thermo
        "h_kJ_per_kg", "s_kJ_per_kgK", "u_kJ_per_kg",
        "v_m3_per_kg", "rho_kg_per_m3",
        # optional VLE info (only for 2ph by default)
        "xL_mole", "yV_mole", "wL_mass", "wV_mass",
        # status
        "ok", "error",
    ]

    n_ok = 0
    n_fail = 0
    phase_counts: dict[str, int] = {}

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for c in cases:
            row = {k: "" for k in fieldnames}

            row["T_K"] = f"{c.T_K:.6f}"
            row["T_C"] = f"{(c.T_K - 273.15):.6f}"
            row["P_Pa"] = f"{c.P_Pa:.6f}"
            row["P_kPa"] = f"{(c.P_Pa / 1000.0):.6f}"
            row["P_bar"] = f"{(c.P_Pa / 1e5):.6f}"
            row["X_mass"] = f"{c.X:.8f}"

            try:
                out = props_tpx(T_K=c.T_K, P_Pa=c.P_Pa, X=c.X, strict=args.strict)

                phase = str(out.get("phase", "")).strip()
                q = out.get("q", "")

                row["phase"] = phase
                row["q"] = str(q)
                row["z_mole"] = str(out.get("z_mole", ""))

                phase_counts[phase] = phase_counts.get(phase, 0) + 1

                # thermo (required)
                h = out.get("h_J_per_kg", None)
                s = out.get("s_J_per_kgK", None)
                u = out.get("u_J_per_kg", None)
                v = out.get("v_m3_per_kg", None)
                rho = out.get("rho_kg_per_m3", None)

                if _is_blank(h) or _is_blank(s) or _is_blank(u) or _is_blank(v) or _is_blank(rho):
                    raise ValueError(
                        "Missing one or more required thermo keys: "
                        f"h_J_per_kg={h!r}, s_J_per_kgK={s!r}, u_J_per_kg={u!r}, v_m3_per_kg={v!r}, rho_kg_per_m3={rho!r}"
                    )

                row["h_kJ_per_kg"] = f"{float(h)/1000.0:.8f}"
                row["s_kJ_per_kgK"] = f"{float(s)/1000.0:.8f}"
                row["u_kJ_per_kg"] = f"{float(u)/1000.0:.8f}"
                row["v_m3_per_kg"] = f"{float(v):.12e}"
                row["rho_kg_per_m3"] = f"{float(rho):.8f}"

                # VLE endpoints
                # Source keys: out["xL"], out["yV"], out["wL"], out["wV"]
                xL_raw = out.get("xL", "")
                yV_raw = out.get("yV", "")
                wL_raw = out.get("wL", "")
                wV_raw = out.get("wV", "")

                if args.keep_vle:
                    # Old behavior: always carry endpoints if present
                    row["xL_mole"] = str(xL_raw)
                    row["yV_mole"] = str(yV_raw)
                    row["wL_mass"] = str(wL_raw)
                    row["wV_mass"] = str(wV_raw)
                else:
                    # Hygiene: only for 2ph
                    if phase == "2ph":
                        xL = _as_float(xL_raw, name="xL")
                        yV = _as_float(yV_raw, name="yV")
                        wL_ = _as_float(wL_raw, name="wL")
                        wV_ = _as_float(wV_raw, name="wV")

                        # enforce ordering (NH3 richer in vapor): xL < yV and wL < wV
                        if (xL > yV) or (wL_ > wV_):
                            if args.auto_order_vle:
                                if xL > yV:
                                    xL, yV = yV, xL
                                if wL_ > wV_:
                                    wL_, wV_ = wV_, wL_
                            else:
                                raise ValueError(
                                    f"2ph VLE endpoints not ordered: xL={xL} yV={yV} wL={wL_} wV={wV_}"
                                )

                        row["xL_mole"] = f"{xL}"
                        row["yV_mole"] = f"{yV}"
                        row["wL_mass"] = f"{wL_}"
                        row["wV_mass"] = f"{wV_}"
                    # else: leave blanks for single-phase rows

                row["ok"] = "1"
                n_ok += 1

                if not args.only_fail:
                    w.writerow(row)

            except Exception as e:
                row["ok"] = "0"
                row["error"] = f"{type(e).__name__}: {e}"
                n_fail += 1
                w.writerow(row)

    print("=== verify_grid ===")
    print(f"out: {out_path}")
    print(f"cases: {len(cases)} | ok: {n_ok} | fail: {n_fail}")
    if phase_counts:
        parts = ", ".join(f"{k or '<blank>'}={v}" for k, v in sorted(phase_counts.items(), key=lambda kv: kv[0]))
        print(f"phase counts: {parts}")
    if args.only_fail:
        print("(only failures written)")


if __name__ == "__main__":
    main()

__all__ = [
    "Case",
    "build_cases",
    "ensure_parent",
    "main",
]
