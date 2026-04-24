#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoolProp constants / "trivial inputs" sandbox (v7.2.x compatible).

What it does
------------
For each requested fluid, it logs:

1) Numeric "trivial" (state-independent) thermodynamic_properties via the Python overload:
      CP.PropsSI(output_key, fluid)
   with a fallback to the 6-argument form:
      CP.PropsSI(output_key, "", 0, "", 0, fluid)

2) String metadata (CAS, aliases, etc.) via:
      CP.get_fluid_param_string(fluid, param)

3) Library-level info via:
      CP.get_global_param_string(key)

4) A few config doubles (e.g. universal gas constant) via:
      CP.get_config_double(...)

Extra rigor (optional)
----------------------
If your Python wrapper exposes CP.get_csv_parameter_list(), the script can:
- Enumerate CoolProp parameters
- Filter to those for which CP.is_trivial_parameter(idx) is True
- Attempt to evaluate each trivial parameter for each fluid

This gets close to: "list all constants CoolProp knows about" (for your build).

Run examples
------------
python coolprop_constants.py --fluid Nitrogen --fluid Water
python coolprop_constants.py --backend HEOS --fluid Nitrogen --out out.json

Notes
-----
- Some global keys may be unsupported by your build (you'll see ok=false with an error string).
- Many "nice-sounding" keys are not valid output strings in CoolProp; those will be logged as failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from typing import Any

import CoolProp.CoolProp as CP


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _ts_utc_iso_z() -> str:
    # timezone-aware, avoids deprecated utcnow()
    # Example: "2026-02-20T19:01:42.105054Z"
    s = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return s.replace("+00:00", "Z")


def _try_props_trivial(key: str, fluid: str) -> dict[str, Any]:
    """
    Try to evaluate a trivial (state-independent) numeric property.

    Primary:
      PropsSI(key, fluid)

    Fallback:
      PropsSI(key, "", 0, "", 0, fluid)
    """
    out: dict[str, Any] = {"key": key, "ok": False, "value": None, "method": None, "error": None}

    # Preferred (Python overload)
    try:
        v = CP.PropsSI(key, fluid)
        out.update(ok=True, value=float(v), method="PropsSI(key, fluid)")
        if not _finite(out["value"]):
            out.update(ok=False, error=f"non-finite: {out['value']!r}")
        return out
    except Exception as e1:
        # Fallback to 6-arg dummy inputs
        try:
            v = CP.PropsSI(key, "", 0.0, "", 0.0, fluid)
            out.update(ok=True, value=float(v), method='PropsSI(key,"",0,"",0,fluid)')
            if not _finite(out["value"]):
                out.update(ok=False, error=f"non-finite: {out['value']!r}")
            return out
        except Exception as e2:
            out.update(error=f"{type(e1).__name__}: {e1} | fallback {type(e2).__name__}: {e2}")
            return out


def _try_fluid_param_string(param: str, fluid: str) -> dict[str, Any]:
    out: dict[str, Any] = {"param": param, "ok": False, "value": None, "error": None}
    try:
        out.update(ok=True, value=str(CP.get_fluid_param_string(fluid, param)))
    except Exception as e:
        out.update(error=f"{type(e).__name__}: {e}")
    return out


def _try_global_param_string(key: str) -> dict[str, Any]:
    out: dict[str, Any] = {"key": key, "ok": False, "value": None, "error": None}
    try:
        out.update(ok=True, value=str(CP.get_global_param_string(key)))
    except Exception as e:
        out.update(error=f"{type(e).__name__}: {e}")
    return out


def _try_config_double(name: str, token: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "ok": False, "value": None, "error": None}
    try:
        out.update(ok=True, value=float(CP.get_config_double(token)))
    except Exception as e:
        out.update(error=f"{type(e).__name__}: {e}")
    return out


def _safe_get_parameter_info(idx: int, field: str) -> str | None:
    try:
        return str(CP.get_parameter_information(int(idx), str(field)))
    except Exception:
        return None


def _enumerate_trivial_parameter_keys() -> list[dict[str, Any]]:
    """
    Enumerate parameter keys from CoolProp, and return only those marked trivial.

    Returns list of dicts with at least:
      - key (string)
      - idx (int)
      - trivial (bool)
      - units (maybe)
      - description (maybe)
    """
    if not hasattr(CP, "get_csv_parameter_list"):
        return []

    try:
        csv_txt = CP.get_csv_parameter_list()
    except Exception:
        return []

    txt = str(csv_txt).strip()
    if not txt:
        return []

    # Use csv reader for robustness (quoted fields, commas in descriptions, etc.)
    rows_out: list[dict[str, Any]] = []
    reader = csv.reader(txt.splitlines())

    header_seen = False
    for row in reader:
        if not row:
            continue
        # First column is the key in the current CoolProp CSV list format.
        key = (row[0] or "").strip().strip('"').strip("'").strip()
        if not key:
            continue

        # Skip header if present
        if not header_seen and key.lower() in {"key", "parameter", "name"}:
            header_seen = True
            continue

        try:
            idx = int(CP.get_parameter_index(key))
            trivial = bool(CP.is_trivial_parameter(idx))
        except Exception:
            continue

        if trivial:
            rows_out.append(
                {
                    "key": key,
                    "idx": idx,
                    "trivial": True,
                    "units": _safe_get_parameter_info(idx, "units"),
                    "description": _safe_get_parameter_info(idx, "description"),
                }
            )

    return rows_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="HEOS", help="CoolProp backend, e.g. HEOS, REFPROP (if installed)")
    ap.add_argument("--fluid", action="append", required=True, help="Fluid name, e.g. Nitrogen, Water")
    ap.add_argument("--out", default="", help="Optional JSON output path")
    ap.add_argument("--no-auto", action="store_true", help="Skip auto enumeration of trivial keys")
    ap.add_argument("--max-auto-keys", type=int, default=50, help="Preview limit for auto trivial keys in global section")
    ap.add_argument("--include-all-auto-keys", action="store_true", help="Include ALL auto trivial keys in global section")
    ns = ap.parse_args()

    def backend_prefix(fluid: str) -> str:
        f = str(fluid).strip()
        if "::" in f:
            return f
        return f"{ns.backend}::{f}" if ns.backend else f

    numeric_trivial_candidates = [
        # critical point
        "Tcrit", "pcrit", "rhocrit", "rhomolar_critical",
        # triple point
        "Ttriple", "ptriple",
        # limits
        "Tmin", "Tmax", "pmax",
        # reducing point (corresponding states)
        "T_reducing", "p_reducing", "rhomolar_reducing",
        # pure-fluid constants
        "molemass", "molar_mass", "M",
        "acentric",
        # intentionally include a couple of "often guessed" but invalid keys to prove behavior
        "acentric_factor",
        "dipole_moment",
    ]

    string_params = [
        "CAS", "aliases", "Name", "ASHRAE34", "formula",
        "InChI", "InChIKey", "SMILES",
        "REFPROPName", "BibTeX", "description",
    ]

    global_keys = [
        "version", "gitrevision", "gitbranch", "fluids_list", "backend_list",
    ]

    config_doubles: list[tuple[str, Any]] = []
    for name in ["R_U_CODATA", "R_U_SI"]:
        token = getattr(CP, name, None)
        if token is not None:
            config_doubles.append((name, token))

    trivial_param_keys: list[dict[str, Any]] = []
    if not ns.no_auto:
        trivial_param_keys = _enumerate_trivial_parameter_keys()

    report: dict[str, Any] = {
        "ts": _ts_utc_iso_z(),
        "coolprop_version": None,
        "global": {
            "strings": [_try_global_param_string(k) for k in global_keys],
            "config_doubles": [_try_config_double(n, tok) for (n, tok) in config_doubles],
            "auto_trivial_keys": {
                "available": bool(trivial_param_keys),
                "count": len(trivial_param_keys),
                "keys": [],
            },
        },
        "fluids": [],
    }

    # best-effort version
    try:
        report["coolprop_version"] = str(CP.get_global_param_string("version"))
    except Exception:
        report["coolprop_version"] = None

    # Global preview of auto keys
    if trivial_param_keys:
        if ns.include_all_auto_keys:
            report["global"]["auto_trivial_keys"]["keys"] = trivial_param_keys
        else:
            report["global"]["auto_trivial_keys"]["keys"] = trivial_param_keys[: max(0, int(ns.max_auto_keys))]

    for fluid in ns.fluid:
        f_full = backend_prefix(fluid)

        f_entry: dict[str, Any] = {
            "fluid": str(fluid),
            "fluid_full": f_full,
            "numeric_trivial_candidates": [],
            "numeric_trivial_auto": [],
            "string_params": [],
            "derived": {},
        }

        # candidate numeric trivial constants
        for k in numeric_trivial_candidates:
            f_entry["numeric_trivial_candidates"].append(_try_props_trivial(k, f_full))

        # auto-enumerated trivial constants (if available)
        if trivial_param_keys:
            for row in trivial_param_keys:
                f_entry["numeric_trivial_auto"].append(_try_props_trivial(row["key"], f_full))

        # string metadata (base fluid name; backend prefix not needed)
        for p in string_params:
            f_entry["string_params"].append(_try_fluid_param_string(p, str(fluid)))

        # Derived: R_spec = R_u / M
        derived: dict[str, Any] = {}

        mm = None
        for cand in ["molemass", "molar_mass", "M"]:
            try:
                v = CP.PropsSI(cand, f_full)
                if _finite(v) and float(v) > 0:
                    mm = float(v)
                    derived["molar_mass_key_used"] = cand
                    derived["molar_mass_kg_per_mol"] = mm
                    break
            except Exception:
                continue

        Ru = None
        # Prefer CODATA if present
        tok = getattr(CP, "R_U_CODATA", None)
        if tok is not None:
            try:
                Ru = float(CP.get_config_double(tok))
                derived["R_u_source"] = "get_config_double(R_U_CODATA)"
                derived["R_u_J_per_molK"] = Ru
            except Exception:
                Ru = None
        # fallback: if R_U_SI exists and CODATA did not
        if Ru is None:
            tok2 = getattr(CP, "R_U_SI", None)
            if tok2 is not None:
                try:
                    Ru = float(CP.get_config_double(tok2))
                    derived["R_u_source"] = "get_config_double(R_U_SI)"
                    derived["R_u_J_per_molK"] = Ru
                except Exception:
                    Ru = None

        if mm is not None and Ru is not None:
            derived["R_spec_J_per_kgK"] = Ru / mm

        f_entry["derived"] = derived
        report["fluids"].append(f_entry)

    txt = json.dumps(report, indent=2, sort_keys=False)
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"Wrote: {ns.out}")
    else:
        print(txt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
