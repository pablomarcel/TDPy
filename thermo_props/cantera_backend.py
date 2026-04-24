# thermo_props/cantera_backend.py
from __future__ import annotations

"""thermo_props.cantera_backend

Cantera backend (low-level).

This module provides a CoolProp-like property function, **CTPropsSI**, intended
for reacting-mixture / chemical-equilibrium work where Cantera is the source of truth.

Key upgrades vs. the initial implementation
-------------------------------------------
- **Aggressive memoization** for repeated CTPropsSI calls.
  This matters a lot for SciPy solvers that repeatedly evaluate the same
  thermodynamic properties (e.g., standard-state ḡ° calls inside equilibrium problems).
- Cached parsing of fluid specs.
- Cached configured Solution templates (mechanism + phase + composition) and
  fast cloning when available.
- Optional cache inspection/clearing helpers.

Design goals
------------
- Import-time light: Cantera is imported lazily, so tdpy can run without it.
- Independent: no dependency on CoolProp or other tdpy backends.
- EES-ish ergonomics: call CTPropsSI like PropsSI:

    CTPropsSI(out, in1, v1, in2, v2, fluid)

Fluid specification
-------------------
The ``fluid`` argument is a flexible string that tells Cantera which mechanism
to load, what phase to use (if applicable), what composition to set, and
whether to equilibrate.

Supported examples:
  - "gri30.yaml"                          -> ct.Solution("gri30.yaml")
  - "gri30.yaml::gri30"                   -> ct.Solution("gri30.yaml", "gri30")
  - "gri30.yaml|X=CH4:1,O2:2,N2:7.52"     -> set mole fractions (X)
  - "gri30.yaml|Y=..."                    -> set mass fractions (Y)
  - "gri30.yaml|equil=TP"                 -> equilibrate after setting state
  - Bracket form is also accepted:
        "gri30.yaml[ X=CH4:1,O2:2,N2:7.52 | equil=TP ]"

Supported inputs (SI, order-agnostic)
------------------------------------
- (T, P)         : K, Pa
- (T, D)         : K, kg/m^3  (aliases: rho, Dmass, density)
- (Hmass, P)     : J/kg, Pa   (aliases: h, H)
- (Smass, P)     : J/kg/K, Pa (aliases: s, S)
- (HMOLAR, P)    : J/kmol, Pa (converted to mass-basis using MW)
- (SMOLAR, P)    : J/kmol/K, Pa (converted to mass-basis using MW)

Supported outputs (SI / Cantera conventions)
-------------------------------------------
- T, P, D (density, kg/m^3)
- Hmass, Smass, Umass, Gmass
- Cpmass, Cvmass
- HMOLAR, SMOLAR, UMOLAR, GMOLAR
- CPMOLAR, CVMOLAR
- MOLAR_MASS (mean molecular weight, kg/kmol)
- Z (compressibility factor, computed from P, rho, T, MW)
- mu / viscosity (Pa*s) and k / conductivity (W/m/K) if transport is available
- Pr (Prandtl number) if transport is available
- Composition component access:
    * X[0], Y[0]               (by index)
    * X[CH4], Y[O2]            (by species name)
    * X:CH4, Y:O2              (alternate form)
- Chemical potentials:
    * chemical_potential[CH4]  (J/kmol)
    * mu[CH4]                  (alias)

Notes
-----
- This backend is intentionally focused on Cantera use cases and does not aim
  to emulate the full CoolProp key universe.
- No unit conversion is performed; pass SI values.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import numbers
import os
import re
from typing import Any, Iterable, Mapping, Sequence, Tuple

__all__ = [
    # errors
    "CanteraNotInstalled",
    "CanteraCallError",
    # availability
    "cantera_available",
    "cantera_version",
    # cache helpers
    "ctprops_cache_info",
    "clear_ctprops_caches",
    # main API
    "ctprops_si",
    "ctprops_multi",
    "batch_ctprops",
    # shims
    "CTPropsSI",
    # dataclass (optional convenience)
    "CTCall",
]

# ------------------------------ cache sizing ------------------------------

def _env_int(name: str, default: int) -> int:
    s = os.getenv(name, "").strip()
    if not s:
        return default
    try:
        v = int(s)
        return v if v >= 0 else default
    except Exception:
        return default

# How many distinct CTPropsSI calls to memoize (by canonicalized arguments).
# Bump if you do property-table sweeps; lower if you worry about memory.
_CTPROPS_CACHE_MAXSIZE = _env_int("TDPY_CTPROPS_CACHE_MAXSIZE", 8192)

# Cache sizes for parsing and configured Solution templates
_CTPARSE_CACHE_MAXSIZE = _env_int("TDPY_CTPARSE_CACHE_MAXSIZE", 512)
_CTSOL_MECH_CACHE_MAXSIZE = _env_int("TDPY_CTSOL_MECH_CACHE_MAXSIZE", 64)
_CTSOL_CFG_CACHE_MAXSIZE = _env_int("TDPY_CTSOL_CFG_CACHE_MAXSIZE", 128)

# ------------------------------ errors ------------------------------


class CanteraNotInstalled(ImportError):
    """Raised when Cantera cannot be imported."""


class CanteraCallError(RuntimeError):
    """Raised when a Cantera (CTPropsSI) call fails; message includes call context."""


# ------------------------------ lazy import ------------------------------


_ct_mod: Any | None = None


def _import_cantera() -> Any:
    """Import cantera lazily and cache the module."""
    global _ct_mod
    if _ct_mod is not None:
        return _ct_mod
    try:
        import cantera as ct  # type: ignore
    except Exception as e:  # pragma: no cover
        raise CanteraNotInstalled(
            "Cantera is not installed or could not be imported. Install with: pip install cantera"
        ) from e
    _ct_mod = ct
    return _ct_mod


def cantera_available() -> bool:
    try:
        _import_cantera()
        return True
    except Exception:
        return False


def cantera_version() -> str | None:
    try:
        ct = _import_cantera()
        v = getattr(ct, "__version__", None)
        return str(v) if v else None
    except Exception:
        return None


# ------------------------------ small utils ------------------------------


def _to_float(name: str, x: Any) -> float:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be a number, got bool")
    if isinstance(x, numbers.Real):
        y = float(x)
    else:
        try:
            y = float(x)
        except Exception as e:
            raise TypeError(f"{name} must be a number, got {type(x).__name__}: {x!r}") from e
    return y


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _norm_key(key: str) -> str:
    k = str(key).strip()
    if not k:
        raise ValueError("Key is empty")
    return k


# ------------------------------ key normalization ------------------------------


_CT_IN_ALIASES: Mapping[str, str] = {
    # temperature / pressure / density
    "t": "T",
    "temp": "T",
    "temperature": "T",
    "p": "P",
    "press": "P",
    "pressure": "P",
    "d": "D",
    "rho": "D",
    "dmass": "D",
    "density": "D",
    # mass-basis
    "h": "Hmass",
    "hmass": "Hmass",
    "enthalpy": "Hmass",
    "s": "Smass",
    "smass": "Smass",
    "entropy": "Smass",
    "u": "Umass",
    "umass": "Umass",
    "internalenergy": "Umass",
    "internal_energy": "Umass",
    # molar-basis
    "hmolar": "HMOLAR",
    "h_molar": "HMOLAR",
    "smolar": "SMOLAR",
    "s_molar": "SMOLAR",
    "umolar": "UMOLAR",
    "u_molar": "UMOLAR",
}

_CT_OUT_ALIASES: Mapping[str, str] = {
    **_CT_IN_ALIASES,
    # heat capacities / gibbs
    "cp": "Cpmass",
    "cpmass": "Cpmass",
    "cv": "Cvmass",
    "cvmass": "Cvmass",
    "g": "Gmass",
    "gmass": "Gmass",
    # molar heat capacities / gibbs
    "cpmolar": "CPMOLAR",
    "cp_molar": "CPMOLAR",
    "cvmolar": "CVMOLAR",
    "cv_molar": "CVMOLAR",
    "gmolar": "GMOLAR",
    "g_molar": "GMOLAR",
    # mean molecular weight
    "mw": "MOLAR_MASS",
    "molarmass": "MOLAR_MASS",
    "molar_mass": "MOLAR_MASS",
    "m": "MOLAR_MASS",
    # transport
    "mu": "VISCOSITY",
    "viscosity": "VISCOSITY",
    "k": "CONDUCTIVITY",
    "conductivity": "CONDUCTIVITY",
    "thermal_conductivity": "CONDUCTIVITY",
    "pr": "PRANDTL",
    # compressibility
    "z": "Z",
    # composition vectors
    "x": "X",
    "y": "Y",
    # chemical potentials
    "chemical_potential": "CHEMICAL_POTENTIAL",
    "chemicalpotential": "CHEMICAL_POTENTIAL",
    "mu_species": "CHEMICAL_POTENTIAL",
}

@lru_cache(maxsize=256)
def _ct_norm_in(key: str) -> str:
    k = _norm_key(key)
    return _CT_IN_ALIASES.get(k.lower(), k)

@lru_cache(maxsize=256)
def _ct_norm_out(key: str) -> str:
    k = _norm_key(key)
    return _CT_OUT_ALIASES.get(k.lower(), k)

# ------------------------------ fluid spec parsing ------------------------------


@dataclass(frozen=True)
class _CTParsedFluidSpec:
    raw: str
    mech: str
    phase: str | None
    comp_basis: str | None  # "X" (mole) or "Y" (mass)
    comp: str | None        # composition string "CH4:1,O2:2"
    equilibrate: str | None # e.g. "TP", "HP", ...

_CT_BRACKET_RE = re.compile(r"^(?P<head>[^\[]+)\[(?P<tail>.+)\]\s*$")

@lru_cache(maxsize=_CTPARSE_CACHE_MAXSIZE)
def _parse_ct_fluid_spec(fluid: str) -> _CTParsedFluidSpec:
    raw = str(fluid).strip()
    if not raw:
        raise ValueError("CTPropsSI fluid must be a non-empty string.")

    head = raw
    tail = ""

    m = _CT_BRACKET_RE.match(raw)
    if m:
        head = m.group("head").strip()
        tail = m.group("tail").strip()
    else:
        if "|" in raw:
            head, tail = raw.split("|", 1)
            head = head.strip()
            tail = tail.strip()
        elif ";" in raw:
            head, tail = raw.split(";", 1)
            head = head.strip()
            tail = tail.strip()

    mech = head.strip()
    phase: str | None = None
    if "::" in mech:
        mech, phase = mech.split("::", 1)
        mech = mech.strip()
        phase = phase.strip() or None

    comp_basis: str | None = None
    comp: str | None = None
    equilibrate: str | None = None

    if tail:
        toks = [t.strip() for t in re.split(r"[|;]+", tail) if t.strip()]
        for tok in toks:
            if "=" in tok:
                k, v = tok.split("=", 1)
                k_l = k.strip().lower()
                v0 = v.strip()
                if k_l in {"x", "mole", "molefrac", "mole_fractions"}:
                    comp_basis = "X"
                    comp = v0
                    continue
                if k_l in {"y", "mass", "massfrac", "mass_fractions"}:
                    comp_basis = "Y"
                    comp = v0
                    continue
                if k_l in {"comp", "composition"}:
                    comp = v0
                    continue
                if k_l in {"basis"}:
                    s = v0.strip().lower()
                    if s in {"x", "mole", "moles", "molefrac"}:
                        comp_basis = "X"
                    elif s in {"y", "mass", "massfrac"}:
                        comp_basis = "Y"
                    continue
                if k_l in {"equil", "equilibrate", "eq"}:
                    equilibrate = v0.strip().upper()
                    continue
                if k_l in {"phase"}:
                    phase = v0.strip() or None
                    continue
            # no '=' token: heuristics
            if ":" in tok and comp is None:
                comp = tok
                continue
            if tok.strip().upper() in {"TP", "HP", "SP", "TV", "UV", "SV", "PV"} and equilibrate is None:
                equilibrate = tok.strip().upper()
                continue

    return _CTParsedFluidSpec(
        raw=raw,
        mech=mech,
        phase=phase,
        comp_basis=comp_basis,
        comp=comp,
        equilibrate=equilibrate,
    )

# ------------------------------ Solution creation (cached templates) ------------------------------

@lru_cache(maxsize=_CTSOL_MECH_CACHE_MAXSIZE)
def _ct_solution_mech_template(mech: str, phase: str | None) -> Any:
    """Cache mechanism/phase load (expensive). Returns a Solution instance used as template."""
    ct = _import_cantera()
    if phase:
        return ct.Solution(str(mech), str(phase))
    return ct.Solution(str(mech))

@lru_cache(maxsize=_CTSOL_CFG_CACHE_MAXSIZE)
def _ct_solution_configured_template(
    mech: str, phase: str | None, comp_basis: str | None, comp: str | None
) -> Any:
    """Cache a template Solution with composition already assigned (still a template)."""
    templ = _ct_solution_mech_template(mech, phase)
    clone = getattr(templ, "clone", None)
    if callable(clone):
        gas = clone()
    else:  # pragma: no cover
        ct = _import_cantera()
        gas = ct.Solution(str(mech), str(phase)) if phase else ct.Solution(str(mech))

    if comp:
        basis = (comp_basis or "X").upper()
        if basis == "Y":
            gas.Y = str(comp)
        else:
            gas.X = str(comp)
    return gas

def _ct_new_solution(spec: _CTParsedFluidSpec) -> Any:
    """Create a fresh Solution quickly using cached configured templates when possible."""
    templ = _ct_solution_configured_template(spec.mech, spec.phase, spec.comp_basis, spec.comp)
    clone = getattr(templ, "clone", None)
    if callable(clone):
        return clone()
    # fallback (older Cantera): reconstruct
    ct = _import_cantera()
    gas = ct.Solution(str(spec.mech), str(spec.phase)) if spec.phase else ct.Solution(str(spec.mech))
    if spec.comp:
        basis = (spec.comp_basis or "X").upper()
        if basis == "Y":
            gas.Y = str(spec.comp)
        else:
            gas.X = str(spec.comp)
    return gas

# ------------------------------ component suffix parsing ------------------------------

_SUFFIX_BRACKET_RE = re.compile(r"^(?P<base>[^\[]+)\[(?P<comp>.+)\]\s*$")

def _split_component_suffix(out_key: str) -> tuple[str, str | None]:
    """Split keys like 'X[CH4]' or 'mu[0]' or 'X:CH4'."""
    s = str(out_key).strip()
    if not s:
        raise ValueError("Output key is empty")

    m = _SUFFIX_BRACKET_RE.match(s)
    if m:
        return m.group("base").strip(), m.group("comp").strip()

    # alternate 'BASE:COMP' (only for a few bases)
    if ":" in s:
        base, comp = s.split(":", 1)
        base = base.strip()
        comp = comp.strip()
        if base and comp and base.lower() in {"x", "y", "mu", "chemical_potential"}:
            return base, comp

    return s, None

def _component_index(gas: Any, comp: str) -> int:
    """Convert component selector into species index."""
    c = str(comp).strip()
    if not c:
        raise ValueError("Empty component selector")
    if c.isdigit():
        return int(c)
    idx = int(gas.species_index(c))
    if idx < 0:
        raise ValueError(f"Species {c!r} not found in mechanism")
    return idx

# ------------------------------ state setters ------------------------------

def _ct_set_state(gas: Any, in1: str, v1: float, in2: str, v2: float) -> None:
    k1 = _ct_norm_in(in1)
    k2 = _ct_norm_in(in2)
    v1f = _to_float("v1", v1)
    v2f = _to_float("v2", v2)
    vals: dict[str, float] = {k1: v1f, k2: v2f}
    keys = set(vals.keys())

    try:
        if keys == {"T", "P"}:
            gas.TP = float(vals["T"]), float(vals["P"])
            return
        if keys == {"T", "D"}:
            gas.TD = float(vals["T"]), float(vals["D"])
            return

        if keys == {"Hmass", "P"}:
            gas.HP = float(vals["Hmass"]), float(vals["P"])
            return
        if keys == {"Smass", "P"}:
            gas.SP = float(vals["Smass"]), float(vals["P"])
            return

        # molar-basis setters (convert to mass using MW [kg/kmol])
        if keys == {"HMOLAR", "P"}:
            mw = float(getattr(gas, "mean_molecular_weight"))
            if mw <= 0.0 or not _finite(mw):
                raise ValueError("Invalid mean molecular weight for HMOLAR conversion")
            gas.HP = float(vals["HMOLAR"]) / mw, float(vals["P"])
            return
        if keys == {"SMOLAR", "P"}:
            mw = float(getattr(gas, "mean_molecular_weight"))
            if mw <= 0.0 or not _finite(mw):
                raise ValueError("Invalid mean molecular weight for SMOLAR conversion")
            gas.SP = float(vals["SMOLAR"]) / mw, float(vals["P"])
            return
    except Exception as e:
        raise CanteraCallError(
            f"CTPropsSI state-set failed for ({in1!r},{in2!r}) with values ({v1f},{v2f}): {e}"
        ) from e

    raise CanteraCallError(
        "CTPropsSI unsupported input pair. "
        f"Got ({k1!r},{k2!r}). Supported: (T,P), (T,D), (Hmass,P), (Smass,P), (HMOLAR,P), (SMOLAR,P)."
    )

# ------------------------------ output getters ------------------------------

_RU_J_PER_KMOLK = 8314.46261815324  # J/kmol/K

def _ct_get_output(gas: Any, out_key_raw: str) -> float:
    base, comp = _split_component_suffix(out_key_raw)
    out_n = _ct_norm_out(base)

    try:
        if out_n == "T":
            return float(gas.T)
        if out_n == "P":
            return float(gas.P)
        if out_n == "D":
            return float(gas.density)

        # mass-basis
        if out_n == "Hmass":
            return float(gas.enthalpy_mass)
        if out_n == "Smass":
            return float(gas.entropy_mass)
        if out_n == "Umass":
            return float(gas.int_energy_mass)
        if out_n == "Gmass":
            return float(gas.gibbs_mass)
        if out_n == "Cpmass":
            return float(gas.cp_mass)
        if out_n == "Cvmass":
            return float(gas.cv_mass)

        # molar-basis (Cantera convention: J/kmol, J/kmol/K)
        if out_n == "HMOLAR":
            return float(gas.enthalpy_mole)
        if out_n == "SMOLAR":
            return float(gas.entropy_mole)
        if out_n == "UMOLAR":
            return float(gas.int_energy_mole)
        if out_n == "GMOLAR":
            return float(gas.gibbs_mole)
        if out_n == "CPMOLAR":
            return float(gas.cp_mole)
        if out_n == "CVMOLAR":
            return float(gas.cv_mole)

        if out_n == "MOLAR_MASS":
            return float(gas.mean_molecular_weight)

        if out_n == "Z":
            # Z = P / (rho * Rspec * T), where Rspec = Ru / MW
            P = float(gas.P)
            rho = float(gas.density)
            T = float(gas.T)
            mw = float(gas.mean_molecular_weight)
            if rho <= 0.0 or T <= 0.0 or mw <= 0.0:
                raise ValueError("Invalid state for Z calculation")
            Rspec = _RU_J_PER_KMOLK / mw
            return float(P / (rho * Rspec * T))

        if out_n == "VISCOSITY":
            return float(gas.viscosity)
        if out_n == "CONDUCTIVITY":
            return float(gas.thermal_conductivity)
        if out_n == "PRANDTL":
            mu = float(gas.viscosity)
            k = float(gas.thermal_conductivity)
            cp = float(gas.cp_mass)
            if k == 0.0:
                raise ValueError("thermal_conductivity is zero")
            return float(cp * mu / k)

        # composition vectors
        if out_n == "X":
            if comp is None:
                raise ValueError("X requires a component selector like X[CH4] or X[0]")
            i = _component_index(gas, comp)
            return float(gas.X[int(i)])
        if out_n == "Y":
            if comp is None:
                raise ValueError("Y requires a component selector like Y[CH4] or Y[0]")
            i = _component_index(gas, comp)
            return float(gas.Y[int(i)])

        if out_n == "CHEMICAL_POTENTIAL":
            if comp is None:
                raise ValueError("chemical_potential requires a component selector like mu[CH4]")
            i = _component_index(gas, comp)
            return float(gas.chemical_potentials[int(i)])

    except Exception as e:
        raise CanteraCallError(f"CTPropsSI output evaluation failed for out={out_key_raw!r}: {e}") from e

    raise CanteraCallError(f"CTPropsSI unsupported output key: {out_key_raw!r} (normalized={out_n!r}).")

# ------------------------------ CTPropsSI memoization ------------------------------

def _float_key(x: float) -> float:
    """Canonicalize floats for cache keys. Keeps constants stable; reduces noisy near-equality."""
    xf = float(x)
    if not math.isfinite(xf):
        return xf
    # 12 significant digits is typically plenty for thermo state keys.
    return float(f"{xf:.12g}")

def _canon_out_key(out: str) -> str:
    base, comp = _split_component_suffix(out)
    base_n = _ct_norm_out(base)
    if comp is not None:
        comp_s = str(comp).strip()
        # normalize digit selectors; leave species names as-is
        if comp_s.isdigit():
            comp_s = str(int(comp_s))
        return f"{base_n}[{comp_s}]"
    return base_n

def _canon_in_pair(in1: str, v1: float, in2: str, v2: float) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    k1 = _ct_norm_in(in1)
    k2 = _ct_norm_in(in2)
    p1 = (k1, _float_key(_to_float("v1", v1)))
    p2 = (k2, _float_key(_to_float("v2", v2)))
    # order-agnostic for supported pairs
    return tuple(sorted((p1, p2), key=lambda t: t[0]))  # type: ignore[return-value]

@lru_cache(maxsize=_CTPROPS_CACHE_MAXSIZE)
def _ctprops_cached(
    out_c: str,
    pA_k: str,
    pA_v: float,
    pB_k: str,
    pB_v: float,
    fluid_raw: str,
) -> float:
    """Core cached evaluator. Arguments must already be canonicalized."""
    spec = _parse_ct_fluid_spec(fluid_raw)
    gas = _ct_new_solution(spec)

    # set state
    _ct_set_state(gas, pA_k, pA_v, pB_k, pB_v)

    # optional equilibrium
    if spec.equilibrate:
        gas.equilibrate(str(spec.equilibrate))

    # compute output
    y = _ct_get_output(gas, out_c)

    if not _finite(float(y)):
        raise CanteraCallError(
            f"CTPropsSI returned non-finite result for out={out_c!r}, "
            f"in=({pA_k!r},{pB_k!r}), fluid={fluid_raw!r}: {y!r}"
        )
    return float(y)

def ctprops_cache_info() -> dict[str, Any]:
    """Return cache stats (useful for debugging performance)."""
    return {
        "ctprops_cache_maxsize": _CTPROPS_CACHE_MAXSIZE,
        "ctprops_cache_info": getattr(_ctprops_cached, "cache_info", lambda: None)(),
        "ctparse_cache_maxsize": _CTPARSE_CACHE_MAXSIZE,
        "ctparse_cache_info": getattr(_parse_ct_fluid_spec, "cache_info", lambda: None)(),
        "ctsol_mech_cache_maxsize": _CTSOL_MECH_CACHE_MAXSIZE,
        "ctsol_mech_cache_info": getattr(_ct_solution_mech_template, "cache_info", lambda: None)(),
        "ctsol_cfg_cache_maxsize": _CTSOL_CFG_CACHE_MAXSIZE,
        "ctsol_cfg_cache_info": getattr(_ct_solution_configured_template, "cache_info", lambda: None)(),
        "cantera_version": cantera_version(),
    }

def clear_ctprops_caches() -> None:
    """Clear all internal LRU caches."""
    try:
        _ctprops_cached.cache_clear()
    except Exception:
        pass
    try:
        _parse_ct_fluid_spec.cache_clear()
    except Exception:
        pass
    try:
        _ct_solution_mech_template.cache_clear()
    except Exception:
        pass
    try:
        _ct_solution_configured_template.cache_clear()
    except Exception:
        pass

# ------------------------------ public API ------------------------------

def ctprops_si(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    """Cantera-backed property lookup with a CoolProp-like signature (memoized)."""
    out_c = _canon_out_key(out)
    (pA, pB) = _canon_in_pair(in1, v1, in2, v2)
    fluid_s = str(fluid).strip()
    try:
        return _ctprops_cached(out_c, pA[0], pA[1], pB[0], pB[1], fluid_s)
    except CanteraNotInstalled:
        raise
    except Exception as e:
        # Wrap with call context (keep the helpful info from the original version)
        raise CanteraCallError(
            "CTPropsSI call failed.\n"
            f"  out={out!r} (canon={out_c!r})\n"
            f"  in1={in1!r} v1={v1!r}\n"
            f"  in2={in2!r} v2={v2!r}\n"
            f"  fluid={fluid_s!r}\n"
            f"  cause={type(e).__name__}: {e}"
        ) from e

def ctprops_multi(
    outputs: Sequence[str],
    in1: str,
    v1: float,
    in2: str,
    v2: float,
    fluid: str,
) -> dict[str, float]:
    """Compute multiple CT outputs for the same state (state set once)."""
    spec = _parse_ct_fluid_spec(str(fluid).strip())
    gas = _ct_new_solution(spec)
    _ct_set_state(gas, in1, v1, in2, v2)
    if spec.equilibrate:
        gas.equilibrate(str(spec.equilibrate))

    out: dict[str, float] = {}
    for k in outputs:
        out[str(k)] = float(_ct_get_output(gas, str(k)))
    return out

@dataclass(frozen=True)
class CTCall:
    """A single Cantera CTPropsSI call signature (same shape as PropsSI)."""
    out: str
    in1: str
    v1: float
    in2: str
    v2: float
    fluid: str

def batch_ctprops(calls: Iterable[CTCall]) -> list[float]:
    """Execute a batch of CTPropsSI calls (memoized per call)."""
    ys: list[float] = []
    for c in calls:
        ys.append(ctprops_si(c.out, c.in1, c.v1, c.in2, c.v2, c.fluid))
    return ys

# ------------------------------ CoolProp-like shims ------------------------------

def CTPropsSI(out: str, in1: str, v1: float, in2: str, v2: float, fluid: str) -> float:
    """Alias shim so input files can call CTPropsSI the same way they call PropsSI."""
    return ctprops_si(out, in1, v1, in2, v2, fluid)
