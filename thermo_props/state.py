from __future__ import annotations

"""
thermo_props.state

State builder for thermodynamic properties (CoolProp-backed).

Goals:
- Predictable state construction from *two* independent thermodynamic_properties.
- Accept EES-ish aliases (h, s, rho, v, x) and unit-suffixed keys (T_C, P_bar, h_kJkg, ...).
- Return a rich state object with a consistent property dictionary in SI units.

Notes:
- This module builds *single thermodynamic states*. The "solve sets of equations"
  experience lives in `equations/*`.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import math

from utils import with_error_context

from .coolprop_backend import (
    CoolPropCallError,
    CoolPropNotInstalled,
    phase_si,
    props_multi,
    props_si,
)

# ------------------------------ canonical outputs ------------------------------
# CoolProp-native keys for computed thermodynamic_properties (SI), plus a few convenient derived ones.

DEFAULT_OUTPUTS: tuple[str, ...] = (
    "T",        # K
    "P",        # Pa
    "Q",        # quality (0..1), may be NaN in single phase
    "Hmass",    # J/kg
    "Smass",    # J/kg-K
    "Umass",    # J/kg
    "Dmass",    # kg/m^3
    "Cpmass",   # J/kg-K
    "Cvmass",   # J/kg-K
    "A",        # m/s speed of sound
    "V",        # Pa*s viscosity (CoolProp key)
    "L",        # W/m-K thermal conductivity (CoolProp key)
    "Prandtl",  # -
)

# Allowed CoolProp input pairs (after normalization)
_ALLOWED_IN: set[str] = {"T", "P", "Q", "Hmass", "Smass", "Umass", "Dmass"}

# Keys to ignore when picking the "two independent thermodynamic_properties" out of a mapping
_IGNORE_KEYS: set[str] = {
    "fluid",
    "name",
    "label",
    "notes",
    "comment",
    "id",
    "backend",
    "outputs",
    "include_phase",
    "meta",
    "units",
    "unit",
    "basis",
    "options",
    # defensive (in case someone passes a whole thermo_props payload)
    "state",
    "states",
    "given",
    "ask",
    "mode",
}

# ------------------------------ input aliases ------------------------------
# EES-ish aliases for *inputs*.
_CANONICAL_IN_ALIASES: Mapping[str, str] = {
    "t": "T",
    "p": "P",
    "q": "Q",
    "x": "Q",
    "h": "Hmass",
    "s": "Smass",
    "u": "Umass",
    "rho": "Dmass",
    "d": "Dmass",
    "dmass": "Dmass",
    # specific volume: EES uses "v" for m^3/kg
    "v": "v_m3kg",
}

# ------------------------------ unit-suffixed inputs ------------------------------
# v_si = value * factor + offset  (except specific volume which is inverted)
_UNIT_KEYMAP: Mapping[str, tuple[str, float, float]] = {
    # Temperature
    "T": ("T", 1.0, 0.0),
    "T_K": ("T", 1.0, 0.0),
    "T_C": ("T", 1.0, 273.15),
    # Pressure
    "P": ("P", 1.0, 0.0),
    "P_Pa": ("P", 1.0, 0.0),
    "P_kPa": ("P", 1.0e3, 0.0),
    "P_MPa": ("P", 1.0e6, 0.0),
    "P_bar": ("P", 1.0e5, 0.0),
    "P_atm": ("P", 101325.0, 0.0),
    # Enthalpy
    "h": ("Hmass", 1.0, 0.0),
    "Hmass": ("Hmass", 1.0, 0.0),
    "h_Jkg": ("Hmass", 1.0, 0.0),
    "h_kJkg": ("Hmass", 1.0e3, 0.0),
    # Entropy
    "s": ("Smass", 1.0, 0.0),
    "Smass": ("Smass", 1.0, 0.0),
    "s_JkgK": ("Smass", 1.0, 0.0),
    "s_kJkgK": ("Smass", 1.0e3, 0.0),
    # Internal energy
    "u": ("Umass", 1.0, 0.0),
    "Umass": ("Umass", 1.0, 0.0),
    "u_Jkg": ("Umass", 1.0, 0.0),
    "u_kJkg": ("Umass", 1.0e3, 0.0),
    # Density
    "rho": ("Dmass", 1.0, 0.0),
    "rho_kgm3": ("Dmass", 1.0, 0.0),
    "Dmass": ("Dmass", 1.0, 0.0),
    # Specific volume (m^3/kg) -> Dmass (inversion; handled specially)
    "v_m3kg": ("Dmass", -1.0, 0.0),
    # Quality
    "x": ("Q", 1.0, 0.0),
    "q": ("Q", 1.0, 0.0),
    "Q": ("Q", 1.0, 0.0),
}

# ------------------------------ output aliases (user-friendly) ------------------------------

_DERIVED_OUTPUTS: set[str] = {"v", "gamma", "R_eff"}

_OUTPUT_ALIASES: Mapping[str, str] = {
    "t": "T",
    "p": "P",
    "q": "Q",
    "x": "Q",
    "h": "Hmass",
    "s": "Smass",
    "u": "Umass",
    "rho": "Dmass",
    "d": "Dmass",
    "cp": "Cpmass",
    "cv": "Cvmass",
    "a": "A",
    "mu": "V",
    "visc": "V",
    "k": "L",
    "lambda": "L",
    "cond": "L",
    "pr": "Prandtl",
    # derived / postprocessed
    "v": "v",
    "gamma": "gamma",
    "r_eff": "R_eff",
    "reff": "R_eff",
}

# ------------------------------ small helpers ------------------------------


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _as_float(v: Any, key: str) -> float:
    try:
        x = float(v)
    except Exception as e:
        raise TypeError(f"{key} must be a number; got {type(v).__name__}") from e
    if not _is_finite(x):
        raise ValueError(f"{key} must be finite; got {x!r}")
    return x


def _normalize_key(key: str) -> str:
    """
    Normalize input key:
    - exact match in _UNIT_KEYMAP
    - then try a couple case-normalizations for unit-suffixed keys
    - then apply EES-ish aliasing for bare keys (t,p,h,s,u,rho,v,x)
    """
    k = str(key).strip()
    if not k:
        raise ValueError("Empty key in state specification.")

    if k in _UNIT_KEYMAP:
        return k

    # common user inputs: p_bar, t_c, T_c, etc.
    k2 = (k[0].upper() + k[1:]) if k else k
    if k2 in _UNIT_KEYMAP:
        return k2

    k3 = k.upper()
    if k3 in _UNIT_KEYMAP:
        return k3

    kl = k.lower()
    if kl in _CANONICAL_IN_ALIASES:
        return _CANONICAL_IN_ALIASES[kl]

    return k


def _decode_input_item(key: str, value: Any) -> tuple[str, float]:
    """
    Convert a user-facing (key,value) into a CoolProp input key and SI value.

    Supports:
    - Unit-suffixed inputs: T_C, P_bar, h_kJkg, s_kJkgK, ...
    - EES-ish aliases: T,P,h,s,u,rho,v,x
    - Specific volume keys: v_m3kg or v (interpreted as m^3/kg) -> density inversion
    """
    k0 = _normalize_key(key)
    v0 = _as_float(value, key)

    # Specific volume -> density inversion
    if k0 in ("v_m3kg",):
        if v0 <= 0:
            raise ValueError(f"{key} must be > 0 (specific volume). Got {v0}")
        return "Dmass", 1.0 / v0

    # If alias normalized to v_m3kg already, handle it too
    if k0 == "v_m3kg":
        if v0 <= 0:
            raise ValueError(f"{key} must be > 0 (specific volume). Got {v0}")
        return "Dmass", 1.0 / v0

    # Linear transform keys
    if k0 in _UNIT_KEYMAP:
        canon, factor, offset = _UNIT_KEYMAP[k0]
        # v_m3kg handled above; here it's linear
        return canon, v0 * factor + offset

    # Canonical allowed keys
    if k0 in _ALLOWED_IN:
        return k0, v0

    raise ValueError(
        f"Unsupported state input key {key!r}. "
        f"Allowed: {sorted(_ALLOWED_IN)} plus unit keys like T_C, P_bar, h_kJkg, s_kJkgK, rho_kgm3, v_m3kg, x."
    )


def _pick_two_inputs(mapping: Mapping[str, Any]) -> tuple[str, float, str, float]:
    """
    Select exactly two independent thermodynamic_properties from a mapping.

    Rules:
    - Ignore metadata keys (_IGNORE_KEYS).
    - Ignore keys whose value is None.
    - If more than two thermo keys are present, raise.
    """
    items: list[tuple[str, Any]] = []
    for k, v in mapping.items():
        if k in _IGNORE_KEYS:
            continue
        if v is None:
            continue
        items.append((k, v))

    if len(items) != 2:
        keys = [k for k, _ in items]
        raise ValueError(
            "State requires exactly 2 independent thermodynamic_properties; "
            f"got {len(items)} keys: {keys}"
        )

    (k1, v1), (k2, v2) = items
    in1, x1 = _decode_input_item(k1, v1)
    in2, x2 = _decode_input_item(k2, v2)

    if in1 == in2:
        raise ValueError(f"State inputs must be distinct; got {in1!r} twice.")
    return in1, x1, in2, x2


def _normalize_output_key(k: Any) -> str:
    s = str(k).strip()
    if not s:
        return s
    if s in _DERIVED_OUTPUTS:
        return s
    sl = s.lower()
    if sl in _OUTPUT_ALIASES:
        return _OUTPUT_ALIASES[sl]
    return s


def _normalize_outputs(outputs: Sequence[str]) -> tuple[list[str], set[str]]:
    """
    Normalize user-facing output keys to CoolProp keys, tracking derived outputs.

    - If user requests 'v' -> ensure 'Dmass' is computed (postprocess creates v).
    - If user requests 'gamma' or 'R_eff' -> ensure Cp/Cv are computed.
    """
    want_cp: list[str] = []
    want_derived: set[str] = set()

    for o in outputs:
        ok = _normalize_output_key(o)
        if not ok:
            continue
        if ok in _DERIVED_OUTPUTS:
            want_derived.add(ok)
            continue
        want_cp.append(ok)

    # dependencies for derived outputs
    if "v" in want_derived and "Dmass" not in want_cp:
        want_cp.append("Dmass")
    if ("gamma" in want_derived or "R_eff" in want_derived) and (
        "Cpmass" not in want_cp or "Cvmass" not in want_cp
    ):
        if "Cpmass" not in want_cp:
            want_cp.append("Cpmass")
        if "Cvmass" not in want_cp:
            want_cp.append("Cvmass")

    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in want_cp:
        if k not in seen:
            seen.add(k)
            out.append(k)

    return out, want_derived


# ------------------------------ state object ------------------------------


@dataclass(frozen=True)
class ThermoState:
    """
    A thermodynamic state evaluated by CoolProp.

    `props` contains CoolProp-native keys in SI units, plus derived:
      - "v"      : specific volume (m^3/kg)
      - "gamma"  : Cp/Cv (if available)
      - "R_eff"  : Cp - Cv (diagnostic; not constant for real fluids)
    """

    fluid: str
    in1: str
    v1: float
    in2: str
    v2: float
    props: dict[str, float]
    phase: str | None = None

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.props.get(key, default)

    @property
    def T(self) -> float:
        return float(self.props["T"])

    @property
    def P(self) -> float:
        return float(self.props["P"])

    @property
    def h(self) -> float:
        return float(self.props["Hmass"])

    @property
    def s(self) -> float:
        return float(self.props["Smass"])

    @property
    def rho(self) -> float:
        return float(self.props["Dmass"])

    @property
    def v(self) -> float:
        return float(self.props["v"])


# ------------------------------ builders ------------------------------


def _postprocess(props: dict[str, float]) -> dict[str, float]:
    out = dict(props)

    # specific volume
    D = out.get("Dmass", float("nan"))
    if _is_finite(D) and D > 0:
        out["v"] = 1.0 / D
    else:
        out["v"] = float("nan")

    # gamma and R_eff (diagnostics)
    cp = out.get("Cpmass", float("nan"))
    cv = out.get("Cvmass", float("nan"))
    if _is_finite(cp) and _is_finite(cv) and cv != 0.0:
        out["gamma"] = cp / cv
        out["R_eff"] = cp - cv
    else:
        out["gamma"] = float("nan")
        out["R_eff"] = float("nan")

    return out


@with_error_context("thermo_props.state_from_pair")
def state_from_pair(
    fluid: str,
    in1: str,
    v1: float,
    in2: str,
    v2: float,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    include_phase: bool = True,
) -> ThermoState:
    """
    Build a state from two independent thermodynamic_properties (already in SI units / CoolProp keys).

    Example:
        state_from_pair("R134a", "T", 273.15, "Q", 1.0)
        state_from_pair("HEOS::Air", "T", 300.0, "P", 101325.0)
    """
    if not isinstance(fluid, str) or not fluid.strip():
        raise ValueError("fluid must be a non-empty string.")
    if in1 not in _ALLOWED_IN or in2 not in _ALLOWED_IN:
        raise ValueError(f"Inputs must be in {sorted(_ALLOWED_IN)}; got {in1!r}, {in2!r}")
    if not _is_finite(v1) or not _is_finite(v2):
        raise ValueError("Input values must be finite.")
    if in1 == in2:
        raise ValueError("in1 and in2 must be distinct.")

    outs_cp, _derived = _normalize_outputs(outputs)

    # Compute thermodynamic_properties
    props = props_multi(list(outs_cp), in1, float(v1), in2, float(v2), fluid)
    props2 = _postprocess(props)

    ph: str | None = None
    if include_phase:
        try:
            ph = phase_si(in1, float(v1), in2, float(v2), fluid)
        except Exception:
            ph = None

    return ThermoState(
        fluid=fluid,
        in1=in1,
        v1=float(v1),
        in2=in2,
        v2=float(v2),
        props=props2,
        phase=ph,
    )


@with_error_context("thermo_props.state_from_mapping")
def state_from_mapping(
    mapping: Mapping[str, Any],
    *,
    fluid: str | None = None,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    include_phase: bool = True,
) -> ThermoState:
    """
    Build a state from a mapping that contains exactly two independent thermodynamic_properties.

    Accepts keys like:
      - {"T_C": -10, "x": 1.0}
      - {"P_bar": 10, "h_kJkg": 240}
      - {"T": 300, "P": 101325}
      - {"rho_kgm3": 12.3, "T_K": 280}
      - {"v_m3kg": 0.0012, "P_bar": 10}  # specific volume

    You may supply `fluid=` or include "fluid" in the mapping.
    """
    f = (fluid or str(mapping.get("fluid", "")).strip()).strip()
    if not f:
        raise ValueError("Fluid must be provided (argument `fluid=` or mapping['fluid']).")

    in1, v1, in2, v2 = _pick_two_inputs(mapping)
    return state_from_pair(
        f,
        in1,
        v1,
        in2,
        v2,
        outputs=outputs,
        include_phase=include_phase,
    )


# ------------------------------ convenience constructors ------------------------------


def state_TP(fluid: str, T_K: float, P_Pa: float, *, outputs: Sequence[str] = DEFAULT_OUTPUTS) -> ThermoState:
    return state_from_pair(fluid, "T", float(T_K), "P", float(P_Pa), outputs=outputs)


def state_Tx(fluid: str, T_K: float, x: float, *, outputs: Sequence[str] = DEFAULT_OUTPUTS) -> ThermoState:
    return state_from_pair(fluid, "T", float(T_K), "Q", float(x), outputs=outputs)


def state_Px(fluid: str, P_Pa: float, x: float, *, outputs: Sequence[str] = DEFAULT_OUTPUTS) -> ThermoState:
    return state_from_pair(fluid, "P", float(P_Pa), "Q", float(x), outputs=outputs)


def state_Ph(fluid: str, P_Pa: float, h_Jkg: float, *, outputs: Sequence[str] = DEFAULT_OUTPUTS) -> ThermoState:
    return state_from_pair(fluid, "P", float(P_Pa), "Hmass", float(h_Jkg), outputs=outputs)


def state_Ps(fluid: str, P_Pa: float, s_JkgK: float, *, outputs: Sequence[str] = DEFAULT_OUTPUTS) -> ThermoState:
    return state_from_pair(fluid, "P", float(P_Pa), "Smass", float(s_JkgK), outputs=outputs)


def state_Th(fluid: str, T_K: float, h_Jkg: float, *, outputs: Sequence[str] = DEFAULT_OUTPUTS) -> ThermoState:
    return state_from_pair(fluid, "T", float(T_K), "Hmass", float(h_Jkg), outputs=outputs)


# ------------------------------ single-property helper ------------------------------


@with_error_context("thermo_props.prop_from_state")
def prop_from_state(
    fluid: str,
    out: str,
    in1: str,
    v1: float,
    in2: str,
    v2: float,
) -> float:
    """
    Tiny helper for solvers that want *one* property without building a full ThermoState.

    Note:
    - `out` may be a CoolProp key (e.g., "Hmass") or a common alias ("h","s","rho",...).
    - Derived outputs ("v","gamma","R_eff") are not supported here; use state_from_pair().
    """
    out_key = _normalize_output_key(out)
    if out_key in _DERIVED_OUTPUTS:
        raise ValueError(f"Derived output {out!r} is not supported in prop_from_state(); use state_from_pair().")
    return props_si(out_key, in1, float(v1), in2, float(v2), fluid)

__all__ = [
    "DEFAULT_OUTPUTS",
    "ThermoState",
    "state_from_mapping",
    "state_from_pair",
    "prop_from_state",
]
