from __future__ import annotations

"""
thermo_props.core

Facade module expected by `thermo_props.__init__`.

This file provides:
- coolprop_available() -> bool
- State dataclass alias (public name) + ThermoState import
- Props service -> build thermodynamic states + query thermodynamic properties (CoolProp-backed)

Latest facts (your codebase):
- `thermo_props.state` defines `ThermoState` (NOT `State`).
- `thermo_props.__init__` historically imported:
      from .core import Props, State, coolprop_available
  So this module must export `State`.
  We do: State = ThermoState (alias) and also export ThermoState.

Design notes:
- Keep imports safe if CoolProp isn't installed (coolprop_backend guards).
- Support two usage modes:
    (A) EES-ish mapping inputs with unit-suffixed keys (T_C, P_bar, h_kJkg, v_m3kg, ...)
        -> state_from_mapping(...) from state.py
    (B) Low-level direct calls with CoolProp-native keys (T,P,Q,Hmass,Smass,Umass,Dmass)
        -> prop_from_state(...) and/or state_from_pair(...)
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .coolprop_backend import CoolPropCallError, CoolPropNotInstalled, coolprop_available
from .state import (
    DEFAULT_OUTPUTS,
    ThermoState,
    prop_from_state,
    state_from_mapping,
    state_from_pair,
)

# Public alias expected by thermo_props/__init__.py (back-compat)
State = ThermoState

# ------------------------- output aliasing (EES-ish) -------------------------

# The ThermoState.props dict stores mostly CoolProp-native keys:
#   T, P, Q, Hmass, Smass, Umass, Dmass, Cpmass, Cvmass, A, V, L, Prandtl
# plus derived:
#   v, gamma, R_eff
#
# Allow requesting either EES-ish names ("h", "s", "rho", "cp", ...) or the stored keys.
_OUT_ALIASES: Dict[str, str] = {
    # core thermo
    "T": "T",
    "P": "P",
    "x": "Q",
    "Q": "Q",
    "h": "Hmass",
    "H": "Hmass",
    "s": "Smass",
    "S": "Smass",
    "u": "Umass",
    "U": "Umass",
    "rho": "Dmass",
    "D": "Dmass",
    # convenience / derived
    "v": "v",  # derived m^3/kg
    "gamma": "gamma",
    "R_eff": "R_eff",
    # caloric / acoustic / transport
    "cp": "Cpmass",
    "cv": "Cvmass",
    "a": "A",
    "mu": "V",        # viscosity key in DEFAULT_OUTPUTS is "V"
    "k": "L",         # thermal conductivity key in DEFAULT_OUTPUTS is "L"
    "Pr": "Prandtl",
}

_DEFAULT_OUTPUTS: Tuple[str, ...] = tuple(DEFAULT_OUTPUTS)


def _norm_out_key(k: str) -> str:
    kk = str(k).strip()
    if not kk:
        raise ValueError("Empty output key.")
    if kk in _OUT_ALIASES:
        return _OUT_ALIASES[kk]
    lk = kk.lower()
    if lk in _OUT_ALIASES:
        return _OUT_ALIASES[lk]
    # Allow requesting a stored CoolProp key directly, e.g. "Hmass", "Cpmass", "Prandtl"
    return kk


def _require_coolprop(where: str) -> None:
    """
    Raise a consistent, user-friendly error if CoolProp isn't available.

    We intentionally raise CoolPropNotInstalled (subclass of ImportError) so callers
    can catch it as ImportError in CLI/GUI flows.
    """
    if coolprop_available():
        return
    raise CoolPropNotInstalled(
        f"CoolProp is not available, but {where} was called. "
        "Install it with: pip install CoolProp"
    )


# ------------------------- Props service -------------------------

class Props:
    """
    Convenience thermo property service (CoolProp-backed).

    This class delegates "state building" to `state.py` to avoid duplicating
    normalization/unit-conversion logic.

    Typical use:
        props = Props()
        st = props.state("R134a", {"T_C": -10, "x": 1.0})
        print(st.h)              # J/kg
        print(st.props["Hmass"]) # same
        print(props.get("cp", "R134a", {"T_C": -10, "x": 1.0}))

    Notes:
    - `inputs` must specify exactly TWO independent thermodynamic_properties (state.py enforces this).
    - Larger sets of equations belong in `equations`.
    """

    def __init__(
        self,
        default_outputs: Sequence[str] = _DEFAULT_OUTPUTS,
        *,
        include_phase: bool = True,
    ) -> None:
        self.default_outputs = tuple(str(x) for x in default_outputs)
        self.include_phase = bool(include_phase)

    # ---- state builders ----

    def state(
        self,
        fluid: str,
        inputs: Mapping[str, Any],
        *,
        outputs: Optional[Sequence[str]] = None,
        include_phase: Optional[bool] = None,
    ) -> ThermoState:
        """
        Build a ThermoState from a mapping containing exactly two independent inputs.

        `inputs` may contain unit-suffixed keys such as:
          - T_C, T_K
          - P_bar, P_kPa, P_MPa, P_atm
          - h_kJkg, s_kJkgK, u_kJkg
          - rho_kgm3, v_m3kg
          - x or Q (quality)
        """
        _require_coolprop("thermo_props.Props.state(...)")

        outs = tuple(outputs) if outputs is not None else self.default_outputs
        inc_phase = self.include_phase if include_phase is None else bool(include_phase)

        return state_from_mapping(
            dict(inputs),
            fluid=str(fluid),
            outputs=outs,
            include_phase=inc_phase,
        )

    def state_from_pair(
        self,
        fluid: str,
        in1: str,
        v1: float,
        in2: str,
        v2: float,
        *,
        outputs: Optional[Sequence[str]] = None,
        include_phase: Optional[bool] = None,
    ) -> ThermoState:
        """
        Build a ThermoState from two CoolProp-native inputs (already SI).

        Example:
            props.state_from_pair("R134a", "T", 263.15, "Q", 1.0)
        """
        _require_coolprop("thermo_props.Props.state_from_pair(...)")

        outs = tuple(outputs) if outputs is not None else self.default_outputs
        inc_phase = self.include_phase if include_phase is None else bool(include_phase)

        return state_from_pair(
            fluid=str(fluid),
            in1=str(in1),
            v1=float(v1),
            in2=str(in2),
            v2=float(v2),
            outputs=outs,
            include_phase=inc_phase,
        )

    # ---- property getters ----

    def get(
        self,
        out_key: str,
        fluid: str,
        inputs: Mapping[str, Any],
        *,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """
        Get one property from a two-input mapping.

        `out_key` can be:
          - EES-ish: h, s, u, rho, v, cp, cv, a, mu, k, Pr, x
          - or stored key: Hmass, Smass, Umass, Dmass, Cpmass, Cvmass, A, V, L, Prandtl, T, P, Q
        """
        st = self.state(fluid, inputs, outputs=self.default_outputs, include_phase=False)
        k = _norm_out_key(out_key)
        return st.props.get(k, default)

    def get_many(
        self,
        out_keys: Iterable[str],
        fluid: str,
        inputs: Mapping[str, Any],
        *,
        default: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Get multiple thermodynamic_properties from a two-input mapping.
        Returns dict keyed by the ORIGINAL requested keys.
        """
        st = self.state(fluid, inputs, outputs=self.default_outputs, include_phase=False)
        out: Dict[str, Optional[float]] = {}
        for k_req in out_keys:
            k_req_s = str(k_req)
            k = _norm_out_key(k_req_s)
            out[k_req_s] = st.props.get(k, default)
        return out

    # ---- fast single-call helper (CoolProp-native only) ----

    def get_si_from_pair(
        self,
        out: str,
        fluid: str,
        in1: str,
        v1: float,
        in2: str,
        v2: float,
    ) -> float:
        """
        Fast path: one property from two SI inputs using CoolProp-native tokens.

        Example:
            props.get_si_from_pair("Hmass", "R134a", "T", 263.15, "Q", 1.0)
        """
        _require_coolprop("thermo_props.Props.get_si_from_pair(...)")
        return float(prop_from_state(str(fluid), str(out), str(in1), float(v1), str(in2), float(v2)))

__all__ = [
    "Props",
    "State",
    "ThermoState",
    "DEFAULT_OUTPUTS",
    "CoolPropCallError",
    "CoolPropNotInstalled",
    "coolprop_available",
]
