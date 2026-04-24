from __future__ import annotations

"""
thermo_props.api

High-level, EES-ish API surface for thermodynamic property evaluation.

LATEST FACTS / CONSTRAINTS
--------------------------
- app routes problem_type == "thermo_props" to thermo_props.api.run(spec) (preferred)
  with fallback to thermo_props.api.eval_states(spec).
- design.build_thermo_props currently builds an app-facing spec like:

    {
      "backend": "coolprop",
      "fluid": "R134a",
      "states": [
        {"id": "1", "given": {"T_C": -10, "x": 1.0}, "ask": ["P","Hmass"]},
        {"id": "2", "given": {"P_bar": 10, "h_kJkg": 240}, "ask": ["T","Smass"]},
      ],
      "meta": {...}
    }

- This module must therefore:
  * provide run(spec) that understands the above shape (plus some backward-compat shapes),
  * remain thin and stable,
  * keep the CoolProp dependency isolated in coolprop_backend.py,
  * keep robust state construction in state.py.

Public surface:
- state(...) -> ThermoState (computed props in SI)
- prop(...)  -> single property in SI
- props(...) -> property dict in SI
- run(spec)  -> app-facing facade for JSON-driven evaluation (batch or single)
- saturation/isobar helpers for Plotly overlays
"""

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from utils import with_error_context

from .coolprop_backend import CoolPropCallError, props_si
from .state import DEFAULT_OUTPUTS, ThermoState, state_from_mapping


# ------------------------------ primary API ------------------------------

@with_error_context("thermo_props.api.state")
def state(
    *,
    fluid: str | None = None,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    include_phase: bool = True,
    **spec: Any,
) -> ThermoState:
    """
    Build a thermodynamic state from exactly two independent thermodynamic_properties.

    Examples:
        st = state(fluid="R134a", T_C=-10, x=1.0)
        st = state(fluid="HEOS::Air", T=300.0, P_bar=1.01325)

    `spec` must contain exactly two thermo keys (aside from metadata like name/label/id/etc).
    Keys may be EES-ish: T, P, h, s, u, rho, v, x, Q
    Or unit-suffixed: T_C, P_bar, h_kJkg, s_kJkgK, rho_kgm3, v_m3kg, ...
    """
    mapping: dict[str, Any] = dict(spec)
    if fluid is not None:
        mapping["fluid"] = fluid
    return state_from_mapping(mapping, outputs=outputs, include_phase=include_phase)


@with_error_context("thermo_props.api.props")
def props(
    *,
    fluid: str | None = None,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    include_phase: bool = True,
    **spec: Any,
) -> dict[str, float]:
    """
    Return a property dictionary in SI units for a state defined by two thermodynamic_properties.

    Example:
        d = props(fluid="R134a", T_C=35, x=0.0, outputs=("P","Hmass","Smass"))
    """
    st = state(fluid=fluid, outputs=outputs, include_phase=include_phase, **spec)
    return dict(st.props)


@with_error_context("thermo_props.api.prop")
def prop(
    out: str,
    *,
    fluid: str | None = None,
    include_phase: bool = False,
    **spec: Any,
) -> float:
    """
    Return a single property in SI units for a state defined by two thermodynamic_properties.

    Example:
        P = prop("P", fluid="R134a", T_C=35, x=0.0)
        h = prop("Hmass", fluid="HEOS::Air", T=300, P=101325)
    """
    st = state(fluid=fluid, outputs=(out,), include_phase=include_phase, **spec)
    if out not in st.props:
        raise ValueError(f"Property {out!r} was not computed. Available: {sorted(st.props.keys())}")
    return float(st.props[out])


# ------------------------------ app-facing facade ------------------------------

def _as_mapping(x: Any, what: str) -> Mapping[str, Any]:
    if isinstance(x, Mapping):
        return x
    # allow dataclass-ish objects with __dict__
    d = getattr(x, "__dict__", None)
    if isinstance(d, dict):
        return d
    raise TypeError(f"{what} must be a mapping-like object; got {type(x).__name__}.")


def _coerce_outputs(v: Any) -> Sequence[str]:
    if v is None:
        return DEFAULT_OUTPUTS
    if isinstance(v, (list, tuple)):
        out = tuple(str(x) for x in v if str(x).strip())
        return out if out else DEFAULT_OUTPUTS
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return tuple(parts) if parts else DEFAULT_OUTPUTS
    return DEFAULT_OUTPUTS


def _is_given_style_state(m: Mapping[str, Any]) -> bool:
    # design.py emits {"given": {...}, "ask": [...]}
    g = m.get("given", None)
    return isinstance(g, Mapping)


def _eval_one_state(
    state_item: Mapping[str, Any],
    *,
    i: int,
    fluid_default: str | None,
    outputs_default: Sequence[str] | None,
    include_phase: bool,
) -> dict[str, Any]:
    # allow per-state fluid override, else inherit root fluid
    fluid = state_item.get("fluid", None)
    fluid = str(fluid) if fluid is not None else (str(fluid_default) if fluid_default is not None else None)
    if fluid is None:
        raise ValueError("thermo_props: missing 'fluid' (provide at root or per-state).")

    if _is_given_style_state(state_item):
        given_map = _as_mapping(state_item.get("given"), f"states[{i}].given")
        ask = state_item.get("ask", None)
        outputs = _coerce_outputs(ask) if ask is not None else (outputs_default or DEFAULT_OUTPUTS)
        st = state_from_mapping(
            dict(given_map),
            fluid=fluid,
            outputs=outputs,
            include_phase=include_phase,
        )
        return {
            "index": i,
            "id": state_item.get("id", None),
            "name": state_item.get("name", None),
            "label": state_item.get("label", None),
            "fluid": st.fluid,
            "given": dict(given_map),
            "outputs": list(outputs),
            "in1": st.in1,
            "v1": st.v1,
            "in2": st.in2,
            "v2": st.v2,
            "phase": st.phase,
            "props": dict(st.props),
        }

    # Back-compat: allow state defined directly by two keys at the item level
    outputs = outputs_default or DEFAULT_OUTPUTS
    st = state_from_mapping(
        dict(state_item),
        fluid=fluid,
        outputs=outputs,
        include_phase=include_phase,
    )
    return {
        "index": i,
        "id": state_item.get("id", None),
        "name": state_item.get("name", None),
        "label": state_item.get("label", None),
        "fluid": st.fluid,
        "given": {k: v for k, v in dict(state_item).items() if k not in ("id", "name", "label", "fluid", "ask", "outputs", "backend", "meta")},
        "outputs": list(outputs),
        "in1": st.in1,
        "v1": st.v1,
        "in2": st.in2,
        "v2": st.v2,
        "phase": st.phase,
        "props": dict(st.props),
    }


@with_error_context("thermo_props.api.run")
def run(spec: Any) -> dict[str, Any]:
    """
    App-oriented entry point.

    Primary (current) shape produced by design.build_thermo_props:

      {
        "backend": "coolprop",
        "fluid": "R134a",
        "states": [
          {"id":"1","given":{"T_C":-10,"x":1.0},"ask":["P","Hmass"]},
          {"id":"2","given":{"P_bar":10,"h_kJkg":240},"ask":["T","Smass"]}
        ],
        "meta": {...}
      }

    Supported (back-compat) shapes:
      - {"fluid": "...", "state": {...}, "outputs":[...]}
      - {"fluid": "...", "states": [...], "outputs":[...]}  (where each state is direct two-key mapping)
      - {"fluid":"...", "given": {...}, "ask":[...]}  (single-state shorthand)
      - {"fluid":"...", "T":..., "P":..., "outputs":[...]} (direct two-key top-level)

    Returns a JSON-serializable payload.
    """
    m = _as_mapping(spec, "thermo_props spec")

    backend = str(m.get("backend", "coolprop"))
    fluid = m.get("fluid", None)
    fluid = str(fluid) if fluid is not None else None

    include_phase = bool(m.get("include_phase", True))

    # If provided, these apply as defaults (per-state 'ask' can override).
    outputs_default: Sequence[str] | None = None
    if "outputs" in m and m.get("outputs") is not None:
        outputs_default = _coerce_outputs(m.get("outputs"))

    # ---- Single-state: explicit "state" mapping (older)
    if "state" in m and m.get("state") is not None:
        st_map = _as_mapping(m["state"], "spec['state']")
        st = state_from_mapping(
            dict(st_map),
            fluid=fluid,
            outputs=outputs_default or DEFAULT_OUTPUTS,
            include_phase=include_phase,
        )
        return {
            "mode": "single",
            "backend": backend,
            "fluid": st.fluid,
            "in1": st.in1,
            "v1": st.v1,
            "in2": st.in2,
            "v2": st.v2,
            "phase": st.phase,
            "props": dict(st.props),
        }

    # ---- Single-state: design-style shorthand at root {"given": {...}, "ask": [...]}
    if "given" in m and isinstance(m.get("given"), Mapping):
        given_map = _as_mapping(m["given"], "spec['given']")
        ask = m.get("ask", None)
        outputs = _coerce_outputs(ask) if ask is not None else (outputs_default or DEFAULT_OUTPUTS)
        if fluid is None:
            raise ValueError("thermo_props: missing 'fluid' (provide at root).")
        st = state_from_mapping(
            dict(given_map),
            fluid=fluid,
            outputs=outputs,
            include_phase=include_phase,
        )
        return {
            "mode": "single",
            "backend": backend,
            "fluid": st.fluid,
            "in1": st.in1,
            "v1": st.v1,
            "in2": st.in2,
            "v2": st.v2,
            "phase": st.phase,
            "props": dict(st.props),
        }

    # ---- Batch: "states" list (preferred; design.py emits this)
    if "states" in m and m.get("states") is not None:
        items = m["states"]
        if not isinstance(items, (list, tuple)):
            raise TypeError("spec['states'] must be a list of state mappings.")

        out_states: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            st_map = _as_mapping(item, f"spec['states'][{i}]")
            out_states.append(
                _eval_one_state(
                    st_map,
                    i=i,
                    fluid_default=fluid,
                    outputs_default=outputs_default,
                    include_phase=include_phase,
                )
            )

        return {
            "mode": "batch",
            "backend": backend,
            "fluid": fluid,
            "states": out_states,
            "meta": dict(m.get("meta", {}) or {}) if isinstance(m.get("meta", {}) or {}, Mapping) else {},
        }

    # ---- Compatibility: attempt to interpret the top-level mapping directly as a two-key state
    try:
        st = state_from_mapping(
            dict(m),
            fluid=fluid,
            outputs=outputs_default or DEFAULT_OUTPUTS,
            include_phase=include_phase,
        )
        return {
            "mode": "single",
            "backend": backend,
            "fluid": st.fluid,
            "in1": st.in1,
            "v1": st.v1,
            "in2": st.in2,
            "v2": st.v2,
            "phase": st.phase,
            "props": dict(st.props),
        }
    except Exception as e:
        raise ValueError(
            "thermo_props.api.run(spec) could not interpret the input. "
            "Preferred: {'fluid':..., 'states':[{'given':{...}, 'ask':[...]} ...]} "
            "or single-state {'fluid':..., 'given':{...}, 'ask':[...]}."
        ) from e


# Back-compat alias (EespyApp will fall back to this name if run() isn't found)
eval_states = run


# ------------------------------ serialization helpers ------------------------------

def state_to_dict(st: ThermoState) -> dict[str, Any]:
    """
    JSON-friendly representation for saving results.

    Notes:
    - All values are in SI units (CoolProp-native) except derived 'v' (m^3/kg).
    - `phase` is included if available.
    """
    return asdict(st)


# ------------------------------ overlay helpers (Plotly-friendly) ------------------------------

@with_error_context("thermo_props.api.saturation_curve_Ts")
def saturation_curve_Ts(
    fluid: str,
    *,
    n: int = 200,
    T_min_K: float | None = None,
    T_max_K: float | None = None,
) -> dict[str, list[float]]:
    """
    Return saturation dome curves for a pure fluid in T–s space.

    Output keys:
      - "T_K"
      - "sL_JkgK" (sat. liquid)
      - "sV_JkgK" (sat. vapor)

    Notes:
    - Uses CoolProp saturation evaluation via (T, Q).
    - For fluids without a saturation curve (or mixtures), CoolProp may error.
    """
    if n < 10:
        raise ValueError("n must be >= 10")

    if T_min_K is None or T_max_K is None:
        T_min_K2, T_max_K2 = _find_sat_T_bounds(fluid)
        T_min_K = T_min_K if T_min_K is not None else T_min_K2
        T_max_K = T_max_K if T_max_K is not None else T_max_K2

    if not (T_min_K and T_max_K and T_max_K > T_min_K):
        raise CoolPropCallError("Invalid saturation temperature bounds.")

    Ts = [T_min_K + (T_max_K - T_min_K) * i / (n - 1) for i in range(n)]

    T_out: list[float] = []
    sL: list[float] = []
    sV: list[float] = []

    for T in Ts:
        try:
            s_liq = props_si("Smass", "T", float(T), "Q", 0.0, fluid)
            s_vap = props_si("Smass", "T", float(T), "Q", 1.0, fluid)
        except Exception:
            continue
        T_out.append(float(T))
        sL.append(float(s_liq))
        sV.append(float(s_vap))

    if len(T_out) < 5:
        raise CoolPropCallError(
            "Failed to build saturation curve (insufficient points). "
            "This fluid may not support saturation via CoolProp."
        )

    return {"T_K": T_out, "sL_JkgK": sL, "sV_JkgK": sV}


@with_error_context("thermo_props.api.isobars_Ts")
def isobars_Ts(
    fluid: str,
    pressures_Pa: Sequence[float],
    *,
    nT: int = 80,
    T_min_K: float | None = None,
    T_max_K: float | None = None,
) -> list[dict[str, Any]]:
    """
    Generate T–s isobars for overlay (like Klein & Nellis style backgrounds).

    Returns a list of traces (data dicts) suitable for Plotly:
      [{"P_Pa":..., "T_K":[...], "s_JkgK":[...]} ...]
    """
    if nT < 10:
        raise ValueError("nT must be >= 10")
    if not pressures_Pa:
        return []

    if T_min_K is None or T_max_K is None:
        try:
            Tmin, Tmax = _find_sat_T_bounds(fluid)
            T_min_K = T_min_K if T_min_K is not None else Tmin
            T_max_K = T_max_K if T_max_K is not None else Tmax
        except Exception:
            T_min_K = T_min_K if T_min_K is not None else 200.0
            T_max_K = T_max_K if T_max_K is not None else 600.0

    if not (T_min_K and T_max_K and T_max_K > T_min_K):
        raise CoolPropCallError("Invalid isobar temperature bounds.")

    traces: list[dict[str, Any]] = []
    Ts = [T_min_K + (T_max_K - T_min_K) * i / (nT - 1) for i in range(nT)]

    for P in pressures_Pa:
        P = float(P)
        T_out: list[float] = []
        s_out: list[float] = []
        for T in Ts:
            try:
                s = props_si("Smass", "T", float(T), "P", P, fluid)
            except Exception:
                continue
            T_out.append(float(T))
            s_out.append(float(s))
        if len(T_out) >= 5:
            traces.append({"P_Pa": P, "T_K": T_out, "s_JkgK": s_out})

    return traces


# ------------------------------ internal helpers ------------------------------

def _safe_sat_probe(fluid: str) -> bool:
    try:
        _ = props_si("P", "T", 300.0, "Q", 0.0, fluid)
        _ = props_si("P", "T", 300.0, "Q", 1.0, fluid)
        return True
    except Exception:
        return False


def _find_sat_T_bounds(fluid: str) -> tuple[float, float]:
    """
    Best-effort scan for saturation temperature bounds supported by CoolProp via (T,Q) calls.
    """
    if not _safe_sat_probe(fluid):
        raise CoolPropCallError("Fluid does not appear to support saturation via (T,Q) calls.")

    Tmin = 300.0
    for _ in range(80):
        try:
            _ = props_si("P", "T", Tmin, "Q", 0.0, fluid)
            _ = props_si("P", "T", Tmin, "Q", 1.0, fluid)
            Tmin *= 0.98
        except Exception:
            Tmin = Tmin / 0.98
            break

    Tmax = 300.0
    last_ok = Tmax
    for _ in range(160):
        try:
            _ = props_si("P", "T", Tmax, "Q", 0.0, fluid)
            _ = props_si("P", "T", Tmax, "Q", 1.0, fluid)
            last_ok = Tmax
            Tmax *= 1.02
        except Exception:
            break

    Tmax = max(last_ok * 0.999, Tmin + 1.0)
    return float(Tmin), float(Tmax)
