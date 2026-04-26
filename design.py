# design.py
from __future__ import annotations

"""Problem-specification builders for TDPy.

This module converts input mappings from JSON, YAML, or TXT files into the
typed or mapping-like objects consumed by the application service.

Design goals
------------
The builders are intentionally lightweight and conservative:

* File paths are resolved relative to the input file directory.
* GUI and CLI flows share the same builder layer.
* Heavy optional dependencies such as SciPy, GEKKO, and CoolProp are not
  imported here.
* Existing equation-system behavior is preserved for older input files.

Supported problem types
-----------------------
``"nozzle_ideal"``
    Builds a ``NozzleProfileSpec`` for the ideal-gas nozzle solver.

``"thermo_props"``
    Builds a normalized mapping consumed by the thermodynamic-property API.

``"equations"``
    Builds a stable mapping consumed by ``equations.api.solve_system``.

``"optimize"``
    Builds an optimization mapping routed through the equations/optimizer
    facade.

Compatibility notes
-------------------
Existing equation pipelines rely on ``build_equations`` returning a mapping
without adding a top-level ``"solve"`` key. Some GUI-generated inputs may
contain a nested ``"solve"`` block; the equations builder stores that
information in ``meta`` for debugging but keeps the stable top-level shape.

Optimization inputs may interpret nested solve settings because they do not go
through the older ``equations.spec.system_from_mapping`` path in the same way.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from apis import ProblemSpec
from core import NozzleProfileSpec
from in_out import load_geometry_csv
from utils import with_error_context

# Units parsing is optional; if present, we can parse "300 K", "14.7 psi", etc.
try:
    from units import parse_quantity  # type: ignore
except Exception:  # pragma: no cover
    parse_quantity = None  # type: ignore


# ------------------------------ small helpers ------------------------------


class AttrDict(dict):
    """Dictionary that also supports attribute access.

    This is useful because some application code uses ``getattr(spec, ...)``
    while backend code often expects mapping-like payloads.
    """

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
            return
        raise AttributeError(name)


def _resolve_path(p: str | Path, base_dir: Path) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return pp


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key!r}")
    return mapping[key]


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _strip_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _clean_equation_line(raw: str) -> Optional[str]:
    """Normalize one equation line from a text file.

    Blank lines and comment-only lines are ignored. Inline comments introduced
    by ``#``, ``!``, or ``//`` are stripped.
    """
    line = (raw or "").strip()
    if not line:
        return None
    if line.startswith("#") or line.startswith("!"):
        return None
    if "//" in line:
        line = line.split("//", 1)[0].strip()
    if "#" in line:
        line = line.split("#", 1)[0].strip()
    if "!" in line:
        line = line.split("!", 1)[0].strip()
    return line or None


def _read_equations_lines(path: Path) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = _clean_equation_line(raw)
        if s:
            out.append(s)
    return out


def _coerce_float(x: Any, *, default_unit: str | None = None, to_unit: str | None = None) -> float:
    """Convert a number-like value to ``float``.

    Accepted values include integers, floats, numeric strings, and unit strings
    such as ``"300 K"``, ``"14.7 psi"``, or ``"70[kJ/kg]"`` when the optional
    unit parser is available.
    """
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("Empty string cannot be converted to float.")

        if parse_quantity is not None:
            q = parse_quantity(s, default_unit=default_unit, to_unit=to_unit)
            return float(q.value)

        return float(s)

    raise TypeError(f"Expected number-like (int/float/str), got {type(x).__name__}: {x!r}")


def _coerce_param_value(x: Any) -> Any:
    """Coerce a constant or parameter value.

    Numeric values, including unit-bearing numeric strings when units support is
    available, are converted to ``float``. Non-numeric strings and other
    JSON-safe values are kept as-is.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return s
        if parse_quantity is not None:
            try:
                q = parse_quantity(s, default_unit=None, to_unit=None)
                return float(q.value)
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            return s

    return x


def _normalize_backend(x: Any) -> str:
    b = str(x or "auto").strip().lower()
    if b in {"auto", "default"}:
        return "auto"
    if b in {"scipy"}:
        return "scipy"
    if b in {"gekko"}:
        return "gekko"
    return b


def _solve_dict(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a nested solve-options mapping if one is present."""
    s = data.get("solve", None)
    if isinstance(s, Mapping):
        return s
    s2 = data.get("solver", None)
    if isinstance(s2, Mapping):
        return s2
    return {}


def _pick_solve_opt(data: Mapping[str, Any], solve: Mapping[str, Any], key: str, default: Any) -> Any:
    """Prefer a top-level option and otherwise fall back to the solve block."""
    if key in data and data.get(key) is not None:
        return data.get(key)
    if key in solve and solve.get(key) is not None:
        return solve.get(key)
    return default


# ------------------------------ registry ------------------------------


_BUILDERS: Dict[str, Any] = {}


def register(problem_type: str):
    def deco(fn):
        _BUILDERS[problem_type] = fn
        return fn

    return deco


# ------------------------------ (legacy) equations duck-typed objects ------------------------------
# Kept for backwards compatibility with any local code that imported these.
# tdpy itself returns a Mapping for equations builder; these are NOT returned by build_equations().


@dataclass(frozen=True)
class EqVar:
    """Legacy equation-variable container kept for backward compatibility."""

    name: str
    kind: str  # "unknown" or "fixed"
    value: float | None
    guess: float | None = None
    lower: float | None = None
    upper: float | None = None
    units: str | None = None
    desc: str | None = None


@dataclass(frozen=True)
class EquationsSolveSpec:
    """Legacy equation-solve container kept for backward compatibility."""

    equations: List[str]
    variables: List[EqVar]
    params: Dict[str, Any] = field(default_factory=dict)

    backend: str = "auto"
    method: str = "hybr"
    tol: float = 1e-9
    max_iter: int = 200
    max_restarts: int = 2

    meta: Dict[str, Any] = field(default_factory=dict)


# ------------------------------ builders ------------------------------


@register("nozzle_ideal")
@with_error_context("build:nozzle_ideal")
def build_nozzle_ideal(data: Mapping[str, Any], base_dir: Path) -> NozzleProfileSpec:
    """Build the ideal-gas nozzle profile specification."""
    geom_path = _resolve_path(_require(data, "geometry_csv"), base_dir)
    x_mm, D_mm = load_geometry_csv(geom_path)

    k = float(data.get("k", 1.4))
    R = float(data.get("R", 287.058))

    # Backward-compatible + unit-friendly:
    # Accept T0_K / P0_Pa, but also allow T0 / P0 with units.
    T0_raw = data.get("T0_K", data.get("T0", None))
    P0_raw = data.get("P0_Pa", data.get("P0", None))
    if T0_raw is None:
        raise ValueError("Missing required key: 'T0_K' (or 'T0').")
    if P0_raw is None:
        raise ValueError("Missing required key: 'P0_Pa' (or 'P0').")

    T0 = _coerce_float(T0_raw, default_unit="K", to_unit="K")
    P0 = _coerce_float(P0_raw, default_unit="Pa", to_unit="Pa")

    branch = str(data.get("branch_after_throat", "sup")).lower()
    if branch not in ("sub", "sup"):
        raise ValueError("branch_after_throat must be 'sub' or 'sup'")

    return NozzleProfileSpec(
        k=k,
        R=R,
        T0_K=T0,
        P0_Pa=P0,
        x_mm=list(x_mm),
        D_mm=list(D_mm),
        branch_after_throat=branch,  # type: ignore[arg-type]
    )


@register("thermo_props")
@with_error_context("build:thermo_props")
def build_thermo_props(data: Mapping[str, Any], base_dir: Path) -> Mapping[str, Any]:
    """Build a normalized thermodynamic-property evaluation spec.

    Accepted input forms include a single state with ``state`` or ``given``, a
    batch list under ``states``, and a legacy ``given`` plus ``ask`` structure.
    The normalized output has ``backend``, ``fluid``, ``states``, and ``meta``.
    """
    backend = str(data.get("backend", "coolprop"))
    fluid = str(_require(data, "fluid"))

    # outputs alias: outputs -> ask
    outputs_top = [str(x) for x in _as_list(data.get("outputs", None))]
    ask_top = [str(x) for x in _as_list(data.get("ask", None))]
    ask_default = ask_top or outputs_top  # prefer explicit ask; else outputs

    def _extract_given_from_mapping(m: Mapping[str, Any]) -> Dict[str, Any]:
        # Prefer explicit given/state if present
        if "given" in m and m.get("given") is not None:
            g = m.get("given")
            if not isinstance(g, Mapping):
                raise TypeError("'given' must be a mapping.")
            return dict(g)

        if "state" in m and m.get("state") is not None:
            s = m.get("state")
            if not isinstance(s, Mapping):
                raise TypeError("'state' must be a mapping.")
            return dict(s)

        # Flat mapping fallback: strip known control keys
        skip = {
            "id",
            "name",
            "given",
            "state",
            "ask",
            "outputs",
            "backend",
            "fluid",
            "meta",
            "states",
            "states_file",
            "include_phase",
        }
        return {str(k): v for k, v in m.items() if str(k) not in skip}

    def _extract_ask(m: Mapping[str, Any]) -> List[str]:
        # state-level ask wins; else state-level outputs; else top-level ask/outputs
        a = m.get("ask", None)
        o = m.get("outputs", None)
        if a is not None:
            return [str(x) for x in _as_list(a)]
        if o is not None:
            return [str(x) for x in _as_list(o)]
        return list(ask_default)

    states: List[Dict[str, Any]] = []
    states_in = data.get("states", None)

    if states_in is None:
        # single-state
        sid = data.get("id", data.get("name", None))
        given = _extract_given_from_mapping(data)
        if not given:
            raise ValueError("thermo_props requires 'state'/'given' (or flat keys like T_C, P_bar, x, ...).")
        ask = _extract_ask(data)
        states.append(
            {
                "id": (str(sid) if sid is not None else None),
                "given": given,
                "ask": ask,
            }
        )
    else:
        if not isinstance(states_in, list):
            raise TypeError("'states' must be a list of state definitions.")
        for i, st in enumerate(states_in):
            if not isinstance(st, Mapping):
                raise TypeError(f"states[{i}] must be a mapping.")
            sid = st.get("id", st.get("name", None))
            given = _extract_given_from_mapping(st)
            if not given:
                raise ValueError(f"states[{i}] must include 'state'/'given' or flat keys (T_C, P_bar, x, ...).")
            ask = _extract_ask(st)
            states.append(
                {
                    "id": (str(sid) if sid is not None else None),
                    "given": given,
                    "ask": ask,
                }
            )

    meta = dict(data.get("meta", {}) or {})

    # Optional external file reference (kept)
    states_file = data.get("states_file", None)
    if states_file is not None:
        p = _resolve_path(states_file, base_dir)
        meta["states_file"] = str(p)

    # pass-through include_phase if supplied
    if "include_phase" in data and "include_phase" not in meta:
        meta["include_phase"] = bool(data.get("include_phase"))

    return AttrDict({"backend": backend, "fluid": fluid, "states": states, "meta": meta})


@register("equations")
@with_error_context("build:equations")
def build_equations(data: Mapping[str, Any], base_dir: Path) -> Mapping[str, Any]:
    """Build a stable equation-system mapping.

    The returned mapping is compatible with
    ``equations.spec.system_from_mapping``. It accepts the established CLI/app
    shape with solver options, ``params`` or ``constants``, a variables mapping
    or list, and equations supplied inline or through ``equations_file``.

    The builder intentionally does not add a top-level ``solve`` key. Nested
    solve information, when present, is copied into ``meta`` for user-interface
    and debugging purposes.
    """
    backend = _normalize_backend(data.get("backend", data.get("solver", "auto")))
    method = str(data.get("method", "hybr"))
    tol = float(data.get("tol", 1e-9))
    max_iter = int(data.get("max_iter", 200))
    max_restarts = int(data.get("max_restarts", 2))

    # ----- params/constants (merge) -----
    constants: Dict[str, Any] = {}

    def _merge_constants(src: Any) -> None:
        if src is None:
            return
        if not isinstance(src, Mapping):
            raise TypeError("'constants'/'params' must be a mapping.")
        for k, v in src.items():
            constants[str(k)] = _coerce_param_value(v)

    _merge_constants(data.get("params", None))
    _merge_constants(data.get("constants", None))

    # ----- variables (normalize to Mapping[str, Mapping]) -----
    vars_in = data.get("variables", data.get("vars", None))
    if vars_in is None:
        raise ValueError("Missing required key: 'variables' (or 'vars').")

    variables: Dict[str, Dict[str, Any]] = {}

    if isinstance(vars_in, Mapping):
        # allow shorthand: "x": 1.0  -> {"guess": 1.0}
        for name, payload in vars_in.items():
            nm = str(name).strip()
            if not nm:
                raise ValueError(f"Invalid variable name: {name!r}")

            if isinstance(payload, Mapping):
                variables[nm] = dict(payload)
            else:
                variables[nm] = {"guess": payload}

    elif isinstance(vars_in, list):
        # allow list form: [{"name":"x","guess":1.0}, ...]
        for i, item in enumerate(vars_in):
            if not isinstance(item, Mapping):
                raise TypeError(f"variables[{i}] must be a mapping with at least 'name'.")
            nm = str(_require(item, "name")).strip()
            if not nm:
                raise ValueError(f"variables[{i}] has invalid name.")
            d = dict(item)
            d.pop("name", None)
            variables[nm] = d
    else:
        raise TypeError("'variables' must be a mapping or a list of mappings.")

    # ----- equations: inline + file -----
    eqs: List[str] = []

    eq_file = data.get("equations_file", None)
    if eq_file is not None:
        p = _resolve_path(eq_file, base_dir)
        eqs.extend(_read_equations_lines(p))

    inline = data.get("equations", None)
    if inline is not None:
        if isinstance(inline, str):
            s = inline.strip()
            if s:
                eqs.append(s)
        else:
            if not isinstance(inline, list):
                raise TypeError("'equations' must be a list of strings (or a single string).")
            for e in inline:
                s = str(e).strip()
                if s:
                    eqs.append(s)

    if not eqs:
        raise ValueError("No equations provided. Use 'equations' and/or 'equations_file'.")

    meta = dict(data.get("meta", {}) or {})
    if eq_file is not None:
        meta["equations_file"] = str(_resolve_path(eq_file, base_dir))

    # Store solve options in meta too (handy for UIs)
    meta.setdefault("solve", {})
    if isinstance(meta.get("solve"), dict):
        meta["solve"].update(
            {
                "backend": backend,
                "method": method,
                "tol": tol,
                "max_iter": max_iter,
                "max_restarts": max_restarts,
            }
        )

    # Return a Mapping compatible with equations.spec.system_from_mapping()
    # (it understands: backend/method/tol/max_iter/constants/variables/equations/meta)
    return AttrDict(
        {
            "backend": backend,
            "method": method,
            "tol": tol,
            "max_iter": max_iter,
            "max_restarts": max_restarts,
            "constants": constants,
            "variables": variables,
            "equations": eqs,
            "meta": meta,
        }
    )


@register("optimize")
@with_error_context("build:optimize")
def build_optimize(data: Mapping[str, Any], base_dir: Path) -> Mapping[str, Any]:
    """Build an optimization specification mapping.

    The optimization builder accepts objective and sense fields, inline or
    file-based constraints, optional design-variable declarations, and the same
    equation-style variables/constants fields used by ``build_equations``.

    The returned mapping is routed by the equations facade to the optimizer
    backend when ``problem_type`` is ``"optimize"`` or when objective and
    constraint keys are present.
    """
    solve_in = _solve_dict(data)

    # Prefer explicit top-level keys; else solve block; else defaults.
    backend = _normalize_backend(
        _pick_solve_opt(
            data,
            solve_in,
            "backend",
            _pick_solve_opt(data, solve_in, "solver", data.get("backend", data.get("solver", "auto"))),
        )
    )
    method = str(_pick_solve_opt(data, solve_in, "method", data.get("method", "SLSQP")))
    tol = float(_pick_solve_opt(data, solve_in, "tol", data.get("tol", 1e-6)))
    max_iter = int(_pick_solve_opt(data, solve_in, "max_iter", data.get("max_iter", data.get("maxiter", 200))))
    max_restarts = int(_pick_solve_opt(data, solve_in, "max_restarts", data.get("max_restarts", data.get("restarts", 0))))

    # Objective + sense
    objective = data.get("objective", None)
    sense_in = data.get("sense", None)

    if objective is None:
        if data.get("minimize", None) is not None:
            objective = data.get("minimize")
            sense_in = sense_in or "min"
        elif data.get("maximize", None) is not None:
            objective = data.get("maximize")
            sense_in = sense_in or "max"

    if objective is None:
        raise ValueError("optimize requires an 'objective' (or 'minimize'/'maximize').")

    sense = str(sense_in or "min").strip().lower()
    if sense in {"min", "minimize"}:
        sense = "min"
    elif sense in {"max", "maximize"}:
        sense = "max"
    else:
        raise ValueError("optimize.sense must be 'min' or 'max'.")

    # Constraints: inline + file
    constraints: List[str] = []

    c_in = data.get("constraints", data.get("constraint", None))
    if c_in is not None:
        if isinstance(c_in, str):
            s = c_in.strip()
            if s:
                constraints.append(s)
        else:
            if not isinstance(c_in, list):
                raise TypeError("'constraints' must be a list of strings (or a single string).")
            for c in c_in:
                s = str(c).strip()
                if s:
                    constraints.append(s)

    c_file = data.get("constraints_file", None)
    if c_file is not None:
        p = _resolve_path(c_file, base_dir)
        constraints.extend(_read_equations_lines(p))

    # Reuse the stable equations normalizer for vars/constants/equations, but do NOT
    # change its output shape. If caller provided no equations but did provide constraints,
    # feed them in as equations for normalization.
    data2 = dict(data)
    if (data2.get("equations", None) is None) and (data2.get("equations_file", None) is None) and constraints:
        data2["equations"] = list(constraints)

    base_eq = build_equations(data2, base_dir=base_dir)

    # Default constraints to equations if not provided
    if not constraints:
        constraints = list(base_eq.get("equations", []))

    # Design vars (optional)
    dv = data.get("design_vars", data.get("designvars", data.get("design_variables", None)))
    design_vars: List[str] = []
    if dv is not None:
        if isinstance(dv, str):
            design_vars = [p.strip() for p in dv.replace(";", ",").split(",") if p.strip()]
        else:
            if not isinstance(dv, list):
                raise TypeError("'design_vars' must be a list of names (or a comma-separated string).")
            design_vars = [str(x).strip() for x in dv if str(x).strip()]

    meta = dict(base_eq.get("meta", {}) or {})
    if c_file is not None:
        meta["constraints_file"] = str(_resolve_path(c_file, base_dir))

    # Keep solve info in meta for UI/debug (but NOT required by optimizer router).
    meta.setdefault("solve", {})
    if isinstance(meta.get("solve"), dict):
        meta["solve"].update(
            {
                "backend": backend,
                "method": method,
                "tol": tol,
                "max_iter": max_iter,
                "max_restarts": max_restarts,
            }
        )

    out = AttrDict(dict(base_eq))
    out.update(
        {
            "problem_type": "optimize",
            "backend": backend,
            "method": method,
            "tol": tol,
            "max_iter": max_iter,
            "max_restarts": max_restarts,
            "objective": str(objective).strip(),
            "sense": sense,
            "constraints": list(constraints),
            "design_vars": list(design_vars),
            "meta": meta,
        }
    )

    # Keep equations as alias if missing (rare)
    if not out.get("equations"):
        out["equations"] = list(constraints)

    return out


# ------------------------------ entry points ------------------------------


def build_problem(mapping: Mapping[str, Any], in_path: Path) -> ProblemSpec:
    """Normalize an input mapping into a ``ProblemSpec``.

    The input mapping must include ``problem_type``. All remaining keys are
    copied into ``ProblemSpec.data``.
    """
    if "problem_type" not in mapping:
        raise ValueError("Input must include 'problem_type'")
    pt = str(mapping["problem_type"])
    data = dict(mapping)
    data.pop("problem_type", None)
    return ProblemSpec(problem_type=pt, data=data)


def build_spec(problem: ProblemSpec, base_dir: Path) -> Any:
    """Build the solver-specific specification for a ``ProblemSpec``."""
    if problem.problem_type not in _BUILDERS:
        raise ValueError(
            f"Unknown problem_type: {problem.problem_type!r}. "
            f"Known: {sorted(_BUILDERS.keys())}"
        )
    return _BUILDERS[problem.problem_type](problem.data, base_dir)
