from __future__ import annotations

"""
app

Application service: orchestrates parsing, spec-building, solving, and output writing.

Routes:
- nozzle_ideal   : core.NozzleIdealGasSolver
- thermo_props   : thermo_props.api.run(spec) (CoolProp-backed when available)
- equations      : equations.solve_system(system) facade (GEKKO/SciPy backends)
- optimize       : equations.solve_optimize(system) facade (SciPy minimize backend)

Latest facts / pitfalls addressed (2026-02):
- thermo_props inputs exist in TWO shapes in the wild:
    A) normalized (design.py style):
        {fluid, backend, states:[{id,given,ask}], meta}
    B) human-friendly (your demo files):
        {fluid, state:{...}, outputs:[...]}   or   {fluid, states:[{name,...}], outputs:[...]}
  This module adapts both into the normalized shape before calling thermo_props.api.

- equations/optimize inputs exist in TWO shapes:
    A) mapping (recommended; consumed by equations.spec.system_from_mapping):
        {variables:{...}, params/constants:{...}, equations:[...], backend/method/tol/...}
    B) legacy/dataclass-like object (older design.py returned EquationsSolveSpec without .check_square)
  This module converts legacy/dataclass specs into the mapping shape before calling
  equations.solve_system(...), preventing AttributeError(.check_square).

- GUI interpreter outputs often store solve options under:
      "solve": {"backend": "...", "method": "...", "tol": ..., "max_iter": ...}
  Even when backend/method/tol aren't present at the top-level. This module now
  extracts those solve overrides consistently.

Constraints:
- Keep imports light; optional deps are imported only inside routes.
- app.py must call the equations facade (equations.*), not equations.solver directly.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from apis import RunRequest, RunResult
from core import NozzleIdealGasSolver
from design import build_problem, build_spec
from in_out import load_problem, save_json
from utils import ensure_dir, setup_logger, timed, with_error_context

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


# ------------------------------ JSON sanitization ------------------------------


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion to JSON-serializable primitives."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, Mapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if np is not None:
        try:
            if isinstance(x, (np.ndarray,)):
                return x.tolist()
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, (np.integer,)):
                return int(x)
        except Exception:
            pass
    d = getattr(x, "__dict__", None)
    if isinstance(d, dict):
        return _to_jsonable(d)
    return str(x)


def _jsonify_payload(obj: Any) -> Dict[str, Any]:
    """
    Convert solver outputs to a JSON-serializable dict.

    - dict -> sanitize recursively
    - dataclass -> asdict -> sanitize
    - plain object -> __dict__ -> sanitize
    - else -> {"value": str(obj)}
    """
    if isinstance(obj, dict):
        out = _to_jsonable(obj)
        return out if isinstance(out, dict) else {"value": out}
    if is_dataclass(obj):
        out = _to_jsonable(asdict(obj))
        return out if isinstance(out, dict) else {"value": out}
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        out = _to_jsonable(d)
        return out if isinstance(out, dict) else {"value": out}
    return {"value": _to_jsonable(obj)}


# ------------------------------ path helpers ------------------------------


def _resolve_in_path(in_dir: Path, p: Path) -> Path:
    """Resolve input path relative to package in_dir if not absolute."""
    return p if p.is_absolute() else (in_dir / p).resolve()


def _resolve_out_path(out_dir: Path, in_path: Path, out_path: Path | None) -> Path:
    """Resolve output path relative to package out_dir, defaulting to <stem>.out.json."""
    if out_path is None:
        return (out_dir / (in_path.stem + ".out.json")).resolve()
    return out_path if out_path.is_absolute() else (out_dir / out_path).resolve()


# ------------------------------ small parsing helpers ------------------------------


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _split_csv(s: str) -> list[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def _as_outputs_list(x: Any) -> list[str]:
    """
    Accept:
      - ["T","P",...]
      - ("T","P",...)
      - "T,P,Hmass"
    """
    if x is None:
        return []
    if isinstance(x, str):
        return _split_csv(x)
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for it in x:
            if it is None:
                continue
            if isinstance(it, str):
                out.extend(_split_csv(it) if ("," in it) else [it.strip()])
            else:
                out.append(str(it))
        return [k for k in (p.strip() for p in out) if k]
    return _split_csv(str(x))


def _as_mapping(obj: Any) -> Dict[str, Any]:
    """Best-effort: object -> dict (without importing heavy deps)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return dict(asdict(obj))
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return dict(d)
    return {}


def _get_solve_overrides(m: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract solver overrides from either:
      - top-level keys (backend/method/tol/...)
      - nested "solve" or "solver" mapping
    """
    out: Dict[str, Any] = {}

    # nested solve dict wins
    for k in ("solve", "solver"):
        blob = m.get(k, None)
        if isinstance(blob, Mapping):
            out.update({str(kk): vv for kk, vv in blob.items()})

    # then fill in from top-level if missing
    for k in ("backend", "solver", "method", "tol", "max_iter", "max_restarts", "use_units"):
        if k in m and k not in out:
            out[k] = m.get(k)

    # normalize aliases: "solver" -> "backend" where appropriate
    if "backend" not in out and "solver" in out:
        out["backend"] = out.get("solver")

    return out


# ------------------------------ thermo_props spec adapter ------------------------------


def _adapt_thermo_props_spec(spec: Any) -> Dict[str, Any]:
    """
    Normalize thermo_props spec to:
      {backend, fluid, states:[{id,given,ask}], meta}

    Supports BOTH:
      - normalized: {states:[{given,ask,...}], ...}
      - friendly:   {state:{...}, outputs:[...]} or {states:[{name,...}], outputs:[...]}
      - legacy:     {given:{...}, ask:[...]}
    """
    m = _as_mapping(spec)

    backend = str(m.get("backend", m.get("solver", "coolprop")) or "coolprop")
    fluid = m.get("fluid", None)
    if fluid is None:
        raise ValueError("thermo_props spec is missing required key: 'fluid'.")
    fluid = str(fluid)

    # outputs / ask
    outputs = _as_outputs_list(m.get("outputs", None))
    ask_top = _as_outputs_list(m.get("ask", None))
    ask_default = ask_top or outputs

    meta = dict(m.get("meta", {}) or {}) if isinstance(m.get("meta", {}), Mapping) else {}

    # External hint (optional)
    if "states_file" in m and "states_file" not in meta:
        meta["states_file"] = str(m.get("states_file"))

    states_out: list[dict[str, Any]] = []

    # 1) normalized/builder shape: states:[{given, ask, id}]
    if isinstance(m.get("states", None), list):
        for i, st in enumerate(m["states"]):
            if not isinstance(st, Mapping):
                raise TypeError(f"thermo_props.states[{i}] must be a mapping.")
            st_m = dict(st)

            sid = st_m.get("id", st_m.get("name", None))
            sid = str(sid) if sid is not None else None

            if "given" in st_m:
                given = dict(st_m["given"]) if isinstance(st_m["given"], Mapping) else {}
                ask = _as_outputs_list(st_m.get("ask", st_m.get("outputs", ask_default)))
            else:
                # Friendly per-state: {name:..., T_C:..., x:..., ...}
                given = {k: v for k, v in st_m.items() if k not in {"id", "name", "ask", "outputs"}}
                ask = _as_outputs_list(st_m.get("ask", st_m.get("outputs", ask_default)))

            if not given:
                raise ValueError(f"thermo_props.states[{i}] has empty 'given' mapping.")
            if len(ask) == 0 and len(ask_default) == 0:
                # allow empty ask (backend may return defaults), but keep explicit when possible
                ask = []

            states_out.append({"id": sid, "given": given, "ask": ask})

    # 2) friendly single-state: state:{...}
    elif isinstance(m.get("state", None), Mapping):
        sid = m.get("id", m.get("name", None))
        sid = str(sid) if sid is not None else None
        given = dict(m["state"])
        ask = _as_outputs_list(m.get("ask", m.get("outputs", ask_default)))
        if not given:
            raise ValueError("thermo_props.state is empty.")
        states_out.append({"id": sid, "given": given, "ask": ask})

    # 3) legacy single-state: given:{...}
    elif isinstance(m.get("given", None), Mapping):
        sid = m.get("id", m.get("name", None))
        sid = str(sid) if sid is not None else None
        given = dict(m["given"])
        ask = _as_outputs_list(m.get("ask", ask_default))
        if not given:
            raise ValueError("thermo_props.given is empty.")
        states_out.append({"id": sid, "given": given, "ask": ask})

    else:
        raise ValueError(
            "thermo_props spec must include one of: "
            "'states' (list), 'state' (mapping), or 'given' (mapping)."
        )

    return {"backend": backend, "fluid": fluid, "states": states_out, "meta": meta}


# ------------------------------ equations/optimize spec adapter ------------------------------


def _adapt_equations_spec(spec: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize equations/optimize spec to a MAPPING accepted by equations.* facade.

    Preferred mapping shape (works with equations.spec.system_from_mapping):
      {
        "backend": "auto"|"scipy"|"gekko",
        "method": "hybr"|"SLSQP"|...,
        "tol": 1e-9,
        "max_iter": 200,
        "max_restarts": 2,
        "variables": {...},
        "params" or "constants": {...},
        "equations": [...],
        # optimize additions (optional):
        "objective": "...",
        "sense": "min"|"max",
        "constraints": [...]
      }

    Also returns extracted opts dict for the app-level call.
    """
    # If caller already provided a mapping, keep it.
    if isinstance(spec, Mapping):
        m = dict(spec)
    else:
        # Handle legacy/dataclass shape (e.g., design.EquationsSolveSpec)
        m = _as_mapping(spec)

        # If it looks like the legacy EqSpec object: equations(list), variables(list), params(dict)
        eqs = getattr(spec, "equations", None)
        vars_list = getattr(spec, "variables", None)
        params = getattr(spec, "params", None)

        if isinstance(eqs, list) and isinstance(vars_list, list) and isinstance(params, dict):
            variables_map: Dict[str, Any] = {}
            for v in vars_list:
                # v may be design.EqVar or any duck-typed var object
                name = str(getattr(v, "name", ""))
                if not name:
                    continue
                kind = str(getattr(v, "kind", "unknown"))
                fixed = (kind.lower() == "fixed") or (getattr(v, "value", None) is not None and kind.lower() != "unknown")
                payload: Dict[str, Any] = {
                    "fixed": bool(fixed),
                    "guess": getattr(v, "guess", None),
                    "value": getattr(v, "value", None),
                    "lower": getattr(v, "lower", None),
                    "upper": getattr(v, "upper", None),
                }
                # unit/desc are optional
                u = getattr(v, "units", None)
                if u is not None:
                    payload["unit"] = u
                d = getattr(v, "desc", None)
                if d is not None:
                    payload["desc"] = d
                variables_map[name] = payload

            m = {
                "backend": getattr(spec, "backend", "auto"),
                "method": getattr(spec, "method", "hybr"),
                "tol": getattr(spec, "tol", 1e-9),
                "max_iter": getattr(spec, "max_iter", 200),
                "max_restarts": getattr(spec, "max_restarts", 2),
                "variables": variables_map,
                # prefer "params" (your demo JSON), but system_from_mapping also supports "constants"
                "params": dict(params),
                "equations": list(eqs),
                "meta": dict(getattr(spec, "meta", {}) or {}),
            }

    # Extract opts (and set sensible defaults). Nested solve dict should win.
    solve_overrides = _get_solve_overrides(m)

    opts: Dict[str, Any] = {
        "backend": solve_overrides.get("backend", "auto"),
        "method": solve_overrides.get("method", m.get("method", "hybr")),
        "tol": solve_overrides.get("tol", m.get("tol", 1e-9)),
        "max_iter": solve_overrides.get("max_iter", m.get("max_iter", 200)),
        "max_restarts": solve_overrides.get("max_restarts", m.get("max_restarts", 2)),
        "use_units": solve_overrides.get("use_units", m.get("use_units", False)),
    }

    return m, opts


# ------------------------------ app service ------------------------------


class TdpyApp:
    """Application service: orchestrates parsing, solving, and output writing."""

    def __init__(self, in_dir: Path | None = None, out_dir: Path | None = None) -> None:
        pkg_dir = Path(__file__).resolve().parent
        self.in_dir = (in_dir or (pkg_dir / "in")).resolve()
        self.out_dir = (out_dir or (pkg_dir / "out")).resolve()
        ensure_dir(self.in_dir)
        ensure_dir(self.out_dir)
        self.log = setup_logger("tdpy")

    def list_inputs(self) -> Dict[str, Any]:
        items = sorted([p for p in self.in_dir.rglob("*") if p.is_file()])
        return {"in_dir": str(self.in_dir), "files": [str(p.relative_to(self.in_dir)) for p in items]}

    # ------------------------------ main entry ------------------------------

    @timed
    @with_error_context("TdpyApp.run")
    def run(self, req: RunRequest) -> RunResult:
        in_path = _resolve_in_path(self.in_dir, req.in_path)

        raw = load_problem(in_path)
        prob = build_problem(raw, in_path)
        spec = build_spec(prob, base_dir=in_path.parent)

        solver_name, payload = self._solve(problem_type=prob.problem_type, spec=spec)

        out_path = _resolve_out_path(self.out_dir, in_path, req.out_path)

        save_json(
            {
                "solver": solver_name,
                "problem_type": prob.problem_type,
                "in": str(in_path),
                "result": payload,
            },
            out_path,
        )

        plots: Dict[str, str] = {}
        if req.make_plots and go is not None:
            plots = self._make_default_plots(
                problem_type=prob.problem_type,
                payload=payload,
                out_base=out_path.with_suffix(""),
            )

        # Try to reflect solver success if payload includes it; otherwise ok=True means "pipeline ran"
        ok = bool(payload.get("ok", True)) if isinstance(payload, dict) else True

        return RunResult(
            ok=ok,
            solver=solver_name,
            in_path=in_path,
            out_path=out_path,
            payload=payload,
            plots=plots,
        )

    # ------------------------------ routing ------------------------------

    def _solve(self, problem_type: str, spec: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Route to the correct solver backend.

        Returns:
          (solver_name, payload_dict)
        """
        pt = str(problem_type)

        if pt == "nozzle_ideal":
            solver = NozzleIdealGasSolver()
            payload = solver.solve(spec)
            return getattr(solver, "name", "nozzle_ideal"), _jsonify_payload(payload)

        if pt == "thermo_props":
            # Avoid hard import failures when CoolProp isn't installed.
            from thermo_props import api as tp_api  # local import (optional deps)

            spec_norm = _adapt_thermo_props_spec(spec)

            if hasattr(tp_api, "run"):
                payload = tp_api.run(spec_norm)  # type: ignore[misc]
            elif hasattr(tp_api, "eval_states"):
                payload = tp_api.eval_states(spec_norm)  # type: ignore[misc]
            else:
                raise RuntimeError("thermo_props.api must define run(spec) (or eval_states(spec)).")

            backend = str(spec_norm.get("backend", "coolprop") or "coolprop")
            return f"thermo-props:{backend}", _jsonify_payload(payload)

        if pt in ("equations", "optimize"):
            # IMPORTANT: use facade (adapts spec -> backend solver shape)
            system_map, opts = _adapt_equations_spec(spec)

            # Route to the correct facade entry point.
            # Prefer .equations.solve(...) if present (router), else use explicit calls.
            solve_fn = None
            solve_name = None
            try:
                import equations as eq_facade
                if hasattr(eq_facade, "solve"):
                    solve_fn = getattr(eq_facade, "solve")
                    solve_name = "solve"
                elif pt == "optimize" and hasattr(eq_facade, "solve_optimize"):
                    solve_fn = getattr(eq_facade, "solve_optimize")
                    solve_name = "solve_optimize"
                elif hasattr(eq_facade, "solve_system"):
                    solve_fn = getattr(eq_facade, "solve_system")
                    solve_name = "solve_system"
            except Exception:
                solve_fn = None

            if solve_fn is None:
                # Fallback to module-level api (still facade; not solver.py)
                from equations import api as eq_api  # type: ignore
                if pt == "optimize" and hasattr(eq_api, "solve_optimize"):
                    solve_fn = getattr(eq_api, "solve_optimize")
                    solve_name = "solve_optimize"
                elif hasattr(eq_api, "solve_system"):
                    solve_fn = getattr(eq_api, "solve_system")
                    solve_name = "solve_system"
                else:
                    raise RuntimeError("Could not locate equations facade solve function (solve/solve_system/solve_optimize).")

            sol = solve_fn(
                system_map,
                backend=str(opts.get("backend", "auto")),
                method=str(opts.get("method", "hybr")),
                tol=float(opts.get("tol", 1e-9)),
                max_iter=int(opts.get("max_iter", 200)),
                max_restarts=int(opts.get("max_restarts", 2)),
                use_units=bool(opts.get("use_units", False)),
            )

            payload = _jsonify_payload(sol)

            # Prefer backend from payload; else opts
            backend = str(payload.get("backend", opts.get("backend", "unknown")))
            route = "optimize" if pt == "optimize" else "equations"
            return f"{route}:{backend}", payload

        raise ValueError(f"No solver registered for problem_type={pt!r}")

    # ------------------------------ plots ------------------------------

    def _make_default_plots(self, problem_type: str, payload: Dict[str, Any], out_base: Path) -> Dict[str, str]:
        """
        Default plots (conservative / CLI-friendly):
        - nozzle_ideal: Mach, P, T vs x (mm)
        - thermo_props / equations / optimize: none by default
        """
        if go is None:
            return {}

        if str(problem_type) != "nozzle_ideal":
            return {}

        x = payload.get("x_mm", None)
        if x is None:
            return {}

        plots: Dict[str, str] = {}

        def save(fig: Any, name: str) -> str:
            p = out_base.parent / f"{out_base.name}.{name}.html"
            fig.write_html(str(p))
            return str(p)

        if "M" in payload:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=payload["M"], mode="lines+markers", name="Mach"))
            fig.update_layout(title="Mach vs x (mm)", xaxis_title="x (mm)", yaxis_title="Mach")
            plots["mach"] = save(fig, "mach")

        if "P_Pa" in payload:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=payload["P_Pa"], mode="lines+markers", name="P"))
            fig.update_layout(title="Pressure vs x (mm)", xaxis_title="x (mm)", yaxis_title="P (Pa)")
            plots["pressure"] = save(fig, "pressure")

        if "T_K" in payload:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=payload["T_K"], mode="lines+markers", name="T"))
            fig.update_layout(title="Temperature vs x (mm)", xaxis_title="x (mm)", yaxis_title="T (K)")
            plots["temperature"] = save(fig, "temperature")

        return plots
