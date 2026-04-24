# equations/optimizer.py
from __future__ import annotations

"""
equations.optimizer

SciPy-based optimization backend for TDPy.

This module is intentionally *small* and *duck-typed*:
- It consumes the "opt_like" mapping built by equations/api.py (solve_optimize()).
- It reuses the same safe expression compiler used by the equation solver:
    - Objective: safe_eval.compile_expression()
    - Constraints: safe_eval.compile_residual()  (treated as equality constraints: residual == 0)
- It evaluates expressions via safe_eval.eval_compiled() with a restricted scope:
    SAFE_FUNCS/SAFE_CONSTS + injected callables/constants + current variable values.

Design goals:
- Do not break existing equation-solving behavior (this file is only used when
  api.py routes a spec with problem_type == "optimize").
- Be robust to thermo/property-call evaluation errors by applying a penalty
  (avoids SciPy blowing up mid-iteration).
- Keep the returned type compatible with solver.py results (SolveResult).

Important limitation (first iteration):
- Inequality constraints like "g(x) <= 0" are NOT supported as raw DSL because
  safe_eval disallows comparisons. Use:
    - bounds: (preferred) provided by build_spec.py
    - OR encode inequality as a smooth penalty in the objective yourself.
"""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .safe_eval import (
    CompiledExpr,
    ParseError,
    UnsafeExpressionError,
    compile_expression,
    compile_residual,
    eval_compiled,
)

from .solver import SolveResult


def _compile_expression_compat(
    expr: str,
    *,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> CompiledExpr:
    """
    Compatibility wrapper: older safe_eval.compile_expression may not accept
    extra_funcs/extra_consts. Try the richest signature first.
    """
    try:
        return compile_expression(expr, extra_funcs=extra_funcs, extra_consts=extra_consts)  # type: ignore[call-arg]
    except TypeError:
        try:
            return compile_expression(expr, extra_funcs=extra_funcs)  # type: ignore[call-arg]
        except TypeError:
            return compile_expression(expr)  # type: ignore[call-arg]


def _compile_residual_compat(
    expr: str,
    *,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> CompiledExpr:
    """
    Compatibility wrapper: older safe_eval.compile_residual may not accept
    extra_funcs/extra_consts.
    """
    try:
        return compile_residual(expr, extra_funcs=extra_funcs, extra_consts=extra_consts)  # type: ignore[call-arg]
    except TypeError:
        try:
            return compile_residual(expr, extra_funcs=extra_funcs)  # type: ignore[call-arg]
        except TypeError:
            return compile_residual(expr)  # type: ignore[call-arg]


def _eval_compiled_compat(
    c: CompiledExpr,
    *,
    values: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Compatibility wrapper: older safe_eval.eval_compiled may not accept
    extra_funcs/extra_consts.
    """
    try:
        return float(
            eval_compiled(
                c,
                values=values,
                params=params,
                extra_funcs=extra_funcs,
                extra_consts=extra_consts,
            )
        )
    except TypeError:
        try:
            return float(eval_compiled(c, values=values, params=params, extra_funcs=extra_funcs))  # type: ignore[call-arg]
        except TypeError:
            return float(eval_compiled(c, values=values, params=params))  # type: ignore[call-arg]


def _is_mapping(x: Any) -> bool:
    try:
        from collections.abc import Mapping as _Mapping
        return isinstance(x, _Mapping)
    except Exception:  # pragma: no cover
        return hasattr(x, "keys") and hasattr(x, "__getitem__")


def _var_field(v: Any, name: str, default: Any = None) -> Any:
    """Duck-typed accessor for VarLike dataclass or dict."""
    if v is None:
        return default
    if hasattr(v, name):
        try:
            return getattr(v, name)
        except Exception:
            return default
    if isinstance(v, dict):
        return v.get(name, default)
    return default


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def _l2_norm(vals: Sequence[float]) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in vals)))


def _extract_callable_params(params: Mapping[str, Any]) -> Dict[str, Callable[..., Any]]:
    """
    Anything callable in params can be treated as extra_funcs for compilation
    and evaluation. This lets safe_eval validate function names that are only
    provided at runtime (e.g., PropsSI wrappers), without expanding the global
    allowlist.
    """
    out: Dict[str, Callable[..., Any]] = {}
    for k, v in params.items():
        if callable(v):
            out[str(k)] = v  # type: ignore[assignment]
    return out


def _extract_extra_consts(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Non-callable injected names (strings, numbers) can be passed as extra_consts
    to help name extraction. For evaluation, we also pass params directly.
    """
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if callable(v):
            continue
        out[str(k)] = v
    return out


@dataclass(frozen=True)
class _CompiledOpt:
    objective: CompiledExpr
    sense: str
    constraints: List[CompiledExpr]
    unknown_names: List[str]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    x0: List[float]


def _compile_problem(problem: Mapping[str, Any]) -> _CompiledOpt:
    objective = str(problem.get("objective", "") or "").strip()
    if not objective:
        raise ValueError("Optimization spec is missing 'objective'.")

    sense = str(problem.get("sense", "min") or "min").strip().lower()
    if sense not in {"min", "max"}:
        raise ValueError(f"Invalid optimization sense: {sense!r}")

    params = problem.get("params", {})
    if not _is_mapping(params):
        params = {}
    params = dict(params)  # type: ignore[arg-type]

    variables = problem.get("variables", [])
    if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
        raise TypeError("Optimization spec 'variables' must be a list of var-like objects.")

    unknowns: List[Any] = []
    for v in variables:
        unk = _var_field(v, "unknown", None)
        kind = str(_var_field(v, "kind", "") or "").lower()
        val = _var_field(v, "value", None)

        is_unknown = False
        if isinstance(unk, bool):
            is_unknown = bool(unk)
        elif kind:
            is_unknown = kind == "unknown"
        else:
            is_unknown = val is None

        if is_unknown:
            unknowns.append(v)

    if not unknowns:
        unknowns = list(variables)

    unknown_names = [str(_var_field(v, "name", "") or "").strip() for v in unknowns]
    if any(not n for n in unknown_names):
        raise ValueError("One or more optimization variables are missing a valid 'name'.")

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    x0: List[float] = []
    for v in unknowns:
        lo = _as_float(_var_field(v, "lower", None))
        hi = _as_float(_var_field(v, "upper", None))
        bounds.append((lo, hi))

        g = _as_float(_var_field(v, "guess", None))
        if g is None:
            g = _as_float(_var_field(v, "value", None))
        if g is None:
            g = 1.0

        if lo is not None and g < lo:
            g = lo
        if hi is not None and g > hi:
            g = hi
        x0.append(float(g))

    constraints_in = problem.get("constraints", [])
    if constraints_in is None:
        constraints_in = []
    if not isinstance(constraints_in, Sequence) or isinstance(constraints_in, (str, bytes)):
        raise TypeError("Optimization spec 'constraints' must be a list of strings.")
    constraints_str = [str(s) for s in constraints_in if str(s).strip()]

    extra_funcs = _extract_callable_params(params)
    extra_consts = _extract_extra_consts(params)

    try:
        c_obj = _compile_expression_compat(objective, extra_funcs=extra_funcs, extra_consts=extra_consts)
    except (UnsafeExpressionError, ParseError) as e:
        raise ParseError(f"Objective expression could not be compiled: {objective!r}: {e}") from e

    c_cons: List[CompiledExpr] = []
    for s in constraints_str:
        try:
            c = _compile_residual_compat(s, extra_funcs=extra_funcs, extra_consts=extra_consts)
        except (UnsafeExpressionError, ParseError) as e:
            raise ParseError(f"Constraint could not be compiled: {s!r}: {e}") from e
        c_cons.append(c)

    return _CompiledOpt(
        objective=c_obj,
        sense=sense,
        constraints=c_cons,
        unknown_names=unknown_names,
        bounds=bounds,
        x0=x0,
    )


@dataclass
class _PenaltyCounter:
    obj_penalty: int = 0
    cons_penalty: int = 0


def _make_eval_context(
    params: Mapping[str, Any],
    unknown_names: Sequence[str],
    x: Sequence[float],
) -> Dict[str, Any]:
    values = {str(k): float(v) for k, v in zip(unknown_names, x)}
    return {"values": values, "params": dict(params)}


def _eval_objective(
    compiled: CompiledExpr,
    *,
    sense: str,
    params: Mapping[str, Any],
    unknown_names: Sequence[str],
    x: Sequence[float],
    penalty_value: float,
    counter: _PenaltyCounter,
) -> float:
    ctx = _make_eval_context(params, unknown_names, x)
    try:
        y = float(_eval_compiled_compat(compiled, values=ctx["values"], params=ctx["params"]))
    except Exception:
        counter.obj_penalty += 1
        y = float(penalty_value)
    if sense == "max":
        return -y
    return y


def _eval_constraints(
    compiled_list: Sequence[CompiledExpr],
    *,
    params: Mapping[str, Any],
    unknown_names: Sequence[str],
    x: Sequence[float],
    penalty_value: float,
    counter: _PenaltyCounter,
) -> List[float]:
    if not compiled_list:
        return []
    ctx = _make_eval_context(params, unknown_names, x)
    out: List[float] = []
    for c in compiled_list:
        try:
            r = float(_eval_compiled_compat(c, values=ctx["values"], params=ctx["params"]))
        except Exception:
            counter.cons_penalty += 1
            r = float(penalty_value)
        out.append(r)
    return out


def solve_optimize(
    problem: Mapping[str, Any],
    *,
    backend: Optional[str] = None,
    method: Optional[str] = None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
) -> SolveResult:
    """
    Solve an optimization problem.

    Parameters are passed through by equations/api.py, but this function also
    reads defaults from the `problem` mapping itself.

    Expected keys in `problem` (duck-typed; see equations/api.py solve_optimize):
      - objective: str
      - sense: "min" | "max"
      - constraints: list[str]          # treated as equality residuals == 0
      - variables: list[var-like]       # each has name/kind/value/guess/lower/upper
      - params: dict[str, Any]          # numeric constants, symbols, and property callables
      - tol, max_iter, method (optional)

    Returns:
      SolveResult (same shape as equations/solver.py).
    """
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import Bounds, minimize  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SciPy is required for optimization (scipy.optimize.minimize). "
            "Install scipy to use equations.optimizer."
        ) from e

    method_eff = str(method or problem.get("method", "") or "SLSQP").strip()
    tol_eff = float(tol if tol is not None else problem.get("tol", 1e-8))
    max_iter_eff = int(max_iter if max_iter is not None else problem.get("max_iter", 200))

    penalty_value = float(problem.get("thermo_penalty", 1e20))

    compiled = _compile_problem(problem)
    params = problem.get("params", {})
    params = dict(params) if _is_mapping(params) else {}

    lo = [(-np.inf if b[0] is None else float(b[0])) for b in compiled.bounds]
    hi = [(np.inf if b[1] is None else float(b[1])) for b in compiled.bounds]
    bounds = Bounds(lo, hi)

    counter = _PenaltyCounter()

    def fun(x: "np.ndarray") -> float:
        return _eval_objective(
            compiled.objective,
            sense=compiled.sense,
            params=params,
            unknown_names=compiled.unknown_names,
            x=list(map(float, x)),
            penalty_value=penalty_value,
            counter=counter,
        )

    scipy_constraints: List[Dict[str, Any]] = []
    if compiled.constraints:
        for c in compiled.constraints:
            def _make_fun(ci: CompiledExpr) -> Callable[["np.ndarray"], float]:
                def _f(x: "np.ndarray") -> float:
                    rr = _eval_constraints(
                        [ci],
                        params=params,
                        unknown_names=compiled.unknown_names,
                        x=list(map(float, x)),
                        penalty_value=penalty_value,
                        counter=counter,
                    )
                    return float(rr[0])
                return _f

            scipy_constraints.append({"type": "eq", "fun": _make_fun(c)})

    method_supports_constraints = {"SLSQP", "trust-constr", "COBYLA"}
    method_supports_bounds = {"SLSQP", "trust-constr", "L-BFGS-B", "TNC", "Powell"}
    method_used = method_eff

    forced_fallback_note = None
    if scipy_constraints and method_used not in method_supports_constraints:
        forced_fallback_note = f"Requested method {method_used!r} does not support equality constraints; using 'SLSQP'."
        method_used = "SLSQP"
    if (any(b[0] is not None or b[1] is not None for b in compiled.bounds)) and method_used not in method_supports_bounds:
        forced_fallback_note = (forced_fallback_note or "") + f" Requested method {method_eff!r} does not support bounds; using 'SLSQP'."
        method_used = "SLSQP"

    max_restarts_eff = int(problem.get("max_restarts", 0) or 0)
    x0_base = np.array(compiled.x0, dtype=float)

    best = None
    best_meta = None

    def _run_one(x0: "np.ndarray", run_ix: int) -> Tuple[Any, Dict[str, Any]]:
        res = minimize(
            fun,
            x0,
            method=method_used,
            bounds=bounds,
            constraints=scipy_constraints,
            tol=tol_eff,
            options={"maxiter": max_iter_eff},
        )
        x_star = np.array(res.x, dtype=float)
        cons_vals = _eval_constraints(
            compiled.constraints,
            params=params,
            unknown_names=compiled.unknown_names,
            x=list(map(float, x_star)),
            penalty_value=penalty_value,
            counter=counter,
        )
        cons_norm = _l2_norm(cons_vals)
        cons_maxabs = max((abs(v) for v in cons_vals), default=0.0)

        meta = {
            "scipy": {
                "success": bool(getattr(res, "success", False)),
                "status": int(getattr(res, "status", -1)),
                "message": str(getattr(res, "message", "")),
                "nit": int(getattr(res, "nit", 0) or 0),
                "nfev": int(getattr(res, "nfev", 0) or 0),
                "njev": int(getattr(res, "njev", 0) or 0),
                "method": str(method_used),
                "fun": float(getattr(res, "fun", float("nan"))),
            },
            "run_ix": int(run_ix),
            "constraint_norm": float(cons_norm),
            "constraint_maxabs": float(cons_maxabs),
            "penalties": {
                "objective": int(counter.obj_penalty),
                "constraints": int(counter.cons_penalty),
            },
        }
        if forced_fallback_note:
            meta["note"] = forced_fallback_note
        return res, meta

    res0, meta0 = _run_one(x0_base, 0)
    best = res0
    best_meta = meta0

    for k in range(1, max_restarts_eff + 1):
        sgn = -1.0 if (k % 2 == 0) else 1.0
        scale = 1.0 + sgn * min(0.10 * k, 0.5)
        x0 = x0_base * scale

        x0 = np.where(np.abs(x0) < 1e-12, x0_base + sgn * (0.1 * k), x0)

        x0 = np.minimum(np.maximum(x0, bounds.lb), bounds.ub)
        res_k, meta_k = _run_one(x0, k)

        choose_new = False
        if bool(getattr(res_k, "success", False)) and not bool(getattr(best, "success", False)):
            choose_new = True
        elif bool(getattr(res_k, "success", False)) == bool(getattr(best, "success", False)):
            fk = float(getattr(res_k, "fun", float("inf")))
            fb = float(getattr(best, "fun", float("inf")))
            ck = float(meta_k.get("constraint_norm", float("inf")))
            cb = float(best_meta.get("constraint_norm", float("inf"))) if isinstance(best_meta, dict) else float("inf")
            if ck < cb - 1e-12:
                choose_new = True
            elif abs(ck - cb) <= 1e-12 and fk < fb:
                choose_new = True

        if choose_new:
            best = res_k
            best_meta = meta_k

    x_best = list(map(float, getattr(best, "x", compiled.x0)))
    vars_out = {name: float(val) for name, val in zip(compiled.unknown_names, x_best)}
    residuals = _eval_constraints(
        compiled.constraints,
        params=params,
        unknown_names=compiled.unknown_names,
        x=x_best,
        penalty_value=penalty_value,
        counter=counter,
    )
    residual_norm = _l2_norm(residuals)

    msg = str(getattr(best, "message", ""))
    ok = bool(getattr(best, "success", False))
    nfev = int(getattr(best, "nfev", 0) or 0)

    meta: Dict[str, Any] = {
        "objective_value": float(getattr(best, "fun", float("nan"))),
        "sense": str(compiled.sense),
        "unknown_names": list(compiled.unknown_names),
        "bounds": [
            [None if lo is None else float(lo), None if hi is None else float(hi)]
            for lo, hi in compiled.bounds
        ],
        "requested_backend": str(backend or problem.get("backend", "scipy") or "scipy"),
        "requested_method": str(method or problem.get("method", "") or method_eff),
        "backend": "scipy",
        "method": str(method_used),
        "penalty_value": float(penalty_value),
    }
    if isinstance(best_meta, dict):
        meta.update(best_meta)

    return SolveResult(
        ok=ok,
        backend="scipy",
        method=str(method_used),
        message=msg,
        nfev=nfev,
        variables=vars_out,
        residuals=[float(r) for r in residuals],
        residual_norm=float(residual_norm),
        meta=meta,
    )


__all__ = ["solve_optimize"]
