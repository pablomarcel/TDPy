from __future__ import annotations

"""Safe numeric evaluation helpers for the TDPy interpreter.

This module evaluates numeric constants and numeric constant expressions in a
restricted way. It is used by the interpreter layer before the full nonlinear
equation solver is invoked.

Goals
-----
The evaluator is designed to:

* Evaluate simple numeric constants and constant expressions safely.
* Support unit-aware quantity parsing when the optional units layer is enabled.
* Provide useful error messages that identify the failing expression.
* Stay consistent with equation preprocessing, including ``^`` to ``**``
  conversion when the shared preprocessing helper is available.

Security policy
---------------
The evaluator uses an AST whitelist. It does not allow attribute access,
subscripts, comprehensions, lambdas, imports, comparisons, boolean operations,
conditional expressions, starred arguments, or calls to non-whitelisted
functions.
"""

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


# ------------------------------ preprocessing ------------------------------

# Reuse solver-safe preprocessing when available.
try:
    from equations.safe_eval import preprocess_expr  # type: ignore
except Exception:  # pragma: no cover

    def preprocess_expr(s: str) -> str:
        """Minimal fallback preprocessing used when equations.safe_eval is unavailable."""
        return s.replace("^", "**")


# ------------------------------ whitelist configuration ------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def default_numeric_funcs() -> Dict[str, Any]:
    """Return the default numeric-only function allowlist.

    The returned functions perform numeric work only. They do not perform file
    I/O, imports, attribute access, or backend calls.
    """
    funcs: Dict[str, Any] = {
        # core
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
        # exponentials and logs
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "log2": getattr(math, "log2", None),
        # trigonometry
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        # hyperbolic functions
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        # rounding
        "floor": math.floor,
        "ceil": math.ceil,
        # misc
        "hypot": math.hypot,
        "radians": math.radians,
        "degrees": math.degrees,
    }
    return {k: v for k, v in funcs.items() if v is not None}


# ------------------------------ errors ------------------------------

@dataclass(frozen=True)
class NumericEvalContext:
    """Context attached to a numeric evaluation error."""

    expr: str
    where: str = "numeric expression"


class NumericEvalError(ValueError):
    """Error raised when a numeric expression cannot be safely evaluated."""

    def __init__(self, message: str, *, ctx: Optional[NumericEvalContext] = None):
        if ctx is not None:
            super().__init__(f"{ctx.where}: {message} | expr={ctx.expr!r}")
        else:
            super().__init__(message)


# ------------------------------ safe numeric eval ------------------------------

def safe_eval_numeric(
    expr: str,
    *,
    names: Mapping[str, float],
    funcs: Optional[Mapping[str, Any]] = None,
) -> float:
    """Evaluate a numeric expression with an AST whitelist.

    Allowed syntax includes numeric literals, resolved names supplied through
    ``names``, the operators ``+``, ``-``, ``*``, ``/``, ``**``, ``%``, unary
    signs, parentheses, and calls to whitelisted numeric functions.

    Disallowed syntax includes attribute access, indexing, comprehensions,
    lambdas, assignments, comparisons, boolean operations, conditional
    expressions, calls to non-whitelisted functions, and starred arguments.

    Parameters
    ----------
    expr:
        Expression text to evaluate.
    names:
        Mapping of previously resolved numeric constants.
    funcs:
        Optional extra numeric-safe functions.

    Returns
    -------
    float
        Evaluated numeric value.
    """
    s = preprocess_expr(str(expr)).strip()
    ctx = NumericEvalContext(expr=s)

    if not s:
        raise NumericEvalError("Empty expression", ctx=ctx)

    fns = dict(default_numeric_funcs())
    if funcs:
        fns.update(dict(funcs))

    consts: Dict[str, float] = dict(names)
    consts.setdefault("pi", float(math.pi))
    consts.setdefault("e", float(math.e))

    try:
        node = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise NumericEvalError("Invalid syntax", ctx=ctx) from e

    class V(ast.NodeVisitor):
        """AST visitor implementing the numeric whitelist."""

        def visit_Expression(self, n: ast.Expression) -> float:
            return float(self.visit(n.body))

        def visit_Constant(self, n: ast.Constant) -> float:
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise NumericEvalError(f"Non-numeric literal not allowed: {n.value!r}", ctx=ctx)

        def visit_Num(self, n: ast.Num) -> float:  # pragma: no cover
            """Support Python versions that still expose ast.Num."""
            return float(n.n)

        def visit_Name(self, n: ast.Name) -> float:
            nm = n.id
            if nm.startswith("__"):
                raise NumericEvalError("Dunder names are not allowed", ctx=ctx)
            if nm in consts:
                return float(consts[nm])
            raise NumericEvalError(f"Unknown name: {nm!r}", ctx=ctx)

        def visit_UnaryOp(self, n: ast.UnaryOp) -> float:
            if not isinstance(n.op, _ALLOWED_UNARYOPS):
                raise NumericEvalError(f"Unary operator not allowed: {type(n.op).__name__}", ctx=ctx)
            v = float(self.visit(n.operand))
            return +v if isinstance(n.op, ast.UAdd) else -v

        def visit_BinOp(self, n: ast.BinOp) -> float:
            if not isinstance(n.op, _ALLOWED_BINOPS):
                raise NumericEvalError(f"Operator not allowed: {type(n.op).__name__}", ctx=ctx)
            a = float(self.visit(n.left))
            b = float(self.visit(n.right))
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a**b
            if isinstance(n.op, ast.Mod):
                return a % b
            raise NumericEvalError("Unhandled operator", ctx=ctx)

        def visit_Call(self, n: ast.Call) -> float:
            if not isinstance(n.func, ast.Name):
                raise NumericEvalError("Only direct calls f(x) are allowed", ctx=ctx)

            fn = n.func.id
            if fn not in fns:
                raise NumericEvalError(f"Function not allowed: {fn!r}", ctx=ctx)

            for a in n.args:
                if isinstance(a, ast.Starred):
                    raise NumericEvalError("Star-args are not allowed", ctx=ctx)
            for kw in n.keywords:
                if kw.arg is None:
                    raise NumericEvalError("Star-kwargs are not allowed", ctx=ctx)

            args = [float(self.visit(a)) for a in n.args]
            kwargs = {str(kw.arg): float(self.visit(kw.value)) for kw in n.keywords}
            try:
                return float(fns[fn](*args, **kwargs))
            except Exception as e:
                raise NumericEvalError(f"Call failed: {fn}(...) -> {e}", ctx=ctx) from e

        def visit_Attribute(self, n: ast.Attribute) -> float:
            raise NumericEvalError("Attribute access is not allowed", ctx=ctx)

        def visit_Subscript(self, n: ast.Subscript) -> float:
            raise NumericEvalError("Indexing is not allowed", ctx=ctx)

        def visit_Lambda(self, n: ast.Lambda) -> float:  # pragma: no cover
            raise NumericEvalError("Lambda is not allowed", ctx=ctx)

        def visit_Compare(self, n: ast.Compare) -> float:
            raise NumericEvalError("Comparisons are not allowed in numeric constants", ctx=ctx)

        def visit_BoolOp(self, n: ast.BoolOp) -> float:
            raise NumericEvalError("Boolean operations are not allowed in numeric constants", ctx=ctx)

        def visit_IfExp(self, n: ast.IfExp) -> float:
            raise NumericEvalError("Conditional expressions are not allowed", ctx=ctx)

        def generic_visit(self, n: ast.AST) -> float:
            raise NumericEvalError(f"Unsafe or unsupported syntax: {type(n).__name__}", ctx=ctx)

    return float(V().visit(node))


# ------------------------------ units parsing ------------------------------

_FLOAT_RE = re.compile(
    r"""^\s*
    [+-]?
    (?:
        (?:\d+(?:\.\d*)?)|(?:\.\d+)
    )
    (?:[eE][+-]?\d+)?
    \s*$""",
    re.VERBOSE,
)


def _looks_like_plain_float(s: str) -> bool:
    return bool(_FLOAT_RE.match(s))


def try_parse_float_or_quantity(s: str, *, enable_units: bool = True) -> Optional[float]:
    """Try to parse text as a float or a unit-aware quantity.

    The parser first accepts plain finite floats and scientific notation. When
    ``enable_units`` is true, it then tries the optional TDPy units layer.

    Returns
    -------
    float | None
        Parsed finite value, or ``None`` when the text cannot be interpreted as
        a numeric value.
    """
    ss = preprocess_expr(str(s)).strip()
    if not ss:
        return None

    if _looks_like_plain_float(ss):
        try:
            v = float(ss)
            if not math.isfinite(v):
                return None
            return v
        except Exception:
            pass

    if not enable_units:
        return None

    try:
        from units import DEFAULT_REGISTRY, parse_quantity  # type: ignore
    except Exception:
        return None

    try:
        q = parse_quantity(ss, DEFAULT_REGISTRY)
        v = float(q.base_value())
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None
