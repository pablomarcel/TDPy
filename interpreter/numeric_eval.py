from __future__ import annotations

"""
interpreter.numeric_eval

Safe numeric evaluation for the interpreter layer.

Goals:
- Evaluate numeric constants and constant-expressions safely (AST-whitelisted).
- Support simple unit parsing (via units) when enabled.
- Provide helpful error messages for end users ("what token failed and why").
- Stay consistent with equation preprocessing (e.g., '^' -> '**') when available.

IMPORTANT SECURITY NOTES
- No attribute access (obj.x), no subscripts (a[0]), no comprehensions, no lambdas,
  no imports, no calls except a strict whitelist, no starred args/kwargs.
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
        # minimal fallback: handle ^ as power
        return s.replace("^", "**")


# ------------------------------ whitelist configuration ------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def default_numeric_funcs() -> Dict[str, Any]:
    """
    Numeric-only functions, safe to call:
    - no IO, no attributes, no imports.
    - everything returns numeric results (or raises).
    """
    funcs: Dict[str, Any] = {
        # core
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),

        # exponentials / logs
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "log2": getattr(math, "log2", None),

        # trig
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,

        # hyperbolic
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
    expr: str
    where: str = "numeric expression"


class NumericEvalError(ValueError):
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
    """
    Evaluate a numeric expression safely.

    Allowed:
      - numeric literals: 1, 1.2, 1e-3
      - names: must exist in `names` (resolved constants)
      - operators: + - * / ** % and parentheses
      - calls to whitelisted numeric funcs: sin(x), log(x), clamp(x, lo, hi), ...

    Disallowed:
      - attribute access, indexing, comprehensions, lambdas, f-strings
      - assignments, comparisons, boolean ops, if-expressions
      - any call to non-whitelisted function
      - starred args/kwargs
    """
    raw = expr
    s = preprocess_expr(str(expr)).strip()
    ctx = NumericEvalContext(expr=s)

    if not s:
        raise NumericEvalError("Empty expression", ctx=ctx)

    fns = dict(default_numeric_funcs())
    if funcs:
        # allow caller to add/override with additional numeric-safe functions
        fns.update(dict(funcs))

    consts: Dict[str, float] = dict(names)
    consts.setdefault("pi", float(math.pi))
    consts.setdefault("e", float(math.e))

    try:
        node = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise NumericEvalError("Invalid syntax", ctx=ctx) from e

    class V(ast.NodeVisitor):
        def visit_Expression(self, n: ast.Expression) -> float:
            return float(self.visit(n.body))

        def visit_Constant(self, n: ast.Constant) -> float:
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise NumericEvalError(f"Non-numeric literal not allowed: {n.value!r}", ctx=ctx)

        # Python <3.8 compatibility (not needed for you, but harmless)
        def visit_Num(self, n: ast.Num) -> float:  # pragma: no cover
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
                return a ** b
            if isinstance(n.op, ast.Mod):
                return a % b
            raise NumericEvalError("Unhandled operator", ctx=ctx)

        def visit_Call(self, n: ast.Call) -> float:
            if not isinstance(n.func, ast.Name):
                raise NumericEvalError("Only direct calls f(x) are allowed", ctx=ctx)

            fn = n.func.id
            if fn not in fns:
                raise NumericEvalError(f"Function not allowed: {fn!r}", ctx=ctx)

            # Disallow starargs / kwargs expansion.
            # In Python AST: ast.Starred in args, and keyword.arg is None for **kwargs.
            for a in n.args:
                if isinstance(a, ast.Starred):
                    raise NumericEvalError("Star-args (*args) are not allowed", ctx=ctx)
            for kw in n.keywords:
                if kw.arg is None:
                    raise NumericEvalError("Star-kwargs (**kwargs) are not allowed", ctx=ctx)

            args = [float(self.visit(a)) for a in n.args]
            kwargs = {str(kw.arg): float(self.visit(kw.value)) for kw in n.keywords}  # type: ignore[arg-type]
            try:
                return float(fns[fn](*args, **kwargs))
            except Exception as e:
                raise NumericEvalError(f"Call failed: {fn}(...) -> {e}", ctx=ctx) from e

        # Explicitly block common unsafe nodes with clearer messages
        def visit_Attribute(self, n: ast.Attribute) -> float:
            raise NumericEvalError("Attribute access is not allowed (obj.x)", ctx=ctx)

        def visit_Subscript(self, n: ast.Subscript) -> float:
            raise NumericEvalError("Indexing is not allowed (a[0])", ctx=ctx)

        def visit_Lambda(self, n: ast.Lambda) -> float:  # pragma: no cover
            raise NumericEvalError("Lambda is not allowed", ctx=ctx)

        def visit_Compare(self, n: ast.Compare) -> float:
            raise NumericEvalError("Comparisons are not allowed in numeric constants", ctx=ctx)

        def visit_BoolOp(self, n: ast.BoolOp) -> float:
            raise NumericEvalError("Boolean ops are not allowed in numeric constants", ctx=ctx)

        def visit_IfExp(self, n: ast.IfExp) -> float:
            raise NumericEvalError("Conditional expressions are not allowed", ctx=ctx)

        def generic_visit(self, n: ast.AST) -> float:
            raise NumericEvalError(f"Unsafe/unsupported syntax: {type(n).__name__}", ctx=ctx)

    return float(V().visit(node))


# ------------------------------ units parsing ------------------------------

_FLOAT_RE = re.compile(
    r"""^\s*
    [+-]?
    (?:
        (?:\d+(?:\.\d*)?)|(?:\.\d+)
    )
    (?:[eE][+-]?\d+)?    # exponent
    \s*$""",
    re.VERBOSE,
)


def _looks_like_plain_float(s: str) -> bool:
    return bool(_FLOAT_RE.match(s))


def try_parse_float_or_quantity(s: str, *, enable_units: bool = True) -> Optional[float]:
    """
    Try parse, in order:
      1) plain float / scientific notation
      2) unit-aware quantity via units (if enabled)

    Returns None if not parseable.

    Notes:
    - we preprocess '^' -> '**' for consistency, but quantities are typically "300 K"
      and won't contain '^' anyway.
    """
    ss = preprocess_expr(str(s)).strip()
    if not ss:
        return None

    # quick float (strict regex avoids float("nan") / float("inf") surprises)
    if _looks_like_plain_float(ss):
        try:
            v = float(ss)
            # reject NaN/inf (they poison constant resolution)
            if not math.isfinite(v):
                return None
            return v
        except Exception:
            pass

    if not enable_units:
        return None

    # optional units support
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
