# equations/safe_eval.py
from __future__ import annotations

"""Safe expression evaluation for EES-style equation strings.

This module provides a small dependency-free evaluation layer for equation and
expression strings used by the TDPy equation solver.

Use cases
---------
The module supports:

* Converting an equation such as ``"P2 = P1 + dp"`` into a residual expression.
* Compiling and evaluating residual expressions with a restricted AST whitelist.
* Extracting referenced variable names for validation and unknown selection.
* Evaluating warm-start right-hand-side expressions safely.

Security model
--------------
Expressions are parsed with ``ast`` and rejected when they contain unsupported
or unsafe syntax. Attribute access, subscripts, comprehensions, lambdas,
method calls, imports, comparisons, boolean operations, and other non-arithmetic
constructs are not allowed.

Evaluation uses::

    eval(code, {"__builtins__": {}}, safe_scope)

String-literal policy
---------------------
Short string literals are allowed because thermodynamic-property helper calls
often require keys such as ``"T"``, ``"P"``, or ``"D"``. To reduce misuse,
strings are blocked from arithmetic operations such as string concatenation and
are limited by ``_MAX_STRING_LITERAL_LEN``.

Thermodynamic helper calls
--------------------------
The solver may evaluate controlled property helper calls supplied by the runtime
scope, including CoolProp, Cantera, LiBr-H2O, and NH3-H2O helper names. These
names are included in a conservative allowlist so the AST validator accepts the
call syntax. The function object must still be provided by the runtime scope
for evaluation to succeed.
"""

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple


# ------------------------------ errors ------------------------------

class UnsafeExpressionError(ValueError):
    """Raised when an expression contains unsupported or unsafe syntax."""


class ParseError(ValueError):
    """Raised when an equation or expression cannot be parsed or evaluated."""


# ------------------------------ limits ------------------------------

_MAX_EXPR_LEN = 10_000
_MAX_STRING_LITERAL_LEN = 256
_MAX_CONST_POW_EXP = 1_000


# ------------------------------ safe namespace ------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sign(x: float) -> float:
    return -1.0 if x < 0 else (1.0 if x > 0 else 0.0)


def _step(x: float) -> float:
    return 0.0 if x < 0 else 1.0


def _cbrt(x: float) -> float:
    f = getattr(math, "cbrt", None)
    if callable(f):
        return float(f(x))
    if x == 0:
        return 0.0
    return float(math.copysign(abs(x) ** (1.0 / 3.0), x))


_SAFE_FUNCS_BASE: Dict[str, Any] = {
    # basic
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "clamp": _clamp,
    "sign": _sign,
    "step": _step,

    # exp/logs
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "exp": math.exp,
    "expm1": getattr(math, "expm1", None),
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": getattr(math, "log2", None),
    "log1p": getattr(math, "log1p", None),

    # trig
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": getattr(math, "atan2", None),

    # hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,

    # rounding
    "floor": math.floor,
    "ceil": math.ceil,

    # misc
    "hypot": getattr(math, "hypot", None),
    "degrees": getattr(math, "degrees", None),
    "radians": getattr(math, "radians", None),
    "erf": getattr(math, "erf", None),
    "erfc": getattr(math, "erfc", None),
    "gamma": getattr(math, "gamma", None),
    "lgamma": getattr(math, "lgamma", None),
}

SAFE_FUNCS: Dict[str, Callable[..., Any]] = {
    k: v for k, v in _SAFE_FUNCS_BASE.items() if callable(v)
}

SAFE_CONSTS: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": getattr(math, "tau", 2.0 * math.pi),
    "inf": float("inf"),
    "nan": float("nan"),

    # common aliases
    "PI": math.pi,
    "E": math.e,
}

# Conservative allowlist of callable names that are permitted to appear as
# f(...), even if the compiler is not explicitly told about them through
# extra_funcs. Evaluation still requires the actual function to be present in
# the runtime scope.
_SAFE_CALL_NAMES: Set[str] = {
    # CoolProp and humid air
    "PropsSI",
    "PhaseSI",
    "HAPropsSI",

    # Cantera backend
    "CTPropsSI",
    "CTPropsMulti",
    "CTBatchProps",

    # Optional internal Cantera aliases
    "ctprops_si",
    "ctprops_multi",
    "batch_ctprops",
    "cantera_available",

    # Optional Cantera cache helpers
    "ctprops_cache_info",
    "clear_ctprops_caches",

    # CoolProp AbstractState wrappers
    "ASPropsSI",
    "ASPropsMulti",
    "ASBatchProps",

    # Convenience wrappers
    "FugacitySI",
    "FugacityCoeffSI",
    "LnFugacityCoeffSI",
    "ChemicalPotentialSI",

    # Optional internal AbstractState aliases
    "as_props_si",
    "as_props_multi",
    "batch_as_props",

    # Optional humid-air aliases
    "ha_props_si",
    "ha_props_multi",
    "batch_ha_props",

    # LiBr-H2O entry points and helper calls
    "LiBrPropsSI",
    "LiBrH2OPropsSI",
    "LiBrX_TP",
    "LiBrH_TX",
    "LiBrRho_TXP",
    "LiBrT_HX",
    "LiBrPropsMulti",
    "LiBrBatchProps",
    "librh2o_props_si",
    "librh2o_props_multi",
    "batch_librh2o_props",

    # NH3-H2O backend and helper calls
    "NH3H2O_TPX",
    "NH3H2O_STATE_TPX",
    "NH3H2O",
    "NH3H2O_STATE",
    "prop_tpx",
    "state_tpx",
    "props_multi_tpx",
    "batch_prop_tpx",
    "NH3H2OPropsSI",
    "NH3H2OPropsMulti",
    "NH3H2OBatchProps",

    # Availability helpers
    "abstractstate_available",
    "nh3h2o_available",
}

_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
)

_ALLOWED_UNARYOPS = (
    ast.UAdd,
    ast.USub,
)

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
)


# ------------------------------ public API ------------------------------

@dataclass(frozen=True)
class CompiledExpr:
    """Compiled expression metadata and code object.

    Parameters
    ----------
    raw:
        Original user expression string. This may include an equals sign when
        compiled through ``compile_residual``.
    residual:
        Normalized residual expression string. For plain expressions, this is
        the expression itself.
    code:
        Compiled Python code object ready for restricted evaluation.
    names:
        Referenced identifiers after excluding safe functions, safe constants,
        and injected extras.
    """

    raw: str
    residual: str
    code: Any
    names: List[str]


def preprocess_expr(s: str) -> str:
    """Apply minimal EES-style preprocessing to an expression.

    The preprocessing performs these transformations:

    * ``^`` becomes ``**``.
    * Unicode minus becomes ``-``.
    * Unicode multiplication symbols become ``*``.
    * Unicode division becomes ``/``.
    * Leading and trailing whitespace are stripped.
    """
    if s is None:
        raise ParseError("Expression is None.")
    out = str(s).strip()
    if len(out) > _MAX_EXPR_LEN:
        raise ParseError(f"Expression too long ({len(out)} chars). Limit is {_MAX_EXPR_LEN}.")
    out = out.replace("−", "-")
    out = out.replace("^", "**")
    out = out.replace("×", "*").replace("·", "*").replace("⋅", "*")
    out = out.replace("÷", "/")
    return out


def is_identifier(name: str) -> bool:
    """Return whether ``name`` is a simple Python-style identifier.

    The accepted pattern is an ASCII-style identifier beginning with a letter or
    underscore and followed by letters, digits, or underscores.
    """
    if not isinstance(name, str):
        return False
    s = name.strip()
    if not s:
        return False
    if not (s[0].isalpha() or s[0] == "_"):
        return False
    for ch in s:
        if not (ch.isalnum() or ch == "_"):
            return False
    return True


def split_assignment(eq: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect a simple assignment expression.

    The accepted shape is ``lhs = rhs`` where ``lhs`` is a plain identifier.
    If the expression is not a simple assignment, ``(None, None)`` is returned.
    """
    s = preprocess_expr(eq)

    if "==" in s and "=" not in s.replace("==", ""):
        s = s.replace("==", "=")

    if "=" not in s:
        return None, None

    lhs, rhs = s.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    if not lhs or not rhs:
        return None, None

    if not is_identifier(lhs):
        return None, None

    return lhs, rhs


def normalize_equation_to_residual(eq: str) -> str:
    """Convert an equation into a residual expression.

    Examples
    --------
    ``"P2 = P1 + dp"`` becomes ``"(P2) - (P1 + dp)"``.

    ``"a == b"`` becomes ``"(a) - (b)"``.

    ``"f(x)"`` is returned as ``"f(x)"`` and interpreted as an existing
    residual expression.
    """
    s = preprocess_expr(eq)

    if "==" in s and "=" not in s.replace("==", ""):
        s = s.replace("==", "=")

    if "=" not in s:
        return s

    lhs, rhs = s.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    if not lhs or not rhs:
        raise ParseError(f"Malformed equation with '=': {eq!r}")

    return f"({lhs}) - ({rhs})"


class _Validator(ast.NodeVisitor):
    """Strict AST validator for safe arithmetic expressions.

    Allowed syntax includes numeric constants, short string constants, names,
    arithmetic binary and unary operations, and direct calls to approved
    function names.
    """

    def __init__(
        self,
        *,
        allow_strings: bool = True,
        extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
        extra_consts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.allow_strings = bool(allow_strings)
        self.extra_funcs = set((extra_funcs or {}).keys())
        self.extra_consts = set((extra_consts or {}).keys())
        self.names: Set[str] = set()

    def generic_visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpressionError(
                f"Unsupported syntax node: {type(node).__name__}. "
                "Only arithmetic expressions and approved function calls are allowed."
            )
        return super().generic_visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        self.names.add(str(node.id))

    def visit_Constant(self, node: ast.Constant) -> Any:
        val = node.value
        if isinstance(val, (int, float)):
            return
        if isinstance(val, str):
            if not self.allow_strings:
                raise UnsafeExpressionError("String literals are not allowed here.")
            if len(val) > _MAX_STRING_LITERAL_LEN:
                raise UnsafeExpressionError(
                    f"String literal too long ({len(val)} > {_MAX_STRING_LITERAL_LEN})."
                )
            return
        raise UnsafeExpressionError(
            f"Unsupported constant type: {type(val).__name__}. Only int, float, or short string is allowed."
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError(f"Unsupported unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError(f"Unsupported binary operator: {type(node.op).__name__}")

        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                expv = float(node.right.value)
                if abs(expv) > _MAX_CONST_POW_EXP:
                    raise UnsafeExpressionError(
                        f"Exponent too large ({expv}); limit is ±{_MAX_CONST_POW_EXP}."
                    )

        if isinstance(node.op, ast.Add):
            if (
                isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
            ) or (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
            ):
                raise UnsafeExpressionError("String concatenation is not allowed.")

        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError(
                "Only direct function calls like f(x) are allowed. "
                "Attribute access and method calls are forbidden."
            )

        fname = str(node.func.id)
        allowed = (
            fname in SAFE_FUNCS
            or fname in self.extra_funcs
            or fname in _SAFE_CALL_NAMES
        )
        if not allowed:
            raise UnsafeExpressionError(
                f"Function {fname!r} is not allowed. "
                "Use math-like safe functions or approved thermo helper names only."
            )

        for kw in node.keywords:
            if kw.arg is None:
                raise UnsafeExpressionError("Star-args are not allowed in function calls.")
            self.visit(kw.value)

        for arg in node.args:
            self.visit(arg)


def _compile_checked(
    expr: str,
    *,
    allow_strings: bool = True,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> CompiledExpr:
    src = preprocess_expr(expr)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ParseError(f"Could not parse expression: {expr!r}: {e}") from e

    validator = _Validator(
        allow_strings=allow_strings,
        extra_funcs=extra_funcs,
        extra_consts=extra_consts,
    )
    validator.visit(tree)

    try:
        code = compile(tree, "<equation>", "eval")
    except Exception as e:
        raise ParseError(f"Could not compile expression: {expr!r}: {e}") from e

    excluded = set(SAFE_FUNCS.keys()) | set(SAFE_CONSTS.keys())
    excluded |= set((extra_funcs or {}).keys())
    excluded |= set((extra_consts or {}).keys())
    names = sorted(n for n in validator.names if n not in excluded)

    return CompiledExpr(raw=expr, residual=src, code=code, names=names)


def compile_expression(
    expr: str,
    *,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> CompiledExpr:
    """Compile a plain arithmetic expression without equation normalization."""
    return _compile_checked(
        expr,
        allow_strings=True,
        extra_funcs=extra_funcs,
        extra_consts=extra_consts,
    )


def compile_residual(
    eq: str,
    *,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> CompiledExpr:
    """Compile an equation into a residual expression.

    If ``eq`` has no equals sign, it is treated as an existing residual
    expression.
    """
    residual = normalize_equation_to_residual(eq)
    c = _compile_checked(
        residual,
        allow_strings=True,
        extra_funcs=extra_funcs,
        extra_consts=extra_consts,
    )
    return CompiledExpr(raw=eq, residual=residual, code=c.code, names=c.names)


def _build_scope(
    *,
    values: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    scope.update(SAFE_FUNCS)
    scope.update(SAFE_CONSTS)

    if extra_consts:
        scope.update(dict(extra_consts))
    if extra_funcs:
        scope.update(dict(extra_funcs))

    if params:
        scope.update(dict(params))
    if values:
        scope.update(dict(values))

    return scope


def eval_compiled(
    c: CompiledExpr,
    *,
    values: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> float:
    """Evaluate a compiled expression safely and coerce the result to float."""
    scope = _build_scope(
        values=values,
        params=params,
        extra_funcs=extra_funcs,
        extra_consts=extra_consts,
    )
    try:
        out = eval(c.code, {"__builtins__": {}}, scope)
    except Exception as e:
        raise ParseError(f"Error evaluating expression {c.raw!r}: {e}") from e

    try:
        return float(out)
    except Exception as e:
        raise ParseError(
            f"Expression {c.raw!r} did not evaluate to a numeric value; got {type(out).__name__}: {out!r}"
        ) from e


def eval_expression(
    expr: str,
    *,
    values: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    extra_funcs: Optional[Mapping[str, Callable[..., Any]]] = None,
    extra_consts: Optional[Mapping[str, Any]] = None,
) -> float:
    """Compile and evaluate a plain expression."""
    c = compile_expression(
        expr,
        extra_funcs=extra_funcs,
        extra_consts=extra_consts,
    )
    return eval_compiled(
        c,
        values=values,
        params=params,
        extra_funcs=extra_funcs,
        extra_consts=extra_consts,
    )


__all__ = [
    "CompiledExpr",
    "ParseError",
    "SAFE_CONSTS",
    "SAFE_FUNCS",
    "UnsafeExpressionError",
    "compile_expression",
    "compile_residual",
    "eval_compiled",
    "eval_expression",
    "is_identifier",
    "normalize_equation_to_residual",
    "preprocess_expr",
    "split_assignment",
]
