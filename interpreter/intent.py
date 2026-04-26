from __future__ import annotations

"""Intent helpers for the TDPy interpreter package.

This module is the intention layer for converting parsed text into solver-ready
pieces.

Responsibilities
----------------
The module intentionally stays narrow:

* Normalize equation strings from EES-like syntax into solver-friendly syntax.
* Parse constant assignments from ``given`` sections.
* Parse guess lines in several human-friendly formats.
* Extract identifiers conservatively for unknown inference.
* Resolve numeric constants through safe multi-pass evaluation.

Design notes
------------
This module does not parse full text files. Full text parsing belongs in
``parse.py``. This module provides primitives used by ``build_spec.py``.

Thermodynamic-property calls are handled carefully. The identifier inference
logic treats common CoolProp and Cantera function names as functions rather than
unknown variables. It also treats common fluid-spec tokens as symbolic text so
numeric constant resolution does not emit unnecessary warnings for inputs such
as ``fluid = Helium`` or ``spec = gri30.yaml|X=CH4:1``.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .numeric_eval import NumericEvalError, safe_eval_numeric, try_parse_float_or_quantity

# Reuse solver-safe preprocessing when available.
try:
    from equations.safe_eval import preprocess_expr  # type: ignore
except Exception:  # pragma: no cover

    def preprocess_expr(s: str) -> str:
        """Minimal fallback preprocessing used when equations.safe_eval is unavailable."""
        return s.replace("^", "**")


# ------------------------------ regex ------------------------------

# name [=|==|:=] rhs
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(==|=|:=)\s*(.+?)\s*$")

# Guess forms such as "? x = 1", "guess: x = 1", and "init x = 1".
_GUESS_PREFIX_RE = re.compile(r"^\s*(\?|guess|init)\s*[:\s]+(.+)$", re.IGNORECASE)

# Inline guess form: "x ?= 1".
_GUESS_INLINE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\?=\s*(.+?)\s*$")

# Defensive directive filters.
_REPORT_LINE_RE = re.compile(r"^\s*report\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SOLVE_LINE_RE = re.compile(r"^\s*(solve|solver)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Optimizer directive-like lines. These are not equations; they are interpreted
# by build_spec.py when optimization mode is enabled. The RHS is optional so
# section-header forms such as "objective:" do not become equations.
_OBJECTIVE_LINE_RE = re.compile(r"^\s*(objective|minimize|maximize)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_CONSTRAINTS_LINE_RE = re.compile(r"^\s*(constraints|constraint)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_DESIGNVARS_LINE_RE = re.compile(
    r"^\s*(design_vars|designvars|design_variables|designvariables)\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)
_BOUNDS_LINE_RE = re.compile(r"^\s*(bounds|bound)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _is_directive_line(line: str) -> bool:
    """Return whether a line is a known directive rather than solver content."""
    s = _strip_inline_comment(line).strip()
    if not s:
        return False
    if _REPORT_LINE_RE.match(s) or _SOLVE_LINE_RE.match(s):
        return True
    if (
        _OBJECTIVE_LINE_RE.match(s)
        or _CONSTRAINTS_LINE_RE.match(s)
        or _DESIGNVARS_LINE_RE.match(s)
        or _BOUNDS_LINE_RE.match(s)
    ):
        return True
    return False


# Lightweight function-call detector for name extraction heuristics.
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


# ------------------------------ vocab ------------------------------

_BUILTIN_CONSTS: Set[str] = {"pi", "e"}

# Function names that should not be treated as variables. Keep aligned with the
# equations.safe_eval allowlist and interpreter.build_spec.
_MATH_FUNC_NAMES: Set[str] = {
    # core
    "abs",
    "min",
    "max",
    "pow",
    "clamp",
    # exp/logs
    "sqrt",
    "exp",
    "log",
    "ln",
    "log10",
    "log2",
    # trig
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    # hyperbolic
    "sinh",
    "cosh",
    "tanh",
    # rounding
    "floor",
    "ceil",
    # misc
    "hypot",
    "radians",
    "degrees",
    # thermo
    "PropsSI",
    "PhaseSI",
    "HAPropsSI",
    "CTPropsSI",
    "CTPropsMulti",
    "CTBatchProps",
    "ctprops_si",
    "ctprops_multi",
    "batch_ctprops",
    "cantera_available",
    "ctprops_cache_info",
    "clear_ctprops_caches",
    "ASPropsSI",
    "ASPropsMulti",
    "ASBatchProps",
    "as_props_si",
    "as_props_multi",
    "batch_as_props",
    "abstractstate_available",
    "FugacitySI",
    "FugacityCoeffSI",
    "LnFugacityCoeffSI",
    "ChemicalPotentialSI",
    "LiBrPropsSI",
    "LiBrH2OPropsSI",
    "LiBrPropsMulti",
    "LiBrBatchProps",
    "librh2o_props_si",
    "librh2o_props_multi",
    "batch_librh2o_props",
    "NH3H2O",
    "NH3H2O_STATE",
    "NH3H2O_TPX",
    "NH3H2O_STATE_TPX",
    "NH3H2OPropsSI",
    "NH3H2OPropsMulti",
    "NH3H2OBatchProps",
    "nh3h2o_available",
    "state_tpx",
    "prop_tpx",
    "props_multi_tpx",
    "batch_prop_tpx",
    # special
    "erf",
    "erfc",
    "gamma",
    "lgamma",
}

# Words that users might type that should never become unknowns.
_RESERVED_WORDS: Set[str] = {
    "title",
    "given",
    "givens",
    "constants",
    "const",
    "params",
    "parameters",
    "guess",
    "guesses",
    "init",
    "inits",
    "variables",
    "vars",
    "equations",
    "eqs",
    "report",
    "output",
    "solve",
    "solver",
    "objective",
    "minimize",
    "maximize",
    "constraints",
    "constraint",
    "design_vars",
    "designvars",
    "design_variables",
    "designvariables",
    "bounds",
    "bound",
    "note",
    "notes",
    "units",
}

# Lowercased views for case-insensitive filtering.
_BUILTIN_CONSTS_LC = {s.lower() for s in _BUILTIN_CONSTS}
_RESERVED_WORDS_LC = {s.lower() for s in _RESERVED_WORDS}
_MATH_FUNC_NAMES_LC = {s.lower() for s in _MATH_FUNC_NAMES}


# ------------------------------ helpers ------------------------------

def _strip_string_literals(expr: str) -> str:
    """Remove content inside quoted strings before identifier extraction."""
    out: list[str] = []
    q: str | None = None
    esc = False

    for ch in expr:
        if q is None:
            if ch in ("'", '"'):
                q = ch
                out.append(" ")
            else:
                out.append(ch)
        else:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == q:
                q = None
                out.append(" ")
            else:
                continue

    return "".join(out)


def _strip_inline_comment(line: str) -> str:
    """Strip ``#`` and ``//`` inline comments while respecting quoted strings."""
    s = line
    out: list[str] = []
    q: str | None = None
    esc = False
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if q is None:
            if ch in ("'", '"'):
                q = ch
                out.append(ch)
                i += 1
                continue

            if ch == "#":
                break
            if ch == "/" and i + 1 < n and s[i + 1] == "/":
                break

            out.append(ch)
            i += 1
            continue

        # inside quotes
        if esc:
            esc = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\":
            esc = True
            out.append(ch)
            i += 1
            continue
        if ch == q:
            q = None
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


# This token regex is used only to suppress numeric-eval warnings for
# fluid/mechanism specs. It does not affect solver parsing.
_SYMBOL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_:\-./|=,+\[\]\(\)]+$")


def _is_symbolic_constant_rhs(rhs: str) -> bool:
    """Return whether an RHS looks like a symbolic text constant."""
    r = rhs.strip()
    if not r:
        return False

    if (len(r) >= 2) and ((r[0] == r[-1]) and r[0] in ("'", '"')):
        return True

    if any(ch.isspace() for ch in r):
        return False

    if any(op in r for op in ("+", "*", "/", "^")):
        if "*" in r or "/" in r or "^" in r:
            return False

    if not _SYMBOL_TOKEN_RE.fullmatch(r):
        return False

    return any(ch.isalpha() for ch in r)


# ------------------------------ data model ------------------------------

@dataclass
class IntentDraft:
    """Optional richer structure for future one-shot text interpretation.

    The current interpreter path uses ``build_spec.py`` as the main builder.
    This dataclass is retained as a stable intermediate shape for future
    extensions and for callers that may already import it.
    """

    title: Optional[str]
    equations: List[str]
    constants_expr: Dict[str, str]
    constants_val: Dict[str, float]
    guesses: Dict[str, float]
    report: List[str]
    solve_overrides: Dict[str, Any]
    warnings: List[str]
    reserved_words: Set[str] = field(default_factory=lambda: set(_RESERVED_WORDS) | set(_BUILTIN_CONSTS))
    func_names: Set[str] = field(default_factory=lambda: set(_MATH_FUNC_NAMES))


# ------------------------------ parsing primitives ------------------------------

def parse_guess_line(line: str, *, enable_units: bool = True) -> Optional[Tuple[str, float]]:
    """Parse a human-friendly guess line.

    Supported forms include ``"? x = 1.2"``, ``"guess: x = 1.2"``,
    ``"init x = 1.2"``, ``"x ?= 1.2"``, and ``"x = 1.2"`` when the right-hand
    side is numeric or a parseable quantity.

    Returns
    -------
    tuple[str, float] | None
        ``(name, value)`` when parsing succeeds; otherwise ``None``.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return None

    if _is_directive_line(s):
        return None

    m_inline = _GUESS_INLINE_RE.match(s)
    if m_inline:
        name = m_inline.group(1)
        val = m_inline.group(2).strip()
        fv = try_parse_float_or_quantity(val, enable_units=enable_units)
        if fv is None:
            return None
        return name, float(fv)

    m_pref = _GUESS_PREFIX_RE.match(s)
    if m_pref:
        rest = m_pref.group(2).strip()
        m = _ASSIGN_RE.match(rest)
        if not m:
            parts = rest.split()
            if len(parts) >= 2:
                name = parts[0]
                val = " ".join(parts[1:])
                fv = try_parse_float_or_quantity(val, enable_units=enable_units)
                if fv is not None and re.fullmatch(r"[A-Za-z_]\w*", name or ""):
                    return name, float(fv)
            return None

        name = m.group(1)
        rhs = m.group(3).strip()
        fv = try_parse_float_or_quantity(rhs, enable_units=enable_units)
        if fv is None:
            return None
        return name, float(fv)

    m = _ASSIGN_RE.match(s)
    if m:
        name = m.group(1)
        rhs = m.group(3).strip()
        fv = try_parse_float_or_quantity(rhs, enable_units=enable_units)
        if fv is None:
            return None
        return name, float(fv)

    return None


def parse_constant_assignment(line: str) -> Optional[Tuple[str, str]]:
    """Parse a constant assignment and return the name and RHS expression.

    Accepted forms include ``"g = 9.81"`` and ``"A := pi*r^2"``. The right-hand
    side is returned after solver-safe preprocessing.

    The function does not decide whether the right-hand side is numeric or
    symbolic. That decision is made by the builder layer.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return None

    if _is_directive_line(s):
        return None

    m = _ASSIGN_RE.match(s)
    if not m:
        return None

    name = m.group(1)
    rhs = preprocess_expr(m.group(3).strip())
    return name, rhs


def looks_like_equation(line: str) -> bool:
    """Return whether a line should stay in the equation stream.

    Empty lines, comment-only lines, ``report`` directives, and ``solve``
    directives return ``False``. Other lines return ``True``.

    Optimizer directives such as ``maximize:``, ``bounds:``, and
    ``design_vars:`` are allowed through the equation stream so ``build_spec.py``
    can peel them off and construct optimization specs.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return False
    if _REPORT_LINE_RE.match(s) or _SOLVE_LINE_RE.match(s):
        return False
    return True


def normalize_equation(line: str) -> str:
    """Normalize common user syntax for solver consumption."""
    s = _strip_inline_comment(line).strip()
    if not s:
        return ""

    s = preprocess_expr(s)

    if ":=" in s:
        s = s.replace(":=", "=")
    if "==" in s:
        s = s.replace("==", "=")

    return s.strip()


# ------------------------------ identifier extraction ------------------------------

def extract_names_fallback(expr: str) -> Set[str]:
    """Extract likely variable names from an expression.

    The extractor strips comments and quoted string literals before scanning.
    It filters built-in constants, reserved directive words, known function
    names, and dunder-style names.
    """
    expr0 = _strip_inline_comment(expr)
    expr2 = _strip_string_literals(expr0)

    names = set(re.findall(r"[A-Za-z_]\w*", expr2))

    names = {
        n
        for n in names
        if n.lower() not in _BUILTIN_CONSTS_LC and n.lower() not in _RESERVED_WORDS_LC
    }

    for fn in _CALL_RE.findall(expr2):
        if fn.lower() in _MATH_FUNC_NAMES_LC:
            names.discard(fn)

    names = {
        n
        for n in names
        if n.lower() not in _MATH_FUNC_NAMES_LC and not n.startswith("__")
    }
    return names


# ------------------------------ constant resolution ------------------------------

def resolve_constants(
    constants_expr: Mapping[str, str],
    *,
    enable_units: bool = True,
) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    """Resolve numeric constants with a multi-pass strategy.

    Resolution order is:

    1. Direct numeric or unit-aware parsing.
    2. Safe numeric evaluation using constants that have already been resolved.
    3. Preservation of unresolved expressions for the caller.

    Returns
    -------
    tuple[dict[str, float], dict[str, str], list[str]]
        Resolved numeric constants, unresolved expression constants, and
        warning messages.
    """
    warnings: List[str] = []
    resolved: Dict[str, float] = {}
    unresolved: Dict[str, str] = {k: preprocess_expr(v.strip()) for k, v in dict(constants_expr).items()}

    for k, rhs in list(unresolved.items()):
        fv = try_parse_float_or_quantity(rhs, enable_units=enable_units)
        if fv is not None:
            resolved[k] = float(fv)
            unresolved.pop(k, None)

    progress = True
    it = 0
    while progress and unresolved and it < 50:
        it += 1
        progress = False

        for k, rhs in list(unresolved.items()):
            if _is_symbolic_constant_rhs(rhs):
                continue
            try:
                v = safe_eval_numeric(rhs, names=resolved)
            except NumericEvalError:
                continue
            except Exception as e:
                warnings.append(f"Failed to evaluate constant {k}={rhs!r}: {e}")
                continue

            resolved[k] = float(v)
            unresolved.pop(k, None)
            progress = True

    for k, rhs in unresolved.items():
        if _is_symbolic_constant_rhs(rhs):
            continue
        refs = extract_names_fallback(rhs)
        refs = {r for r in refs if r not in resolved and r.lower() not in _BUILTIN_CONSTS_LC}
        if refs:
            warnings.append(f"Constant {k!r} could not be resolved; references unknown names: {sorted(refs)}")

    return resolved, unresolved, warnings
