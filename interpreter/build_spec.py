# interpreter/build_spec.py
from __future__ import annotations

"""
interpreter.build_spec

Turn ParsedInput (from parse.py) into a TDPy equation-system JSON spec.

Core goals:
- Robust against chaotic user input
- Pull "given" assignments into constants when safe
- Infer unknowns from equations without accidentally inventing variables
  (e.g., "report:" or "solve:" lines, math function names, strings inside quotes, etc.)
- Preserve user intent with warnings when we ignore or reinterpret something

Important mixed-thermo behavior:
- Lines like `fluid = Helium` are treated as a *symbol assignment* (string-like),
  NOT an equation. We will substitute `fluid` with `"Helium"` in later equations,
  so the system remains square and we don't invent `fluid`/`Helium` unknowns.

Assignment pulling rule (equation stream):
- We avoid "pulling" assignment-style equations that call thermo property functions
  (PropsSI/HAPropsSI/LiBrPropsSI/CTPropsSI/...) into numeric constants. Those lines should
  remain equations so the solver can warm-start / safely evaluate them.

Thermo-in-given rule (EES-like):
- If a line appears in `given:` and its RHS is a thermo/property call **and**
  it depends only on already-known constants/symbols, we attempt to evaluate it
  immediately into a numeric constant using `equations.safe_eval` + injected
  thermo-call wrappers. This allows "standard-state precomputes" to behave like
  EES and avoids slow repeated thermo calls in the nonlinear solve loop.

Optimizer upgrade note (Feb 2026):
- Recognize optimizer directives/sections:
    * objective: / minimize: / maximize:
    * constraints:
    * design_vars: (optional)
    * bounds: (optional)
- When an objective is present, emit spec with problem_type='optimize', including
  objective/sense/constraints while preserving legacy equation-system behavior.

Robustness patch (Feb 2026):
- bounds directives often appear as one-liners like:
    bounds: x: [0, 1]; y: [0, 1]
  We split *top-level* items (not inside [..] or (..)) so all variables receive bounds.

LATEST FACTS / STABILITY PATCH (Feb 2026):
- Solve overrides coming from .txt files often arrive as strings (because parse.py keeps values as text).
  The solver expects real types for some keys:
    warm_start, auto_guess, use_units (bool)
    warm_start_passes (int)
    thermo_penalty (float)
  If left as strings, Python truthiness (e.g., bool("0") == True) can silently enable expensive
  warm-start/auto-guess loops and make solvers appear to "hang" (especially with Cantera calls).
  This module now coerces these keys robustly.
"""

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .models import InterpretConfig, InterpretResult, ParsedInput
from .intent import (
    looks_like_equation,
    normalize_equation,
    parse_constant_assignment,
    parse_guess_line,
    resolve_constants,
)

# Numeric constant evaluator (used to re-resolve expressions once thermo givens fold)
try:
    from .numeric_eval import NumericEvalError, safe_eval_numeric, try_parse_float_or_quantity  # type: ignore
except Exception:  # pragma: no cover
    NumericEvalError = Exception  # type: ignore
    safe_eval_numeric = None  # type: ignore
    try_parse_float_or_quantity = None  # type: ignore

# Safe evaluator that supports controlled function injection (PropsSI/CTPropsSI/etc.)
try:
    from equations.safe_eval import compile_expression, eval_expression, ParseError, preprocess_expr  # type: ignore
except Exception:  # pragma: no cover
    compile_expression = None  # type: ignore
    eval_expression = None  # type: ignore
    ParseError = Exception  # type: ignore
    preprocess_expr = None  # type: ignore


# ------------------------------ local vocab ------------------------------

_BUILTIN_CONSTS: Set[str] = {"pi", "e"}

# Common directive / section words that should NEVER become unknowns
_RESERVED_WORDS: Set[str] = {
    # sections
    "title",
    "given", "givens", "constants", "const", "params", "parameters",
    "guess", "guesses", "init", "inits", "variables", "vars",
    "equations", "eqs",
    "report", "output",
    "solve", "solver",
    "objective", "minimize", "maximize",
    "constraints", "constraint",
    "design_vars", "designvars", "design_variables", "designvariables",
    "bounds", "bound",
    # common “english-ish” tokens users may type
    "note", "notes", "units",
}

# Math / helper functions we allow in equation strings; should not become unknowns
# Keep aligned with interpreter.intent and equations.safe_eval allowlist.
_KNOWN_FUNC_NAMES: Set[str] = {
    # core
    "abs", "min", "max", "pow", "clamp",
    # exp/logs
    "sqrt", "exp", "log", "ln", "log10", "log2",
    # trig
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    # hyperbolic
    "sinh", "cosh", "tanh",
    # rounding
    "floor", "ceil",
    # misc
    "hypot", "degrees", "radians",
    # special
    "erf", "erfc", "gamma", "lgamma",

    # thermo (CoolProp + humid air)
    "PropsSI", "PhaseSI", "HAPropsSI",
    # CoolProp AbstractState wrappers (fugacity, etc.)
    "ASPropsSI",
    # Cantera (equilibrium / reacting mixtures) PropsSI-like interface
    "CTPropsSI", "CTPropsMulti", "CTBatchProps",
    "ctprops_si", "ctprops_multi", "batch_ctprops",
    "cantera_available",
    # fugacity helpers
    "FugacitySI", "FugacityCoeffSI", "LnFugacityCoeffSI", "ChemicalPotentialSI",
    # optional internal/alternate names
    "abstractstate_available", "as_props_si", "as_props_multi", "batch_as_props",

    # LiBr–H2O (ASHRAE-style property engine; PropsSI-like)
    "LiBrPropsSI", "LiBrH2OPropsSI",
    "LiBrPropsMulti", "LiBrBatchProps",
    # LiBr convenience helpers (common user-facing wrappers)
    "LiBrX_TP", "LiBrH_TX", "LiBrRho_TXP", "LiBrT_HX",
    # optional internal/alternate names (safe to ignore for unknown inference)
    "librh2o_props_si", "librh2o_props_multi", "batch_librh2o_props",

    # NH3–H2O (Ibrahim & Klein 1993) native property engine (CoolProp-like signature)
    "NH3H2O", "NH3H2O_STATE", "NH3H2O_TPX", "NH3H2O_STATE_TPX",
    # Optional PropsSI-like aliases
    "NH3H2OPropsSI", "NH3H2OPropsMulti", "NH3H2OBatchProps",
    # optional internal/alternate names
    "nh3h2o_available",
    "state_tpx", "prop_tpx", "props_multi_tpx", "batch_prop_tpx",
}

# Thermo/property function names we treat as "do not constant-fold at interpret time"
# in the equation stream (solver will handle safe evaluation / warm-start).
_THERMO_CALL_NAMES: Set[str] = {
    # CoolProp pure + HA
    "PropsSI", "PhaseSI", "HAPropsSI",
    # CoolProp AbstractState wrappers
    "ASPropsSI",
    # Cantera
    "CTPropsSI", "CTPropsMulti", "CTBatchProps",
    "ctprops_si", "ctprops_multi", "batch_ctprops",
    # fugacity helpers
    "FugacitySI", "FugacityCoeffSI", "LnFugacityCoeffSI", "ChemicalPotentialSI",
    # internal/alternate names
    "as_props_si", "as_props_multi", "batch_as_props",

    # LiBr–H2O
    "LiBrPropsSI", "LiBrH2OPropsSI",
    "LiBrPropsMulti", "LiBrBatchProps",
    "LiBrX_TP", "LiBrH_TX", "LiBrRho_TXP", "LiBrT_HX",
    "librh2o_props_si", "librh2o_props_multi", "batch_librh2o_props",

    # NH3–H2O
    "NH3H2O", "NH3H2O_STATE", "NH3H2O_TPX", "NH3H2O_STATE_TPX",
    "NH3H2OPropsSI", "NH3H2OPropsMulti", "NH3H2OBatchProps",
    "prop_tpx", "props_multi_tpx", "state_tpx", "batch_prop_tpx",
}

# Lowercased views for case-insensitive filtering (identifier extraction only).
_BUILTIN_CONSTS_LC = {s.lower() for s in _BUILTIN_CONSTS}
_RESERVED_WORDS_LC = {s.lower() for s in _RESERVED_WORDS}
_KNOWN_FUNC_NAMES_LC = {s.lower() for s in _KNOWN_FUNC_NAMES}
_THERMO_CALL_NAMES_LC = {s.lower() for s in _THERMO_CALL_NAMES}

# "State-like" variable names such as h1, P2, T3, x6 (very common in textbook problems)
# These must NOT be interpreted as symbol-strings.
_STATEVAR_RE = re.compile(r"^[A-Za-z]{1,3}\d+$")

# If a user writes an *unquoted* identifier assignment in the equation stream, we only treat it
# as a symbol constant for these LHS names (common, safe metadata-like knobs).
# Everything else requires quotes to be treated as a symbol constant inside equations: foo = "Bar"
_SYMBOL_LHS_WHITELIST: Set[str] = {
    "fluid",
    "working_fluid", "workingfluid",
    "refrigerant", "ref",
    "coolant",
    "substance", "medium",
    "gas", "liquid",
    "backend", "provider", "engine", "model",
}

# ------------------------------ directive detection ------------------------------

_DIRECTIVE_KV_RE = re.compile(r"^\s*([A-Za-z_][\w ]*)\s*:\s*(.+?)\s*$")
_REPORT_BARE_RE = re.compile(r"^\s*report\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SOLVE_BARE_RE = re.compile(r"^\s*(solve|solver)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Also allow bare (no colon) objective tokens (common when users type like EES)
_OBJECTIVE_BARE_RE = re.compile(r"^\s*(objective|minimize|maximize)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_OBJECTIVE_NOCOLON_RE = re.compile(r"^\s*(minimize|maximize)\b\s+(.+?)\s*$", re.IGNORECASE)

_CONSTRAINTS_BARE_RE = re.compile(r"^\s*(constraints|constraint)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_DESIGNVARS_BARE_RE = re.compile(r"^\s*(design_vars|designvars|design_variables|designvariables)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_BOUNDS_BARE_RE = re.compile(r"^\s*(bounds|bound)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _strip_inline_comment(line: str) -> str:
    """
    Strip inline comments while respecting quoted strings.

    Supports:
      - '#' comments
      - '//' comments
    """
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
        out.append(ch)
        if esc:
            esc = False
            i += 1
            continue
        if ch == "\\":
            esc = True
            i += 1
            continue
        if ch == q:
            q = None
            i += 1
            continue

        i += 1

    return "".join(out)


def _looks_like_directive_line(line: str) -> bool:
    """
    Defensive: treat 'report: ...' / 'solve: ...' / '<section>: ...' as directives,
    not equations, even if they slipped through parse.py.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return False

    if _REPORT_BARE_RE.match(s) or _SOLVE_BARE_RE.match(s):
        return True
    if _OBJECTIVE_BARE_RE.match(s) or _CONSTRAINTS_BARE_RE.match(s) or _DESIGNVARS_BARE_RE.match(s) or _BOUNDS_BARE_RE.match(s):
        return True
    if _OBJECTIVE_NOCOLON_RE.match(s):
        return True

    m = _DIRECTIVE_KV_RE.match(s)
    if not m:
        # Section headers like "bounds:" (empty) might have been dropped by parse.py, but if they
        # survive, treat them as directives.
        if re.match(r"^\s*(objective|minimize|maximize|constraints|constraint|design_vars|designvars|bounds|bound)\s*:\s*$", s, re.IGNORECASE):
            return True
        return False

    key = re.sub(r"\s+", "", m.group(1).strip().lower())
    return key in {
        "title",
        "given", "givens", "constants", "const", "params", "parameters",
        "guess", "guesses", "init", "inits", "variables", "vars",
        "equations", "eqs",
        "report", "output",
        "solve", "solver",
        "objective", "minimize", "maximize",
        "constraints", "constraint",
        "design_vars", "designvars", "design_variables", "designvariables",
        "bounds", "bound",
    }


# ------------------------------ optimizer directive parsing ------------------------------

def _normalize_expr(expr: str) -> str:
    """Preprocess expression to be solver-friendly ('^'->'**', etc.)."""
    s = _strip_inline_comment(expr).strip()
    if not s:
        return ""
    if preprocess_expr is not None:
        try:
            return preprocess_expr(s).strip()  # type: ignore[misc]
        except Exception:
            pass
    return s.replace("^", "**").strip()


def _parse_objective_directive_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse:
      - 'minimize: <expr>'
      - 'maximize: <expr>'
      - 'objective: <expr>' (defaults to minimize unless expr begins with 'max ' / 'maximize ')
      - 'objective:' with empty RHS (allowed; objective may appear on subsequent line in a section)
      - bare: 'maximize x*y' / 'minimize x*y'

    Returns (sense, expr) where sense is 'min' or 'max', or None if not an objective line.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return None

    m0 = _OBJECTIVE_NOCOLON_RE.match(s)
    if m0:
        key = (m0.group(1) or "").strip().lower()
        rhs = (m0.group(2) or "").strip()
        return ("max" if key == "maximize" else "min"), _normalize_expr(rhs)

    m = _OBJECTIVE_BARE_RE.match(s)
    if not m:
        return None

    key = (m.group(1) or "").strip().lower()
    rhs = (m.group(2) or "").strip()

    if key == "maximize":
        return "max", _normalize_expr(rhs)
    if key == "minimize":
        return "min", _normalize_expr(rhs)

    # key == 'objective'
    rhs_norm = _normalize_expr(rhs)
    rhs_lc = rhs_norm.lstrip().lower()

    # Allow 'objective: max f(x)' shorthand.
    for pfx, sense in [("maximize ", "max"), ("max ", "max"), ("minimize ", "min"), ("min ", "min")]:
        if rhs_lc.startswith(pfx):
            return sense, rhs_norm.lstrip()[len(pfx):].strip()

    return "min", rhs_norm


def _parse_constraints_directive_line(line: str) -> Optional[str]:
    """Parse inline 'constraints: <eq>' / 'constraint: <eq>' directive."""
    s = _strip_inline_comment(line).strip()
    if not s:
        return None
    m = _CONSTRAINTS_BARE_RE.match(s)
    if not m:
        return None
    rhs = (m.group(2) or "").strip()
    if not rhs:
        return None
    return normalize_equation(rhs)


def _parse_design_vars_directive_line(line: str) -> Optional[List[str]]:
    """Parse inline 'design_vars: x, y, z' directive."""
    s = _strip_inline_comment(line).strip()
    if not s:
        return None
    m = _DESIGNVARS_BARE_RE.match(s)
    if not m:
        return None
    rhs = (m.group(2) or "").strip()
    if not rhs:
        return []
    return _parse_design_vars_blob(rhs)


def _parse_bounds_directive_line(line: str) -> Optional[str]:
    """Parse inline 'bounds: ...' directive blob (returns RHS)."""
    s = _strip_inline_comment(line).strip()
    if not s:
        return None
    m = _BOUNDS_BARE_RE.match(s)
    if not m:
        return None
    rhs = (m.group(2) or "").strip()
    return rhs


def _parse_design_vars_blob(blob: str) -> List[str]:
    parts = [p for p in re.split(r"[,\s]+", blob.strip()) if p]
    out: List[str] = []
    for p in parts:
        if re.fullmatch(r"[A-Za-z_]\w*", p):
            out.append(p)
    return out


def _parse_float_maybe_units(s: str, *, enable_units: bool) -> Optional[float]:
    ss = s.strip()
    if not ss:
        return None
    if ss.lower() in {"-inf", "-infty", "-infinity"}:
        return float("-inf")
    if ss.lower() in {"inf", "+inf", "infty", "infinity", "+infinity"}:
        return float("inf")

    if try_parse_float_or_quantity is not None:  # type: ignore[name-defined]
        v = try_parse_float_or_quantity(ss, enable_units=enable_units)  # type: ignore[misc]
        if v is not None:
            return float(v)

    try:
        return float(ss)
    except Exception:
        return None


def _split_top_level(blob: str, seps: str = ";,") -> List[str]:
    """
    Split a blob on separators that are at top-level (not inside quotes, [..], or (..)).
    Used for parsing one-line bounds directives like:
        x:[0,1]; y:[0,1]
        x:[0,1], y:[0,1]
    """
    s = str(blob or "")
    if not s.strip():
        return []

    out: List[str] = []
    buf: List[str] = []

    q: str | None = None
    esc = False
    depth_sq = 0
    depth_par = 0

    def flush() -> None:
        part = "".join(buf).strip()
        buf.clear()
        if part:
            out.append(part)

    for ch in s:
        if q is None:
            if ch in ("'", '"'):
                q = ch
                buf.append(ch)
                continue
            if ch == "[":
                depth_sq += 1
                buf.append(ch)
                continue
            if ch == "]":
                depth_sq = max(0, depth_sq - 1)
                buf.append(ch)
                continue
            if ch == "(":
                depth_par += 1
                buf.append(ch)
                continue
            if ch == ")":
                depth_par = max(0, depth_par - 1)
                buf.append(ch)
                continue

            if depth_sq == 0 and depth_par == 0 and ch in seps:
                flush()
                continue

            buf.append(ch)
            continue

        # inside quotes
        buf.append(ch)
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == q:
            q = None
            continue

    flush()
    return out


def _split_on_name_colon_at_top_level(blob: str) -> List[str]:
    """
    Fallback splitter for blobs like:
        x:[0,1] y:[0,1]
    Splits on whitespace before a NAME: token at top-level.
    """
    s = str(blob or "").strip()
    if not s:
        return []

    parts = _split_top_level(s, seps=";,")
    if len(parts) > 1:
        return parts

    # If we still have a single item, try splitting on whitespace lookahead for NAME:
    # but only when not inside brackets/parentheses.
    tokens: List[str] = []
    buf: List[str] = []

    q: str | None = None
    esc = False
    depth_sq = 0
    depth_par = 0

    def flush() -> None:
        part = "".join(buf).strip()
        buf.clear()
        if part:
            tokens.append(part)

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]

        if q is None:
            if ch in ("'", '"'):
                q = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "[":
                depth_sq += 1
                buf.append(ch)
                i += 1
                continue
            if ch == "]":
                depth_sq = max(0, depth_sq - 1)
                buf.append(ch)
                i += 1
                continue
            if ch == "(":
                depth_par += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ")":
                depth_par = max(0, depth_par - 1)
                buf.append(ch)
                i += 1
                continue

            # check for split point: whitespace at top-level followed by NAME:
            if depth_sq == 0 and depth_par == 0 and ch.isspace():
                # peek forward over whitespace
                j = i
                while j < n and s[j].isspace():
                    j += 1
                if j < n:
                    m = re.match(r"([A-Za-z_]\w*)\s*:", s[j:])
                    if m:
                        flush()
                        i = j
                        continue

            buf.append(ch)
            i += 1
            continue

        # inside quotes
        buf.append(ch)
        if esc:
            esc = False
            i += 1
            continue
        if ch == "\\":
            esc = True
            i += 1
            continue
        if ch == q:
            q = None
            i += 1
            continue
        i += 1

    flush()
    return tokens if tokens else parts


def _parse_bounds_line(line: str, *, enable_units: bool) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    """
    Parse bounds line forms (intentionally permissive inside a bounds section):
      - 'x: [0, 1]' or 'x: (0, 1)'
      - 'x [0, 1]'  or 'x (0, 1)'
      - 'x 0 1'
      - 'x >= 0'
      - 'x <= 1'
      - '0 <= x <= 1'
    Returns (name, lo, hi).
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return None

    # bullets like "- x: [0,1]" or "* x: [0,1]"
    s = re.sub(r"^\s*[-*]\s*", "", s).strip()
    if not s:
        return None

    # 0 <= x <= 1
    m = re.match(r"^\s*(.+?)\s*<=\s*([A-Za-z_]\w*)\s*<=\s*(.+?)\s*$", s)
    if m:
        lo = _parse_float_maybe_units(m.group(1), enable_units=enable_units)
        name = m.group(2)
        hi = _parse_float_maybe_units(m.group(3), enable_units=enable_units)
        return name, lo, hi

    # x >= 0 / x <= 1
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*(<=|>=)\s*(.+?)\s*$", s)
    if m:
        name = m.group(1)
        op = m.group(2)
        val = _parse_float_maybe_units(m.group(3), enable_units=enable_units)
        if val is None:
            return None
        return (name, None, val) if op == "<=" else (name, val, None)

    # x: [...]
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*:\s*(.+?)\s*$", s)
    if m:
        name = m.group(1)
        rhs = m.group(2).strip()
    else:
        # x [...]
        m2 = re.match(r"^\s*([A-Za-z_]\w*)\s+(.+?)\s*$", s)
        if not m2:
            return None
        name = m2.group(1)
        rhs = m2.group(2).strip()

    # bracketed tuple/list
    m = re.match(r"^[\[(]\s*(.+?)\s*,\s*(.+?)\s*[\])]\s*$", rhs)
    if m:
        lo = _parse_float_maybe_units(m.group(1), enable_units=enable_units)
        hi = _parse_float_maybe_units(m.group(2), enable_units=enable_units)
        return name, lo, hi

    # '0, 1'
    m = re.match(r"^\s*(.+?)\s*,\s*(.+?)\s*$", rhs)
    if m:
        lo = _parse_float_maybe_units(m.group(1), enable_units=enable_units)
        hi = _parse_float_maybe_units(m.group(2), enable_units=enable_units)
        return name, lo, hi

    # '0 1'
    parts = rhs.split()
    if len(parts) == 2:
        lo = _parse_float_maybe_units(parts[0], enable_units=enable_units)
        hi = _parse_float_maybe_units(parts[1], enable_units=enable_units)
        return name, lo, hi

    return None


def _parse_bounds_blob(blob: str, *, enable_units: bool) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """
    Parse a one-line bounds blob that might contain multiple variable specs.

    Examples:
      "x: [0, 1]; y: [0, 1]"
      "x: [0, 1], y: [0, 1]"
      "x: [0, 1] y: [0, 1]"
      "0 <= x <= 1; 0 <= y <= 1"
    """
    s = _strip_inline_comment(blob).strip()
    if not s:
        return []

    parts = _split_on_name_colon_at_top_level(s)
    out: List[Tuple[str, Optional[float], Optional[float]]] = []
    for part in parts:
        b = _parse_bounds_line(part, enable_units=enable_units)
        if b is not None:
            out.append(b)
            continue
    return out


def _merge_bounds(bounds_map: Dict[str, Tuple[Optional[float], Optional[float]]], name: str, lo: Optional[float], hi: Optional[float]) -> None:
    cur = bounds_map.get(name)
    if cur is None:
        bounds_map[name] = (lo, hi)
        return
    lo0, hi0 = cur
    lo_new = lo0 if lo is None else lo
    hi_new = hi0 if hi is None else hi
    bounds_map[name] = (lo_new, hi_new)


# ------------------------------ identifier extraction helpers ------------------------------

def _strip_string_literals(expr: str) -> str:
    """
    Remove content inside single/double-quoted strings so identifier extraction
    doesn't turn string tokens into variables.
    """
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


def _extract_names_ordered(expr: str) -> List[str]:
    """Ordered identifier extraction (stable UI)."""
    expr0 = _strip_inline_comment(expr)
    expr2 = _strip_string_literals(expr0)
    toks = re.findall(r"[A-Za-z_]\w*", expr2)
    seen: Set[str] = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _filter_identifiers(
    names: Sequence[str],
    *,
    constants: Set[str],
    reserved_lc: Set[str],
    funcs_lc: Set[str],
) -> List[str]:
    out: List[str] = []
    for n in names:
        s = str(n)
        if not s:
            continue
        if s in constants:
            continue
        sl = s.lower()
        if sl in reserved_lc:
            continue
        if sl in funcs_lc:
            continue
        if s.startswith("__"):
            continue
        out.append(s)
    return out


# ------------------------------ symbol (string-ish) assignments ------------------------------

_RHS_IDENT_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*$")
_RHS_SPEC_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_\.\+\-:]*?(?:::[A-Za-z0-9_\.\+\-:]+)+)\s*$")
_RHS_QUOTED_RE = re.compile(r"""^\s*(['"])(.*?)\1\s*$""")


def _parse_symbol_rhs_info(rhs: str) -> Optional[Tuple[str, bool]]:
    """Return (symbol, was_quoted) if rhs is symbolic-like, else None."""
    m_q = _RHS_QUOTED_RE.match(rhs)
    if m_q:
        return m_q.group(2), True

    m_s = _RHS_SPEC_RE.match(rhs)
    if m_s:
        return m_s.group(1), False

    m_i = _RHS_IDENT_RE.match(rhs)
    if m_i:
        return m_i.group(1), False

    return None


def _quote_symbol(sym: str) -> str:
    s = sym.replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{s}\""


def _substitute_symbol_constants(expr: str, symbols: Mapping[str, str]) -> str:
    """Replace bare identifiers like fluid -> "Helium" outside of quotes."""
    if not symbols:
        return expr

    segs: List[Tuple[str, bool]] = []
    buf: List[str] = []
    q: str | None = None
    esc = False

    def flush(is_string: bool) -> None:
        if buf:
            segs.append(("".join(buf), is_string))
            buf.clear()

    for ch in expr:
        if q is None:
            if ch in ("'", '"'):
                flush(False)
                q = ch
                buf.append(ch)
            else:
                buf.append(ch)
        else:
            buf.append(ch)
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == q:
                q = None
                flush(True)

    if buf:
        flush(q is not None)

    out_parts: List[str] = []
    for seg, is_string in segs:
        if is_string:
            out_parts.append(seg)
            continue
        s2 = seg
        for name, sym in symbols.items():
            s2 = re.sub(rf"\b{re.escape(name)}\b", _quote_symbol(sym), s2)
        out_parts.append(s2)

    return "".join(out_parts)


def _is_statevar_name(name: str) -> bool:
    return bool(_STATEVAR_RE.match(name.strip()))


def _should_symbolize_assignment(
    *,
    lhs: str,
    rhs_sym: str,
    was_quoted: bool,
    guesses: Mapping[str, float],
    funcs_lc: Set[str],
    reserved_lc: Set[str],
    context: str,
) -> bool:
    """Decide whether `lhs = rhs_sym` should be treated as a "symbol constant"."""
    lhs0 = lhs.strip()
    rhs0 = rhs_sym.strip()

    if not lhs0 or not rhs0:
        return False

    lhs_l = lhs0.lower()
    rhs_l = rhs0.lower()

    if lhs_l in funcs_lc or lhs_l in reserved_lc:
        return False
    if rhs_l in funcs_lc or rhs_l in reserved_lc:
        return False

    if was_quoted:
        return True

    if _is_statevar_name(lhs0) or _is_statevar_name(rhs0):
        return False

    if lhs0 in guesses or rhs0 in guesses:
        return False

    if context == "equation":
        return lhs_l in _SYMBOL_LHS_WHITELIST

    if lhs_l in _SYMBOL_LHS_WHITELIST:
        return True
    if "::" in rhs0:
        return True

    return False


# ------------------------------ thermo-call detection ------------------------------

_THERMO_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _rhs_has_thermo_call(rhs: str) -> bool:
    """True if rhs includes a call to a known thermo/property function."""
    rhs2 = _strip_string_literals(_strip_inline_comment(rhs))
    for fn in _THERMO_CALL_RE.findall(rhs2):
        if fn.lower() in _THERMO_CALL_NAMES_LC:
            return True
    return False


# ------------------------------ thermo eval helpers (given folding) ------------------------------

_cached_thermo_funcs: Dict[str, Any] | None = None


def _thermo_eval_funcs() -> Dict[str, Any]:
    """
    Build a mapping of thermo function names to callables for safe_eval injection.
    This is intentionally "best effort": if a backend is not installed, the callable
    won't be injected and evaluation will fall back to "keep as equation".
    """
    global _cached_thermo_funcs
    if _cached_thermo_funcs is not None:
        return dict(_cached_thermo_funcs)

    funcs: Dict[str, Any] = {}

    # CoolProp backend wrappers
    try:
        from thermo_props.coolprop_backend import (  # type: ignore
            PropsSI,
            PhaseSI,
            HAPropsSI,
            ASPropsSI,
            FugacitySI,
            FugacityCoeffSI,
            LnFugacityCoeffSI,
            ChemicalPotentialSI,
        )
        funcs.update(
            {
                "PropsSI": PropsSI,
                "PhaseSI": PhaseSI,
                "HAPropsSI": HAPropsSI,
                "ASPropsSI": ASPropsSI,
                "FugacitySI": FugacitySI,
                "FugacityCoeffSI": FugacityCoeffSI,
                "LnFugacityCoeffSI": LnFugacityCoeffSI,
                "ChemicalPotentialSI": ChemicalPotentialSI,
            }
        )
    except Exception:
        pass

    # Cantera backend wrappers (CTPropsSI family)
    try:
        from thermo_props.cantera_backend import (  # type: ignore
            CTPropsSI,
            ctprops_si,
            ctprops_multi,
            batch_ctprops,
            cantera_available,
        )
        funcs.update(
            {
                "CTPropsSI": CTPropsSI,
                "ctprops_si": ctprops_si,
                "ctprops_multi": ctprops_multi,
                "batch_ctprops": batch_ctprops,
                "cantera_available": cantera_available,
            }
        )
    except Exception:
        pass

    # LiBr–H2O backend (optional)
    try:
        from thermo_props.librh2o_backend import (  # type: ignore
            LiBrPropsSI,
            LiBrH2OPropsSI,
            LiBrX_TP,
            LiBrH_TX,
            LiBrRho_TXP,
            LiBrT_HX,
        )
        funcs.update(
            {
                "LiBrPropsSI": LiBrPropsSI,
                "LiBrH2OPropsSI": LiBrH2OPropsSI,
                "LiBrX_TP": LiBrX_TP,
                "LiBrH_TX": LiBrH_TX,
                "LiBrRho_TXP": LiBrRho_TXP,
                "LiBrT_HX": LiBrT_HX,
            }
        )
    except Exception:
        pass

    # NH3–H2O backend (optional)
    try:
        from thermo_props.nh3h2o_backend import (  # type: ignore
            NH3H2O,
            NH3H2O_STATE,
            NH3H2O_TPX,
            NH3H2O_STATE_TPX,
            NH3H2OPropsSI,
        )
        funcs.update(
            {
                "NH3H2O": NH3H2O,
                "NH3H2O_STATE": NH3H2O_STATE,
                "NH3H2O_TPX": NH3H2O_TPX,
                "NH3H2O_STATE_TPX": NH3H2O_STATE_TPX,
                "NH3H2OPropsSI": NH3H2OPropsSI,
            }
        )
    except Exception:
        pass

    _cached_thermo_funcs = dict(funcs)
    return dict(funcs)


def _can_fold_given_expr(rhs: str, *, resolved: Mapping[str, Any], symbols: Mapping[str, str], reserved_lc: Set[str], funcs_lc: Set[str]) -> bool:
    """
    True if rhs references only already-resolved constant names, symbol constants, or builtins.
    (We still allow function names, which are filtered by funcs_lc.)
    """
    allowed = set(resolved.keys()) | set(symbols.keys()) | set(_BUILTIN_CONSTS)
    names_all = _extract_names_ordered(rhs)
    unknownish = _filter_identifiers(names_all, constants=allowed, reserved_lc=reserved_lc, funcs_lc=funcs_lc)
    return not unknownish


def _try_eval_given_thermo(rhs: str, *, params: Mapping[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Attempt to evaluate rhs (which may include thermo calls) to a float using safe_eval.
    Returns (ok, value, warning_message_if_any).
    """
    if compile_expression is None or eval_expression is None:
        return False, None, "safe_eval unavailable (cannot fold thermo givens)"

    funcs = _thermo_eval_funcs()
    if not funcs:
        return False, None, "no thermo functions available in interpreter scope"

    try:
        c = compile_expression(rhs, extra_funcs=funcs, extra_consts=None)
        v = eval_expression(c, values=None, params=params, extra_funcs=funcs, extra_consts=None)
        if not (isinstance(v, (int, float))):
            return False, None, f"expression did not evaluate to a number: {v!r}"
        fv = float(v)
        if not (fv == fv):  # NaN
            return False, None, "expression evaluated to NaN"
        if fv in (float("inf"), float("-inf")):
            return False, None, "expression evaluated to infinity"
        return True, fv, None
    except ParseError as e:
        return False, None, f"safe_eval error: {e}"
    except Exception as e:
        return False, None, f"eval error: {type(e).__name__}: {e}"


# ------------------------------ solve overrides ------------------------------

_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off"}


def _coerce_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None


def _coerce_jsonish(v: str) -> Any:
    s = str(v).strip()
    if not s:
        return s
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return v
    return v


def _coerce_solve_overrides(raw: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Coerce solve overrides from parse.py / UI into stable types.

    Returns (solve_overrides, warnings)
    """
    out: Dict[str, Any] = {}
    warns: List[str] = []

    def _as_float(x: Any) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    def _as_int(x: Any) -> Optional[int]:
        try:
            return int(float(x))
        except Exception:
            return None

    for k, v in raw.items():
        key_raw = str(k).strip()
        key = key_raw.lower().strip()

        # Preserve non-scalar mappings/lists as-is (e.g., scipy_options might already be a dict)
        if isinstance(v, (dict, list, tuple)):
            out[key_raw] = v
            continue

        val_str = str(v).strip()

        # ---- canonical numeric keys ----
        if key in {"tol", "rtol", "atol"}:
            x = _as_float(val_str)
            if x is not None:
                out["tol"] = float(x)
            else:
                warns.append(f"Solve override ignored (tol not numeric): {key_raw}={val_str!r}")
            continue

        if key in {"max_iter", "maxiter", "maxiterations", "iterations"}:
            x = _as_int(val_str)
            if x is not None:
                out["max_iter"] = int(x)
            else:
                warns.append(f"Solve override ignored (max_iter not int): {key_raw}={val_str!r}")
            continue

        if key in {"max_restarts", "restarts"}:
            x = _as_int(val_str)
            if x is not None:
                out["max_restarts"] = int(x)
            else:
                warns.append(f"Solve override ignored (max_restarts not int): {key_raw}={val_str!r}")
            continue

        # ---- booleans that MUST be real bools ----
        if key in {"warm_start", "warmstart"}:
            b = _coerce_bool(val_str)
            if b is None:
                warns.append(f"Solve override ignored (warm_start not boolean-ish): {key_raw}={val_str!r}")
            else:
                out["warm_start"] = b
            continue

        if key in {"auto_guess", "autoguess"}:
            b = _coerce_bool(val_str)
            if b is None:
                warns.append(f"Solve override ignored (auto_guess not boolean-ish): {key_raw}={val_str!r}")
            else:
                out["auto_guess"] = b
            continue

        if key in {"use_units", "units", "enable_units"}:
            b = _coerce_bool(val_str)
            if b is None:
                warns.append(f"Solve override ignored (use_units not boolean-ish): {key_raw}={val_str!r}")
            else:
                out["use_units"] = b
            continue

        # ---- warm-start tuning ----
        if key in {"warm_start_passes", "warmstart_passes", "warm_start_iters", "warmstart_iters"}:
            x = _as_int(val_str)
            if x is not None:
                out["warm_start_passes"] = int(x)
            else:
                warns.append(f"Solve override ignored (warm_start_passes not int): {key_raw}={val_str!r}")
            continue

        if key in {"warm_start_mode", "warmstart_mode"}:
            # keep string, but normalize
            m = val_str.strip().lower()
            if m in {"override", "conservative"}:
                out["warm_start_mode"] = m
            else:
                out["warm_start_mode"] = val_str
            continue

        # ---- thermo penalty (float) ----
        if key in {"thermo_penalty", "penalty", "thermo_penalty_scale", "penalty_scale"}:
            x = _as_float(val_str)
            if x is not None:
                out["thermo_penalty"] = float(x)
            else:
                warns.append(f"Solve override ignored (thermo_penalty not numeric): {key_raw}={val_str!r}")
            continue

        # ---- backend/method ----
        if key in {"backend", "solver"}:
            out["backend"] = val_str
            continue

        if key in {"method"}:
            # Do NOT over-normalize here; optimizer uses method names like SLSQP.
            m = val_str.strip()
            ml = m.lower()
            if ml in {"newton", "nr", "newtonraphson", "newton-raphson"}:
                out["method"] = "hybr"
                warns.append("Solve method 'newton' mapped to SciPy root method 'hybr' (system solve).")
            else:
                out["method"] = m
            continue

        # ---- scipy options (allow jsonish) ----
        if key in {"scipy_options", "scipy_opts", "scipy_opt"}:
            out["scipy_options"] = _coerce_jsonish(val_str)
            continue

        # Default: keep as scalar string, but try to coerce obvious numeric/bool
        b = _coerce_bool(val_str)
        if b is not None:
            out[key_raw] = b
            continue
        x = _as_int(val_str)
        if x is not None and str(x) == val_str:
            out[key_raw] = x
            continue
        xf = _as_float(val_str)
        if xf is not None:
            out[key_raw] = xf
            continue

        out[key_raw] = val_str

    return out, warns


# ------------------------------ builder ------------------------------

def build_from_parsed(parsed: ParsedInput, *, cfg: InterpretConfig) -> InterpretResult:
    warnings: List[str] = []
    errors: List[str] = []

    reserved_lc: Set[str] = set(_RESERVED_WORDS_LC) | set(_BUILTIN_CONSTS_LC)
    funcs_lc: Set[str] = set(_KNOWN_FUNC_NAMES_LC)

    # Local copies (do NOT mutate ParsedInput in-place)
    report_names: List[str] = list(parsed.report_names or [])
    solve_overrides_raw: Dict[str, Any] = dict(parsed.solve_overrides or {})
    ignored_lines: List[str] = list(parsed.ignored_lines or [])

    # Optimizer intent (optional)
    objective_expr: Optional[str] = None
    objective_sense: str = "min"
    constraints_lines_raw: List[str] = []
    design_vars: List[str] = []
    bounds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # Also accept optional parsed sections if parse.py exposes them.
    parsed_objective_lines: List[str] = list(getattr(parsed, "objective_lines", []) or [])
    parsed_constraints_lines: List[str] = list(getattr(parsed, "constraints_lines", []) or getattr(parsed, "constraint_lines", []) or [])
    parsed_design_vars_lines: List[str] = list(getattr(parsed, "design_vars_lines", []) or getattr(parsed, "designvars_lines", []) or [])
    parsed_bounds_lines: List[str] = list(getattr(parsed, "bounds_lines", []) or [])

    # 1) Collect givens / guesses from explicit sections
    constants_expr: Dict[str, str] = {}
    symbol_constants: Dict[str, str] = {}  # e.g., {"fluid": "Helium"}
    guesses: Dict[str, float] = {}

    # Guess lines
    for ln in parsed.guess_lines:
        g = parse_guess_line(ln, enable_units=cfg.enable_units)
        if g is None:
            warnings.append(f"Unparsed guess line (ignored): {ln!r}")
            continue
        guesses[g[0]] = float(g[1])

    # Given lines
    for ln in parsed.given_lines:
        s0 = _strip_inline_comment(ln).strip()
        if not s0:
            continue

        # Optimizer directives that slipped into 'given:'
        obj = _parse_objective_directive_line(s0)
        if obj is not None:
            sense, expr = obj
            objective_sense = sense
            if expr:
                if objective_expr and expr != objective_expr:
                    warnings.append("Multiple objective expressions found; last one wins.")
                objective_expr = expr
            warnings.append(f"Moved directive line from given to objective: {ln!r}")
            continue

        c_rhs = _parse_constraints_directive_line(s0)
        if c_rhs is not None:
            constraints_lines_raw.append(c_rhs)
            warnings.append(f"Moved directive line from given to constraints: {ln!r}")
            continue

        dvs = _parse_design_vars_directive_line(s0)
        if dvs is not None:
            design_vars.extend(dvs)
            warnings.append(f"Moved directive line from given to design_vars: {ln!r}")
            continue

        b_blob = _parse_bounds_directive_line(s0)
        if b_blob is not None:
            parsed_bounds = _parse_bounds_blob(b_blob, enable_units=cfg.enable_units)
            if not parsed_bounds:
                warnings.append(f"Bounds directive could not be parsed: {ln!r}")
            for nm, lo, hi in parsed_bounds:
                _merge_bounds(bounds_map, nm, lo, hi)
            warnings.append(f"Moved directive line from given to bounds: {ln!r}")
            continue

        a = parse_constant_assignment(s0)
        if a is None:
            warnings.append(f"Unparsed given line (ignored): {ln!r}")
            continue

        name, rhs = a

        # Symbol/string-like given? (fluid = Helium)
        info = _parse_symbol_rhs_info(rhs)
        if info is not None:
            sym, was_quoted = info
            if _should_symbolize_assignment(
                lhs=name,
                rhs_sym=sym,
                was_quoted=was_quoted,
                guesses=guesses,
                funcs_lc=funcs_lc,
                reserved_lc=reserved_lc,
                context="given",
            ):
                symbol_constants[name] = sym
                warnings.append(f"Interpreted given as symbol constant: {name} = {sym!r}")
                continue

        constants_expr[name] = rhs

    # 1b) Optional optimizer sections (if parse.py provided them)
    for ln in parsed_objective_lines:
        s0 = _strip_inline_comment(ln).strip()
        if not s0:
            continue
        obj = _parse_objective_directive_line(s0)
        if obj is not None:
            sense, expr = obj
            objective_sense = sense
            if expr:
                if objective_expr and expr != objective_expr:
                    warnings.append("Multiple objective expressions found; last one wins.")
                objective_expr = expr
            continue
        expr = _normalize_expr(s0)
        if expr:
            if objective_expr and expr != objective_expr:
                warnings.append("Multiple objective expressions found; last one wins.")
            objective_expr = expr

    for ln in parsed_constraints_lines:
        s0 = _strip_inline_comment(ln).strip()
        if not s0:
            continue
        if _looks_like_directive_line(s0):
            # allow inline 'constraints: <eq>' lines inside constraints section
            c_rhs = _parse_constraints_directive_line(s0)
            if c_rhs is not None:
                constraints_lines_raw.append(c_rhs)
            continue
        constraints_lines_raw.append(normalize_equation(s0))

    for ln in parsed_design_vars_lines:
        s0 = _strip_inline_comment(ln).strip()
        if not s0:
            continue
        dvs = _parse_design_vars_directive_line(s0)
        if dvs is not None:
            design_vars.extend(dvs)
            continue
        design_vars.extend(_parse_design_vars_blob(s0))

    for ln in parsed_bounds_lines:
        s0 = _strip_inline_comment(ln).strip()
        if not s0:
            continue
        # If bounds lines came with prefix like "bounds: x: [0,1]" parse RHS blob too.
        b_blob = _parse_bounds_directive_line(s0)
        if b_blob is not None:
            for nm, lo, hi in _parse_bounds_blob(b_blob, enable_units=cfg.enable_units):
                _merge_bounds(bounds_map, nm, lo, hi)
            continue

        b = _parse_bounds_line(s0, enable_units=cfg.enable_units)
        if b is not None:
            _merge_bounds(bounds_map, b[0], b[1], b[2])

    # 2) Scan equation_lines
    eq_lines_raw: List[str] = []

    # Optional "block mode" for objective/constraints/bounds if headers like "bounds:" survived.
    pending_objective_sense: Optional[str] = None
    in_constraints_block = False
    in_bounds_block = False
    in_designvars_block = False

    for ln in parsed.equation_lines:
        s0 = _strip_inline_comment(ln).strip()
        if not s0:
            # blank line ends any block mode
            pending_objective_sense = None
            in_constraints_block = False
            in_bounds_block = False
            in_designvars_block = False
            continue

        # Handle header-only lines like "bounds:" / "constraints:" if they survived parsing.
        if re.match(r"^\s*(constraints|constraint)\s*:\s*$", s0, re.IGNORECASE):
            in_constraints_block = True
            in_bounds_block = False
            in_designvars_block = False
            pending_objective_sense = None
            continue
        if re.match(r"^\s*(bounds|bound)\s*:\s*$", s0, re.IGNORECASE):
            in_bounds_block = True
            in_constraints_block = False
            in_designvars_block = False
            pending_objective_sense = None
            continue
        if re.match(r"^\s*(design_vars|designvars|design_variables|designvariables)\s*:\s*$", s0, re.IGNORECASE):
            in_designvars_block = True
            in_constraints_block = False
            in_bounds_block = False
            pending_objective_sense = None
            continue
        if re.match(r"^\s*(objective|minimize|maximize)\s*:\s*$", s0, re.IGNORECASE):
            # objective block: next non-directive line becomes objective (once)
            m = re.match(r"^\s*(objective|minimize|maximize)\s*:\s*$", s0, re.IGNORECASE)
            key = (m.group(1) if m else "objective").strip().lower()
            pending_objective_sense = "max" if key == "maximize" else "min"
            in_constraints_block = False
            in_bounds_block = False
            in_designvars_block = False
            continue

        # If we are in an objective block and this line is not a directive, capture it as the objective expression.
        if pending_objective_sense is not None and not _looks_like_directive_line(s0):
            expr = _normalize_expr(s0)
            if expr:
                objective_sense = pending_objective_sense
                if objective_expr and expr != objective_expr:
                    warnings.append("Multiple objective expressions found; last one wins.")
                objective_expr = expr
            pending_objective_sense = None
            continue

        # If we are in a constraints block, every non-directive line is a constraint equation.
        if in_constraints_block and not _looks_like_directive_line(s0):
            constraints_lines_raw.append(normalize_equation(s0))
            continue

        # If we are in a design_vars block, accumulate tokens.
        if in_designvars_block and not _looks_like_directive_line(s0):
            design_vars.extend(_parse_design_vars_blob(s0))
            continue

        # If we are in a bounds block, parse each line as bounds.
        if in_bounds_block and not _looks_like_directive_line(s0):
            b = _parse_bounds_line(s0, enable_units=cfg.enable_units)
            if b is not None:
                _merge_bounds(bounds_map, b[0], b[1], b[2])
            else:
                warnings.append(f"Unparsed bounds line (ignored): {ln!r}")
            continue

        # directive lines that slipped into equations
        if _looks_like_directive_line(s0):
            # Optimizer directives
            obj = _parse_objective_directive_line(s0)
            if obj is not None:
                sense, expr = obj
                objective_sense = sense
                if expr:
                    if objective_expr and expr != objective_expr:
                        warnings.append("Multiple objective expressions found; last one wins.")
                    objective_expr = expr
                warnings.append(f"Moved directive line from equations to objective: {ln!r}")
                continue

            c_rhs = _parse_constraints_directive_line(s0)
            if c_rhs is not None:
                constraints_lines_raw.append(c_rhs)
                warnings.append(f"Moved directive line from equations to constraints: {ln!r}")
                continue

            dvs = _parse_design_vars_directive_line(s0)
            if dvs is not None:
                design_vars.extend(dvs)
                warnings.append(f"Moved directive line from equations to design_vars: {ln!r}")
                continue

            b_blob = _parse_bounds_directive_line(s0)
            if b_blob is not None:
                parsed_bounds = _parse_bounds_blob(b_blob, enable_units=cfg.enable_units)
                if not parsed_bounds:
                    warnings.append(f"Bounds directive could not be parsed: {ln!r}")
                for nm, lo, hi in parsed_bounds:
                    _merge_bounds(bounds_map, nm, lo, hi)
                warnings.append(f"Moved directive line from equations to bounds: {ln!r}")
                continue

            m = _DIRECTIVE_KV_RE.match(s0)
            if m:
                k = re.sub(r"\s+", "", m.group(1).strip().lower())
                v = m.group(2).strip()

                if k in {"report", "output"}:
                    parts = re.split(r"[,\s]+", v)
                    report_names.extend([p for p in parts if p])
                    warnings.append(f"Moved directive line from equations to report: {ln!r}")
                    continue

                if k in {"solve", "solver"}:
                    solve_overrides_raw.setdefault("_blob", "")
                    solve_overrides_raw["_blob"] = (str(solve_overrides_raw["_blob"]) + " " + v).strip()
                    warnings.append(f"Moved directive line from equations to solve overrides: {ln!r}")
                    continue

            ignored_lines.append(ln)
            warnings.append(f"Ignored non-equation directive line: {ln!r}")
            continue

        # Guess lines embedded in chaos stream
        gg = parse_guess_line(s0, enable_units=cfg.enable_units)
        if gg is not None:
            s0_l = s0.lower()
            if s0.startswith("?") or " ?=" in s0 or s0_l.startswith("guess ") or s0_l.startswith("init "):
                guesses[gg[0]] = float(gg[1])
                continue

        s_norm = normalize_equation(s0)
        if not s_norm:
            continue

        # Symbol assignment in equation stream? (fluid = Helium)
        a = parse_constant_assignment(s_norm)
        if a and not cfg.keep_assignments_as_equations:
            name, rhs = a
            info = _parse_symbol_rhs_info(rhs)
            if info is not None:
                sym, was_quoted = info
                if _should_symbolize_assignment(
                    lhs=name,
                    rhs_sym=sym,
                    was_quoted=was_quoted,
                    guesses=guesses,
                    funcs_lc=funcs_lc,
                    reserved_lc=reserved_lc,
                    context="equation",
                ):
                    symbol_constants[name] = sym
                    warnings.append(f"Interpreted assignment as symbol constant (not an equation): {name} = {sym!r}")
                    continue
                # else keep as equation

        if looks_like_equation(s_norm):
            eq_lines_raw.append(s_norm)
        else:
            ignored_lines.append(ln)
            warnings.append(f"Ignored non-equation line in equations section: {ln!r}")

    # 3) Normalize equations further and pull assignment-style lines into constants_expr when appropriate
    equations: List[str] = []
    pulled_as_constants: Dict[str, str] = {}

    for s in eq_lines_raw:
        a = parse_constant_assignment(s)

        if a and not cfg.keep_assignments_as_equations:
            name, rhs = a

            # If RHS includes thermo/property calls, keep as equation (no pulling).
            if _rhs_has_thermo_call(rhs):
                equations.append(s)
                continue

            allowed_constants = (
                set(constants_expr.keys())
                | set(pulled_as_constants.keys())
                | set(symbol_constants.keys())
                | set(_BUILTIN_CONSTS)
            )

            rhs_names_all = _extract_names_ordered(rhs)
            rhs_names_unknownish = _filter_identifiers(
                rhs_names_all,
                constants=allowed_constants,
                reserved_lc=reserved_lc,
                funcs_lc=funcs_lc,
            )

            if not rhs_names_unknownish:
                pulled_as_constants[name] = rhs
                continue

        equations.append(s)

    # merge pulled constants (do not overwrite explicit givens)
    for k, rhs in pulled_as_constants.items():
        constants_expr.setdefault(k, rhs)

    # 4) Resolve constants:
    #    - numeric (units + math) via resolve_constants
    #    - thermo-like givens via safe_eval (if resolvable from already-known constants)
    thermo_expr: Dict[str, str] = {}
    numeric_expr: Dict[str, str] = {}

    for k, rhs in constants_expr.items():
        if _rhs_has_thermo_call(rhs):
            thermo_expr[k] = rhs
        else:
            numeric_expr[k] = rhs

    const_val, const_unresolved_num, const_warn = resolve_constants(numeric_expr, enable_units=cfg.enable_units)
    warnings.extend(const_warn)

    # Attempt to fold thermo givens into constants when they are resolvable.
    # Then re-run numeric expression resolution for expressions that may depend on those thermo constants.
    unresolved_thermo = dict(thermo_expr)
    unresolved_num = dict(const_unresolved_num)

    # Params scope for safe_eval (numeric constants + symbol constants + builtins)
    # NOTE: symbol constants are stored as plain strings; that's OK for thermo calls.
    def _params_scope() -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        p.update(const_val)
        p.update(symbol_constants)
        # builtins provided by safe_eval itself, but harmless to include
        p.update({"pi": 3.141592653589793, "e": 2.718281828459045})
        return p

    progress = True
    it = 0
    max_it = 50
    while progress and (unresolved_thermo or unresolved_num) and it < max_it:
        it += 1
        progress = False

        # (A) thermo constants first (they can unlock numeric expressions)
        for k, rhs in list(unresolved_thermo.items()):
            if not _can_fold_given_expr(rhs, resolved=const_val, symbols=symbol_constants, reserved_lc=reserved_lc, funcs_lc=funcs_lc):
                continue

            ok, v, wmsg = _try_eval_given_thermo(rhs, params=_params_scope())
            if ok and v is not None:
                const_val[k] = float(v)
                unresolved_thermo.pop(k, None)
                progress = True
            else:
                # don't spam; only warn the first time we *could* attempt
                if wmsg and it == 1:
                    warnings.append(f"Thermo given {k!r} not folded (will remain equation): {wmsg}")

        # (B) numeric expressions that depend on newly folded thermo constants
        if safe_eval_numeric is not None:
            for k, rhs in list(unresolved_num.items()):
                try:
                    v = safe_eval_numeric(rhs, names=const_val)  # type: ignore[misc]
                except NumericEvalError:
                    continue
                except Exception as e:
                    warnings.append(f"Failed to evaluate constant {k}={rhs!r}: {e}")
                    continue
                const_val[k] = float(v)
                unresolved_num.pop(k, None)
                progress = True

    # Any remaining unresolved constants: keep them as equations "k = expr"
    for k, rhs in unresolved_num.items():
        equations.append(f"{k} = {rhs}")
        warnings.append(f"Kept as equation (could not resolve as numeric constant): {k} = {rhs}")

    for k, rhs in unresolved_thermo.items():
        equations.append(f"{k} = {rhs}")
        warnings.append(f"Kept as equation (thermo/property expression in given): {k} = {rhs}")

    # 4b) Apply symbol substitutions into equations (so we don't create junk unknowns)
    if symbol_constants:
        equations = [_substitute_symbol_constants(e, symbol_constants) for e in equations]
        if constraints_lines_raw:
            constraints_lines_raw = [_substitute_symbol_constants(c, symbol_constants) for c in constraints_lines_raw]
        if objective_expr:
            objective_expr = _substitute_symbol_constants(objective_expr, symbol_constants)

    # 5) Determine unknowns by scanning identifiers in constraints/equations (stable order)
    const_names: Set[str] = set(const_val.keys()) | set(symbol_constants.keys()) | set(_BUILTIN_CONSTS)

    # Optimization intent: objective present → emit optimize spec
    is_optimize = bool(objective_expr and str(objective_expr).strip())

    constraints: List[str] = []
    if is_optimize:
        if constraints_lines_raw:
            constraints = [c for c in constraints_lines_raw if str(c).strip()]
            if equations:
                warnings.append(
                    "Optimization: both 'constraints' and 'equations' were provided; "
                    "treating equations section as additional constraints."
                )
                constraints.extend(equations)
        else:
            constraints = list(equations)

        if not constraints:
            errors.append("Optimization requested (objective provided) but no constraints/equations were found.")
    else:
        if constraints_lines_raw:
            # Don't drop user input if they added a constraints section but forgot the objective.
            warnings.append(
                "Constraints were provided but no objective was found; "
                "treating constraints as additional equations."
            )
            equations.extend([c for c in constraints_lines_raw if str(c).strip()])

    scan_lines: List[str] = constraints if is_optimize else equations

    unknowns_ordered: List[str] = []
    seen_u: Set[str] = set()

    for e in scan_lines:
        names_all = _extract_names_ordered(e)
        names_filtered = _filter_identifiers(
            names_all,
            constants=const_names,
            reserved_lc=reserved_lc,
            funcs_lc=funcs_lc,
        )
        for n in names_filtered:
            if n not in seen_u:
                seen_u.add(n)
                unknowns_ordered.append(n)

    # Include objective identifiers in unknown set (optimizer mode)
    if is_optimize and objective_expr:
        names_all = _extract_names_ordered(str(objective_expr))
        names_filtered = _filter_identifiers(
            names_all,
            constants=const_names,
            reserved_lc=reserved_lc,
            funcs_lc=funcs_lc,
        )
        for n in names_filtered:
            if n not in seen_u:
                seen_u.add(n)
                unknowns_ordered.append(n)

    unknowns = unknowns_ordered

    # 5b) Warnings for guesses that won't be used
    for gname in sorted(guesses.keys()):
        if gname in const_val:
            warnings.append(f"Guess ignored for numeric constant {gname!r}.")
        elif gname in symbol_constants:
            warnings.append(f"Guess ignored for symbol constant {gname!r}.")
        elif gname not in seen_u:
            warnings.append(
                f"Guess provided for unused/unknown variable {gname!r} "
                "(ignored unless it appears in equations)."
            )

    # 5c) Optimizer metadata warnings
    if design_vars:
        for dv in list(dict.fromkeys(design_vars)):
            if dv in const_val:
                warnings.append(f"Design var {dv!r} is a numeric constant; optimizer flags ignored.")
            elif dv in symbol_constants:
                warnings.append(f"Design var {dv!r} is a symbol constant; optimizer flags ignored.")
            elif dv not in seen_u:
                warnings.append(
                    f"Design var {dv!r} does not appear in constraints/equations/objective "
                    "(it will not be part of the optimization variables)."
                )

    if bounds_map:
        for bn in sorted(bounds_map.keys()):
            if bn not in seen_u:
                warnings.append(
                    f"Bounds were provided for {bn!r} but it does not appear in constraints/equations/objective "
                    "(bounds ignored)."
                )

    # 6) Build variable list with guesses
    variables: List[Dict[str, Any]] = []
    if is_optimize and not design_vars:
        # New optimizer default: if the user did not list design_vars, assume all unknowns are design variables.
        design_set: Set[str] = set(unknowns)
    else:
        design_set = set(dict.fromkeys(design_vars)) if design_vars else set()

    for name in unknowns:
        g = guesses.get(name, cfg.default_guess)
        v: Dict[str, Any] = {"name": name, "guess": float(g)}
        if design_set:
            v["is_design"] = (name in design_set)

        if name in bounds_map:
            lo, hi = bounds_map[name]
            # Avoid non-JSON-safe infinities; treat +/-inf as unbounded.
            if lo is not None:
                if isinstance(lo, float) and (lo == float("inf") or lo == float("-inf")):
                    lo = None
                else:
                    v["lower"] = float(lo)
            if hi is not None:
                if isinstance(hi, float) and (hi == float("inf") or hi == float("-inf")):
                    hi = None
                else:
                    v["upper"] = float(hi)

        variables.append(v)

    # 7) Solve block
    solve: Dict[str, Any] = {
        "backend": cfg.backend,
        "method": cfg.method,
        "tol": float(cfg.tol),
        "max_iter": int(cfg.max_iter),
        "max_restarts": int(cfg.max_restarts),
    }

    solve_overrides, solve_warn = _coerce_solve_overrides(solve_overrides_raw)
    warnings.extend(solve_warn)

    if "_blob" in solve_overrides:
        warnings.append(f"Unparsed solve override blob: {solve_overrides.get('_blob')!r} (ignored)")
        solve_overrides.pop("_blob", None)

    solve.update(solve_overrides)

    # 8) Report handling
    report: List[str] = []
    if report_names:
        report = [r for r in report_names if str(r).strip()]
        known_for_report = set(const_val.keys()) | set(symbol_constants.keys()) | set(unknowns) | set(_BUILTIN_CONSTS)
        for r in report:
            if r not in known_for_report:
                warnings.append(
                    f"Report name {r!r} not found in constants/unknowns "
                    "(will print if solver returns it)."
                )
    else:
        if cfg.infer_report == "all":
            report = list(sorted(set(list(const_val.keys()) + list(symbol_constants.keys()) + list(unknowns))))
        elif cfg.infer_report == "unknowns":
            report = list(unknowns)
        else:
            report = []

    title = cfg.title or parsed.title

    # 9) Square-check (EES rule) for pure equation solving
    if not is_optimize:
        if len(equations) != len(unknowns):
            warnings.append(
                "System is not square (EES rule): "
                f"equations={len(equations)} vs unknowns={len(unknowns)}. "
                "Solver may reject this. Consider adding/removing equations, "
                "or converting more assignments into givens."
            )

    # 10) Emit spec (numeric constants only; symbol constants are inlined into equations)
    if is_optimize:
        # Optimization spec: constraints are residual equations, objective is scalar expression.
        spec: Dict[str, Any] = {
            "problem_type": "optimize",
            "title": title or "Untitled optimization problem",
            "constants": {k: float(v) for k, v in const_val.items()},
            "objective": str(objective_expr),
            "sense": str(objective_sense),
            "constraints": list(constraints),
            # Compatibility: also expose constraints under 'equations' for tooling that expects it.
            "equations": list(constraints),
            "variables": variables,
            "solve": solve,
            "report": report,
        }
        if design_vars:
            spec["design_vars"] = list(dict.fromkeys(design_vars))  # preserve order, unique
    else:
        spec = {
            "problem_type": "equations",
            "title": title or "Untitled equation system",
            "constants": {k: float(v) for k, v in const_val.items()},
            "equations": equations,
            "variables": variables,
            "solve": solve,
            "report": report,
        }

    ok = not errors
    return InterpretResult(
        ok=ok,
        spec=spec,
        warnings=warnings,
        errors=errors,
        meta={
            "n_equations": (len(constraints) if is_optimize else len(equations)),
            "n_unknowns": len(unknowns),
            "n_constants": len(const_val),
            "n_symbol_constants": len(symbol_constants),
            "symbol_constants": dict(symbol_constants),
            "constants_unresolved": sorted(list(set(unresolved_num.keys()) | set(unresolved_thermo.keys()))),
            "pulled_assignment_constants": list(pulled_as_constants.keys()),
            "ignored_lines": ignored_lines,
            "guesses_seen": sorted(list(guesses.keys())),
            "bounds_names": sorted(list(bounds_map.keys())),
        },
    )
