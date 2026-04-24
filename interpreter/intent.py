from __future__ import annotations

"""
interpreter.intent

This module is the "intention layer" for the interpreter package.

Responsibilities (keep it narrow, but robust):
- Normalize equation strings (EES-ish syntax → solver-friendly)
- Parse constants ("given") assignments
- Parse guess lines (multiple human-friendly forms)
- Extract identifiers conservatively for unknown inference (without inventing junk vars)
- Resolve constants numerically via safe numeric evaluation (multi-pass)

Design notes:
- This module should NOT parse full text files (that's parse.py).
- It provides the primitives used by build_spec.py.

Cantera / thermo note (Feb 2026):
- This module only participates in *identifier inference* and *numeric-only*
  constant evaluation. Thermo/property calls (PropsSI/CTPropsSI/etc.) are folded
  (or kept as equations) by build_spec.py.
- However, we still treat common Cantera/CoolProp "fluid spec" tokens as
  *symbolic* so numeric constant resolution doesn't emit noisy warnings when users
  forget quotes (e.g., gri30.yaml|X=CH4:1).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .numeric_eval import NumericEvalError, safe_eval_numeric, try_parse_float_or_quantity

# Reuse solver-safe preprocessing when available (handles '^' → '**', implicit stuff, etc.)
try:
    from equations.safe_eval import preprocess_expr  # type: ignore
except Exception:  # pragma: no cover
    def preprocess_expr(s: str) -> str:
        # minimal fallback: handle ^ as power
        return s.replace("^", "**")


# ------------------------------ regex ------------------------------

# name [=|==|:=] rhs
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(==|=|:=)\s*(.+?)\s*$")

# "? x = 1", "guess: x = 1", "init x = 1", "guess x = 1"
_GUESS_PREFIX_RE = re.compile(r"^\s*(\?|guess|init)\s*[:\s]+(.+)$", re.IGNORECASE)

# "x ?= 1"
_GUESS_INLINE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\?=\s*(.+?)\s*$")

# Directive-ish lines (defensive filters)
_REPORT_LINE_RE = re.compile(r"^\s*report\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SOLVE_LINE_RE = re.compile(r"^\s*(solve|solver)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Optimizer directive-ish lines (defensive filters)
# These are NOT equations; they are interpreted by build_spec.py when optimization mode is enabled.
# Keep the RHS optional so section-header style lines like "objective:" don't become equations.
_OBJECTIVE_LINE_RE = re.compile(r"^\s*(objective|minimize|maximize)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_CONSTRAINTS_LINE_RE = re.compile(r"^\s*(constraints|constraint)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_DESIGNVARS_LINE_RE = re.compile(r"^\s*(design_vars|designvars|design_variables|designvariables)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_BOUNDS_LINE_RE = re.compile(r"^\s*(bounds|bound)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _is_directive_line(line: str) -> bool:
    # True if the line is a known directive (report/solve/objective/constraints/design_vars/bounds).
    #
    # Important design choice:
    # - We still *recognize* optimizer directives here (so they can be ignored in
    #   contexts like parsing givens/guesses), BUT we do NOT necessarily *drop*
    #   them from the equation stream. (See looks_like_equation().)
    s = _strip_inline_comment(line).strip()
    if not s:
        return False
    if _REPORT_LINE_RE.match(s) or _SOLVE_LINE_RE.match(s):
        return True
    if _OBJECTIVE_LINE_RE.match(s) or _CONSTRAINTS_LINE_RE.match(s) or _DESIGNVARS_LINE_RE.match(s) or _BOUNDS_LINE_RE.match(s):
        return True
    return False

# Very lightweight "function call present" detector (for name extraction heuristics)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


# ------------------------------ vocab ------------------------------

_BUILTIN_CONSTS: Set[str] = {"pi", "e"}

# Function names that should NOT be treated as variables.
# Keep aligned with equations.safe_eval allowlist + interpreter.build_spec.
_MATH_FUNC_NAMES: Set[str] = {
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
    "hypot", "radians", "degrees",

    # thermo (numeric only; GEKKO backend doesn't support it)
    "PropsSI", "PhaseSI", "HAPropsSI",

    # Cantera (independent thermo/chem backend; PropsSI-like)
    "CTPropsSI", "CTPropsMulti", "CTBatchProps",
    "ctprops_si", "ctprops_multi", "batch_ctprops",
    "cantera_available",
    # optional cache helpers (safe to ignore for unknown inference)
    "ctprops_cache_info", "clear_ctprops_caches",

    # thermo (CoolProp AbstractState wrapper; for fugacity, etc.)
    "ASPropsSI", "ASPropsMulti", "ASBatchProps",
    "as_props_si", "as_props_multi", "batch_as_props",
    "abstractstate_available",
    "FugacitySI", "FugacityCoeffSI", "LnFugacityCoeffSI", "ChemicalPotentialSI",

    # LiBr–H2O (ASHRAE-style) property engine (PropsSI-like)
    "LiBrPropsSI", "LiBrH2OPropsSI",
    "LiBrPropsMulti", "LiBrBatchProps",
    # optional internal/alternate names (safe to ignore for unknown inference)
    "librh2o_props_si", "librh2o_props_multi", "batch_librh2o_props",

    # NH3–H2O (Ibrahim & Klein 1993) native property engine (PropsSI-like)
    # Primary names exposed to equation text:
    "NH3H2O", "NH3H2O_STATE", "NH3H2O_TPX", "NH3H2O_STATE_TPX",
    # Optional "PropsSI-like" aliases you may expose later:
    "NH3H2OPropsSI", "NH3H2OPropsMulti", "NH3H2OBatchProps",
    # optional internal/alternate names (safe to ignore for unknown inference)
    "nh3h2o_available",
    "state_tpx", "prop_tpx", "props_multi_tpx", "batch_prop_tpx",

    # special
    "erf", "erfc", "gamma", "lgamma",
}

# Words that users might type that should never become unknowns.
# This is a second layer beyond parse.py (defensive).
_RESERVED_WORDS: Set[str] = {
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
    "note", "notes", "units",
}

# Lowercased views for case-insensitive filtering (identifier extraction only).
_BUILTIN_CONSTS_LC = {s.lower() for s in _BUILTIN_CONSTS}
_RESERVED_WORDS_LC = {s.lower() for s in _RESERVED_WORDS}
_MATH_FUNC_NAMES_LC = {s.lower() for s in _MATH_FUNC_NAMES}


# ------------------------------ helpers ------------------------------

def _strip_string_literals(expr: str) -> str:
    """
    Remove content inside single/double-quoted strings so identifier extraction
    doesn't turn string tokens into variables.

    Example: PropsSI("D","T",T,"P",P,fluid)
      - "D", "T", "P" must NOT create unknowns D/T/P
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
            # inside quotes
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
    """
    Strip inline comments while respecting quoted strings.

    Supports:
      - '#' comments
      - '//' comments

    This is defensive: parse.py may already strip comments, but intent is used
    by multiple callers and should tolerate inline comments.
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

            # start of comment?
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


# Accept a broader set of characters commonly used in fluid/mechanism specs.
# This is used ONLY to suppress numeric-eval warnings; it doesn't affect the solver.
_SYMBOL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_:\-./|=,+\[\]\(\)]+$")


def _is_symbolic_constant_rhs(rhs: str) -> bool:
    """
    Heuristic: RHS that is likely meant to be a *string token* (not numeric),
    e.g.:
      fluid = Helium
      backend = coolprop
      fluid = HEOS::Ammonia
      spec_CH4 = gri30.yaml|X=CH4:1

    We treat these as "symbolic constants" so resolve_constants() doesn't emit
    noisy warnings about unknown names.
    """
    r = rhs.strip()
    if not r:
        return False

    # quoted -> definitely symbolic
    if (len(r) >= 2) and ((r[0] == r[-1]) and r[0] in ("'", '"')):
        return True

    # single token with no whitespace
    if any(ch.isspace() for ch in r):
        return False

    # avoid obvious math operators; allow sign in front if it's still a token
    if any(op in r for op in ("+", "*", "/", "^")):
        # Allow plus/comma inside a mechanism token, but only if it still looks like a token
        # (Handled by _SYMBOL_TOKEN_RE). If it contains other math ops, treat as non-symbolic.
        if "*" in r or "/" in r or "^" in r:
            return False

    if not _SYMBOL_TOKEN_RE.fullmatch(r):
        return False

    # must contain at least one letter (avoid "123" being considered symbolic)
    return any(ch.isalpha() for ch in r)


# ------------------------------ data model ------------------------------

@dataclass
class IntentDraft:
    """
    Optional richer structure if you later want to do "one-shot interpret"
    from a text blob, but currently build_spec.py is the main builder.

    Keep fields stable; build_spec can ignore unused items.
    """
    title: Optional[str]
    equations: List[str]
    constants_expr: Dict[str, str]   # name -> expr string (to be evaluated if possible)
    constants_val: Dict[str, float]  # resolved numeric constants
    guesses: Dict[str, float]
    report: List[str]
    solve_overrides: Dict[str, Any]
    warnings: List[str]

    # Extra introspection that other modules may read if present
    reserved_words: Set[str] = field(default_factory=lambda: set(_RESERVED_WORDS) | set(_BUILTIN_CONSTS))
    func_names: Set[str] = field(default_factory=lambda: set(_MATH_FUNC_NAMES))


# ------------------------------ parsing primitives ------------------------------

def parse_guess_line(line: str, *, enable_units: bool = True) -> Optional[Tuple[str, float]]:
    """
    Supports guess forms:
      - "? x = 1.2"
      - "guess: x = 1.2"
      - "init x = 1.2"
      - "x ?= 1.2"
      - "x = 1.2"  (ONLY if RHS is numeric/quantity; intended for guess sections)

    Returns (name, float_value) or None.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return None

    # Ignore directives that might have slipped in
    if _is_directive_line(s):
        return None

    # Inline "?=" form
    m_inline = _GUESS_INLINE_RE.match(s)
    if m_inline:
        name = m_inline.group(1)
        val = m_inline.group(2).strip()
        fv = try_parse_float_or_quantity(val, enable_units=enable_units)
        if fv is None:
            return None
        return name, float(fv)

    # Prefix form "? ..." / "guess: ..." / "init ..."
    m_pref = _GUESS_PREFIX_RE.match(s)
    if m_pref:
        rest = m_pref.group(2).strip()
        m = _ASSIGN_RE.match(rest)
        if not m:
            # allow "? x 1.23" (super loose) -> split on whitespace
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

    # Fallback: treat "x = 1.2" as a guess ONLY if RHS is numeric/quantity
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
    """
    Parse "g = 9.81" or "A := pi*r^2" (rhs returned preprocessed).
    Returns (name, rhs_expr).

    NOTE:
    - This intentionally does NOT decide whether rhs is numeric vs string.
      build_spec.py can interpret string constants (e.g., fluid = Helium).
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return None

    # Ignore directives
    if _is_directive_line(s):
        return None

    m = _ASSIGN_RE.match(s)
    if not m:
        return None

    name = m.group(1)
    rhs = preprocess_expr(m.group(3).strip())
    return name, rhs


def looks_like_equation(line: str) -> bool:
    """
    Conservative test used by the file parser to decide what belongs in the
    equation stream.

    Rules:
    - empty/comment-only lines -> False
    - report/solve directives -> False (they belong to their own sections)
    - EVERYTHING else -> True

    Important for optimization:
    - Optimizer directives like 'maximize: ...', 'bounds: ...', 'design_vars: ...'
      are allowed to flow through the equation stream so build_spec.py can peel
      them off and construct an optimization spec. If we drop them here, the
      objective never reaches build_spec and the problem incorrectly routes as a
      square equation system.
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return False
    if _REPORT_LINE_RE.match(s) or _SOLVE_LINE_RE.match(s):
        return False
    return True


def normalize_equation(line: str) -> str:
    """
    Normalize common user syntax:
      - '^' -> '**' (via preprocess_expr)
      - '==' -> '=' (EES-ish equality)
      - ':=' -> '=' (EES-ish)
    """
    s = _strip_inline_comment(line).strip()
    if not s:
        return ""

    s = preprocess_expr(s)

    # unify := and ==
    if ":=" in s:
        s = s.replace(":=", "=")
    if "==" in s:
        s = s.replace("==", "=")

    return s.strip()


# ------------------------------ identifier extraction ------------------------------

def extract_names_fallback(expr: str) -> Set[str]:
    """
    Conservative identifier extraction.

    Key behavior:
    - Strips inline comments
    - Strips string literals first, so tokens like "D" in PropsSI("D",...) do NOT
      become unknowns.
    - Finds identifiers via regex
    - Filters out builtin consts, reserved words, and known function names (case-insensitive)
    - Bans dunder names
    """
    expr0 = _strip_inline_comment(expr)
    expr2 = _strip_string_literals(expr0)

    names = set(re.findall(r"[A-Za-z_]\w*", expr2))

    # remove obvious non-vars (case-insensitive)
    names = {n for n in names if n.lower() not in _BUILTIN_CONSTS_LC and n.lower() not in _RESERVED_WORDS_LC}

    # remove function names that appear as calls (stronger signal)
    for fn in _CALL_RE.findall(expr2):
        if fn.lower() in _MATH_FUNC_NAMES_LC:
            names.discard(fn)

    # also remove any leftover known funcs (case-insensitive)
    names = {n for n in names if n.lower() not in _MATH_FUNC_NAMES_LC and not n.startswith("__")}
    return names


# ------------------------------ constant resolution ------------------------------

def resolve_constants(
    constants_expr: Mapping[str, str],
    *,
    enable_units: bool = True,
) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    """
    Multi-pass constant evaluation (numeric-only):
      - if RHS parses as float/quantity -> resolve
      - else if RHS is an expression in terms of already-resolved constants -> resolve via safe_eval_numeric
      - else keep in unresolved mapping

    Returns:
      resolved: name -> float
      unresolved: name -> rhs_expr (string)
      warnings: list[str]
    """
    warnings: List[str] = []
    resolved: Dict[str, float] = {}
    unresolved: Dict[str, str] = {k: preprocess_expr(v.strip()) for k, v in dict(constants_expr).items()}

    # first pass: direct numeric
    for k, rhs in list(unresolved.items()):
        fv = try_parse_float_or_quantity(rhs, enable_units=enable_units)
        if fv is not None:
            resolved[k] = float(fv)
            unresolved.pop(k, None)

    # iterative expression resolution
    progress = True
    it = 0
    while progress and unresolved and it < 50:
        it += 1
        progress = False

        for k, rhs in list(unresolved.items()):
            # if it looks symbolic (fluid/backend tokens), don't try numeric eval here
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

    # Final warnings: unresolved constants that reference unknown identifiers
    for k, rhs in unresolved.items():
        if _is_symbolic_constant_rhs(rhs):
            continue
        refs = extract_names_fallback(rhs)
        refs = {r for r in refs if r not in resolved and r.lower() not in _BUILTIN_CONSTS_LC}
        if refs:
            warnings.append(f"Constant {k!r} could not be resolved; references unknown names: {sorted(refs)}")

    return resolved, unresolved, warnings
