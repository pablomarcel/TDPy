from __future__ import annotations

"""Text parser for the TDPy interpreter package.

This module converts human-friendly equation text into a lightweight
``ParsedInput`` object. It does not build the final solver specification. The
builder layer in ``build_spec.py`` performs semantic interpretation after this
parser has classified raw lines.

Parser responsibilities
-----------------------
The parser recognizes:

* title directives
* section headers such as ``given:``, ``guess:``, ``equations:``, and ``report:``
* inline directives such as ``report: x, y`` and ``solve: tol=1e-6``
* optimizer directives such as ``objective:``, ``constraints:``, and
  ``design_vars:``
* chaotic mixed input where directives appear inside an equation block
* numeric constant assignments that can be pulled into ``given_lines``

Design notes
------------
The parser intentionally keeps output close to the original text. It returns
raw equation-like lines, raw given lines, raw guess lines, raw report names, raw
solve overrides, and ignored lines for diagnostics. Downstream modules perform
normalization, unknown inference, unit parsing, and solver-spec construction.
"""

import re
from typing import Any, Dict, List, Optional

from .models import ParsedInput


_SECTION_ALIASES = {
    "constants": "given",
    "const": "given",
    "given": "given",
    "params": "given",
    "parameters": "given",
    "guesses": "guess",
    "guess": "guess",
    "inits": "guess",
    "init": "guess",
    "initial": "guess",
    "variables": "guess",   # many users put guesses under "variables"
    "vars": "guess",
    "equations": "equations",
    "eqs": "equations",
    "report": "report",
    "output": "report",
    "solve": "solve",
    "solver": "solve",
}

# Optimizer directive keys.
# These must not be discarded by the parser because build_spec.py consumes them.
# They are kept as raw equation_lines so downstream code can interpret them.
_OPTIMIZER_DIRECTIVE_KEYS = {
    "objective",
    "minimize",
    "maximize",
    "constraint",
    "constraints",
    "design_vars",
    "designvars",
    "design_variables",
    "designvariables",
    "bound",
    "bounds",
}

# Title directive used anywhere in the text.
_TITLE_RE = re.compile(r"^\s*(title)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Section header with empty value, for example "given:" or "equations:".
_SECTION_HEADER_RE = re.compile(r"^\s*([A-Za-z_][\w ]*)\s*:\s*$")

# Inline directive, for example "report: x, y" or "solve: tol=1e-6".
_DIRECTIVE_RE = re.compile(r"^\s*([A-Za-z_][\w ]*)\s*:\s*(.+?)\s*$")

# Allow "report x, y, z" without a colon.
_REPORT_BARE_RE = re.compile(r"^\s*report\b\s+(.+?)\s*$", re.IGNORECASE)

# Allow "solve tol=..., max_iter=..." without a colon.
_SOLVE_BARE_RE = re.compile(r"^\s*solve\b\s+(.+?)\s*$", re.IGNORECASE)

# Guess markers for chaotic input.
_GUESS_PREFIX_RE = re.compile(r"^\s*(\?|guess\b|init\b|initial\b)\s*(.+?)\s*$", re.IGNORECASE)

# Constant assignment candidate: NAME = RHS.
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")

# Numeric-ish RHS for "given" inference. This is intentionally strict and
# allows values such as 1, 1.2, -3, 2e-6, and (2e-6)*3/4.
_NUMERIC_EXPR_RE = re.compile(r"^[\s0-9eE\.\+\-\*/\(\)]+$")

# Reserved directive words that should never become variable names.
_RESERVED_WORDS = {
    "title",
    "given",
    "constants",
    "const",
    "params",
    "parameters",
    "guess",
    "guesses",
    "init",
    "inits",
    "initial",
    "variables",
    "vars",
    "equations",
    "eqs",
    "report",
    "output",
    "solve",
    "solver",
    # optimizer keys
    "objective",
    "minimize",
    "maximize",
    "constraint",
    "constraints",
    "design_vars",
    "designvars",
    "design_variables",
    "designvariables",
    "bound",
    "bounds",
}


def _strip_inline_comment(line: str) -> str:
    """Remove simple inline comments from a line.

    The parser assumes equation text does not contain quoted strings that use
    comment tokens. It strips text after ``#`` or ``//``.
    """
    for token in ("//", "#"):
        idx = line.find(token)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def _is_blank(line: str) -> bool:
    """Return whether a line is blank or whitespace-only."""
    return not line or not line.strip()


def _normalize_key(s: str) -> str:
    """Normalize a directive or section key for matching."""
    return re.sub(r"\s+", "", s.strip().lower())


def _split_names_csv_ws(s: str) -> List[str]:
    """Split a report-name list separated by commas or whitespace."""
    parts = re.split(r"[,\s]+", s.strip())
    return [p for p in parts if p]


def _parse_solve_kv_blob(blob: str, *, out: Dict[str, Any]) -> None:
    """Parse a loose solve-options blob into an output mapping.

    Accepted items include ``tol=1e-6``, ``max_iter=50``, and
    ``method=hybr``. Values are kept as strings so downstream code can apply
    consistent type coercion.
    """
    items = [t.strip() for t in re.split(r"[;,]+", blob) if t.strip()]
    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
        elif ":" in item:
            k, v = item.split(":", 1)
        else:
            # Ignore junk in solve blobs.
            continue
        out[k.strip()] = v.strip()


def _looks_like_numeric_expr(rhs: str) -> bool:
    """Return whether a right-hand side looks numeric-only."""
    s = rhs.strip()
    if not s:
        return False
    if not any(ch.isdigit() for ch in s):
        return False
    return bool(_NUMERIC_EXPR_RE.fullmatch(s))


def parse_text(text: str) -> ParsedInput:
    """Parse human-friendly equation text into ``ParsedInput``.

    The parser is designed for clean and chaotic human input.

    Key behavior
    ------------
    Directives are recognized anywhere, even inside an equations section. For
    example, ``report: P_2, m_p`` is not treated as an equation.

    Optimizer directives such as ``objective:``, ``minimize:``, ``maximize:``,
    ``design_vars:``, ``bounds:``, and ``constraints:`` are intentionally
    preserved as raw equation lines so ``build_spec.py`` can interpret them.

    In equations mode, lines such as ``P_atm = 100000`` are inferred as given
    constants when the right-hand side is numeric-only.

    Guess lines can be marked with prefixes such as ``?``, ``guess``, or
    ``init``.

    Returns
    -------
    ParsedInput
        Lightweight parser output that keeps raw lines for downstream
        interpretation.
    """
    title: Optional[str] = None
    section: str = "equations"

    equation_lines: List[str] = []
    given_lines: List[str] = []
    guess_lines: List[str] = []
    report_names: List[str] = []
    solve_overrides: Dict[str, Any] = {}
    ignored_lines: List[str] = []

    # Optional optimizer section-header support:
    #
    #   objective:
    #     x*y
    #
    #   bounds:
    #     x: [0,1]
    #
    # The parser re-prefixes those lines as "objective: x*y" so build_spec can
    # consume them without changing the ParsedInput schema.
    opt_section: Optional[str] = None

    for raw in text.splitlines():
        line = _strip_inline_comment(raw)
        if _is_blank(line):
            continue

        # 1) title: ... (always recognized)
        m_title = _TITLE_RE.match(line)
        if m_title:
            title = m_title.group(2).strip()
            continue

        # 2) report ... (inline anywhere)
        m_report_bare = _REPORT_BARE_RE.match(line)
        if m_report_bare:
            opt_section = None
            report_names.extend(_split_names_csv_ws(m_report_bare.group(1)))
            continue

        # 3) solve ... (inline anywhere)
        m_solve_bare = _SOLVE_BARE_RE.match(line)
        if m_solve_bare:
            opt_section = None
            _parse_solve_kv_blob(m_solve_bare.group(1), out=solve_overrides)
            continue

        # 4) section header such as "given:" or "equations:"
        m_sec = _SECTION_HEADER_RE.match(line)
        if m_sec:
            key_norm = _normalize_key(m_sec.group(1))

            # Optimizer section headers are kept without changing the
            # ParsedInput schema.
            if key_norm in _OPTIMIZER_DIRECTIVE_KEYS:
                opt_section = key_norm
                section = "equations"
                continue

            opt_section = None
            section = _SECTION_ALIASES.get(key_norm, section)
            continue

        # 5) directive "key: value" anywhere, including chaotic equations text.

        # 5a) Optimizer block-style lines. If we are currently inside an
        # optimizer pseudo-section such as:
        #
        #   bounds:
        #     x: [0, 1]
        #     y: [0, 1]
        #
        # then "x: [0, 1]" would otherwise match the generic directive regex and
        # be discarded. Preserve it by re-prefixing with the active optimizer key.
        if opt_section is not None:
            equation_lines.append(f"{opt_section}: {line}")
            continue

        m_dir = _DIRECTIVE_RE.match(line)
        if m_dir:
            key_raw = m_dir.group(1)
            val = m_dir.group(2).strip()
            key_norm = _normalize_key(key_raw)
            alias = _SECTION_ALIASES.get(key_norm, None)

            # Optimizer directives must be preserved for build_spec.
            if key_norm in _OPTIMIZER_DIRECTIVE_KEYS:
                opt_section = None
                equation_lines.append(f"{key_raw.strip()}: {val}")
                continue

            if alias == "report":
                opt_section = None
                report_names.extend(_split_names_csv_ws(val))
                continue

            if alias == "solve":
                opt_section = None
                _parse_solve_kv_blob(val, out=solve_overrides)
                continue

            if alias == "given":
                opt_section = None
                if val:
                    given_lines.append(val)
                else:
                    section = "given"
                continue

            if alias == "guess":
                opt_section = None
                if val:
                    guess_lines.append(val)
                else:
                    section = "guess"
                continue

            if alias == "equations":
                opt_section = None
                if val:
                    equation_lines.append(val)
                else:
                    section = "equations"
                continue

            # Unknown directive key: treat it as not an equation.
            opt_section = None
            ignored_lines.append(line)
            continue

        # 6) Per-section handling.

        if section == "given":
            opt_section = None
            given_lines.append(line)
            continue

        if section == "guess":
            opt_section = None
            m_g = _GUESS_PREFIX_RE.match(line)
            if m_g:
                guess_lines.append(m_g.group(2).strip())
            else:
                guess_lines.append(line)
            continue

        if section == "report":
            opt_section = None
            report_names.extend(_split_names_csv_ws(line))
            continue

        if section == "solve":
            opt_section = None
            if "=" in line or ":" in line:
                _parse_solve_kv_blob(line, out=solve_overrides)
            else:
                ignored_lines.append(line)
            continue

        # 7) equations/default section.

        m_g2 = _GUESS_PREFIX_RE.match(line)
        if m_g2:
            opt_section = None
            guess_lines.append(m_g2.group(2).strip())
            continue

        # Infer given constants from NAME = numeric expression.
        m_asn = _ASSIGN_RE.match(line)
        if m_asn:
            lhs = m_asn.group(1).strip()
            rhs = m_asn.group(2).strip()
            if lhs.lower() not in _RESERVED_WORDS and _looks_like_numeric_expr(rhs):
                opt_section = None
                given_lines.append(f"{lhs} = {rhs}")
                continue

        # Optional optimizer section-header support. If the user wrote:
        #
        #   objective:
        #     x*y
        #
        # re-prefix it as "objective: x*y".
        if opt_section in {
            "objective",
            "minimize",
            "maximize",
            "design_vars",
            "designvars",
            "design_variables",
            "designvariables",
            "bounds",
            "bound",
        }:
            equation_lines.append(f"{opt_section}: {line}")
            continue

        equation_lines.append(line)

    return ParsedInput(
        title=title,
        equation_lines=equation_lines,
        given_lines=given_lines,
        guess_lines=guess_lines,
        report_names=report_names,
        solve_overrides=solve_overrides,
        ignored_lines=ignored_lines,
    )
