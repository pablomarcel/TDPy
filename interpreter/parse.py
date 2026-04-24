from __future__ import annotations

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

# --- Optimizer directive keys ---
# These must NOT be discarded by the parser, because build_spec.py consumes them.
# We keep them as raw "equation_lines" so downstream can interpret them.
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

# Title directive (always handled anywhere)
_TITLE_RE = re.compile(r"^\s*(title)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# "section:" header with empty value, e.g. "given:" "equations:" etc.
_SECTION_HEADER_RE = re.compile(r"^\s*([A-Za-z_][\w ]*)\s*:\s*$")

# "key: value" directive (inline), e.g. "report: x, y" or "solve: tol=1e-6"
_DIRECTIVE_RE = re.compile(r"^\s*([A-Za-z_][\w ]*)\s*:\s*(.+?)\s*$")

# allow "report x, y, z" (no colon)
_REPORT_BARE_RE = re.compile(r"^\s*report\b\s+(.+?)\s*$", re.IGNORECASE)

# allow "solve tol=..., max_iter=..." (no colon)
_SOLVE_BARE_RE = re.compile(r"^\s*solve\b\s+(.+?)\s*$", re.IGNORECASE)

# guess markers for chaotic input
_GUESS_PREFIX_RE = re.compile(r"^\s*(\?|guess\b|init\b|initial\b)\s*(.+?)\s*$", re.IGNORECASE)

# constant assignment candidate: NAME = RHS
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")

# numeric-ish RHS for "given" inference (strict on purpose)
# allows: 1, 1.2, -3, 2e-6, (2e-6)*3/4, etc.
_NUMERIC_EXPR_RE = re.compile(r"^[\s0-9eE\.\+\-\*/\(\)]+$")

# reserved directive words that should never become variable names
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
    # simple but effective: remove #... and //...
    # (we assume no quoted strings in equation input)
    for token in ("//", "#"):
        idx = line.find(token)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def _is_blank(line: str) -> bool:
    return not line or not line.strip()


def _normalize_key(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def _split_names_csv_ws(s: str) -> List[str]:
    # report can be "x, y, z" or "x y z"
    parts = re.split(r"[,\s]+", s.strip())
    return [p for p in parts if p]


def _parse_solve_kv_blob(blob: str, *, out: Dict[str, Any]) -> None:
    """
    Parse a loose "tol=1e-6, max_iter=50, method=hybr" blob.
    Values are kept as strings; downstream can coerce types.
    """
    # allow comma/semicolon separated kv items
    items = [t.strip() for t in re.split(r"[;,]+", blob) if t.strip()]
    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
        elif ":" in item:
            k, v = item.split(":", 1)
        else:
            # ignore junk in solve blobs
            continue
        out[k.strip()] = v.strip()


def _looks_like_numeric_expr(rhs: str) -> bool:
    s = rhs.strip()
    if not s:
        return False
    # must contain at least one digit
    if not any(ch.isdigit() for ch in s):
        return False
    # only allow numeric-expression characters (strict; no names)
    return bool(_NUMERIC_EXPR_RE.fullmatch(s))


def parse_text(text: str) -> ParsedInput:
    """
    Parser designed for BOTH clean and chaotic human input.

    Key behaviors:
    - Directives (title/report/solve/given/guess) are recognized ANYWHERE, even inside equations.
      Example: "report: P_2, m_p" will NOT be treated as an equation.
    - Optimizer directives (objective/minimize/maximize/design_vars/bounds/constraints) are
      intentionally preserved as raw equation lines so build_spec.py can interpret them.
    - In "equations" mode, lines like "P_atm = 100000" are inferred as GIVEN constants
      if the RHS is a numeric expression (no symbol names).
    - Guess lines can be marked with "?" or "guess ..." / "init ..." prefixes.

    Output stays lightweight: we return raw lines for downstream interpretation.
    """
    title: Optional[str] = None
    section: str = "equations"

    equation_lines: List[str] = []
    given_lines: List[str] = []
    guess_lines: List[str] = []
    report_names: List[str] = []
    solve_overrides: Dict[str, Any] = {}
    ignored_lines: List[str] = []

    # Optional optimizer section header support:
    #   objective:
    #     x*y
    #   bounds:
    #     x: [0,1]
    # The parser will re-prefix those lines as "objective: x*y" so build_spec can see them.
    opt_section: Optional[str] = None

    for raw in text.splitlines():
        line = _strip_inline_comment(raw)
        if _is_blank(line):
            continue

        # 1) title: ... (always)
        m_title = _TITLE_RE.match(line)
        if m_title:
            title = m_title.group(2).strip()
            continue

        # 2) report: ... (inline anywhere)
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

        # 4) section header like "given:" / "equations:" etc.
        m_sec = _SECTION_HEADER_RE.match(line)
        if m_sec:
            key_norm = _normalize_key(m_sec.group(1))

            # optimizer section headers (kept without changing ParsedInput schema)
            if key_norm in _OPTIMIZER_DIRECTIVE_KEYS:
                opt_section = key_norm
                section = "equations"
                continue

            opt_section = None
            section = _SECTION_ALIASES.get(key_norm, section)
            continue

        # 5) directive "key: value" anywhere (including chaotic equations section)

        # 5a) optimizer section header support (block-style):
        # If we are currently inside an optimizer pseudo-section like:
        #   bounds:
        #     x: [0, 1]
        #     y: [0, 1]
        # then lines like "x: [0, 1]" would otherwise match the generic directive regex
        # and be discarded. Preserve them by re-prefixing with the active optimizer key.
        if opt_section is not None:
            equation_lines.append(f"{opt_section}: {line}")
            continue
        m_dir = _DIRECTIVE_RE.match(line)
        if m_dir:
            key_raw = m_dir.group(1)
            val = m_dir.group(2).strip()
            key_norm = _normalize_key(key_raw)
            alias = _SECTION_ALIASES.get(key_norm, None)

            # optimizer directives must be preserved for build_spec
            if key_norm in _OPTIMIZER_DIRECTIVE_KEYS:
                opt_section = None
                # preserve the full directive line (not just the RHS)
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
                # allow: "given: P_1 = 450000" or "constants: g=9.81"
                if val:
                    given_lines.append(val)
                else:
                    section = "given"
                continue

            if alias == "guess":
                opt_section = None
                # allow: "guess: P_2 = 225000"
                if val:
                    guess_lines.append(val)
                else:
                    section = "guess"
                continue

            # If it's a known section word but with a value, handle sensibly:
            if alias == "equations":
                opt_section = None
                # allow: "equations: x+y=1" (rare but harmless)
                if val:
                    equation_lines.append(val)
                else:
                    section = "equations"
                continue

            # Unknown directive key: treat as NOT an equation (safer than blowing up later)
            opt_section = None
            ignored_lines.append(line)
            continue

        # 6) per-section handling (with extra "chaos" inference)
        if section == "given":
            opt_section = None
            given_lines.append(line)
            continue

        if section == "guess":
            opt_section = None
            # allow "? P_2 = 225000" even inside guess section
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
            # allow multiple kv items per line
            if "=" in line or ":" in line:
                # If it's a blob, parse it as such
                _parse_solve_kv_blob(line, out=solve_overrides)
            else:
                ignored_lines.append(line)
            continue

        # 7) equations/default section (most important)
        # 7a) guess prefixes even in equations mode
        m_g2 = _GUESS_PREFIX_RE.match(line)
        if m_g2:
            opt_section = None
            guess_lines.append(m_g2.group(2).strip())
            continue

        # 7b) infer "given" constants from NAME = <numeric_expr>
        m_asn = _ASSIGN_RE.match(line)
        if m_asn:
            lhs = m_asn.group(1).strip()
            rhs = m_asn.group(2).strip()
            if lhs.lower() not in _RESERVED_WORDS and _looks_like_numeric_expr(rhs):
                opt_section = None
                given_lines.append(f"{lhs} = {rhs}")
                continue

        # 7c) optional optimizer section header support:
        # If the user did:
        #   objective:
        #     x*y
        # we re-prefix it to become "objective: x*y"
        if opt_section in {"objective", "minimize", "maximize", "design_vars", "designvars",
                           "design_variables", "designvariables", "bounds", "bound"}:
            equation_lines.append(f"{opt_section}: {line}")
            continue

        # Otherwise treat as an equation line
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
