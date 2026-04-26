from __future__ import annotations

"""Input and output utilities for TDPy.

This module provides lightweight helpers for reading and writing problem files.

Supported input formats
-----------------------
JSON
    Structured problem definitions.

YAML
    Human-readable configuration files when PyYAML is installed.

TXT
    EES-style key-value text files with simple scalar and list coercion.

Design notes
------------
The module intentionally stays dependency-light. PyYAML is optional, and Plotly
is only required when ``save_plotly_html`` is called.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from utils import ensure_dir, json_default, with_error_context

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ------------------------------ dirs ------------------------------

def package_dir() -> Path:
    """Return the directory containing this module."""
    return Path(__file__).resolve().parent


def in_dir() -> Path:
    """Return the default input directory."""
    return package_dir() / "in"


def out_dir() -> Path:
    """Return the default output directory."""
    return package_dir() / "out"


# ------------------------------ TXT (EES-ish) parsing ------------------------------

def _strip_inline_comments(line: str) -> str:
    """Strip simple inline comments from one text line.

    Comment markers are ``//``, ``#``, and ``!``. The parser is intentionally
    simple and does not attempt quote-aware parsing.
    """
    s = line
    for c in ("//", "#", "!"):
        if c in s:
            s = s.split(c, 1)[0]
    return s.strip()


@with_error_context("load_text_kv")
def load_text_kv(path: str | Path) -> Dict[str, Any]:
    """Parse an EES-style key-value text file.

    Supported line forms include ``key = value`` and ``key: value``. Blank lines
    and comment-only lines are skipped.

    Values are coerced on a best-effort basis into booleans, ``None``, integers,
    floats, JSON objects or arrays, comma-separated lists, or raw strings.

    Examples
    --------
    The following input lines are accepted::

        key = value
        key2: value2
        list = 1, 2, 3
    """
    p = Path(path)
    out: Dict[str, Any] = {}

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("!"):
            continue

        line = _strip_inline_comments(line)
        if not line:
            continue

        if "=" in line:
            k, v = [s.strip() for s in line.split("=", 1)]
        elif ":" in line:
            k, v = [s.strip() for s in line.split(":", 1)]
        else:
            continue

        if not k:
            continue

        out[k] = _coerce(v)

    return out


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _coerce(v: str) -> Any:
    s = v.strip()
    if not s:
        return ""

    # quoted strings stay strings (and prevent list-splitting on commas)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return _strip_quotes(s)

    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    if sl in ("none", "null"):
        return None

    # JSON-ish object/array
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            # fall back to scalar coercion if it's not valid JSON
            pass

    # Comma-separated list (simple)
    if "," in s and not (s.startswith("{") or s.startswith("[") or s.startswith("(")):
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        if parts:
            return [_coerce_scalar(p) for p in parts]

    return _coerce_scalar(s)


def _coerce_scalar(s: str) -> Any:
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    if sl in ("none", "null"):
        return None

    # int (base-10 only; avoid "08" surprises)
    try:
        if s.startswith("0") and s not in ("0", "0.0") and not s.startswith(("0.", "0e", "0E")):
            raise ValueError
        return int(s, 10)
    except Exception:
        pass

    # float
    try:
        return float(s)
    except Exception:
        return s


# ------------------------------ load/save problems ------------------------------

@with_error_context("load_problem")
def load_problem(path: str | Path) -> Dict[str, Any]:
    """Load a problem mapping from JSON, YAML, or TXT.

    The returned object is always a plain dictionary suitable for
    ``build_problem`` or ``build_spec``. JSON and YAML roots must be mappings.
    TXT files are parsed with ``load_text_kv``.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object (mapping/dict).")
        return dict(data)

    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("pyyaml not installed; can't read YAML inputs.")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping (dict).")
        return dict(data)

    if ext == ".txt":
        return load_text_kv(p)

    raise ValueError(f"Unsupported input format: {ext!r} (expected .json/.yaml/.yml/.txt)")


@with_error_context("save_json")
def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Save a mapping as formatted JSON and return the output path."""
    p = Path(path)
    ensure_dir(p)
    txt = json.dumps(dict(data), indent=2, default=json_default)
    if not txt.endswith("\n"):
        txt += "\n"
    p.write_text(txt, encoding="utf-8")
    return p


@with_error_context("save_yaml")
def save_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    """Save a mapping as YAML and return the output path.

    PyYAML must be installed to use this helper.
    """
    if yaml is None:
        raise ImportError("pyyaml not installed; can't write YAML outputs.")
    p = Path(path)
    ensure_dir(p)
    txt = yaml.safe_dump(dict(data), sort_keys=False)
    if not txt.endswith("\n"):
        txt += "\n"
    p.write_text(txt, encoding="utf-8")
    return p


# ------------------------------ CSV helpers ------------------------------

@with_error_context("load_geometry_csv")
def load_geometry_csv(path: str | Path) -> Tuple[List[float], List[float]]:
    """Return ``x_mm`` and ``D_mm`` arrays from a geometry CSV file.

    The CSV file must include headers named ``x_mm`` and ``D_mm``. The function
    name is kept for backward compatibility with the existing nozzle solver.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV has no headers.")

        # normalize headers for robustness
        fieldnames = [h.strip() for h in r.fieldnames]
        if "x_mm" not in fieldnames or "D_mm" not in fieldnames:
            raise ValueError(f"Geometry CSV missing headers 'x_mm' and/or 'D_mm'. Found: {fieldnames}")

        x: List[float] = []
        d: List[float] = []
        for row in r:
            # DictReader keys use original header strings; map through stripped headers
            # safest approach: read by expected keys directly
            x.append(float(row["x_mm"]))
            d.append(float(row["D_mm"]))

    return x, d


@with_error_context("save_table_csv")
def save_table_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Save a sequence of row mappings to a CSV file.

    If ``fieldnames`` is not provided, the column names are inferred from the
    first row. Values are written through ``csv.DictWriter``.
    """
    p = Path(path)
    ensure_dir(p)

    if not rows:
        p.write_text("", encoding="utf-8")
        return p

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    fns = list(fieldnames)

    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fns})

    return p


# ------------------------------ Plotly helper ------------------------------

def save_plotly_html(fig: Any, path: str | Path) -> str:
    """Save a Plotly figure as HTML and return the string path.

    Plotly must be installed and the caller must pass an object that implements
    ``write_html``.
    """
    p = Path(path)
    ensure_dir(p)
    fig.write_html(str(p))
    return str(p)
