# interpreter/api.py
from __future__ import annotations

"""
interpreter.api

Public API for the interpreter package.

Pipeline:
    text/file -> parse_text() -> build_from_parsed() -> InterpretResult

Design goals:
- Keep this layer thin and deterministic.
- Return JSON-ready specs through InterpretResult.spec.
- Preserve warnings/errors/meta from lower layers.
- Support both text and file entry points.
- Keep imports package-local and lightweight.
"""

import json
from pathlib import Path
from typing import Any, Optional

from .build_spec import build_from_parsed
from .models import InterpretConfig, InterpretResult
from .parse import parse_text


def interpret_text(text: str, *, cfg: Optional[InterpretConfig] = None) -> InterpretResult:
    """
    Interpret raw user text into a TDPy-ready JSON spec.

    This is the main API for the interpreter package:
      text -> parse_text() -> build_from_parsed() -> InterpretResult(spec=JSON-ready dict)
    """
    cfg = cfg or InterpretConfig()
    parsed = parse_text(text)
    res = build_from_parsed(parsed, cfg=cfg)

    # Optional strict mode: elevate warnings to errors.
    if getattr(cfg, "strict_warnings", False) and res.warnings:
        errs = list(res.errors) + [f"(strict) {w}" for w in res.warnings]
        return InterpretResult(
            ok=False,
            spec=res.spec,
            warnings=list(res.warnings),
            errors=errs,
            meta=dict(res.meta),
        )

    return res


def interpret_file(
    path: str | Path,
    *,
    cfg: Optional[InterpretConfig] = None,
    encoding: str = "utf-8",
) -> InterpretResult:
    """
    Interpret a .txt file into a TDPy JSON spec.

    - If cfg.title is not provided, defaults to file stem.
    - Preserves cfg values by creating a shallow updated InterpretConfig.
    """
    cfg0 = cfg or InterpretConfig()
    p = Path(path)
    txt = p.read_text(encoding=encoding)

    # If user didn't set title explicitly, use the file stem.
    if cfg0.title is None:
        cfg0 = InterpretConfig(**{**cfg0.__dict__, "title": p.stem})

    return interpret_text(txt, cfg=cfg0)


def write_spec_json(
    res: InterpretResult,
    outfile: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """
    Write only the JSON spec portion of InterpretResult to disk.

    This intentionally writes only res.spec and not warnings/errors/meta.
    """
    p = Path(outfile)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(res.spec, indent=indent, sort_keys=sort_keys) + "\n"
    p.write_text(payload, encoding="utf-8")


def write_full_result_json(
    res: InterpretResult,
    outfile: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """
    Write the full interpreter result to disk (spec + diagnostics).

    Useful for debugging interpreter behavior, regression tests, and UI integration.
    """
    p = Path(outfile)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ok": bool(res.ok),
        "spec": res.spec,
        "warnings": list(res.warnings or []),
        "errors": list(res.errors or []),
        "meta": dict(res.meta or {}),
    }
    p.write_text(json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n", encoding="utf-8")


__all__ = [
    "interpret_text",
    "interpret_file",
    "write_spec_json",
    "write_full_result_json",
]
