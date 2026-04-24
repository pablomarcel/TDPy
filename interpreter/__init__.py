# interpreter/__init__.py
from __future__ import annotations

"""
interpreter

Turn human-friendly text files into TDPy-ready JSON specs.

Public API:
  - interpret_text(...)
  - interpret_file(...)
  - write_spec_json(...)
  - write_full_result_json(...)

Typical usage:

    from interpreter import interpret_file, InterpretConfig

    cfg = InterpretConfig(title="my_case", tol=1e-6, max_iter=50)
    res = interpret_file("in/case.txt", cfg=cfg)
    if res.ok:
        # res.spec is JSON-ready
        ...

CLI:

    python -m interpreter.cli --in in/demo/reversible.txt --out in/demo/reversible.json
"""

from .api import (
    interpret_file,
    interpret_text,
    write_full_result_json,
    write_spec_json,
)
from .models import InterpretConfig, InterpretResult, ParsedInput

__all__ = [
    "interpret_text",
    "interpret_file",
    "write_spec_json",
    "write_full_result_json",
    "InterpretConfig",
    "InterpretResult",
    "ParsedInput",
]
