# interpreter/cli.py
from __future__ import annotations

"""
interpreter.cli

CLI for turning human-friendly equation text into a TDPy JSON spec.

Examples:
    python -m interpreter.cli --in in/demo/reversible.txt --out in/demo/reversible.json
    python -m interpreter.cli --in - --print
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .api import interpret_file, write_spec_json

# Optional: if api.py exposes interpret_text(), use it for stdin mode.
try:  # pragma: no cover
    from .api import interpret_text  # type: ignore
except Exception:  # pragma: no cover
    interpret_text = None

from .models import InterpretConfig


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _dump_json(obj: object, *, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="interpreter",
        description="Interpret equation text into a TDPy JSON spec.",
    )

    ap.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input .txt file, or '-' for stdin.",
    )
    ap.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output JSON spec path, or '-' for stdout. Default: <infile>.json (or stdout if --in '-').",
    )

    # Behavior / UX
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail (non-zero exit) if interpreter emits any warnings (or errors).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings/meta output (errors still print).",
    )
    ap.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print JSON spec to stdout (same as --out '-').",
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for stdout printing (default: 2).",
    )

    # Config mapping 1:1 to InterpretConfig
    ap.add_argument("--title", default=None, help="Override title.")
    ap.add_argument("--backend", default="auto", help="Solve backend to emit (auto|scipy|gekko).")
    ap.add_argument("--method", default="hybr", help="Solve method to emit (e.g., hybr, lm, ...).")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance to emit in solve block.")
    ap.add_argument("--max-iter", type=int, default=50, help="Max iterations to emit in solve block.")
    ap.add_argument("--max-restarts", type=int, default=2, help="Max restarts to emit in solve block.")
    ap.add_argument(
        "--default-guess",
        type=float,
        default=1.0,
        help="Default guess assigned to unknowns without explicit guesses (default: 1.0).",
    )
    ap.add_argument(
        "--report",
        default="unknowns",
        choices=["unknowns", "all", "none"],
        help="Auto report selection if user doesn't provide report lines.",
    )

    # Assignments behavior
    g_assign = ap.add_mutually_exclusive_group()
    g_assign.add_argument(
        "--keep-assignments",
        action="store_true",
        help="Keep trivial 'x = 3' as equations (do NOT pull into constants).",
    )
    g_assign.add_argument(
        "--pull-assignments",
        action="store_true",
        help="Pull trivial 'x = 3' into constants when possible (default behavior).",
    )

    # Units behavior
    g_units = ap.add_mutually_exclusive_group()
    g_units.add_argument(
        "--units",
        action="store_true",
        help="Enable unit parsing like '300 K' (default).",
    )
    g_units.add_argument(
        "--no-units",
        action="store_true",
        help="Disable unit parsing like '300 K'.",
    )

    args = ap.parse_args(argv)

    # Resolve output behavior
    outfile: Optional[str] = args.outfile
    if args.do_print:
        outfile = "-"

    # If reading from stdin and user didn't specify output, default to stdout
    if args.infile == "-" and outfile is None:
        outfile = "-"

    cfg = InterpretConfig(
        title=args.title,
        backend=str(args.backend),
        method=str(args.method),
        tol=float(args.tol),
        max_iter=int(args.max_iter),
        max_restarts=int(args.max_restarts),
        infer_report=str(args.report),
        keep_assignments_as_equations=bool(args.keep_assignments) and not bool(args.pull_assignments),
        enable_units=(not bool(args.no_units)),  # default True
        default_guess=float(args.default_guess),
        strict_warnings=bool(args.strict),
    )

    # Interpret
    if args.infile == "-":
        text = sys.stdin.read()
        if interpret_text is not None:
            res = interpret_text(text, cfg=cfg)  # type: ignore[misc]
        else:
            # Fallback: keep behavior deterministic if api lacks interpret_text()
            from .parse import parse_text
            from .build_spec import build_from_parsed

            parsed = parse_text(text)
            res = build_from_parsed(parsed, cfg=cfg)
    else:
        res = interpret_file(args.infile, cfg=cfg)

    # Emit warnings/errors/meta to stderr (so stdout can be pure JSON when requested)
    if not args.quiet:
        if res.warnings:
            _eprint("Interpreter warnings:")
            for w in res.warnings:
                _eprint(f"  - {w}")

        meta = getattr(res, "meta", None)
        if isinstance(meta, dict) and meta:
            parts = []
            for k in ("n_equations", "n_unknowns", "n_constants"):
                if k in meta:
                    parts.append(f"{k}={meta[k]}")
            if "constants_unresolved" in meta and meta["constants_unresolved"]:
                parts.append(f"constants_unresolved={meta['constants_unresolved']}")
            if "pulled_assignment_constants" in meta and meta["pulled_assignment_constants"]:
                parts.append(f"pulled={meta['pulled_assignment_constants']}")
            if parts:
                _eprint("Interpreter meta: " + ", ".join(parts))

    if not res.ok and res.errors:
        _eprint("Interpreter errors:")
        for e in res.errors:
            _eprint(f"  - {e}")

    # Non-zero exit on fatal errors, or on warnings when --strict is set.
    if not res.ok:
        return 2

    if args.strict and res.warnings:
        return 3

    # Output
    if outfile == "-":
        print(_dump_json(res.spec, indent=args.indent))
        return 0

    if outfile is None:
        if args.infile == "-":
            print(_dump_json(res.spec, indent=args.indent))
            return 0
        p_in = Path(args.infile)
        outfile = str(p_in.with_suffix(".json"))

    write_spec_json(res, outfile)
    if not args.quiet:
        _eprint(f"Wrote spec JSON: {outfile}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
