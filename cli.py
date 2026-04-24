#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
cli

CLI-first runner for EES-like workflows.

Commands:
- list-inputs   : list files under in
- run           : run any problem definition (routes by problem_type)
- props         : convenience alias for thermo_props inputs
- eqn           : convenience alias for equations inputs
- opt           : convenience alias for optimization inputs (problem_type=optimize)
- menu          : interactive start menu (if available)

Robustness upgrades:
- Resolves relative --in paths against in automatically (while still allowing absolute paths).
- Resolves relative --out paths against out automatically.
- Optional CLI overrides (backend/method/tol/iters/restarts/units/warm-start/scipy-opts) that DO NOT override
  the spec unless explicitly provided on the CLI.
- Backwards compatible with older RunRequest signatures (opts passed only if supported).

Warm-start support:
- --warm-start / --no-warm-start
- --warm-start-passes N
- --warm-start-mode override|conservative

SciPy options passthrough:
- --scipy-opt key=value  (repeatable)
  which becomes opts["scipy_options"] = {...} for the solver/optimizer.

Notes (Feb 2026):
- Optimization runs use the same CLI surface as equations runs; the input's problem_type controls routing.
  The `opt` command is a convenience wrapper; it does not rewrite the input.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------- Import shim so `python cli.py ...` works with absolute imports ----------
if __package__ in (None, ""):
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from apis import RunRequest  # type: ignore
    from app import TdpyApp  # type: ignore
else:
    from apis import RunRequest
    from app import TdpyApp


# ------------------------------ paths ------------------------------

def _pkg_dir() -> Path:
    return Path(__file__).resolve().parent


def _in_dir() -> Path:
    return _pkg_dir() / "in"


def _out_dir() -> Path:
    return _pkg_dir() / "out"


def _resolve_in_path(user_path: str) -> Path:
    """
    Resolve an input path. Priority:
      1) If it exists as provided (absolute or relative to CWD), use it.
      2) If not, try under in/<user_path>.
      3) If user_path already includes 'in', normalize it.
    """
    p = Path(user_path)

    # 1) as provided
    if p.is_absolute() and p.exists():
        return p
    if not p.is_absolute() and p.exists():
        return p.resolve()

    # 2) normalize if they passed "in/..."
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "" and parts[1] == "in":
        p2 = Path(*parts[2:])
        cand = _in_dir() / p2
        if cand.exists():
            return cand.resolve()

    # 3) under in
    cand = _in_dir() / p
    if cand.exists():
        return cand.resolve()

    # last resort: return the most likely candidate for a good error message
    return cand


def _resolve_out_path(user_out: Optional[str], in_path: Path) -> Path:
    """
    Resolve an output path. If user_out is None:
      - default to out/<stem>.out.json
    If user_out is relative:
      - place under out/<user_out>
    If absolute:
      - use as-is
    """
    if user_out is None or str(user_out).strip() == "":
        return (_out_dir() / f"{in_path.stem}.out.json").resolve()

    p = Path(user_out)
    if p.is_absolute():
        return p
    return (_out_dir() / p).resolve()


# ------------------------------ output helpers ------------------------------

def _print_list_inputs(payload: Dict[str, Any], verbose: bool) -> None:
    in_dir = str(payload.get("in_dir", "") or "")
    files = payload.get("files", []) or []
    print(in_dir)
    for rel in files:
        if verbose:
            print(str(Path(in_dir) / str(rel)))
        else:
            print(str(rel))


def _maybe_print_run_summary(res: Any) -> None:
    """
    Best-effort user-friendly summary without assuming specific response schema.
    """
    msg = getattr(res, "message", None)
    ok = getattr(res, "ok", None)
    solver = getattr(res, "solver", None)
    if msg is not None:
        print(str(msg))
    if solver is not None:
        print(f"solver: {solver}")
    if ok is not None and bool(ok) is False and msg is None:
        print("run failed (ok=false)")


# ------------------------------ parsing helpers ------------------------------

def _coerce_scalar(s: str) -> Any:
    """
    Convert a string token into bool/int/float when it is obviously one,
    else return the original string.
    """
    t = s.strip()
    if not t:
        return t

    lo = t.lower()
    if lo in {"true", "yes", "y", "on"}:
        return True
    if lo in {"false", "no", "n", "off"}:
        return False
    if lo in {"none", "null"}:
        return None

    # int?
    try:
        if lo.startswith(("0x", "-0x")):
            return int(t, 16)
        if lo.startswith(("0b", "-0b")):
            return int(t, 2)
        if lo.startswith(("0o", "-0o")):
            return int(t, 8)
        if "." not in t and "e" not in lo:
            return int(t)
    except Exception:
        pass

    # float?
    try:
        return float(t)
    except Exception:
        return t


def _parse_kv(token: str) -> Tuple[str, Any]:
    """
    Parse "key=value" into (key, coerced_value).
    """
    if "=" not in token:
        raise ValueError(f"Expected key=value, got {token!r}")
    k, v = token.split("=", 1)
    key = k.strip()
    if not key:
        raise ValueError(f"Empty key in {token!r}")
    return key, _coerce_scalar(v)


# ------------------------------ CLI parser ------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tdpy",
        description="tdpy — console-first EES-like runner (JSON/YAML/TXT inputs).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # list-inputs
    p_list = sub.add_parser("list-inputs", help="List files under in")
    p_list.add_argument(
        "--verbose",
        action="store_true",
        help="Print absolute paths (useful for copy/paste).",
    )

    # Common run overrides (ONLY apply if explicitly provided)
    def add_run_overrides(ap: argparse.ArgumentParser) -> None:
        ap.add_argument(
            "--backend",
            default=argparse.SUPPRESS,
            help="Override solve.backend for this run only (e.g., auto|scipy|gekko).",
        )

        method_help = (
            "Override solve.method for this run only.\n"
            "For equation systems (scipy root): hybr, lm, broyden1, broyden2, anderson, krylov, df-sane, "
            "linearmixing, diagbroyden, excitingmixing.\n"
            "For optimization (scipy minimize): SLSQP, L-BFGS-B, TNC, trust-constr, COBYLA, Nelder-Mead, Powell.\n"
            "Aliases like 'hybrid', 'newton', 'broyden' are typically normalized by the solver."
        )
        ap.add_argument(
            "--method",
            default=argparse.SUPPRESS,
            help=method_help,
        )

        ap.add_argument(
            "--tol",
            type=float,
            default=argparse.SUPPRESS,
            help="Override solve.tol for this run only.",
        )
        ap.add_argument(
            "--max-iter",
            type=int,
            default=argparse.SUPPRESS,
            help="Override solve.max_iter for this run only.",
        )
        ap.add_argument(
            "--max-restarts",
            type=int,
            default=argparse.SUPPRESS,
            help="Override solve.max_restarts for this run only.",
        )

        # units (mutually exclusive, but still only set if explicitly provided)
        g_units = ap.add_mutually_exclusive_group()
        g_units.add_argument(
            "--use-units",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable unit parsing (override).",
        )
        g_units.add_argument(
            "--no-units",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Disable unit parsing (override).",
        )

        # warm-start tooling
        g_ws = ap.add_mutually_exclusive_group()
        g_ws.add_argument(
            "--warm-start",
            action="store_true",
            dest="warm_start",
            default=argparse.SUPPRESS,
            help="Enable warm-start prepass to improve initial guesses (override).",
        )
        g_ws.add_argument(
            "--no-warm-start",
            action="store_false",
            dest="warm_start",
            default=argparse.SUPPRESS,
            help="Disable warm-start prepass (override).",
        )
        ap.add_argument(
            "--warm-start-passes",
            type=int,
            default=argparse.SUPPRESS,
            help="Override warm-start passes (number of prepass sweeps over assignments).",
        )
        ap.add_argument(
            "--warm-start-mode",
            choices=["override", "conservative"],
            default=argparse.SUPPRESS,
            help="Override warm-start mode (override|conservative).",
        )

        # SciPy options passthrough
        ap.add_argument(
            "--scipy-opt",
            action="append",
            default=argparse.SUPPRESS,
            metavar="K=V",
            help="Pass through a SciPy option as key=value (repeatable). "
                 "Becomes opts['scipy_options'] = {...}.",
        )

        ap.add_argument(
            "--dry-run",
            action="store_true",
            help="Print resolved input/output paths and exit without running.",
        )

    # run (generic)
    p_run = sub.add_parser("run", help="Run a problem definition file (any problem_type)")
    p_run.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, or absolute, or relative to CWD.",
    )
    p_run.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out (or absolute). If omitted, defaults to <stem>.out.json under out.",
    )
    p_run.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate default Plotly HTML plots alongside the JSON output (if plotly is installed and solver supports it).",
    )
    add_run_overrides(p_run)

    # props (thermo_props convenience)
    p_props = sub.add_parser("props", help="Run a thermo_props problem file (convenience wrapper)")
    p_props.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, or absolute, or relative to CWD.",
    )
    p_props.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out (or absolute). If omitted, defaults to <stem>.out.json under out.",
    )
    add_run_overrides(p_props)

    # eqn (equations convenience)
    p_eqn = sub.add_parser("eqn", help="Run an equations problem file (convenience wrapper)")
    p_eqn.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, or absolute, or relative to CWD.",
    )
    p_eqn.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out (or absolute). If omitted, defaults to <stem>.out.json under out.",
    )
    add_run_overrides(p_eqn)

    # opt (optimization convenience)
    p_opt = sub.add_parser("opt", help="Run an optimization problem file (convenience wrapper)")
    p_opt.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, or absolute, or relative to CWD.",
    )
    p_opt.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out (or absolute). If omitted, defaults to <stem>.out.json under out.",
    )
    add_run_overrides(p_opt)

    # menu (interactive)
    sub.add_parser("menu", help="Launch interactive start menu")

    return p


def _collect_opts_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build an opts dict ONLY from explicitly provided override flags.
    Using argparse.SUPPRESS prevents accidental overriding of spec defaults.
    """
    d = vars(args)
    opts: Dict[str, Any] = {}

    # simple key map from CLI arg -> opts key
    mapping = {
        "backend": "backend",
        "method": "method",
        "tol": "tol",
        "max_iter": "max_iter",
        "max_restarts": "max_restarts",
        "warm_start": "warm_start",
        "warm_start_passes": "warm_start_passes",
        "warm_start_mode": "warm_start_mode",
    }
    for k_cli, k_opt in mapping.items():
        if k_cli in d:
            opts[k_opt] = d[k_cli]

    # units flags are mutually exclusive by construction
    if "use_units" in d and bool(d["use_units"]):
        opts["use_units"] = True
    if "no_units" in d and bool(d["no_units"]):
        opts["use_units"] = False

    # SciPy options dict
    if "scipy_opt" in d:
        raw_list = d.get("scipy_opt") or []
        sci: Dict[str, Any] = {}
        for token in raw_list:
            try:
                k, v = _parse_kv(str(token))
            except Exception as e:
                raise SystemExit(f"Invalid --scipy-opt {token!r}: {e}") from e
            sci[str(k)] = v
        if sci:
            opts["scipy_options"] = sci

    return opts


def _make_run_request(
    *,
    in_path: Path,
    out_path: Path,
    make_plots: bool,
    opts: Dict[str, Any],
) -> Any:
    """
    Backwards compatible RunRequest construction:
    - If RunRequest supports opts=..., pass it.
    - Otherwise ignore opts.
    """
    try:
        return RunRequest(
            in_path=in_path,
            out_path=out_path,
            make_plots=bool(make_plots),
            opts=opts,  # type: ignore[arg-type]
        )
    except TypeError:
        # Older RunRequest signature: no opts
        return RunRequest(
            in_path=in_path,
            out_path=out_path,
            make_plots=bool(make_plots),
        )


def _run_file(
    app: TdpyApp,
    infile: str,
    outfile: Optional[str],
    *,
    make_plots: bool = False,
    args: Optional[argparse.Namespace] = None
) -> int:
    in_path = _resolve_in_path(infile)
    if not in_path.exists():
        print(f"Input file not found: {infile!r}")
        print(f"Tried: {str(in_path)}")
        return 2

    out_path = _resolve_out_path(outfile, in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # opts overrides (only from explicit CLI flags)
    opts: Dict[str, Any] = _collect_opts_from_args(args) if args is not None else {}

    if args is not None and getattr(args, "dry_run", False):
        print(f"in : {str(in_path)}")
        print(f"out: {str(out_path)}")
        if opts:
            print(f"opts: {opts}")
        return 0

    req = _make_run_request(
        in_path=in_path,
        out_path=out_path,
        make_plots=bool(make_plots),
        opts=opts,
    )

    res = app.run(req)

    # Best-effort summary
    _maybe_print_run_summary(res)

    # Always print the output path for scripting convenience
    outp = getattr(res, "out_path", None)
    if outp is not None:
        print(str(outp))
    else:
        # fall back to our resolved out path
        print(str(out_path))

    ok = bool(getattr(res, "ok", False))
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = TdpyApp()

    if args.cmd == "list-inputs":
        payload = app.list_inputs()
        _print_list_inputs(payload, verbose=bool(args.verbose))
        return 0

    if args.cmd == "run":
        return _run_file(app, infile=args.infile, outfile=args.outfile, make_plots=bool(args.make_plots), args=args)

    if args.cmd == "props":
        # Convenience wrapper; routing is still controlled by the input's problem_type.
        return _run_file(app, infile=args.infile, outfile=args.outfile, make_plots=False, args=args)

    if args.cmd == "eqn":
        # Convenience wrapper; routing is still controlled by the input's problem_type.
        return _run_file(app, infile=args.infile, outfile=args.outfile, make_plots=False, args=args)

    if args.cmd == "opt":
        # Convenience wrapper; routing is still controlled by the input's problem_type.
        return _run_file(app, infile=args.infile, outfile=args.outfile, make_plots=False, args=args)

    if args.cmd == "menu":
        # Avoid importing optional UI/menu deps unless needed; keeps CLI lightweight.
        try:
            if __package__ in (None, ""):
                from main import main as menu_main  # type: ignore
            else:
                from main import main as menu_main  # type: ignore
        except Exception:
            print("tdpy menu is unavailable (main import failed). Use `tdpy list-inputs` / `tdpy run`.")
            return 2

        menu_main()
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
