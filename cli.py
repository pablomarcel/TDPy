#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Command-line interface for TDPy.

The CLI is intentionally file-driven and conservative. It preserves the
existing engineering workflows while adding a Sphinx skeleton helper for
GitHub Pages deployments.

Commands
--------
``list-inputs``
    List files under the package ``in`` directory.

``run``
    Run any problem definition and route by ``problem_type``.

``props``
    Convenience wrapper for thermodynamic-property inputs.

``eqn``
    Convenience wrapper for nonlinear equation-system inputs.

``opt``
    Convenience wrapper for optimization inputs.

``menu``
    Launch the optional interactive text menu.

``sphinx-skel``
    Generate a conservative single-site Sphinx documentation skeleton.

Deployment notes
----------------
The ``sphinx-skel`` command follows the lessons learned from the other
engineering tool projects:

- dynamic reStructuredText heading underlines
- conservative Sphinx-safe generated files
- ``_static/.gitkeep`` and ``_templates/.gitkeep``
- minimal project-standard Sphinx Makefile
- importable-module filtering for autodoc
- deploy-safe mock imports for optional scientific and GUI dependencies

Typical documentation command::

    python -m cli sphinx-skel docs

Then build locally with::

    make -C docs html
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

# ---------- Import shim so `python cli.py ...` works with absolute imports ----------
if __package__ in (None, ""):
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from apis import RunRequest  # type: ignore
    from app import TdpyApp  # type: ignore
else:
    # TDPy is currently a root-layout project. The stable runtime imports these
    # modules by their top-level names, so keep the absolute imports for
    # compatibility with the existing codebase and editable installs.
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
    """Resolve an input path against the current directory and then ``in``."""
    p = Path(user_path)

    # 1) as provided
    if p.is_absolute() and p.exists():
        return p
    if not p.is_absolute() and p.exists():
        return p.resolve()

    # 2) normalize if the caller passed "in/..."
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

    # Last resort: return the likely candidate for a useful error message.
    return cand


def _resolve_out_path(user_out: Optional[str], in_path: Path) -> Path:
    """Resolve an output path, defaulting to ``out/<input-stem>.out.json``."""
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
    """Print a best-effort summary without assuming a specific response schema."""
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
    """Convert a string token into bool, int, float, ``None``, or string."""
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

    try:
        return float(t)
    except Exception:
        return t


def _parse_kv(token: str) -> Tuple[str, Any]:
    """Parse ``key=value`` into a key and a coerced scalar value."""
    if "=" not in token:
        raise ValueError(f"Expected key=value, got {token!r}")
    k, v = token.split("=", 1)
    key = k.strip()
    if not key:
        raise ValueError(f"Empty key in {token!r}")
    return key, _coerce_scalar(v)


# ------------------------------ Sphinx skeleton helpers ------------------------------

_RST_CHARS = ("=", "-", "~", "^")


# Conservative modules for a root-layout TDPy documentation site.
# ``units.core`` is intentionally not listed because the current runtime uses
# top-level ``units.py`` and packaging both ``units.py`` and a ``units`` package
# can create import ambiguity. The public stable module is ``units``.
_MODULES: tuple[str, ...] = (
    # Application layer
    "cli",
    "app",
    "apis",
    "design",
    "core",
    "in_out",
    "units",
    "utils",
    "main",
    # Equations package
    "equations.api",
    "equations.solver",
    "equations.optimizer",
    "equations.safe_eval",
    "equations.spec",
    # Interpreter package
    "interpreter.cli",
    "interpreter.api",
    "interpreter.build_spec",
    "interpreter.intent",
    "interpreter.models",
    "interpreter.numeric_eval",
    "interpreter.parse",
    # Thermodynamic properties package
    "thermo_props.api",
    "thermo_props.core",
    "thermo_props.state",
    "thermo_props.coolprop_backend",
    "thermo_props.cantera_backend",
    "thermo_props.librh2o_ashrae_backend",
    "thermo_props.nh3h2o_backend",
    "thermo_props.ammonia_water",
    # GUI modules
    "gui_core_dpg",
    "gui_log_dpg",
    "gui_utils_dpg",
)


def _rst_heading(title: str, level: int = 0) -> str:
    """Return a Sphinx-safe reStructuredText heading."""
    ch = _RST_CHARS[min(max(level, 0), len(_RST_CHARS) - 1)]
    text = str(title).strip() or "Untitled"
    return f"{text}\n{ch * len(text)}\n"


def _is_importable(module_name: str) -> bool:
    """Return whether a module can be located without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _write_text(path: Path, text: str, *, force: bool = False) -> bool:
    """Write text if missing or if ``force`` is enabled."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _generate_conf_py() -> str:
    """Generate a conservative root-layout Sphinx ``conf.py``."""
    return '''# Generated by TDPy cli.py
from __future__ import annotations

import sys
from pathlib import Path

# docs -> repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "TDPy"
author = "Pablo Marcel Montijo"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_static_path = ["_static"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "CoolProp",
    "CoolProp.CoolProp",
    "cantera",
    "dearpygui",
    "dearpygui.dearpygui",
    "gekko",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "pandas",
    "plotly",
    "plotly.graph_objects",
    "scipy",
    "scipy.interpolate",
    "scipy.linalg",
    "scipy.optimize",
    "sympy",
    "yaml",
    "pyfiglet",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
'''


def _module_group(module_name: str) -> str:
    """Return a readable documentation group for a module name."""
    if module_name.startswith("equations."):
        return "Equations Package"
    if module_name.startswith("interpreter."):
        return "Interpreter Package"
    if module_name.startswith("thermo_props."):
        return "Thermo Props Package"
    if module_name.startswith("gui_"):
        return "GUI Modules"
    return "Application Layer"


def _generate_api_rst(modules: Sequence[str]) -> str:
    """Generate an API page for the importable modules."""
    parts: list[str] = [_rst_heading("API Reference", 0)]

    grouped: Dict[str, list[str]] = {}
    for mod in modules:
        grouped.setdefault(_module_group(mod), []).append(mod)

    group_order = [
        "Application Layer",
        "Equations Package",
        "Interpreter Package",
        "Thermo Props Package",
        "GUI Modules",
    ]

    for group in group_order:
        mods = grouped.get(group, [])
        if not mods:
            continue

        parts.append(_rst_heading(group, 1))
        for mod in mods:
            parts.append(_rst_heading(mod, 2))
            parts.append(
                f".. automodule:: {mod}\n"
                "   :members:\n"
                "   :undoc-members:\n"
                "   :show-inheritance:\n\n"
            )

    return "\n".join(parts).rstrip() + "\n"


def _generate_index_rst() -> str:
    """Generate the documentation root page."""
    return (
        _rst_heading("TDPy Documentation", 0)
        + "\n"
        + "TDPy is a Python-first engineering toolkit for thermodynamics, "
          "property evaluation, nonlinear equation solving, optimization, "
          "and EES-style engineering workflows.\n\n"
        + ".. toctree::\n"
          "   :maxdepth: 2\n"
          "   :caption: Contents:\n\n"
          "   api\n"
    )


def _generate_makefile() -> str:
    """Generate the minimal project-standard Sphinx Makefile."""
    return (
        "# Minimal Sphinx Makefile\n"
        ".PHONY: html clean\n"
        "html:\n"
        "\t+sphinx-build -b html . _build/html\n"
        "clean:\n"
        "\t+rm -rf _build\n"
    )


def create_sphinx_skeleton(dest: str | Path, *, force: bool = False) -> Path:
    """Create a conservative Sphinx skeleton for the root-layout TDPy project."""
    out_dir = Path(dest).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure repository root is visible for importability checks when running
    # directly from the project root or via ``python -m cli``.
    root = Path(__file__).resolve().parent
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    importable_modules = [m for m in _MODULES if _is_importable(m)]

    # Avoid an empty API page if some optional package discovery behaves oddly.
    if not importable_modules:
        importable_modules = ["cli", "app", "apis", "core", "design", "in_out", "units", "utils"]

    _write_text(out_dir / "conf.py", _generate_conf_py(), force=force)
    _write_text(out_dir / "index.rst", _generate_index_rst(), force=force)
    _write_text(out_dir / "api.rst", _generate_api_rst(importable_modules), force=force)
    _write_text(out_dir / "Makefile", _generate_makefile(), force=force)

    _touch(out_dir / "_static" / ".gitkeep")
    _touch(out_dir / "_templates" / ".gitkeep")

    return out_dir


# ------------------------------ CLI parser ------------------------------

def _add_run_overrides(ap: argparse.ArgumentParser) -> None:
    """Add run-time override flags to a run-like parser."""
    ap.add_argument(
        "--backend",
        default=argparse.SUPPRESS,
        help="Override solve.backend for this run only (for example: auto, scipy, or gekko).",
    )

    method_help = (
        "Override solve.method for this run only. "
        "For equation systems, common SciPy root methods include hybr, lm, "
        "broyden1, broyden2, anderson, krylov, df-sane, linearmixing, "
        "diagbroyden, and excitingmixing. For optimization, common methods "
        "include SLSQP, L-BFGS-B, TNC, trust-constr, COBYLA, Nelder-Mead, "
        "and Powell."
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

    g_units = ap.add_mutually_exclusive_group()
    g_units.add_argument(
        "--use-units",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable unit parsing for this run only.",
    )
    g_units.add_argument(
        "--no-units",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable unit parsing for this run only.",
    )

    g_ws = ap.add_mutually_exclusive_group()
    g_ws.add_argument(
        "--warm-start",
        action="store_true",
        dest="warm_start",
        default=argparse.SUPPRESS,
        help="Enable warm-start prepass to improve initial guesses.",
    )
    g_ws.add_argument(
        "--no-warm-start",
        action="store_false",
        dest="warm_start",
        default=argparse.SUPPRESS,
        help="Disable warm-start prepass.",
    )
    ap.add_argument(
        "--warm-start-passes",
        type=int,
        default=argparse.SUPPRESS,
        help="Override warm-start passes.",
    )
    ap.add_argument(
        "--warm-start-mode",
        choices=["override", "conservative"],
        default=argparse.SUPPRESS,
        help="Override warm-start mode.",
    )

    ap.add_argument(
        "--scipy-opt",
        action="append",
        default=argparse.SUPPRESS,
        metavar="K=V",
        help="Pass a SciPy option as key=value. May be repeated.",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved input/output paths and exit without running.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the TDPy command-line parser."""
    p = argparse.ArgumentParser(
        prog="tdpy",
        description="TDPy — console-first EES-like runner for JSON, YAML, and TXT inputs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # list-inputs
    p_list = sub.add_parser("list-inputs", help="List files under in")
    p_list.add_argument(
        "--verbose",
        action="store_true",
        help="Print absolute paths.",
    )

    # run (generic)
    p_run = sub.add_parser("run", help="Run a problem definition file")
    p_run.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, absolute, or relative to the current directory.",
    )
    p_run.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out, or absolute. Defaults to <stem>.out.json.",
    )
    p_run.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate default Plotly HTML plots when supported.",
    )
    _add_run_overrides(p_run)

    # props
    p_props = sub.add_parser("props", help="Run a thermo_props problem file")
    p_props.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, absolute, or relative to the current directory.",
    )
    p_props.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out, or absolute. Defaults to <stem>.out.json.",
    )
    _add_run_overrides(p_props)

    # eqn
    p_eqn = sub.add_parser("eqn", help="Run an equations problem file")
    p_eqn.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, absolute, or relative to the current directory.",
    )
    p_eqn.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out, or absolute. Defaults to <stem>.out.json.",
    )
    _add_run_overrides(p_eqn)

    # opt
    p_opt = sub.add_parser("opt", help="Run an optimization problem file")
    p_opt.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input path relative to in, absolute, or relative to the current directory.",
    )
    p_opt.add_argument(
        "--out",
        dest="outfile",
        default=None,
        help="Output path relative to out, or absolute. Defaults to <stem>.out.json.",
    )
    _add_run_overrides(p_opt)

    # menu
    sub.add_parser("menu", help="Launch interactive start menu")

    # sphinx-skel
    p_sphinx = sub.add_parser(
        "sphinx-skel",
        help="Create a conservative Sphinx docs skeleton for GitHub Pages.",
    )
    p_sphinx.add_argument(
        "dest",
        nargs="?",
        default="docs",
        help="Destination directory. Default: docs",
    )
    p_sphinx.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated files.",
    )

    return p


def _collect_opts_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build an opts dict only from explicitly provided override flags."""
    d = vars(args)
    opts: Dict[str, Any] = {}

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

    if "use_units" in d and bool(d["use_units"]):
        opts["use_units"] = True
    if "no_units" in d and bool(d["no_units"]):
        opts["use_units"] = False

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
    """Construct ``RunRequest`` while remaining compatible with older versions."""
    try:
        return RunRequest(
            in_path=in_path,
            out_path=out_path,
            make_plots=bool(make_plots),
            opts=opts,  # type: ignore[arg-type]
        )
    except TypeError:
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
    args: Optional[argparse.Namespace] = None,
) -> int:
    """Resolve paths, create a run request, execute it, and print the output path."""
    in_path = _resolve_in_path(infile)
    if not in_path.exists():
        print(f"Input file not found: {infile!r}")
        print(f"Tried: {str(in_path)}")
        return 2

    out_path = _resolve_out_path(outfile, in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    _maybe_print_run_summary(res)

    outp = getattr(res, "out_path", None)
    if outp is not None:
        print(str(outp))
    else:
        print(str(out_path))

    ok = bool(getattr(res, "ok", False))
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)

    # Keep Sphinx skeleton generation lightweight. It should not need the app
    # service or any solver backend to run successfully.
    if args.cmd == "sphinx-skel":
        out_dir = create_sphinx_skeleton(args.dest, force=bool(args.force))
        print(str(out_dir))
        return 0

    app = TdpyApp()

    if args.cmd == "list-inputs":
        payload = app.list_inputs()
        _print_list_inputs(payload, verbose=bool(args.verbose))
        return 0

    if args.cmd == "run":
        return _run_file(
            app,
            infile=args.infile,
            outfile=args.outfile,
            make_plots=bool(args.make_plots),
            args=args,
        )

    if args.cmd == "props":
        return _run_file(
            app,
            infile=args.infile,
            outfile=args.outfile,
            make_plots=False,
            args=args,
        )

    if args.cmd == "eqn":
        return _run_file(
            app,
            infile=args.infile,
            outfile=args.outfile,
            make_plots=False,
            args=args,
        )

    if args.cmd == "opt":
        return _run_file(
            app,
            infile=args.infile,
            outfile=args.outfile,
            make_plots=False,
            args=args,
        )

    if args.cmd == "menu":
        # Avoid importing optional UI/menu dependencies unless needed.
        try:
            if __package__ in (None, ""):
                from main import main as menu_main  # type: ignore
            else:
                from main import main as menu_main  # type: ignore
        except Exception:
            print("TDPy menu is unavailable. Use `tdpy list-inputs` or `tdpy run`.")
            return 2

        menu_main()
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
