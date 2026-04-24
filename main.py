#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
main

Tiny interactive TUI that rides on the same TdpyApp service.

Design goals:
- Keep this as a thin convenience wrapper; the real orchestration lives in TdpyApp.
- Keep imports lightweight (pyfiglet is optional).
- Work with all current problem types automatically because everything goes through:
    TdpyApp.run(RunRequest(...))

Upgrades:
- Adds an import-shim so `python main.py` works (mirrors cli.py pattern).
- Adds a "quick shortcuts" panel and a simple "last used input" memory for the session.
- Improves UX: clear errors, prints output path, lists available inputs on failure.
- Keeps behavior deterministic and dependency-light.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# ---------- Import shim so `python main.py` works with absolute imports ----------
if __package__ in (None, ""):
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from app import TdpyApp  # type: ignore
    from apis import RunRequest  # type: ignore
    from utils import setup_logger  # type: ignore
else:
    from app import TdpyApp
    from apis import RunRequest
    from utils import setup_logger

try:
    from pyfiglet import Figlet  # type: ignore
except Exception:  # pragma: no cover
    Figlet = None


def _banner() -> str:
    title = "TDPy"
    if Figlet is None:
        return f"=== {title} ==="
    f = Figlet(font="slant")
    return f.renderText(title)


def _menu() -> str:
    return (
        "\n"
        "✨ Choose an option:\n"
        "  1) Run a problem (from in or absolute path)\n"
        "  2) List available inputs\n"
        "  3) About\n"
        "  4) Exit\n"
    )


def _shortcuts(last_in: str | None) -> str:
    last = f"last: {last_in}" if last_in else "last: (none)"
    return (
        "\n"
        "⌨️  Shortcuts:\n"
        f"  - Enter 1 then leave blank to reuse ({last})\n"
        "  - Type 'q' at any prompt to go back\n"
        "  - Input paths can be relative to in (e.g. nozzle/foo.json) or absolute\n"
    )


def _about() -> str:
    return (
        "TDPy — a Python-first EES-like runner.\n"
        "Console-first today; later: PySide6 GUI on top of the same app service.\n"
        "Modules:\n"
        "  - thermo_props: CoolProp-backed property calls\n"
        "  - equations: GEKKO/SciPy nonlinear equation solving\n"
        "  - units: lightweight conversions\n"
    )


def _print_listing(app: TdpyApp) -> None:
    listing = app.list_inputs()
    print(f"📁 {listing.get('in_dir', '')}")
    for f in listing.get("files", []):
        print(f"  - {f}")


def _ask(prompt: str) -> str:
    """
    Input helper.
    - returns raw stripped input
    - user can enter 'q' to indicate 'go back' (caller decides what that means)
    """
    return input(prompt).strip()


def main(argv: Optional[list[str]] = None) -> int:
    _ = setup_logger("main")
    app = TdpyApp()

    last_in: str | None = None

    print(_banner())

    while True:
        print(_menu())
        print(_shortcuts(last_in))
        choice = _ask("👉 Enter choice: ").lower()

        if choice in {"4", "exit", "quit"}:
            print("👋 bye")
            return 0

        if choice == "3":
            print(_about())
            continue

        if choice == "2":
            _print_listing(app)
            continue

        if choice == "1":
            print("Tip: paste a path relative to in (e.g. nozzle/foo.json)")
            print("     or an absolute path (e.g. /Users/you/.../foo.json)\n")

            rel = _ask("📄 Input file path (blank=reuse last): ")
            if rel.lower() == "q":
                continue

            if not rel:
                if last_in:
                    rel = last_in
                    print(f"↩️  using last input: {rel}")
                else:
                    print("🤔 no input given (and no last input to reuse)")
                    continue

            make_plots_raw = _ask("📈 Create plotly plots? (y/N): ").lower()
            if make_plots_raw == "q":
                continue
            make_plots = make_plots_raw == "y"

            try:
                res = app.run(RunRequest(in_path=Path(rel), make_plots=make_plots))
                last_in = rel

                print(f"✅ Done! Output: {res.out_path}")
                if res.plots:
                    print("📎 Plots:")
                    for k, v in res.plots.items():
                        print(f"   - {k}: {v}")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("\nAvailable inputs:")
                _print_listing(app)

            continue

        print("🤔 invalid choice")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
