#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Dear PyGui utility helpers for TDPy.

This module provides standalone helpers used by the Dear PyGui frontend.

Feature areas
-------------
Filesystem helpers
    Project-root discovery, input/output directory helpers, relative-path
    formatting, and unique output-path generation.

Subprocess helpers
    Background command execution with stdout/stderr capture and callback-based
    line streaming.

File helpers
    JSON and text loading/saving utilities used by the GUI.

Platform helpers
    OS-specific file and folder opening.

Dear PyGui helpers
    File-dialog payload normalization and input-pattern resolution.

The module avoids relative imports so it can be used from ``python gui_core_dpg.py``
and from module-style invocations during development.
"""

import glob as _glob
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# ------------------------------ paths ------------------------------

def find_repo_root(start: Optional[str | Path] = None) -> Path:
    """Find the TDPy repository root.

    The search walks upward from a few likely starting locations and looks for
    files that strongly indicate the project root: ``__init__.py``, ``cli.py``,
    and ``app.py``. If discovery is ambiguous, the directory containing this
    module is returned as the safest fallback.
    """
    candidates: List[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parent)

    seen: set[Path] = set()
    for base in candidates:
        d = base
        for _ in range(35):
            if d in seen:
                break
            seen.add(d)

            markers = (
                (d / "__init__.py").exists(),
                (d / "cli.py").exists(),
                (d / "app.py").exists(),
            )
            if all(markers):
                return d

            if d.parent == d:
                break
            d = d.parent

    # Best fallback: the directory containing this file.
    return Path(__file__).resolve().parent


def in_dir(repo_root: Path) -> Path:
    """Return the resolved default input directory for a repository root."""
    return (repo_root / "in").resolve()


def out_dir(repo_root: Path) -> Path:
    """Return the resolved default output directory for a repository root."""
    return (repo_root / "out").resolve()


def ensure_dir(p: str | Path) -> Path:
    """Ensure a directory exists and return the normalized path.

    When ``p`` looks like a file path because it has a suffix, the parent
    directory is created. Otherwise, ``p`` itself is created.
    """
    pp = Path(p).expanduser()
    target = pp.parent if pp.suffix else pp
    target.mkdir(parents=True, exist_ok=True)
    return pp


def is_inside(child: Path, parent: Path) -> bool:
    """Return whether ``child`` is located inside ``parent``."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def rel_to_in(path: Path, repo_root: Path) -> str:
    """Return a CLI-friendly path relative to ``in`` when possible."""
    p = Path(path).resolve()
    inroot = in_dir(repo_root)
    if is_inside(p, inroot):
        return str(p.relative_to(inroot)).replace("\\", "/")
    return str(p)


def unique_path(base: Path) -> Path:
    """Generate a non-existing path by appending a numeric suffix.

    The suffix is inserted before the file extension. For example,
    ``result.json`` may become ``result_1.json``.
    """
    base = Path(base)
    if not base.exists():
        return base
    stem = base.stem
    suf = base.suffix
    parent = base.parent
    for k in range(1, 10_000):
        cand = parent / f"{stem}_{k}{suf}"
        if not cand.exists():
            return cand
    return parent / f"{stem}_{os.getpid()}{suf}"


# ------------------------------ DearPyGui file dialog helpers ------------------------------

def extract_dpg_file_dialog_path(app_data: Any) -> str:
    """Normalize Dear PyGui file-dialog payloads across versions.

    Different Dear PyGui versions return selected file paths under slightly
    different keys. This helper checks the known shapes and returns a string
    path when one is available.
    """
    if not isinstance(app_data, dict):
        return ""
    sel = app_data.get("selections")
    if isinstance(sel, dict) and sel:
        try:
            return str(next(iter(sel.values())))
        except Exception:
            pass
    for k in ("file_path_name", "file_path", "path"):
        v = app_data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def resolve_input_pattern(path: str | Path, *, prefer_exts: Tuple[str, ...] = (".txt", ".json")) -> Optional[Path]:
    """Resolve a GUI input path or glob pattern to a concrete file.

    Supported examples include absolute files, wildcard paths, and bare stems.
    For wildcard matches, preferred extensions are selected first.

    Examples
    --------
    The following inputs are supported::

        /path/to/foo.*
        /path/to/*.txt
        /path/to/foo
    """
    p = Path(path).expanduser()
    s = str(p)

    if p.exists() and p.is_file():
        return p

    if any(ch in s for ch in ("*", "?", "[")):
        matches = [Path(m) for m in sorted(_glob.glob(s))]
        matches = [m for m in matches if m.is_file()]
        if not matches:
            return None
        for ext in prefer_exts:
            for m in matches:
                if m.suffix.lower() == ext.lower():
                    return m
        return matches[0]

    if s.endswith(".*"):
        base = s[:-2]
        for ext in prefer_exts:
            cand = Path(base + ext)
            if cand.exists() and cand.is_file():
                return cand
        parent = Path(base).parent
        stem = Path(base).name
        globbed = [Path(m) for m in sorted(_glob.glob(str(parent / (stem + ".*"))))]
        globbed = [m for m in globbed if m.is_file()]
        if not globbed:
            return None
        for ext in prefer_exts:
            for m in globbed:
                if m.suffix.lower() == ext.lower():
                    return m
        return globbed[0]

    if p.suffix == "":
        for ext in prefer_exts:
            cand = p.with_suffix(ext)
            if cand.exists() and cand.is_file():
                return cand

    return None


# ------------------------------ open helpers ------------------------------

def open_path(path: str | os.PathLike[str] | Path) -> bool:
    """Open a file or folder with the operating-system default handler."""
    p = Path(path).expanduser()
    if not p.exists():
        return False
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(p)])
            return True
        if os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
        subprocess.Popen(["xdg-open", str(p)])
        return True
    except Exception:
        return False


# ------------------------------ IO helpers ------------------------------

def load_text(path: Path) -> str:
    """Load text from a file using UTF-8 with replacement for bad bytes."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def save_text(path: Path, text: str) -> None:
    """Save text to a file using UTF-8 and create parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from a file."""
    data = json.loads(load_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def save_json(path: Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    """Save a mapping as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=False) + "\n", encoding="utf-8")


# ------------------------------ subprocess runner ------------------------------

@dataclass
class CmdResult:
    """Result returned by the asynchronous command runner."""

    returncode: int
    stdout: str
    stderr: str


def run_cmd_async(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    on_line: Optional[Callable[[str], None]] = None,
    on_done: Optional[Callable[[CmdResult], None]] = None,
) -> threading.Thread:
    """Run a command in a background thread.

    Parameters
    ----------
    cmd:
        Command and arguments.
    cwd:
        Optional working directory.
    env:
        Optional environment mapping.
    timeout:
        Optional timeout in seconds.
    on_line:
        Callback invoked for each captured stdout or stderr line. Stderr lines
        are prefixed with ``"STDERR: "``.
    on_done:
        Callback invoked once with ``CmdResult``.

    Returns
    -------
    threading.Thread
        The started daemon thread.
    """

    def _worker() -> None:
        try:
            p = subprocess.Popen(
                list(cmd),
                cwd=str(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            if on_done:
                on_done(CmdResult(1, "", f"{e}"))
            return

        out_lines: List[str] = []
        err_lines: List[str] = []

        def _drain(stream, sink: List[str], prefix: str) -> None:
            if stream is None:
                return
            for line in stream:
                if line is None:
                    continue
                s = line.rstrip("\n")
                sink.append(s)
                if on_line:
                    on_line(f"{prefix}{s}")

        t1 = threading.Thread(target=_drain, args=(p.stdout, out_lines, ""), daemon=True)
        # Many CLIs write warnings to stderr; prefix as STDERR instead of ERR.
        t2 = threading.Thread(target=_drain, args=(p.stderr, err_lines, "STDERR: "), daemon=True)
        t1.start()
        t2.start()

        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            if on_done:
                on_done(CmdResult(124, "\n".join(out_lines), "\n".join(err_lines + ["timeout"])))
            return

        t1.join(timeout=0.5)
        t2.join(timeout=0.5)

        if on_done:
            on_done(CmdResult(p.returncode or 0, "\n".join(out_lines), "\n".join(err_lines)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def last_nonempty_line(text: str) -> str:
    """Return the last non-empty line in a text block."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def preview_cmd(cmd: Sequence[str], *, prefix: str = "runroot") -> str:
    """Build a compact command preview string for the GUI."""
    if not cmd:
        return ""
    out: List[str] = []
    if prefix:
        out.append(prefix)
    if cmd[0] == sys.executable:
        out.append("python")
        out.extend(cmd[1:])
    else:
        out.extend(cmd)
    return " ".join(out)


__all__ = [
    "CmdResult",
    "ensure_dir",
    "extract_dpg_file_dialog_path",
    "find_repo_root",
    "in_dir",
    "is_inside",
    "last_nonempty_line",
    "load_json",
    "load_text",
    "open_path",
    "out_dir",
    "preview_cmd",
    "rel_to_in",
    "resolve_input_pattern",
    "run_cmd_async",
    "save_json",
    "save_text",
    "unique_path",
]
