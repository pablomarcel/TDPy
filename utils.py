from __future__ import annotations

"""
utils

Small, dependency-light utilities used across 

Goals:
- deterministic, CLI-friendly behavior
- safe JSON serialization helpers
- consistent error-context wrapping
- predictable logging without duplicate handlers
- light-weight dict override utilities (for CLI --set style overrides)
"""

import functools
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, TypeVar

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    """
    Ensure a directory exists.

    If `path` looks like a file path (has a suffix), create its parent directory.
    Otherwise, create the directory itself.
    """
    p = Path(path)
    target = p.parent if p.suffix else p
    if str(target) and not target.exists():
        target.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

_LOGGER_INIT: set[str] = set()


def _parse_env_log_level(env_value: str, default: int) -> int:
    s = str(env_value).strip()
    if not s:
        return default
    # Allow numeric levels ("10") or names ("DEBUG")
    try:
        return int(s)
    except Exception:
        return int(getattr(logging, s.upper(), default))


def setup_logger(name: str = "tdpy", level: int = logging.INFO) -> logging.Logger:
    """
    Deterministic logger setup.

    - Avoids duplicate handlers even if called repeatedly
    - Default INFO; can be overridden via env TDPY_LOG_LEVEL (e.g. DEBUG, INFO, WARNING, 10, 20)
    """
    env_level = os.environ.get("TDPY_LOG_LEVEL")
    if env_level is not None:
        level = _parse_env_log_level(env_level, level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers. Also tolerate external logging configuration.
    if name in _LOGGER_INIT or logger.handlers:
        return logger

    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    _LOGGER_INIT.add(name)
    return logger


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def json_default(obj: Any) -> Any:
    """
    Default handler for json.dumps.

    Supports:
      - dataclasses
      - pathlib.Path
      - simple iterables (set/tuple) -> list
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_dumps(data: Any, *, indent: int = 2) -> str:
    """Convenience wrapper for consistent JSON output."""
    return json.dumps(data, indent=indent, default=json_default)


# -----------------------------------------------------------------------------
# Timing / error-context decorators
# -----------------------------------------------------------------------------

def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function calls and log elapsed seconds (DEBUG level)."""
    logger = logging.getLogger("tdpy")  # don't rebuild handlers on every call

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.debug("%s took %.6fs", fn.__name__, dt)
        return out

    return wrapper


def with_error_context(context: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory: wraps exceptions with extra context.

    Keeps the original exception as __cause__ so tracebacks stay useful.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                et = type(e).__name__
                msg = str(e)
                raise RuntimeError(f"{context}: {et}: {msg}") from e

        return wrapper

    return deco


# -----------------------------------------------------------------------------
# Env var helper
# -----------------------------------------------------------------------------

def env_path(*parts: str, default: str | None = None) -> str | None:
    """
    Read env var like TDPY_FOO_BAR for parts ("foo","bar").
    Returns string or default.
    """
    key = "_".join(["TDPY", *parts]).upper()
    return os.environ.get(key, default)


# -----------------------------------------------------------------------------
# Dict / override helpers (used for CLI --set and future GUI editing)
# -----------------------------------------------------------------------------

def deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge src into dst (in-place) and return dst.
    Dicts merge deeply, other types overwrite.
    """
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            deep_update(dst[k], v)  # type: ignore[arg-type]
        else:
            dst[k] = v
    return dst


def dotted_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Get mapping["a"]["b"]["c"] using path "a.b.c"."""
    cur: Any = mapping
    for part in str(path).split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dotted_set(mapping: MutableMapping[str, Any], path: str, value: Any) -> None:
    """
    Set mapping["a"]["b"]["c"] = value using path "a.b.c".
    Creates intermediate dicts as needed.
    """
    parts = str(path).split(".")
    if not parts or any(p.strip() == "" for p in parts):
        raise ValueError(f"Invalid dotted path: {path!r}")

    cur: MutableMapping[str, Any] = mapping
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[p] = nxt
        cur = nxt  # type: ignore[assignment]
    cur[parts[-1]] = value


def coerce_scalar(text: str) -> Any:
    """
    Coerce a string into bool/int/float if it looks like one; otherwise return the original string.
    """
    s = str(text).strip()
    if s == "":
        return ""

    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"

    # int (avoid octal-ish surprises)
    try:
        if s.startswith("0") and s not in ("0", "0.0") and not s.startswith("0."):
            raise ValueError
        return int(s)
    except Exception:
        pass

    # float
    try:
        return float(s)
    except Exception:
        return s


def parse_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    """
    Parse CLI-style overrides:
      ["a.b=3", "foo=true", "bar.baz=1e-6"]

    Returns a nested dict suitable for deep_update().
    """
    out: Dict[str, Any] = {}
    for raw in pairs:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override must be key=value; got {raw!r}")
        k, v = item.split("=", 1)
        dotted_set(out, k.strip(), coerce_scalar(v))
    return out


# -----------------------------------------------------------------------------
# Optional dependency hints (nice UX for CLI)
# -----------------------------------------------------------------------------

def require_package(import_name: str, pip_name: Optional[str] = None) -> None:
    """Raise a clear ImportError for optional dependencies."""
    try:
        __import__(import_name)
    except Exception as e:
        pkg = pip_name or import_name
        raise ImportError(
            f"Missing optional dependency {import_name!r}. Install with: pip install {pkg}"
        ) from e


# -----------------------------------------------------------------------------
# Small convenience helpers
# -----------------------------------------------------------------------------

def now_iso() -> str:
    """UTC-ish timestamp string; good enough for meta fields."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def clamp(x: float, lo: float | None = None, hi: float | None = None) -> float:
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x
