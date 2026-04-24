from __future__ import annotations

"""
equations.spec

Pure-data specifications for an EES-like nonlinear equation system.

Design goals:
- lightweight (no SciPy/GEKKO imports)
- dataclass-based (JSON-friendly)
- GUI-friendly fields (desc, unit, bounds, label)
- robust input coercion across multiple historical JSON shapes

Supported input shapes
----------------------
A) "spec-native" (older / internal)
   {
     "vars": {...}, "params": {...}, "equations": [...], "meta": {...}
   }

B) "app/CLI" (current JSON)
   {
     "problem_type": "equations",
     "solver": "scipy" | "gekko" | "auto",
     "method": "hybr" | "krylov" | "anderson" | "broyden1" | ...,
     "tol": ...,
     "max_iter": ...,
     "constants": {...},
     "variables": {...} | [ {name, guess, ...}, ... ],
     "equations": [...]
   }

C) "solve block" (preferred overrides)
   {
     "solve": {
        "backend":"scipy",
        "method":"hybr",
        "tol":1e-8,
        "max_iter":200,
        "max_restarts":2,

        # Warm-start / guess prepass (optional; solver layer implements behavior)
        "warm_start": true,
        "warm_start_passes": 3,
        "warm_start_tol": 1e-6,
        "warm_start_relax": 1.0,
        "warm_start_only_assignments": false
     },
     "variables": {...},
     "constants": {...},
     "equations": [...]
   }

Quality-of-life supported:
- variables may be a list of dicts with "name"
- equations may be a single string
- vars/variables entries may be shorthand scalars: {"x": 1.0} meaning unknown x with guess 1.0
- params/constants allow non-floats (fluid="Nitrogen") and quantity strings if units are available
- variable fixed/unknown detection supports:
    fixed: true/false
    kind: "fixed"|"unknown"
    unknown: true/false
  plus an inference rule:
    if user provides "value" but not "guess" and no explicit fixed/unknown/kind flags,
    treat it as fixed (matches common JSON expectations).

Notes:
- This module *stores* warm-start configuration in `EquationSystemSpec.solve` and does
  minimal coercion. The actual warm-start algorithm lives in solver.py.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple


# ------------------------------ errors ------------------------------

class SpecError(ValueError):
    """Raised when an equation system spec is malformed."""


# ------------------------------ optional units helpers ------------------------------
# Compatibility with multiple units implementations used in your repo:
#   (a) parse_quantity(text, DEFAULT_REGISTRY) -> object with .base_value()
#   (b) parse_quantity(text) -> quantity-like
try:
    from units import DEFAULT_REGISTRY, parse_quantity  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_REGISTRY = None  # type: ignore[assignment]
    parse_quantity = None  # type: ignore[assignment]


def _has_units() -> bool:
    return parse_quantity is not None


def _quantity_to_float(q: Any) -> float:
    """
    Best-effort extraction from a parsed quantity object.

    Supports objects with:
      - .base_value() method (older implementation)
      - .base_value attribute
      - .magnitude / .value
      - fallback float(q)
    """
    if q is None:
        raise SpecError("parse_quantity returned None.")

    for attr in ("magnitude", "value"):
        if hasattr(q, attr):
            try:
                return float(getattr(q, attr))
            except Exception:
                pass

    if hasattr(q, "base_value"):
        bv = getattr(q, "base_value")
        try:
            return float(bv() if callable(bv) else bv)
        except Exception:
            pass

    try:
        return float(q)
    except Exception as e:
        raise SpecError(f"Could not convert quantity to float: {q!r}") from e


def _parse_quantity_value(text: str) -> float:
    """
    Call parse_quantity with best-effort signature compatibility.
    """
    if parse_quantity is None:
        raise SpecError("Units subsystem is not available (parse_quantity import failed).")

    s = str(text).strip()
    if not s:
        raise SpecError("Quantity string is empty.")

    if DEFAULT_REGISTRY is not None:
        try:
            q = parse_quantity(s, DEFAULT_REGISTRY)  # type: ignore[misc]
            return _quantity_to_float(q)
        except TypeError:
            pass
        except Exception:
            pass

    try:
        q = parse_quantity(s)  # type: ignore[misc]
        return _quantity_to_float(q)
    except Exception as e:
        raise SpecError(f"Could not parse quantity: {text!r}") from e


def _coerce_number_or_quantity(v: Any, *, unit_hint: str | None = None, what: str = "value") -> float:
    """
    Coerce a numeric value to float, optionally supporting:
      - strings like "300 K" (quantity parsing)
      - numeric strings like "300" with a unit_hint
      - numeric values with unit_hint

    If unit support is unavailable, falls back to plain float conversion.
    """
    if isinstance(v, bool):
        raise SpecError(f"{what} must be a number/quantity, not bool.")

    # Numeric
    if isinstance(v, (int, float)):
        x = float(v)
        if unit_hint and _has_units():
            # Prefer registry conversion if available
            try:
                if DEFAULT_REGISTRY is not None and hasattr(DEFAULT_REGISTRY, "has") and hasattr(DEFAULT_REGISTRY, "to_base"):
                    if DEFAULT_REGISTRY.has(unit_hint):  # type: ignore[union-attr]
                        return float(DEFAULT_REGISTRY.to_base(x, unit_hint))  # type: ignore[union-attr]
            except Exception:
                pass
            # Fallback: treat as "<x> <unit_hint>"
            try:
                return _parse_quantity_value(f"{x} {unit_hint}")
            except Exception:
                return x
        return x

    # String
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise SpecError(f"{what} must not be empty.")

        if _has_units():
            try:
                return _parse_quantity_value(s)
            except Exception:
                pass

        try:
            x = float(s)
        except Exception as e:
            raise SpecError(f"Could not parse {what} as a number/quantity: {v!r}") from e

        if unit_hint and _has_units():
            try:
                if DEFAULT_REGISTRY is not None and hasattr(DEFAULT_REGISTRY, "has") and hasattr(DEFAULT_REGISTRY, "to_base"):
                    if DEFAULT_REGISTRY.has(unit_hint):  # type: ignore[union-attr]
                        return float(DEFAULT_REGISTRY.to_base(x, unit_hint))  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                return _parse_quantity_value(f"{x} {unit_hint}")
            except Exception:
                pass

        return x

    raise SpecError(f"{what} must be a number or quantity string; got {type(v).__name__}.")


def _coerce_optional_number(v: Any, *, unit_hint: str | None = None, what: str = "value") -> float | None:
    if v is None:
        return None
    return _coerce_number_or_quantity(v, unit_hint=unit_hint, what=what)


def _coerce_param_value(v: Any) -> Any:
    """
    Parameters/constants may be:
    - float/int -> float
    - quantity string like "101.325 kPa" -> float (if units available)
    - arbitrary strings (e.g., fluid name) -> kept as string
    - other JSON scalars -> kept as-is
    """
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return ""
        if _has_units():
            try:
                return _parse_quantity_value(s)
            except Exception:
                return s
        return s
    return v


# ------------------------------ specs ------------------------------

@dataclass(frozen=True)
class VarSpec:
    """
    Variable specification.

    Unknown vs fixed:
      - fixed=True  -> known constant inside solve; requires `value` (guess ignored)
      - fixed=False -> unknown; requires `guess` (or value-as-guess)

    Units:
      - `unit` is a display hint and also used to interpret numeric guess/bounds.
      - guess/value may be a quantity string ("300 K") regardless of unit hint.
    """
    name: str
    value: float | None = None
    guess: float | None = None
    unit: str | None = None
    desc: str | None = None
    lower: float | None = None
    upper: float | None = None
    fixed: bool = False
    label: str | None = None  # GUI-friendly; optional

    @property
    def unknown(self) -> bool:
        return not self.fixed

    @property
    def kind(self) -> str:
        return "fixed" if self.fixed else "unknown"

    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (self.lower, self.upper)

    def guess_value(self, default: float = 1.0) -> float:
        """Preference: guess -> value -> default."""
        if self.guess is not None:
            return float(self.guess)
        if self.value is not None:
            return float(self.value)
        return float(default)


@dataclass(frozen=True)
class ParamSpec:
    """Parameter (constant) specification. Value may be float OR non-float (e.g., fluid string)."""
    name: str
    value: Any
    unit: str | None = None
    desc: str | None = None
    label: str | None = None  # GUI-friendly; optional


EquationKind = Literal["expr", "residual"]


@dataclass(frozen=True)
class EquationSpec:
    """
    Equation specification.

    kind="expr":
      - expr: expression string representing a constraint (residual == 0)

    kind="residual":
      - fn: name of a built-in residual provider (string key)
      - args: JSON-friendly kwargs payload for that residual provider
    """
    kind: EquationKind = "expr"
    expr: str | None = None
    fn: str | None = None
    args: Dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "expr":
            if not self.expr or not str(self.expr).strip():
                raise SpecError("EquationSpec(kind='expr') requires a non-empty expr.")
        elif self.kind == "residual":
            if not self.fn or not str(self.fn).strip():
                raise SpecError("EquationSpec(kind='residual') requires a non-empty fn.")
        else:
            raise SpecError(f"Unknown equation kind: {self.kind!r}")


@dataclass(frozen=True)
class EquationSystemSpec:
    """
    A full system definition: variables + parameters + equations.

    Optional solver configuration:
      - top-level hints: backend/method/tol/max_iter/max_restarts/use_units
      - ALSO stores raw solve block (solve: {...}) if provided by input JSON

    Backends are free to ignore hints. The solver layer decides behavior.
    """
    vars: Dict[str, VarSpec]
    equations: List[EquationSpec]
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # optional solve hints (legacy / simple)
    backend: str = "auto"
    method: str = "hybr"
    tol: float = 1e-9
    max_iter: int = 200
    max_restarts: int = 2
    use_units: bool = False

    # raw solve block (preferred override format in newer JSON)
    solve: Dict[str, Any] = field(default_factory=dict)

    @property
    def variables(self) -> List[VarSpec]:
        return list(self.vars.values())

    def params_for_eval(self) -> Dict[str, Any]:
        return {k: p.value for k, p in self.params.items()}

    def n_equations(self) -> int:
        return len(self.equations)

    def unknown_names(self) -> List[str]:
        return [v.name for v in self.vars.values() if not v.fixed]

    def n_unknowns(self) -> int:
        return sum(1 for v in self.vars.values() if not v.fixed)

    def validate(self) -> None:
        _validate_names(self.vars.keys(), what="vars")
        _validate_names(self.params.keys(), what="params")

        for k, v in self.vars.items():
            if k != v.name:
                raise SpecError(f"VarSpec key/name mismatch: key={k!r} name={v.name!r}")
            _validate_bounds(v)
            _validate_var_required_fields(v)

        for k, p in self.params.items():
            if k != p.name:
                raise SpecError(f"ParamSpec key/name mismatch: key={k!r} name={p.name!r}")

        if not self.equations:
            raise SpecError("EquationSystemSpec must include at least one equation.")

        if self.tol <= 0:
            raise SpecError(f"tol must be > 0; got {self.tol}")
        if self.max_iter <= 0:
            raise SpecError(f"max_iter must be > 0; got {self.max_iter}")
        if self.max_restarts < 0:
            raise SpecError(f"max_restarts must be >= 0; got {self.max_restarts}")

        # Warm-start hints are optional; we only do light coercion checks here.
        ws = self.solve.get("warm_start", None)
        if ws is not None and not isinstance(ws, (bool, int, str)):
            raise SpecError(f"solve.warm_start must be boolean-like; got {type(ws).__name__}")

    def check_square(self) -> None:
        """EES-style determinism: number of equations equals number of unknowns."""
        self.validate()
        ne = self.n_equations()
        nu = self.n_unknowns()
        if ne != nu:
            raise SpecError(
                f"System is not square: equations={ne} unknowns={nu}. Unknowns={self.unknown_names()}"
            )


# ------------------------------ helpers ------------------------------

def _validate_names(names: Any, what: str) -> None:
    bad: List[str] = []
    for n in names:
        s = str(n)
        if not s or s.strip() != s:
            bad.append(s)
            continue
        if not (s[0].isalpha() or s[0] == "_"):
            bad.append(s)
            continue
        for ch in s:
            if not (ch.isalnum() or ch == "_"):
                bad.append(s)
                break
    if bad:
        raise SpecError(f"Invalid {what} names: {bad}")


def _validate_bounds(v: VarSpec) -> None:
    lo, hi = v.lower, v.upper
    if lo is not None and hi is not None and hi < lo:
        raise SpecError(f"Invalid bounds for {v.name!r}: upper < lower ({hi} < {lo})")


def _validate_var_required_fields(v: VarSpec) -> None:
    if v.fixed:
        if v.value is None:
            raise SpecError(f"Fixed variable {v.name!r} must have a value.")
    else:
        if v.guess is None and v.value is None:
            raise SpecError(f"Unknown variable {v.name!r} must have a guess (or value-as-guess).")


_BACKEND_ALIASES: Dict[str, str] = {
    "": "auto",
    "none": "auto",
    "default": "auto",
    "auto": "auto",
    "scipy": "scipy",
    "root": "scipy",
    "scipy-root": "scipy",
    "optimize": "scipy",
    "gekko": "gekko",
    "ipopt": "gekko",
    "apopt": "gekko",
}


def _normalize_backend(x: Any) -> str:
    b = str(x or "auto").strip().lower()
    return _BACKEND_ALIASES.get(b, b)


def _as_dict_if_mapping(x: Any) -> Dict[str, Any]:
    if isinstance(x, Mapping):
        return dict(x)
    return {}


def _coerce_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if s in {"0", "false", "no", "n", "f", "off"}:
        return False
    return default


def _infer_fixed_flag(payload: Mapping[str, Any]) -> bool:
    """
    Determine fixed/unknown intent with multiple supported knobs:
      - fixed: true/false (wins)
      - unknown: true/false
      - kind: "fixed" | "unknown"
    Plus inference:
      - If "value" is present but "guess" is not (and no explicit flags),
        treat as fixed.
    Default: unknown (fixed=False).
    """
    if "fixed" in payload:
        return _coerce_bool(payload.get("fixed"), default=False)
    if "unknown" in payload:
        return not _coerce_bool(payload.get("unknown"), default=True)
    if "kind" in payload:
        k = str(payload.get("kind") or "").strip().lower()
        if k == "fixed":
            return True
        if k == "unknown":
            return False

    has_value = payload.get("value", None) is not None
    has_guess = payload.get("guess", None) is not None
    if has_value and not has_guess:
        return True

    return False


# ------------------------------ builders ------------------------------

def system_from_mapping(m: Mapping[str, Any]) -> EquationSystemSpec:
    """
    Construct an EquationSystemSpec from a JSON/YAML mapping.

    Robustness upgrades:
    - accepts variables as mapping OR list[{name,...}]
    - accepts equations as list OR single string
    - merges params + constants (constants wins)
    - tolerates shorthand scalar variable entries: {"x": 1.0}
    - supports solve-block overrides under "solve": {...}
    - validates problem_type if present
    """
    if not isinstance(m, Mapping):
        raise SpecError("Equation system must be a mapping/dict.")

    # If present, enforce problem_type
    pt = m.get("problem_type", None)
    if pt is not None:
        spt = str(pt).strip().lower()
        if spt and spt != "equations":
            raise SpecError(f"Invalid problem_type for equations spec: {pt!r} (expected 'equations').")

    # ---- solve block (preferred override) ----
    solve_block = _as_dict_if_mapping(m.get("solve", None))

    # ---- solver hints (optional), with solve-block override precedence ----
    backend = _normalize_backend(
        solve_block.get("backend", None)
        or solve_block.get("solver", None)
        or m.get("backend", None)
        or m.get("solver", None)
        or "auto"
    )
    method = str(solve_block.get("method", None) or m.get("method", None) or "hybr")

    tol_raw = solve_block.get("tol", None)
    if tol_raw is None:
        tol_raw = m.get("tol", 1e-9)
    tol = float(tol_raw or 1e-9)

    mi_raw = (
        solve_block.get("max_iter", None)
        or solve_block.get("maxiter", None)
        or m.get("max_iter", None)
        or m.get("maxiter", None)
        or 200
    )
    max_iter = int(mi_raw)

    mr_raw = (
        solve_block.get("max_restarts", None)
        or solve_block.get("restarts", None)
        or m.get("max_restarts", None)
        or 2
    )
    max_restarts = int(mr_raw)

    use_units = _coerce_bool(
        solve_block.get("use_units", None),
        default=_coerce_bool(m.get("use_units", False)),
    )

    # Warm-start keys: we don’t force them into top-level fields (to keep API stable),
    # but we DO normalize common aliases so solver.py can rely on consistent keys.
    solve_norm = dict(solve_block)
    if "warm_start_passes" not in solve_norm and "warm_start_max_passes" in solve_norm:
        solve_norm["warm_start_passes"] = solve_norm.get("warm_start_max_passes")
    if "warm_start_relax" not in solve_norm and "warm_start_damping" in solve_norm:
        solve_norm["warm_start_relax"] = solve_norm.get("warm_start_damping")

    # ---- variables ----
    vars_out: Dict[str, VarSpec] = {}

    def _add_var(nm: str, payload: Any) -> None:
        if not nm or nm.strip() != nm:
            raise SpecError(f"Invalid variable name: {nm!r}")

        # Shorthand scalar: {"x": 1.0} -> unknown with guess 1.0
        if not isinstance(payload, Mapping):
            guess = _coerce_optional_number(payload, unit_hint=None, what=f"vars[{nm}]")
            if guess is None:
                raise SpecError(f"Unknown variable {nm!r} must have a guess/value.")
            vars_out[nm] = VarSpec(name=nm, value=None, guess=float(guess), fixed=False)
            return

        unit_hint = payload.get("unit", payload.get("units", None))
        unit_hint = str(unit_hint) if unit_hint is not None else None

        fixed = _infer_fixed_flag(payload)

        desc = payload.get("desc", None)
        desc = str(desc) if desc is not None else None
        label = payload.get("label", None)
        label = str(label) if label is not None else None

        # bounds
        lower: float | None = None
        upper: float | None = None
        b = payload.get("bounds", None)
        if isinstance(b, (list, tuple)) and len(b) == 2:
            lower = _coerce_optional_number(b[0], unit_hint=unit_hint, what=f"vars[{nm}].bounds[0]")
            upper = _coerce_optional_number(b[1], unit_hint=unit_hint, what=f"vars[{nm}].bounds[1]")
        else:
            lower = _coerce_optional_number(payload.get("lower", None), unit_hint=unit_hint, what=f"vars[{nm}].lower")
            upper = _coerce_optional_number(payload.get("upper", None), unit_hint=unit_hint, what=f"vars[{nm}].upper")

        if fixed:
            v_raw = payload.get("value", None)
            if v_raw is None:
                # allow fixed variables provided as a "guess" only (common in quick GUIs)
                v_raw = payload.get("guess", None)
            value = _coerce_optional_number(v_raw, unit_hint=unit_hint, what=f"vars[{nm}].value")
            vars_out[nm] = VarSpec(
                name=nm,
                value=value,
                guess=None,
                unit=unit_hint,
                desc=desc,
                lower=lower,
                upper=upper,
                fixed=True,
                label=label,
            )
        else:
            # Unknown: prefer guess; allow value-as-guess
            g_raw = payload.get("guess", None)
            if g_raw is None:
                g_raw = payload.get("value", None)
            guess = _coerce_optional_number(g_raw, unit_hint=unit_hint, what=f"vars[{nm}].guess/value")
            vars_out[nm] = VarSpec(
                name=nm,
                value=None,
                guess=guess,
                unit=unit_hint,
                desc=desc,
                lower=lower,
                upper=upper,
                fixed=False,
                label=label,
            )

    raw_vars: Any
    if "variables" in m:
        raw_vars = m.get("variables", {})
    elif "vars" in m:
        raw_vars = m.get("vars", {})
    else:
        raise SpecError("Equation system mapping must include either 'variables' or 'vars'.")

    if isinstance(raw_vars, Mapping):
        for name, payload in raw_vars.items():
            _add_var(str(name), payload)
    elif isinstance(raw_vars, list):
        for i, payload in enumerate(raw_vars):
            if not isinstance(payload, Mapping):
                raise SpecError(f"variables[{i}] must be a mapping with at least 'name'.")
            nm = str(payload.get("name", "")).strip()
            if not nm:
                raise SpecError(f"variables[{i}] missing required 'name'.")
            _add_var(nm, payload)
    else:
        raise SpecError("'variables'/'vars' must be a mapping or a list.")

    # ---- parameters / constants (merge) ----
    params_out: Dict[str, ParamSpec] = {}

    def _merge_params(src: Any, *, src_name: str) -> None:
        if src is None:
            return
        if not isinstance(src, Mapping):
            raise SpecError(f"'{src_name}' must be a mapping.")
        for name, payload in src.items():
            nm = str(name).strip()
            if not nm:
                raise SpecError(f"Invalid param name: {name!r}")

            if isinstance(payload, Mapping):
                unit_hint = payload.get("unit", payload.get("units", None))
                unit_hint = str(unit_hint) if unit_hint is not None else None
                desc = payload.get("desc", None)
                desc = str(desc) if desc is not None else None
                label = payload.get("label", None)
                label = str(label) if label is not None else None

                v_raw = payload.get("value", None)

                vv: Any
                if isinstance(v_raw, (int, float)) and not isinstance(v_raw, bool):
                    vv = _coerce_number_or_quantity(v_raw, unit_hint=unit_hint, what=f"{src_name}[{nm}].value")
                elif isinstance(v_raw, str) and _has_units():
                    # attempt quantity parse, but keep string if it isn't a quantity
                    try:
                        vv = _parse_quantity_value(v_raw.strip())
                    except Exception:
                        vv = v_raw.strip()
                else:
                    vv = _coerce_param_value(v_raw)

                params_out[nm] = ParamSpec(name=nm, value=vv, unit=unit_hint, desc=desc, label=label)
            else:
                params_out[nm] = ParamSpec(name=nm, value=_coerce_param_value(payload))

    _merge_params(m.get("params", None), src_name="params")
    _merge_params(m.get("constants", None), src_name="constants")  # constants override

    # ---- equations ----
    raw_eqs = m.get("equations", [])
    if isinstance(raw_eqs, str):
        raw_eqs = [raw_eqs]
    if not isinstance(raw_eqs, list):
        raise SpecError("'equations' must be a list (or a single string).")

    eqs_out: List[EquationSpec] = []
    for idx, item in enumerate(raw_eqs):
        if item is None:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                eqs_out.append(EquationSpec(kind="expr", expr=s))
            continue
        if not isinstance(item, Mapping):
            raise SpecError(f"equations[{idx}] must be a string or a mapping/dict.")
        kind = str(item.get("kind", "expr"))
        eqs_out.append(
            EquationSpec(
                kind=kind,  # type: ignore[arg-type]
                expr=item.get("expr"),
                fn=item.get("fn"),
                args=dict(item.get("args", {})) if isinstance(item.get("args", {}), Mapping) else {},
                label=item.get("label"),
            )
        )

    # ---- meta ----
    meta = dict(m.get("meta", {})) if isinstance(m.get("meta", {}), Mapping) else {}
    for k in ("problem_type", "title", "note", "notes"):
        if k in m and k not in meta:
            meta[k] = m.get(k)

    spec = EquationSystemSpec(
        vars=vars_out,
        params=params_out,
        equations=eqs_out,
        meta=meta,
        backend=backend,
        method=method,
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        use_units=use_units,
        solve=solve_norm,
    )
    spec.validate()
    return spec


__all__ = [
    "SpecError",
    "VarSpec",
    "ParamSpec",
    "EquationSpec",
    "EquationSystemSpec",
    "system_from_mapping",
]
