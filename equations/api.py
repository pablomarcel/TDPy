# equations/api.py
from __future__ import annotations

"""
equations.api

Public, stable facade for the EES-ish nonlinear equation solving layer.

Role of this file:
- Accept either an EquationSystemSpec or a JSON/YAML mapping
- Validate/normalize the spec
- Adapt rich spec -> solver.py's duck-typed shape
- Keep imports light; heavy deps remain optional and are imported lazily

Upgrades in this version:
- Works with the *new* spec.py model (VarSpec/ParamSpec/EquationSpec/EquationSystemSpec)
- Supports spec.solve (raw solve-block dict) and forwards it to solver.py
- Preserves non-numeric params (e.g., fluid="Nitrogen") for PropsSI(..., fluid)
- API args override spec/solve-block ONLY when explicitly provided
  (backend/method/tol/max_iter/max_restarts/use_units)
- Best-effort unit conversion support via units if present
- Explicit backend availability checks (helpful errors) when user requests scipy/gekko
- Registers property functions into the evaluation context when equations reference them:
    * PropsSI / PhaseSI
    * HAPropsSI
    * CTPropsSI / CTPropsMulti / CTBatchProps (Cantera)
    * LiBrPropsSI / LiBrH2OPropsSI (LiBr–H2O engine)
    * LiBr helper calls (EES-style convenience):
        - LiBrX_TP, LiBrH_TX, LiBrRho_TXP, LiBrT_HX
    * NH3–H2O (ammonia-water) property engine + helpers (native backend):
        - NH3H2O / NH3H2O_STATE / NH3H2O_TPX / NH3H2O_STATE_TPX
        - optional aliases: prop_tpx / state_tpx / props_multi_tpx / batch_prop_tpx
        - optional PropsSI-like aliases if you expose them:
            NH3H2OPropsSI / NH3H2OPropsMulti / NH3H2OBatchProps

Notes:
- If equations contain Python-callable property-function calls (PropsSI/PhaseSI/HAPropsSI/CTPropsSI/ASPropsSI/LiBr*/NH3H2O*),
  GEKKO backends cannot handle them. In that case we require SciPy (or auto -> SciPy).

Contract note (Feb 2026):
- `thermo_props.coolprop_backend` is the *central contract/registration point* for thermo callables.
  Cantera is implemented in `thermo_props.cantera_backend`, but we prefer to access CTPropsSI
  via coolprop_backend's delegation wrappers to keep a single import surface.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import re

from .spec import (
    EquationSpec,
    EquationSystemSpec,
    ParamSpec,
    VarSpec,
    system_from_mapping,
)

if TYPE_CHECKING:  # pragma: no cover
    from .solver import SolveResult as Solution  # noqa: F401
else:
    Solution = Any  # runtime-friendly alias


BackendKind = Literal["auto", "gekko", "scipy"]


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend isn't installed/available."""


# Public aliases expected by equations/__init__.py
EquationSystem = EquationSystemSpec
Var = VarSpec
Param = ParamSpec


# ------------------------------ backend availability ------------------------------


def _has_gekko() -> bool:
    try:
        import gekko  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _has_scipy() -> bool:
    try:
        import scipy  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _normalize_backend_name(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"", "none"}:
        return "auto"
    if s in {"root", "scipy-root", "optimize"}:
        return "scipy"
    if s in {"ipopt", "apopt"}:
        return "gekko"
    if s in {"scipyoptimize"}:
        return "scipy"
    return s


def _ensure_backend_available(backend: str) -> None:
    b = _normalize_backend_name(backend)
    if b in {"", "none", "auto"}:
        return
    if b == "gekko" and not _has_gekko():
        raise BackendUnavailableError("Backend 'gekko' requested but GEKKO is not installed.")
    if b == "scipy" and not _has_scipy():
        raise BackendUnavailableError("Backend 'scipy' requested but SciPy is not installed.")


# ------------------------------ optional units integration ------------------------------


@dataclass(frozen=True)
class _UnitAdapter:
    """
    Tiny wrapper for optional unit conversion.

    Expected capability (best-effort):
      - convert(value, from_units, to_units) -> float
    """

    convert: Any  # Callable[[float, str, str], float]


def _try_units_adapter() -> Optional[_UnitAdapter]:
    """
    Attempt to import units lazily.

    Supported minimal APIs (best-effort):
      - convert(value: float, from_units: str, to_units: str) -> float
      OR
      - DEFAULT_REGISTRY with .convert(value, from_units, to_units)
    """
    try:
        import units as _units
    except Exception:
        return None

    conv = getattr(_units, "convert", None)
    if callable(conv):
        return _UnitAdapter(convert=conv)

    reg = getattr(_units, "DEFAULT_REGISTRY", None)
    if reg is not None and callable(getattr(reg, "convert", None)):
        return _UnitAdapter(convert=reg.convert)

    return None


def _norm_units(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip()
    return s if s else None


# ------------------------------ adapter types ------------------------------


@dataclass
class _VarLike:
    """
    Internal variable object matching what solver.py expects via duck-typing.

    solver.py checks unknowns using (in order):
      - v.unknown (bool) if present
      - v.kind == "unknown" if present
      - v.value is None
    """

    name: str
    kind: str
    value: float | None
    guess: float | None = None
    lower: float | None = None
    upper: float | None = None

    @property
    def unknown(self) -> bool:
        return str(self.kind).lower() == "unknown"


@dataclass
class _SpecLike:
    """
    Internal spec object matching what solver.py expects via duck-typing.
    """

    equations: List[str]
    variables: List[_VarLike]
    params: Dict[str, Any]

    solve: Dict[str, Any] | None = None

    backend: str | None = None
    solver: str | None = None
    method: str | None = None
    tol: float | None = None
    max_iter: int | None = None
    maxiter: int | None = None
    max_restarts: int | None = None


# ------------------------------ conversion helpers ------------------------------


def _equations_to_strings(eqs: Sequence[EquationSpec]) -> List[str]:
    out: List[str] = []
    for e in eqs:
        if e.kind == "expr":
            out.append(str(e.expr))
            continue
        if e.kind == "residual":
            raise NotImplementedError(
                "EquationSpec(kind='residual') is not implemented yet in the backend adapter. "
                "For now use kind='expr' with an explicit equation string."
            )
        raise ValueError(f"Unknown EquationSpec.kind: {e.kind!r}")
    return out


def _extract_base_units(spec: EquationSystemSpec) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    meta = getattr(spec, "meta", None)
    if not isinstance(meta, Mapping):
        return None, None
    u = meta.get("units")
    if not isinstance(u, Mapping):
        return None, None
    vars_u = u.get("vars")
    params_u = u.get("params")
    vars_base = dict(vars_u) if isinstance(vars_u, Mapping) else None
    params_base = dict(params_u) if isinstance(params_u, Mapping) else None
    return vars_base, params_base


def _params_to_values(
    params_map: Mapping[str, ParamSpec],
    *,
    units: Optional[_UnitAdapter] = None,
    base_units: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in params_map.values():
        name = str(p.name)
        val: Any = p.value

        if isinstance(val, (int, float)) and not isinstance(val, bool):
            x = float(val)
            if units is not None and base_units is not None:
                u_from = _norm_units(getattr(p, "unit", None))
                u_to = _norm_units(base_units.get(name))
                if u_from and u_to and u_from != u_to:
                    try:
                        x = float(units.convert(x, u_from, u_to))
                    except Exception:
                        pass
            out[name] = x
        else:
            out[name] = val

    return out


def _vars_to_varlikes(
    vars_map: Mapping[str, VarSpec],
    *,
    units: Optional[_UnitAdapter] = None,
    base_units: Optional[Mapping[str, str]] = None,
) -> List[_VarLike]:
    out: List[_VarLike] = []

    for v in vars_map.values():
        name = str(v.name)

        u_from = _norm_units(getattr(v, "unit", None))
        u_to = _norm_units(base_units.get(name)) if base_units is not None else None

        def _conv(x: float | None) -> float | None:
            if x is None:
                return None
            if units is None or base_units is None:
                return float(x)
            if not u_from or not u_to or u_from == u_to:
                return float(x)
            try:
                return float(units.convert(float(x), u_from, u_to))
            except Exception:
                return float(x)

        lo2 = _conv(v.lower)
        hi2 = _conv(v.upper)

        if bool(v.fixed):
            if v.value is None:
                raise ValueError(f"Fixed variable {name!r} is missing a value.")
            value = _conv(float(v.value))
            out.append(
                _VarLike(
                    name=name,
                    kind="fixed",
                    value=float(value) if value is not None else float(v.value),
                    guess=None,
                    lower=lo2,
                    upper=hi2,
                )
            )
        else:
            guess = _conv(float(v.guess_value(default=1.0)))
            out.append(
                _VarLike(
                    name=name,
                    kind="unknown",
                    value=None,
                    guess=float(guess) if guess is not None else float(v.guess_value(default=1.0)),
                    lower=lo2,
                    upper=hi2,
                )
            )

    return out


def _strip_solve_keys(solve: Dict[str, Any], keys: Sequence[str]) -> None:
    for k in keys:
        if k in solve:
            solve.pop(k, None)


def _import_first(candidates: Sequence[str]) -> Any | None:
    for mod in candidates:
        try:
            return import_module(f"thermo_props.{mod}")
        except Exception:
            continue
    return None


# ------------------------------ property-function detection + injection ------------------------------

_RE_PROPS = re.compile(r"\bPropsSI\s*\(")
_RE_PHASE = re.compile(r"\bPhaseSI\s*\(")
_RE_HA = re.compile(r"\bHAPropsSI\s*\(")

# Cantera (CTPropsSI family) + optional cache helpers
_RE_CT = re.compile(
    r"\b(?:"
    r"CTPropsSI|CTPropsMulti|CTBatchProps|"
    r"ctprops_si|ctprops_multi|batch_ctprops|"
    r"cantera_available|"
    r"ctprops_cache_info|clear_ctprops_caches"
    r")\s*\("
)

# AbstractState family (CoolProp AbstractState wrappers)
_RE_AS_PROPS = re.compile(r"\bASPropsSI\s*\(")
_RE_AS_MULTI = re.compile(r"\bASPropsMulti\s*\(")
_RE_AS_BATCH = re.compile(r"\bASBatchProps\s*\(")
_RE_FUG = re.compile(r"\bFugacitySI\s*\(")
_RE_FUGCOEFF = re.compile(r"\bFugacityCoeffSI\s*\(")
_RE_LNPHI = re.compile(r"\bLnFugacityCoeffSI\s*\(")
_RE_CHEMPOT = re.compile(r"\bChemicalPotentialSI\s*\(")

# AbstractState internal aliases (if users call the backend helpers directly)
_RE_AS_SI = re.compile(r"\bas_props_si\s*\(")
_RE_AS_MULTI2 = re.compile(r"\bas_props_multi\s*\(")
_RE_AS_BATCH2 = re.compile(r"\bbatch_as_props\s*\(")


# LiBr family (PropsSI-like + helpers + internal aliases)
_RE_LIBR = re.compile(
    r"\b(?:"
    r"LiBrPropsSI|LiBrH2OPropsSI|"
    r"LiBrX_TP|LiBrH_TX|LiBrRho_TXP|LiBrT_HX|"
    r"LiBrPropsMulti|LiBrBatchProps|"
    r"librh2o_props_si|librh2o_props_multi|batch_librh2o_props"
    r")\s*\("
)
_RE_LIBR_X_TP = re.compile(r"\bLiBrX_TP\s*\(")
_RE_LIBR_H_TX = re.compile(r"\bLiBrH_TX\s*\(")
_RE_LIBR_RHO = re.compile(r"\bLiBrRho_TXP\s*\(")
_RE_LIBR_T_HX = re.compile(r"\bLiBrT_HX\s*\(")

# NH3–H2O family (native backend + optional aliases + optional PropsSI-like)
_RE_NH3H2O = re.compile(
    r"\b(?:"
    r"NH3H2O|NH3H2O_STATE|NH3H2O_TPX|NH3H2O_STATE_TPX|"
    r"NH3H2OPropsSI|NH3H2OPropsMulti|NH3H2OBatchProps|"
    r"prop_tpx|state_tpx|props_multi_tpx|batch_prop_tpx"
    r")\s*\("
)


def _needs_python_property_funcs(eq_strings: Sequence[str]) -> bool:
    """
    True if equations contain Python-callable property functions.

    NOTE: we do NOT include the cache helper names here because they are not numeric
    thermo-property calls and typically shouldn't appear in equations.
    """
    text = "\n".join(eq_strings)
    return bool(
        _RE_PROPS.search(text)
        or _RE_PHASE.search(text)
        or _RE_HA.search(text)
        or _RE_AS_PROPS.search(text)
        or _RE_AS_MULTI.search(text)
        or _RE_AS_BATCH.search(text)
        or _RE_FUG.search(text)
        or _RE_FUGCOEFF.search(text)
        or _RE_LNPHI.search(text)
        or _RE_CHEMPOT.search(text)
        or _RE_AS_SI.search(text)
        or _RE_AS_MULTI2.search(text)
        or _RE_AS_BATCH2.search(text)
        or _RE_LIBR.search(text)
        or _RE_NH3H2O.search(text)
        or re.search(r"\b(?:CTPropsSI|CTPropsMulti|CTBatchProps|ctprops_si|ctprops_multi|batch_ctprops)\s*\(", text)
    )


def _inject_property_functions(eq_strings: Sequence[str], params: Dict[str, Any]) -> None:
    text = "\n".join(eq_strings)

    # -------------------- CoolProp contract surface --------------------
    # We import this only if needed.
    needs_cp = bool(
        _RE_PROPS.search(text)
        or _RE_PHASE.search(text)
        or _RE_HA.search(text)
        or _RE_AS_PROPS.search(text)
        or _RE_AS_MULTI.search(text)
        or _RE_AS_BATCH.search(text)
        or _RE_FUG.search(text)
        or _RE_FUGCOEFF.search(text)
        or _RE_LNPHI.search(text)
        or _RE_CHEMPOT.search(text)
        or _RE_AS_SI.search(text)
        or _RE_AS_MULTI2.search(text)
        or _RE_AS_BATCH2.search(text)
        or _RE_CT.search(text)
    )

    _cp = None
    if needs_cp:
        try:
            from thermo_props import coolprop_backend as _cp  # local import
        except Exception:
            _cp = None

        if _cp is None:
            missing: List[str] = []
            if _RE_PROPS.search(text):
                missing.append("PropsSI")
            if _RE_PHASE.search(text):
                missing.append("PhaseSI")
            if _RE_HA.search(text):
                missing.append("HAPropsSI")
            if _RE_AS_PROPS.search(text) or _RE_AS_SI.search(text):
                missing.append("ASPropsSI")
            if _RE_AS_MULTI.search(text) or _RE_AS_MULTI2.search(text):
                missing.append("ASPropsMulti")
            if _RE_AS_BATCH.search(text) or _RE_AS_BATCH2.search(text):
                missing.append("ASBatchProps")
            if _RE_FUG.search(text):
                missing.append("FugacitySI")
            if _RE_FUGCOEFF.search(text):
                missing.append("FugacityCoeffSI")
            if _RE_LNPHI.search(text):
                missing.append("LnFugacityCoeffSI")
            if _RE_CHEMPOT.search(text):
                missing.append("ChemicalPotentialSI")
            if _RE_CT.search(text):
                missing.append("CTPropsSI")
            raise BackendUnavailableError(
                "Equation system references thermo property functions "
                f"({', '.join(missing)}), but thermo_props.coolprop_backend could not be imported."
            )

    # register CoolProp functions
    if _cp is not None:
        if _RE_PROPS.search(text):
            params.setdefault("PropsSI", _cp.props_si)
        if _RE_PHASE.search(text):
            params.setdefault("PhaseSI", _cp.phase_si)
        if _RE_HA.search(text):
            params.setdefault("HAPropsSI", _cp.haprops_si)
            params.setdefault("ha_props_si", _cp.ha_props_si)
            params.setdefault("ha_props_multi", _cp.ha_props_multi)
            params.setdefault("batch_ha_props", _cp.batch_ha_props)

        # AbstractState wrappers
        if (
            _RE_AS_PROPS.search(text)
            or _RE_AS_MULTI.search(text)
            or _RE_AS_BATCH.search(text)
            or _RE_FUG.search(text)
            or _RE_FUGCOEFF.search(text)
            or _RE_LNPHI.search(text)
            or _RE_CHEMPOT.search(text)
            or _RE_AS_SI.search(text)
            or _RE_AS_MULTI2.search(text)
            or _RE_AS_BATCH2.search(text)
        ):
            fn_avail = getattr(_cp, "abstractstate_available", None)
            if callable(fn_avail) and not bool(fn_avail()):
                raise BackendUnavailableError(
                    "Equation system references CoolProp AbstractState functions (ASPropsSI/Fugacity*), "
                    "but CoolProp AbstractState is unavailable in this environment."
                )

            fn_as = getattr(_cp, "as_props_si", None) or getattr(_cp, "ASPropsSI", None)
            if not callable(fn_as):
                raise BackendUnavailableError(
                    "Equation system references ASPropsSI/Fugacity* calls, but coolprop_backend "
                    "does not expose as_props_si (expected callable)."
                )
            params.setdefault("ASPropsSI", fn_as)
            params.setdefault("as_props_si", fn_as)

            fn_multi = getattr(_cp, "as_props_multi", None) or getattr(_cp, "ASPropsMulti", None)
            if callable(fn_multi):
                params.setdefault("ASPropsMulti", fn_multi)
                params.setdefault("as_props_multi", fn_multi)
            elif _RE_AS_MULTI.search(text) or _RE_AS_MULTI2.search(text):
                raise BackendUnavailableError(
                    "Equation system references ASPropsMulti/as_props_multi, but coolprop_backend does not provide it."
                )

            fn_batch = getattr(_cp, "batch_as_props", None) or getattr(_cp, "ASBatchProps", None)
            if callable(fn_batch):
                params.setdefault("ASBatchProps", fn_batch)
                params.setdefault("batch_as_props", fn_batch)
            elif _RE_AS_BATCH.search(text) or _RE_AS_BATCH2.search(text):
                raise BackendUnavailableError(
                    "Equation system references ASBatchProps/batch_as_props, but coolprop_backend does not provide it."
                )

            for _nm, _re_pat in (
                ("FugacitySI", _RE_FUG),
                ("FugacityCoeffSI", _RE_FUGCOEFF),
                ("LnFugacityCoeffSI", _RE_LNPHI),
                ("ChemicalPotentialSI", _RE_CHEMPOT),
            ):
                _fn = getattr(_cp, _nm, None)
                if callable(_fn):
                    params.setdefault(_nm, _fn)
                elif _re_pat.search(text):
                    raise BackendUnavailableError(
                        f"Equation system references {_nm}, but coolprop_backend does not provide it."
                    )

        # Cantera wrappers are delegated via coolprop_backend contract
        if _RE_CT.search(text):
            fn_avail = getattr(_cp, "cantera_available", None)
            if callable(fn_avail):
                params.setdefault("cantera_available", fn_avail)

            needs_ct_calls = bool(
                re.search(r"\b(?:CTPropsSI|CTPropsMulti|CTBatchProps|ctprops_si|ctprops_multi|batch_ctprops)\s*\(", text)
            )
            if needs_ct_calls and callable(fn_avail) and not bool(fn_avail()):
                raise BackendUnavailableError(
                    "Equation system references Cantera property functions (CTPropsSI/CTPropsMulti/CTBatchProps), "
                    "but Cantera is unavailable in this environment. Install with: pip install cantera"
                )

            fn_si = getattr(_cp, "ctprops_si", None) or getattr(_cp, "CTPropsSI", None)
            fn_multi = getattr(_cp, "ctprops_multi", None) or getattr(_cp, "CTPropsMulti", None)
            fn_batch = getattr(_cp, "batch_ctprops", None) or getattr(_cp, "CTBatchProps", None)

            if callable(fn_si):
                params.setdefault("CTPropsSI", fn_si)
                params.setdefault("ctprops_si", fn_si)
            elif re.search(r"\b(?:CTPropsSI|ctprops_si)\s*\(", text):
                raise BackendUnavailableError(
                    "Equation system references CTPropsSI/ctprops_si, but coolprop_backend does not provide it."
                )

            if callable(fn_multi):
                params.setdefault("CTPropsMulti", fn_multi)
                params.setdefault("ctprops_multi", fn_multi)
            elif re.search(r"\b(?:CTPropsMulti|ctprops_multi)\s*\(", text):
                raise BackendUnavailableError(
                    "Equation system references CTPropsMulti/ctprops_multi, but coolprop_backend does not provide it."
                )

            if callable(fn_batch):
                params.setdefault("CTBatchProps", fn_batch)
                params.setdefault("batch_ctprops", fn_batch)
            elif re.search(r"\b(?:CTBatchProps|batch_ctprops)\s*\(", text):
                raise BackendUnavailableError(
                    "Equation system references CTBatchProps/batch_ctprops, but coolprop_backend does not provide it."
                )

            # Optional cache helpers (safe to inject if referenced)
            fn_ci = getattr(_cp, "ctprops_cache_info", None)
            fn_cc = getattr(_cp, "clear_ctprops_caches", None)
            if callable(fn_ci):
                params.setdefault("ctprops_cache_info", fn_ci)
            if callable(fn_cc):
                params.setdefault("clear_ctprops_caches", fn_cc)


    # -------------------- LiBr–H2O engine + helpers --------------------
    if _RE_LIBR.search(text):
        _lb = _import_first(("librh2o_ashrae_backend", "librh2o_backend"))
        if _lb is None:
            raise BackendUnavailableError(
                "Equation system references LiBr–H2O property functions, but no LiBr backend module "
                "could be imported from thermo_props."
            )

        lb_si = getattr(_lb, "librh2o_props_si", None) or getattr(_lb, "LiBrPropsSI", None)
        lb_multi = getattr(_lb, "librh2o_props_multi", None) or getattr(_lb, "LiBrPropsMulti", None)
        lb_batch = getattr(_lb, "batch_librh2o_props", None) or getattr(_lb, "LiBrBatchProps", None)

        if not callable(lb_si):
            raise BackendUnavailableError("LiBr backend imported but does not expose librh2o_props_si.")

        params.setdefault("LiBrPropsSI", lb_si)
        params.setdefault("LiBrH2OPropsSI", lb_si)
        params.setdefault("librh2o_props_si", lb_si)

        if callable(lb_multi):
            params.setdefault("LiBrPropsMulti", lb_multi)
            params.setdefault("librh2o_props_multi", lb_multi)
        if callable(lb_batch):
            params.setdefault("LiBrBatchProps", lb_batch)
            params.setdefault("batch_librh2o_props", lb_batch)

        def _lb_try(out_codes: Sequence[str], pairs: Sequence[Tuple[str, Any]]) -> Any:
            norm_pairs: List[Tuple[str, Any]] = []
            for k, v in pairs:
                kk = str(k)
                vv: Any = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                norm_pairs.append((kk, vv))

            def _call(out_key: str, use_pairs: Sequence[Tuple[str, Any]]) -> Any:
                args: List[Any] = [out_key]
                for kk, vv in use_pairs:
                    args.extend([kk, vv])
                return lb_si(*args)  # type: ignore[misc]

            last_exc: Exception | None = None

            for out_key in out_codes:
                try:
                    return _call(out_key, norm_pairs)
                except Exception as e:
                    last_exc = e

                if len(norm_pairs) >= 3:
                    try:
                        return _call(out_key, norm_pairs[:2])
                    except Exception as e:
                        last_exc = e

                alt_pairs: List[Tuple[str, Any]] = []
                for kk, vv in norm_pairs:
                    if kk == "X":
                        alt_pairs.append(("x", vv))
                    elif kk == "x":
                        alt_pairs.append(("X", vv))
                    else:
                        alt_pairs.append((kk, vv))

                if alt_pairs != norm_pairs:
                    try:
                        return _call(out_key, alt_pairs)
                    except Exception as e:
                        last_exc = e

                    if len(alt_pairs) >= 3:
                        try:
                            return _call(out_key, alt_pairs[:2])
                        except Exception as e:
                            last_exc = e

            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LiBr property call failed (no attempts executed).")

        if _RE_LIBR_X_TP.search(text):
            def LiBrX_TP(T: Any, P: Any) -> Any:  # noqa: N802
                return _lb_try(out_codes=("X", "x"), pairs=(("T", T), ("P", P)))
            params.setdefault("LiBrX_TP", LiBrX_TP)

        if _RE_LIBR_H_TX.search(text):
            def LiBrH_TX(T: Any, X: Any) -> Any:  # noqa: N802
                return _lb_try(out_codes=("H", "h"), pairs=(("T", T), ("X", X)))
            params.setdefault("LiBrH_TX", LiBrH_TX)

        if _RE_LIBR_RHO.search(text):
            def LiBrRho_TXP(T: Any, X: Any, P: Any) -> Any:  # noqa: N802
                return _lb_try(out_codes=("D", "rho", "RHO"), pairs=(("T", T), ("X", X), ("P", P)))
            params.setdefault("LiBrRho_TXP", LiBrRho_TXP)

        if _RE_LIBR_T_HX.search(text):
            def LiBrT_HX(H: Any, X: Any) -> Any:  # noqa: N802
                return _lb_try(out_codes=("T",), pairs=(("H", H), ("X", X)))
            params.setdefault("LiBrT_HX", LiBrT_HX)

    # -------------------- NH3–H2O native backend + helpers --------------------
    if _RE_NH3H2O.search(text):
        ref_names = (
            "NH3H2O",
            "NH3H2O_STATE",
            "NH3H2O_TPX",
            "NH3H2O_STATE_TPX",
            "NH3H2OPropsSI",
            "NH3H2OPropsMulti",
            "NH3H2OBatchProps",
            "prop_tpx",
            "state_tpx",
            "props_multi_tpx",
            "batch_prop_tpx",
        )
        has_user_funcs = any(k in params for k in ref_names)

        _aw = None
        if not has_user_funcs:
            _aw = _import_first(("nh3h2o_backend", "nh3h2o_native_backend", "nh3h2o_ik_backend", "nh3h2o_ik93_backend"))

        if _aw is None and not has_user_funcs:
            raise BackendUnavailableError(
                "Equation system references NH3–H2O property calls (NH3H2O*/prop_tpx/state_tpx), "
                "but no NH3–H2O backend module could be imported from thermo_props."
            )

        if _aw is not None:
            fn_prop = getattr(_aw, "NH3H2O_TPX", None) or getattr(_aw, "prop_tpx", None) or getattr(_aw, "NH3H2O", None)
            fn_state = getattr(_aw, "NH3H2O_STATE_TPX", None) or getattr(_aw, "state_tpx", None) or getattr(_aw, "NH3H2O_STATE", None)
            fn_multi = getattr(_aw, "props_multi_tpx", None) or getattr(_aw, "NH3H2OPropsMulti", None)
            fn_batch = getattr(_aw, "batch_prop_tpx", None) or getattr(_aw, "NH3H2OBatchProps", None)
            fn_propssi = getattr(_aw, "NH3H2OPropsSI", None)

            if callable(fn_prop):
                params.setdefault("NH3H2O_TPX", fn_prop)
                params.setdefault("prop_tpx", fn_prop)
                params.setdefault("NH3H2O", fn_prop)
            if callable(fn_state):
                params.setdefault("NH3H2O_STATE_TPX", fn_state)
                params.setdefault("state_tpx", fn_state)
                params.setdefault("NH3H2O_STATE", fn_state)
            if callable(fn_multi):
                params.setdefault("props_multi_tpx", fn_multi)
                params.setdefault("NH3H2OPropsMulti", fn_multi)
            if callable(fn_batch):
                params.setdefault("batch_prop_tpx", fn_batch)
                params.setdefault("NH3H2OBatchProps", fn_batch)
            if callable(fn_propssi):
                params.setdefault("NH3H2OPropsSI", fn_propssi)

            missing: List[str] = []
            if re.search(r"\bNH3H2O_TPX\s*\(", text) and not callable(params.get("NH3H2O_TPX")):
                missing.append("NH3H2O_TPX")
            if re.search(r"\bNH3H2O_STATE_TPX\s*\(", text) and not callable(params.get("NH3H2O_STATE_TPX")):
                missing.append("NH3H2O_STATE_TPX")
            if re.search(r"\bprop_tpx\s*\(", text) and not callable(params.get("prop_tpx")):
                missing.append("prop_tpx")
            if re.search(r"\bstate_tpx\s*\(", text) and not callable(params.get("state_tpx")):
                missing.append("state_tpx")
            if re.search(r"\bNH3H2OPropsSI\s*\(", text) and not callable(params.get("NH3H2OPropsSI")):
                missing.append("NH3H2OPropsSI")
            if missing:
                raise BackendUnavailableError(
                    "NH3–H2O backend imported but missing required callables referenced by equations: "
                    + ", ".join(missing)
                )


# ------------------------------ public API ------------------------------



def _is_optimize_mapping(m: Mapping[str, Any]) -> bool:
    """Return True if the incoming mapping represents an optimization problem."""
    try:
        pt = str(m.get("problem_type", "") or "").strip().lower()
    except Exception:
        pt = ""
    if pt == "optimize":
        return True
    # Back-compat / permissive detection
    if "objective" in m or "constraints" in m:
        return True
    return False


def _mapping_params_to_values(m: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build the evaluation-parameter dict from an interpreter-produced mapping.

    The interpreter/build_spec layer typically emits numeric `constants`.
    We also accept `params` if the caller provided them directly.
    """
    params: Dict[str, Any] = {}

    raw_params = m.get("params", None)
    if isinstance(raw_params, Mapping):
        params.update(dict(raw_params))

    raw_constants = m.get("constants", None)
    if isinstance(raw_constants, Mapping):
        params.update(dict(raw_constants))

    return params


def _mapping_vars_to_varlikes(var_items: Sequence[Any]) -> List[_VarLike]:
    """Convert variable entries from interpreter specs into the internal _VarLike shape."""
    out: List[_VarLike] = []

    # If the mapping already contains VarSpec objects, reuse the existing converter.
    if var_items and all(isinstance(v, VarSpec) for v in var_items):  # type: ignore[arg-type]
        return _vars_to_varlikes(var_items)  # type: ignore[arg-type]

    def _as_float(x: Any) -> float | None:
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    for v in var_items:
        if isinstance(v, Mapping):
            name = str(v.get("name", v.get("var", v.get("id", ""))) or "").strip()
            if not name:
                continue
            kind = str(v.get("kind", "") or ("unknown" if v.get("unknown", False) else "unknown")).strip() or "unknown"
            value = _as_float(v.get("value", None))
            guess = _as_float(v.get("guess", v.get("init", v.get("x0", None))))
            lower = _as_float(v.get("lower", v.get("lb", None)))
            upper = _as_float(v.get("upper", v.get("ub", None)))
            out.append(_VarLike(name=name, kind=kind, value=value, guess=guess, lower=lower, upper=upper))
        elif isinstance(v, VarSpec):
            # Single VarSpec among dicts/strings: handle it.
            out.extend(_vars_to_varlikes([v]))
        elif isinstance(v, str) and v.strip():
            out.append(_VarLike(name=v.strip(), kind="unknown", value=None))
        else:
            # Ignore unknown shapes
            continue

    return out


def solve_optimize(
    problem: Mapping[str, Any],
    *,
    backend: Optional[BackendKind] = None,
    method: Optional[str] = None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    max_restarts: Optional[int] = None,
    use_units: Optional[bool] = None,
) -> Solution:
    """
    Solve an optimization problem produced by interpreter/build_spec.py.

    Expected mapping keys (build_spec output):
      - problem_type: "optimize"
      - objective: "<expr>"
      - sense: "min"|"max"
      - constraints: ["<eq/residual>", ...]   (residual==0 constraints)
      - variables: [...], constants: {...}, solve: {...}

    Backend notes:
      - Current optimizer implementation is SciPy-based (scipy.optimize.minimize).
      - GEKKO optimization is *not* wired here (yet).
    """
    if not isinstance(problem, Mapping):
        raise TypeError("solve_optimize expects a Mapping[str, Any]")

    objective = str(problem.get("objective", "") or "").strip()
    if not objective:
        raise ValueError("Optimization spec is missing 'objective'.")

    sense = str(problem.get("sense", problem.get("objective_sense", "min")) or "min").strip().lower()
    if sense not in {"min", "max"}:
        raise ValueError(f"Invalid optimization 'sense': {sense!r} (expected 'min' or 'max').")

    constraints = problem.get("constraints", None)
    if constraints is None:
        constraints = problem.get("equations", [])
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise TypeError("Optimization spec 'constraints' must be a list of strings.")
    constraint_strings = [str(c) for c in constraints if str(c).strip()]

    solve_block_raw = problem.get("solve", None)
    solve_block_in: Dict[str, Any] = dict(solve_block_raw) if isinstance(solve_block_raw, Mapping) else {}

    spec_backend_pref = _normalize_backend_name(
        solve_block_in.get("backend", solve_block_in.get("solver", None))
        or problem.get("backend", None)
        or problem.get("solver", None)
        or "auto"
    )
    spec_method_pref = str(solve_block_in.get("method", None) or problem.get("method", None) or "SLSQP")
    spec_tol_pref = float(solve_block_in.get("tol", None) or problem.get("tol", None) or 1e-6)
    spec_max_iter_pref = int(
        solve_block_in.get("max_iter", solve_block_in.get("maxiter", None))
        or problem.get("max_iter", None)
        or problem.get("maxiter", None)
        or 200
    )
    spec_max_restarts_pref = int(
        solve_block_in.get("max_restarts", solve_block_in.get("restarts", None))
        or problem.get("max_restarts", None)
        or 0
    )
    spec_use_units_pref = bool(
        solve_block_in.get("use_units", None)
        if "use_units" in solve_block_in
        else bool(problem.get("use_units", False))
    )

    eff_backend = _normalize_backend_name(backend) if backend is not None else spec_backend_pref
    eff_method = str(method).strip() if method is not None else spec_method_pref
    eff_tol = float(tol) if tol is not None else spec_tol_pref
    eff_max_iter = int(max_iter) if max_iter is not None else spec_max_iter_pref
    eff_max_restarts = int(max_restarts) if max_restarts is not None else spec_max_restarts_pref
    eff_use_units = bool(use_units) if use_units is not None else bool(spec_use_units_pref)

    # For now, optimization is SciPy-based.
    if eff_backend in {"gekko"}:
        raise BackendUnavailableError(
            "Optimization requested with backend 'gekko', but the optimization API is currently SciPy-based. "
            "Use backend='scipy' (or omit backend for auto)."
        )
    if eff_backend in {"", "none", "auto"}:
        eff_backend = "scipy"

    # Build evaluation params + inject thermo/property callables as needed.
    params = _mapping_params_to_values(problem)

    all_exprs = list(constraint_strings) + [objective]
    _inject_property_functions(all_exprs, params)

    # Optimization always requires SciPy at present.
    if not _has_scipy():
        raise BackendUnavailableError("Optimization requires SciPy, but SciPy is not installed. Install with: pip install scipy")

    # Additional check: if the user used Python-callable thermo functions and explicitly asked for GEKKO, we already rejected.
    _ensure_backend_available(eff_backend)

    # Forward any remaining solve-block keys to the optimizer implementation.
    solve_block_forward = dict(solve_block_in)
    _strip_solve_keys(
        solve_block_forward,
        [
            "backend", "solver",
            "method",
            "tol",
            "max_iter", "maxiter",
            "max_restarts", "restarts",
            "use_units",
        ],
    )

    var_items_raw = problem.get("variables", problem.get("vars", []))

    # Optimization variables may arrive in two common shapes:
    #  (A) interpreter/build_spec shape: a list of variable dicts
    #      e.g. [{"name":"x","guess":0.2,"lower":0,"upper":1}, ...]
    #  (B) equation-system mapping shape: a mapping name->payload
    #      e.g. {"x":{"guess":0.2,"lower":0,"upper":1}, "y":{"guess":0.8}}
    # Support both for robustness.
    if isinstance(var_items_raw, Mapping):
        var_items_list: List[Any] = []
        for k, v in var_items_raw.items():
            nm = str(k).strip()
            if not nm:
                continue
            if isinstance(v, Mapping):
                d: Dict[str, Any] = {"name": nm}
                d.update(dict(v))
                var_items_list.append(d)
            else:
                var_items_list.append({"name": nm, "guess": v})
        var_items: Any = var_items_list
    else:
        var_items = var_items_raw

    if not isinstance(var_items, Sequence) or isinstance(var_items, (str, bytes)):
        raise TypeError("Optimization spec 'variables' must be a list or mapping.")
    varlikes = _mapping_vars_to_varlikes(list(var_items))

    # Lazy import: optimizer module may not exist until you add it (next step).
    try:
        from .optimizer import solve_optimize as _solve_optimize_impl  # type: ignore
    except Exception as e:
        raise BackendUnavailableError(
            "Optimization spec detected, but equations.optimizer is not available yet. "
            "Create equations/optimizer.py with a solve_optimize(...) entry point."
        ) from e

    # Duck-typed spec passed downstream (keeps API stable while optimizer evolves)
    opt_like = {
        "objective": str(objective),
        "sense": str(sense),
        "constraints": list(constraint_strings),
        "variables": varlikes,
        "params": params,
        "solve": solve_block_forward if solve_block_forward else {},
        "backend": str(eff_backend),
        "method": str(eff_method),
        "tol": float(eff_tol),
        "max_iter": int(eff_max_iter),
        "max_restarts": int(eff_max_restarts),
        "use_units": bool(eff_use_units),
        "title": str(problem.get("title", "") or ""),
    }

    return _solve_optimize_impl(
        opt_like,
        backend=str(eff_backend),
        method=str(eff_method),
        tol=float(eff_tol),
        max_iter=int(eff_max_iter),
    )


def solve(
    system: Union[EquationSystemSpec, Mapping[str, Any]],
    *,
    backend: Optional[BackendKind] = None,
    method: Optional[str] = None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    max_restarts: Optional[int] = None,
    use_units: Optional[bool] = None,
) -> Solution:
    """Convenience alias that routes equation systems and optimization problems."""
    return solve_system(
        system,
        backend=backend,
        method=method,
        tol=tol,
        max_iter=max_iter,
        max_restarts=max_restarts,
        use_units=use_units,
    )


def solve_system(
    system: Union[EquationSystemSpec, Mapping[str, Any]],
    *,
    backend: Optional[BackendKind] = None,
    method: Optional[str] = None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    max_restarts: Optional[int] = None,
    use_units: Optional[bool] = None,
) -> Solution:
    """
    Solve an EES-ish nonlinear equation system.

    Accepts:
      - EquationSystemSpec, OR
      - a JSON/YAML mapping compatible with spec.system_from_mapping()

    Override semantics:
      - If an API kwarg is None, we fall back to spec.solve / spec fields / defaults.
      - If provided, the API kwarg overrides spec/solve-block.
    """
    if isinstance(system, Mapping):
        # Optimization routing (build_spec emits problem_type='optimize' + objective/constraints)
        if _is_optimize_mapping(system):
            return solve_optimize(
                system,
                backend=backend,
                method=method,
                tol=tol,
                max_iter=max_iter,
                max_restarts=max_restarts,
                use_units=use_units,
            )
        spec = system_from_mapping(system)
    else:
        spec = system

    spec.check_square()

    solve_block_raw = getattr(spec, "solve", None)
    solve_block_in: Dict[str, Any] = dict(solve_block_raw) if isinstance(solve_block_raw, Mapping) else {}

    spec_backend_pref = _normalize_backend_name(
        solve_block_in.get("backend", solve_block_in.get("solver", None))
        or getattr(spec, "backend", None)
        or getattr(spec, "solver", None)
        or "auto"
    )
    spec_method_pref = str(solve_block_in.get("method", None) or getattr(spec, "method", None) or "hybr")
    spec_tol_pref = float(solve_block_in.get("tol", None) or getattr(spec, "tol", None) or 1e-9)
    spec_max_iter_pref = int(
        solve_block_in.get("max_iter", solve_block_in.get("maxiter", None))
        or getattr(spec, "max_iter", None)
        or getattr(spec, "maxiter", None)
        or 200
    )
    spec_max_restarts_pref = int(
        solve_block_in.get("max_restarts", solve_block_in.get("restarts", None))
        or getattr(spec, "max_restarts", None)
        or 2
    )
    spec_use_units_pref = bool(
        solve_block_in.get("use_units", None)
        if "use_units" in solve_block_in
        else getattr(spec, "use_units", False)
    )

    eff_backend = _normalize_backend_name(backend) if backend is not None else spec_backend_pref
    eff_method = str(method).strip() if method is not None else spec_method_pref
    eff_tol = float(tol) if tol is not None else spec_tol_pref
    eff_max_iter = int(max_iter) if max_iter is not None else spec_max_iter_pref
    eff_max_restarts = int(max_restarts) if max_restarts is not None else spec_max_restarts_pref
    eff_use_units = bool(use_units) if use_units is not None else bool(spec_use_units_pref)

    units_adapter = _try_units_adapter() if eff_use_units else None
    var_base_units, param_base_units = _extract_base_units(spec) if units_adapter is not None else (None, None)

    eq_strings = _equations_to_strings(spec.equations)
    varlikes = _vars_to_varlikes(spec.vars, units=units_adapter, base_units=var_base_units)
    params = _params_to_values(spec.params, units=units_adapter, base_units=param_base_units)

    _inject_property_functions(eq_strings, params)

    needs_props = _needs_python_property_funcs(eq_strings)
    if needs_props:
        if eff_backend == "gekko":
            raise BackendUnavailableError(
                "This equation system uses property-function calls (PropsSI/PhaseSI/HAPropsSI/CTPropsSI/ASPropsSI/LiBr*/NH3H2O*), "
                "which require the SciPy backend. GEKKO cannot evaluate these Python-callable thermo functions."
            )
        if eff_backend == "auto":
            eff_backend = "scipy"
        if not _has_scipy():
            raise BackendUnavailableError(
                "This equation system uses property-function calls (PropsSI/PhaseSI/HAPropsSI/CTPropsSI/ASPropsSI/LiBr*/NH3H2O*), "
                "but SciPy is not installed. Install with: pip install scipy"
            )

    _ensure_backend_available(eff_backend)

    solve_block_forward = dict(solve_block_in)
    _strip_solve_keys(
        solve_block_forward,
        [
            "backend", "solver",
            "method",
            "tol",
            "max_iter", "maxiter",
            "max_restarts", "restarts",
            "use_units",
        ],
    )

    spec_like = _SpecLike(
        equations=eq_strings,
        variables=varlikes,
        params=params,
        solve=solve_block_forward if solve_block_forward else {},
        backend=eff_backend,
        solver=eff_backend,
        method=eff_method,
        tol=eff_tol,
        max_iter=eff_max_iter,
        maxiter=eff_max_iter,
        max_restarts=eff_max_restarts,
    )

    from .solver import solve_system as _solve_system_impl  # local import

    return _solve_system_impl(
        spec_like,
        backend=str(eff_backend),
        method=str(eff_method),
        tol=float(eff_tol),
        max_iter=int(eff_max_iter),
        max_restarts=int(eff_max_restarts),
    )


__all__ = [
    "BackendUnavailableError",
    "BackendKind",
    "EquationSystem",
    "Param",
    "Solution",
    "Var",
    "solve",
    "solve_optimize",
    "solve_system",
]
