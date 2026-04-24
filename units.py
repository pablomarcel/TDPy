from __future__ import annotations

"""
units

Tiny, dependency-light unit parsing + conversion.

This module is intentionally minimal and stable:

Public API
----------
- UnitError
- Quantity(value, unit)  # lightweight container
- parse_quantity(text, default_unit=None, to_unit=None) -> Quantity
- convert(value, from_unit, to_unit) -> float

Design goals
------------
- No heavy dependencies (no Pint required).
- Robust parsing of strings like:
    "300 K", "300K", "300[K]"
    "-10 C", "14.7 psi", "101.3 kPa"
    "70 kJ/kg", "7.5e6 Pa", "2.1e2[J/kg-K]"
- Conversions supported for the core units used by tdpy:
  temperature, pressure, length, mass, time, energy, power,
  specific energy (J/kg), specific entropy (J/kg-K).
"""

import re
from dataclasses import dataclass
from typing import Dict, Tuple


# ------------------------------ errors ------------------------------


class UnitError(ValueError):
    """Raised for unit parsing/conversion errors."""


# ------------------------------ data types ------------------------------


@dataclass(frozen=True)
class Quantity:
    """
    A numeric value with an associated unit string.

    Notes
    -----
    - The canonical attribute is `unit` for backward compatibility.
    - `units` is provided as an alias (some callers prefer that spelling).
    """

    value: float
    unit: str = ""

    @property
    def units(self) -> str:  # alias
        return self.unit

    def to(self, unit: str) -> "Quantity":
        if not unit:
            # Explicit conversion to "no unit" is nonsensical; keep behavior strict.
            raise UnitError("Cannot convert to an empty unit.")
        if not self.unit:
            raise UnitError(
                "Cannot convert a quantity with no unit. "
                "Provide default_unit when parsing, or create Quantity(value, unit)."
            )
        return Quantity(convert(self.value, self.unit, unit), _norm_unit(unit))


# ------------------------------ unit tables ------------------------------
# Each entry: unit -> (scale, offset) such that:
#   value_SI = value * scale + offset
#
# Offsets are only used for temperature scales.
#
# SI bases used:
#   pressure: Pa
#   temperature: K
#   length: m
#   mass: kg
#   time: s
#   energy: J
#   power: W
#   spec_energy: J/kg
#   spec_entropy: J/kg-K


_TEMP_TO_K: Dict[str, Tuple[float, float]] = {
    "k": (1.0, 0.0),
    "kelvin": (1.0, 0.0),
    "c": (1.0, 273.15),
    "degc": (1.0, 273.15),
    "celsius": (1.0, 273.15),
    "f": (5.0 / 9.0, 255.3722222222222),  # (F-32)*5/9 + 273.15
    "degf": (5.0 / 9.0, 255.3722222222222),
    "fahrenheit": (5.0 / 9.0, 255.3722222222222),
    "r": (5.0 / 9.0, 0.0),  # Rankine to K
    "degr": (5.0 / 9.0, 0.0),
    "rankine": (5.0 / 9.0, 0.0),
}

_PRESS_TO_PA: Dict[str, float] = {
    "pa": 1.0,
    "pascal": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "gpa": 1e9,
    "bar": 1e5,
    "mbar": 1e2,
    "atm": 101325.0,
    "torr": 133.32236842105263,
    "mmhg": 133.32236842105263,
    "psi": 6894.757293168361,
    "psia": 6894.757293168361,
    "psig": 6894.757293168361,  # treated as psi magnitude (no gauge offset model here)
}

_LENGTH_TO_M: Dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "km": 1e3,
    "in": 0.0254,
    "inch": 0.0254,
    "ft": 0.3048,
    "feet": 0.3048,
}

_MASS_TO_KG: Dict[str, float] = {
    "kg": 1.0,
    "g": 1e-3,
    "lbm": 0.45359237,
    "lb": 0.45359237,
}

_TIME_TO_S: Dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "min": 60.0,
    "hr": 3600.0,
    "h": 3600.0,
}

_ENERGY_TO_J: Dict[str, float] = {
    "j": 1.0,
    "kj": 1e3,
    "mj": 1e6,
    "btu": 1055.05585262,
}

_POWER_TO_W: Dict[str, float] = {
    "w": 1.0,
    "kw": 1e3,
    "mw": 1e6,
    "hp": 745.6998715822702,
}

# Composite helpers (common thermo ones)
_SPEC_ENERGY: Dict[str, float] = {
    "j/kg": 1.0,
    "j/kgm": 1.0,  # tolerant typo-ish forms (rare; kept lenient)
    "kj/kg": 1e3,
    "mj/kg": 1e6,
    "btu/lbm": _ENERGY_TO_J["btu"] / _MASS_TO_KG["lbm"],
}

# J/kg-K, kJ/kg-K, Btu/lbm-R
_SPEC_ENTROPY: Dict[str, float] = {
    "j/kg-k": 1.0,
    "j/kg/k": 1.0,
    "j/(kg*k)": 1.0,
    "kj/kg-k": 1e3,
    "kj/kg/k": 1e3,
    "kj/(kg*k)": 1e3,
    "btu/lbm-r": _ENERGY_TO_J["btu"] / (_MASS_TO_KG["lbm"] * (5.0 / 9.0)),
    "btu/lbm/degr": _ENERGY_TO_J["btu"] / (_MASS_TO_KG["lbm"] * (5.0 / 9.0)),
}

# Generic map: unit -> (type, scale, offset)
_UNIT_DB: Dict[str, Tuple[str, float, float]] = {}


def _reg(table: Dict[str, float], utype: str) -> None:
    for k, s in table.items():
        _UNIT_DB[k] = (utype, float(s), 0.0)


_reg(_PRESS_TO_PA, "pressure")
_reg(_LENGTH_TO_M, "length")
_reg(_MASS_TO_KG, "mass")
_reg(_TIME_TO_S, "time")
_reg(_ENERGY_TO_J, "energy")
_reg(_POWER_TO_W, "power")
_reg(_SPEC_ENERGY, "spec_energy")
_reg(_SPEC_ENTROPY, "spec_entropy")
for k, (scale, offset) in _TEMP_TO_K.items():
    _UNIT_DB[k] = ("temperature", float(scale), float(offset))


# ------------------------------ normalization ------------------------------


def _clean_unit_expr(u: str) -> str:
    """
    Normalize a unit expression conservatively.

    - lowercases
    - removes spaces and degree symbol
    - strips surrounding brackets/parentheses
    - keeps separators '/', '-', '*' because we use them for composite lookups
    """
    s = str(u).strip().lower()
    if not s:
        return ""
    s = s.replace("°", "")
    # remove surrounding wrappers
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip().lower()
    s = s.replace(" ", "")

    # common normalizations for entropy units
    # e.g. "j/kgk" -> "j/kg-k"
    if s.endswith("kgk") and (s.startswith("j/") or s.startswith("kj/") or s.startswith("mj/")):
        # "j/kgk" or "kj/kgk"
        s = s.replace("kgk", "kg-k")
    if s.endswith("kg-k") and (s.startswith("j/") or s.startswith("kj/") or s.startswith("mj/")):
        # already good
        pass

    # tolerate "lbmr" or "lbm-r" style entropy
    s = s.replace("lbmr", "lbm-r")

    return s


def _norm_unit(u: str | None) -> str:
    if u is None:
        return ""
    return _clean_unit_expr(u)


# ------------------------------ conversion ------------------------------


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a scalar between compatible units.

    Supported types:
    - temperature, pressure, length, mass, time, energy, power, spec_energy, spec_entropy
    """
    fu = _norm_unit(from_unit)
    tu = _norm_unit(to_unit)

    if not fu and not tu:
        return float(value)
    if fu == tu:
        return float(value)

    if fu not in _UNIT_DB:
        raise UnitError(f"Unknown unit: {from_unit!r}")
    if tu not in _UNIT_DB:
        raise UnitError(f"Unknown unit: {to_unit!r}")

    ftype, fscale, foffset = _UNIT_DB[fu]
    ttype, tscale, toffset = _UNIT_DB[tu]
    if ftype != ttype:
        raise UnitError(f"Incompatible units: {from_unit!r} ({ftype}) -> {to_unit!r} ({ttype})")

    if ftype == "temperature":
        # K = value * scale + offset
        si = float(value) * fscale + foffset
        return (si - toffset) / tscale

    si = float(value) * fscale
    return si / tscale


# ------------------------------ parsing ------------------------------

_Q_RE = re.compile(
    r"""^\s*
    (?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*
    (?:
        (?:\[\s*(?P<u1>[^\]]+)\s*\]) |
        (?P<u2>[^#;!]+)
    )?
    \s*$""",
    re.VERBOSE,
)


def parse_quantity(
    text: str, *, default_unit: str | None = None, to_unit: str | None = None
) -> Quantity:
    """
    Parse a numeric string with optional unit.

    Examples:
      - "101.3 kPa"
      - "5bar"
      - "300[K]"
      - "-10 C"
      - "70 kJ/kg"
      - "1.2e5 Pa"

    Behavior:
      - If no unit is provided:
          * if default_unit is given: uses it
          * else: returns unit="" (no unit)
      - If to_unit is provided: converts to that unit (requires a known from-unit).
    """
    s = str(text).strip()
    m = _Q_RE.match(s)
    if not m:
        raise UnitError(f"Could not parse quantity: {text!r}")

    val = float(m.group("val"))

    unit_raw = (m.group("u1") or m.group("u2") or "").strip()
    # strip inline comments if user passed them inside unit capture
    if unit_raw:
        unit_raw = unit_raw.split("#", 1)[0].split(";", 1)[0].split("!", 1)[0].strip()

    unit = _norm_unit(unit_raw) if unit_raw else ""

    if not unit:
        unit = _norm_unit(default_unit) if default_unit else ""
        q = Quantity(val, unit)
        if to_unit:
            if not q.unit:
                raise UnitError(
                    f"Cannot convert {text!r} to {to_unit!r} because no unit was provided "
                    "and no default_unit was specified."
                )
            return q.to(to_unit)
        return q

    q = Quantity(val, unit)
    return q.to(to_unit) if to_unit else q
