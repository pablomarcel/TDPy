# units/__init__.py
from __future__ import annotations

"""
units

Lightweight units and conversion helpers.

Design goals:
- Simple, dependency-free
- Explicit coverage of common thermo / fluids engineering units
- Friendly parsing for CLI/JSON usage
- Stable convenience API for the rest of TDPy

Public API
----------
- UnitError
- UnitDef
- UnitRegistry
- Quantity
- DEFAULT_REGISTRY
- default_registry()
- parse_quantity(text, default_unit=None, to_unit=None) -> Quantity
- convert(value, from_unit, to_unit) -> float
- convert_value(value, from_unit, to_unit, registry) -> float
"""

import re
from typing import Optional

from .core import Quantity, UnitDef, UnitError, UnitRegistry


# ------------------------------ normalization helpers ------------------------------

def _clean_unit_expr(u: str) -> str:
    """
    Normalize a unit expression conservatively.

    - lowercases
    - removes spaces and degree symbol
    - strips surrounding brackets/parentheses
    - keeps separators '/', '-', '*' because we use them for curated composite lookups
    """
    s = str(u).strip().lower()
    if not s:
        return ""
    s = s.replace("°", "")
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip().lower()
    s = s.replace(" ", "")

    # common entropy cleanup
    if s.endswith("kgk") and (s.startswith("j/") or s.startswith("kj/") or s.startswith("mj/")):
        s = s.replace("kgk", "kg-k")

    s = s.replace("lbmr", "lbm-r")
    return s


def _norm_unit(u: str | None) -> str:
    if u is None:
        return ""
    return _clean_unit_expr(u)


# ------------------------------ default registry ------------------------------

def default_registry() -> UnitRegistry:
    """
    Create the default registry covering common thermo / fluids units.

    Base units by dimension:
      temperature      -> K
      pressure         -> Pa
      length           -> m
      area             -> m2
      volume           -> m3
      mass             -> kg
      time             -> s
      velocity         -> m/s
      energy           -> J
      power            -> W
      spec_energy      -> J/kg
      spec_entropy     -> J/(kg*K)
      density          -> kg/m3
      mass_flow        -> kg/s
      volumetric_flow  -> m3/s
    """
    r = UnitRegistry()

    # --- temperature (base: K)
    r.add("k", "temperature", 1.0, 0.0, canonical="K")
    r.add("kelvin", "temperature", 1.0, 0.0, canonical="K")
    r.add("c", "temperature", 1.0, 273.15, canonical="C")
    r.add("degc", "temperature", 1.0, 273.15, canonical="C")
    r.add("celsius", "temperature", 1.0, 273.15, canonical="C")
    r.add("f", "temperature", 5.0 / 9.0, 255.3722222222222, canonical="F")
    r.add("degf", "temperature", 5.0 / 9.0, 255.3722222222222, canonical="F")
    r.add("fahrenheit", "temperature", 5.0 / 9.0, 255.3722222222222, canonical="F")
    r.add("r", "temperature", 5.0 / 9.0, 0.0, canonical="R")
    r.add("degr", "temperature", 5.0 / 9.0, 0.0, canonical="R")
    r.add("rankine", "temperature", 5.0 / 9.0, 0.0, canonical="R")

    # --- pressure (base: Pa)
    r.add("pa", "pressure", 1.0, 0.0, canonical="Pa")
    r.add("pascal", "pressure", 1.0, 0.0, canonical="Pa")
    r.add("kpa", "pressure", 1e3, 0.0, canonical="kPa")
    r.add("mpa", "pressure", 1e6, 0.0, canonical="MPa")
    r.add("gpa", "pressure", 1e9, 0.0, canonical="GPa")
    r.add("bar", "pressure", 1e5, 0.0, canonical="bar")
    r.add("mbar", "pressure", 1e2, 0.0, canonical="mbar")
    r.add("atm", "pressure", 101325.0, 0.0, canonical="atm")
    r.add("torr", "pressure", 133.32236842105263, 0.0, canonical="torr")
    r.add("mmhg", "pressure", 133.32236842105263, 0.0, canonical="mmHg")
    r.add("psi", "pressure", 6894.757293168361, 0.0, canonical="psi")
    r.add("psia", "pressure", 6894.757293168361, 0.0, canonical="psia")
    r.add("psig", "pressure", 6894.757293168361, 0.0, canonical="psig")

    # --- length (base: m)
    r.add("m", "length", 1.0, 0.0, canonical="m")
    r.add("meter", "length", 1.0, 0.0, canonical="m")
    r.add("metre", "length", 1.0, 0.0, canonical="m")
    r.add("mm", "length", 1e-3, 0.0, canonical="mm")
    r.add("cm", "length", 1e-2, 0.0, canonical="cm")
    r.add("km", "length", 1e3, 0.0, canonical="km")
    r.add("in", "length", 0.0254, 0.0, canonical="in")
    r.add("inch", "length", 0.0254, 0.0, canonical="in")
    r.add("ft", "length", 0.3048, 0.0, canonical="ft")
    r.add("feet", "length", 0.3048, 0.0, canonical="ft")

    # --- area (base: m2)
    r.add("m2", "area", 1.0, 0.0, canonical="m2")
    r.add("cm2", "area", 1e-4, 0.0, canonical="cm2")
    r.add("mm2", "area", 1e-6, 0.0, canonical="mm2")
    r.add("in2", "area", 0.00064516, 0.0, canonical="in2")
    r.add("ft2", "area", 0.09290304, 0.0, canonical="ft2")

    # --- volume (base: m3)
    r.add("m3", "volume", 1.0, 0.0, canonical="m3")
    r.add("l", "volume", 1e-3, 0.0, canonical="L")
    r.add("cm3", "volume", 1e-6, 0.0, canonical="cm3")
    r.add("mm3", "volume", 1e-9, 0.0, canonical="mm3")
    r.add("in3", "volume", 1.6387064e-5, 0.0, canonical="in3")
    r.add("ft3", "volume", 0.028316846592, 0.0, canonical="ft3")

    # --- mass (base: kg)
    r.add("kg", "mass", 1.0, 0.0, canonical="kg")
    r.add("g", "mass", 1e-3, 0.0, canonical="g")
    r.add("lbm", "mass", 0.45359237, 0.0, canonical="lbm")
    r.add("lb", "mass", 0.45359237, 0.0, canonical="lbm")

    # --- time (base: s)
    r.add("s", "time", 1.0, 0.0, canonical="s")
    r.add("sec", "time", 1.0, 0.0, canonical="s")
    r.add("min", "time", 60.0, 0.0, canonical="min")
    r.add("hr", "time", 3600.0, 0.0, canonical="hr")
    r.add("h", "time", 3600.0, 0.0, canonical="hr")

    # --- velocity (base: m/s)
    r.add("m/s", "velocity", 1.0, 0.0, canonical="m/s")
    r.add("ft/s", "velocity", 0.3048, 0.0, canonical="ft/s")

    # --- energy (base: J)
    r.add("j", "energy", 1.0, 0.0, canonical="J")
    r.add("kj", "energy", 1e3, 0.0, canonical="kJ")
    r.add("mj", "energy", 1e6, 0.0, canonical="MJ")
    r.add("btu", "energy", 1055.05585262, 0.0, canonical="Btu")

    # --- power (base: W)
    r.add("w", "power", 1.0, 0.0, canonical="W")
    r.add("kw", "power", 1e3, 0.0, canonical="kW")
    r.add("mw", "power", 1e6, 0.0, canonical="MW")
    r.add("hp", "power", 745.6998715822702, 0.0, canonical="hp")

    # --- specific energy (base: J/kg)
    r.add("j/kg", "spec_energy", 1.0, 0.0, canonical="J/kg")
    r.add("j/kgm", "spec_energy", 1.0, 0.0, canonical="J/kg")
    r.add("kj/kg", "spec_energy", 1e3, 0.0, canonical="kJ/kg")
    r.add("mj/kg", "spec_energy", 1e6, 0.0, canonical="MJ/kg")
    r.add("btu/lbm", "spec_energy", 1055.05585262 / 0.45359237, 0.0, canonical="Btu/lbm")

    # --- specific entropy (base: J/kg-K)
    r.add("j/kg-k", "spec_entropy", 1.0, 0.0, canonical="J/(kg*K)")
    r.add("j/kg/k", "spec_entropy", 1.0, 0.0, canonical="J/(kg*K)")
    r.add("j/(kg*k)", "spec_entropy", 1.0, 0.0, canonical="J/(kg*K)")
    r.add("kj/kg-k", "spec_entropy", 1e3, 0.0, canonical="kJ/(kg*K)")
    r.add("kj/kg/k", "spec_entropy", 1e3, 0.0, canonical="kJ/(kg*K)")
    r.add("kj/(kg*k)", "spec_entropy", 1e3, 0.0, canonical="kJ/(kg*K)")
    r.add("btu/lbm-r", "spec_entropy", 1055.05585262 / (0.45359237 * (5.0 / 9.0)), 0.0, canonical="Btu/(lbm*R)")
    r.add("btu/lbm/degr", "spec_entropy", 1055.05585262 / (0.45359237 * (5.0 / 9.0)), 0.0, canonical="Btu/(lbm*R)")

    # --- density (base: kg/m3)
    r.add("kg/m3", "density", 1.0, 0.0, canonical="kg/m3")
    r.add("g/cm3", "density", 1000.0, 0.0, canonical="g/cm3")

    # --- mass flow (base: kg/s)
    r.add("kg/s", "mass_flow", 1.0, 0.0, canonical="kg/s")
    r.add("kg/min", "mass_flow", 1.0 / 60.0, 0.0, canonical="kg/min")
    r.add("lbm/s", "mass_flow", 0.45359237, 0.0, canonical="lbm/s")
    r.add("lbm/min", "mass_flow", 0.45359237 / 60.0, 0.0, canonical="lbm/min")

    # --- volumetric flow (base: m3/s)
    r.add("m3/s", "volumetric_flow", 1.0, 0.0, canonical="m3/s")
    r.add("l/s", "volumetric_flow", 1e-3, 0.0, canonical="L/s")
    r.add("l/min", "volumetric_flow", 1e-3 / 60.0, 0.0, canonical="L/min")
    r.add("cfm", "volumetric_flow", 0.028316846592 / 60.0, 0.0, canonical="cfm")

    return r


DEFAULT_REGISTRY = default_registry()


# ------------------------------ conversion convenience ------------------------------

def convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a scalar using DEFAULT_REGISTRY.

    This is the convenience function most of the rest of TDPy expects.
    """
    fu = _norm_unit(from_unit)
    tu = _norm_unit(to_unit)
    if not fu and not tu:
        return float(value)
    if fu == tu:
        return float(value)
    return DEFAULT_REGISTRY.convert(float(value), fu, tu)


def convert_value(value: float, from_unit: str, to_unit: str, registry: UnitRegistry) -> float:
    """
    Convenience wrapper for an explicit registry.
    """
    return registry.convert(float(value), _norm_unit(from_unit), _norm_unit(to_unit))


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
    text: str,
    *,
    default_unit: str | None = None,
    to_unit: str | None = None,
    registry: UnitRegistry = DEFAULT_REGISTRY,
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
    if unit_raw:
        unit_raw = unit_raw.split("#", 1)[0].split(";", 1)[0].split("!", 1)[0].strip()

    unit = _norm_unit(unit_raw) if unit_raw else ""

    if not unit:
        unit = _norm_unit(default_unit) if default_unit else ""
        q = Quantity(val, unit, registry)
        if to_unit:
            if not q.unit:
                raise UnitError(
                    f"Cannot convert {text!r} to {to_unit!r} because no unit was provided "
                    "and no default_unit was specified."
                )
            return q.to(_norm_unit(to_unit))
        return q

    q = Quantity(val, unit, registry)
    return q.to(_norm_unit(to_unit)) if to_unit else q


__all__ = [
    "UnitError",
    "UnitDef",
    "UnitRegistry",
    "Quantity",
    "DEFAULT_REGISTRY",
    "default_registry",
    "parse_quantity",
    "convert",
    "convert_value",
]
