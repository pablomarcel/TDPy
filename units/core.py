# units/core.py
from __future__ import annotations

"""
units.core

Core units implementation.

This file is intentionally dependency-free and small, but still structured:
- UnitDef: linear mapping to base units in a dimension group
- UnitRegistry: add/get/convert units
- Quantity: (value, unit) wrapper with `.to()`

Design goals:
- deterministic conversions
- tiny surface area
- no parser logic here
- suitable as the stable "math + registry" layer for the rest of TDPy
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class UnitError(ValueError):
    """Raised when units are unknown or incompatible."""


# -----------------------------------------------------------------------------
# Core: linear unit conversions within a dimension group
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitDef:
    """
    Linear mapping to a dimension's base unit:

        base = value * factor + offset

    For most units offset = 0.
    Temperatures are handled via offsets (C, F, R).
    """
    dim: str
    factor: float
    offset: float = 0.0
    canonical: Optional[str] = None


def _norm(unit: str) -> str:
    """Normalize a unit token."""
    return str(unit).strip().replace("°", "").replace(" ", "")


class UnitRegistry:
    """
    Registry of unit definitions.

    Each unit belongs to a dimension group; conversions are only allowed
    within the same dimension group.
    """

    def __init__(self) -> None:
        self._units: Dict[str, UnitDef] = {}

    # ----- registration -----

    def add(
        self,
        unit: str,
        dim: str,
        factor: float,
        offset: float = 0.0,
        canonical: Optional[str] = None,
    ) -> None:
        key = _norm(unit)
        if not key:
            raise UnitError("Unit token cannot be empty.")
        self._units[key] = UnitDef(
            dim=dim,
            factor=float(factor),
            offset=float(offset),
            canonical=canonical or key,
        )

    # ----- lookup -----

    def has(self, unit: str) -> bool:
        return _norm(unit) in self._units

    def get(self, unit: str) -> UnitDef:
        key = _norm(unit)
        if key not in self._units:
            raise UnitError(f"Unknown unit: {unit!r}")
        return self._units[key]

    def dim(self, unit: str) -> str:
        return self.get(unit).dim

    # ----- conversions -----

    def to_base(self, value: float, from_unit: str) -> float:
        u = self.get(from_unit)
        return float(value) * u.factor + u.offset

    def from_base(self, base_value: float, to_unit: str) -> float:
        u = self.get(to_unit)
        if u.factor == 0.0:
            raise UnitError(f"Invalid unit factor for {to_unit!r}")
        return (float(base_value) - u.offset) / u.factor

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        uf = self.get(from_unit)
        ut = self.get(to_unit)
        if uf.dim != ut.dim:
            raise UnitError(
                f"Incompatible units: {from_unit!r} ({uf.dim}) -> {to_unit!r} ({ut.dim})"
            )
        base = self.to_base(float(value), from_unit)
        return self.from_base(base, to_unit)

    # ----- introspection -----

    def list_units(self, dim: Optional[str] = None) -> Dict[str, UnitDef]:
        if dim is None:
            return dict(self._units)
        return {k: v for k, v in self._units.items() if v.dim == dim}


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    registry: UnitRegistry

    @property
    def dim(self) -> str:
        return self.registry.dim(self.unit)

    def to(self, unit: str) -> "Quantity":
        v = self.registry.convert(self.value, self.unit, unit)
        return Quantity(v, unit, self.registry)

    def base_value(self) -> float:
        return self.registry.to_base(self.value, self.unit)

    def as_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "dim": self.dim}


__all__ = [
    "UnitError",
    "UnitDef",
    "UnitRegistry",
    "Quantity",
]