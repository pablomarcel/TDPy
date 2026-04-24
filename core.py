from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from utils import timed, with_error_context

Branch = Literal["sub", "sup"]


def area_mach(M: float, k: float) -> float:
    """Return A/A* as a function of Mach number for a perfect gas."""
    if M <= 0:
        raise ValueError("Mach must be positive.")
    t = (2.0 / (k + 1.0)) * (1.0 + 0.5 * (k - 1.0) * M * M)
    expo = (k + 1.0) / (2.0 * (k - 1.0))
    return (1.0 / M) * (t ** expo)


def _bisect_root(f, lo: float, hi: float, tol: float = 1e-10, maxit: int = 200) -> float:
    flo = f(lo)
    fhi = f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("Root is not bracketed.")
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def mach_from_area_ratio(r: float, k: float, branch: Branch) -> float:
    """Invert area-Mach relation for A/A* = r."""
    if r < 1.0:
        raise ValueError(f"area ratio must be >= 1; got {r}")
    if abs(r - 1.0) < 1e-12:
        return 1.0

    def f(M: float) -> float:
        return area_mach(M, k) - r

    if branch == "sub":
        return _bisect_root(f, 1e-9, 0.999999999)
    return _bisect_root(f, 1.000000001, 50.0)


def static_from_stag(T0: float, P0: float, M: float, k: float, R: float) -> Dict[str, float]:
    T = T0 / (1.0 + 0.5 * (k - 1.0) * M * M)
    P = P0 * (1.0 + 0.5 * (k - 1.0) * M * M) ** (-k / (k - 1.0))
    a = math.sqrt(k * R * T)
    V = M * a
    rho = P / (R * T)
    v = 1.0 / rho
    return {"T": T, "P": P, "a": a, "V": V, "rho": rho, "v": v}


def choked_mass_flow(Astar: float, T0: float, P0: float, k: float, R: float) -> float:
    term = (2.0 / (k + 1.0)) ** ((k + 1.0) / (2.0 * (k - 1.0)))
    return Astar * P0 * math.sqrt(k / (R * T0)) * term


@dataclass(frozen=True)
class NozzleProfileSpec:
    k: float
    R: float
    T0_K: float
    P0_Pa: float
    x_mm: List[float]
    D_mm: List[float]
    branch_after_throat: Branch = "sup"


class Solver:
    name: str = "solver-base"
    def solve(self, spec: Any) -> Dict[str, Any]:
        raise NotImplementedError


class NozzleIdealGasSolver(Solver):
    name = "nozzle-ideal-gas"

    @timed
    @with_error_context("NozzleIdealGasSolver.solve")
    def solve(self, spec: NozzleProfileSpec) -> Dict[str, Any]:
        if len(spec.x_mm) != len(spec.D_mm):
            raise ValueError("x_mm and D_mm must have same length.")
        D_m = [v * 1e-3 for v in spec.D_mm]
        A = [math.pi * (d ** 2) / 4.0 for d in D_m]
        i_throat = min(range(len(A)), key=lambda i: A[i])
        Astar = A[i_throat]
        A_ratio = [Ai / Astar for Ai in A]

        M: List[float] = []
        for i, r in enumerate(A_ratio):
            if i == i_throat:
                M.append(1.0)
            elif i < i_throat:
                M.append(mach_from_area_ratio(r, spec.k, "sub"))
            else:
                M.append(mach_from_area_ratio(r, spec.k, spec.branch_after_throat))

        states = [static_from_stag(spec.T0_K, spec.P0_Pa, Mi, spec.k, spec.R) for Mi in M]
        mdot = choked_mass_flow(Astar, spec.T0_K, spec.P0_Pa, spec.k, spec.R)

        return {
            "x_mm": spec.x_mm,
            "D_mm": spec.D_mm,
            "A_m2": A,
            "Astar_m2": Astar,
            "A_ratio": A_ratio,
            "M": M,
            "T_K": [s["T"] for s in states],
            "P_Pa": [s["P"] for s in states],
            "a_mps": [s["a"] for s in states],
            "V_mps": [s["V"] for s in states],
            "rho_kgm3": [s["rho"] for s in states],
            "mdot_kgps": mdot,
            "meta": {
                "k": spec.k,
                "R": spec.R,
                "T0_K": spec.T0_K,
                "P0_Pa": spec.P0_Pa,
                "i_throat": i_throat,
                "branch_after_throat": spec.branch_after_throat,
            },
        }
