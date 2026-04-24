import warnings
import numpy as np
from typing import Dict, Any
from iapws.ammonia import H2ONH3
from scipy.optimize import minimize_scalar

# Suppress math warnings triggered by intermediate solver guesses
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_ammonia_properties(
        t_celsius: float,
        p_bar: float,
        x_nh3: float,
        phase: str = "liquid"
) -> Dict[str, Any]:
    """
    Calculates thermodynamic thermodynamic_properties for NH3-H2O using the
    Tillner-Roth & Friend (IAPWS) model.
    """
    state = H2ONH3()
    t_kelvin = t_celsius + 273.15
    p_target_mpa = p_bar / 10.0

    # Define density bounds based on target phase
    # Liquid: 500-1100 kg/m3 | Vapor: 0.1-100 kg/m3
    bounds = (400, 1100) if phase == "liquid" else (0.1, 200)

    def objective(rho: float) -> float:
        try:
            # iapws requires standard float types
            res = state._prop(float(rho), t_kelvin, x_nh3)
            return (res['P'] - p_target_mpa) ** 2
        except (ValueError, ZeroDivisionError, TypeError):
            return 1e10

    # Bounded solver ensures physically valid (positive) density
    res = minimize_scalar(objective, bounds=bounds, method='bounded')

    if not res.success:
        return {"status": "error", "message": "Optimization failed to converge"}

    # Extract final thermodynamic_properties
    rho_final = float(res.x)
    props = state._prop(rho_final, t_kelvin, x_nh3)

    # Validate convergence to target pressure
    if abs(props['P'] - p_target_mpa) > 0.05:
        return {"status": "error", "message": f"No physical {phase} state found."}

    return {
        "status": "success",
        "phase": phase,
        "temperature_c": t_celsius,
        "pressure_bar": round(float(props['P'] * 10), 2),
        "density_kg_m3": round(rho_final, 2),
        "enthalpy_kj_kg": round(float(props['h']), 2),
        "entropy_kj_kgk": round(float(props['s']), 4)
    }


# --- Execution Example ---
if __name__ == "__main__":
    # Example: Solution leaving the Generator (High P, High T)
    result = get_ammonia_properties(t_celsius=90, p_bar=15, x_nh3=0.4)

    print(f"{'Property':<20} | {'Value':<10}")
    print("-" * 35)
    for key, value in result.items():
        print(f"{key:<20} | {value}")
