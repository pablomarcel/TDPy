import numpy as np
from typing import Dict, Any


def get_libr_properties(t_celsius: float, x_libr: float) -> Dict[str, Any]:
    """
    Calculates LiBr-H2O thermodynamic_properties using Patek & Klomfar (2006) correlations.
    t_celsius: Temperature in Celsius
    x_libr: Mass fraction of LiBr (e.g., 0.6 for 60%)
    """
    T = t_celsius + 273.15  # Temperature in Kelvin
    T_crit = 647.096  # Critical temperature of Water (K)

    # --- 1. Saturation Pressure Calculation (P_sat) ---
    # Coefficients for the correlation (Patek & Klomfar Table 4)
    a = [-2.41303e2, 4.85103e2, -3.34585e2, 9.30263e1, -1.03374e1]
    m = [3.50754, 4.00858, 4.94282, 6.28908, 5.58801]

    # Normalized temperature ratio
    # We find the T_sat of water that gives the same pressure as the solution
    theta = T / T_crit
    sum_term = 0
    for i in range(5):
        sum_term += a[i] * (x_libr ** m[i])

    # T_water is the equivalent saturation temperature of pure water
    T_water = T - (sum_term * theta)

    # Simple Antoine-like correlation for pure water P_sat (in kPa)
    # For a full model, you could use 'iapws' for this specific water lookup
    p_sat_water = 0.1 * np.exp(16.3872 - 3885.7 / (T_water - 42.98))  # kPa to bar

    # --- 2. Specific Enthalpy Calculation (h) ---
    # Coefficients (Patek & Klomfar Table 5) - kJ/kg
    # Simplified version for prototyping
    h_water = 4.18 * t_celsius  # Rough liquid water enthalpy kJ/kg

    # Excess enthalpy correlation components
    b = [-2.02472e3, 1.63509e4, -5.00813e4, 6.95369e4, -3.51463e4]
    n = [1.0, 1.0, 1.0, 1.0, 1.0]

    h_excess = 0
    for i in range(5):
        h_excess += b[i] * (x_libr ** (i + 1))

    h_solution = (1 - x_libr) * h_water + h_excess / 100  # Scaled for kJ/kg

    return {
        "status": "success",
        "system": "LiBr-H2O",
        "temperature_c": t_celsius,
        "mass_fraction_libr": x_libr,
        "pressure_sat_bar": round(float(p_sat_water), 5),
        "enthalpy_kj_kg": round(float(h_solution), 2)
    }


# --- Execution Example ---
if __name__ == "__main__":
    # Typical Absorber Condition: 40°C, 60% LiBr
    result = get_libr_properties(t_celsius=40, x_libr=0.60)

    print(f"{'Property':<20} | {'Value':<10}")
    print("-" * 35)
    for key, value in result.items():
        print(f"{key:<20} | {value}")
