from CoolProp.CoolProp import PropsSI

# Define parameters
T = 313.15  # Temperature in Kelvin (40°C)
P = 101325  # Pressure in Pa (Atmospheric)
x = 0.55  # Mass fraction of LiBr (55% LiBr, 45% H2O)

# Construct fluid string: INCOMP::LiBr[mass_fraction]
fluid = f"INCOMP::LiBr[{x}]"

try:
    # 1. Calculate Density (kg/m^3)
    rho = PropsSI('D', 'T', T, 'P', P, fluid)

    # 2. Calculate Specific Enthalpy (J/kg)
    h = PropsSI('H', 'T', T, 'P', P, fluid)

    # 3. Calculate Specific Heat Capacity (J/kg-K)
    cp = PropsSI('C', 'T', T, 'P', P, fluid)

    print(f"Properties for {x * 100}% LiBr solution at {T - 273.15:.1f}°C:")
    print(f"Density: {rho:.2f} kg/m^3")
    print(f"Enthalpy: {h / 1000:.2f} kJ/kg")
    print(f"Specific Heat: {cp:.2f} J/kg-K")

except Exception as e:
    print(f"Error: {e}")
