from CoolProp.CoolProp import PropsSI

# T = 300 K, P = 101325 Pa, x = 0.2 (20% Ammonia)
fluid = "INCOMP::MAM[0.2]"

density = PropsSI('D', 'T', 300, 'P', 101325, fluid)
enthalpy = PropsSI('H', 'T', 300, 'P', 101325, fluid)

print(f"Ammonia-Water Density: {density:.2f} kg/m^3")
