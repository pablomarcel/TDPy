from CoolProp.CoolProp import PropsSI

T = 350.0          # K
x = 0.55           # mass fraction LiBr
fluid = f"INCOMP::LiBr[{x}]"

P_eq = PropsSI("P", "T", T, "Q", 0, fluid)      # equilibrium vapor pressure over solution
P_sat_water = PropsSI("P", "T", T, "Q", 0, "Water")

print("T [K] =", T)
print("LiBr x =", x)
print("P_eq over LiBr [Pa] =", P_eq)
print("P_sat pure water [Pa] =", P_sat_water)
