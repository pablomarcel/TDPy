import CoolProp.CoolProp as CP

# 'Propane', 'n-Propane', and 'C3H8' are all valid aliases
fluid = "C3H8"

def check_state(label):
    # Testing at Standard Ambient Temperature and Pressure (SATP)
    T = 298.15  # 25 C
    P = 101325  # 1 atm
    h = CP.PropsSI('H', 'T', T, 'P', P, fluid)
    s = CP.PropsSI('S', 'T', T, 'P', P, fluid)
    print(f"--- {label} ---")
    print(f"H: {h/1000:7.2f} kJ/kg")
    print(f"S: {s/1000:7.2f} kJ/kg/K\n")

# 1. Default (Usually IIR convention: h=200, s=1 at 0C sat liq)
CP.set_reference_state(fluid, "DEF")
check_state("Default (IIR)")

# 2. Normal Boiling Point (h=0, s=0 at 1atm sat liq)
CP.set_reference_state(fluid, "NBP")
check_state("Preset: NBP")

# 3. MANUAL CONTROL (Example: Force h=0, s=0 at 25C, 1atm)
# We need the density (rho) at our desired 'zero' point first
T_ref = 298.15
P_ref = 101325
rho_ref = CP.PropsSI('D', 'T', T_ref, 'P', P_ref, fluid)

# Signature: set_reference_state(fluid, T, rho, h, s)
CP.set_reference_state(fluid, T_ref, rho_ref, 0.0, 0.0)
check_state("Manual: Zero at 25C/1atm")
