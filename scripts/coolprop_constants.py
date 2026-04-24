import CoolProp.CoolProp as CP
# Get Molar Mass in kg/mol
molar_mass = CP.PropsSI('M', 'Nitrogen')
# R_spec = R_u / M
R_spec = CP.get_config_double(CP.R_U_CODATA) / molar_mass

Tcrit = CP.PropsSI("Tcrit", "Water")
pcrit = CP.PropsSI("pcrit", "Water")

# Get the acentric factor for Nitrogen
omega = CP.PropsSI('acentric', 'T', 0, 'P', 0, 'Nitrogen')

print(f"Acentric factor of Nitrogen: {omega}")
# Expected output: ~0.0372
