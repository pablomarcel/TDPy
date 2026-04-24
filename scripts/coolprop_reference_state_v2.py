import CoolProp.CoolProp as CP

fluid = "C3H8"
T_ref = 298.15 # 25 C
P_ref = 101325 # 1 atm (or use 'Q', 0 for saturated liquid)

# 1. Calculate the OFFSET at your textbook's "Zero Point"
# If your book says H=0 at 25C Sat Liquid, use 'Q', 0 here.
# If your book says H=0 at 25C 1atm Gas, use 'P', P_ref here.
h_offset = CP.PropsSI('H', 'T', T_ref, 'Q', 0, fluid)
s_offset = CP.PropsSI('S', 'T', T_ref, 'Q', 0, fluid)

print(f"[Log] Calculated Offsets: H={h_offset:.2f}, S={s_offset:.2f}")

# 2. Create a wrapper function to "Math the Math"
def get_props(output, input1_name, input1_val, input2_name, input2_val):
    val = CP.PropsSI(output, input1_name, input1_val, input2_name, input2_val, fluid)
    if output == 'H':
        return val - h_offset
    if output == 'S':
        return val - s_offset
    return val

# 3. VERIFY
h_new = get_props('H', 'T', T_ref, 'Q', 0)
s_new = get_props('S', 'T', T_ref, 'Q', 0)

print(f"\n--- Manual Offset Results ---")
print(f"Verified H: {h_new:12.4f} J/kg (Exactly 0.0)")
print(f"Verified S: {s_new:12.4f} J/kg/K (Exactly 0.0)")

# 4. Check the state that was giving you 47.14 kJ/kg
h_textbook = get_props('H', 'T', 298.15, 'P', 101325)
print(f"Corrected H (25C, 1atm): {h_textbook/1000:7.2f} kJ/kg")
