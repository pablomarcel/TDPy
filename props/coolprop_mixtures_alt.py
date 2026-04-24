from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP

def show(label: str, fluid: str, T: float, P: float) -> None:
    rho = PropsSI("D", "T", T, "P", P, fluid)
    h   = PropsSI("H", "T", T, "P", P, fluid)
    cp  = PropsSI("C", "T", T, "P", P, fluid)
    print(f"{label:18s} | {fluid:35s} | rho={rho:10.3f}  h={h:12.3f}  cp={cp:10.3f}")

def has_binary_pair(fluid_a: str, fluid_b: str) -> bool:
    # Works for HEOS fluids; returns CAS strings like "7664-41-7"
    cas_a = CP.get_fluid_param_string(fluid_a, "CAS")
    cas_b = CP.get_fluid_param_string(fluid_b, "CAS")
    pairs = CP.get_global_param_string("mixture_binary_pairs_list")
    return (f"{cas_a}&{cas_b}" in pairs) or (f"{cas_b}&{cas_a}" in pairs)

if __name__ == "__main__":
    T = 300.0
    P = 101325.0

    show("pure", "Ammonia", T, P)
    show("pure", "Water",   T, P)

    # Prove mixtures work (pseudo-pure refrigerant blend)
    show("pseudo-mixture", "R407C", T, P)

    # NH3-H2O: guard
    if has_binary_pair("Ammonia", "Water"):
        show("mix(HEOS)", "Ammonia[0.30]&Water[0.70]", T, P)
    else:
        print("mix(HEOS)          | Ammonia+Water unsupported in CoolProp (no binary pair data)")

    # REFPROP attempt (if installed)
    try:
        show("mix(REFPROP)", "REFPROP::Ammonia[0.30]&Water[0.70]", T, P)
    except Exception as e:
        print("REFPROP backend not available via CoolProp:", type(e).__name__, e)

    show("INCOMP", "INCOMP::MITSW[0.035]", 320.0, P)
