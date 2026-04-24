from CoolProp.CoolProp import PropsSI

def incomp_solution_limits(name: str, x_try: float = 0.2) -> dict:
    # Need *some* valid x to build the fluid string; we'll try a few guesses.
    guesses = [x_try, 0.1, 0.15, 0.25, 0.3, 0.05, 0.01, 0.4, 0.5, 0.6, 0.7]
    last_err = None

    for x in guesses:
        fluid = f"INCOMP::{name}[{x}]"
        try:
            # Just to verify this x is acceptable
            PropsSI("D", "T", 300, "P", 101325, fluid)

            return {
                "fluid": fluid,
                "x_working": x,
                "Tmin_K": PropsSI("Tmin", "T", 0, "P", 0, fluid),
                "Tmax_K": PropsSI("Tmax", "T", 0, "P", 0, fluid),
                "x_min":  PropsSI("fraction_min", "T", 0, "P", 0, fluid),
                "x_max":  PropsSI("fraction_max", "T", 0, "P", 0, fluid),
                "T_freeze_K": PropsSI("T_freeze", "T", 0, "P", 0, fluid),
            }
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not find a valid x for {name}. Last error: {last_err}")

if __name__ == "__main__":
    for name in ["MAM", "MAM2", "LiBr", "MEG", "MPG", "MITSW"]:
        try:
            info = incomp_solution_limits(name)
            print(name, info)
        except Exception as e:
            print(name, "->", e)
