import CoolProp
print(CoolProp.CoolProp.get_global_param_string("FluidsList"))

import CoolProp.CoolProp as CP
print(CP.get_global_param_string("predefined_mixtures"))

pairs = CP.get_global_param_string("mixture_binary_pairs_list")
print("NH3-H2O pair present?",
      ("7664-41-7&7732-18-5" in pairs) or ("7732-18-5&7664-41-7" in pairs))
