import CoolProp.CoolProp as CP

print("PURE INCOMP:")
print(CP.get_global_param_string("incompressible_list_pure"))

print("\nINCOMP SOLUTIONS:")
print(CP.get_global_param_string("incompressible_list_solution"))
