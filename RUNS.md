# RUNS — `tdpy` Command Cookbook

Updated: **2026-04-21**  
Run these from the **thermodynamics root** (the directory that contains ``).  
Outputs default to `out/`.

## 0) Help

```bash
python -m cli \
  --help
```

Per-command help:

```bash
python -m cli run \
  --help
```

```bash
python -m cli props \
  --help
```

```bash
python -m cli eqn \
  --help
```

## 1) Text UI (console front-end)

Interactive menu:

```bash
python -m main
```

Or via CLI subcommand:

```bash
python -m cli menu
```

## 2) List available input cases

List relative paths under `in`:

```bash
python -m cli list-inputs
```

To print absolute paths for copy/paste:

```bash
python -m cli list-inputs \
  --verbose
```

## 3) Nozzle (ideal gas, quasi-1D isentropic)

### 3a) Run the demo TXT case (EES-ish `key=value`)

```bash
python -m cli run \
  --in demo/nozzle_case.txt
```

By default, output is written to:

- `out/nozzle_case.out.json`

### 3b) Choose an explicit output name

```bash
python -m cli run \
  --in demo/nozzle_case.txt \
  --out nozzle_case.custom.out.json
```

### 3c) Generate default plots (Plotly) during the run

```bash
python -m cli run \
  --in demo/nozzle_case.txt \
  --make-plots
```

Typical outputs:

- `out/nozzle_case.out.json`
- `out/nozzle_case.out.mach.html`
- `out/nozzle_case.out.pressure.html`
- `out/nozzle_case.out.temperature.html`

## 4) Thermodynamic properties (CoolProp) — `thermo_props`

### 4.0) Quick note on input shapes

The thermo props runner accepts **single-state** or **batch** specs.  
You can express a state either:
- as two top-level keys (e.g. `T_C` + `x`), **or**
- as a nested `state: {...}`, **or**
- as `states: [{...}, {...}]`.

Keys can be EES-ish or unit-suffixed (examples: `T_C`, `T_K`, `P_bar`, `P_kPa`, `h_kJkg`, `s_kJkgK`, `rho_kgm3`, `v_m3kg`, `x`).

### 4.1) Create a single-state demo input (JSON)

Create: `in/demo/props_r134a_Tx.json`

```json
{
  "problem_type": "thermo_props",
  "backend": "coolprop",
  "fluid": "R134a",

  "states": [
    {
      "id": "sat_vapor_Tx",
      "given": { "T_C": -10, "x": 1.0 },
      "ask": ["T", "P", "Q", "Hmass", "Smass", "Dmass", "Cpmass", "Cvmass", "A", "V", "L", "Prandtl"]
    }
  ],

  "meta": {
    "title": "R134a saturated vapor at -10C",
    "note": "Single-state thermo_props evaluation using (T, x) inputs."
  }
}
```

Run it (generic runner):

```bash
python -m cli run \
  --in demo/props_r134a_Tx.json
```

Or via the convenience command:

```bash
python -m cli props \
  --in demo/props_r134a_Tx.json
```

### 4.2) Same single-state case, but “flat” top-level inputs (TXT)

Create: `in/demo/props_air_TP.txt`

```txt
# props_air_TP.txt
# EES-ish key=value format (parsed by io)
# Thermo props: provide exactly two independent state inputs

problem_type = thermo_props
fluid = HEOS::Air

# --- given (two independent props) ---
given.T_K = 300
given.P_bar = 1.01325

# --- ask / outputs ---
ask = T,P,Hmass,Smass,Dmass,Cpmass,Cvmass,A
```

Run:

```bash
python -m cli props \
  --in demo/props_air_TP.txt
```

### 4.3) Batch states (JSON)

Create: `in/demo/props_batch.json`

```json
{
  "problem_type": "thermo_props",
  "backend": "coolprop",
  "fluid": "R134a",

  "states": [
    {
      "id": "sat_vapor_-10C",
      "given": { "T_C": -10, "x": 1.0 },
      "ask": ["T", "P", "Q", "Hmass", "Smass", "Dmass", "Cpmass", "Cvmass", "A"]
    },
    {
      "id": "sat_liquid_35C",
      "given": { "T_C": 35, "x": 0.0 },
      "ask": ["T", "P", "Q", "Hmass", "Smass", "Dmass", "Cpmass", "Cvmass", "A"]
    },
    {
      "id": "superheated",
      "given": { "P_bar": 8.0, "T_C": 60.0 },
      "ask": ["T", "P", "Q", "Hmass", "Smass", "Dmass", "Cpmass", "Cvmass", "A"]
    }
  ],

  "meta": {
    "title": "R134a thermo batch states",
    "note": "Three states: saturated vapor at -10C, saturated liquid at 35C, and a superheated point at 8 bar & 60C."
  }
}
```

Run:

```bash
python -m cli props \
  --in demo/props_batch.json \
  --out props_batch.out.json
```

## 5) Equation solving — `equations`

### 5.1) Minimal 2×2 nonlinear system (JSON)

Solve:
- `x + y = 2`
- `x*y = 1`

Create: `in/demo/eqn_xy.json`

```json
{
  "problem_type": "equations",
  "backend": "auto",
  "method": "hybr",
  "tol": 1e-10,
  "max_iter": 200,

  "variables": {
    "x": { "guess": 1.5 },
    "y": { "guess": 0.5 }
  },

  "equations": [
    "x + y - 2",
    "x*y - 1"
  ],

  "meta": {
    "title": "Solve simple 2x2 system",
    "note": "Find (x,y) such that x+y=2 and x*y=1."
  }
}
```

Run (generic runner):

```bash
python -m cli run \
  --in demo/eqn_xy.json \
  --out eqn_xy.out.json
```

Or via the convenience command:

```bash
python -m cli eqn \
  --in demo/eqn_xy.json
```

### 5.2) Same thing, but include a fixed parameter (JSON)

Create: `in/demo/eqn_circle.json`

```json
{
  "problem_type": "equations",
  "backend": "auto",
  "method": "hybr",
  "tol": 1e-10,
  "max_iter": 200,

  "constants": {
    "r": 2.0,
    "k": 1.0
  },

  "variables": {
    "x": { "guess": 1.0 },
    "y": { "guess": 1.7 }
  },

  "equations": [
    "x*x + y*y - r*r",
    "y - k*x"
  ]
}
```

Run:

```bash
python -m cli eqn \
  --in demo/eqn_circle.json \
  --out eqn_circle.out.json
```

### 5.3) Equations from a text file (JSON + equations_file)

Create: `in/demo/eqn_file.txt`

```txt
# eqn_file.txt
# one equation per line (blank lines + comments allowed)

x + y - 2
x*y - 1
```

Create: `in/demo/eqn_from_file.json`

```json
{
  "problem_type": "equations",
  "backend": "auto",
  "method": "hybr",
  "tol": 1e-10,
  "max_iter": 200,

  "variables": {
    "x": { "guess": 1.5 },
    "y": { "guess": 0.5 }
  },

  "equations_file": "eqn_file.txt"
}
```

Run:

```bash
python -m cli eqn \
  --in demo/eqn_from_file.json \
  --out eqn_from_file.out.json
```

## 6) Quick Python “smoke tests” (no CLI)

### 6.1) Thermo props call

```bash
python - << 'PY'
from thermo_props import props

d = props(fluid="HEOS::Air", T_K=300.0, P_bar=1.01325, outputs=("T","P","Hmass","Smass","Dmass"))
print(d)
PY
```

### 6.2) Equation solve (file-driven via app)

```bash
python - << 'PY'
from pathlib import Path
from app import TdpyApp
from apis import RunRequest

app = TdpyApp()
res = app.run(RunRequest(in_path=Path("demo/eqn_xy.json")))
print("ok:", res.ok)
print("out:", res.out_path)
print("payload keys:", sorted(res.payload.keys()))
PY
```

---

## 7) Developer notes

- `tdpy` is meant to feel like **EES**, but the internal design stays Pythonic:
  - parsing/validation in `io.py` + `design.py`
  - orchestration in `app.py`
  - CLI wiring in `cli.py`
  - `thermo_props` is CoolProp-backed (optional dependency)
  - `equations` routes to the equation solver backend you’ve wired (SciPy/GEKKO/auto)
- Keep `in/demo/*` as the canonical smoke-test folder for new features.

## 8) ad-hoc debug

```bash
python - << 'PY'
from pathlib import Path
from io import load_problem, in_dir
p = in_dir() / "demo/props_r134a_Tx.json"
print("FILE:", p)
print(load_problem(p))
PY
```

## 9) simple problem

```bash
python -m cli run \
  --in demo/problem.json
```

## 10) reversible work

```bash
python -m cli run \
  --in demo/reversible.json
```

## 11) interpret - structured input txt

```bash
python -m interpreter.cli \
  --in in/demo/Example_05_2_1_reversible_irreversible_work.txt \
  --out in/demo/reversible_interpreted.json
```

## 12) interpret - chaotic input txt

```bash
python -m interpreter.cli \
  --in in/demo/Example_05_2_1_reversible_chaos.txt \
  --out in/demo/reversible_chaos.json
```

## 13) solve interpreted case

```bash
python -m cli run \
  --in demo/reversible_interpreted.json
```

## 14) solve interpreted case - chaos

```bash
python -m cli run \
  --in demo/reversible_chaos.json
```

## 15) interpret - helium balloon

```bash
python -m interpreter.cli \
  --in in/demo/Example_03_5_2_helium_balloon.txt \
  --out in/demo/helium_balloon.json
```

## 16) solve - helium balloon

```bash
python -m cli run \
  --in demo/helium_balloon.json
```

## 17) solve - helium balloon - manual guess values

```bash
python -m cli run \
  --in demo/helium_balloon_guess.json
```

## 18) RipGrep

rg -n "PropsSI|RESERVED|ignore|builtin" eespy
rg -n "PropsSI" equations
rg -n "Interpreted assignment as symbol constant" eespy
rg -n "residual_norm" eespy
rg -n "5e-8" eespy
rg -n "ParsedInput|equation_lines|given_lines|guess_lines|report_names|\"equations\"|\"variables\"|\"report\"|solve_overrides" interpreter
rg -n "eespy"

## 19) Ammonia Water Mixture

```bash
python nh3h2o/ammonia_cooling_trf.py \
  --backend trf \
  --mode cycle
```

## 20) Verify Grid

```bash
python nh3h2o/verify_grid.py \
  --TminC 0 --TmaxC 120 --nT 7 \
  --Pbars 2,5,10,20,50 \
  --X 0.1,0.3,0.5,0.7,0.9 \
  --strict \
  --out out/nh3h2o_grid_absorption.csv
```

awk -F, 'NR==1 || ($13+0)<=0 || ($14+0)<=0 {print}' nh3h2o/out/nh3h2o_grid.csv

# q must be -0.001, 1.001, or between 0 and 1
awk -F, 'NR==1 || ($8!="") && !($8==-0.001 || $8==1.001 || ($8>=0 && $8<=1)) {print}' nh3h2o/out/nh3h2o_grid.csv

# entropy should be finite (not blank, not "nan")
awk -F, 'NR==1 || $11=="" || $11=="nan" || $11=="NaN" {print}' nh3h2o/out/nh3h2o_grid.csv

# Phase ↔ q consistency (must match your own convention)

awk -F, 'NR==1 {print; next} ($7=="L"   && $8!=-0.001) || ($7=="g"   && $8!=1.001)  || ($7=="2ph" && !($8>=0 && $8<=1)) {print}' nh3h2o/out/nh3h2o_grid.csv

# Two-phase rows must have xL/yV/wL/wV populated + ordered

awk -F, 'NR==1 {print; next} $7=="2ph" && ($15=="" || $16=="" || $17=="" || $18=="" || !($15+0 < $16+0) || !($17+0 < $18+0)) {print}' nh3h2o/out/nh3h2o_grid.csv
awk -F, 'NR>1 && $7=="2ph" && ($15=="" || $16=="" || $17=="" || $18=="" || !($15+0 < $16+0) || !($17+0 < $18+0)) {c++} END{print c+0}' nh3h2o/out/nh3h2o_grid.csv

# Single-phase rows should NOT carry saturation fields (optional hygiene)

awk -F, 'NR>1 && ($7=="L" || $7=="g") && ($15!="" || $16!="" || $17!="" || $18!="") {c++} END{print c+0}' nh3h2o/out/nh3h2o_grid.csv

## 21) Smoke Test for NH3H2O

```bash
python - <<'PY'
from thermo_props.coolprop_backend import props_si
print(props_si("Hmass","T",283.15,"P",1.0e6,"NH3H2O|X=0.5|strict=1"))
PY
```

## 22) CoolProp Reference State Shifting

```bash
python sandbox/coolprop_refstate_sandbox.py
```

```bash
python sandbox/coolprop_refstate_sandbox.py \
  --fluid n-Propane
```

```bash
python sandbox/coolprop_refstate_sandbox.py \
  --fluid Ammonia
```

```bash
python sandbox/coolprop_refstate_sandbox.py \
  --backend REFPROP \
  --fluid R134a
```

```bash
python sandbox/coolprop_refstate_sandbox.py \
  --out out/refstate_demo.json
```

```bash
python - <<'PY'
import CoolProp.CoolProp as CP

fluid = "R134a"
CP.set_reference_state(fluid, "NBP")

for P in (100000.0, 101325.0):
    h = CP.PropsSI("HMASS","P",P,"Q",0,fluid)  # J/kg, sat liquid
    s = CP.PropsSI("SMASS","P",P,"Q",0,fluid)  # J/kg/K, sat liquid
    print(f"P={P:9.1f} Pa  HMASS={h: .6f} J/kg  SMASS={s: .6f} J/kg/K")
PY
```

# NBP 1 bar vs 1 atm

```bash
python sandbox/coolprop_refstate_sandbox_universal.py \
  --fluid R134a \
  --modes NBP \
  --nbp-P 100000 \
  --nbp-P 101325
```

# Custom reference state example (set h=s=0 at 300 K, 1 atm)

```bash
python sandbox/coolprop_refstate_sandbox_universal.py \
  --fluid R134a \
  --modes CUSTOM,DEF \
  --enable-custom \
  --custom-T0 300 \
  --custom-P0 101325 \
  --custom-hmolar0 0 \
  --custom-smolar0 0
```

## 23) CoolProp Constants

```bash
python sandbox/coolprop_constants.py \
  --fluid Nitrogen \
  --fluid Water
```

```bash
python sandbox/coolprop_constants.py \
  --backend HEOS \
  --fluid Nitrogen \
  --out out/coolprop_constants.json
```

## 24) Cantera - Reformation of Methane

```bash
python sandbox/cantera_reformation_methane.py \
  --T 1100 \
  --Pbar 10 \
  --mech gri30.yaml \
  --book_yH2 0.4899
```

## 25) GUI - Dear PyGui

```bash
python -m gui_core_dpg
```

### Sphinx commands

python -m cli sphinx-skel docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html