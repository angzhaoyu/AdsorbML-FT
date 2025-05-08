# MK (Microkinetic) Component

This directory contains the Microkinetic (MK) modeling component for analyzing reaction networks and calculating surface coverages and reaction rates.

## Files

- `cal_MK.py`: Python module with functions for microkinetic modeling
- `cal_MK.ipynb`: Jupyter notebook with examples and demonstrations
- `MK.xlsx`: Excel file containing reaction data and rate constants

## Overview

The MK component uses microkinetic modeling to:
- Calculate steady-state surface coverages of adsorbates
- Determine reaction rates for different pathways
- Analyze the kinetics of catalytic reactions
- Predict product distributions

## Key Functions

The MK component includes the following key functions:

1. `get_ra(R, P, k)` - Calculate reaction rate for a given reaction
2. `get_RP(eq_name, dict)` - Extract reactants and products from a reaction equation
3. `find_k(df_Eak, eq_1)` - Find rate constants for a given reaction
4. `get_ads(eq_name)` - Identify adsorbates and small molecules in a reaction
5. `initial_ads(eqs)` - Initialize adsorbate concentrations
6. `ad_bound(y, dict_bound)` - Apply boundary conditions
7. `MK(dict_ads, eqs, df, dict_bound)` - Calculate steady-state coverages and reaction rates
8. `cal_coverge(df_1, dict_bound, ads)` - Calculate coverage and reaction rates

## Boundary Conditions

The MK model requires boundary conditions to be specified for the gas-phase and liquid-phase species. These are set in the `dict_bound` dictionary:

```python
dict_bound = {
    "CO(g)": 1,      # CO gas concentration
    "H": 1,          # H concentration
    "H2O(l)": 1,     # Water concentration
    "CH2O(g)": 0,    # Formaldehyde concentration (initially 0)
    "CH4(g)": 0,     # Methane concentration (initially 0)
    "CH3OH(l)": 0,   # Methanol concentration (initially 0)
    "*": 1           # Free surface sites
}
```

## Usage Example

```python
import pandas as pd
from cal_MK import cal_coverge

# Load reaction data
df_1 = pd.read_excel("MK.xlsx")  # Without pre-exponential factor

# Define boundary conditions
dict_bound = {
    "CO(g)": 1,
    "H": 1,
    "H2O(l)": 1,
    "CH2O(g)": 0,
    "CH4(g)": 0,
    "CH3OH(l)": 0,
    "*": 1
}

# Specify target products
mol = ["CO(g)", "CH2O(g)", "CH3OH(l)", "CH4(g)"]

# Calculate coverages and rates
coverage, rate = cal_coverge(df_1, dict_bound, mol)
print("Coverage:", coverage)
print("Reaction rates:", rate)
```

## Reaction Format

Reactions should be specified in the format:
```
R1+R2-P1+P2
```

Where:
- `R1`, `R2` are reactants
- `P1`, `P2` are products
- `-` separates reactants and products
- `+` separates multiple reactants or products

For example:
```
CO+*-*CO       # CO adsorption
*CO+H-*CHO     # CO hydrogenation to CHO
*CH3OH-CH3OH+* # Methanol desorption
```

## Rate Constants

Each reaction has forward (k_forward) and reverse (k_reverse) rate constants. These are stored in the MK.xlsx file and used to calculate the reaction rates at steady state.

The rate constants should be provided in the format:
```
[k_forward, k_reverse]
```

For example:
```
[1.0e+8, 1.0e+3]  # Forward rate: 1.0e+8, Reverse rate: 1.0e+3
```
