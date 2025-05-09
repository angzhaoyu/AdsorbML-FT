# AdsorbML-FT
AdsorbML-FT is a process developed based on the fine-tuning scheme of AdsorbML for accurate adsorption energy prediction and effective structural optimization of catalyst surfaces.The integrated MK (Microkinetic) component delivers steady-state approximation solutions for reaction network analysis.
## Overview

This comprehensive toolkit offers:
- Automated generation of diverse adsorption configurations through sophisticated heuristic and random methods
- High-precision structure optimization powered by state-of-the-art machine learning models
- Flexible training and fine-tuning capabilities for adsorption energy prediction models
- Robust processing and analysis of complex adsorption data
- Microkinetic modeling for reaction pathway analysis and product distribution prediction

## Installation

### Prerequisites

Before using AdsorbML-FT, you need to install FairChem:

```bash
# Follow the FairChem installation instructions
# https://github.com/facebookresearch/fairchem
```

### Setup

1. Clone this repository:
```bash
git clone https://github.com/angzhaoyu/AdsorbML-FT.git
cd AdsorbML-FT
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Activate FairChem environment

```bash
# Activate your FairChem environment
conda activate fair-chem
```

### 2. Navigate to the project directory

```bash
cd path/to/AdsorbML-FT/AdsorbML-FT
```

### 3. Initialize the project structure

```bash
python ./script/ml_start.py
```

<details>
<summary>Click to see execution output</summary>

```
Directory Structure:
##########################################################

├── 00-modeling
│   ├── 00-slab_mol/mol_cif # Molecules
│   └── 00-slab_mol/slab_cif # Slabs
│
├── 01-data_traj/        # Trajectory data
│   ├── *_traj/          # Adsorption structure trajectory files
│   ├── slab_traj/       # Catalyst surface trajectory files
│   └── E_mol            # Adsorption molecule energies
│
├── 02-FT/               # Fine-tuning results
└── 03-opt/              # Optimized structures

Usage Instructions:
0. In 00-modeling/00-slab_mol:
   - Place adsorption molecules in mol_cif
   - Place catalyst slabs in slab_cif

1. In 01-data/01-data_traj:
   - Place adsorption structure trajectory files in "*_traj" folder
   - Place catalyst surface trajectory files in "slab_traj" folder
   - Place molecule energies in "E_mol" file

2. Next step - Create database: Run "ml_incar.py" and "ml_lmdb.py"

##########################################################
```
</details>

This will create the following directory structure:
```
├── 00-modeling
│   ├── 00-slab_mol/mol_cif  # Molecules
│   └── 00-slab_mol/slab_cif # Slabs
│
├── 01-data
│   ├── 01-data_traj/        # Trajectory data
│   │   ├── *_traj/          # Adsorption structure trajectory files
│   │   ├── slab_traj/       # Catalyst surface trajectory files
│   │   └── E_mol            # Adsorption molecule energies
│   ├── 02-data_db/          # Database files
│   └── 03-data_lmdb/        # LMDB files for training
│
├── 02-FT/                   # Fine-tuning results
└── 03-opt/                  # Optimized structures
```

### 4. Generate INCAR file

```bash
python ./script/ml_incar.py
```

<details>
<summary>Click to see execution output</summary>

```
ml_incar Instructions
##########################################################

Please verify the ml_incar parameters!!!

Usage Instructions:
1. If you want to run "ml_lmdb.py", confirm "ns", default is 3!

2. If you want to run "ml_train.py",
    Make sure you have only 1 lmdb, or provide the complete lmdb path!
    Training epochs: 30, batch size: 3

##########################################################
```
</details>

This will create an INCAR file with your parameters. The ml_incar file contains:

<details>
<summary>Click to see ml_incar content</summary>

```
Global Parameters

# Model Path
checkpoint_path = /home/zlb/wzy-shell/fairchem_checkpoints/eq2_31M_ec4_allmd.pt  # Do not use double quotes

# Modeling Parameters

num_sites = 10  # Additional 10 structures from random heuristic, minimum 1
opt_cif = True  # Default: start calculation directly. Note: pay attention to optimization parameters
num_n = 5       # Lowest 5 structures. If not enough, can be increased
supply_traj = False  # Will not model or calculate
supply_model = False  # Automatic modeling. If set to True, you can provide your own POSCAR in ads_slab
calc_moldes = True    # Calculate optimized structures
num_cf = 20           # Number of configurations



# Database Parameters
ns = 3                # Default 1000 to include all trajectories
DFT_db = False        # Do not save slab and DFT database
save_csv = True       # Save database as table
split = [8,2,0]       # Train:Validation:Test ratio, typically 8:2

# Training Parameters
src = ./01-data/03-data_lmdb  # Path to training data, containing train and val directories
max_epochs = 30
batch_size = 3
eval_batch_size = 3
yml = False           # Default: no yml. If available for single instance, provide path



# Optimization Parameters
Optcif Parameters
cpu = False
opt_fmax = 0.02       # Force convergence criterion for model optimization
opt_steps = 500       # Maximum number of steps
min_constraints = 32  # Fix the lowest n atoms
tags0 = 32            # n, set the lowest n atoms to 0 (fixed layer)
tags1 = 64            # m, set atoms from n to m to 0 (surface layer). Calculate yourself if adsorbate is below surface
```
</details>

### 5. Prepare your structures

Place your surface and adsorption molecules in the following folders:
```
├── 00-modeling
│   ├── 00-slab_mol/mol_cif  # Molecules
│   └── 00-slab_mol/slab_cif # Slabs
```

### 6. Generate and optimize structures

```bash
python ./script/ml_modeling.py
```

<details>
<summary>Click to see execution output (partial)</summary>

```
/home/zlb/anaconda3/envs/fair-chem/lib/python3.12/site-packages/fairchem/core/models/scn/spherical_harmonics.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
/home/zlb/anaconda3/envs/fair-chem/lib/python3.12/site-packages/fairchem/core/models/escn/so3.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
/home/zlb/anaconda3/envs/fair-chem/lib/python3.12/site-packages/fairchem/core/models/equiformer_v2/wigner.py:10: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
/home/zlb/anaconda3/envs/fair-chem/lib/python3.12/site-packages/fairchem/core/common/relaxation/ase_utils.py:200: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
WARNING:root:Detected old config, converting to new format. Consider updating to avoid potential incompatibilities.
INFO:root:local rank base: 0
INFO:root:amp: true
cmd:
  checkpoint_dir: /home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/checkpoints/2025-05-08-17-25-20
  commit: core:None,experimental:NA
  identifier: ''
  logs_dir: /home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/logs/wandb/2025-05-08-17-25-20
  print_every: 100
  results_dir: /home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/results/2025-05-08-17-25-20
  seed: null
  timestamp_id: 2025-05-08-17-25-20
  version: 1.8.1
dataset:
  format: trajectory_lmdb_v2
  grad_target_mean: 0.0
  grad_target_std: 2.887317180633545
  key_mapping:
    force: forces
    y: energy
  normalize_labels: true
  target_mean: -0.7554450631141663
  target_std: 2.887317180633545
  transforms:
    normalizer:
      energy:
        mean: -0.7554450631141663
        stdev: 2.887317180633545
      forces:
        mean: 0.0
        stdev: 2.887317180633545
...

      Step     Time          Energy          fmax
BFGS:    0 17:26:21       16.373116       73.380187
BFGS:    1 17:26:21        7.967644       44.792383
BFGS:    2 17:26:21        2.211291       10.458543
...
```
</details>

This will use heuristic and random methods to generate a series of structures and optimize them using the model specified in the ml_incar checkpoint path.

### 7. DFT calculations

Take the structures from 03-poscar or 04-low5-poscar and perform DFT calculations. Set the NSW parameter in the VASP INCAR according to your needs (default is 3).

> **Note:** You can skip the VASP calculation process if you already have trajectory files.

### 8. Prepare trajectory files

Place your trajectory files in the following directories:
```
/AdsorbML-FT/AdsorbML-FT/01-data/01-data_traj/ads_traj    # Structures with adsorbates
/AdsorbML-FT/AdsorbML-FT/01-data/01-data_traj/slab_traj   # Slab structures
```

The adsorption energy calculation differs from FairChem's approach. We directly use:
```
Eads = E(ads) - E(slab) - E(mol) 
```
E(mol) can be set in /AdsorbML-FT/AdsorbML-FT/01-data/01-data_traj/E_mol

#### 9 Database Creation and Model Training

After placing trajectory files in the appropriate directories:
```
/AdsorbML-FT/AdsorbML-FT/01-data/01-data_traj/ads_traj    # Structures with adsorbates
/AdsorbML-FT/AdsorbML-FT/01-data/01-data_traj/slab_traj   # Slab structures
```

We ran the ml_lmdb.py script to create the database:

<details>
<summary>Click to see ml_lmdb.py output</summary>

```
If you see progress bars, it means the script ran successfully!
Next step: Run "ml_train.py"
```
</details>

Then we ran ml_train.py to train the model:

<details>
<summary>Click to see ml_train.py output (partial)</summary>

```
INFO:root:local rank base: 0
INFO:root:amp: true
cmd:
  checkpoint_dir: /home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/checkpoints/2025-05-08-19-13-20
  commit: core:None,experimental:NA
  identifier: ''
  logs_dir: /home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/logs/wandb/2025-05-08-19-13-20
  print_every: 100
  results_dir: /home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/results/2025-05-08-19-13-20
  seed: null
  timestamp_id: 2025-05-08-19-13-20
  version: 1.8.1
dataset:
  format: trajectory_lmdb_v2
  grad_target_mean: 0.0
  grad_target_std: 2.887317180633545
  key_mapping:
    force: forces
    y: energy
  normalize_labels: true
  target_mean: -0.7554450631141663
  target_std: 2.887317180633545
  transforms:
    normalizer:
      energy:
        mean: -0.7554450631141663
        stdev: 2.887317180633545
      forces:
        mean: 0.0
        stdev: 2.887317180633545
...
```
</details>

The trained model was saved in the 02-FT/ads_ns3 directory:
```
bs3_max30_eq2_31M_ec4_allmd.pt
bs3_max30_eq2_31M_ec4_allmd.txt
bs3_max30_eq2_31M_ec4_allmd.yml
```


### 10. Optimize structures with the trained model

1. Copy the initial structures from the ads_slab directory to a new directory in 03-opt:

```bash
# Create a directory with a name ending in "_cif"
mkdir -p AdsorbML-FT/AdsorbML-FT/03-opt/Cu3M1-CO_cif

# Copy structures from ads_slab
cp AdsorbML-FT/AdsorbML-FT/00-modeling/01-ads_slab/* AdsorbML-FT/AdsorbML-FT/03-opt/Cu3M1-CO_cif/
```

> **Note:** The directory name can be anything but must end with "_cif".





2. Copy the ml_incar file to the 03-opt directory:

```bash
cp ml_incar AdsorbML-FT/AdsorbML-FT/03-opt/
```

3. Edit the ml_incar file to use your trained model:

```bash
# Edit the checkpoint_path parameter to point to your trained model
# For example:
# checkpoint_path = ./02-FT/your_model_name/bs3_max30_eq2_31M_ec4_allmd.pt
```

4. Run the optimization:

```bash
cd AdsorbML-FT/AdsorbML-FT/03-opt/
python ../script/opt_cif.py
```

<details>
<summary>Click to see opt_cif.py output (partial)</summary>

```
      Step     Time          Energy          fmax
BFGS:    0 19:25:54       -0.046185        2.206761
BFGS:    1 19:25:54       -0.075067        2.803944
BFGS:    2 19:25:54       -0.115827        1.059300
...    
BFGS:   37 19:25:55       -0.669343        0.009228
      Step     Time          Energy          fmax
BFGS:    0 19:25:55       -0.774622        1.011775
...
```
</details>

### 11. Extract the lowest energy structures

After optimization, run the get_lower_cif.py script to extract the structures with the lowest energy:

```bash
python ../script/get_lower_cif.py
```

This will create a new directory with "_lower" suffix containing the lowest energy structures.

<details>
<summary>Click to see lowest_energy_files.csv</summary>

```
File,Energy (eV),Base Name
Cu3Ag1_111_CO_19.traj,-0.7326459288597107,Cu3Ag1_111_CO
Cu3Al1_111_CO_20.traj,-0.6951969861984253,Cu3Al1_111_CO
Cu3As1_111_CO_6.traj,-0.7857128977775574,Cu3As1_111_CO
Cu3Au1_111_CO_7.traj,-0.605774998664856,Cu3Au1_111_CO
Cu3Cd1_111_CO_9.traj,-0.9426915645599365,Cu3Cd1_111_CO
Cu3Co1_111_CO_21.traj,-2.2576684951782227,Cu3Co1_111_CO
Cu3Cu1_111_CO_4.traj,-0.7947379350662231,Cu3Cu1_111_CO
Cu3Ga1_111_CO_11.traj,-0.7107882499694824,Cu3Ga1_111_CO
Cu3Ge1_111_CO_6.traj,-0.6272360682487488,Cu3Ge1_111_CO
Cu3In1_111_CO_2.traj,-0.6652964353561401,Cu3In1_111_CO
Cu3Ni1_111_CO_1.traj,-1.806041955947876,Cu3Ni1_111_CO
Cu3Pd1_111_CO_18.traj,-1.2043360471725464,Cu3Pd1_111_CO
Cu3Pt1_111_CO_13.traj,-1.647141933441162,Cu3Pt1_111_CO
Cu3Rh1_111_CO_3.traj,-2.0691423416137695,Cu3Rh1_111_CO
Cu3Sb1_111_CO_14.traj,-0.49816271662712097,Cu3Sb1_111_CO
Cu3Sn1_111_CO_17.traj,-0.563944935798645,Cu3Sn1_111_CO
Cu3Zn1_111_CO_3.traj,-0.9388298392295837,Cu3Zn1_111_CO
```
</details>

The Cu3M1-CO_lower directory contains the optimized CIF files for each of the lowest energy structures.


## Configuration

The `ml_incar` file contains important parameters for the workflow:

- `checkpoint_path`: Path to the model checkpoint
- `num_sites`: Number of adsorption sites to generate
- `min_constraints`: Minimum constraints for structure optimization
- `tags0`, `tags1`: Tags for atom classification
- `opt_fmax`: Force convergence criterion
- `opt_steps`: Maximum optimization steps
- `num_cf`: Number of configurations to generate
- `num_n`: Number of lowest energy structures to save
- `calc_moldes`: Whether to calculate molecular descriptors
- `supply_traj`: Whether to supply trajectory files
- `supply_model`: Whether to supply a model
- `ns`: Number of structures to include in database
- `save_csv`: Whether to save CSV files
- `DFT_db`: Whether to save DFT database
- `split`: Train/val/test split ratio

## Part 2: MK Component

The MK (Microkinetic) component provides sophisticated tools for steady-state approximation solutions to analyze complex reaction networks and calculate surface coverages and reaction rates with high precision.

### 2.1 MK Overview

The advanced microkinetic modeling capabilities enable researchers to:
- Calculate accurate steady-state surface coverages of adsorbates under various reaction conditions
- Determine precise reaction rates for competing reaction pathways
- Analyze the detailed kinetics of catalytic reactions at the molecular level
- Predict product distributions for multi-step reaction networks
- Identify rate-determining steps in catalytic cycles


### 2.2 Boundary Conditions

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


### 2.3 Usage

To use the MK component:

1. Prepare a table of reactions with rate constants
2. Define boundary conditions for gas and liquid species
3. Specify the target products to analyze
4. Run the microkinetic model to calculate coverages and rates

```python
# Example usage
import pandas as pd
from MK.cal_MK import cal_coverge

# Load reaction data
df_1 = pd.read_excel("./MK/MK.xlsx")

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
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FairChem](https://github.com/facebookresearch/fairchem) project for providing the machine learning framework
- [Open Catalyst Project](https://opencatalystproject.org/) for datasets and models
- Contributors to the AdsorbML-FT project for their valuable input and feedback

