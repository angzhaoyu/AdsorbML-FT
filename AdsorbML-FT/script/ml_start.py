#!/usr/bin/env python3
"""
Initialize the AdsorbML-FT project structure.
This script creates the necessary directories and copies the E_mol file if available.
"""

import os
from pathlib import Path
import shutil


# Create directory structure
os.makedirs(f"00-modeling/00-slab_mol/mol_cif", exist_ok=True)
os.makedirs(f"00-modeling/01-ads_slab", exist_ok=True)
os.makedirs(f"00-modeling/02-traj", exist_ok=True)
os.makedirs(f"00-modeling/03-poscar", exist_ok=True)
os.makedirs(f"01-data/01-data_traj", exist_ok=True)
os.makedirs(f"01-data/01-data_traj/slab_traj", exist_ok=True)
os.makedirs(f"01-data/02-data_db", exist_ok=True)
os.makedirs(f"01-data/03-data_lmdb", exist_ok=True)
os.makedirs(f"02-FT", exist_ok=True)
os.makedirs(f"03-opt", exist_ok=True)

# Source file path for E_mol (molecular energies)
src_file = "/home/zlb/wzy-shell/fairchem_checkpoints/E_mol"    # Source file to copy
dst_file = os.path.join("01-data/01-data_traj/", "E_mol")      # Destination path
try:
    # Check if destination file already exists
    if os.path.exists(dst_file):
        pass
    else:
        # Check if source file exists
        if not os.path.isfile(src_file):
            raise FileNotFoundError(f"Source file {src_file} does not exist")
        # Copy file with metadata
        shutil.copy2(src_file, dst_file)  # Use copy2 to preserve metadata
except Exception as e:
    print(f"Operation failed: {str(e)}")

# Directory structure explanation
txt_sm = """Directory Structure:
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
"""
print(txt_sm)
