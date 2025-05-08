#!/usr/bin/env python3

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
import os
from ase.io import read,write
from ase.optimize import BFGS
#from ase.visualize import view
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.constraints import FixAtoms
import numpy as np
import re
import sys

def find_files_with_extension(directory, extension):
    found_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    if found_files:
        pass
        #print("Found the following files:")
    else:
        print(f"No files with {extension_to_find} extension found.")
    return found_files

def read_cif(path):
    with open(path, 'r') as file:
        if "_cell_" in file.read():
            atoms = read(path,)
        else:
            atoms = read(path, format= "vasp")
    return atoms

def get_filename_from_path(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    string_without_dot = re.sub(r'\.+$', '', file_name)
    return string_without_dot

def find_folders_with_extension(foler_path,extension):
    cif_folders = []
    folders = [entry.name for entry in os.scandir(foler_path) if entry.is_dir()]
    for folder in folders :
        l = len(extension)
        if extension == folder[-l:]:
            cif_folders.append(folder)
    return cif_folders

def plt_cif(slab):
    fig, axs = plt.subplots(1, 2)
    plot_atoms(slab, axs[0]);
    plot_atoms(slab, axs[1], rotation=('-90x'))
    axs[0].set_axis_off()
    axs[1].set_axis_off()

def initial_constraints(atoms, n):
    atom_positions = atoms.get_positions()
    z_positions = [pos[2] for pos in atom_positions]
    sorted_indices = sorted(enumerate(z_positions), key=lambda x: x[1])
    min_32_indices = [index for index, value in sorted_indices[:n]]
    fix_constraint = FixAtoms(indices=min_32_indices)
    atoms.set_constraint(fix_constraint)
    return atoms

def set_tags_from_constraints(atoms, tags0, tags1):
    assert atoms.constraints
    atom_positions = atoms.get_positions()
    z_positions = [pos[2] for pos in atom_positions]
    sorted_indices = sorted(enumerate(z_positions), key=lambda x: x[1])
    tags2_indices = [index for index, value in sorted_indices[tags1:]]
    tags1_indices = [index for index, value in sorted_indices[tags0:tags1]]
    tags0_indices = [index for index, value in sorted_indices[:tags0]]
    num_atoms = len(atoms)
    tags = np.ones(num_atoms, dtype=int)
    tags[tags2_indices] = 2
    tags[tags1_indices] = 1
    tags[tags0_indices] = 0
    atoms.set_tags(tags)
    return atoms


def parse_config(file_path, required_vars):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            if key in required_vars:
                value = value.split('#')[0].strip()
                config[key] = value
    return config

if __name__ == "__main__":
    required_parameters = ['checkpoint_path', "cpu",'tags0', 'tags1','opt_fmax', "opt_steps", 'min_constraints']

    try:
      ml_incar = parse_config('ml_incar', required_parameters)
    except FileNotFoundError:
      print('Please run "ml_incar.py" first to generate the configuration file!')
      sys.exit(1)


    checkpoint_path = str(ml_incar.get('checkpoint_path').strip('"\''))
    cpu  = {'true': True, 'false': False}.get(ml_incar.get('cpu').lower(), False)
    tags0 = int(ml_incar.get('tags0'))
    tags1 = int(ml_incar.get('tags1'))
    opt_fmax = float(ml_incar.get('opt_fmax'))
    opt_steps = int(ml_incar.get('opt_steps'))
    min_constraints = int(ml_incar.get('min_constraints'))
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu= cpu)
    cif_folders = find_folders_with_extension("./" , "_cif")
    for cif_folder in cif_folders:
        cif_paths = find_files_with_extension(cif_folder, ".cif")
        name_begin = cif_folder[:-4]
        os.makedirs(f"{name_begin}_traj", exist_ok=True)
        os.makedirs(f"{name_begin}_poscar", exist_ok=True)
        for cif_path in cif_paths:
            cif_name = get_filename_from_path(cif_path)
            ads_slab = read_cif(cif_path, )
            ads_slab = initial_constraints(ads_slab, min_constraints)
            ads_slab = set_tags_from_constraints(ads_slab, tags0,tags1)
            ads_slab.calc = calc
            opt = BFGS(ads_slab, trajectory=f"{name_begin}_traj/{cif_name}.traj")
            opt.run(fmax=opt_fmax, steps=opt_steps)
            write(f"{name_begin}_poscar/{cif_name}.cif", ads_slab ,format="vasp")