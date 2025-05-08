#!/usr/bin/env python3

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
import ase.io
from ase.io import read
from ase.io import write
from ase.optimize import BFGS
import json
from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
import os
from glob import glob
import pandas as pd
from fairchem.data.oc.utils import DetectTrajAnomaly
from fairchem.data.oc.utils.vasp import write_vasp_input_files
from ase.constraints import FixAtoms
# Optional - see below
import numpy as np
from dscribe.descriptors import SOAP
from scipy.spatial.distance import pdist, squareform
from x3dase.visualize import view_x3d_n
import re
from pathlib import Path
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

class Slab_new:
    def __init__(
        self,
        slab_atoms = None):
        self.atoms = slab_atoms
        if np.any(self.atoms.get_tags() == 1):
            pass
        else:
            self.atoms.set_tags(1)
            #print("yous tags is 1")

    def has_surface_tagged(self):
        return np.any(self.atoms.get_tags() == 1)

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

def get_filename_from_path(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    string_without_dot = re.sub(r'\.+$', '', file_name)
    return string_without_dot 

def initial_constraints(atoms, n):
    atom_positions = atoms.get_positions()
    z_positions = [pos[2] for pos in atom_positions]
    sorted_indices = sorted(enumerate(z_positions), key=lambda x: x[1])
    min_32_indices = [index for index, value in sorted_indices[:n]]
    fix_constraint = FixAtoms(indices=min_32_indices)
    atoms.set_constraint(fix_constraint)
    return atoms
    
def read_cif(path):
    with open(path, 'r') as file:
        if "_cell_" in file.read():
            atoms = read(path,)
        else:
            atoms = read(path, format= "vasp")
    return atoms

def plt_cif(slab):
    fig, axs = plt.subplots(1, 2)
    plot_atoms(slab, axs[0]);
    plot_atoms(slab, axs[1], rotation=('-90x'))
    axs[0].set_axis_off()
    axs[1].set_axis_off()

def deduplicate(configs_for_deduplication: list,
                adsorbate_binding_index: int,
                cosine_similarity = 1e-3,
               ):  
    energies_for_deduplication = np.array([atoms.get_potential_energy() for atoms in configs_for_deduplication])
    soap = SOAP(
        species=np.unique(configs_for_deduplication[0].get_chemical_symbols()),
        r_cut = 2.0,
        n_max=6,
        l_max=3,
        periodic=True,
    )
    ads_len = list(configs_for_deduplication[0].get_tags()).count(2)
    position_idx = -1*(ads_len-adsorbate_binding_index)
    soap_desc = []
    for config in configs_for_deduplication:
        soap_ex = soap.create(config, centers=[position_idx])
        soap_desc.extend(soap_ex)
    soap_descs = np.vstack(soap_desc)
    distance = squareform(pdist(soap_descs, metric="cosine"))
    bool_matrix = np.where(distance <= cosine_similarity, 1, 0)
    idxs_to_keep = []
    pass_idxs = []
    for idx, row in enumerate(bool_matrix):
        if idx in pass_idxs:
            continue    
        elif sum(row) == 1:
            idxs_to_keep.append(idx)
        else:
            same_idxs = [row_idx for row_idx, val in enumerate(row) if val == 1]
            pass_idxs.extend(same_idxs)
            min_e = min(energies_for_deduplication[same_idxs])
            idxs_to_keep.append(list(energies_for_deduplication).index(min_e))
    return idxs_to_keep

def get_absorbate(adsorbate_smiles):
    #adsorbate_path = Path(ocdata.__file__).parent / Path('databases/pkls/adsorbates.pkl')
    adsorbate = Adsorbate(adsorbate_smiles_from_db=adsorbate_smiles)
    return adsorbate

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

def get_output(num_n):
    traj_paths = find_files_with_extension("./00-modeling/02-traj/", "traj")
    file_path = "./00-modeling/00-slab_mol"
    os.makedirs(f"./00-modeling/03-poscar", exist_ok=True)
    os.makedirs(f"./00-modeling/04-low{num_n}_poscar", exist_ok=True)
    mole_file_path = file_path+"/mol_cif"
    slab_file_path = file_path+"/slab_cif"
    mole_paths = find_files_with_extension(mole_file_path,extension="cif")
    slab_paths = find_files_with_extension(slab_file_path,extension="cif")
    results = []
    for traj_path in traj_paths:
        traj_name = get_filename_from_path(traj_path)
        traj = read(traj_path, ":")
        initial_atoms = traj[0]
        final_atoms = traj[-1]
        write(f"./00-modeling/03-poscar/{traj_name}.cif", final_atoms ,format="vasp")
        atom_tags = initial_atoms.get_tags()
        detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
        anom = (
            detector.is_adsorbate_dissociated()
            or detector.is_adsorbate_desorbed()
            or detector.has_surface_changed()
            or detector.is_adsorbate_intercalated()
        )
        rx_energy = traj[-1].get_potential_energy()
        results.append({"file_id": traj_name,"slab_id": "_".join(traj_name.split("_")[:-2]),"ads_id": traj_name.split("_")[-2], "relaxed_atoms": traj[-1],
                        "relaxed_energy_ml": rx_energy, "anomolous": anom})
        df = pd.DataFrame(results)
        
        df = df[~df.anomolous].copy().reset_index()
        df = df.sort_values(by=["slab_id" ,"ads_id","relaxed_energy_ml"])
        
    for slab_path in slab_paths:
        slab_name = get_filename_from_path(slab_path)
        for mole_path in mole_paths:
            mol_name = get_filename_from_path(mole_path) 
            adsorbate = get_absorbate("*OH")
            mole = read(mole_path)
            mole.cell = [0,0,0]
            adsorbate.atoms = mole
            df_slab = df[df["slab_id"] == slab_name]
            df_slab_ads = df_slab[df_slab["ads_id"] == mol_name]
            configs_for_deduplication =  df_slab_ads.relaxed_atoms.tolist()
            idxs_to_keep = deduplicate(configs_for_deduplication,adsorbate.binding_indices[0])
            end_df = df_slab_ads.iloc[idxs_to_keep]
            low_e_values = np.round(end_df.sort_values(by = "relaxed_energy_ml").relaxed_energy_ml.tolist()[0:num_n],3)
            configs_for_dft = end_df.sort_values(by = "relaxed_energy_ml").relaxed_atoms.tolist()[0:num_n]
            config_idxs = end_df.sort_values(by = "relaxed_energy_ml").file_id.tolist()[0:num_n]
            for idx, config in enumerate(configs_for_dft):
                write(f"./00-modeling/04-low{num_n}_poscar/{config_idxs[idx]}.cif", config ,format="vasp")

if __name__ == "__main__":
    required_parameters = ['checkpoint_path', "num_sites",'min_constraints','tags0', 'tags1','opt_fmax', "opt_steps", "num_cf", "num_n",'calc_moldes','supply_traj','supply_model']
    try:
        ml_incar = parse_config('ml_incar', required_parameters)
    except FileNotFoundError:
        print('请先运行 "ml_incar.sh" 生成配置文件！')
        sys.exit(1)    
    try:
        checkpoint_path = str(ml_incar['checkpoint_path'])
        num_sites = int(ml_incar['num_sites'])
        num_n = int(ml_incar['num_n'])
        min_constraints = int(ml_incar['min_constraints'])
        opt_fmax = float(ml_incar['opt_fmax'])
        num_cf = int(ml_incar['num_cf'])
        opt_steps = int(ml_incar['opt_steps'])
        supply_traj = str(ml_incar['supply_traj'])
        supply_model = str(ml_incar['supply_model'])
        calc_moldes = str(ml_incar['calc_moldes'])
        tags0 = int(ml_incar.get('tags0'))
        tags1 = int(ml_incar.get('tags1'))

    except (ValueError, TypeError) as e:
      raise ValueError(f"训练参数缺失！")
    os.makedirs(f"00-modeling/01-ads_slab", exist_ok=True)
    os.makedirs(f"00-modeling/02-traj", exist_ok=True)
    os.makedirs(f"00-modeling/03-poscar", exist_ok=True)
    if supply_traj == "True":
        get_output(num_n)
    if supply_traj == "False":
        file_path = "./00-modeling/00-slab_mol"
        mole_file_path = file_path+"/mol_cif"
        slab_file_path = file_path+"/slab_cif"
        mole_paths = find_files_with_extension(mole_file_path,extension="cif")
        slab_paths = find_files_with_extension(slab_file_path,extension="cif")
        if supply_model == "False" :
            filename_adslabs = {}
            for slab_path in slab_paths:
                slab = read_cif(slab_path)
                slab = initial_constraints(slab, min_constraints)
                slab_new = Slab_new(slab_atoms=slab)
                slab_name = get_filename_from_path(slab_path)

                for mole_path in mole_paths:
                    adsorbate = get_absorbate("*OH")
                    mol_name = get_filename_from_path(mole_path)
                    mole = read(mole_path)
                    mole.cell = [0,0,0]
                    adsorbate.atoms = mole
                    heuristic_adslabs = AdsorbateSlabConfig(slab_new, adsorbate, mode="heuristic")
                    random_adslabs = AdsorbateSlabConfig(slab_new, adsorbate, mode="random_site_heuristic_placement", num_sites = num_sites)
                    adslabs = [*heuristic_adslabs.atoms_list, *random_adslabs.atoms_list]
                    for j,i in enumerate(adslabs):
                        filename_adslabs[f'{slab_name+"_"+mol_name+"_"+str(j)}'] = i

                        write(f"./00-modeling/01-ads_slab/{slab_name+"_"+mol_name+"_"+str(j)}.cif", i ,format="vasp")
                        if j > num_cf:
                            break
    
        if supply_model == "True" :
            ads_paths = find_files_with_extension("./00-modeling/01-ads_slab/",extension="cif")
            filename_adslabs = {}
            for ads_path in ads_paths:
                ads_atoms = read_cif(ads_paths)
                ads_atoms = initial_constraints(ads_atoms, min_constraints)
                ads_atoms = set_tags_from_constraints(ads_atoms, tags0,tags1)
                ads_name = get_filename_from_path(ads_path)
                filename_adslabs[f'{slab_name+"_"+mol_name+"_"+str(j)}'] = ads_atoms

        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
        if calc_moldes == "False":
            pass
        if calc_moldes == "True":
            for filename, adslab in filename_adslabs.items():
                adslab.calc = calc
                adslab = initial_constraints(adslab, min_constraints)
                adslab = set_tags_from_constraints(adslab, tags0,tags1)
                opt = BFGS(adslab, trajectory=f"./00-modeling/02-traj/{filename}.traj")
                opt.run(fmax=opt_fmax, steps=opt_steps)            
        get_output(num_n)