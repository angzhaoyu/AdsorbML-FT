#!/usr/bin/env python3
import os
from ase.db import connect
import re
import torch
from ase.calculators.singlepoint import SinglePointCalculator
import pandas as pd
import numpy as np
import lmdb
from fairchem.core.preprocessing import AtomsToGraphs
from collections import OrderedDict
from ase.io import read
import sys
import shutil
import ast
from tqdm import tqdm
import pickle

np.random.seed(42)

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

def wirte_traj_to_db(file_path, db_path, ns):
    if os.path.exists(db_path):
        os.remove(db_path)
    traj_paths = find_files_with_extension(file_path, ".traj")
    with connect(db_path) as db:
        for traj_path in traj_paths:
            ad_name = get_filename_from_path(traj_path)
            if ns == -1:
                slab_s = read(traj_path,index= ns)
                db.write(slab_s,name=ad_name)
            else:
                ads_traj = read(traj_path,index=":")
                for ads_s in ads_traj[:ns]:
                    if np.max(np.abs(ads_s.get_forces())) > 2:
                        continue
                    db.write(ads_s,name=ad_name)
    return db_path

def print_db(db_path, save =True):
    with connect(db_path) as db:
        headers = ["id","name","formula","energy","natoms","fmax"]
        datas =[]
        for row in db.select(limit=10000):
            datas.append([row.id,row.name,row.formula,row.energy,row.natoms,row.fmax])
        df = pd.DataFrame(datas, columns=headers)
        print(df.head(5))
        if save:
            df.to_csv(f"./01-data/02-data_db/{get_filename_from_path(db_path)}.csv", index=False)

def get_ads_db(db_DFT_path, db_slab_path, E_ads_db_path, E_mol):
    if os.path.exists(E_ads_db_path):
        os.remove(E_ads_db_path)
    db_DFT = connect(db_DFT_path)
    db_slab = connect(db_slab_path)
    slab_names_energys = {}
    for slab_row in db_slab.select():
        slab_names_energys[slab_row.name] = slab_row.toatoms().get_potential_energy()


    with connect(E_ads_db_path) as E_ads_db:
        for DFT_row in db_DFT.select():
            DFT_names = DFT_row.name
            #print(DFT_names)
            DFT_atoms = DFT_row.toatoms()
            energy_dft = DFT_atoms.get_potential_energy()
            energy_slab = None
            for slab_name_,slab_energy_  in slab_names_energys.items():
                if  slab_name_  in DFT_names:
                    energy_slab = slab_energy_
            if energy_slab is None:
                raise Exception("Please provide slab information")

            energy_mol = None
            for mol_name_, mol_energy_ in E_mol.items():
                if  mol_name_  in DFT_names :
                    energy_mol = mol_energy_
            if energy_mol is None:
                raise Exception("Please provide E_mol information")

            energy_ads = energy_dft - energy_slab - energy_mol
            energy_mol = None
            energy_slab = None
            energy_dft = None
            if energy_ads > 10:
                continue
            calc = SinglePointCalculator(atoms=DFT_atoms, energy=energy_ads, forces =DFT_atoms.get_forces())
            DFT_atoms.calc = calc
            E_ads_db.write(DFT_atoms, name = DFT_names)

    return E_ads_db_path

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

def split_array(array, split):
    a = []
    number = split[0]+split[1]+split[2]
    num = len(array)//number+1

    if split[2] != 0:
        array1 = array[:-((split[1]+split[2])*num)]
        array2 = array[-((split[1]+split[2])*num):-(split[2]*num)]
        array3 = array[-(split[2]*num):]
        a.append(array1)
        a.append(array2)
        a.append(array3)
    else:
        array1 = array[:-(split[1]*num)]
        array2 = array[-((split[1]+split[2])*num):]
        array3 = []
        a.append(array1)
        a.append(array2)
        a.append(array3)
    return a

def write_lmdb(lmdb_path, data_a2g, sid_set, fid_set):
    memory_size = sys.getsizeof(data_a2g)
    lmdb1 = lmdb.open(
        "%s" % lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,)

    for id, data in tqdm(enumerate(data_a2g), total=len(data_a2g)):
        data.sid = torch.LongTensor([sid_set[id]])
        data.fid = torch.LongTensor([fid_set[id]])
        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A
        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue
        txn = lmdb1.begin(write=True)
        txn.put(f"{id}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = lmdb1.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(len(data_a2g), protocol=-1))
    txn.commit()

    lmdb1.sync()
    lmdb1.close()

    return lmdb_path

def get_sid_fid_from_db(db):
    names = []
    for row in db.select():
        names.append(row.name)
    names_list = list(OrderedDict.fromkeys(names))
    sid_set = []
    for j in names:
        for sid,l in enumerate(names_list):
            if j == l :
                sid_set.append(sid)
    n = 0
    a = 0
    fid_set=[]
    for id in range(len(sid_set)):
        if n == sid_set[id]:
            a += 1
        else:
            a = 1
        n = sid_set[id]
        fid_set.append(a-1)
    return sid_set ,fid_set

def shuft_a2g_sid_fid(a2g,sid,fid):
    random_indices = np.random.permutation(len(a2g))
    new_a2g = []
    new_sid = []
    new_fid = []
    for i in random_indices:
        new_a2g.append(a2g[i])
        new_sid.append(sid[i])
        new_fid.append(fid[i])
    return new_a2g , new_sid, new_fid


def write_lmdb_finally(db_path,split):
    lmdb_name = get_filename_from_path(db_path)
    os.makedirs("./01-data/03-data_lmdb/"+lmdb_name+"/train/", exist_ok=True)
    train_path = ("./01-data/03-data_lmdb/"+lmdb_name +"/train/"+lmdb_name+"train.lmdb")
    os.makedirs("./01-data/03-data_lmdb/"+lmdb_name+"/val/", exist_ok=True)
    val_path = ("./01-data/03-data_lmdb/"+lmdb_name+"/val/" +lmdb_name+"val.lmdb")

    if split[2] == 0:
        pass
    else:
        os.makedirs("./01-data/03-data_lmdb/"+lmdb_name+"/test/", exist_ok=True)
        test_path = ("./01-data/03-data_lmdb/"+lmdb_name+"/test/"+lmdb_name+"test.lmdb")

    db = connect(db_path)
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,     # False for test data
        r_forces=True,     # False for test data
        r_distances=True,
        r_fixed=True,
    )

    data_a2g = a2g.convert_all(db, disable_tqdm=True)
    sid_set, fid_set = get_sid_fid_from_db(db)


    data_a2g, sid_set, fid_set = shuft_a2g_sid_fid(data_a2g, sid_set, fid_set)

    train_a2g, val_a2g, test_a2g = split_array(data_a2g,split)
    train_sid, val_sid, test_sid = split_array(sid_set,split)
    train_fid, val_fid, test_fid = split_array(fid_set,split)

    train_a2g, train_sid, train_fid = shuft_a2g_sid_fid(train_a2g, train_sid, train_fid)
    val_a2g, val_sid, val_fid = shuft_a2g_sid_fid(val_a2g, val_sid, val_fid)
    test_a2g, test_sid, test_fid = shuft_a2g_sid_fid(test_a2g, test_sid, test_fid)

    train_ = write_lmdb(train_path, train_a2g, train_sid, train_fid)
    val_ = write_lmdb(val_path, val_a2g, val_sid, val_fid )
    if split[2] == 0 :
        pass
    else:
        test_ = write_lmdb(test_path, test_a2g, test_sid, test_fid)

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        raise ValueError(f"Unable to parse string: {s}")

def parse_value(s):
    return float(eval(s.strip()))

def parse_energy_file_to_dict(file_path=None, text_content=None):
    if file_path is None and text_content is None:
        raise ValueError("Must provide either file_path or text_content")

    if file_path:
        with open(file_path, 'r') as f:
            text_content = f.read()

    energy_dict = {}
    for line in text_content.split('\n'):
        if not line.strip():
            continue
        for pair in line.split(','):
            if '=' not in pair:
                continue
            key, value_str = pair.split('=', 1)
            energy_dict[key.strip()] = parse_value(value_str)
    return energy_dict

def remove_f(folder_path):
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Folder '{folder_path}' has been deleted")
            except Exception as e:
                print(f"Deletion failed: {e}")
    else:
        print(f"Folder '{folder_path}' does not exist")
def remove_path(target_path):
    """Delete a file or folder"""
    if os.path.isfile(target_path):
        os.remove(target_path)  # Delete file
    elif os.path.isdir(target_path):
        shutil.rmtree(target_path)  # Delete folder


if __name__ == "__main__":
    # Get command line arguments
    os.makedirs(f"01-data/02-data_db", exist_ok=True)
    os.makedirs(f"01-data/03-data_lmdb", exist_ok=True)
    E_mol_path ='./01-data/01-data_traj/E_mol'
    E_mol= parse_energy_file_to_dict(file_path =E_mol_path)
    required_parameters = ['ns', "save_csv", "DFT_db","split"]
    try:
      ml_incar = parse_config('ml_incar', required_parameters)
    except FileNotFoundError:
      print('Please run "ml_incar.py" first to generate the configuration file!')
      sys.exit(1)
    try:
        ns = int(ml_incar['ns'])
        save_csv = ml_incar['save_csv']
        DFT_db = ml_incar['DFT_db']
        split = ml_incar['split']
        split = string_to_list(split)

    except (ValueError, TypeError) as e:
      raise ValueError(f"Training parameters missing!")

    folders = [entry.name for entry in os.scandir('./01-data/01-data_traj/') if entry.is_dir()]

    if len(find_files_with_extension('./01-data/01-data_traj/slab_traj', ".traj")) == 0 :
        raise Exception("Please provide slab or adsorption trajectory files")

    slab_file_paths = []
    ads_file_paths = []
    for traj_file_paths in folders:
        if "slab" in traj_file_paths:
            slab_file_paths.append(traj_file_paths)
        else:
            ads_file_paths.append(traj_file_paths)
    if len(slab_file_paths) < 1:
        raise Exception("Please provide slab information")

    for ads_file_path in ads_file_paths:
        slab_db_path ="./01-data/02-data_db/" + slab_file_paths[0]+f"_ns{ns}.db"
        DFT_db_path ="./01-data/02-data_db/" + "DFT_" +get_filename_from_path(ads_file_path)[:-5]+f"_ns{ns}.db"
        ads_db_path ="./01-data/02-data_db/" + get_filename_from_path(ads_file_path)[:-5]+f"_ns{ns}.db"
        slab_db_path = wirte_traj_to_db(f"./01-data/01-data_traj/{slab_file_paths[0]}", slab_db_path, -1)
        DFT_db_path = wirte_traj_to_db(f"./01-data/01-data_traj/{ads_file_path}", DFT_db_path,ns)
        E_ads_db_path = get_ads_db(DFT_db_path, slab_db_path, ads_db_path,  E_mol)


        if DFT_db == "False" :
          remove_path(slab_db_path)
          remove_path(DFT_db_path)
        if save_csv == "True":
          save_csv  =  True
        print_db(E_ads_db_path,save= save_csv)
        write_lmdb_finally(E_ads_db_path,split)

        txt_sm = """If you see progress bars, it means the script ran successfully!  \nNext step: Run "ml_train.py" """
        print(txt_sm)


