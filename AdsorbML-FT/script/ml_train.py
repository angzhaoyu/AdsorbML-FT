#!/usr/bin/env python3

from fairchem.core.datasets import LmdbDataset
import numpy as np
from fairchem.core.common.tutorial_utils import generate_yml_config
import time
import re
from fairchem.core.common.tutorial_utils import fairchem_main
import sys
import subprocess
import os
import torch
import shutil
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
    required_parameters = ['checkpoint_path', 'src',  "yml", 'max_epochs', 'batch_size','eval_batch_size']
    try:
      ml_incar = parse_config('ml_incar', required_parameters)
    except FileNotFoundError:
      print('Please run "ml_incar.py" first to generate the configuration file!')
      sys.exit(1)

    try:
      checkpoint_path = str(ml_incar['checkpoint_path'])
      src = str(ml_incar['src'])
      yml = str(ml_incar['yml'])
      max_epochs = int(ml_incar['max_epochs'])
      batch_size = int(ml_incar['batch_size'])
      eval_batch_size = int(ml_incar['eval_batch_size'])
    except (ValueError, TypeError) as e:
      raise ValueError(f"Training parameters missing!")

    if len(find_files_with_extension("./",".lmdb")) == 0:
      raise ValueError(f"Please provide LMDB database!")

    if len([entry.name for entry in os.scandir(src) if entry.is_dir() and entry.name == "train"]) > 0:
      pass
    else:
        src = f"{src}/{[entry.name for entry in os.scandir(src) if entry.is_dir()][0]}"

    # Get command line arguments
    element_c = "bs"+str(batch_size)+"_"+ "max"+str(max_epochs) +"_"
    train_src = src + "/train"
    val_src =  src + "/val"
    FT_folder_name = f"{get_filename_from_path(src)}"
    id_name = get_filename_from_path(checkpoint_path)
    os.makedirs(f"./02-FT/{FT_folder_name}", exist_ok=True)
    config_path = f"./02-FT/{FT_folder_name}/{element_c}{id_name}.yml"
    train_txt = f"./02-FT/{FT_folder_name}/{element_c}{id_name}.txt"
    train_dataset = LmdbDataset({"src": train_src})
    energies = []
    forces = []
    for data in train_dataset:
      energies.append(data.energy)
      for i in range(3):
        for j in range(len(data.forces)):
          forces.append(data.forces[j][i])

    target_mean = np.mean(energies)
    target_std = np.std(energies)
    grad_target_mean = np.mean(np.array(forces))
    grad_target_std = np.std(np.array(forces))
    len_train = len(train_dataset)
    target_mean,grad_target_mean,grad_target_std,len_train

    if yml == "False":
      yml = generate_yml_config(checkpoint_path, config_path,
                        delete=['slurm', 'cmd', 'logger',  'task', 'model_attributes','optim.load_balancing',
                                'dataset', 'test_dataset', 'val_dataset'],
                        update={'gpus': 1,
                                'task.dataset': 'lmdb',
                                'optim.eval_every': int(f'{len_train}'),
                                'optim.max_epochs': int(f'{max_epochs}'),
                                'optim.num_workers' : 0,
                                #'logger' :  'tensorboard',
                                'optim.batch_size': int(f'{batch_size}') ,
                                'optim.eval_batch_size' : int(f'{eval_batch_size}'),
                                # Train data
                                'dataset.train.src': "%s" % train_src,
                                'dataset.train.normalize_labels': True,
                                'dataset.train.target_mean': float(f'{target_mean}'),
                                'dataset.train.target_std': float(f'{target_std}'),
                                'dataset.train.grad_target_mean': float(f'{grad_target_mean}'),
                                'dataset.train.grad_target_std': float(f'{grad_target_mean}'),
                                # val data
                                'dataset.val.src': "%s" % val_src,
                                })


    t0 = time.time()
    command = f"python {fairchem_main()} --mode train --config-yml {yml} --checkpoint {checkpoint_path}  --run-dir fine-tuning  --identifier {id_name} --amp > {train_txt} 2>&1 "
    subprocess.run(command, shell=True, check=True)
    print(f'Elapsed time = {time.time() - t0:1.1f} seconds')
    with open('%s' % train_txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        if "checkpoint_dir:" in line:
            parts = line.split(':')
            cpdir = parts[-1].strip()
            break
    checkpoint_cpdir = cpdir + "/checkpoint.pt"
    modified_model_path = f"./02-FT/{FT_folder_name}/{element_c}{id_name}.pt"
    model = torch.load(checkpoint_cpdir)
    del model['scheduler']
    del model['epoch']
    del model['step']
    model['config']['trainer'] = 'equiformerv2_forces'
    torch.save(model, modified_model_path)
    folder_path = "fine-tuning"
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                #print(f"Folder '{folder_path}' has been deleted")
            except Exception as e:
                print(f"Deletion failed: {e}")
    else:
        print(f"Folder '{folder_path}' does not exist")