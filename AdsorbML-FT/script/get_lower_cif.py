#!/usr/bin/env python3
import os
import pandas as pd
import shutil  # Import the shutil module for file operations
from ase.io import read

def extract_last_step_energy(traj_file):
    """Read trajectory file and extract energy from the last step"""
    try:
        # Use index=-1 for efficiency if only the last frame is needed
        last_step = read(traj_file, index=-1)
        energy = last_step.get_potential_energy()
        return energy
    except Exception as e:
        print(f"Error reading {traj_file}: {e}")
        return None

def find_poscar_dir(current_dir='.'):
    """Find directory ending with '_poscar' in the current directory"""
    for item in os.listdir(current_dir):
        if os.path.isdir(os.path.join(current_dir, item)) and item.endswith('_poscar'):
            return item
    return None

def find_traj_dir(current_dir='.'):
    """Find directory ending with '_traj' in the current directory"""
    for item in os.listdir(current_dir):
        if os.path.isdir(os.path.join(current_dir, item)) and item.endswith('_traj'):
            return item
    return None

def main():
    current_directory = find_traj_dir(current_dir='.')
    traj_files = [f for f in os.listdir(current_directory) if f.endswith('.traj')]

    if not traj_files:
        print("No .traj files found in the current directory!")
        return

    data = []
    #print("Extracting energies...")
    for traj_file in traj_files:
        #print(traj_file,"traj_dir")
        energy = extract_last_step_energy(f"{current_directory}/{traj_file}")
        if energy is not None:
            data.append({'File': traj_file, 'Energy (eV)': energy})
        else:
            print(f"Could not extract energy from {traj_file}")

    if not data:
       print("No valid energy data could be extracted from any .traj file.")
       return


    df = pd.DataFrame(data)
    df['Base Name'] = df['File'].str.rsplit('_', n=1).str[0]
    df_sorted = df.sort_values(by=['Base Name', 'Energy (eV)'])
    df_min_energy = df.loc[df.groupby('Base Name')['Energy (eV)'].idxmin()]
    df_min_energy.to_csv('lowest_energy_files.csv', index=False)

    poscar_dir_name = find_poscar_dir("./")

    if not poscar_dir_name:
        print(f"Error: No directory ending with '_poscar' found in '{current_directory}'. Cannot copy files.")
        return

    source_dir_path = os.path.join(current_directory, poscar_dir_name)
    # Create new folder name
    lower_dir_name = poscar_dir_name.replace('_poscar', '_lower')
    #print(lower_dir_name)
    os.makedirs(lower_dir_name, exist_ok=True)

    # --- 5. Copy files ---
    copied_count = 0
    skipped_count = 0
    for index, row in df_min_energy.iterrows():
        min_energy_traj_file = row['File']
        # Replace .traj extension with .cif
        base_filename = os.path.splitext(min_energy_traj_file)[0]
        cif_filename = base_filename + '.cif' # Corrected logic to handle potential dots in base name

        source_file_path = f"{poscar_dir_name}/{cif_filename}"

        dest_file_path = f"{lower_dir_name}/{cif_filename}"

        # Check if source file exists
        if os.path.exists(source_file_path):
            try:
                shutil.copy2(source_file_path, dest_file_path) # copy2 preserves metadata
                #print(f"  Copied '{cif_filename}' from '{poscar_dir_name}' to '{lower_dir_name}'")
                copied_count += 1
            except Exception as e:
                #print(f"  Error copying {source_file_path} to {dest_file_path}: {e}")
                skipped_count += 1
        else:
            #print(f"  Warning: Source file '{cif_filename}' not found in '{source_dir_path}'. Skipping.")
            skipped_count += 1

    #print(f"\nFile copying finished. Copied: {copied_count}, Skipped (not found or error): {skipped_count}")


if __name__ == "__main__":
    main()