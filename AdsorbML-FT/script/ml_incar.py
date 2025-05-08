#!/usr/bin/env python3
"""
Generate ml_incar configuration file.
This script copies the ml_incar template file to the current directory.
"""
import shutil
import os

txt_ml_incar = '''ml_incar Instructions
##########################################################

Please verify the ml_incar parameters!!!

Usage Instructions:
1. If you want to run "ml_lmdb.py", confirm "ns", default is 3!

2. If you want to run "ml_train.py",
    Make sure you have only 1 lmdb, or provide the complete lmdb path!
    Training epochs: 30, batch size: 3

##########################################################
'''

def main():
    # Source file path
    src = "/home/zlb/work/wzy/AdsorbML-FT/AdsorbML-FT/script/ml_incar"

    # Destination path (current directory)
    dst = os.path.join(os.getcwd(), "ml_incar")

    try:
        # Execute copy operation
        shutil.copy2(src, dst)
        print(f"{txt_ml_incar}")
    except FileNotFoundError:
        print(f"Error: Source file {src} does not exist")
    except PermissionError:
        print(f"Error: No permission to read {src} or write to current directory")
    except Exception as e:
        print(f"Error during copy operation: {e}")

if __name__ == "__main__":
    main()