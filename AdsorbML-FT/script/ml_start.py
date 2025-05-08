#!/usr/bin/env python3

import os
from pathlib import Path
import shutil


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

src_file = "/home/zlb/wzy-shell/fairchem_checkpoints/E_mol"    # 要复制的源文件
dst_file = os.path.join("01-data/01-data_traj/", "E_mol")  # 目标路径（当前目录下的a.txt）
try:
    # 检查目标文件是否已存在
    if os.path.exists(dst_file):
        pass
    else:
        # 检查源文件是否存在
        if not os.path.isfile(src_file):
            raise FileNotFoundError(f"源文件 {src_file} 不存在")
        # 执行文件复制
        shutil.copy2(src_file, dst_file)  # 使用copy2保留元数据        
except Exception as e:
    print(f"操作失败：{str(e)}")

txt_sm =  """文件夹结构说明：
##########################################################

├── 00-modeling
│   ├── 00-slab_mol/mol_cif #小分子
│   └── 00-slab_mol/slab_cif #板
│
├── 01-data_traj/        # 存放轨迹数据
│   ├── *_traj/          # 存放吸附结构轨迹文件
│   ├── slab_traj/       # 存放催化剂表面轨迹文件
│   └── E_mol            # 放入吸附小分子
│
├── 02-FT/                   # 储存微调结果
└── 03-opt/                  # 优化结构

使用说明：
0. 在01-data_traj中：
   - 在mol_cif中放入吸附小分子
   - 在slab_cif中放入催化剂

1. 在01-data_traj中：
   - 在"*_traj"文件夹中放入"吸附结构轨迹"文件
   - 在"slab_traj"文件夹中放入"催化剂表面"轨迹文件
   - 在"E_mol"中放入小分子能量

2. 下一步得到数据库：运行"ml_incar.sh" 和 "ml_lmdb.py"。

##########################################################
"""
print(txt_sm)
