#!/usr/bin/env python3
import shutil
import os

txt_ml_incar ='''ml_incar说明
##########################################################

请确认ml_incar的参数!!!

使用说明：
1. 如果你要运行"ml_lmdb.py"，确认"ns", 默认为 3！

2. 如果你要运行"ml_train.py"，
    请确保只有1个lmdb，或者给出完整的lmdb路径！
    训练次数为30，批次为3

##########################################################
'''
def main():
    # 源文件路径
    src = "/home/zlb/wzy-shell/fairchem_checkpoints/ml_incar"
    
    # 目标路径（当前目录）
    dst = os.path.join(os.getcwd(), "ml_incar")
    
    try:
        # 执行拷贝操作
        shutil.copy2(src, dst)
        print(f"{txt_ml_incar}")
    except FileNotFoundError:
        print(f"错误：源文件 {src} 不存在")
    except PermissionError:
        print(f"错误：没有权限读取 {src} 或写入当前目录")
    except Exception as e:
        print(f"拷贝过程中发生错误: {e}")

if __name__ == "__main__":
    main()