import os
import shutil
from pathlib import Path

def extract_first_sdf_with_prefix(source_dir, target_dir):
    """
    从源目录的每个子目录中提取第一个sdf文件，并以子目录名作为前缀保存到目标目录
    
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源目录下的所有子目录
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        
        # 确保是目录而不是文件
        if os.path.isdir(subdir_path):
            # 查找子目录中的所有sdf文件
            sdf_files = list(Path(subdir_path).rglob("*.sdf"))
            
            if sdf_files:
                # 按文件名排序并取第一个
                sdf_files.sort()
                first_sdf = sdf_files[0]
                
                # 构建源文件和目标文件路径
                src_file = first_sdf
                # dst_file = os.path.join(target_dir, f"{subdir}.sdf")
                dst_file = os.path.join(target_dir, first_sdf.name)
                
                # 复制文件
                shutil.copy2(src_file, dst_file)
                print(f"已复制: {src_file} -> {dst_file}")
            else:
                print(f"子目录 {subdir} 中没有找到sdf文件")

# 使用示例
source_directory = "data/baseline_ligands/gen"  # 替换为你的源目录
target_directory = "data/baseline_ligands/PMDM"  # 替换为你想要保存的目标目录

extract_first_sdf_with_prefix(source_directory, target_directory)