import os
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm  # Progress bar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# 配置日志
logging.basicConfig(
    filename='/share/home/grp-huangxd/zhangzhiyong/lead_optimization/PromptDiffModel/data/docking_results/docking_process.log',  # 日志文件位置
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
)

# 1. 使用 smina 进行配体-受体对接并提取分数
def run_docking(ligand_sdf, receptor_pdb, ref_ligand_sdf, output_dir, file_prefix, mol_idx, num_poses=1):
    """ 使用 smina 进行对接，生成对接构象并提取分数 """
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(output_dir, f'{file_prefix}_mol{mol_idx}_docked.pdb')
    
    smina_command = [
        'smina',
        '--receptor', receptor_pdb,
        '--ligand', ligand_sdf,
        '--num_modes', str(num_poses),
        '--autobox_ligand', ref_ligand_sdf,
        '--out', output_file,
        '--log', os.path.join(output_dir, f'{file_prefix}_mol{mol_idx}_log.txt'),
        '--score_only'
    ]
    
    try:
        result = subprocess.run(smina_command, check=True, capture_output=True, text=True)
        logging.info(f"对接命令执行成功: {' '.join(smina_command)}")

        # 从日志中提取分数
        affinity = extract_affinity_from_log(result.stdout)
        return affinity
    except subprocess.CalledProcessError as e:
        logging.error(f"对接失败: {e}\n命令: {' '.join(smina_command)}\n错误输出: {e.stderr}")
        return None

def extract_affinity_from_log(log_text):
    """从smina日志中提取最小亲和力分数（多个构象时取最小值）"""
    # 匹配表格中的亲和力分数行（包含多构象情况）
    table_pattern = r"^\s*\d+\s+([-\d.]+)\s+[\d.]+\s+[\d.]+"
    matches = re.findall(table_pattern, log_text, re.MULTILINE)

    
    if matches:
        # 将匹配值转为浮点数并返回最小值
        scores = [float(m) for m in matches]
        return min(scores)
    
    
    logging.warning("无法从日志中提取亲和力分数")
    return None


# 2. 处理单个SDF文件中的所有分子
def process_sdf_file(sdf_path, receptor_dir, ref_ligand_dir, output_dir, file_prefix, num_poses=1):
    """处理单个SDF文件中的所有分子"""
    results = []
    
    # 查找受体和参考配体文件
    receptor_pdb = find_file_by_prefix(receptor_dir, file_prefix, '.pdb')
    ref_ligand_sdf = find_file_by_prefix(ref_ligand_dir, file_prefix, '.sdf')

    
    if not receptor_pdb or not ref_ligand_sdf:
        logging.warning(f"未找到受体或参考配体文件: {file_prefix}")
        return results
    
    # 读取SDF文件中的所有分子
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol_idx, mol in enumerate(suppl):
        if mol is None:
            continue
        
        smiles = Chem.MolToSmiles(mol) if mol else ''
            
        # 创建临时SDF文件用于当前分子
        temp_sdf = os.path.join(output_dir, f'temp_{file_prefix}_mol{mol_idx}.sdf')
        writer = Chem.SDWriter(temp_sdf)
        writer.write(mol)
        writer.close()
        
        # 运行对接
        affinity = run_docking(
            temp_sdf, 
            receptor_pdb, 
            ref_ligand_sdf, 
            output_dir,
            file_prefix,
            mol_idx,
            num_poses
        )
        
        # 删除临时文件
        os.remove(temp_sdf)
        
        if affinity is not None:
            results.append({
                'smiles':smiles,
                'file_prefix': file_prefix,
                'mol_index': mol_idx,
                'affinity': affinity,
                'sdf_file': os.path.basename(sdf_path)
            })
    
    return results

# 3. 批量运行对接任务（多线程）
def batch_docking(ligand_dir, receptor_dir, ref_ligand_dir, output_dir, num_poses=1, max_threads=144):
    """批量处理所有SDF文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有SDF文件
    sdf_files = [f for f in os.listdir(ligand_dir) if f.endswith('.sdf')]
    all_results = []
    
    # 创建线程池
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {}
        
        # 提交所有任务
        for sdf_file in sdf_files:
            sdf_path = os.path.join(ligand_dir, sdf_file)
            file_prefix = '_'.join(sdf_file.split('_')[:-1])  # 移除序号部分
            
            future = executor.submit(
                process_sdf_file,
                sdf_path,
                receptor_dir,
                ref_ligand_dir,
                output_dir,
                file_prefix,
                num_poses
            )
            futures[future] = sdf_file
        
        # 使用tqdm进度条
        with tqdm(total=len(futures), desc="Processing SDF Files", unit="file") as pbar:
            for future in as_completed(futures):
                sdf_file = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    pbar.update(1)
                    pbar.set_postfix(file=sdf_file, results=len(results))
                except Exception as e:
                    logging.error(f"处理文件失败: {sdf_file}, 错误: {e}")
                    pbar.update(1)
    
    # 保存结果到CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'docking_results.csv')
        results_df.to_csv(csv_path, index=False)
        logging.info(f"保存对接结果到: {csv_path}")
        print(f"保存对接结果到: {csv_path}")
    
    return all_results

# 4. 根据文件前缀查找文件
def find_file_by_prefix(directory, prefix, extension):
    """根据文件前缀和扩展名查找文件"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if files:
        return os.path.join(directory, files[0])  # 返回找到的第一个文件路径
    else:
        return None

# 示例使用
if __name__ == "__main__":
    # 定义输入输出目录
    ligand_dir = "data/custom_protein/output_ligand"  # 包含SDF文件的目录
    receptor_dir = "data/custom_protein/test"  # 包含PDB文件的目录
    ref_ligand_dir = "data/custom_protein/output_ligand"  # 包含参考配体SDF文件的目录
    output_dir = 'evaluation_results/docking_ref'  # 输出目录
    
    num_poses = 1  # 每个分子生成1个对接构象
    
    # 运行批量对接
    results = batch_docking(ligand_dir, receptor_dir, ref_ligand_dir, output_dir, num_poses)
    print(f"所有步骤已完成！共处理 {len(results)} 个分子")