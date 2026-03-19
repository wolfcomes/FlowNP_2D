import os
import subprocess
import pandas as pd
from tqdm import tqdm  # Progress bar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
import re

# 配置日志
logging.basicConfig(
    filename='/share/home/grp-huangxd/zhangzhiyong/lead_optimization/PromptDiffModel/data/docking_results/docking_process.log',  # 日志文件位置
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# 1. 使用 smina 进行配体-受体对接并提取分数
def run_docking(ligand_sdf, receptor_pdb, ref_ligand_sdf,
                output_dir, file_prefix, mol_idx,
                num_poses=1, mode="dock"):
    """ 使用 smina 进行对接或打分，支持 dock/score/min 模式 """
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{file_prefix}_mol{mol_idx}_docked.pdb')
    log_file = os.path.join(output_dir, f'{file_prefix}_mol{mol_idx}_log.txt')

    # 基础命令
    smina_command = [
        'smina',
        '--receptor', receptor_pdb,
        '--ligand', ligand_sdf,
    ]

    if mode == "dock":
        smina_command += [
            '--num_modes', str(num_poses),
            '--autobox_ligand', ref_ligand_sdf,
            '--out', output_file,
            '--log', log_file
        ]
    elif mode == "score":
        smina_command += ['--score_only']
    elif mode == "min":
        # smina_command += ['--local_only', '--out', output_file]
        smina_command += ['--local_only']
    else:
        raise ValueError(f"未知 mode: {mode}, 应为 dock/score/min")

    try:
        result = subprocess.run(smina_command, check=True, capture_output=True, text=True)
        logging.info(f"对接命令执行成功: {' '.join(smina_command)}")

        # 解析亲和力
        if mode == "dock":
            with open(log_file, "r") as f:
                log_text = f.read()
            affinity = extract_affinity_from_log(log_text)
        else:
            affinity = extract_affinity_from_log(result.stdout)

        return affinity
    except subprocess.CalledProcessError as e:
        logging.error(f"对接失败: {e}\n命令: {' '.join(smina_command)}\n错误输出: {e.stderr}")
        return None


def extract_affinity_from_log(log_text):
    """从 smina 输出中提取最小亲和力分数"""
    # 1. 表格模式（dock）
    table_pattern = r"^\s*\d+\s+([-\d.]+)\s+[\d.]+\s+[\d.]+"
    matches = re.findall(table_pattern, log_text, re.MULTILINE)
    if matches:
        scores = [float(m) for m in matches]
        return min(scores)

    # 2. Affinity: -7.2 形式（score_only / local_only）
    affinity_pattern = r"Affinity:\s+(-?\d+\.\d+)"
    match = re.search(affinity_pattern, log_text)
    if match:
        return float(match.group(1))

    logging.warning("无法从日志中提取亲和力分数")
    return None


# 2. 处理单个SDF文件中的所有分子
def process_sdf_file(sdf_path, receptor_dir, ref_ligand_dir,
                     output_dir, file_prefix, num_poses=1, mode="dock"):
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
            num_poses,
            mode
        )

        # 删除临时文件
        os.remove(temp_sdf)

        if affinity is not None:
            results.append({
                'smiles': smiles,
                'file_prefix': file_prefix,
                'mol_index': mol_idx,
                'affinity': affinity,
                'sdf_file': os.path.basename(sdf_path),
                'mode': mode
            })

    return results


# 3. 批量运行对接任务（多线程）
def batch_docking(ligand_dir, receptor_dir, ref_ligand_dir,
                  output_dir, num_poses=1, max_threads=144, mode="dock"):
    """批量处理所有SDF文件"""
    os.makedirs(output_dir, exist_ok=True)

    sdf_files = [f for f in os.listdir(ligand_dir) if f.endswith('.sdf')]
    all_results = []

    with ThreadPoolExecutor(max_threads) as executor:
        futures = {}

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
                num_poses,
                mode
            )
            futures[future] = sdf_file

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

    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, f'docking_results_{mode}.csv')
        results_df.to_csv(csv_path, index=False)
        logging.info(f"保存对接结果到: {csv_path}")
        print(f"保存对接结果到: {csv_path}")

    return all_results


# 4. 根据文件前缀查找文件
def find_file_by_prefix(directory, prefix, extension):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if files:
        return os.path.join(directory, files[0])
    else:
        return None


# 示例使用
if __name__ == "__main__":
    ligand_dir = "results/generation_results"   # 包含SDF文件的目录
    # ligand_dir = "data/custom_protein/output_ligand"
    # ligand_dir = "data/baseline_ligands/PMDM"
    receptor_dir = "data/custom_protein/test"          # 包含PDB文件的目录
    ref_ligand_dir = "data/custom_protein/output_ligand"  # 参考配体目录
    output_dir = 'evaluation_results/score-only'      # 输出目录

    num_poses = 1
    mode = "score"   # 可选: "dock" / "score" / "min"

    results = batch_docking(ligand_dir, receptor_dir, ref_ligand_dir,
                            output_dir, num_poses, max_threads=144, mode=mode)
    print(f"所有步骤已完成！共处理 {len(results)} 个分子, mode={mode}")
