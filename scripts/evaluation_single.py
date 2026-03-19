import os
import subprocess
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen
import re
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
# 尝试导入自定义SA_Score
SA_score_dir = os.path.join(script_dir, "SA_Score")
sys.path.append(SA_score_dir)
try:
    import sascorer
    SA_AVAILABLE = True
    print("成功导入自定义SA_Score模块")
except ImportError:
    print("警告：无法导入SA_Score模块，SA_score将不会被计算")
    SA_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def calculate_molecular_properties(mol):
    """计算分子性质"""
    
    # 基础性质
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # 计算SA分数
    if SA_AVAILABLE:
        sa_score = sascorer.calculateScore(mol)
    else:
        sa_score = None
    
    # 计算QED
    try:
        qed = QED.qed(mol)
    except:
        qed = None
    
    # 计算Lipinski规则符合度 (0-5)
    lipinski_score = 0
    if mw <= 500: lipinski_score += 1
    if logp <= 5: lipinski_score += 1
    if hbd <= 5: lipinski_score += 1
    if hba <= 10: lipinski_score += 1
    if rotatable_bonds <=10: lipinski_score += 1
    
    return {
        'Molecular_Weight': mw,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'Rotatable_Bonds': rotatable_bonds,
        'TPSA': tpsa,
        'QED': qed,
        'SA_Score': sa_score,
        'Lipinski_Score': lipinski_score
    }

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

def run_smina_docking(ligand_sdf, receptor_pdb, ref_ligand_sdf=None, mode="score"):
    """
    使用 smina 进行配体-受体对接
    
    参数:
    - ligand_sdf: 配体SDF文件路径
    - receptor_pdb: 受体PDB文件路径  
    - ref_ligand_sdf: 参考配体SDF文件路径(可选)
    - mode: 模式 ("score"/"dock"/"min")
    """
    
    # 构建smina命令
    smina_command = [
        'smina',
        '--receptor', receptor_pdb,
        '--ligand', ligand_sdf,
    ]
    
    if mode == "dock" and ref_ligand_sdf:
        smina_command += [
            '--num_modes', '1',
            '--autobox_ligand', ref_ligand_sdf,
        ]
    elif mode == "score":
        smina_command += ['--score_only']
    elif mode == "min":
        smina_command += ['--minimize']
    
    try:
        # 运行smina
        result = subprocess.run(smina_command, check=True, capture_output=True, text=True)
        
        # 解析亲和力分数
        affinity = extract_affinity_from_log(result.stdout)
        return affinity
        
    except subprocess.CalledProcessError as e:
        logging.error(f"smina运行失败: {e}\n错误输出: {e.stderr}")
        return None

def process_single_molecule(args):
    """处理单个分子（并行处理函数）"""
    mol_index, mol, receptor_pdb, ref_ligand_sdf, mode, temp_dir = args
    
    if mol is None:
        return None
    
    # 创建临时SDF文件
    temp_sdf = os.path.join(temp_dir, f'temp_mol_{mol_index}.sdf')
    try:
        writer = Chem.SDWriter(temp_sdf)
        writer.write(mol)
        writer.close()
        
        # 计算分子性质
        properties = calculate_molecular_properties(mol)
        if properties is None:
            return None
        
        # 运行对接
        docking_score = run_smina_docking(temp_sdf, receptor_pdb, ref_ligand_sdf, mode)
        
        # 整合结果
        result = {
            'Molecule_Index': mol_index,
            'SMILES': Chem.MolToSmiles(mol) if mol else '',
            'Docking_Score': docking_score,
            'Docking_Mode': mode
        }
        result.update(properties)
        
        return result
    except Exception as e:
        logging.error(f"处理分子 {mol_index} 时出错: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(temp_sdf):
            os.remove(temp_sdf)

def main():
    """主函数：处理命令行参数并执行并行对接评估"""
    parser = argparse.ArgumentParser(description='SDF文件中多个分子与PDB并行对接评估')
    parser.add_argument('--ligand', type=str, required=True, help='配体SDF文件路径（可包含多个分子）')
    parser.add_argument('--receptor', type=str, required=True, help='受体PDB文件路径')
    parser.add_argument('--ref_ligand', type=str, help='参考配体SDF文件路径(可选，用于dock模式)')
    parser.add_argument('--mode', type=str, default='score', choices=['score', 'dock', 'min'], 
                       help='对接模式: score(打分)/dock(对接)/min(最小化)')
    parser.add_argument('--output', type=str, default='docking_results.csv', help='输出结果CSV文件')
    parser.add_argument('--max_molecules', type=int, default=0, help='最大处理分子数（0表示处理所有分子）')
    parser.add_argument('--threads', type=int, default=4, help='并行线程数（默认4）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.ligand):
        print(f"错误: 配体文件不存在: {args.ligand}")
        return
    
    if not os.path.exists(args.receptor):
        print(f"错误: 受体文件不存在: {args.receptor}")
        return
    
    if args.ref_ligand and not os.path.exists(args.ref_ligand):
        print(f"错误: 参考配体文件不存在: {args.ref_ligand}")
        return
    
    print(f"开始处理: {args.ligand} vs {args.receptor}")
    print(f"模式: {args.mode}")
    print(f"并行线程数: {args.threads}")
    if args.max_molecules > 0:
        print(f"最大处理分子数: {args.max_molecules}")
    
    # 读取SDF文件中的所有分子
    suppl = Chem.SDMolSupplier(args.ligand)
    molecules = []
    for i, mol in enumerate(suppl):
        if mol is not None:
            molecules.append((i, mol))
        if args.max_molecules > 0 and len(molecules) >= args.max_molecules:
            break
    
    print(f"找到 {len(molecules)} 个有效分子")
    
    if not molecules:
        print("错误: 没有找到有效分子")
        return
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 准备并行任务参数
        task_args = [
            (mol_index, mol, args.receptor, args.ref_ligand, args.mode, temp_dir)
            for mol_index, mol in molecules
        ]
        
        all_results = []
        successful_count = 0
        failed_count = 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # 提交所有任务
            future_to_mol = {
                executor.submit(process_single_molecule, args): args[0] 
                for args in task_args
            }
            
            # 使用tqdm显示进度条
            with tqdm(total=len(future_to_mol), desc="处理分子", unit="mol") as pbar:
                for future in as_completed(future_to_mol):
                    mol_index = future_to_mol[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                            successful_count += 1
                            pbar.set_postfix({
                                '成功': successful_count, 
                                '失败': failed_count,
                                '当前分数': f"{result['Docking_Score']:.2f}" if result['Docking_Score'] else 'N/A'
                            })
                        else:
                            failed_count += 1
                            pbar.set_postfix({
                                '成功': successful_count, 
                                '失败': failed_count
                            })
                    except Exception as e:
                        logging.error(f"分子 {mol_index} 处理异常: {e}")
                        failed_count += 1
                        pbar.set_postfix({
                            '成功': successful_count, 
                            '失败': failed_count
                        })
                    finally:
                        pbar.update(1)
    
    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 按对接分数排序（分数越低越好）
        results_df = results_df.sort_values('Docking_Score', ascending=True)
        
        # 保存到CSV
        results_df.to_csv(args.output, index=False)
        
        # 输出统计信息
        print("\n" + "="*60)
        print("并行对接评估完成!")
        print("="*60)
        print(f"成功处理: {successful_count}/{len(molecules)} 个分子")
        print(f"失败: {failed_count} 个分子")
        
        if successful_count > 0:
            valid_scores = results_df['Docking_Score'].dropna()
            if len(valid_scores) > 0:
                print(f"对接分数范围: {valid_scores.min():.2f} 到 {valid_scores.max():.2f}")
                print(f"平均对接分数: {valid_scores.mean():.2f}")
                print(f"最佳对接分数: {valid_scores.iloc[0]:.2f}")
            
            print(f"\n结果已保存到: {args.output}")
            
            # 显示前5个最佳结果
            print("\n前5个最佳对接分子:")
            print("-" * 100)
        
    else:
        print("错误: 没有成功处理任何分子")

if __name__ == "__main__":
    main()