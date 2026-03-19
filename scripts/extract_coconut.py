import argparse
from pathlib import Path
import warnings
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# 抑制RDKit警告
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)

def is_3d_molecule(mol):
    """检查分子是否为3D结构（非平面结构）"""
    if mol is None or mol.GetNumAtoms() < 4:
        return False
    
    conf = mol.GetConformer()
    if conf is None:
        return False
    
    # 获取原子坐标
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    
    positions = np.array(positions)
    
    # 中心化坐标
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    
    # 计算奇异值
    _, s, _ = np.linalg.svd(centered)
    
    # 检查最小奇异值是否足够大（非平面结构）
    return s[2] > 1e-2

def process_sdf_to_csv(sdf_path, output_csv, output_sdf=None, max_molecules=None, max_atoms=None):
    """处理SDF文件并保存合格分子到CSV和SDF"""
    print(f"加载分子文件: {sdf_path}")
    if max_atoms:
        print(f"最大原子数限制: {max_atoms}")
    
    all_smiles = []
    valid_mols = []  # 存储有效的分子对象
    stats = {
        'total': 0,
        'failed': 0,
        'non_3d': 0,
        'stereo_issues': 0,
        'too_many_atoms': 0
    }
    
    # 使用宽松的参数读取分子
    mol_reader = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    
    with tqdm(desc="处理分子") as pbar:
        for mol in mol_reader:
            stats['total'] += 1
            pbar.update(1)
            
            if mol is None:
                stats['failed'] += 1
                continue

            try:
                # 检查原子数限制
                if max_atoms and mol.GetNumAtoms() > max_atoms:
                    stats['too_many_atoms'] += 1
                    continue
                
                # 尝试进行分子标准化处理
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                    except:
                        stats['failed'] += 1
                        continue
                
                # 生成SMILES表示
                try:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                except:
                    stats['stereo_issues'] += 1
                    try:
                        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                    except:
                        stats['failed'] += 1
                        continue
                
                # 检查是否为3D结构
                if not is_3d_molecule(mol):
                    stats['non_3d'] += 1
                    continue
                
                all_smiles.append(smiles)
                valid_mols.append(mol)  # 保存有效的分子对象
                
                # 如果达到最大分子数限制则停止
                if max_molecules and len(all_smiles) >= max_molecules:
                    break
                    
            except Exception:
                stats['failed'] += 1
                continue

    # 保存结果到CSV
    with open(output_csv, 'w') as f:
        f.write("smiles\n")
        for smi in all_smiles:
            f.write(f"{smi}\n")
    
    # 保存结果到SDF
    if output_sdf:
        with Chem.SDWriter(str(output_sdf)) as writer:
            for mol in valid_mols:
                if mol is not None:
                    writer.write(mol)
    
    # 打印统计信息
    print("\n处理完成!")
    print(f"输入分子总数: {stats['total']}")
    print(f"有效3D分子数: {len(all_smiles)}")
    print(f"失败分子数: {stats['failed']}")
    print(f"非3D分子数: {stats['non_3d']}")
    print(f"立体化学问题数: {stats['stereo_issues']}")
    if max_atoms:
        print(f"超过原子数限制的分子数: {stats['too_many_atoms']}")
    print(f"CSV结果已保存至: {output_csv}")
    if output_sdf:
        print(f"SDF结果已保存至: {output_sdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从SDF文件中提取有效的3D分子SMILES')
    parser.add_argument('--sdf', type=str, required=True, help='输入的SDF文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出的CSV文件路径')
    parser.add_argument('--output_sdf', type=str, default=None, help='输出的SDF文件路径（可选）')
    parser.add_argument('--max_molecules', type=int, default=None, help='最大处理分子数（可选）')
    parser.add_argument('--max_atoms', type=int, default=None, help='最大原子数限制（可选）')
    
    args = parser.parse_args()
    
    # 验证文件路径
    sdf_path = Path(args.sdf)
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF文件不存在: {sdf_path}")
    
    # 如果没有指定输出SDF路径，则基于CSV路径自动生成
    if args.output_sdf is None:
        csv_path = Path(args.output)
        output_sdf_path = csv_path.with_suffix('.sdf')
    else:
        output_sdf_path = Path(args.output_sdf)
    
    process_sdf_to_csv(
        sdf_path=args.sdf,
        output_csv=args.output,
        output_sdf=output_sdf_path,
        max_molecules=args.max_molecules,
        max_atoms=args.max_atoms
    )