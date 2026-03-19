import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import warnings
from scipy import stats
from collections import Counter

warnings.filterwarnings('ignore')

class SDFComparator:
    def __init__(self):
        self.molecules1 = []
        self.molecules2 = []
        
    def read_sdf_file(self, file_path, max_molecules=1000):
        """读取SDF文件并返回分子列表"""
        molecules = []
        try:
            suppl = Chem.SDMolSupplier(file_path)
            for i, mol in enumerate(suppl):
                if mol is not None and i < max_molecules:
                    molecules.append(mol)
                if len(molecules) >= max_molecules:
                    break
            print(f"从 {file_path} 成功读取 {len(molecules)} 个分子")
            return molecules
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return []
    
    def extract_coordinates(self, molecules):
        """从分子中提取所有原子的坐标"""
        all_coords = []
        for mol in molecules:
            if mol is not None:
                conf = mol.GetConformer()
                if conf is not None:
                    for atom in mol.GetAtoms():
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        all_coords.append([pos.x, pos.y, pos.z])
        return np.array(all_coords)
    
    def extract_atom_types(self, molecules):
        """从分子中提取原子类型"""
        atom_types = []
        for mol in molecules:
            if mol is not None:
                for atom in mol.GetAtoms():
                    atom_types.append(atom.GetSymbol())
        return atom_types
    
    def extract_bond_types(self, molecules):
        """从分子中提取化学键类型"""
        bond_types = []
        for mol in molecules:
            if mol is not None:
                for bond in mol.GetBonds():
                    # 获取键类型
                    bond_type = bond.GetBondType()
                    # 获取连接的原子类型
                    atom1 = bond.GetBeginAtom().GetSymbol()
                    atom2 = bond.GetEndAtom().GetSymbol()
                    # 创建标准化的键描述（按字母顺序排序原子）
                    atom_pair = tuple(sorted([atom1, atom2]))
                    bond_description = f"{atom_pair[0]}-{atom_pair[1]}({bond_type})"
                    bond_types.append(bond_description)
        return bond_types
    
    def compare_coordinates(self, coords1, coords2):
        """比较两个坐标集的分布"""
        print("\n" + "="*50)
        print("坐标分布比较")
        print("="*50)
        
        if len(coords1) == 0 or len(coords2) == 0:
            print("错误: 没有可用的坐标数据")
            return
        
        # 基本统计信息
        print(f"文件1坐标数量: {len(coords1)}")
        print(f"文件2坐标数量: {len(coords2)}")
        
        # 坐标范围比较
        for i, coord_name in enumerate(['X', 'Y', 'Z']):
            coord1 = coords1[:, i]
            coord2 = coords2[:, i]
            
            print(f"\n{coord_name}坐标比较:")
            print(f"  文件1 - 均值: {np.mean(coord1):.3f}, 标准差: {np.std(coord1):.3f}, 范围: [{np.min(coord1):.3f}, {np.max(coord1):.3f}]")
            print(f"  文件2 - 均值: {np.mean(coord2):.3f}, 标准差: {np.std(coord2):.3f}, 范围: [{np.min(coord2):.3f}, {np.max(coord2):.3f}]")
            
            # t检验
            t_stat, p_value = stats.ttest_ind(coord1, coord2)
            print(f"  t检验 p值: {p_value:.6f} ({'显著差异' if p_value < 0.05 else '无显著差异'})")
    
    def compare_atom_types(self, atoms1, atoms2):
        """比较两个原子类型分布的分布"""
        print("\n" + "="*50)
        print("原子类型分布比较")
        print("="*50)
        
        if len(atoms1) == 0 or len(atoms2) == 0:
            print("错误: 没有可用的原子类型数据")
            return
        
        # 统计原子类型频率
        counter1 = Counter(atoms1)
        counter2 = Counter(atoms2)
        
        # 获取所有原子类型
        all_atoms = set(list(counter1.keys()) + list(counter2.keys()))
        
        print(f"文件1原子总数: {len(atoms1)}")
        print(f"文件2原子总数: {len(atoms2)}")
        print(f"发现的原子类型: {sorted(all_atoms)}")
        
        # 创建比较表格
        comparison_data = []
        for atom in sorted(all_atoms):
            count1 = counter1.get(atom, 0)
            count2 = counter2.get(atom, 0)
            freq1 = count1 / len(atoms1) * 100
            freq2 = count2 / len(atoms2) * 100
            comparison_data.append({
                'Atom': atom,
                'Count1': count1,
                'Count2': count2,
                'Freq1(%)': f"{freq1:.2f}",
                'Freq2(%)': f"{freq2:.2f}",
                'Difference': f"{abs(freq1 - freq2):.2f}"
            })
        
        # 显示比较表格
        df = pd.DataFrame(comparison_data)
        print("\n原子类型分布比较表:")
        print(df.to_string(index=False))
        
        # 卡方检验
        observed = []
        for atom in sorted(all_atoms):
            observed.append([counter1.get(atom, 0), counter2.get(atom, 0)])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        print(f"\n卡方检验: χ² = {chi2:.3f}, p值 = {p_value:.6f}")
        print(f"原子类型分布 {'有显著差异' if p_value < 0.05 else '无显著差异'}")
        
        return df
    
    def compare_bond_types(self, bonds1, bonds2):
        """比较两个化学键类型分布的分布"""
        print("\n" + "="*50)
        print("化学键类型分布比较")
        print("="*50)
        
        if len(bonds1) == 0 or len(bonds2) == 0:
            print("错误: 没有可用的化学键数据")
            return
        
        # 统计化学键类型频率
        counter1 = Counter(bonds1)
        counter2 = Counter(bonds2)
        
        # 获取所有化学键类型
        all_bonds = set(list(counter1.keys()) + list(counter2.keys()))
        
        print(f"文件1化学键总数: {len(bonds1)}")
        print(f"文件2化学键总数: {len(bonds2)}")
        print(f"发现的化学键类型数量: {len(all_bonds)}")
        
        # 创建比较表格
        comparison_data = []
        for bond in sorted(all_bonds):
            count1 = counter1.get(bond, 0)
            count2 = counter2.get(bond, 0)
            freq1 = count1 / len(bonds1) * 100 if len(bonds1) > 0 else 0
            freq2 = count2 / len(bonds2) * 100 if len(bonds2) > 0 else 0
            comparison_data.append({
                'Bond_Type': bond,
                'Count1': count1,
                'Count2': count2,
                'Freq1(%)': f"{freq1:.2f}",
                'Freq2(%)': f"{freq2:.2f}",
                'Difference': f"{abs(freq1 - freq2):.2f}"
            })
        
        # 显示比较表格（只显示前20个最常见的键类型）
        df = pd.DataFrame(comparison_data)
        print(f"\n化学键类型分布比较表 (显示前20个):")
        print(df.head(20).to_string(index=False))
        
        if len(df) > 20:
            print(f"... 还有 {len(df) - 20} 个其他键类型")
        
        # 卡方检验
        observed = []
        for bond in sorted(all_bonds):
            observed.append([counter1.get(bond, 0), counter2.get(bond, 0)])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        print(f"\n卡方检验: χ² = {chi2:.3f}, p值 = {p_value:.6f}")
        print(f"化学键类型分布 {'有显著差异' if p_value < 0.05 else '无显著差异'}")
        
        # 显示最常见的化学键类型
        print(f"\n文件1中最常见的化学键类型:")
        for bond, count in counter1.most_common(10):
            freq = count / len(bonds1) * 100
            print(f"  {bond}: {count}次 ({freq:.2f}%)")
        
        print(f"\n文件2中最常见的化学键类型:")
        for bond, count in counter2.most_common(10):
            freq = count / len(bonds2) * 100
            print(f"  {bond}: {count}次 ({freq:.2f}%)")
        
        return df
    
    def compare_sdf_files(self, file1, file2, sample_size=1000):
        """主比较函数"""
        print(f"开始比较SDF文件:")
        print(f"文件1: {file1}")
        print(f"文件2: {file2}")
        print(f"采样大小: {sample_size} 个分子")
        
        # 读取文件
        self.molecules1 = self.read_sdf_file(file1, sample_size)
        self.molecules2 = self.read_sdf_file(file2, sample_size)
        
        if len(self.molecules1) == 0 or len(self.molecules2) == 0:
            print("错误: 无法读取足够的分子数据进行比较")
            return
        
        # 提取坐标、原子类型和化学键类型
        print("\n提取坐标信息...")
        coords1 = self.extract_coordinates(self.molecules1)
        coords2 = self.extract_coordinates(self.molecules2)
        
        print("提取原子类型信息...")
        atoms1 = self.extract_atom_types(self.molecules1)
        atoms2 = self.extract_atom_types(self.molecules2)
        
        print("提取化学键类型信息...")
        bonds1 = self.extract_bond_types(self.molecules1)
        bonds2 = self.extract_bond_types(self.molecules2)
        
        # 进行比较
        self.compare_coordinates(coords1, coords2)
        atom_df = self.compare_atom_types(atoms1, atoms2)
        bond_df = self.compare_bond_types(bonds1, bonds2)
        
        return {
            'coordinates_file1': coords1,
            'coordinates_file2': coords2,
            'atoms_file1': atoms1,
            'atoms_file2': atoms2,
            'bonds_file1': bonds1,
            'bonds_file2': bonds2,
            'atom_comparison_df': atom_df,
            'bond_comparison_df': bond_df
        }

def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='比较两个SDF文件的坐标、原子类型和化学键分布')
    parser.add_argument('--file1', type=str, default='data/coconut_raw/coconut_sdf_3d-06-2025.sdf', 
                       help='第一个SDF文件路径 (默认: file1.sdf)')
    parser.add_argument('--file2', type=str, default='results/FlowNP.sdf',
                       help='第二个SDF文件路径 (默认: file2.sdf)')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='从每个文件中抽取的分子数量 (默认: 1000)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file1):
        print(f"错误: 文件 {args.file1} 不存在")
        print("请使用 --file1 参数指定正确的文件路径")
        return
    
    if not os.path.exists(args.file2):
        print(f"错误: 文件 {args.file2} 不存在")
        print("请使用 --file2 参数指定正确的文件路径")
        return
    
    # 创建比较器并执行比较
    comparator = SDFComparator()
    results = comparator.compare_sdf_files(args.file1, args.file2, sample_size=args.sample_size)
    
    print("\n比较完成!")

if __name__ == "__main__":
    main()