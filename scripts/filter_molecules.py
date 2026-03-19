from rdkit import Chem
from rdkit.Chem import AllChem

def filter_molecules(input_sdf, output_sdf, min_atoms=70, max_molecules=100):
    """
    从输入SDF文件中筛选原子数量大于min_atoms的分子，最多max_molecules个，写入输出SDF文件
    
    参数:
        input_sdf: 输入SDF文件路径
        output_sdf: 输出SDF文件路径
        min_atoms: 最小原子数阈值 (默认70)
        max_molecules: 最大输出分子数 (默认100)
    """
    supplier = Chem.SDMolSupplier(input_sdf)
    writer = Chem.SDWriter(output_sdf)
    
    count = 0
    for mol in supplier:
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > min_atoms:
                writer.write(mol)
                count += 1
                print(f"找到符合条件的分子 {count}: {num_atoms} 个原子")
                
                if count >= max_molecules:
                    break
    
    writer.close()
    print(f"\n完成! 共筛选出 {count} 个分子，已保存到 {output_sdf}")

if __name__ == "__main__":
    input_file = "data/coconut_raw/coconut_sdf_3d-06-2025.sdf"  # 替换为你的输入SDF文件路径
    output_file = "filtered_output.sdf"  # 输出文件路径
    
    filter_molecules(input_file, output_file)