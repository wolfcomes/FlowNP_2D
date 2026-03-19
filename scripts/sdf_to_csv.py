import sys
import os
from rdkit import Chem
import pandas as pd

def has_fragments(smiles):
    """检查SMILES是否含有断点（未连接的原子或键）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # 无效SMILES
    
    # 检查是否有未连接的原子（如 "." 分隔的片段）
    if "." in smiles:
        return True
    
    # 检查分子中是否有未连接的原子（如游离原子）
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0 and atom.GetAtomicNum() != 0:
            return True
    
    return False

def process_sdf_file(input_sdf, output_csv=None):
    """
    处理单个SDF文件，返回包含有效SMILES的DataFrame
    
    参数:
        input_sdf (str): 输入的SDF文件路径
        output_csv (str): 可选，输出的CSV文件路径
        
    返回:
        pd.DataFrame: 包含有效SMILES的DataFrame
    """
    supplier = Chem.SDMolSupplier(input_sdf)
    data = []
    valid_count = 0
    invalid_count = 0

    for mol in supplier:
        if mol is None:
            invalid_count += 1
            continue
        
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if not has_fragments(smiles):
                name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
                data.append({'Name': name, 'smiles': smiles, 'Source': os.path.basename(input_sdf)})
                valid_count += 1
            else:
                invalid_count += 1
        except:
            invalid_count += 1
    
    df = pd.DataFrame(data)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"文件 {input_sdf} 处理完成！有效分子（无断点）: {valid_count}, 无效分子（含断点/错误）: {invalid_count}")
        print(f"结果已保存到: {output_csv}")
    else:
        print(f"文件 {input_sdf} 处理完成！有效分子（无断点）: {valid_count}, 无效分子（含断点/错误）: {invalid_count}")
    
    return df

def sdf_to_smiles_csv(input_path, output_csv):
    """
    将SDF文件或目录下的所有SDF文件转换为CSV，仅保留无断点的SMILES
    
    参数:
        input_path (str): 输入的SDF文件路径或目录路径
        output_csv (str): 输出的CSV文件路径
    """
    try:
        all_data = []
        
        if os.path.isdir(input_path):
            # 处理目录下的所有SDF文件
            print(f"正在处理目录: {input_path}")
            for filename in os.listdir(input_path):
                if filename.lower().endswith('.sdf'):
                    filepath = os.path.join(input_path, filename)
                    df = process_sdf_file(filepath)
                    all_data.append(df)
            
            if not all_data:
                print("目录中没有找到SDF文件！")
                return
                
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(output_csv, index=False)
            print(f"所有文件处理完成！结果已合并保存到: {output_csv}")
            
        elif os.path.isfile(input_path) and input_path.lower().endswith('.sdf'):
            # 处理单个SDF文件
            process_sdf_file(input_path, output_csv)
        else:
            print("错误: 输入路径必须是SDF文件或包含SDF文件的目录！")
            sys.exit(1)

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python sdf_to_smiles_nofrag.py 输入文件或目录 输出文件.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_file = sys.argv[2]
    sdf_to_smiles_csv(input_path, output_file)