import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    from rdkit.Chem import SAscore
    SA_AVAILABLE = True
    print("成功导入RDKit SA_Score模块")
except ImportError:
    print("RDKit中的SA_score模块不可用，尝试导入自定义模块")
    
    # 尝试导入自定义SA_Score
    SA_score_dir = os.path.join(script_dir, "SA_Score")
    sys.path.append(SA_score_dir)
    try:
        import sascorer
        SA_AVAILABLE = True
        print("成功导入自定义SA_Score模块")
    except ImportError:
        print("警告：无法导入SA_Score模块，SA_score将不会被计算")

# 计算分子性质
def calculate_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 基础性质
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    
    # 计算QED
    qed = QED.qed(mol)
    
    # 计算SA分数 (Synthetic Accessibility)
    sa_score = globals()['sascorer'].calculateScore(mol)
    
    # 计算Lipinski规则符合度 (0-4)
    lipinski_score = 0
    if mw <= 500: lipinski_score += 1
    if logp <= 5: lipinski_score += 1
    if hbd <= 5: lipinski_score += 1
    if hba <= 10: lipinski_score += 1
    if rotatable_bonds <= 10: lipinski_score += 1
    
    return {
        'MW': mw,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'RotatableBonds': rotatable_bonds,
        'QED': qed,
        'SA': sa_score,
        'Lipinski_score': lipinski_score
    }


# 处理单个CSV文件
def process_docking_file(file_path, group_name):
    df = pd.read_csv(file_path)
    print(f"Processing {file_path} with {len(df)} molecules...")
    
    # 确保有SMILES列
    if 'smiles' not in df.columns:
        print(f"Error: 'smiles' column not found in {file_path}")
        return None
    
    # 计算分子性质
    properties = []
    for smiles in df['smiles']:
        prop = calculate_molecular_properties(smiles)
        properties.append(prop)
    
    # 合并结果
    prop_df = pd.DataFrame(properties)
    result_df = pd.concat([df, prop_df], axis=1)
    result_df['group'] = group_name
    
    # 过滤无效分子
    valid_df = result_df.dropna(subset=['LogP', 'QED', 'SA'])
    print(f"Valid molecules: {len(valid_df)}/{len(df)}")
    
    return valid_df

# 绘制性质分布图
def plot_property_distributions(combined_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图样式
    sns.set(style="whitegrid", palette="muted")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # 定义要绘制的性质
    properties = ['affinity', 'LogP', 'QED', 'SA', 'MW', 'Lipinski_score']
    property_names = {
        'affinity': 'Vina Score (kcal/mol)',
        'LogP': 'LogP (Lipophilicity)',
        'QED': 'QED (Drug Likeness)',
        'SA': 'SA (Synthetic Accessibility)',
        'MW': 'Molecular Weight (Da)',
        'Lipinski_score': 'Lipinski Rule Compliance'
    }
    
    # 1. 绘制密度曲线图
    print("Plotting density distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Molecular Property Distributions", fontsize=20)
    
    for ax, prop in zip(axes.flatten(), properties):
        # 计算每组数据的密度曲线
        for group, data in combined_df.groupby('group'):
            values = data[prop].dropna()
            if len(values) > 1:
                kde = gaussian_kde(values)
                x = np.linspace(values.min(), values.max(), 500)
                y = kde(x)
                ax.plot(x, y, label=group, lw=2)
        
        ax.set_title(property_names[prop])
        ax.set_xlabel(property_names[prop])
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), 
               bbox_to_anchor=(0.5, -0.05), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'density_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    

# 生成统计表
def generate_statistics_table(combined_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义统计指标
    stats_functions = {
        'Count': 'count',
        'Mean': 'mean',
        'Median': 'median',
        'Std Dev': 'std',
        'Min': 'min',
        'Max': 'max',
        'Q1': lambda x: x.quantile(0.25),
        'Q3': lambda x: x.quantile(0.75)
    }
    
    # 定义要统计的性质
    properties = ['affinity', 'LogP', 'QED', 'SA', 'MW', 'Lipinski_score']
    
    # 生成统计表
    stats_tables = []
    for prop in properties:
        prop_stats = combined_df.groupby('group')[prop].agg(list(stats_functions.values()))
        prop_stats.columns = list(stats_functions.keys())
        prop_stats['Property'] = prop
        stats_tables.append(prop_stats)
    
    # 合并所有统计结果
    all_stats = pd.concat(stats_tables)
    all_stats = all_stats.reset_index().set_index(['Property', 'group'])
    
    # 保存为CSV和Excel
    stats_csv = os.path.join(output_dir, 'property_statistics.csv')
    
    all_stats.to_csv(stats_csv)
    
    print(f"Statistics saved to:\n{stats_csv}")
    return all_stats

# 主函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze docking results and molecular properties')
    parser.add_argument('input_files', nargs='+', help='Input CSV files with docking results')
    parser.add_argument('--output_dir', default='analysis_results', help='Output directory for results')
    args = parser.parse_args()
    
    # 处理所有输入文件
    all_data = []
    for file_path in args.input_files:
        group_name = os.path.splitext(os.path.basename(file_path))[0]
        df = process_docking_file(file_path, group_name)
        if df is not None and not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("No valid data to process!")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存合并后的数据
    combined_file = os.path.join(args.output_dir, 'combined_results.csv')
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined data saved to: {combined_file}")
    
    # 生成分布图
    plot_property_distributions(combined_df, args.output_dir)
    
    # 生成统计表
    stats_table = generate_statistics_table(combined_df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()