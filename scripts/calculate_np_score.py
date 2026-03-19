#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

# 导入NP_score模块
script_dir = os.path.dirname(os.path.realpath(__file__))
NP_score_dir = os.path.join(script_dir, "NP_Score")
sys.path.append(NP_score_dir)
from npscorer import readNPModel, scoreMol

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='计算CSV文件中SMILES的NP-score并可视化')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入的CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出结果的CSV文件路径')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='包含SMILES的列名')
    parser.add_argument('--max_molecules', '-m', type=int, default=None, 
                       help='最大计算分子数量（随机选择），默认计算所有分子')
    parser.add_argument('--random_seed', type=int, default=42, 
                       help='随机种子，用于可重复的随机选择')
    parser.add_argument('--plot', action='store_true', help='是否生成NP-score分布图')
    parser.add_argument('--plot_output', type=str, default=None, help='NP-score分布图的输出路径')
    return parser.parse_args()

def load_model():
    """加载NP-score模型"""
    # 使用用户下载的模型
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join(script_dir, "NP_Score", "publicnp.model.gz")
    
    if os.path.exists(model_file):
        return readNPModel(model_file)
    else:
        raise FileNotFoundError(f"NP-score模型文件未找到: {model_file}")

def calculate_np_score(smiles, fscore):
    """计算单个SMILES的NP-score"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        
        # 使用scoreMol函数计算NP-score
        return scoreMol(mol, fscore)
    except Exception as e:
        print(f"计算NP-score时出错: {e}")
        return np.nan

def process_csv_file(input_file, smiles_col, fscore, max_molecules=None, random_seed=42):
    """处理CSV文件并计算每个SMILES的NP-score"""
    print(f"读取CSV文件: {input_file}")
    
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None

    # 检查SMILES列是否存在
    if smiles_col not in df.columns:
        print(f"错误: CSV文件中不存在列 '{smiles_col}'")
        return None

    # 如果指定了最大分子数量，则随机选择分子
    original_count = len(df)
    if max_molecules is not None and max_molecules < original_count:
        print(f"从{original_count}个分子中随机选择{max_molecules}个分子进行计算...")
        np.random.seed(random_seed)
        selected_indices = np.random.choice(df.index, size=max_molecules, replace=False)
        df = df.loc[selected_indices].copy()
        df.reset_index(drop=True, inplace=True)
        print(f"已选择{len(df)}个分子进行NP-score计算")
    else:
        print(f"计算所有{len(df)}个分子的NP-score...")
    
    # 计算NP-score
    df['np_score'] = df[smiles_col].apply(lambda x: calculate_np_score(x, fscore))
    
    # 计算有效分子数量
    valid_count = df['np_score'].notna().sum()
    print(f"成功计算了{valid_count}个分子的NP-score (总共{len(df)}个)")
    
    # 计算统计数据
    if valid_count > 0:
        print(f"NP-score统计数据:")
        print(f"  平均值: {df['np_score'].mean():.4f}")
        print(f"  中位数: {df['np_score'].median():.4f}")
        print(f"  最小值: {df['np_score'].min():.4f}")
        print(f"  最大值: {df['np_score'].max():.4f}")
        print(f"  标准差: {df['np_score'].std():.4f}")
    
    return df

def plot_np_score_distribution(df, output_path=None):
    """绘制NP-score分布图"""
    if df is None or 'np_score' not in df.columns:
        print("没有有效的NP-score数据可供绘图")
        return
    
    # 过滤掉无效值
    valid_scores = df['np_score'].dropna()
    if len(valid_scores) == 0:
        print("没有有效的NP-score数据可供绘图")
        return

    plt.figure(figsize=(10, 6))
    
    # 创建主图
    plt.subplot(1, 2, 1)
    sns.histplot(valid_scores, kde=True)
    plt.title('NP-score distribution')
    plt.xlabel('NP-score')
    plt.ylabel('Density')
    
    # 创建箱形图
    plt.subplot(1, 2, 2)
    sns.boxplot(y=valid_scores)
    plt.title('NP-score boxplot')
    plt.ylabel('NP-score')
    
    plt.tight_layout()
    
    # 保存或显示图形
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"NP-score分布图已保存到: {output_path}")
    else:
        plt.show()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载NP-score模型
    try:
        fscore = load_model()
    except Exception as e:
        print(f"加载NP-score模型时出错: {e}")
        return
    
    # 处理CSV文件
    df = process_csv_file(args.input, args.smiles_col, fscore, 
                         args.max_molecules, args.random_seed)
    if df is None:
        return
    
    # 保存结果
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"结果已保存到: {args.output}")
    
    # 绘制分布图
    if args.plot:
        plot_output = args.plot_output if args.plot_output else args.input.rsplit('.', 1)[0] + '_np_score_distribution.png'
        plot_np_score_distribution(df, plot_output)

if __name__ == "__main__":
    main()