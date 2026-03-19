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
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='从多个CSV文件中分析分子性质差异')
    parser.add_argument('--input_files', '-i', nargs='+', required=True, help='输入的CSV文件路径列表')
    parser.add_argument('--output_dir', '-o', type=str, default='.', help='输出结果的目录')
    parser.add_argument('--smiles_col', type=str, default=None, help='包含SMILES的列名，如果为None则自动检测')
    parser.add_argument('--smiles_cols', type=str, nargs='+', help='每个文件对应的SMILES列名，顺序需与输入文件一致')
    parser.add_argument('--n_molecules', '-n', type=int, default=1000, help='每个文件要分析的分子数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

def detect_smiles_column(df):
    """尝试自动检测SMILES列"""
    possible_column_names = ['smiles', 'SMILES', 'canonical_smiles', 'Canonical_SMILES', 'smi', 
                            'mol_string', 'molecule', 'structure', 'canonical_smiles', 'can_smiles', 
                            'smiles_string', 'smile', 'Smile']
    
    # 按优先级检查列名
    for col in possible_column_names:
        if col in df.columns:
            # 验证是否实际包含SMILES
            sample = df[col].iloc[0] if len(df) > 0 else ""
            if isinstance(sample, str) and len(sample) > 0:
                try:
                    mol = Chem.MolFromSmiles(sample)
                    if mol is not None:
                        return col
                except:
                    pass
    
    # 检查所有列，查找可能包含SMILES的列
    for col in df.columns:
        if df[col].dtype == object:  # 字符串列
            sample = df[col].iloc[0] if len(df) > 0 else ""
            if isinstance(sample, str) and len(sample) > 0:
                try:
                    mol = Chem.MolFromSmiles(sample)
                    if mol is not None:
                        return col
                except:
                    pass
    
    return None

def load_and_process_csv(file_path, smiles_col, n_molecules, random_state=None):
    """加载CSV文件并提取SMILES"""
    print(f"处理文件: {file_path}")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None
    
    # 如果未提供SMILES列，尝试自动检测
    if smiles_col is None:
        smiles_col = detect_smiles_column(df)
        if smiles_col:
            print(f"自动检测到SMILES列: '{smiles_col}'")
        else:
            print(f"错误: 无法在文件 {file_path} 中检测到SMILES列")
            print(f"可用列: {list(df.columns)}")
            return None
    
    # 检查SMILES列是否存在
    if smiles_col not in df.columns:
        print(f"错误: 文件 {file_path} 中不存在列 '{smiles_col}'")
        print(f"可用列: {list(df.columns)}")
        return None
    
    # 随机采样n_molecules个分子（如果文件中的分子数量足够）
    if len(df) > n_molecules:
        df = df.sample(n=n_molecules, random_state=random_state)
    
    print(f"提取 {len(df)} 个分子的SMILES...")
    
    # 提取SMILES和来源信息
    result_df = pd.DataFrame({
        'SMILES': df[smiles_col],
        'source': os.path.basename(file_path)
    })
    
    return result_df

def plot_tsne_distribution(combined_df, output_dir):
    """绘制基于分子指纹的t-SNE分布图"""
    if combined_df is None or len(combined_df) == 0:
        print("没有数据可供绘图")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 修改数据来源名称
    def clean_source_name(source):
        # 去掉 _np_score.csv 后缀
        source = source.replace('_np_score.csv', '')
        # 替换特定名称
        if source == 'generated_coconut':
            return 'generated_without_pocket'
        elif source == 'generated_from_npz':
            return 'generated_with_pocket'
        return source
    
    # 应用名称修改
    combined_df['clean_source'] = combined_df['source'].apply(clean_source_name)
    
    # t-SNE 分布图 (基于分子指纹)
    print("\n开始生成基于分子指纹的t-SNE分布图...")

    if 'SMILES' not in combined_df.columns:
        print("警告：combined_df中缺少'SMILES'列，无法生成基于指纹的t-SNE图。")
        return
    
    fingerprints = []
    valid_indices = []  # 跟踪具有有效SMILES的行索引
    valid_smiles = []   # 保存有效的SMILES
    
    print("正在计算分子指纹...")
    for idx, smiles in combined_df['SMILES'].items():
        # 检查是否为有效值并转换为字符串
        if pd.isna(smiles):
            continue
            
        # 转换为字符串
        smiles_str = str(smiles)
        
        # 检查字符串长度
        if len(smiles_str.strip()) == 0:
            continue
            
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol:
                # 使用Morgan指纹 (类似于ECFP4)，半径为2，2048位
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(np.array(fp))
                valid_indices.append(idx)
                valid_smiles.append(smiles_str)
        except Exception as e:
            print(f"处理SMILES '{smiles_str}' 时出错: {e}")
            continue
    
    print(f"成功计算了 {len(fingerprints)} 个分子指纹。")

    if not fingerprints:
        print("警告：未能从SMILES生成任何有效的分子指纹，跳过t-SNE图。")
        return
    
    features = np.array(fingerprints)
    # 获取与有效指纹相对应的来源标签和其他信息
    labels = combined_df.loc[valid_indices, 'clean_source'].str.rstrip('.csv')
    original_sources = combined_df.loc[valid_indices, 'source']  # 原始来源名称

    # t-SNE的perplexity参数值必须小于样本数
    n_samples = features.shape[0]
    perplexity_value = min(30.0, float(n_samples - 1))
    
    if perplexity_value > 1:
        print(f"正在执行t-SNE降维（样本数={n_samples}, perplexity={int(perplexity_value)}）... 这可能需要一些时间。")
        tsne = TSNE(n_components=2, 
                    random_state=42, 
                    perplexity=perplexity_value, 
                    n_iter=1000,
                    init='pca',
                    learning_rate='auto')
        # 直接对指纹进行降维，无需标准化
        tsne_results = tsne.fit_transform(features)
        print("t-SNE降维完成。")

        # 创建用于绘图的DataFrame
        plot_df = pd.DataFrame({
            't-SNE Dimension 1': tsne_results[:, 0],
            't-SNE Dimension 2': tsne_results[:, 1],
            'source': labels,
            'original_source': original_sources,
            'SMILES': valid_smiles
        })
        
        # 保存t-SNE坐标数据到CSV文件
        tsne_coords_file = os.path.join(output_dir, 'tsne_coordinates.csv')
        plot_df.to_csv(tsne_coords_file, index=False)
        print(f"已保存t-SNE坐标数据: {tsne_coords_file}")

        # 绘制t-SNE散点图
        plt.figure(figsize=(12, 12))
        sns.set_style("white")
        
        # 获取颜色映射
        unique_sources = plot_df['source'].unique()
        palette = sns.color_palette("hsv", len(unique_sources))
        color_dict = dict(zip(unique_sources, palette))
        
        scatter_plot = sns.scatterplot(
            x='t-SNE Dimension 1', y='t-SNE Dimension 2',
            hue='source',
            palette=color_dict,
            data=plot_df,
            legend="full",
            alpha=0.7,
            s=50
        )
        
        # 为每个来源类别绘制椭圆
        for source_name in plot_df['source'].unique():
            source_data = plot_df[plot_df['source'] == source_name]
            x = source_data['t-SNE Dimension 1']
            y = source_data['t-SNE Dimension 2']
            
            # 计算均值和协方差
            if len(x) > 1:  # 需要至少2个点才能计算协方差
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                cov = np.cov(x, y)
                
                # 计算椭圆参数
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                
                # 获取对应的颜色
                ellipse_color = color_dict[source_name]
                
                # 绘制椭圆 (2个标准差范围，覆盖约95%的数据)
                ellipse = Ellipse(
                    xy=(mean_x, mean_y),
                    width=lambda_[0]*2*2,  # 2倍标准差，乘以2是因为width是直径
                    height=lambda_[1]*2*2,
                    angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                    edgecolor=ellipse_color,
                    fc='None',
                    lw=3,
                    alpha=1.0
                )
                plt.gca().add_patch(ellipse)
                
        
        plt.title('t-SNE Visualization of Chemical Space (based on Morgan Fingerprints)', fontsize=18)
        plt.xlabel('t-SNE Dimension 1', fontsize=14)
        plt.ylabel('t-SNE Dimension 2', fontsize=14)
        plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 1])

        # 保存图像
        output_file = os.path.join(output_dir, 'tsne_fingerprint_distribution.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"已保存基于分子指纹的t-SNE分布图: {output_file}")
    else:
        print("警告：有效指纹数量过少，无法执行t-SNE。")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有输入文件
    all_data = []
    
    # 处理smiles_cols参数
    smiles_cols_dict = {}
    if args.smiles_cols:
        if len(args.smiles_cols) != len(args.input_files):
            print(f"警告: 提供的SMILES列名数量 ({len(args.smiles_cols)}) 与输入文件数量 ({len(args.input_files)}) 不匹配")
        else:
            for i, file_path in enumerate(args.input_files):
                smiles_cols_dict[file_path] = args.smiles_cols[i]
    
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            continue
        
        # 确定当前文件的SMILES列
        current_smiles_col = smiles_cols_dict.get(file_path) if smiles_cols_dict else args.smiles_col
        
        df = load_and_process_csv(
            file_path, 
            current_smiles_col, 
            args.n_molecules, 
            random_state=args.seed
        )
        
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("没有有效数据可供分析")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 保存合并后的数据
    output_file = os.path.join(args.output_dir, 'combined_smiles.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"已将合并数据保存到: {output_file}")
    
    # 绘制t-SNE分布图
    plot_tsne_distribution(combined_df, args.output_dir)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()