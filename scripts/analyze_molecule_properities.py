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
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski, AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds, CalcNumRings

# 导入用于t-SNE分析的新模块
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# 导入NP_score模块
script_dir = os.path.dirname(os.path.realpath(__file__))
NP_score_dir = os.path.join(script_dir, "NP_Score")
SA_score_dir = os.path.join(script_dir, "SA_Score")
sys.path.append(NP_score_dir)
sys.path.append(SA_score_dir)

# 设置默认值
readNPModel = None
scoreMol = None
SA_AVAILABLE = False

# 尝试导入NP_Score
try:
    from npscorer import readNPModel, scoreMol
    print("成功导入NP_Score模块")
except ImportError:
    print("警告：无法导入NP_Score模块，NP_score将不会被计算")

# 尝试导入SA_Score
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


def calculate_properties(smiles, np_model=None):
    """计算单个SMILES的分子性质"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {}
        
        # 添加SMILES列，以便后续用于指纹计算
        properties['SMILES'] = smiles
        
        # 计算QED（药物类似性）
        properties['QED'] = QED.qed(mol)
        
        # 计算氢键受体数量 (HA)
        properties['HA'] = CalcNumHBA(mol)
        
        # 计算氢键供体数量 (HB)
        properties['HB'] = CalcNumHBD(mol)
        
        # 计算LogP（脂水分配系数）
        properties['LogP'] = Crippen.MolLogP(mol)
        
        # 计算分子量 (MW)
        properties['MW'] = Descriptors.MolWt(mol)
        
        # 新增：计算可旋转键数量
        properties['RotatableBonds'] = CalcNumRotatableBonds(mol)
        
        # 新增：计算环数量
        properties['Rings'] = CalcNumRings(mol)
        
        # 新增：计算手性中心数量
        properties['ChiralCenters'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # 计算NP_score（天然产物相似性分数）
        if np_model is not None and 'scoreMol' in globals() and scoreMol is not None:
            properties['NP_score'] = scoreMol(mol, np_model)
        else:
            properties['NP_score'] = np.nan
            
        # 计算SA_score（合成可及性分数）
        if SA_AVAILABLE:
            # 使用自定义sascorer模块
            if 'sascorer' in globals() and hasattr(globals()['sascorer'], 'calculateScore'):
                properties['SA'] = globals()['sascorer'].calculateScore(mol)
            # 使用RDKit SAscore模块
            elif 'SAscore' in globals() and hasattr(globals()['SAscore'], 'calculateScore'):
                properties['SA'] = SAscore.calculateScore(mol)
            else:
                properties['SA'] = np.nan
        else:
            properties['SA'] = np.nan
            
        return properties
        
    except Exception as e:
        print(f"计算分子性质时出错: {e}")
        return None


def load_and_process_csv(file_path, smiles_col, n_molecules, np_model=None, random_state=None):
    """加载CSV文件并计算分子性质"""
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
    
    print(f"计算 {len(df)} 个分子的性质...")
    
    # 计算分子性质
    properties_list = []
    valid_count = 0
    
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        if not isinstance(smiles, str) or len(smiles) == 0:
            continue
            
        properties = calculate_properties(smiles, np_model)
        
        if properties:
            properties_list.append(properties)
            valid_count += 1
    
    print(f"成功计算了 {valid_count} 个分子的性质")
    
    if valid_count == 0:
        return None
    
    # 创建包含所有性质的DataFrame
    result_df = pd.DataFrame(properties_list)
    
    # 添加文件来源信息
    result_df['source'] = os.path.basename(file_path)
    
    return result_df


def plot_property_distributions(combined_df, output_dir):
    """绘制分子性质分布图和t-SNE图"""
    if combined_df is None or len(combined_df) == 0:
        print("没有数据可供绘图")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设定图表样式
    sns.set(style="whitegrid")
    
    # 更新性质列表，包含新增的性质
    property_names = ['QED', 'HA', 'HB', 'LogP', 'MW', 'RotatableBonds', 'Rings', 'ChiralCenters', 'NP_score', 'SA']
    
    # 英文映射
    property_names_en = {
        'QED': 'QED (Drug Likeness)',
        'HA': 'HA (Hydrogen Bond Acceptors)',
        'HB': 'HB (Hydrogen Bond Donors)',
        'LogP': 'LogP (Lipophilicity)',
        'MW': 'MW (Molecular Weight)',
        'RotatableBonds': 'Rotatable Bonds',
        'Rings': 'Number of Rings',
        'ChiralCenters': 'Chiral Centers',
        'NP_score': 'NP_score (Natural Product Likeness)',
        'SA': 'SA (Synthetic Accessibility)',
    }

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
    
    # 按图类型分组 - 小提琴图
    fig_violin, axes_violin = plt.subplots(4, 3, figsize=(20, 20))
    fig_violin.suptitle("Violin Plots of Molecular Properties", fontsize=16)
    axes_violin = axes_violin.flatten()
    
    # 按图类型分组 - 箱形图
    fig_box, axes_box = plt.subplots(4, 3, figsize=(20, 20))
    fig_box.suptitle("Box Plots of Molecular Properties", fontsize=16)
    axes_box = axes_box.flatten()
    
    # 按图类型分组 - 散点图
    fig_strip, axes_strip = plt.subplots(4, 3, figsize=(20, 20))
    fig_strip.suptitle("Strip Plots of Molecular Properties", fontsize=16)
    axes_strip = axes_strip.flatten()
    
    # 绘制所有性质的图
    for i, prop in enumerate(property_names):
        if prop not in combined_df.columns:
            continue
            
        # 过滤掉无效值
        valid_data = combined_df.dropna(subset=[prop])
        if len(valid_data) == 0:
            continue
        
        # 1. 小提琴图
        if i < len(axes_violin):
            sns.violinplot(x='clean_source', y=prop, data=valid_data, ax=axes_violin[i])
            axes_violin[i].set_title(f'{property_names_en[prop]}')
            axes_violin[i].set_xlabel('Data Source')
            axes_violin[i].set_ylabel(property_names_en[prop])
            axes_violin[i].tick_params(axis='x', rotation=45)
        
        # 2. 箱形图
        if i < len(axes_box):
            sns.boxplot(x='clean_source', y=prop, data=valid_data, ax=axes_box[i])
            axes_box[i].set_title(f'{property_names_en[prop]}')
            axes_box[i].set_xlabel('Data Source')
            axes_box[i].set_ylabel(property_names_en[prop])
            axes_box[i].tick_params(axis='x', rotation=45)
        
        # 3. 散点图
        if i < len(axes_strip):
            sns.stripplot(x='clean_source', y=prop, data=valid_data, jitter=True, alpha=0.5, ax=axes_strip[i])
            axes_strip[i].set_title(f'{property_names_en[prop]}')
            axes_strip[i].set_xlabel('Data Source')
            axes_strip[i].set_ylabel(property_names_en[prop])
            axes_strip[i].tick_params(axis='x', rotation=45)
    
    # 调整布局并保存图像
    fig_violin.tight_layout(rect=[0, 0, 1, 0.97])
    fig_box.tight_layout(rect=[0, 0, 1, 0.97])
    fig_strip.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图像
    fig_violin.savefig(os.path.join(output_dir, 'violin_plots.png'), dpi=300)
    fig_box.savefig(os.path.join(output_dir, 'box_plots.png'), dpi=300)
    fig_strip.savefig(os.path.join(output_dir, 'strip_plots.png'), dpi=300)
    
    plt.close(fig_violin)
    plt.close(fig_box)
    plt.close(fig_strip)
    
    print("已保存小提琴图、箱形图和散点图")
    
    # 创建密度图 - 分组展示
    fig_density, axes = plt.subplots(2, 4, figsize=(18, 9))  # 缩小图形尺寸
    # fig_density.suptitle("Molecular Property Distributions", fontsize=22)  # 缩小标题字体

    sns.set_style("white")
    all_lines, all_labels = [], []

    # 第一行：基础理化性质
    row1_props = ['NP_score','LogP', 'MW', 'QED']
    # 第二行：结构特征和评分性质
    row2_props = ['RotatableBonds', 'Rings', 'ChiralCenters','SA']

    # 绘制第一行性质
    for i, prop in enumerate(row1_props):
        if prop not in combined_df.columns: continue
        valid_data = combined_df.dropna(subset=[prop])
        if len(valid_data) == 0: continue
        ax = axes[0, i]
        for source, group in valid_data.groupby('clean_source'):
            line = sns.kdeplot(group[prop], label=source, ax=ax, fill=False, linewidth=2)
            if i == 0:
                all_lines.append(line.get_lines()[-1])
                all_labels.append(source.split('.')[0])
        # ax.set_title(property_names_en[prop], fontsize=18)
        ax.set_ylabel('Density' if i == 0 else '', fontsize=14)
        ax.set_xlabel(property_names_en[prop], fontsize=14)
        # 移除辅助线标识
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.get_legend(): ax.get_legend().remove()

    # 绘制第二行性质
    for i, prop in enumerate(row2_props):
        if prop not in combined_df.columns: continue
        valid_data = combined_df.dropna(subset=[prop])
        if len(valid_data) == 0: continue
        ax = axes[1, i]
        for source, group in valid_data.groupby('clean_source'):
            sns.kdeplot(group[prop], label=source, ax=ax, fill=False, linewidth=2)
        # ax.set_title(property_names_en[prop], fontsize=12)
        ax.set_ylabel('Density' if i == 0 else '', fontsize=14)
        ax.set_xlabel(property_names_en[prop], fontsize=14)
        # 移除辅助线标识
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.get_legend(): ax.get_legend().remove()

    # 添加图例到右上角
    if all_lines:
        # 在最后一个子图的右上角添加图例
        last_ax = axes[0, 0]
        last_ax.legend(all_lines, all_labels, loc='upper left', frameon=False, fontsize=10)

    fig_density.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_dir, 'all_properties_density.png')
    fig_density.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig_density)
    print(f"已保存所有性质的密度曲线图: {output_file}")
    
    # 创建相关性热图
    plt.figure(figsize=(14, 12))
    # 仅选择数值类型的列进行相关性计算
    numeric_df = combined_df[property_names].select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Molecular Property Correlations', fontsize=16)
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'property_correlations.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"已保存相关性热图: {output_file}")
    
    # t-SNE 分布图 (基于分子指纹)
    print("\n开始生成基于分子指纹的t-SNE分布图...")

    if 'SMILES' not in combined_df.columns:
        print("警告：combined_df中缺少'SMILES'列，无法生成基于指纹的t-SNE图。")
    else:
        fingerprints = []
        valid_indices = []  # 跟踪具有有效SMILES的行索引
        valid_smiles = []   # 保存有效的SMILES
        
        print("正在计算分子指纹...")
        for idx, smiles in combined_df['SMILES'].items():
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 使用Morgan指纹 (类似于ECFP4)，半径为2，2048位
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(np.array(fp))
                valid_indices.append(idx)
                valid_smiles.append(smiles)
        
        print(f"成功计算了 {len(fingerprints)} 个分子指纹。")

        if not fingerprints:
            print("警告：未能从SMILES生成任何有效的分子指纹，跳过t-SNE图。")
        else:
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
                            max_iter=1000,
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

    # 创建各来源数据的统计比较表格
    combined_df_for_stats = combined_df.copy()
    combined_df_for_stats['source'] = combined_df_for_stats['clean_source']
    # 仅对数值列进行聚合
    stats_df = combined_df_for_stats.groupby('source')[property_names].agg(['mean', 'median', 'std', 'min', 'max'])
    stats_file = os.path.join(output_dir, 'property_statistics.csv')
    stats_df.to_csv(stats_file)
    print(f"\n已保存统计数据: {stats_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载NP_score模型（如果可用）
    np_model = None
    if 'readNPModel' in globals() and readNPModel is not None:
        try:
            model_file = os.path.join(script_dir, "NP_Score", "publicnp.model.gz")
            if os.path.exists(model_file):
                np_model = readNPModel(model_file)
                print("成功加载NP_score模型")
            else:
                print(f"警告: NP_score模型文件不存在: {model_file}")
        except Exception as e:
            print(f"加载NP_score模型时出错: {e}")
    
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
            np_model, 
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
    output_file = os.path.join(args.output_dir, 'combined_properties.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"已将合并数据保存到: {output_file}")
    
    # 绘制分子性质分布图
    plot_property_distributions(combined_df, args.output_dir)
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()