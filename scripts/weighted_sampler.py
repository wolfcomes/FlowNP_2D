import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import json
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sample molecules based on precomputed weights')
    parser.add_argument('--weights_file', type=Path, required=True, help='Path to the weights file (train_molecular_weights.pt)')
    parser.add_argument('--output_file', type=Path, required=True, help='Output CSV file path')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of molecules to sample')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def load_weights_data(weights_file):
    """Load the weights data from file."""
    print(f"Loading weights data from {weights_file}")
    data = torch.load(weights_file)
    
    weights = data['weights'].numpy()
    all_smiles = data['all_smiles']
    cluster_labels = data.get('cluster_labels', None)
    valid_indices = data.get('valid_indices', None)
    
    print(f"Loaded {len(all_smiles)} molecules with weights")
    print(f"Weight statistics: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}")
    
    return weights, all_smiles, cluster_labels, valid_indices

def weighted_sampling(smiles_list, weights, n_samples, random_state=42):
    """Perform weighted sampling of molecules."""
    print(f"Performing weighted sampling of {n_samples} molecules...")
    
    # 确保权重是概率分布（和为1）
    sampling_probs = (1/weights**4) / (1/weights**4).sum()
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 进行加权采样
    sampled_indices = np.random.choice(
        len(smiles_list), 
        size=n_samples, 
        replace=True,  # 允许重复采样
        p=sampling_probs
    )
    
    # 获取采样的分子
    sampled_smiles = [smiles_list[i] for i in sampled_indices]
    sampled_weights = [weights[i] for i in sampled_indices]
    
    return sampled_smiles, sampled_weights, sampled_indices

def analyze_sampling_results(sampled_smiles, sampled_weights, all_weights, cluster_labels=None):
    """Analyze the sampling results."""
    print("\n=== Sampling Analysis ===")
    
    # 基本统计
    print(f"Sampled {len(sampled_smiles)} molecules")
    print(f"Sampled weight range: {min(sampled_weights):.6f} - {max(sampled_weights):.6f}")
    print(f"Average sampled weight: {np.mean(sampled_weights):.6f}")
    print(f"Median sampled weight: {np.median(sampled_weights):.6f}")
    
    # 检查重复样本
    unique_sampled = set(sampled_smiles)
    print(f"Unique molecules sampled: {len(unique_sampled)}/{len(sampled_smiles)}")
    
    # 检查权重分布
    high_weight_threshold = np.percentile(all_weights, 90)  # 前10%的高权重
    high_weight_samples = sum(1 for w in sampled_weights if w >= high_weight_threshold)
    print(f"High-weight samples (>P90): {high_weight_samples}/{len(sampled_weights)}")
    
    return {
        'n_samples': len(sampled_smiles),
        'n_unique': len(unique_sampled),
        'weight_min': float(min(sampled_weights)),
        'weight_max': float(max(sampled_weights)),
        'weight_mean': float(np.mean(sampled_weights)),
        'high_weight_count': high_weight_samples
    }

def main():
    args = parse_args()
    
    # 加载权重数据
    weights, all_smiles, cluster_labels, valid_indices = load_weights_data(args.weights_file)
    
    # 进行加权采样
    sampled_smiles, sampled_weights, sampled_indices = weighted_sampling(
        all_smiles, weights, args.n_samples, args.random_state
    )
    
    # 分析采样结果
    analysis_results = analyze_sampling_results(sampled_smiles, sampled_weights, weights, cluster_labels)
    
    # 创建输出DataFrame
    output_data = []
    for i, (smiles, weight, idx) in enumerate(zip(sampled_smiles, sampled_weights, sampled_indices)):
        output_data.append({
            'sample_id': i,
            'original_index': idx,
            'smiles': smiles,
            'weight': weight,
            'cluster_label': cluster_labels[i] if cluster_labels is not None and i < len(cluster_labels) else -1
        })
    
    df = pd.DataFrame(output_data)
    
    # 保存为CSV
    df.to_csv(args.output_file, index=False)
    print(f"\nSampled molecules saved to {args.output_file}")
    
    # 保存采样统计信息
    stats_file = args.output_file.with_suffix('.sampling_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Sampling statistics saved to {stats_file}")
    
    # 打印一些示例
    print("\n=== First 10 sampled molecules ===")
    for i, row in df.head(10).iterrows():
        print(f"{i+1:2d}. SMILES: {row['smiles'][:50]}... Weight: {row['weight']:.6f}")

if __name__ == "__main__":
    main()