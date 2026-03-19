import argparse
import pickle
from pathlib import Path
import json
import warnings

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

# 导入GPU相关的库
try:
    import cuml
    import cupy as cp
    print("Successfully imported cuML and CuPy. GPU acceleration is available.")
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: cuML or CuPy not found. Falling back to CPU. Clustering will be very slow.")
    GPU_AVAILABLE = False
    # 如果没有GPU，使用scikit-learn的KMeans
    from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate molecular weights based on rarity using GPU-accelerated clustering')
    parser.add_argument('--config', type=Path, required=True, help='Config file path')
    parser.add_argument('--output_dir', type=Path, default=None, help='Output directory for weights')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs for fingerprint/descriptor calculation')
    # K-means的n_clusters参数
    parser.add_argument('--n_clusters', type=int, default=1000, help='Number of clusters for K-means')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for K-means')
    return parser.parse_args()

def calculate_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    """Calculate Morgan fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        # 将RDKit指纹对象转换为numpy数组，便于后续处理
        arr = np.zeros((1,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None

def calculate_descriptors(smiles):
    """Calculate molecular descriptors for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
        ]
        return np.array(descriptors)
    except:
        return None

def gpu_kmeans_clustering(fingerprints_np, n_clusters=1000, random_state=42):
    """
    Perform K-means clustering on fingerprints using GPU (cuML).
    
    Args:
        fingerprints_np (np.ndarray): A 2D numpy array of fingerprints (n_molecules, n_bits).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.
                           
    Returns:
        np.ndarray: Cluster labels for each molecule.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("cuML/CuPy not available for GPU clustering. Please check your RAPIDS installation.")

    print("--- Starting GPU K-means Clustering ---")
    
    # 1. 将数据传输到GPU
    print(f"Transferring {fingerprints_np.shape[0]} fingerprints to GPU...")
    fingerprints_gpu = cp.asarray(fingerprints_np, dtype=cp.float32)

    # 2. 初始化K-means
    print(f"Initializing cuML KMeans with n_clusters={n_clusters} and random_state={random_state}")
    kmeans = cuml.KMeans(n_clusters=n_clusters, random_state=random_state)

    # 3. 执行聚类
    print("Fitting KMeans model on GPU...")
    kmeans.fit(fingerprints_gpu)
    
    # 4. 获取预测标签
    labels = kmeans.labels_
    
    # 5. 将结果传回CPU
    print("Transferring labels back to CPU...")
    labels_cpu = labels.get()
    
    # 清理GPU内存
    del fingerprints_gpu
    del labels
    cp.get_default_memory_pool().free_all_blocks()
    
    print("--- GPU K-means Clustering Finished ---")
    return labels_cpu, kmeans

def cpu_kmeans_clustering(fingerprints_np, n_clusters=1000, random_state=42):
    """
    Perform K-means clustering on fingerprints using CPU (scikit-learn).
    """
    print("--- Starting CPU K-means Clustering ---")
    
    print(f"Initializing KMeans with n_clusters={n_clusters} and random_state={random_state}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    print("Fitting KMeans model on CPU...")
    kmeans.fit(fingerprints_np)
    
    labels_cpu = kmeans.labels_
    
    print("--- CPU K-means Clustering Finished ---")
    return labels_cpu, kmeans

def generate_molecular_weights(config_path, output_dir=None, n_jobs=1, n_clusters=1000, random_state=42):
    """Generate molecular weights based on cluster rarity."""
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_config = config['dataset']
    processed_data_dir = Path(dataset_config['processed_data_dir'])
    
    if output_dir is None:
        output_dir = processed_data_dir
    output_dir.mkdir(exist_ok=True)
    
    train_file = processed_data_dir / 'train_data_processed.pt'
    print(f"Loading training data from {train_file}")
    train_data = torch.load(train_file)
    train_smiles = train_data['smiles']
    
    print(f"Loaded {len(train_smiles)} training molecules")
    
    # --- 分子描述符计算 (保持不变) ---
    print("Calculating molecular descriptors...")
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(calculate_descriptors, train_smiles), total=len(train_smiles), desc="Calculating descriptors"))
    
    descriptors_list, valid_indices, valid_smiles = [], [], []
    for i, desc in enumerate(results):
        if desc is not None:
            descriptors_list.append(desc)
            valid_indices.append(i)
            valid_smiles.append(train_smiles[i])
    
    descriptors_array = np.array(descriptors_list)
    print(f"Successfully calculated descriptors for {len(descriptors_array)} molecules")
    
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors_array)
    
    # --- 摩根指纹计算 ---
    print("Calculating Morgan fingerprints for clustering...")
    with Pool(n_jobs) as pool:
        fp_results = list(tqdm(pool.imap(calculate_morgan_fingerprint, valid_smiles), total=len(valid_smiles), desc="Calculating fingerprints"))

    fingerprints_list, valid_fp_indices = [], []
    for i, fp in enumerate(fp_results):
        if fp is not None:
            fingerprints_list.append(fp)
            valid_fp_indices.append(i)
            
    # 将指纹列表转换为一个大的2D numpy数组
    fingerprints_np = np.vstack(fingerprints_list)
    
    final_valid_indices = [valid_indices[i] for i in valid_fp_indices]
    final_descriptors = descriptors_scaled[valid_fp_indices]
    final_smiles = [valid_smiles[i] for i in valid_fp_indices]
    
    print(f"Final dataset for clustering: {len(final_smiles)} molecules")
    
    # --- K-means聚类 ---
    if GPU_AVAILABLE:
        cluster_labels, kmeans_model = gpu_kmeans_clustering(
            fingerprints_np,
            n_clusters=n_clusters,
            random_state=random_state
        )
    else:
        cluster_labels, kmeans_model = cpu_kmeans_clustering(
            fingerprints_np,
            n_clusters=n_clusters,
            random_state=random_state
        )
    
    # --- 基于K-means结果计算权重 ---
    print("Calculating weights based on cluster rarity...")
    
    # 获取每个簇的大小
    unique_labels, cluster_sizes_arr = np.unique(cluster_labels, return_counts=True)
    
    # 创建从簇标签到大小的映射字典
    cluster_size_map = dict(zip(unique_labels, cluster_sizes_arr))
    
    weights = np.ones(len(final_smiles))
    
    for i, label in enumerate(tqdm(cluster_labels, desc="Assigning weights")):
        cluster_size = cluster_size_map[label]
        # 使用簇大小的倒数作为权重基础（小簇权重高，大簇权重低）
        # 添加平滑处理避免极端值
        weights[i] = 1.0 / (cluster_size**0.5)  # 平方根倒数

    # 添加权重下限和上限，避免极端值
    min_weight = 0.0001  # 最小权重
    max_weight = 10.0  # 最大权重
    weights = np.clip(weights, min_weight, max_weight)

    # 归一化到平均值为1
    weights = weights / weights.mean()
    
    # 创建适用于所有训练分子的完整权重数组
    full_weights = np.ones(len(train_smiles))
    for idx, weight in zip(final_valid_indices, weights):
        full_weights[idx] = weight
        
    # --- 保存结果 ---
    weights_file = output_dir / 'train_molecular_weights.pt'
    torch.save({
        'weights': torch.from_numpy(full_weights.astype(np.float32)),
        'cluster_sizes': cluster_sizes_arr.tolist(),
        'n_clusters': n_clusters,
        'valid_indices': final_valid_indices,
        'all_smiles': train_smiles,
        'cluster_labels': cluster_labels
    }, weights_file)
    
    # 计算一些统计信息
    cluster_sizes_sorted = sorted(cluster_sizes_arr, reverse=True)
    n_small_clusters = sum(1 for size in cluster_sizes_arr if size <= 5)  # 小簇数量
    
    cluster_info = {
        'method': 'KMeans',
        'n_clusters': int(n_clusters),
        'cluster_sizes': [int(size) for size in cluster_sizes_sorted],
        'min_cluster_size': int(min(cluster_sizes_arr)),
        'max_cluster_size': int(max(cluster_sizes_arr)),
        'mean_cluster_size': float(np.mean(cluster_sizes_arr)),
        'median_cluster_size': float(np.median(cluster_sizes_arr)),
        'n_small_clusters': int(n_small_clusters),
        'random_state': int(random_state)
    }
        
    with open(output_dir / 'cluster_info.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    print(f"\nGenerated weights for {len(train_smiles)} molecules")
    print(f"Found {n_clusters} clusters.")
    print(f"Cluster size range: {min(cluster_sizes_arr)} - {max(cluster_sizes_arr)}")
    print(f"Weights saved to {weights_file}")
    
    return full_weights

if __name__ == "__main__":
    args = parse_args()
    generate_molecular_weights(
        config_path=args.config,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )