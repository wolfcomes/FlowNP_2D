import torch
import torch.nn as nn
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import pytorch_lightning as pl
from src.models.flowmol import FlowMol
from src.model_utils.load import read_config_file
import os

class ActivationAnalyzer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.setup_hooks()
        
    def setup_hooks(self):
        """注册前向钩子来捕获各层激活"""
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]  # 取第一个输出
                self.activations[name] = {
                    'output': output.detach().cpu(),
                    'module_type': type(module).__name__
                }
            return hook
        
        # 遍历所有模块并注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                 nn.LSTM, nn.GRU, nn.Embedding,
                                 nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Mish)):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def analyze_forward_pass(self, batch_size: int = 100, n_timesteps: int = 20):
        """执行前向传播并分析激活"""
        self.activations.clear()
        
        # 准备模拟输入数据
        device = next(self.model.parameters()).device
        
        try:
            # 尝试使用模型的采样方法来获得真实的激活
            if hasattr(self.model, 'sample_random_sizes'):
                with torch.no_grad():
                    molecules = self.model.sample_random_sizes(
                        batch_size, 
                        device=device, 
                        n_timesteps=n_timesteps,
                        xt_traj=False,
                        ep_traj=False
                    )
            else:
                # 备用方法：创建模拟输入
                self._analyze_with_mock_input(device, batch_size)
                
        except Exception as e:
            print(f"使用真实采样失败: {e}, 使用模拟输入")
            self._analyze_with_mock_input(device, batch_size)
    
    def _analyze_with_mock_input(self, device, batch_size):
        """使用模拟输入进行分析"""
        # 这里需要根据你的模型结构调整
        # 假设模型需要原子数量和坐标
        n_atoms = torch.randint(5, 50, (batch_size,), device=device)
        
        # 创建模拟的原子坐标和特征
        coords_list = []
        features_list = []
        
        for n in n_atoms:
            coords = torch.randn(n, 3, device=device)  # 3D坐标
            features = torch.randn(n, 10, device=device)  # 假设10个特征
            
            coords_list.append(coords)
            features_list.append(features)
        
        # 执行前向传播（需要根据你的模型调整）
        try:
            with torch.no_grad():
                # 这里调用模型的前向方法
                if hasattr(self.model, 'forward'):
                    # 需要根据你的模型签名调整
                    pass
        except Exception as e:
            print(f"模拟前向传播失败: {e}")

    def compute_layer_statistics(self) -> pd.DataFrame:
        """计算各层的统计信息"""
        stats = []
        
        for name, activation_data in self.activations.items():
            output = activation_data['output']
            module_type = activation_data['module_type']
            
            if output.numel() == 0:
                continue
                
            flat_output = output.flatten()
            stats.append({
                'layer_name': name,
                'module_type': module_type,
                'mean': flat_output.mean().item(),
                'std': flat_output.std().item(),
                'min': flat_output.min().item(),
                'max': flat_output.max().item(),
                'abs_mean': flat_output.abs().mean().item(),
                'dead_ratio': (flat_output.abs() < 1e-6).float().mean().item(),
                'saturated_ratio': (flat_output.abs() > 10.0).float().mean().item(),
                'num_parameters': self._count_parameters(name),
                'output_shape': str(list(output.shape))
            })
        
        return pd.DataFrame(stats)
    
    def _count_parameters(self, layer_name: str) -> int:
        """计算指定层的参数数量"""
        try:
            module = dict(self.model.named_modules())[layer_name]
            return sum(p.numel() for p in module.parameters())
        except:
            return 0
    
    def analyze_activation_distributions(self):
        """分析激活分布模式"""
        distribution_info = {}
        
        for name, activation_data in self.activations.items():
            output = activation_data['output']
            if output.numel() == 0:
                continue
                
            flat_output = output.flatten().numpy()
            
            # 计算分布特征
            hist, bins = np.histogram(flat_output, bins=50, density=True)
            peak_height = hist.max()
            peak_position = bins[np.argmax(hist)]
            
            distribution_info[name] = {
                'module_type': activation_data['module_type'],
                'skewness': float(pd.Series(flat_output).skew()),
                'kurtosis': float(pd.Series(flat_output).kurtosis()),
                'peak_height': peak_height,
                'peak_position': peak_position,
                'entropy': -np.sum(hist * np.log(hist + 1e-10)),
                'is_bimodal': self._check_bimodal(hist)
            }
        
        return distribution_info
    
    def _check_bimodal(self, hist: np.ndarray, threshold: float = 0.5) -> bool:
        """检查分布是否是双峰的"""
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(hist[i])
        
        if len(peaks) >= 2:
            # 检查第二高峰是否足够高
            peaks_sorted = sorted(peaks, reverse=True)
            return peaks_sorted[1] > threshold * peaks_sorted[0]
        return False

def create_visualizations(stats_df: pd.DataFrame, distribution_info: Dict, output_dir: Path):
    """创建可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. 各层激活均值分布
    ax = axes[0]
    sns.boxplot(data=stats_df, x='module_type', y='mean', ax=ax)
    ax.set_title('Activation Means by Layer Type')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. 各层激活标准差分布
    ax = axes[1]
    sns.boxplot(data=stats_df, x='module_type', y='std', ax=ax)
    ax.set_title('Activation Std by Layer Type')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. 死亡神经元比例
    ax = axes[2]
    sns.boxplot(data=stats_df, x='module_type', y='dead_ratio', ax=ax)
    ax.set_title('Dead Neuron Ratio by Layer Type')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. 参数数量分布
    ax = axes[3]
    stats_df['log_params'] = np.log10(stats_df['num_parameters'] + 1)
    sns.scatterplot(data=stats_df, x='log_params', y='std', hue='module_type', ax=ax, s=60)
    ax.set_title('Parameter Count vs Activation Std')
    ax.set_xlabel('Log10(Parameter Count + 1)')
    
    # 5. 偏度和峰度散点图
    ax = axes[4]
    skewness = [info['skewness'] for info in distribution_info.values()]
    kurtosis = [info['kurtosis'] for info in distribution_info.values()]
    module_types = [info['module_type'] for info in distribution_info.values()]
    
    scatter = ax.scatter(skewness, kurtosis, c=[hash(t) % 10 for t in module_types], cmap='tab10', s=60)
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.set_title('Distribution Shape Analysis')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 6. 层重要性热图（基于激活方差）
    ax = axes[5]
    important_layers = stats_df.nlargest(20, 'std')[['layer_name', 'std', 'module_type']]
    pivot_data = important_layers.pivot_table(
        values='std', 
        index='layer_name', 
        columns='module_type', 
        aggfunc='mean'
    ).fillna(0)
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=ax, cmap='YlOrRd')
        ax.set_title('Top 20 Layers by Activation Variance')
    else:
        ax.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center')
        ax.set_title('Top Layers Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建详细的统计表格
    stats_df.to_csv(output_dir / 'layer_statistics.csv', index=False)
    
    # 创建分布信息表格
    dist_df = pd.DataFrame.from_dict(distribution_info, orient='index')
    dist_df.to_csv(output_dir / 'distribution_analysis.csv')

def analyze_model_diversity(model: FlowMol, output_dir: Path, n_samples: int = 100):
    """分析模型多样性相关特征"""
    print("开始模型激活分析...")
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建分析器
    analyzer = ActivationAnalyzer(model)
    
    # 执行分析
    print("执行前向传播分析...")
    analyzer.analyze_forward_pass(batch_size=n_samples)
    
    # 计算统计信息
    print("计算层统计信息...")
    stats_df = analyzer.compute_layer_statistics()
    
    # 分析激活分布
    print("分析激活分布模式...")
    distribution_info = analyzer.analyze_activation_distributions()
    
    # 创建可视化
    print("生成可视化图表...")
    create_visualizations(stats_df, distribution_info, output_dir)
    
    # 清理钩子
    analyzer.remove_hooks()
    
    # 生成分析报告
    generate_analysis_report(stats_df, distribution_info, output_dir)
    
    print(f"分析完成！结果保存在: {output_dir}")

def generate_analysis_report(stats_df: pd.DataFrame, distribution_info: Dict, output_dir: Path):
    """生成详细的分析报告"""
    report = []
    report.append("=== 模型激活分析报告 ===\n")
    
    # 总体统计
    report.append("## 总体统计")
    report.append(f"分析层数: {len(stats_df)}")
    report.append(f"激活函数类型: {', '.join(stats_df['module_type'].unique())}")
    report.append(f"总参数数量: {stats_df['num_parameters'].sum():,}")
    report.append("")
    
    # 潜在问题检测
    report.append("## 潜在问题检测")
    
    # 检查死亡神经元
    high_dead_ratio = stats_df[stats_df['dead_ratio'] > 0.5]
    if not high_dead_ratio.empty:
        report.append("🚨 高死亡神经元比例 (>50%):")
        for _, row in high_dead_ratio.iterrows():
            report.append(f"   - {row['layer_name']}: {row['dead_ratio']:.1%}")
    else:
        report.append("✅ 死亡神经元比例正常")
    
    # 检查激活饱和
    high_saturation = stats_df[stats_df['saturated_ratio'] > 0.3]
    if not high_saturation.empty:
        report.append("🚨 高饱和神经元比例 (>30%):")
        for _, row in high_saturation.iterrows():
            report.append(f"   - {row['layer_name']}: {row['saturated_ratio']:.1%}")
    else:
        report.append("✅ 激活饱和比例正常")
    
    # 检查双峰分布（可能表示模式分离）
    bimodal_layers = [name for name, info in distribution_info.items() if info['is_bimodal']]
    if bimodal_layers:
        report.append("📊 检测到双峰分布（可能表示模式分离）:")
        for layer in bimodal_layers[:5]:  # 只显示前5个
            report.append(f"   - {layer}")
    else:
        report.append("📊 未检测到明显的双峰分布")
    
    # 最重要的层（基于激活方差）
    report.append("\n## 最重要的层（基于激活方差）")
    top_layers = stats_df.nlargest(10, 'std')
    for i, (_, row) in enumerate(top_layers.iterrows(), 1):
        report.append(f"{i}. {row['layer_name']} ({row['module_type']}): std={row['std']:.3f}")
    
    # 改进建议
    report.append("\n## 改进建议")
    
    dead_layers = stats_df[stats_df['dead_ratio'] > 0.8]
    if not dead_layers.empty:
        report.append("🔧 考虑替换以下层的激活函数:")
        for _, row in dead_layers.iterrows():
            report.append(f"   - {row['layer_name']}: 当前 {row['module_type']}, 建议尝试 LeakyReLU")
    
    low_variance_layers = stats_df[stats_df['std'] < 0.01]
    if not low_variance_layers.empty:
        report.append("🔧 以下层激活方差过低，可能限制表达能力:")
        for _, row in low_variance_layers.iterrows():
            report.append(f"   - {row['layer_name']}: std={row['std']:.3f}")
    
    # 写入报告文件
    report_text = '\n'.join(report)
    with open(output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)

def parse_args():
    p = argparse.ArgumentParser(description='Model Activation Analysis Script')
    p.add_argument('--model_dir', type=Path, help='Path to model directory')
    p.add_argument('--checkpoint', type=Path, help='Path to checkpoint file')
    p.add_argument('--output_dir', type=Path, help='Path to output directory', default=Path('activation_analysis'))
    p.add_argument('--n_samples', type=int, default=100, help='Number of samples for analysis')
    
    args = p.parse_args()
    
    if args.model_dir is not None and args.checkpoint is not None:
        raise ValueError('Only specify model_dir or checkpoint but not both')
    
    if args.model_dir is None and args.checkpoint is None:
        raise ValueError('Must specify model_dir or checkpoint')
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # 加载模型
    if args.model_dir is not None:
        model_dir = args.model_dir
        checkpoint_file = args.model_dir / 'checkpoints' / 'last.ckpt'
    else:
        model_dir = args.checkpoint.parent.parent
        checkpoint_file = args.checkpoint
    
    print(f"Loading model from: {checkpoint_file}")
    model = FlowMol.load_from_checkpoint(checkpoint_file)
    
    # 创建输出目录
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行分析
    analyze_model_diversity(model, output_dir, args.n_samples)