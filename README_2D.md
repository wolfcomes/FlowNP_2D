# FlowNP_2D

`FlowNP_2D` 现在只保留纯 2D 分子生成主路径：

- 不再训练 `x` 坐标特征
- 不再走 `pocket/crossdock` 条件生成路径
- 不再使用 GVP、RBF 距离特征或动态图重建

当前训练特征固定为：

- `a`: atom type
- `c`: atom charge
- `e`: bond type

## 主路径

活跃运行链路如下：

`process_qm9.py / process_coconut.py -> MoleculeDataset -> MoleculeDataModule -> FlowMol -> CTMCVectorField2D -> train.py / test.py`

## 预处理输出

预处理脚本会为每个数据集产出下列文件：

- `train_data_processed.pt`
- `val_data_processed.pt`
- `test_data_processed.pt`
- `train_data_n_atoms_histogram.pt`
- `train_data_marginal_dists.pt`

训练实际读取的字段为：

- `smiles`
- `atom_types`
- `atom_charges`
- `bond_types`
- `bond_idxs`
- `node_idx_array`
- `edge_idx_array`

如果数据里还保留 `positions`，它只作为兼容字段存在，不参与 2D 训练。

## 预处理命令

QM9:

```bash
python process_qm9.py --config configs/qm9_ctmc.yaml
```

COCONUT:

```bash
python process_coconut.py --config configs/coconut_ctmc.yaml
```

## 训练命令

QM9:

```bash
python train.py --config configs/qm9_ctmc.yaml
```

COCONUT:

```bash
python train.py --config configs/coconut_ctmc.yaml
```

调试训练：

```bash
python train.py --config configs/qm9_ctmc.yaml --debug
```

## 采样命令

```bash
python test.py --checkpoint /path/to/checkpoints/last.ckpt --n_mols 100 --output_file sampled.sdf
```

导出时如果图中没有显式坐标，RDKit 会自动生成 2D 坐标后再写出 SDF。

## 配置说明

当前活跃配置只有：

- `configs/qm9_ctmc.yaml`
- `configs/coconut_ctmc.yaml`

这些配置已经移除了：

- `x` 的 scheduler / prior / loss
- `rbf_*`
- `n_vec_channels`
- `n_hidden_scalars`
- `n_message_gvps`
- `n_update_gvps`
- `enable_dynamic_graph`
- `knn_connectivity`
- `sde`

2D 向量场的核心参数为：

- `n_hidden`
- `n_hidden_edge_feats`
- `n_recycles`
- `n_molecule_updates`
- `convs_per_update`
- `separate_mol_updaters`
- `message_norm`
- `stochasticity`
- `high_confidence_threshold`

## 兼容性

- 旧的 3D checkpoint 不兼容当前 2D 配置
- 旧的 pocket/crossdock 训练路径已经从主流程移除
