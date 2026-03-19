import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical

from src.data_processing.utils import get_batch_idxs, get_edge_batch_idxs
from src.models.interpolant_scheduler import InterpolantScheduler
from src.models.utils import campbell_step, gat_step, build_continuous_inv_temp_func, build_cat_temp_schedule, build_fw_schedule

import dgl
import dgl.function as fn
from typing import Union, Callable


class ScalarGVPConv2D(nn.Module):
    """Scalar-only analogue of GVPConv for the 2D categorical model."""

    def __init__(
        self,
        scalar_size: int,
        edge_feat_size: int,
        message_norm: Union[float, str] = 10,
        dropout: float = 0.0,
        use_dst_feats: bool = False,
    ):
        super().__init__()

        self.scalar_size = scalar_size
        self.edge_feat_size = edge_feat_size
        self.message_norm = message_norm
        self.use_dst_feats = use_dst_feats

        message_input_dim = scalar_size + edge_feat_size
        if use_dst_feats:
            message_input_dim += scalar_size

        self.edge_message = nn.Sequential(
            nn.Linear(message_input_dim, scalar_size),
            nn.SiLU(),
            nn.Linear(scalar_size, scalar_size),
            nn.SiLU(),
        )
        self.node_update_fn = nn.Sequential(
            nn.Linear(scalar_size, scalar_size),
            nn.SiLU(),
            nn.Linear(scalar_size, scalar_size),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.message_layer_norm = nn.LayerNorm(scalar_size)
        self.update_layer_norm = nn.LayerNorm(scalar_size)

        if isinstance(self.message_norm, str) and self.message_norm not in ["mean", "sum"]:
            raise ValueError(
                f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}"
            )

        if self.message_norm == "mean":
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

    def forward(self, g: dgl.DGLGraph, scalar_feats: torch.Tensor, edge_feats: torch.Tensor):
        with g.local_scope():
            g.ndata["h"] = scalar_feats
            if edge_feats is not None:
                g.edata["a"] = edge_feats

            if g.num_edges() > 0:
                g.apply_edges(self.message)
                g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"))
                scalar_msg = g.ndata["scalar_msg"]
            else:
                scalar_msg = torch.zeros_like(scalar_feats)

            if not isinstance(self.message_norm, str):
                scalar_msg = scalar_msg / self.message_norm

            scalar_msg = self.dropout(scalar_msg)
            scalar_feat = self.message_layer_norm(g.ndata["h"] + scalar_msg)

            scalar_residual = self.node_update_fn(scalar_feat)
            scalar_residual = self.dropout(scalar_residual)
            scalar_feat = self.update_layer_norm(scalar_feat + scalar_residual)

        return scalar_feat

    def message(self, edges):
        scalar_feats = [edges.src["h"]]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data["a"])
        if self.use_dst_feats:
            scalar_feats.append(edges.dst["h"])

        scalar_feats = torch.cat(scalar_feats, dim=1)
        scalar_message = self.edge_message(scalar_feats)
        return {"scalar_msg": scalar_message}


class CTMCVectorField2D(nn.Module):
    """
    2D版本的CTMC向量场,用于2D分子生成。
    去除所有3D几何信息,只处理原子类型、电荷和键类型。
    使用去向量化的GVP风格消息传递。
    """

    def __init__(self, n_atom_types: int,
                    canonical_feat_order: list,
                    interpolant_scheduler: InterpolantScheduler,
                    n_charges: int,
                    n_bond_types: int, 
                    n_random_node_feats: int = 8,
                    n_hidden: int = 256,
                    n_hidden_edge_feats: int = 128,
                    n_recycles: int = 1,
                    n_molecule_updates: int = 2, 
                    convs_per_update: int = 2,
                    separate_mol_updaters: bool = False,
                    message_norm: Union[float, str] = 100,
                    exclude_charges: bool = False,
                    has_mask: bool = True,
                    self_conditioning: bool = False,
                    stochasticity: float = 0.0, 
                    high_confidence_threshold: float = 0.0, 
                    dfm_type: str = 'campbell', 
                    cat_temperature_schedule: Union[str, Callable, float] = 1.0,
                    cat_temp_decay_max: float = 0.8,
                    cat_temp_decay_a: float = 2,
                    forward_weight_schedule: Union[str, Callable, float] = 'beta',
                    fw_beta_a: float = 0.25, fw_beta_b: float = 0.25, fw_beta_max: float = 10.0,
                    **kwargs):
        
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_random_node_feats = n_random_node_feats
        self.n_hidden = n_hidden
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters = separate_mol_updaters
        self.exclude_charges = exclude_charges
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = list(canonical_feat_order)
        self.sde = False  # 2D版本不使用SDE
        self.self_conditioning = self_conditioning

        if self.exclude_charges:
            self.n_charges = 0
            n_charges = 0

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.continuous_inv_temp_schedule = None
        self.continuous_inv_temp_func = build_continuous_inv_temp_func(None, None)

        self.n_cat_feats = {
            'a': n_atom_types,
            'c': n_charges,
            's': 2,
            'e': n_bond_types,
            'se': 2,
        }

        self.node_feats = [feat for feat in ['a', 'c', 's'] if feat in self.canonical_feat_order]
        self.edge_feats = [feat for feat in ['e', 'se'] if feat in self.canonical_feat_order]

        n_mask_feats = int(has_mask)
        t_dim = 1
        node_input_dim = sum(self.n_cat_feats[feat] + n_mask_feats for feat in self.node_feats) + t_dim
        edge_input_dim = sum(self.n_cat_feats[feat] + n_mask_feats for feat in self.edge_feats)

        # 节点特征嵌入 (2D版本: 只处理分类特征和时间)
        self.scalar_embedding = nn.Sequential(
            nn.Linear(node_input_dim, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.LayerNorm(n_hidden)
        )
        if self.n_random_node_feats > 0:
            self.random_node_embedding = nn.Sequential(
                nn.Linear(self.n_random_node_feats, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, n_hidden),
            )
            self.random_node_gate = nn.Parameter(torch.zeros(1))
        else:
            self.random_node_embedding = None
            self.random_node_gate = None

        # 边特征嵌入
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_dim, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats)
        )

        # 使用去向量化的GVP风格卷积层
        self.conv_layers = []
        for conv_idx in range(convs_per_update*n_molecule_updates):
            self.conv_layers.append(
                ScalarGVPConv2D(
                    scalar_size=n_hidden,
                    edge_feat_size=n_hidden_edge_feats,
                    message_norm=message_norm,
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # 边更新层
        self.edge_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.edge_updaters.append(EdgeUpdate2D(n_hidden, n_hidden_edge_feats))

        # 输出头
        self.node_output_heads = nn.ModuleDict(
            {
                feat: nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.SiLU(),
                    nn.Linear(n_hidden, self.n_cat_feats[feat]),
                )
                for feat in self.node_feats
            }
        )

        self.edge_output_heads = nn.ModuleDict(
            {
                feat: nn.Sequential(
                    nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.Linear(n_hidden_edge_feats, self.n_cat_feats[feat]),
                )
                for feat in self.edge_feats
            }
        )

        self.eta = stochasticity
        self.hc_thresh = high_confidence_threshold
        self.dfm_type = dfm_type

        # 配置分类特征的温度调度
        self.cat_temperature_schedule = cat_temperature_schedule
        self.cat_temp_decay_max = cat_temp_decay_max
        self.cat_temp_decay_a = cat_temp_decay_a
        self.cat_temp_func = build_cat_temp_schedule(
            cat_temperature_schedule=cat_temperature_schedule,
            cat_temp_decay_max=cat_temp_decay_max,
            cat_temp_decay_a=cat_temp_decay_a)
        
        # 配置前向权重调度
        self.forward_weight_schedule = forward_weight_schedule
        self.fw_beta_a = fw_beta_a
        self.fw_beta_b = fw_beta_b
        self.fw_beta_max = fw_beta_max
        self.forward_weight_func = build_fw_schedule(
            forward_weight_schedule=forward_weight_schedule,
            fw_beta_a=fw_beta_a,
            fw_beta_b=fw_beta_b,
            fw_beta_max=fw_beta_max)

        if self.dfm_type not in ['campbell', 'gat']:
            raise ValueError(f"Invalid dfm_type: {self.dfm_type}")

        self.mask_idxs = {
            'a': self.n_atom_types,
            'c': self.n_charges,
            's': self.n_cat_feats['s'],
            'e': self.n_bond_types,
            'se': self.n_cat_feats['se'],
        }
    
    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        """
        对条件路径 p(g_t|g_0,g_1) 进行采样 (2D版本,无需3D坐标对齐)。
        """
        _, alpha_t = self.interpolant_scheduler.interpolant_weights(t)

        device = g.device

        # 处理节点分类特征
        for feat in self.node_feats:
            feat_idx = self.canonical_feat_order.index(feat)
            if self.mask_idxs[feat] == 0:
                continue
            xt = g.ndata[f'{feat}_1_true'].argmax(-1)
            alpha_t_feat = alpha_t[:, feat_idx][node_batch_idx]
            xt[torch.rand(g.num_nodes(), device=device) < 1 - alpha_t_feat] = self.mask_idxs[feat]
            g.ndata[f'{feat}_t'] = one_hot(
                xt, num_classes=self.n_cat_feats[feat] + 1
            ).float()

        # 处理边特征
        num_edges = int(g.num_edges() // 2)
        for feat in self.edge_feats:
            feat_idx = self.canonical_feat_order.index(feat)
            alpha_t_e = alpha_t[:, feat_idx][edge_batch_idx][upper_edge_mask]
            et_upper = g.edata[f'{feat}_1_true'][upper_edge_mask].argmax(-1)
            et_upper[torch.rand(num_edges, device=device) < 1 - alpha_t_e] = self.mask_idxs[feat]

            n, d = g.edata[f'{feat}_1_true'].shape
            e_t = torch.zeros((n, d + 1), dtype=g.edata[f'{feat}_1_true'].dtype, device=g.device)
            et_upper_onehot = one_hot(et_upper, num_classes=self.n_cat_feats[feat] + 1).float()
            e_t[upper_edge_mask] = et_upper_onehot
            e_t[~upper_edge_mask] = et_upper_onehot
            g.edata[f'{feat}_t'] = e_t

        return g

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, 
                 upper_edge_mask: torch.Tensor, apply_softmax=False, 
                 remove_com=False, prev_dst_dict=None):
        """预测 g_1 (轨迹终点) 给定 g_t (2D版本)"""
        device = g.device

        with g.local_scope():
            # 收集节点和边特征
            node_scalar_features = [g.ndata[f'{feat}_t'] for feat in self.node_feats]
            node_scalar_features.append(t[node_batch_idx].unsqueeze(-1))

            node_scalar_features = torch.cat(node_scalar_features, dim=-1).to(device)
            node_scalar_features = self.scalar_embedding(node_scalar_features)
            if self.random_node_embedding is not None:
                if 'z_t' in g.ndata:
                    random_node_features = g.ndata['z_t'].to(device)
                else:
                    random_node_features = torch.zeros(
                        g.num_nodes(),
                        self.n_random_node_feats,
                        device=device,
                        dtype=node_scalar_features.dtype,
                    )
                node_scalar_features = node_scalar_features + self.random_node_gate * self.random_node_embedding(random_node_features)

            edge_features = torch.cat([g.edata[f'{feat}_t'] for feat in self.edge_feats], dim=-1)
            edge_features = self.edge_embedding(edge_features)

        # 自条件化处理 (如果启用)
        if self.self_conditioning and prev_dst_dict is None:
            train_self_condition = self.training and (torch.rand(1) > 0.5).item()
            if train_self_condition:
                with torch.no_grad():
                    prev_dst_dict = self.denoise_graph(
                        g, 
                        node_scalar_features.clone(), 
                        edge_features.clone(),
                        node_batch_idx, upper_edge_mask, apply_softmax=True)

        # 通过去噪图
        dst_dict = self.denoise_graph(g, node_scalar_features, edge_features, 
                                     node_batch_idx, upper_edge_mask, apply_softmax)
        return dst_dict

    def denoise_graph(self, g: dgl.DGLGraph,
                      node_scalar_features: torch.Tensor,
                      edge_features: torch.Tensor, 
                      node_batch_idx: torch.Tensor, 
                      upper_edge_mask: torch.Tensor,
                      apply_softmax: bool = False):

        # 多轮消息传递
        for recycle_idx in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                node_scalar_features = conv(g, node_scalar_features, edge_features)

                # 周期性更新边特征
                if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                    if self.separate_mol_updaters:
                        updater_idx = conv_idx // self.convs_per_update
                    else:
                        updater_idx = 0

                    edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features)

        # 预测边logits
        ue_feats = edge_features[upper_edge_mask]
        le_feats = edge_features[~upper_edge_mask]
        edge_pair_features = ue_feats + le_feats

        # 构建预测特征字典
        dst_dict = {}
        for feat in self.node_feats:
            dst_dict[feat] = self.node_output_heads[feat](node_scalar_features)
        for feat in self.edge_feats:
            dst_dict[feat] = self.edge_output_heads[feat](edge_pair_features)

        # 如果需要,对分类特征应用softmax
        if apply_softmax:
            for feat in dst_dict.keys():
                dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1)

        return dst_dict

    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor, n_timesteps: int, 
        visualize=False, 
        dfm_type='campbell',
        stochasticity=8.0, 
        high_confidence_threshold=0.9,
        cat_temp_func=None,
        forward_weight_func=None,
        tspan=None,
        **kwargs):
        """沿向量场积分分子轨迹 (2D版本)"""
        
        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        edge_batch_idx = get_edge_batch_idxs(g)

        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # 设置 g_t = g_0
        for feat in self.canonical_feat_order:
            if feat in self.edge_feats:
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']

        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat in self.edge_feats:
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_1_pred'] = []
        
        dst_dict = None
    
        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False

            step_result = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i, 
                alpha_t_prime_i, 
                node_batch_idx, 
                edge_batch_idx, 
                upper_edge_mask, 
                cat_temp_func=cat_temp_func,
                forward_weight_func=forward_weight_func,
                dfm_type=dfm_type,
                stochasticity=stochasticity, 
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step, 
                prev_dst_dict=dst_dict,
                **kwargs)
            
            g = step_result

            if visualize:
                for feat in self.canonical_feat_order:
                    if feat in self.edge_feats:
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat in self.edge_feats:
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # 设置 g_1 = g_t
        for feat in self.canonical_feat_order:
            if feat in self.edge_feats:
                g_data_src = g.edata
            else:
                g_data_src = g.ndata
            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    if len(feat_traj) > 0:
                        ref_shape = feat_traj[0].shape
                        needs_padding = any(t.shape != ref_shape for t in feat_traj)
                        if needs_padding:
                            from torch.nn.utils.rnn import pad_sequence
                            padded_traj = pad_sequence(feat_traj, batch_first=True, padding_value=0)
                            molecule_dict[feat] = padded_traj
                        else:
                            molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)
            return g, reshaped_traj_frames, upper_edge_mask
        
        return g, upper_edge_mask

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             cat_temp_func: Callable,
             forward_weight_func: Callable, 
             prev_dst_dict: dict = None,
             dfm_type: str = 'campbell',
             stochasticity: float = 8.0,
             high_confidence_threshold: float = 0.9, 
             last_step: bool = False,
             inv_temp_func: Callable = None):

        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        if dfm_type is None:
            dfm_type = self.dfm_type

        if inv_temp_func is None:
            inv_temp_func = lambda t: 1.0
        
        dst_dict = None

        # 预测轨迹终点
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            edge_batch_idx=edge_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=False,
            prev_dst_dict=dst_dict
        )

        dt = s_i - t_i

        # 对节点分类特征进行积分步骤
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat in self.edge_feats:
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1)

            if feat in self.edge_feats:
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1)

            if dfm_type == 'campbell':
                xt, x_1_sampled = \
                campbell_step(p_1_given_t=p_s_1, 
                                xt=xt, 
                                stochasticity=eta, 
                                hc_thresh=hc_thresh, 
                                alpha_t=alpha_t_i[feat_idx], 
                                alpha_t_prime=alpha_t_prime_i[feat_idx],
                                dt=dt, 
                                batch_size=g.batch_size, 
                                batch_num_nodes=g.batch_num_edges()//2 if feat in self.edge_feats else g.batch_num_nodes(), 
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat in self.edge_feats else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)

                xt = gat_step(
                    p_1_given_t=p_s_1, 
                    xt=xt, 
                    alpha_t=alpha_t_i[feat_idx], 
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    forward_weight=forward_weight_func(t_i),
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat in self.edge_feats else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat in self.edge_feats else node_batch_idx,
                )
                                
            # 如果是边特征,需要修改xt和x_1_sampled以包含上下边
            if feat in self.edge_feats:
                e_t = torch.zeros_like(g.edata[f'{feat}_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata[f'{feat}_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled
            
            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled

        return g


class EdgeUpdate2D(nn.Module):
    """2D版本的边更新模块"""
    
    def __init__(self, n_node_scalars, n_edge_feats):
        super().__init__()
        
        input_dim = n_node_scalars*2 + n_edge_feats
        
        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )
        
        self.edge_norm = nn.LayerNorm(n_edge_feats)
    
    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats):
        src_idxs, dst_idxs = g.edges()
        
        mlp_inputs = torch.cat([
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ], dim=-1)
        
        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(mlp_inputs))
        return edge_feats
