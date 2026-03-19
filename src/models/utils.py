import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical
from src.utils.ctmc_utils import purity_sampling
from scipy.optimize import linear_sum_assignment

##############################################################################################################
# module_utils
##############################################################################################################

class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()
        from src.models.gvp import GVP

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)
    
class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_edge_feats, update_edge_w_distance=False, rbf_dim=16):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars*2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):
        

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats


##############################################################################################################
# scheduler_utils
##############################################################################################################
def build_continuous_inv_temp_func(schedule, max_inv_temp=None):

    if schedule is None:
        inv_temp_func = lambda t: 1.0
    elif schedule == 'linear':
        inv_temp_func = lambda t: max_inv_temp*(1 - t)
    elif callable(schedule):
        inv_temp_func = schedule
    else:
        raise ValueError(f'Invalid continuous_inv_temp_schedule: {schedule}')
    return inv_temp_func

def build_cat_temp_schedule(cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a):

    if cat_temperature_schedule == 'decay':
        cat_temp_func = lambda t: cat_temp_decay_max*torch.pow(1-t, cat_temp_decay_a)
    elif isinstance(cat_temperature_schedule, (float, int)):
        cat_temp_func = lambda t: cat_temperature_schedule
    elif callable(cat_temperature_schedule):
        cat_temp_func = cat_temperature_schedule
    else:
        raise ValueError(f"Invalid cat_temperature_schedule: {cat_temperature_schedule}")
    
    return cat_temp_func

def build_fw_schedule(forward_weight_schedule, fw_beta_a, fw_beta_b, fw_beta_max):

    if forward_weight_schedule == 'beta':
        forward_weight_func = lambda t: 1 + fw_beta_max*torch.pow(t, fw_beta_a)*torch.pow(1-t, fw_beta_b)
    elif isinstance(forward_weight_schedule, (float, int)):
        forward_weight_func = lambda t: forward_weight_schedule
    elif callable(forward_weight_schedule):
        forward_weight_func = forward_weight_schedule
    else:
        raise ValueError(f"Invalid forward_weight_schedule: {forward_weight_schedule}")
    
    return forward_weight_func


##############################################################################################################
# algorithm_utils
##############################################################################################################

def compute_ot_permutation(cost_matrix_gpu):

    # 将成本矩阵移至CPU并转换为numpy数组
    cost_matrix_cpu = cost_matrix_gpu.cpu().numpy()
    
    # 使用scipy的linear_sum_assignment求解最优分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)
    
    # 我们需要一个排列p，使得 x_0[p] 最接近 x_1。
    # linear_sum_assignment的结果是 row_ind[i] -> col_ind[i]。
    # 由于row_ind是 [0, 1, 2, ...]，col_ind[i] 就是与第 i 个 x_0 节点匹配的 x_1 节点的索引。
    # 我们希望新的 x_0 (x_0_permuted) 的第 j 个节点，能匹配 x_1 的第 j 个节点。
    # 假设 x_1 的第 j 个节点由 x_0 的第 i 个节点匹配，即 col_ind[i] = j。
    # 那么 x_0_permuted[j] 应该等于 x_0[i]。这需要逆排列。
    permutation = torch.empty_like(torch.from_numpy(col_ind))
    permutation[col_ind] = torch.arange(len(col_ind))
    
    return permutation.to(cost_matrix_gpu.device)

def precompute_distances(g: dgl.DGLGraph, node_positions=None, rbf_dmax = 14, rbf_dim = 32):
    """Precompute the pairwise distances between all nodes in the graph."""
    from src.models.gvp import _rbf, _norm_no_nan

    with g.local_scope():

        if node_positions is None:
            g.ndata['x_d'] = g.ndata['x_t']
        else:
            g.ndata['x_d'] = node_positions

        g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))

        dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
        x_diff = g.edata['x_diff'] / dij
        d = _rbf(dij.squeeze(1), D_max=rbf_dmax, D_count=rbf_dim)
    
    return x_diff, d

def precompute_distances_hetero(g: dgl.DGLGraph, node_positions = None, rbf_dmax = 14, rbf_dim = 32):
    """Precompute the pairwise distances between nodes in a heterogeneous graph.
    
    Args:
        g: 异构图
        node_positions: 可选的节点位置字典 {'ligand': tensor, 'pocket': tensor}
    
    Returns:
        x_diff_dict: 每种边类型的相对位移向量字典
        d_dict: 每种边类型的RBF距离特征字典
    """
    from src.models.gvp import _rbf, _norm_no_nan

    with g.local_scope():
        # 设置节点位置
        if node_positions is None:
            for ntype in g.ntypes:
                if 'x_t' in g.nodes[ntype].data:
                    g.nodes[ntype].data['x_d'] = g.nodes[ntype].data['x_t']
                elif 'pos' in g.nodes[ntype].data:
                    g.nodes[ntype].data['x_d'] = g.nodes[ntype].data['pos']
        else:
            for ntype, pos in node_positions.items():
                g.nodes[ntype].data['x_d'] = pos

        x_diff_dict = {}
        d_dict = {}

        # 为每种边类型计算距离
        for etype in g.canonical_etypes:
            if g.num_edges(etype) == 0:
                continue
                
            # 计算相对位移
            g.apply_edges(fn.u_sub_v('x_d', 'x_d', 'x_diff'), etype=etype)
            
            # 计算距离并归一化
            dij = _norm_no_nan(g.edges[etype].data['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edges[etype].data['x_diff'] / dij
            
            # 计算RBF距离特征
            d = _rbf(dij.squeeze(1), D_max=rbf_dmax, D_count=rbf_dim)
            
            x_diff_dict[etype] = x_diff
            d_dict[etype] = d

    return x_diff_dict, d_dict

##############################################################################################################
# intergration_utils
##############################################################################################################

def campbell_step(p_1_given_t: torch.Tensor,
                    xt: torch.Tensor, 
                    stochasticity: float, 
                    hc_thresh: float, 
                    alpha_t: float, 
                    alpha_t_prime: float,
                    dt,
                    batch_size: int,
                    batch_num_nodes: torch.Tensor,
                    n_classes: int,
                    mask_index:int,
                    last_step: bool, 
                    batch_idx: torch.Tensor,
                ):
    x1 = Categorical(p_1_given_t).sample() # has shape (num_nodes,)

    unmask_prob = dt*( alpha_t_prime + stochasticity*alpha_t  ) / (1 - alpha_t)
    mask_prob = dt*stochasticity

    unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
    mask_prob = torch.clamp(mask_prob, min=0, max=1)

    # sample which nodes will be unmasked
    if hc_thresh > 0:
        # select more high-confidence predictions for unmasking than low-confidence predictions
        will_unmask = purity_sampling(
            xt=xt, x1=x1, x1_probs=p_1_given_t, unmask_prob=unmask_prob,
            mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
            node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=xt.device)
    else:
        # uniformly sample nodes to unmask
        will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
        will_unmask = will_unmask * (xt == mask_index) # only unmask nodes that are currently masked

    if not last_step:
        # compute which nodes will be masked
        will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
        will_mask = will_mask * (xt != mask_index) # only mask nodes that are currently unmasked

        # mask the nodes
        xt[will_mask] = mask_index

    # unmask the nodes
    xt[will_unmask] = x1[will_unmask]

    xt = one_hot(xt, num_classes=n_classes).float()
    x1 = one_hot(x1, num_classes=n_classes).float()
    return xt, x1

# def campbell_step(p_1_given_t: torch.Tensor,
#                     xt: torch.Tensor, 
#                     stochasticity: float, 
#                     hc_thresh: float, 
#                     alpha_t: float, 
#                     alpha_t_prime: float,
#                     dt,
#                     batch_size: int,
#                     batch_num_nodes: torch.Tensor,
#                     n_classes: int,
#                     mask_index:int,
#                     last_step: bool, 
#                     batch_idx: torch.Tensor,
#                 ):
#     """
#     简化版本：只有去掩码 + CTMC转移，没有重掩码
#     """
#     device = xt.device
#     xt_new = xt.clone()

#     # 1. 采样最终状态 x1
#     x1 = Categorical(p_1_given_t).sample()

#     # 只保留去掩码概率，去除掩码概率
#     unmask_prob = dt * alpha_t_prime / (1 - alpha_t)
#     unmask_prob = torch.clamp(unmask_prob, min=0, max=1)

#     # 2. 处理去掩码 (Masked -> Unmasked)
#     is_masked = (xt == mask_index)
#     if is_masked.any():
#         if hc_thresh > 0:
#             will_unmask = purity_sampling(
#                 xt=xt, x1=x1, x1_probs=p_1_given_t, unmask_prob=unmask_prob,
#                 mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
#                 node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=device)
#         else:
#             will_unmask = torch.rand(xt.shape[0], device=device) < unmask_prob
#             will_unmask = will_unmask & is_masked

#         xt_new[will_unmask] = x1[will_unmask]

#     # 3. CTMC转移 (只在非掩码节点上进行)
#     # 注意：现在只考虑当前未被掩码的节点（包括刚去掩码的节点）
#     if not last_step:
#         is_unmasked = (xt_new != mask_index)  # 使用更新后的状态

#         if is_unmasked.any():
#             unmasked_indices = is_unmasked.nonzero().squeeze(-1)
#             current_states = xt_new[unmasked_indices]  # 当前确定的状态

#             # CTMC转移概率
#             transition_prob = stochasticity * dt
#             transition_prob = torch.clamp(transition_prob, min=0, max=1)

#             nodes_will_transition_mask = torch.rand_like(transition_prob) < transition_prob
            
#             if nodes_will_transition_mask.any():
#                 transition_indices = unmasked_indices[nodes_will_transition_mask]
#                 old_states = xt_new[transition_indices]
#                 probs_for_transition = p_1_given_t[transition_indices]

#                 new_states = Categorical(probs_for_transition).sample()
#                 xt_new[transition_indices] = new_states

#     # 4. 返回结果
#     xt_one_hot = one_hot(xt_new, num_classes=n_classes).float()
#     x1_one_hot = one_hot(x1, num_classes=n_classes).float()
    
#     return xt_one_hot, x1_one_hot

def gat_step(
            p_1_given_t: torch.Tensor,
            xt: torch.Tensor, 
            alpha_t: float, 
            alpha_t_prime: float,
            forward_weight: float,
            dt,
            batch_size: int,
            batch_num_nodes: torch.Tensor,
            n_classes: int,
            mask_index:int,
            batch_idx: torch.Tensor,
        ):


    # add a zero-column on to p_1_given_t to represent the mask token
    p_1_given_t = torch.cat([p_1_given_t, torch.zeros_like(p_1_given_t[:, :1])], dim=-1)

    # create a one-hot encoding of xt
    delta_xt = one_hot(xt, num_classes=n_classes).float()

    # compute forward probability velocity
    u_forward = alpha_t_prime / (1 - alpha_t) * (p_1_given_t - delta_xt)

    # create a delta on the mask token
    delta_mask = torch.zeros_like(delta_xt)
    delta_mask[:, mask_index] = 1

    # compute the backward probability velocity
    u_backward = alpha_t_prime / (alpha_t + 1e-8) * (delta_xt - delta_mask)

    # compute the probability velocity
    backward_weight = forward_weight - 1
    pvel = forward_weight*u_forward - backward_weight*u_backward

    # compute the parameters of the transition distritibution
    p_step = delta_xt + dt*pvel

    # clamp p_step to be valid
    p_step = torch.clamp(p_step, min=1.0e-9, max=1)

    # sample x_{t+dt} from the transition distribution
    x_dt = Categorical(p_step).sample()

    # one-hot encode x_{t+dt}
    x_dt = one_hot(x_dt, num_classes=n_classes).float()

    return x_dt


##############################################################################################################
# graph_utils
##############################################################################################################

def reconstruct_graph_dynamic(
        g: dgl.DGLGraph,
        upper_edge_mask: torch.Tensor,
        node_batch_idx: torch.Tensor,
        k: int = 12,
        # cutoff: float = 6.0,
        null_edge_value: float = 1.0,
        device: torch.device = None
    ):
    if device is None:
        device = g.device

    original_batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
    num_total_nodes = g.num_nodes()
    assert num_total_nodes == len(node_batch_idx), "node_batch_idx长度与节点数不匹配"
    assert g.num_edges() == len(upper_edge_mask), "upper_edge_mask长度与边数不匹配"

    # 1. 构建边ID的索引
    src, dst = g.edges()
    edges_initial = src * num_total_nodes + dst

    # 获取节点坐标
    node_positions = g.ndata['x_t'][:, :3]

    # 2. 识别需要保留的边(real边)
    edges_to_keep = torch.tensor([], dtype=torch.long, device=device)


    # 3. 计算每个子图需要多少边
    batch_size = original_batch_size
    num_nodes_per_graph = torch.bincount(node_batch_idx, minlength=batch_size)

    
    # 4. 为每个子图采样需要添加的边ID
    graph_offsets = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(0)[:-1]])
    graph_ranges = torch.stack([graph_offsets, graph_offsets + num_nodes_per_graph], dim=1)
    
    edges_to_add = torch.tensor([], dtype=torch.long, device=device)
    for i in range(batch_size):
        
        start, end = graph_ranges[i]
        num_nodes = end - start
        if num_nodes < 2:
            continue

        # 获取当前子图的节点坐标
        subgraph_nodes = torch.arange(start, end, device=device)
        subgraph_pos = node_positions[subgraph_nodes]
        
        # 计算KNN
        # 使用距离矩阵实现KNN
        diff = subgraph_pos.unsqueeze(1) - subgraph_pos.unsqueeze(0)  # [N, N, 3]
        dist_matrix = torch.norm(diff, dim=2)  # [N, N]
        
        # 获取每个节点的top k+1最近邻(包括自己)
        k = min(k, num_nodes-1)
        topk_values, topk_indices = torch.topk(dist_matrix, k=k+1, largest=False, dim=1)
        
        # 生成边对
        u = torch.repeat_interleave(torch.arange(num_nodes, device=device), k)
        v = topk_indices[:, 1:k+1].flatten()  # 跳过自己(索引0)

        distances = dist_matrix[u, v]
    
        # # 创建掩码：只保留距离小于截断值的边
        # distance_mask = distances < cutoff
        
        # # 应用距离截断
        # u = u[distance_mask]
        # v = v[distance_mask]
        
        # 转换为全局节点索引
        u_global = subgraph_nodes[u]
        v_global = subgraph_nodes[v]
        
        # 创建双向边
        edge_ids = u_global * num_total_nodes + v_global
        edge_ids_reverse = v_global * num_total_nodes + u_global
        
        # 添加到边列表
        edges_to_add = torch.cat([edges_to_add, edge_ids, edge_ids_reverse])
    
    # 去重
    edges_to_add = torch.unique(edges_to_add)

    edges_to_keep = torch.cat([edges_to_keep, edges_to_add])
    edges_to_add = edges_to_add[~torch.isin(edges_to_add, edges_initial)]
    edges_to_remove = edges_initial[~torch.isin(edges_initial, edges_to_keep)]



    # 7. 执行边的删除和添加操作
    # 先删除需要删除的边
    if len(edges_to_remove) > 0:
        src, dst = g.edges()
        current_edge_ids = src * num_total_nodes + dst
        remove_mask = torch.isin(current_edge_ids, edges_to_remove)
        eids_to_remove = remove_mask.nonzero().flatten()
        g.remove_edges(eids_to_remove)


    # 添加新边
    if len(edges_to_add) > 0:
        src_ids = edges_to_add // num_total_nodes
        dst_ids = edges_to_add % num_total_nodes
        
        # 保存旧特征（避免多次访问edata）
        old_feats = {name: g.edata[name] for name in g.edata.keys()}

        
        # 批量添加边（单次操作比多次添加高效）
        g.add_edges(src_ids, dst_ids)
        
        # 设置所有特征
        for feat_name, old_feat in old_feats.items():
            feat_dim = old_feat.shape[-1] if len(old_feat.shape) > 1 else 1
            dtype = old_feat.dtype
            
            # 预分配内存（直接创建最终大小的张量）
            new_feat = torch.zeros((g.num_edges(), *old_feat.shape[1:]), 
                                dtype=dtype, device=device)
            
            # 保留旧特征（向量化复制）
            num_old_edges = old_feat.shape[0]
            new_feat[:num_old_edges] = old_feat
            
            # 批量设置新边特征（避免循环）
            if feat_name == 'e_t':
                # 特殊处理e_t特征
                new_feat[num_old_edges:, 0] = null_edge_value
            else:
                # 其他特征保持默认零值
                pass
                
            g.edata[feat_name] = new_feat
    
    # 8. 更新upper_edge_mask和edge_batch_idx
    src, dst = g.edges()
    edge_batch_idx_new = node_batch_idx[src]
    
    # 创建新的upper_edge_mask
    upper_edge_mask_new = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
    if g.num_edges() > 0:
        current_edge_ids = src * num_total_nodes + dst
        upper_mask = src < dst
        real_or_added = torch.cat([edges_to_keep, edges_to_add]) if len(edges_to_add) > 0 else edges_to_keep
        real_or_added_mask = torch.isin(current_edge_ids, real_or_added)
        upper_edge_mask_new = upper_mask & real_or_added_mask
    
    # 9. 按batch_idx排序
    sorted_indices = torch.argsort(edge_batch_idx_new)
    g = dgl.reorder_graph(
        g,
        node_permute_algo=None,
        edge_permute_algo='custom',
        permute_config={'edges_perm': sorted_indices},
        store_ids=False
    )
    
    # 更新相关变量
    upper_edge_mask_new = upper_edge_mask_new[sorted_indices]
    edge_batch_idx_new = edge_batch_idx_new[sorted_indices]

    # 更新图的批处理信息
    g.set_batch_num_nodes(num_nodes_per_graph)
    g.set_batch_num_edges(torch.bincount(edge_batch_idx_new, minlength=batch_size))


    return g, upper_edge_mask_new, edge_batch_idx_new
