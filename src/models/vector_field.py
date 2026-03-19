import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence

from src.models.gvp import GVPConv, HeteroGVPConv, GVPAttentionConv
from src.data_processing.utils import get_batch_idxs
from src.models.interpolant_scheduler import InterpolantScheduler
from src.models.utils import *
from src.models.self_conditioning import SelfConditioningResidualLayer

from src.data_processing.utils import get_edge_batch_idxs

import dgl
from typing import Union, Callable
import scipy


class CTMCVectorField(nn.Module):

    # uses Continuous-Time Markov Chain (CTMC) to model the flow of cateogrical features (atom type, charge, bond order)
    # CTMC for flow-matching was originally proposed in https://arxiv.org/abs/2402.04997

    # we make some modifications to the original CTMC model:
    # our conditional trajectories interpolate along a progress coordiante alpha_t, which is a function of time t
    # where we set a different alpha_t for each data modality
    # we also do purity sampling in a slightly different way that in theory would be slightly less performant but is
    # computationally much more efficient when working with batched graphs

    def __init__(self, n_atom_types: int,
                    canonical_feat_order: list,
                    interpolant_scheduler: InterpolantScheduler,
                    n_charges: int,
                    n_bond_types: int, 
                    n_vec_channels: int = 16,
                    n_cp_feats: int = 0, 
                    n_hidden_scalars: int = 64,
                    n_hidden_edge_feats: int = 64,
                    n_recycles: int = 1,
                    n_molecule_updates: int = 2, 
                    convs_per_update: int = 2,
                    n_message_gvps: int = 3, 
                    n_update_gvps: int = 3,
                    separate_mol_updaters: bool = False,
                    message_norm: Union[float, str] = 100,
                    update_edge_w_distance: bool = False,
                    rbf_dmax = 20,
                    rbf_dim = 16,
                    exclude_charges: bool = False,
                    continuous_inv_temp_schedule = None,
                    continuous_inv_temp_max: float = 10.0,
                    has_mask: bool = True,
                    enable_dynamic_graph: bool = True,
                    knn_connectivity: int = 12,
                    sde: bool = False,
                    self_conditioning: bool = False,
                    stochasticity: float = 0.0, 
                    high_confidence_threshold: float = 0.0, 
                    dfm_type: str = 'campbell', 
                    cat_temperature_schedule: Union[str, Callable, float] = 0.05,
                    cat_temp_decay_max: float = 0.8,
                    cat_temp_decay_a: float = 2,
                    forward_weight_schedule: Union[str, Callable, float] = 'beta',
                    fw_beta_a: float = 0.25, fw_beta_b: float = 0.25, fw_beta_max: float = 10.0,
                    **kwargs):
        
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.exclude_charges = exclude_charges
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order
        self.enable_dynamic_graph = enable_dynamic_graph
        self.n_cp_feats = n_cp_feats
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.knn_connectivity = knn_connectivity
        self.sde = sde
        self.self_conditioning = self_conditioning

        if self.exclude_charges:
            self.n_charges = 0
            n_charges = 0

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, 'n_vec_channels must be >= 3'

        self.continuous_inv_temp_schedule = continuous_inv_temp_schedule
        self.continouts_inv_temp_max = continuous_inv_temp_max
        self.continuous_inv_temp_func = build_continuous_inv_temp_func(self.continuous_inv_temp_schedule, self.continouts_inv_temp_max) 

        self.n_cat_feats = { # number of possible values for each categorical variable (not including mask tokens in the case of CTMC)
            'a': n_atom_types,
            'c': n_charges,
            'e': n_bond_types
        }

        n_mask_feats = int(has_mask)
        t_dim = 1
        if self.exclude_charges:
            mask_dim_node = 1
        else:
            mask_dim_node = 2


        self.scalar_embedding = nn.Sequential(
            nn.Linear(n_atom_types + n_charges + t_dim + mask_dim_node*n_mask_feats, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types + n_mask_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats)
        )

        self.conv_layers = []
        for conv_idx in range(convs_per_update*n_molecule_updates):
            self.conv_layers.append(GVPConv(
                scalar_size=n_hidden_scalars,
                vector_size=n_vec_channels,
                n_cp_feats=n_cp_feats,
                edge_feat_size=n_hidden_edge_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim
            )
            )

        # for conv_idx in range(convs_per_update*n_molecule_updates):
        #     self.conv_layers.append(GVPAttentionConv(
        #         scalar_size=n_hidden_scalars,
        #         vector_size=n_vec_channels,
        #         n_cp_feats=n_cp_feats,
        #         edge_feat_size=n_hidden_edge_feats,
        #         n_message_gvps=n_message_gvps,
        #         n_update_gvps=n_update_gvps,
        #         # message_norm=message_norm,
        #         rbf_dmax=rbf_dmax,
        #         rbf_dim=rbf_dim
        #     )
        #     )

        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(NodePositionUpdate(n_hidden_scalars, n_vec_channels, n_gvps=3, n_cp_feats=n_cp_feats))
            self.edge_updaters.append(EdgeUpdate(n_hidden_scalars, n_hidden_edge_feats, update_edge_w_distance=update_edge_w_distance, rbf_dim=rbf_dim))


        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges)
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types)
        )

        self.eta = stochasticity # default stochasticity parameter, 0 means no stochasticity
        self.hc_thresh = high_confidence_threshold # the threshold for for calling a prediction high-confidence, 0 means no purity sampling
        self.dfm_type = dfm_type

        # configure temperature schedule for categorical features
        self.cat_temperature_schedule = cat_temperature_schedule
        self.cat_temp_decay_max = cat_temp_decay_max
        self.cat_temp_decay_a = cat_temp_decay_a
        self.cat_temp_func = build_cat_temp_schedule(
            cat_temperature_schedule=cat_temperature_schedule,
            cat_temp_decay_max=cat_temp_decay_max,
            cat_temp_decay_a=cat_temp_decay_a)
        
        # configure forward weight schedule
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

        self.mask_idxs = { # for each categorical feature, the index of the mask token
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types,
        }

        if self.self_conditioning: 
            self.self_conditioning_residual_layer = SelfConditioningResidualLayer(
                    n_atom_types=n_atom_types,
                    n_charges=n_charges,
                    n_bond_types=n_bond_types,
                    node_embedding_dim=n_hidden_scalars,
                    edge_embedding_dim=n_hidden_edge_feats,
                    rbf_dim=rbf_dim,
                    rbf_dmax=rbf_dmax
                )
    
    
    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        """
        通过最优传输对齐节点后，对条件路径 p(g_t|g_0,g_1) 进行SDE采样。
        噪声仅应用于连续坐标'x'。
        """

        _, alpha_t = self.interpolant_scheduler.interpolant_weights(t)

        device = g.device

        graphs = dgl.unbatch(g)
        permuted_x0_list = []
        for g_i in graphs:
            x_0_i = g_i.ndata['x_0']
            x_1_i = g_i.ndata['x_1_true']
            cost_matrix = torch.cdist(x_0_i, x_1_i)**2
            permutation = compute_ot_permutation(cost_matrix)
            permuted_x0_list.append(x_0_i[permutation])
        permuted_x0_batch = torch.cat(permuted_x0_list, dim=0)

        x_idx = self.canonical_feat_order.index('x')

        dst_weight = alpha_t[:, x_idx][node_batch_idx].unsqueeze(-1)
        src_weight = 1 - dst_weight
        mu_t = src_weight * permuted_x0_batch + dst_weight * g.ndata['x_1_true']

        if self.sde:
            sigma_t = self.interpolant_scheduler.sigma_t(t, self.eta)
            sigma_t_expanded = sigma_t[:, x_idx][node_batch_idx].unsqueeze(-1)
            epsilon = torch.randn_like(mu_t)
            g.ndata['x_t'] = mu_t + sigma_t_expanded * epsilon
        else:
            g.ndata['x_t'] = mu_t
        
        t_node = t[node_batch_idx]

        for feat, feat_idx in zip(['a', 'c'], [1, 2]):
            if self.mask_idxs[feat] == 0:
                continue
            xt = g.ndata[f'{feat}_1_true'].argmax(-1)
            alpha_t_feat = alpha_t[:, feat_idx][node_batch_idx]
            xt[torch.rand(g.num_nodes(), device=device) < 1 - alpha_t_feat] = self.mask_idxs[feat]
            g.ndata[f'{feat}_t'] = one_hot(xt, num_classes=self.n_cat_feats[feat] + 1)

        num_edges = int(g.num_edges() // 2)
        e_idx = self.canonical_feat_order.index('e')
        alpha_t_e = alpha_t[:, e_idx][edge_batch_idx][upper_edge_mask]
        et_upper = g.edata['e_1_true'][upper_edge_mask].argmax(-1)
        et_upper[torch.rand(num_edges, device=device) < 1 - alpha_t_e] = self.mask_idxs['e']
        
        n, d = g.edata['e_1_true'].shape
        e_t = torch.zeros((n, d + 1), dtype=g.edata['e_1_true'].dtype, device=g.device)
        et_upper_onehot = one_hot(et_upper, num_classes=self.n_cat_feats['e'] + 1).float()
        e_t[upper_edge_mask] = et_upper_onehot
        e_t[~upper_edge_mask] = et_upper_onehot
        g.edata['e_t'] = e_t

        return g

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False, prev_dst_dict=None):
        """Predict x_1 (trajectory destination) given x_t"""
        device = g.device


        with g.local_scope():
            # gather node and edge features for input to convolutions
            node_scalar_features = [
                g.ndata['a_t'],
                t[node_batch_idx].unsqueeze(-1)
            ]

            # if we are not excluding charges, include them in the node scalar features
            if not self.exclude_charges:
                node_scalar_features.append(g.ndata['c_t'])

            node_scalar_features = torch.cat(node_scalar_features, dim=-1).to(device)

            node_scalar_features = self.scalar_embedding(node_scalar_features)

            node_positions = g.ndata['x_t']

            num_nodes = g.num_nodes()

            # initialize the vector features for every node to be zeros
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            # i thought setting the first three channels to the identity matrix would be a good idea,
            # but this actually breaks rotational equivariance
            # node_vec_features[:, :3, :] = torch.eye(3, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)

            # edge_features = torch.cat([g.edata['e_t'], t[edge_batch_idx].unsqueeze(-1)], dim=-1).to(device)
            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            # New
            # p_a = torch.tensor([7.6395e-01, 3.9876e-02, 1.8962e-01, 0.0000e+00, 3.8502e-04, 3.3426e-03,1.8099e-03, 9.2530e-04, 7.2942e-05, 1.0245e-05, 0.0],device = device)
            # p_c = torch.tensor([0.0000e+00, 7.2973e-04, 9.9806e-01, 1.2040e-03, 1.2269e-07, 1.7177e-06, 0.0],device = device)
            # p_e = torch.tensor([9.4538e-01, 3.8098e-02, 4.2158e-03, 6.0659e-05, 1.2240e-02, 0.0],device = device)

            # marginal_node = [p_a.repeat(node_scalar_features.shape[0], 1), t[node_batch_idx].unsqueeze(-1)]
            # if not self.exclude_charges:
            #     marginal_node.append(p_c.repeat(node_scalar_features.shape[0], 1))
            # marginal_node = torch.cat(marginal_node, dim=-1).to(device)
            # marginal_node_scalar_features = self.scalar_embedding(marginal_node)
            # node_scalar_features = (t[node_batch_idx].unsqueeze(-1)/2+0.5)*node_scalar_features + (0.5-t[node_batch_idx].unsqueeze(-1)/2)*marginal_node_scalar_features

            # marginal_edge = torch.cat([p_e.repeat(edge_features.shape[0], 1), t[edge_batch_idx].unsqueeze(-1)], dim=-1).to(device)
            # marginal_edge_features = self.edge_embedding(marginal_edge)
            # edge_features = (t[edge_batch_idx].unsqueeze(-1)/2+0.5)*edge_features + (0.5-t[edge_batch_idx].unsqueeze(-1)/2)*marginal_edge_features

        # if we are using self-conditoning, and prev_dist_dict is None, then
        # we must be in the process of training a self-conditioning model, and need to enter the following logic branch:
        # with p = 0.5, we do a gradient-stopped pass through denoise_graph to get predicted endpoint, 
        # then set prev_dst_dict to this predicted endpoint
        # for the other 0.5 of the time, we do nothing!
        # also if we are in the first timestep of inference, we need to do generate the first predicted endpoint
        if self.self_conditioning and prev_dst_dict is None:

            train_self_condition = self.training and (torch.rand(1) > 0.5).item()
            inference_first_step = not self.training and (t == 0).all().item()

            if train_self_condition or inference_first_step:
                with torch.no_grad():
                    prev_dst_dict = self.denoise_graph(
                        g, 
                        node_scalar_features.clone(), 
                        node_vec_features.clone(), 
                        node_positions.clone(), 
                        edge_features.clone(),
                        node_batch_idx, upper_edge_mask, apply_softmax=True, remove_com=False)

        if self.self_conditioning and prev_dst_dict is not None:
            # if prev_dst_dict is not none, we need to pass through the self-conditioning residual block
            node_scalar_features, node_positions, node_vec_features, edge_features = self.self_conditioning_residual_layer(
                g, node_scalar_features, node_positions, node_vec_features, edge_features, 
                prev_dst_dict, node_batch_idx, upper_edge_mask
            )

        # now, pass through denoising graph
        dst_dict = self.denoise_graph(g, node_scalar_features, node_vec_features, node_positions, edge_features, node_batch_idx, upper_edge_mask, apply_softmax, remove_com)
        return dst_dict


    def denoise_graph(self, g: dgl.DGLGraph,
                      node_scalar_features: torch.Tensor,
                      node_vec_features: torch.Tensor,
                      node_positions: torch.Tensor,
                      edge_features: torch.Tensor, 
                      node_batch_idx: torch.Tensor, 
                      upper_edge_mask: torch.Tensor,
                      apply_softmax: bool = False,
                      remove_com: bool = False):

        x_diff, d = precompute_distances(g)
        for recycle_idx in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):

                # perform a single convolution which updates node scalar and vector features (but not positions)
                node_scalar_features, node_vec_features = conv(g, 
                        scalar_feats=node_scalar_features, 
                        coord_feats=node_positions,
                        vec_feats=node_vec_features,
                        edge_feats=edge_features,
                        x_diff=x_diff,
                        d=d
                )

                # every convs_per_update convolutions, update the node positions and edge features
                if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:

                    if self.separate_mol_updaters:
                        updater_idx = conv_idx // self.convs_per_update
                    else:
                        updater_idx = 0

                    node_positions = self.node_position_updaters[updater_idx](node_scalar_features, node_positions, node_vec_features)

                    x_diff, d = precompute_distances(g, node_positions)

                    edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features, d=d)

            
        # predict final charges and atom type logits
        node_scalar_features = self.node_output_head(node_scalar_features)
        atom_type_logits = node_scalar_features[:, :self.n_atom_types]
        if not self.exclude_charges:
            atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

        # predict the final edge logits
        ue_feats = edge_features[upper_edge_mask]
        le_feats = edge_features[~upper_edge_mask]
        edge_logits = self.to_edge_logits(ue_feats + le_feats)

        # project node positions back into zero-COM subspace
        if remove_com:
            g.ndata['x_1_pred'] = node_positions
            g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
            node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

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
        """Integrate the trajectories of molecules along the vector field."""
        
        # TODO: this overrides EndpointVectorField.integrate just because it has some extra arguments
        # we should refactor this so that we don't have to copy the entire function

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)

            # log_t = torch.linspace(-4, 0, n_timesteps, device=g.device)  # 从-4到0
            # t = torch.exp(log_t)  # 映射到[0, 1]区间
            # t[0] = 0  # 确保第一个点是0
            # t[-1] = 1  # 确保最后一个点是1
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']
        
        if self.enable_dynamic_graph:
            g, upper_edge_mask, edge_batch_idx = reconstruct_graph_dynamic(g, upper_edge_mask, node_batch_idx, k=self.knn_connectivity)


        edge_batch_idx = get_edge_batch_idxs(g)



        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
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
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False
            


            # compute next step and set x_t = x_s
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
            
            if isinstance(step_result, tuple):
                g, upper_edge_mask, edge_batch_idx = step_result

            else:
                g = step_result
            

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat == 'e':
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

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                        # 如果 feat_traj 中张量的形状不一致，用 pad_sequence 填充
                        if len(feat_traj) > 0:
                            # 检查是否需要填充（比较第一个张量的形状）
                            ref_shape = feat_traj[0].shape
                            needs_padding = any(t.shape != ref_shape for t in feat_traj)
                            
                            if needs_padding:
                                # 用 0 填充，并堆叠成一个张量
                                padded_traj = pad_sequence(feat_traj, batch_first=True, padding_value=0)
                                molecule_dict[feat] = padded_traj
                            else:
                                # 直接堆叠
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

        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            edge_batch_idx=edge_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            prev_dst_dict = dst_dict
        )

        dt = s_i - t_i
        sigma_t_i = self.interpolant_scheduler.sigma_t(t_i.unsqueeze(0), self.eta)[0, 0]
        sigma_t_prime_i = self.interpolant_scheduler.sigma_t_prime(t_i.unsqueeze(0), self.eta)[0, 0]

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        x_0 = g.ndata['x_0']


        if self.sde:
            vf = self.vector_field_sde(x_t, x_0, x_1, alpha_t_i[0], alpha_t_prime_i[0], sigma_t_i, sigma_t_prime_i)
        else:
            vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])

        noise = torch.randn_like(x_t)

        if self.sde:
            g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i) + sigma_t_i * noise * torch.sqrt(dt)
        else:
            g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i)

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1) # has shape (num_nodes,)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

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
                                batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(), 
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                # record predicted endpoint for visualization
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)

                xt = gat_step(
                    p_1_given_t=p_s_1, 
                    xt=xt, 
                    alpha_t=alpha_t_i[feat_idx], 
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    forward_weight=forward_weight_func(t_i),
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                )
                                
            
            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled
            
            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled\
        
        if self.enable_dynamic_graph and t_i<0.95:
            g, upper_edge_mask, edge_batch_idx = reconstruct_graph_dynamic(g, upper_edge_mask, node_batch_idx, k=self.knn_connectivity)
            return g, upper_edge_mask, edge_batch_idx


        return g


    def vector_field_sde(self, x_t, x_0, x_1, alpha_t, alpha_t_prime, sigma_t, sigma_t_prime):
        """
        修改为与SDE训练目标一致的向量场
        """

        flow_part = (alpha_t_prime / (1 - alpha_t)) * (x_1 - x_t)

        # 计算线性插值部分
        linear_part = x_1-x_0
        
        # 计算补正项
        # correction_term = (sigma_t_prime / (sigma_t + 1e-8)) * (x_t - linear_part)
        correction_term = (sigma_t**2 / (2*(1-alpha_t))) * (x_t - alpha_t/alpha_t_prime*flow_part)
        # 完整的漂移场（与训练目标一致）
        vf = flow_part + correction_term
        # vf = flow_part
        
        return vf

    def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime):
        vf = (alpha_t_prime / (1 - alpha_t)) * (x_1 - x_t)
        return vf

    


class ContextualCTMCVectorField(CTMCVectorField):
    """
    结合CTMC和交叉注意力的VectorField，用于在蛋白质口袋环境中生成分子
    """
    def __init__(self, *args, 
                 n_hidden_scalars_pocket: int = 256, 
                 n_hidden_edge_feats_pocket: int = 128, 
                 n_vec_channels_pocket: int = 16,
                 **kwargs):
        
        print("Before super(): kwargs keys =", kwargs.keys())
        super().__init__(*args, **kwargs)
        self.n_hidden_scalars_pocket = n_hidden_scalars_pocket
        self.n_hidden_edge_feats_pocket = n_hidden_edge_feats_pocket
        self.n_vec_channels_pocket = n_vec_channels_pocket


        self.super_edge_embedding = nn.Sequential(
            nn.Linear(2, self.n_hidden_edge_feats_pocket),
            nn.SiLU(),
            nn.Linear(self.n_hidden_edge_feats_pocket, self.n_hidden_edge_feats_pocket),
            nn.SiLU(),
            nn.LayerNorm(self.n_hidden_edge_feats_pocket)
        )


        self.pocket_scalar_embedding = nn.Sequential(
            nn.Linear(self.n_atom_types+1, self.n_hidden_scalars_pocket),
            nn.SiLU(),
            nn.Linear(self.n_hidden_scalars_pocket, self.n_hidden_scalars_pocket),
            nn.SiLU(),
            nn.LayerNorm(self.n_hidden_scalars_pocket)
        )

        self.hetero_conv_layers = []
        for conv_idx in range(self.convs_per_update*self.n_molecule_updates):
            self.hetero_conv_layers.append(HeteroGVPConv(
                node_types=['ligand','pocket'],
                edge_types=[('ligand','ll','ligand'), ('pocket','pp','pocket'),
                            ('ligand','lp','pocket'), ('pocket','pl','ligand')],
                scalar_size=self.n_hidden_scalars,
                vector_size=self.n_vec_channels,
                n_cp_feats=self.n_cp_feats,
                edge_feat_size=self.n_hidden_edge_feats,
                n_message_gvps=self.n_message_gvps,
                n_update_gvps=self.n_update_gvps,
                message_norm=self.message_norm,
                rbf_dmax=self.rbf_dmax,
                rbf_dim=self.rbf_dim,
                scalar_size_pocket = self.n_hidden_scalars_pocket,
                edge_feat_size_pocket = self. n_hidden_edge_feats_pocket,
                vector_size_pocket = self.n_vec_channels_pocket
            )
            )
        self.hetero_conv_layers = nn.ModuleList(self.hetero_conv_layers)

    # 精简ligand图特征
    def prepare_ligand_graph(self, ligand_g, t, required_feats):
        ligand_g = ligand_g.local_var()
        device = ligand_g.device
        node_batch_idx, edge_batch_idx = get_batch_idxs(ligand_g)

        # 配体节点处理
        ligand_feats = [
            ligand_g.ndata['a_t'].float(),
            t[node_batch_idx].unsqueeze(-1)
        ]
        if not self.exclude_charges:
            ligand_feats.append(ligand_g.ndata['c_t'])
        ligand_feats = torch.cat(ligand_feats, dim=-1)
        ligand_embedded = self.scalar_embedding(ligand_feats)
        ligand_g.ndata['scalar'] = ligand_embedded

        edge_embeded = self.edge_embedding(ligand_g.edata['e_t'])
        ligand_g.edata['e_t'] = edge_embeded

        # 确保有x_t特征
        if 'x_t' not in ligand_g.ndata:
            ligand_g.ndata['x_t'] = ligand_g.ndata.get('pos', ligand_g.ndata.get('x', torch.zeros_like(ligand_g.ndata['a_t']))).float()

        # 只保留必要的特征
        for feat in list(ligand_g.ndata.keys()):
            if feat not in required_feats:
                del ligand_g.ndata[feat]

        for feat in list(ligand_g.edata.keys()):
            if feat not in required_feats:
                del ligand_g.edata[feat]
        return ligand_g
    
    # 准备口袋图特征
    def prepare_pocket_graph(self, pocket_g, t, required_feats):
        pocket_g = pocket_g.local_var()
        device = pocket_g.device
        pocket_node_batch_idx, _ = get_batch_idxs(pocket_g)
        pocket_feats = [
            pocket_g.ndata['a_1_true'].float(),
            t[pocket_node_batch_idx].unsqueeze(-1)
        ]
        pocket_feats = torch.cat(pocket_feats, dim=-1)
        pocket_embedded = self.pocket_scalar_embedding(pocket_feats)
        pocket_g.ndata['scalar'] = pocket_embedded

        edge_types = torch.zeros((pocket_g.num_edges(), 2), device=device)
        edge_types[:, 0] = 1
        edge_embeded = self.super_edge_embedding(edge_types)
        pocket_g.edata['e_t'] = edge_embeded

        # 确保有x_t特征
        if 'x_t' not in pocket_g.ndata:
            pocket_g.ndata['x_t'] = pocket_g.ndata.get('x_1_true', torch.zeros_like(pocket_g.ndata['a_1_true'])).float()
        
        # 只保留必要的特征
        for feat in list(pocket_g.ndata.keys()):
            if feat not in required_feats:
                del pocket_g.ndata[feat]
                
        for feat in list(pocket_g.edata.keys()):
            if feat not in required_feats:
                del pocket_g.edata[feat]
        return pocket_g
        

    def build_hetero_graph(self, g: dgl.DGLGraph, pocket_g: dgl.DGLGraph, t: float) -> dgl.DGLGraph:
        
            
        device = g.device
        # 确保两个图在相同设备上
        pocket_g = pocket_g.to(device)

        with g.local_scope(), pocket_g.local_scope():

            g = g.clone()
            pocket_g = pocket_g.clone()
            
            ligand_num_nodes = g.batch_num_nodes()
            ligand_num_edges = g.batch_num_edges()
            pocket_num_nodes = pocket_g.batch_num_nodes()
            pocket_num_edges = pocket_g.batch_num_edges()
            batch_size = len(ligand_num_nodes)

            required_feats = {'x_t': torch.float32, 'scalar': torch.float32, 'e_t': torch.float32 }
            
            
            # 处理图数据
            g = self.prepare_ligand_graph(g, t, required_feats)
            pocket_g = self.prepare_pocket_graph(pocket_g, t, required_feats)
            
            batch_size = len(g.batch_num_nodes())
            
            # 计算每个图中配体和口袋节点的累积数量，用于偏移索引
            ligand_nodes_per_graph = g.batch_num_nodes()
            pocket_nodes_per_graph = pocket_g.batch_num_nodes()
            
            ligand_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(ligand_nodes_per_graph, dim=0)])
            pocket_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(pocket_nodes_per_graph, dim=0)])

            all_lp_src = []
            all_lp_dst = []
            
            # 2. 向量化优化的核心：仍然需要循环计算KNN，但只收集索引，不构建图
            # 这是一个折衷方案，避免了在循环中构建DGL图对象的巨大开销。
            # 纯向量化的KNN对于不同大小的图比较复杂，这个方案是性能和实现难度的平衡。
            for i in range(batch_size):
                # 获取当前样本在批处理图中的节点切片
                ligand_start_idx, ligand_end_idx = ligand_cumsum[i], ligand_cumsum[i+1]
                pocket_start_idx, pocket_end_idx = pocket_cumsum[i], pocket_cumsum[i+1]
                
                num_ligand = ligand_nodes_per_graph[i]
                num_pocket = pocket_nodes_per_graph[i]

                if num_ligand == 0 or num_pocket == 0:
                    continue

                ligand_pos = g.ndata['x_t'][ligand_start_idx:ligand_end_idx]
                pocket_pos = pocket_g.ndata['x_t'][pocket_start_idx:pocket_end_idx]
                
                # 计算距离并获取最近的k个口袋节点
                dist_matrix = torch.cdist(ligand_pos, pocket_pos)
                k = min(12, num_pocket)
                _, topk_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1) # dim=1很重要
                
                # 创建相对于当前样本的局部边索引
                # src是 [0, 0, ..., 1, 1, ..., N-1, N-1, ...]
                src_local = torch.arange(num_ligand, device=device).repeat_interleave(k)
                dst_local = topk_indices.flatten()
                
                # 将局部索引转换为批处理大图中的全局索引，并添加到列表中
                all_lp_src.append(src_local + ligand_start_idx)
                all_lp_dst.append(dst_local + pocket_start_idx)

            # 3. 一次性构建异构图
            data_dict = {}
            
            # 1. 添加配体内部边 (直接从批处理图中获取)
            if g.num_edges() > 0:
                ligand_src, ligand_dst = g.edges()
                data_dict[('ligand', 'll', 'ligand')] = (ligand_src, ligand_dst)
            
            # 2. 添加口袋内部边 (直接从批处理图中获取)
            if pocket_g.num_edges() > 0:
                pocket_src, pocket_dst = pocket_g.edges()
                data_dict[('pocket', 'pp', 'pocket')] = (pocket_src, pocket_dst)
            
            # 3. 添加所有收集到的配体-口袋相互作用边
            if len(all_lp_src) > 0:
                final_lp_src = torch.cat(all_lp_src)
                final_lp_dst = torch.cat(all_lp_dst)
                data_dict[('ligand', 'lp', 'pocket')] = (final_lp_src, final_lp_dst)
                data_dict[('pocket', 'pl', 'ligand')] = (final_lp_dst, final_lp_src)

            num_nodes_dict = {
                'ligand': g.num_nodes(),
                'pocket': pocket_g.num_nodes()
            }
            
            # 一次性创建最终的批处理异构图
            hetero_g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, device=device)
            
            # 4. 批量拷贝节点和边特征
            # 节点特征
            hetero_g.nodes['ligand'].data.update(g.ndata)
            hetero_g.nodes['pocket'].data.update(pocket_g.ndata)
            
            # 边特征
            if g.num_edges() > 0:
                hetero_g.edges[('ligand', 'll', 'ligand')].data.update(g.edata)
            if pocket_g.num_edges() > 0:
                hetero_g.edges[('pocket', 'pp', 'pocket')].data.update(pocket_g.edata)
            
            # 相互作用边特征
            if 'pl' in hetero_g.etypes:
                num_edges = hetero_g.num_edges(('pocket', 'pl', 'ligand'))
                edge_types = torch.zeros((num_edges, 2), device=device)
                edge_types[:, 1] = 1
                edge_embeded = self.super_edge_embedding(edge_types)
                
                hetero_g.edges[('ligand', 'lp', 'pocket')].data['e_t'] = edge_embeded
                hetero_g.edges[('pocket', 'pl', 'ligand')].data['e_t'] = edge_embeded
            
            # 恢复批处理信息
            # 设置节点批处理信息
            hetero_g.set_batch_num_nodes({
                'ligand': ligand_num_nodes,
                'pocket': pocket_num_nodes
            })
            
            # 设置边批处理信息
            hetero_g.set_batch_num_edges({
                ('ligand', 'll', 'ligand'): ligand_num_edges,
                ('pocket', 'pp', 'pocket'): pocket_num_edges,
                ('ligand', 'lp', 'pocket'): torch.tensor([len(src) for src in all_lp_src], device=device),
                ('pocket', 'pl', 'ligand'): torch.tensor([len(src) for src in all_lp_src], device=device)
            })
            
            return hetero_g

    
    def forward(self, g: dgl.DGLGraph, 
                pocket_g: dgl.DGLGraph,
                t: torch.Tensor, 
                node_batch_idx: torch.Tensor, 
                upper_edge_mask: torch.Tensor,
                apply_softmax: bool = False,
                remove_com: bool = False):
        
        device = g.device

        hetero_g = self.build_hetero_graph(g, pocket_g, t)

        with g.local_scope(), pocket_g.local_scope():
            # --- 2. 初始化配体特征 ---
             
            node_scalar_features = {'ligand': hetero_g.ndata['scalar']['ligand'],
                                    'pocket': hetero_g.ndata['scalar']['pocket']}

            node_positions = {'ligand':hetero_g.ndata['x_t']['ligand'],
                              'pocket':hetero_g.ndata['x_t']['pocket']}
            node_vec_features = {'ligand': torch.zeros((g.num_nodes(), self.n_vec_channels, 3), device=device),
                                 'pocket': torch.zeros((pocket_g.num_nodes(), self.n_vec_channels_pocket, 3), device=device)}
            
            edge_features = {('ligand','ll','ligand'):hetero_g.edata['e_t'][('ligand','ll','ligand')],
                             ('ligand', 'lp', 'pocket'):hetero_g.edata['e_t'][('ligand', 'lp', 'pocket')],
                             ('pocket', 'pl', 'ligand'):hetero_g.edata['e_t'][('pocket', 'pl', 'ligand')],
                             ('pocket','pp','pocket'):hetero_g.edata['e_t'][('pocket','pp','pocket')]
                             }
            x_diff, d = precompute_distances_hetero(hetero_g, rbf_dmax=self.rbf_dmax, rbf_dim=self.rbf_dim)


            # --- 3. 主循环：等变卷积与交叉注意力注入 ---
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.hetero_conv_layers):

                    # 标准等变卷积
                    node_scalar_features, node_vec_features = conv(
                        hetero_g,
                        scalar_feats=node_scalar_features, 
                        coord_feats=node_positions,
                        vec_feats=node_vec_features,
                        edge_feats=edge_features,
                        x_diff=x_diff,
                        d=d
                    )


                    # 周期性更新坐标和边特征
                    if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                        updater_idx = conv_idx // self.convs_per_update if self.separate_mol_updaters else 0
                        
                        node_positions['ligand'] = self.node_position_updaters[updater_idx](
                            node_scalar_features['ligand'], node_positions['ligand'], node_vec_features['ligand']
                        )
                        x_diff, d = precompute_distances_hetero(hetero_g, node_positions, rbf_dmax=self.rbf_dmax, rbf_dim=self.rbf_dim)
                        edge_features[('ligand','ll','ligand')] = self.edge_updaters[updater_idx](
                            g, node_scalar_features['ligand'], edge_features[('ligand','ll','ligand')], d=d[('ligand','ll','ligand')]
                        )
            
            node_scalar_features = node_scalar_features['ligand']
            edge_features = edge_features[('ligand','ll','ligand')]
            node_positions = node_positions['ligand']


            # --- 4. 输出头 ---
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(
                    g, feat='x_1_pred', op='mean'
                )[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

            # --- 5. 构建输出字典 ---
            dst_dict = {
                'x': node_positions,
                'a': atom_type_logits,
                'e': edge_logits
            }
            if not self.exclude_charges:
                dst_dict['c'] = atom_charge_logits

            if apply_softmax:
                for feat in dst_dict.keys():
                    if feat in ['a', 'c', 'e']:
                        dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1)

            return dst_dict
        
    def integrate(self, g: dgl.DGLGraph, pocket_g: dgl.DGLGraph, node_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor, n_timesteps: int, 
        visualize=False, 
        dfm_type='campbell',
        stochasticity=8.0, 
        high_confidence_threshold=0.9,
        cat_temp_func=None,
        forward_weight_func=None,
        tspan=None,
        **kwargs):
        """Integrate the trajectories of molecules along the vector field."""
        
        # TODO: this overrides EndpointVectorField.integrate just because it has some extra arguments
        # we should refactor this so that we don't have to copy the entire function

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']
        
        if self.enable_dynamic_graph:
            g, upper_edge_mask, edge_batch_idx = reconstruct_graph_dynamic(g, upper_edge_mask, node_batch_idx, k=self.knn_connectivity)


        edge_batch_idx = get_edge_batch_idxs(g)



        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
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
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False
            


            # compute next step and set x_t = x_s
            step_result = self.step(g, pocket_g, s_i, t_i, alpha_t_i, alpha_s_i, 
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
                **kwargs)
            
            if isinstance(step_result, tuple):
                g, upper_edge_mask, edge_batch_idx = step_result

            else:
                g = step_result
            

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat == 'e':
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

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames, upper_edge_mask
        
        return g, upper_edge_mask
    
    def step(self, g: dgl.DGLGraph, pocket_g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
            alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
            node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
            cat_temp_func: Callable,
            forward_weight_func: Callable, 
            dfm_type: str = 'campbell',
            stochasticity: float = 8.0,
            high_confidence_threshold: float = 0.9, 
            last_step: bool = False,
            inv_temp_func: Callable = None,):

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
        

        dst_dict = self(
            g, 
            pocket_g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        
        dt = s_i - t_i
        sigma_t_i = self.interpolant_scheduler.sigma_t(t_i.unsqueeze(0), self.eta)[0, 0]


        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        noise = torch.randn_like(x_t)

        g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i) + sigma_t_i * noise * torch.sqrt(dt)
        # g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i)

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1) # has shape (num_nodes,)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

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
                                batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(), 
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                # record predicted endpoint for visualization
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)

                xt = gat_step(
                    p_1_given_t=p_s_1, 
                    xt=xt, 
                    alpha_t=alpha_t_i[feat_idx], 
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    forward_weight=forward_weight_func(t_i),
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                )
                                   
            
            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled
            
            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled\
        
        if self.enable_dynamic_graph and t_i < 0.95:
            g, upper_edge_mask, edge_batch_idx = reconstruct_graph_dynamic(g, upper_edge_mask, node_batch_idx, k=self.knn_connectivity)
            return g, upper_edge_mask, edge_batch_idx


        return g

