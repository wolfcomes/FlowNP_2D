from pathlib import Path
from typing import Dict, List

import dgl
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from src.analysis.molecule_builder import SampledMolecule
from src.data_processing.priors import edge_prior, inference_prior_register
from src.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask
from src.models.interpolant_scheduler import InterpolantScheduler
from src.models.lr_scheduler import LRScheduler
from src.models.vector_field_2d import CTMCVectorField2D


class FlowMol(pl.LightningModule):
    canonical_feat_order = ["a", "c", "e"]
    node_feats = ["a", "c"]
    edge_feats = ["e"]

    def __init__(
        self,
        atom_type_map: List[str],
        n_atoms_hist_file: str,
        marginal_dists_file: str,
        n_atom_charges: int = 6,
        n_bond_types: int = 5,
        sample_interval: float = 1.0,
        n_mols_to_sample: int = 64,
        time_scaled_loss: bool = True,
        exclude_charges: bool = False,
        weight_ae: bool = False,
        target_blur: float = 0.0,
        total_loss_weights: Dict[str, float] = None,
        lr_scheduler_config: dict = None,
        interpolant_scheduler_config: dict = None,
        vector_field_config: dict = None,
        prior_config: dict = None,
        default_n_timesteps: int = 250,
        explicit_aromaticity: bool = False,
    ):
        super().__init__()

        self.canonical_feat_order = list(self.canonical_feat_order)
        self.node_feats = list(self.node_feats)
        self.edge_feats = list(self.edge_feats)
        self.total_loss_weights = dict(total_loss_weights or {})
        self.lr_scheduler_config = dict(lr_scheduler_config or {})
        self.prior_config = dict(prior_config or {})

        self.atom_type_map = atom_type_map
        self.n_atom_types = len(atom_type_map)
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = n_bond_types if explicit_aromaticity else n_bond_types - 1
        self.time_scaled_loss = time_scaled_loss
        self.exclude_charges = exclude_charges
        self.weight_ae = weight_ae
        self.target_blur = target_blur
        self.n_atoms_hist_file = n_atoms_hist_file
        self.marginal_dists_file = marginal_dists_file
        self.default_n_timesteps = default_n_timesteps
        self.explicit_aromaticity = explicit_aromaticity

        if self.target_blur < 0.0:
            raise ValueError("target_blur must be non-negative")

        processed_data_dir = Path(self.marginal_dists_file).parent
        if not processed_data_dir.exists():
            repo_root = Path(__file__).resolve().parents[2]
            self.marginal_dists_file = repo_root / self.marginal_dists_file
            self.n_atoms_hist_file = repo_root / self.n_atoms_hist_file

        if self.exclude_charges:
            self.node_feats.remove("c")
            self.canonical_feat_order.remove("c")
            self.total_loss_weights.pop("c", None)

        self.n_cat_dict = {
            "a": self.n_atom_types,
            "c": self.n_atom_charges,
            "e": self.n_bond_types,
        }

        for feat in self.canonical_feat_order:
            self.total_loss_weights.setdefault(feat, 1.0)

        self.build_n_atoms_dist(self.n_atoms_hist_file)

        self.interpolant_scheduler = InterpolantScheduler(
            canonical_feat_order=self.canonical_feat_order,
            **(interpolant_scheduler_config or {}),
        )
        self.vector_field = CTMCVectorField2D(
            n_atom_types=self.n_atom_types,
            canonical_feat_order=self.canonical_feat_order,
            interpolant_scheduler=self.interpolant_scheduler,
            n_charges=n_atom_charges,
            n_bond_types=self.n_bond_types,
            exclude_charges=self.exclude_charges,
            **(vector_field_config or {}),
        )

        self.sample_interval = sample_interval
        self.n_mols_to_sample = n_mols_to_sample
        self.last_sample_marker = 0.0
        self.last_epoch_exact = 0.0

        self.save_hyperparameters()

        marginal_dists = torch.load(self.marginal_dists_file)
        self.p_a, _, self.p_e, _ = marginal_dists[:4]

    def configure_loss_fns(self, device):
        reduction = "none"
        categorical_loss_fn = nn.CrossEntropyLoss

        cat_kwargs = {"ignore_index": -100}
        self.loss_fn_dict = {
            "a": categorical_loss_fn(reduction=reduction, **cat_kwargs),
            "c": categorical_loss_fn(reduction=reduction, **cat_kwargs),
            "e": categorical_loss_fn(reduction=reduction, **cat_kwargs),
        }

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):
        if not hasattr(self, "batches_per_epoch"):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

        epoch_exact = self.current_epoch + batch_idx / self.batches_per_epoch
        self.last_epoch_exact = epoch_exact
        self.lr_scheduler.step_lr(epoch_exact)

        if epoch_exact - self.last_sample_marker >= self.sample_interval:
            self.last_sample_marker = epoch_exact
            self.eval()
            with torch.no_grad():
                self.sample_random_sizes(n_molecules=self.n_mols_to_sample, device=g.device)
            self.train()

        losses = self(g)

        train_log_dict = {"epoch_exact": epoch_exact}
        for key, value in losses.items():
            train_log_dict[f"{key}_train_loss"] = value

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat] * losses[feat]
            self.log(f"{feat}_loss", losses[feat], prog_bar=True, on_step=True, sync_dist=True)

        self.log_dict(train_log_dict, sync_dist=True)
        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, sync_dist=True)
        return total_loss

    def validation_step(self, g: dgl.DGLGraph, batch_idx: int):
        losses = self(g)

        val_log_dict = {"epoch_exact": self.last_epoch_exact}
        for key, value in losses.items():
            val_log_dict[f"{key}_val_loss"] = value

        self.log_dict(val_log_dict, batch_size=g.batch_size, sync_dist=True)

        total_loss = torch.zeros(1, device=g.device, requires_grad=False)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat] * losses[feat]

        self.log(
            "val_total_loss",
            total_loss,
            prog_bar=True,
            batch_size=g.batch_size,
            on_step=True,
            sync_dist=True,
        )
        return total_loss

    def forward(self, g: dgl.DGLGraph):
        batch_size = g.batch_size
        device = g.device

        if not hasattr(self, "loss_fn_dict"):
            self.configure_loss_fns(device=device)

        if self.exclude_charges:
            self.loss_fn_dict.pop("c", None)

        node_batch_idx, edge_batch_idx = get_batch_idxs(g)
        upper_edge_mask = get_upper_edge_mask(g)
        t = torch.rand(batch_size, device=device).float()

        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)
        g = self.add_random_node_features(g)
        vf_output = self.vector_field(
            g,
            t,
            node_batch_idx=node_batch_idx,
            edge_batch_idx=edge_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )

        targets = {}
        for feat in self.canonical_feat_order:
            data_src = g.edata if feat == "e" else g.ndata
            target = data_src[f"{feat}_1_true"]

            if feat == "e":
                target = target[upper_edge_mask]

            if self.target_blur == 0.0:
                target = target.argmax(dim=-1)
            else:
                target = target + torch.randn_like(target) * self.target_blur
                target = fn.softmax(target, dim=-1).argmax(dim=-1)

            if feat == "e":
                xt_idxs = data_src[f"{feat}_t"][upper_edge_mask].argmax(-1)
            else:
                xt_idxs = data_src[f"{feat}_t"].argmax(-1)
            target[xt_idxs != self.n_cat_dict[feat]] = -100
            targets[feat] = target

        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == "e":
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            target = targets[feat]
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target) * weight
            losses[feat] = losses[feat][target != -100].mean()

        return losses

    def sample_prior(self, g, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        num_nodes = g.num_nodes()
        device = g.device

        for feat in self.node_feats:
            prior_fn = inference_prior_register["ctmc"]
            args = [num_nodes, self.n_cat_dict[feat]]
            kwargs = self.prior_config[feat]["kwargs"]
            g.ndata[f"{feat}_0"] = prior_fn(*args, **kwargs).to(device)

        g.edata["e_0"] = edge_prior(
            upper_edge_mask,
            self.prior_config["e"],
            explicit_aromaticity=self.explicit_aromaticity,
        ).to(device)
        g = self.add_random_node_features(g)
        return g

    def add_random_node_features(self, g: dgl.DGLGraph):
        n_random_node_feats = getattr(self.vector_field, "n_random_node_feats", 0)
        if n_random_node_feats <= 0:
            return g

        g.ndata["z_t"] = torch.randn(
            g.num_nodes(),
            n_random_node_feats,
            device=g.device,
            dtype=torch.float32,
        )
        return g

    def configure_optimizers(self):
        weight_decay = self.lr_scheduler_config.get("weight_decay", 0)
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr_scheduler_config["base_lr"],
            weight_decay=weight_decay,
        )
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return optimizer

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def sample_n_atoms(self, n_molecules: int, **kwargs):
        n_atoms = self.n_atoms_dist.sample((n_molecules,), **kwargs)
        return self.n_atoms_map[n_atoms]

    def sample_random_sizes(
        self,
        n_molecules: int,
        device="cuda:0",
        stochasticity=None,
        high_confidence_threshold=None,
        xt_traj=False,
        ep_traj=False,
        **kwargs,
    ):
        atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)
        return self.sample(
            atoms_per_molecule,
            device=device,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            **kwargs,
        )

    @torch.no_grad()
    def sample(
        self,
        n_atoms: torch.Tensor,
        n_timesteps: int = None,
        device="cuda:0",
        stochasticity=None,
        high_confidence_threshold=None,
        xt_traj=False,
        ep_traj=False,
        **kwargs,
    ):
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        visualize = xt_traj or ep_traj

        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        graphs = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            graphs.append(dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device))

        g = dgl.batch(graphs)
        upper_edge_mask = get_upper_edge_mask(g)
        node_batch_idx, _ = get_batch_idxs(g)
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        integrate_kwargs = {
            "upper_edge_mask": upper_edge_mask,
            "n_timesteps": n_timesteps,
            "visualize": visualize,
            "stochasticity": stochasticity,
            "high_confidence_threshold": high_confidence_threshold,
        }
        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames, upper_edge_mask = itg_result
        elif isinstance(itg_result, tuple):
            g, upper_edge_mask = itg_result
        else:
            g = itg_result

        g.edata["ue_mask"] = upper_edge_mask
        g = g.to("cpu")

        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):
            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(
                SampledMolecule(
                    *args,
                    ctmc_mol=True,
                    build_xt_traj=xt_traj,
                    build_ep_traj=ep_traj,
                    exclude_charges=self.exclude_charges,
                    explicit_aromaticity=self.explicit_aromaticity,
                )
            )

        return molecules
