from pathlib import Path

import dgl
import torch
from torch.nn.functional import one_hot

from src.data_processing.priors import coupled_node_prior, edge_prior


def collate(graphs):
    return dgl.batch(graphs)


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super().__init__()

        self.prior_config = prior_config
        self.dataset_config = dataset_config
        self.explicit_aromaticity = dataset_config["explicit_aromaticity"]
        self.n_bond_types = 5 if self.explicit_aromaticity else 4

        processed_data_dir = Path(dataset_config["processed_data_dir"])
        if not processed_data_dir.exists():
            processed_data_dir = Path(__file__).resolve().parents[2] / processed_data_dir
            if not processed_data_dir.exists():
                raise FileNotFoundError(
                    f"processed data directory {dataset_config['processed_data_dir']} not found"
                )

        self.processed_data_dir = processed_data_dir

        if dataset_config["dataset_name"] not in {"geom", "qm9", "geom_5conf", "coconut"}:
            raise NotImplementedError("unsupported dataset_name")

        data_file = processed_data_dir / f"{split}_data_processed.pt"
        data_dict = torch.load(data_file)

        self.atom_types = data_dict["atom_types"]
        self.atom_charges = data_dict["atom_charges"]
        self.bond_types = data_dict["bond_types"]
        self.bond_idxs = data_dict["bond_idxs"]
        self.node_idx_array = data_dict["node_idx_array"]
        self.edge_idx_array = data_dict["edge_idx_array"]

    def __len__(self):
        return self.node_idx_array.shape[0]

    def __getitem__(self, idx):
        node_start_idx = self.node_idx_array[idx, 0]
        node_end_idx = self.node_idx_array[idx, 1]
        edge_start_idx = self.edge_idx_array[idx, 0]
        edge_end_idx = self.edge_idx_array[idx, 1]

        atom_types = self.atom_types[node_start_idx:node_end_idx].float()
        atom_charges = self.atom_charges[node_start_idx:node_end_idx].long()
        bond_types = self.bond_types[edge_start_idx:edge_end_idx].int()
        bond_idxs = self.bond_idxs[edge_start_idx:edge_end_idx].long()

        n_atoms = atom_types.shape[0]
        adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int32)
        adj[bond_idxs[:, 0], bond_idxs[:, 1]] = bond_types

        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        edge_labels = one_hot(edge_labels.to(torch.int64), num_classes=self.n_bond_types).float()
        atom_charges = one_hot(atom_charges + 2, num_classes=6).float()

        g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
        g.edata["e_1_true"] = edge_labels
        g.ndata["a_1_true"] = atom_types
        g.ndata["c_1_true"] = atom_charges

        dst_dict = {
            "a": atom_types,
            "c": atom_charges,
        }
        prior_node_feats = coupled_node_prior(dst_dict=dst_dict, prior_config=self.prior_config)
        for feat, prior_feat in prior_node_feats.items():
            g.ndata[f"{feat}_0"] = prior_feat

        upper_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        upper_edge_mask[: upper_edge_idxs.shape[1]] = True
        g.edata["e_0"] = edge_prior(
            upper_edge_mask,
            self.prior_config["e"],
            explicit_aromaticity=self.explicit_aromaticity,
        )

        return g
