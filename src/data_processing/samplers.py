from torch.utils.data import Sampler, DistributedSampler
from src.data_processing.dataset import MoleculeDataset
import torch


class SameSizeMoleculeSampler(Sampler):

    def __init__(self, dataset: MoleculeDataset, batch_size: int, idxs: torch.Tensor = None, shuffle: bool = True, max_num_edges: int = 40000):
        super().__init__(dataset)
        self.dataset: MoleculeDataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_num_edges = max_num_edges

        if idxs is None:
            self.idxs = torch.arange(len(dataset))
        else:
            self.idxs = idxs


        node_idx_array = self.dataset.node_idx_array

        if idxs is not None:
            node_idx_array = node_idx_array[idxs]

        self.num_nodes = node_idx_array[:, 1] - node_idx_array[:, 0] # array of shape (indicies.shape[0],) containing the number of nodes in each graph

    def _batch_size_for_n_nodes(self, n_nodes: int) -> int:
        n_edges_per_mol = n_nodes * n_nodes - n_nodes
        if n_edges_per_mol <= 0:
            return self.batch_size
        return max(1, min(self.batch_size, self.max_num_edges // n_edges_per_mol))

    def _iter_batches(self):
        n_nodes_values = self.num_nodes.unique()
        if self.shuffle:
            n_nodes_values = n_nodes_values[torch.randperm(len(n_nodes_values))]
        else:
            n_nodes_values = torch.sort(n_nodes_values).values

        all_batches = []
        for n_nodes in n_nodes_values.tolist():
            idxs_with_n_nodes = self.idxs[torch.where(self.num_nodes == n_nodes)[0]]
            if self.shuffle:
                idxs_with_n_nodes = idxs_with_n_nodes[torch.randperm(len(idxs_with_n_nodes))]

            batch_size = self._batch_size_for_n_nodes(int(n_nodes))
            for start_idx in range(0, len(idxs_with_n_nodes), batch_size):
                all_batches.append(idxs_with_n_nodes[start_idx:start_idx + batch_size])

        if self.shuffle and len(all_batches) > 1:
            batch_order = torch.randperm(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in batch_order]

        return all_batches

    def __iter__(self):
        yield from self._iter_batches()

    def __len__(self):
        return len(self._iter_batches())


class SameSizeDistributedMoleculeSampler(DistributedSampler):

    def __init__(self, dataset: MoleculeDataset, batch_size: int, max_num_edges: int = 40000, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.max_num_edges = max_num_edges

    def __iter__(self):
        indicies = list(super().__iter__())
        indicies = torch.tensor(indicies)
        batch_sampler = SameSizeMoleculeSampler(
            self.dataset,
            self.batch_size,
            idxs=indicies,
            shuffle=self.shuffle,
            max_num_edges=self.max_num_edges,
        )
        return iter(batch_sampler)
    
    def __len__(self):
        return max(1, self.num_samples // max(1, self.batch_size))
