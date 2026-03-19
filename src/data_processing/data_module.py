import dgl
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_processing.dataset import MoleculeDataset
from src.data_processing.samplers import (
    SameSizeDistributedMoleculeSampler,
    SameSizeMoleculeSampler,
)


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config: dict,
        dm_prior_config: dict,
        batch_size: int,
        num_workers: int = 0,
        distributed: bool = False,
        max_num_edges: int = 40000,
    ):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_config = dm_prior_config
        self.max_num_edges = int(max_num_edges)
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.load_dataset("train")
            self.val_dataset = self.load_dataset("val")

    def load_dataset(self, split: str):
        return MoleculeDataset(split, self.dataset_config, prior_config=self.prior_config)

    def _build_batch_sampler(self, dataset, batch_size: int, shuffle: bool):
        if self.distributed:
            return SameSizeDistributedMoleculeSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                max_num_edges=self.max_num_edges,
            )

        return SameSizeMoleculeSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            max_num_edges=self.max_num_edges,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._build_batch_sampler(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            collate_fn=dgl.batch,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=self._build_batch_sampler(
                self.val_dataset, batch_size=self.batch_size * 2, shuffle=False
            ),
            collate_fn=dgl.batch,
            num_workers=self.num_workers,
        )
