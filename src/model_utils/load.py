from pathlib import Path

import yaml

from src.data_processing.data_module import MoleculeDataModule
from src.models.flowmol import FlowMol


def read_config_file(config_file: Path) -> dict:
    with open(config_file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def model_from_config(config: dict, seed_ckpt: Path = None) -> FlowMol:
    atom_type_map = config["dataset"]["atom_map"]
    sample_interval = config["training"]["evaluation"]["sample_interval"]
    mols_to_sample = config["training"]["evaluation"]["mols_to_sample"]
    explicit_aromaticity = config["dataset"]["explicit_aromaticity"]

    processed_data_dir = Path(config["dataset"]["processed_data_dir"])
    n_atoms_hist_filepath = processed_data_dir / "train_data_n_atoms_histogram.pt"
    marginal_dists_file = processed_data_dir / "train_data_marginal_dists.pt"

    model_kwargs = dict(
        atom_type_map=atom_type_map,
        n_atoms_hist_file=n_atoms_hist_filepath,
        marginal_dists_file=marginal_dists_file,
        sample_interval=sample_interval,
        n_mols_to_sample=mols_to_sample,
        explicit_aromaticity=explicit_aromaticity,
        vector_field_config=config["vector_field"],
        interpolant_scheduler_config=config["interpolant_scheduler"],
        lr_scheduler_config=config["lr_scheduler"],
        **config["mol_fm"],
    )

    if seed_ckpt is not None:
        # Allow warm-starting from checkpoints created before edge-aware node
        # updates were added to the 2D vector field.
        return FlowMol.load_from_checkpoint(seed_ckpt, strict=False, **model_kwargs)
    return FlowMol(**model_kwargs)


def data_module_from_config(config: dict) -> MoleculeDataModule:
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    distributed = config["training"]["trainer_args"]["devices"] > 1
    max_num_edges = config["training"].get("max_num_edges", 40000)

    return MoleculeDataModule(
        dataset_config=config["dataset"],
        dm_prior_config=config["mol_fm"]["prior_config"],
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        max_num_edges=max_num_edges,
    )
