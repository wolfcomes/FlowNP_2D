import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
import argparse
import numpy as np
import dgl
from process_coconut import resolve_coconut_sdf_file
from process_qm9 import resolve_qm9_support_files
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from src.analysis.molecule_builder import build_molecule
from src.model_utils.load import data_module_from_config, model_from_config
from src.model_utils.sweep_config import register_hyperparameter_args
from src.models.vector_field_2d import CTMCVectorField2D
from src.models.interpolant_scheduler import InterpolantScheduler
from src.models.flowmol import FlowMol
from src.data_processing.geom import featurize_molecule


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_processed_split(
    processed_dir: Path, split_prefix: str, n_graphs: int = 1, n_atoms: int = 2
) -> None:
    atom_types = []
    atom_charges = []
    bond_types = []
    bond_idxs = []
    node_idx_array = []
    edge_idx_array = []
    smiles = []

    node_offset = 0
    edge_offset = 0
    for graph_idx in range(n_graphs):
        atom_types.append(torch.tensor([[1, 0]] * n_atoms, dtype=torch.bool))
        atom_charges.append(torch.zeros(n_atoms, dtype=torch.int32))

        graph_bond_idxs = []
        if n_atoms > 1:
            graph_bond_idxs = [[i, i + 1] for i in range(n_atoms - 1)]
            bond_types.append(torch.ones(len(graph_bond_idxs), dtype=torch.int32))
            bond_idxs.append(torch.tensor(graph_bond_idxs, dtype=torch.int32))
        else:
            bond_types.append(torch.zeros(0, dtype=torch.int32))
            bond_idxs.append(torch.zeros((0, 2), dtype=torch.int32))

        node_idx_array.append([node_offset, node_offset + n_atoms])
        edge_idx_array.append([edge_offset, edge_offset + len(graph_bond_idxs)])
        smiles.append("C" * max(n_atoms, 1))

        node_offset += n_atoms
        edge_offset += len(graph_bond_idxs)

    data = {
        "smiles": smiles,
        "atom_types": torch.cat(atom_types, dim=0),
        "atom_charges": torch.cat(atom_charges, dim=0),
        "bond_types": torch.cat(bond_types, dim=0),
        "bond_idxs": torch.cat(bond_idxs, dim=0),
        "node_idx_array": torch.tensor(node_idx_array, dtype=torch.int32),
        "edge_idx_array": torch.tensor(edge_idx_array, dtype=torch.int32),
    }
    torch.save(data, processed_dir / f"{split_prefix}_processed.pt")


def _write_training_artifacts(processed_dir: Path) -> None:
    torch.save(
        (torch.tensor([2], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)),
        processed_dir / "train_data_n_atoms_histogram.pt",
    )
    p_a = torch.tensor([1.0, 0.0], dtype=torch.float32)
    p_c = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    p_e = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    p_c_given_a = torch.ones(2, 6, dtype=torch.float32) / 6
    torch.save((p_a, p_c, p_e, p_c_given_a), processed_dir / "train_data_marginal_dists.pt")


def _minimal_config(processed_dir: Path) -> dict:
    return {
        "dataset": {
            "atom_map": ["C", "H"],
            "dataset_name": "qm9",
            "dataset_size": None,
            "processed_data_dir": str(processed_dir),
            "raw_data_dir": "unused",
            "explicit_aromaticity": False,
        },
        "interpolant_scheduler": {
            "schedule_type": {"a": "linear", "c": "linear", "e": "linear"},
            "params": {"a": 1.0, "c": 1.0, "e": 1.0},
        },
        "lr_scheduler": {
            "base_lr": 1e-4,
            "restart_interval": 10,
            "restart_type": "linear",
            "warmup_length": 1.0,
            "weight_decay": 0.0,
        },
        "mol_fm": {
            "prior_config": {
                "a": {"align": False, "kwargs": {}},
                "c": {"align": False, "kwargs": {}},
                "e": {"align": False, "kwargs": {}},
            },
            "target_blur": 0.0,
            "time_scaled_loss": False,
            "total_loss_weights": {"a": 1.0, "c": 1.0, "e": 1.0},
            "weight_ae": False,
            "exclude_charges": False,
        },
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "evaluation": {
                "sample_interval": 1.0,
                "mols_to_sample": 1,
                "val_loss_interval": 1.0,
            },
            "trainer_args": {
                "devices": 1,
            },
        },
        "vector_field": {
            "n_hidden": 32,
            "n_hidden_edge_feats": 16,
            "n_recycles": 1,
            "n_molecule_updates": 1,
            "convs_per_update": 1,
        },
        "checkpointing": {},
    }


def _make_two_atom_batch(edge_state_idx: int) -> dgl.DGLGraph:
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph = dgl.batch([graph])

    graph.ndata["a_t"] = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )
    graph.ndata["c_t"] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    edge_state = torch.zeros(5, dtype=torch.float32)
    edge_state[edge_state_idx] = 1.0
    graph.edata["e_t"] = edge_state.repeat(graph.num_edges(), 1)
    return graph


class ContractTests(unittest.TestCase):
    def test_qm9_config_loads_pure_2d_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            for split in ("train_data", "val_data"):
                _write_minimal_processed_split(processed_dir, split)
            _write_training_artifacts(processed_dir)

            config = _minimal_config(processed_dir)

            dm = data_module_from_config(config)
            dm.setup("fit")
            batch = next(iter(dm.train_dataloader()))

            self.assertNotIn("x_1_true", batch.ndata)
            self.assertIn("a_1_true", batch.ndata)
            self.assertIn("c_1_true", batch.ndata)
            self.assertIn("e_1_true", batch.edata)

            model = model_from_config(config)
            self.assertEqual(model.canonical_feat_order, ["a", "c", "e"])

    def test_process_qm9_uses_local_featurizer_import(self):
        source = (REPO_ROOT / "process_qm9.py").read_text()
        self.assertIn("from src.data_processing.geom import MoleculeFeaturizer", source)

    def test_process_coconut_does_not_gate_on_3d_molecules(self):
        source = (REPO_ROOT / "process_coconut.py").read_text()
        self.assertNotIn("is_3d_molecule", source)
        self.assertNotIn("Non-3D molecules skipped", source)

    def test_main_loader_path_has_no_pocket_runtime_dependencies(self):
        load_source = (REPO_ROOT / "src/model_utils/load.py").read_text()
        data_module_source = (REPO_ROOT / "src/data_processing/data_module.py").read_text()
        dataset_source = (REPO_ROOT / "src/data_processing/dataset.py").read_text()

        self.assertNotIn("PocketFlowMol", load_source)
        self.assertNotIn("PocketLigandDataModule", load_source)
        self.assertNotIn("PocketLigandDataset", data_module_source)
        self.assertNotIn("class PocketLigandDataset", dataset_source)

    def test_pure_2d_model_path_has_no_active_x_or_dynamic_graph_runtime(self):
        flowmol_source = (REPO_ROOT / "src/models/flowmol.py").read_text()
        vf_source = (REPO_ROOT / "src/models/vector_field_2d.py").read_text()

        self.assertNotIn("PocketFlowMol", flowmol_source)
        self.assertNotIn("reconstruct_graph_dynamic", flowmol_source)
        self.assertIn("canonical_feat_order", flowmol_source)
        self.assertIn('"a"', flowmol_source)
        self.assertIn('"c"', flowmol_source)
        self.assertIn('"e"', flowmol_source)
        self.assertIn("self.sde = False", vf_source)

    def test_active_2d_configs_have_no_x_scheduler_or_3d_vector_keys(self):
        for path_str in ("configs/qm9_ctmc.yaml", "configs/coconut_ctmc.yaml"):
            config = yaml.safe_load((REPO_ROOT / path_str).read_text())

            self.assertNotIn("x", config["interpolant_scheduler"]["params"])
            self.assertNotIn("x", config["interpolant_scheduler"]["schedule_type"])
            self.assertNotIn("x", config["mol_fm"]["prior_config"])
            self.assertNotIn("x", config["mol_fm"]["total_loss_weights"])

            vector_cfg = config["vector_field"]
            for bad_key in (
                "rbf_dmax",
                "rbf_dim",
                "n_vec_channels",
                "n_hidden_scalars",
                "n_message_gvps",
                "n_update_gvps",
                "enable_dynamic_graph",
                "knn_connectivity",
                "sde",
            ):
                self.assertNotIn(bad_key, vector_cfg)

    def test_build_molecule_can_generate_rdkit_coords_without_input_positions(self):
        mol = build_molecule(
            positions=None,
            atom_types=["C", "C"],
            atom_charges=[0, 0],
            bond_src_idxs=[0],
            bond_dst_idxs=[1],
            bond_types=[1],
        )
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumConformers(), 1)

    def test_build_molecule_preserves_provided_2d_positions(self):
        mol = build_molecule(
            positions=torch.tensor([[0.0, 0.0], [1.5, 0.0]], dtype=torch.float32),
            atom_types=["C", "C"],
            atom_charges=[0, 0],
            bond_src_idxs=[0],
            bond_dst_idxs=[1],
            bond_types=[1],
        )
        self.assertIsNotNone(mol)
        conf = mol.GetConformer()
        self.assertFalse(conf.Is3D())
        coords = conf.GetPositions()
        self.assertAlmostEqual(float(coords[0][0]), 0.0, places=6)
        self.assertAlmostEqual(float(coords[1][0]), 1.5, places=6)
        self.assertAlmostEqual(float(coords[0][2]), 0.0, places=6)
        self.assertAlmostEqual(float(coords[1][2]), 0.0, places=6)

    def test_build_molecule_prefers_3d_embedding_without_input_positions(self):
        mol = build_molecule(
            positions=None,
            atom_types=["C"] * 6,
            atom_charges=[0] * 6,
            bond_src_idxs=[0, 1, 2, 3, 4, 5],
            bond_dst_idxs=[1, 2, 3, 4, 5, 0],
            bond_types=[1, 1, 1, 1, 1, 1],
        )
        self.assertIsNotNone(mol)
        conf = mol.GetConformer()
        self.assertTrue(conf.Is3D())
        z_values = [float(coord[2]) for coord in conf.GetPositions()]
        self.assertGreater(max(z_values) - min(z_values), 1e-3)

    def test_build_molecule_falls_back_to_2d_coords_when_3d_embedding_fails(self):
        fallback_mol = Chem.MolFromSmiles("CC")
        fallback_conf = Chem.Conformer(fallback_mol.GetNumAtoms())
        fallback_conf.Set3D(False)
        fallback_conf.SetAtomPosition(0, Point3D(-0.75, 0.0, 0.0))
        fallback_conf.SetAtomPosition(1, Point3D(0.75, 0.0, 0.0))
        fallback_mol.AddConformer(fallback_conf)
        fallback_mol.SetProp("_fallback", "1")

        with patch(
            "src.analysis.molecule_builder._build_2d_fallback_molecule",
            return_value=fallback_mol,
        ) as mocked_build_2d, patch(
            "src.analysis.molecule_builder._embed_plausible_conformer", return_value=None
        ) as mocked_embed:
            mol = build_molecule(
                positions=None,
                atom_types=["C", "C"],
                atom_charges=[0, 0],
                bond_src_idxs=[0],
                bond_dst_idxs=[1],
                bond_types=[1],
            )

        self.assertIsNotNone(mol)
        mocked_build_2d.assert_called_once()
        mocked_embed.assert_called_once()
        self.assertTrue(mol.HasProp("_fallback"))
        self.assertEqual(mol.GetNumConformers(), 1)
        self.assertFalse(mol.GetConformer().Is3D())

    def test_build_molecule_keeps_unsanitizable_topology_as_2d_fallback(self):
        mol = build_molecule(
            positions=None,
            atom_types=["C"] * 6,
            atom_charges=[0] * 6,
            bond_src_idxs=[0, 0, 0, 0, 0],
            bond_dst_idxs=[1, 2, 3, 4, 5],
            bond_types=[1, 1, 1, 1, 1],
        )
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 6)
        self.assertEqual(mol.GetNumBonds(), 5)
        self.assertEqual(mol.GetNumConformers(), 1)
        self.assertFalse(mol.GetConformer().Is3D())

    def test_model_forward_runs_without_x_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            for split in ("train_data", "val_data"):
                _write_minimal_processed_split(processed_dir, split)
            _write_training_artifacts(processed_dir)

            config = _minimal_config(processed_dir)
            dm = data_module_from_config(config)
            dm.setup("fit")
            batch = next(iter(dm.train_dataloader()))

            model = model_from_config(config)
            losses = model(batch)
            self.assertEqual(set(losses.keys()), {"a", "c", "e"})

    def test_data_module_respects_max_num_edges_when_batching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            for split in ("train_data", "val_data"):
                _write_minimal_processed_split(processed_dir, split, n_graphs=4, n_atoms=3)
            _write_training_artifacts(processed_dir)

            config = _minimal_config(processed_dir)
            config["training"]["batch_size"] = 4
            config["training"]["max_num_edges"] = 6

            dm = data_module_from_config(config)
            dm.setup("fit")

            train_batch = next(iter(dm.train_dataloader()))
            val_batch = next(iter(dm.val_dataloader()))

            self.assertEqual(train_batch.batch_size, 1)
            self.assertEqual(val_batch.batch_size, 1)

    def test_model_from_config_loads_seed_checkpoint_non_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            _write_training_artifacts(processed_dir)

            config = _minimal_config(processed_dir)

            with patch("src.model_utils.load.FlowMol.load_from_checkpoint") as mocked_loader:
                model_from_config(config, seed_ckpt=Path("legacy.ckpt"))

            mocked_loader.assert_called_once()
            _, kwargs = mocked_loader.call_args
            self.assertEqual(kwargs["strict"], False)

    def test_sweep_config_only_registers_2d_hyperparameters(self):
        parser = argparse.ArgumentParser()
        register_hyperparameter_args(parser)
        option_dests = {action.dest for action in parser._actions}

        self.assertNotIn("x_loss_weight", option_dests)
        self.assertNotIn("x_cos_param", option_dests)
        self.assertNotIn("n_vec_channels", option_dests)
        self.assertNotIn("n_hidden_scalars", option_dests)
        self.assertNotIn("rbf_dmax", option_dests)
        self.assertIn("n_hidden", option_dests)
        self.assertIn("n_hidden_edge_feats", option_dests)

    def test_pocket_only_loader_and_config_are_removed(self):
        self.assertFalse((REPO_ROOT / "src/model_utils/load_pocket.py").exists())
        self.assertFalse((REPO_ROOT / "configs/crossdock_ctmc.yaml").exists())

    def test_preprocessing_scripts_import_torch_before_numpy(self):
        for rel_path in ("process_qm9.py", "process_coconut.py"):
            source = (REPO_ROOT / rel_path).read_text()
            torch_idx = source.index("import torch")
            numpy_idx = source.index("import numpy")
            self.assertLess(torch_idx, numpy_idx, rel_path)

    def test_resolve_qm9_support_files_prefers_local_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_raw = root / "data" / "qm9_raw"
            fallback_raw = root / "FlowNP" / "data" / "qm9_raw"
            local_raw.mkdir(parents=True)
            fallback_raw.mkdir(parents=True)

            (local_raw / "gdb9.sdf").write_text("sdf")
            (fallback_raw / "gdb9.sdf.csv").write_text("csv")
            (fallback_raw / "uncharacterized.txt").write_text("header\n" * 12)

            sdf_file, csv_file, bad_file = resolve_qm9_support_files(local_raw, root / "FlowNP")
            self.assertEqual(sdf_file, local_raw / "gdb9.sdf")
            self.assertEqual(csv_file, fallback_raw / "gdb9.sdf.csv")
            self.assertEqual(bad_file, fallback_raw / "uncharacterized.txt")

    def test_resolve_coconut_sdf_file_uses_available_sdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            sdf_file = raw_dir / "coconut_sdf_3d-06-2025.sdf"
            sdf_file.write_text("sdf")
            self.assertEqual(resolve_coconut_sdf_file(raw_dir), sdf_file)

    def test_qm9_preprocessing_uses_relaxed_sdf_reader(self):
        source = (REPO_ROOT / "process_qm9.py").read_text()
        self.assertIn("RDLogger.DisableLog", source)
        self.assertIn("sanitize=False", source)

    def test_featurize_molecule_skips_aromatic_bonds_in_kekulized_mode(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        AllChem.Compute2DCoords(mol)

        aromatic_adj = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        aromatic_adj[0, 1] = aromatic_adj[1, 0] = 1.5

        with patch("src.data_processing.geom.Chem.rdmolops.GetAdjacencyMatrix", return_value=aromatic_adj):
            result = featurize_molecule(mol, {"C": 0}, explicit_aromaticity=False)

        self.assertEqual(result, (None, None, None, None, None, None))

    def test_qm9_preprocessing_does_not_expect_legacy_aromatic_return_value(self):
        source = (REPO_ROOT / "process_qm9.py").read_text()
        self.assertNotIn("bond_order_counts, atom_aromatic = mol_featurizer.featurize_molecules", source)

    def test_vector_field_uses_scalar_gvp_style_convs(self):
        source = (REPO_ROOT / "src/models/vector_field_2d.py").read_text()
        self.assertIn("class ScalarGVPConv2D", source)
        self.assertNotIn("dglnn.GraphConv", source)

    def test_vector_field_handles_zero_in_degree_nodes(self):
        scheduler = InterpolantScheduler(
            canonical_feat_order=["a", "c", "e"],
            schedule_type={"a": "linear", "c": "linear", "e": "linear"},
            params={"a": 1.0, "c": 1.0, "e": 1.0},
        )
        model = CTMCVectorField2D(
            n_atom_types=2,
            canonical_feat_order=["a", "c", "e"],
            interpolant_scheduler=scheduler,
            n_charges=6,
            n_bond_types=4,
            n_hidden=16,
            n_hidden_edge_feats=8,
            n_recycles=1,
            n_molecule_updates=1,
            convs_per_update=1,
        )
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=3)
        graph = dgl.batch([graph])
        graph.ndata["a_t"] = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        graph.ndata["c_t"] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        graph.edata["e_t"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )

        outputs = model(
            graph,
            t=torch.tensor([0.5], dtype=torch.float32),
            node_batch_idx=torch.zeros(3, dtype=torch.long),
            edge_batch_idx=torch.zeros(2, dtype=torch.long),
            upper_edge_mask=torch.tensor([True, False]),
        )

        self.assertEqual(outputs["a"].shape, (3, 2))
        self.assertEqual(outputs["c"].shape, (3, 6))
        self.assertEqual(outputs["e"].shape, (1, 4))

    def test_vector_field_node_predictions_change_when_edge_states_change(self):
        torch.manual_seed(0)
        scheduler = InterpolantScheduler(
            canonical_feat_order=["a", "c", "e"],
            schedule_type={"a": "linear", "c": "linear", "e": "linear"},
            params={"a": 1.0, "c": 1.0, "e": 1.0},
        )
        model = CTMCVectorField2D(
            n_atom_types=2,
            canonical_feat_order=["a", "c", "e"],
            interpolant_scheduler=scheduler,
            n_charges=6,
            n_bond_types=4,
            n_hidden=16,
            n_hidden_edge_feats=8,
            n_recycles=1,
            n_molecule_updates=1,
            convs_per_update=1,
        )
        model.eval()

        no_bond_graph = _make_two_atom_batch(edge_state_idx=0)
        triple_bond_graph = _make_two_atom_batch(edge_state_idx=3)

        t = torch.tensor([0.5], dtype=torch.float32)
        node_batch_idx = torch.zeros(2, dtype=torch.long)
        edge_batch_idx = torch.zeros(2, dtype=torch.long)
        upper_edge_mask = torch.tensor([True, False])

        no_bond_outputs = model(
            no_bond_graph,
            t=t,
            node_batch_idx=node_batch_idx,
            edge_batch_idx=edge_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )
        triple_bond_outputs = model(
            triple_bond_graph,
            t=t,
            node_batch_idx=node_batch_idx,
            edge_batch_idx=edge_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )

        self.assertFalse(torch.allclose(no_bond_outputs["a"], triple_bond_outputs["a"]))
        self.assertFalse(torch.allclose(no_bond_outputs["c"], triple_bond_outputs["c"]))

    def test_loss_fns_do_not_weight_atom_or_bond_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            _write_training_artifacts(processed_dir)

            config = _minimal_config(processed_dir)
            config["mol_fm"]["weight_ae"] = True

            model = model_from_config(config)
            model.configure_loss_fns(device=torch.device("cpu"))

            self.assertIsNone(model.loss_fn_dict["a"].weight)
            self.assertIsNone(model.loss_fn_dict["e"].weight)

    def test_vector_field_accepts_random_node_inputs(self):
        scheduler = InterpolantScheduler(
            canonical_feat_order=["a", "c", "e"],
            schedule_type={"a": "linear", "c": "linear", "e": "linear"},
            params={"a": 1.0, "c": 1.0, "e": 1.0},
        )
        model = CTMCVectorField2D(
            n_atom_types=2,
            canonical_feat_order=["a", "c", "e"],
            interpolant_scheduler=scheduler,
            n_charges=6,
            n_bond_types=4,
            n_hidden=16,
            n_hidden_edge_feats=8,
            n_recycles=1,
            n_molecule_updates=1,
            convs_per_update=1,
            n_random_node_feats=4,
        )
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        graph = dgl.batch([graph])
        graph.ndata["a_t"] = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        graph.ndata["c_t"] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        graph.ndata["z_t"] = torch.randn(2, 4)
        graph.edata["e_t"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )

        outputs = model(
            graph,
            t=torch.tensor([0.5], dtype=torch.float32),
            node_batch_idx=torch.zeros(2, dtype=torch.long),
            edge_batch_idx=torch.zeros(2, dtype=torch.long),
            upper_edge_mask=torch.tensor([True, False]),
        )

        self.assertEqual(model.random_node_embedding[0].in_features, 4)
        self.assertEqual(outputs["a"].shape, (2, 2))
        self.assertEqual(outputs["c"].shape, (2, 6))
        self.assertEqual(outputs["e"].shape, (1, 4))

    def test_sample_prior_adds_random_node_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            _write_training_artifacts(processed_dir)

            model = FlowMol(
                atom_type_map=["C", "H"],
                n_atoms_hist_file=processed_dir / "train_data_n_atoms_histogram.pt",
                marginal_dists_file=processed_dir / "train_data_marginal_dists.pt",
                prior_config={
                    "a": {"align": False, "kwargs": {}},
                    "c": {"align": False, "kwargs": {}},
                    "e": {"align": False, "kwargs": {}},
                },
                vector_field_config={
                    "n_hidden": 16,
                    "n_hidden_edge_feats": 8,
                    "n_recycles": 1,
                    "n_molecule_updates": 1,
                    "convs_per_update": 1,
                    "n_random_node_feats": 4,
                },
                interpolant_scheduler_config={
                    "schedule_type": {"a": "linear", "c": "linear", "e": "linear"},
                    "params": {"a": 1.0, "c": 1.0, "e": 1.0},
                },
                lr_scheduler_config={
                    "base_lr": 1e-4,
                    "restart_interval": 1,
                    "restart_type": "linear",
                    "warmup_length": 0.0,
                },
            )
            graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
            graph = dgl.batch([graph])
            graph = model.sample_prior(
                graph,
                node_batch_idx=torch.zeros(2, dtype=torch.long),
                upper_edge_mask=torch.tensor([True, False]),
            )

            self.assertIn("z_t", graph.ndata)
            self.assertEqual(graph.ndata["z_t"].shape, (2, 4))

    def test_test_script_loads_legacy_checkpoints_non_strict(self):
        source = (REPO_ROOT / "test.py").read_text()
        self.assertIn("strict=False", source)
