# FlowNP_2D Pure 2D Design

**Date:** 2026-03-12

**Goal:** Remove pocket-related and 3D-related runtime paths from `FlowNP_2D`, and make the project support end-to-end 2D preprocessing and 2D model training for `QM9` and `COCONUT`.

## Context

The current repository is named `FlowNP_2D`, but the active training path is still mixed with legacy assumptions:

- 2D README and runtime behavior do not fully match.
- `pocket/crossdock` code still exists in the main import graph.
- configs still contain `x` and other 3D hyperparameters.
- preprocessing outputs do not fully match what training expects.
- `process_qm9.py` contains an invalid import path.

The result is that the project is not yet a clean, runnable 2D pipeline.

## Scope

This change covers:

- 2D preprocessing for `QM9`
- 2D preprocessing for `COCONUT`
- 2D-only dataset loading
- 2D-only model loading and training
- 2D-only sampling/export

This change does not preserve active `pocket/crossdock` training support in the main path.

## Architecture

The repository will be converged onto one active path:

`process_*.py -> MoleculeDataset -> MoleculeDataModule -> FlowMol -> CTMCVectorField2D -> train.py/test.py`

All mainline runtime dependencies on these concepts will be removed:

- pocket graphs
- crossdock-specific training path
- `x` as a trained feature
- 3D priors
- GVP-based geometry modeling
- RBF distance features
- dynamic graph reconstruction based on coordinates

RDKit-generated 2D coordinates will remain only as an output-time visualization/export detail.

## Data Design

Preprocessing scripts will emit the filenames the trainer already expects:

- `train_data_processed.pt`
- `val_data_processed.pt`
- `test_data_processed.pt`
- `train_data_n_atoms_histogram.pt`
- `train_data_marginal_dists.pt`

Processed molecule records will be centered on categorical chemistry features:

- `smiles`
- `atom_types`
- `atom_charges`
- `bond_types`
- `bond_idxs`
- `node_idx_array`
- `edge_idx_array`

`positions` may be retained only for compatibility, but it will not be required by training.

`QM9` preprocessing will be fixed to:

- import the local `MoleculeFeaturizer`
- write the correct processed filenames
- save marginal distributions in the structure expected by training

`COCONUT` preprocessing will be fixed to:

- stop filtering molecules by 3D-ness
- write the correct processed filenames
- save the same training-compatible distribution files

## Model Design

`FlowMol` becomes the only active model class in the main path.

- `canonical_feat_order = ['a', 'c', 'e']`
- no `x` loss
- no `x` prior
- no coordinate diffusion
- no pocket-conditioned subclass in the active path

`CTMCVectorField2D` becomes the only active vector field implementation in the main path.

Accepted config stays limited to 2D-relevant parameters such as:

- `n_hidden`
- `n_hidden_edge_feats`
- `n_recycles`
- `n_molecule_updates`
- `convs_per_update`
- `separate_mol_updaters`
- categorical temperature / forward schedules

Legacy 3D parameters will be removed from active configs and ignored by the runtime path:

- `rbf_dmax`
- `rbf_dim`
- `n_vec_channels`
- `n_hidden_scalars`
- `n_message_gvps`
- `n_update_gvps`
- `enable_dynamic_graph`
- `knn_connectivity`
- `sde`

## Training Design

`src/model_utils/load.py` will only construct:

- `FlowMol`
- `MoleculeDataModule`

`train.py` will remain the single training entrypoint and will run only the pure 2D path.

Configs for `QM9` and `COCONUT` will be rewritten as pure 2D configs:

- remove `x` from scheduler params
- remove `x` from priors
- remove `x` from loss weights
- replace legacy vector-field keys with 2D keys

`crossdock_ctmc.yaml` will be removed from the active workflow. If retained in tree, it must be clearly marked deprecated and unused.

## Sampling And Export

`test.py` will continue to sample from `FlowMol`.

When sampled graphs do not contain coordinates, molecule export will use RDKit 2D coordinate generation in `src/analysis/molecule_builder.py`.

This keeps SDF export working without preserving any 3D generation path.

## Verification

The implementation is complete only if all of the following hold:

- `process_qm9.py` produces training-compatible processed artifacts
- `process_coconut.py` produces training-compatible processed artifacts
- `python train.py --config configs/qm9_ctmc.yaml --debug` completes at least one train and validation step
- `python train.py --config configs/coconut_ctmc.yaml --debug` completes at least one train and validation step, assuming processed data exists
- `python test.py --checkpoint <ckpt> --n_mols 4` exports sampled molecules without requiring `x`
- no active import chain required for training depends on pocket-only classes

## Risks

- Some dead 3D helper code may remain in non-mainline files unless explicitly removed.
- Existing checkpoints are not expected to be compatible with the final 2D-only configs.
- Analysis utilities that assume coordinates may still need local cleanup if they are later brought back into active use.

## Repository Note

No git repository is available at `/data/home/zhangzhiyong` or `/data/home/zhangzhiyong/FlowNP_2D`, so this design document cannot be committed in the current environment.
