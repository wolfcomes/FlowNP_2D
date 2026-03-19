# Scaffold CTMC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Bemis-Murcko scaffold node and edge labels to the 2D FlowNP pipeline as jointly generated CTMC categorical variables.

**Architecture:** Extend preprocessing to persist scaffold atom and bond masks, then thread those masks through dataset loading into two new categorical features: node scaffold `s` and edge scaffold `se`. Update `FlowMol` and `CTMCVectorField2D` so `s/se` use the same prior, noising, denoising, loss, and sampling pipeline as `a/c/e`.

**Tech Stack:** Python, PyTorch, DGL, RDKit, unittest

---

### Task 1: Add failing preprocessing tests for scaffold masks

**Files:**
- Modify: `tests/test_2d_contract.py`
- Modify: `src/data_processing/geom.py`

**Step 1: Write the failing test**

Add tests that:

- featurize a ring-containing molecule and assert scaffold masks exist and contain at least one positive atom/bond
- featurize a chain molecule and assert scaffold masks exist and are all zero

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_2d_contract.ContractTests.test_featurize_molecule_emits_scaffold_masks_for_ring_system tests.test_2d_contract.ContractTests.test_featurize_molecule_allows_empty_scaffold_masks_for_chain -q`

Expected: failure because `featurize_molecule()` does not yet return scaffold masks.

**Step 3: Write minimal implementation**

Implement scaffold extraction in `src/data_processing/geom.py` and return:

- `scaffold_atom_mask`
- `scaffold_bond_mask`

**Step 4: Run test to verify it passes**

Run the same `python -m unittest ... -q` command and expect success.

**Step 5: Commit**

```bash
git add tests/test_2d_contract.py src/data_processing/geom.py
git commit -m "feat: add scaffold masks to molecule featurization"
```

### Task 2: Add failing dataset tests for scaffold node and edge labels

**Files:**
- Modify: `tests/test_2d_contract.py`
- Modify: `src/data_processing/dataset.py`
- Modify: `process_qm9.py`
- Modify: `process_coconut.py`

**Step 1: Write the failing test**

Add a dataset contract test that builds a minimal processed split with scaffold mask tensors and asserts:

- `batch.ndata["s_1_true"]` exists
- `batch.edata["se_1_true"]` exists
- upper and lower edge scaffold labels match
- `s_0` and `se_0` priors exist

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_2d_contract.ContractTests.test_qm9_dataset_loads_scaffold_node_and_edge_features -q`

Expected: failure because processed scaffold fields are not loaded or converted.

**Step 3: Write minimal implementation**

Update preprocessing scripts to concatenate and save scaffold masks. Update `MoleculeDataset` to load scaffold masks, convert them to node and pairwise edge one-hot labels, and generate priors.

**Step 4: Run test to verify it passes**

Run the same dataset test command and expect success.

**Step 5: Commit**

```bash
git add tests/test_2d_contract.py process_qm9.py process_coconut.py src/data_processing/dataset.py
git commit -m "feat: load scaffold labels into processed datasets"
```

### Task 3: Add failing model tests for scaffold categorical features

**Files:**
- Modify: `tests/test_2d_contract.py`
- Modify: `src/models/flowmol.py`
- Modify: `src/models/vector_field_2d.py`
- Modify: `src/data_processing/priors.py`

**Step 1: Write the failing test**

Add model tests that assert:

- `model.canonical_feat_order == ["a", "c", "s", "e", "se"]`
- a forward pass returns losses for `a`, `c`, `s`, `e`, `se`
- `sample_prior()` adds `s_0` and `se_0`
- `sample_conditional_path()` and one vector-field pass consume and produce `s/se`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_2d_contract.ContractTests.test_model_tracks_scaffold_features_in_loss_and_prior -q`

Expected: failure because model feature dictionaries and logits only support `a/c/e`.

**Step 3: Write minimal implementation**

Extend `FlowMol`, `CTMCVectorField2D`, and edge prior setup to include `s` and `se`, with dedicated logits and generalized node/edge feature handling.

**Step 4: Run test to verify it passes**

Run the same model test command and expect success.

**Step 5: Commit**

```bash
git add tests/test_2d_contract.py src/models/flowmol.py src/models/vector_field_2d.py src/data_processing/priors.py
git commit -m "feat: model scaffold node and edge ctcm states"
```

### Task 4: Update configs and end-to-end contract coverage

**Files:**
- Modify: `configs/qm9_ctmc.yaml`
- Modify: `configs/coconut_ctmc.yaml`
- Modify: `tests/test_2d_contract.py`

**Step 1: Write the failing test**

Extend config contract tests to require:

- scheduler params include `s` and `se`
- prior config includes `s` and `se`
- total loss weights include `s` and `se`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_2d_contract.ContractTests.test_active_2d_configs_include_scaffold_feature_keys -q`

Expected: failure because configs only declare `a/c/e`.

**Step 3: Write minimal implementation**

Update both active 2D configs with scaffold scheduler, prior, and loss entries.

**Step 4: Run test to verify it passes**

Run the same config test command and expect success.

**Step 5: Commit**

```bash
git add tests/test_2d_contract.py configs/qm9_ctmc.yaml configs/coconut_ctmc.yaml
git commit -m "chore: configure scaffold feature training"
```

### Task 5: Verify integrated behavior

**Files:**
- No additional code changes expected

**Step 1: Run focused regression suite**

Run:

```bash
python -m unittest tests.test_2d_contract -q
```

Expected: all contract tests pass.

**Step 2: Run a minimal data/model smoke check**

Run:

```bash
python -m unittest \
  tests.test_2d_contract.ContractTests.test_qm9_config_loads_pure_2d_pipeline \
  tests.test_2d_contract.ContractTests.test_model_tracks_scaffold_features_in_loss_and_prior \
  -q
```

Expected: success, confirming the pipeline can construct data and model objects with scaffold features.

**Step 3: Commit**

```bash
git add -A
git commit -m "test: verify scaffold ctcm integration"
```
