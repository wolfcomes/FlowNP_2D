# Random Node Init And Unweighted Losses Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add random node input features to the 2D CTMC model during both training and sampling, and remove atom/bond class weighting from categorical losses.

**Architecture:** Keep the existing CTMC categorical path for `a/c/e` unchanged, but inject an unsupervised Gaussian node feature `z` into the node encoder so generation can break symmetry from a fully masked complete graph. Loss computation remains endpoint-classification-only, with plain cross-entropy for all categorical outputs.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, DGL, unittest, RDKit

---

### Task 1: Lock The New Contract In Tests

**Files:**
- Modify: `tests/test_2d_contract.py`
- Test: `tests/test_2d_contract.py`

**Step 1: Write the failing test**

Add tests that assert:
- `FlowMol.configure_loss_fns()` does not attach class weights to `a` or `e`
- `CTMCVectorField2D` consumes a random node feature `z_t` without changing output shapes
- `FlowMol.sample_prior()` creates a random node feature for inference graphs

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_2d_contract.py -k "loss_fns_do_not_weight_atom_or_bond_classes or vector_field_accepts_random_node_inputs or sample_prior_adds_random_node_features" -v`

Expected: FAIL because the current model does not expose the new behavior.

**Step 3: Write minimal implementation**

Update the model to pass only those behaviors.

**Step 4: Run test to verify it passes**

Run the same `pytest` command and confirm PASS.

### Task 2: Wire Random Node Inputs Through Training And Sampling

**Files:**
- Modify: `src/models/flowmol.py`
- Modify: `src/models/vector_field_2d.py`
- Test: `tests/test_2d_contract.py`

**Step 1: Add random-node feature plumbing**

- Add a configurable random-node input dimension to the 2D vector field
- Sample Gaussian `z` features for nodes during training forward passes and sampling prior creation
- Feed `z` into the node embedding path only
- Do not include `z` in outputs, losses, or sampled molecule conversion

**Step 2: Re-run focused tests**

Run: `pytest tests/test_2d_contract.py -k "vector_field_accepts_random_node_inputs or sample_prior_adds_random_node_features" -v`

Expected: PASS

### Task 3: Remove Atom And Bond Class Weighting

**Files:**
- Modify: `src/models/flowmol.py`
- Test: `tests/test_2d_contract.py`

**Step 1: Simplify categorical loss construction**

- Remove the `weight_ae` weighting path from active loss construction for `a` and `e`
- Keep `ignore_index=-100`

**Step 2: Re-run focused tests**

Run: `pytest tests/test_2d_contract.py -k "loss_fns_do_not_weight_atom_or_bond_classes" -v`

Expected: PASS

### Task 4: Verify End-To-End Behavior

**Files:**
- Test: `tests/test_2d_contract.py`
- Test: `src/models/flowmol.py`
- Test: `src/models/vector_field_2d.py`

**Step 1: Run focused contract coverage**

Run: `pytest tests/test_2d_contract.py -k "loss_fns_do_not_weight_atom_or_bond_classes or vector_field_accepts_random_node_inputs or sample_prior_adds_random_node_features" -v`

Expected: PASS

**Step 2: Run a small runtime sanity check**

Run a short Python command that loads the QM9 checkpoint and verifies sampling still executes with the updated model code.
