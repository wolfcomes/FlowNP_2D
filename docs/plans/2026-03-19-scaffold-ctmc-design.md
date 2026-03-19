# Scaffold CTMC Design

**Date:** 2026-03-19

**Goal:** Add Bemis-Murcko scaffold information to the 2D FlowNP pipeline as jointly generated categorical graph variables for nodes and edges.

## Context

The current 2D pipeline only models atom types, charges, and bond types:

- node features: `a`, `c`
- edge features: `e`

This leaves scaffold structure implicit. Based on the current generation behavior, the main weakness is not only local chemistry validity but also poor alignment with training-set scaffold and fingerprint distributions, especially on `COCONUT`.

Simply concatenating ground-truth scaffold labels as training-time inputs would create train/inference mismatch, because those labels are unavailable during generation. The scaffold signal therefore needs to be modeled as part of the generative state itself.

## Scope

This change covers:

- preprocessing scaffold atom and bond labels for `QM9` and `COCONUT`
- storing scaffold labels in processed dataset artifacts
- loading scaffold labels into `DGLGraph` objects
- modeling scaffold node and edge states in `FlowMol` and `CTMCVectorField2D`
- training and sampling scaffold labels with the same CTMC machinery used for `a`, `c`, and `e`

This change does not yet implement:

- backbone-first generation
- scaffold-conditioned decoration
- motif-level or hierarchical generation

## Representation

Two new categorical features will be added:

- node scaffold flag: `s`
  - `0`: non-scaffold atom
  - `1`: scaffold atom
- edge scaffold flag: `se`
  - `0`: non-scaffold edge
  - `1`: scaffold edge

Both features will be treated as categorical CTMC variables with a mask state during noising and sampling.

## Data Design

Preprocessing will compute scaffold labels from the final RDKit molecule actually used for featurization:

1. sanitize
2. kekulize if running in kekulized mode
3. remove hydrogens if the atom map excludes hydrogen
4. compute Bemis-Murcko scaffold on that molecule

This ordering matters because it keeps atom and bond indices aligned with the graph representation used downstream.

Processed split files will gain:

- `scaffold_atom_mask`
- `scaffold_bond_mask`

`scaffold_bond_mask` will align with the existing `bond_idxs` array for real bonds only.

During dataset loading:

- `scaffold_atom_mask` becomes `g.ndata["s_1_true"]`
- `scaffold_bond_mask` is lifted into the full pairwise edge representation and becomes `g.edata["se_1_true"]`

Non-bonded atom pairs will always have `se = 0`.

## Model Design

`FlowMol` feature sets will expand to:

- `canonical_feat_order = ["a", "c", "s", "e", "se"]`
- `node_feats = ["a", "c", "s"]`
- `edge_feats = ["e", "se"]`

`CTMCVectorField2D` will:

- include `s` and `se` in `n_cat_feats`
- include mask indices for `s` and `se`
- noise and sample `s` and `se` through the same CTMC path logic
- embed `s_t` into node inputs and `se_t` into edge inputs
- predict dedicated logits for `s` and `se`

The embedding dimensions should be computed from active feature sets rather than hard-coded from `a/c/e` only.

## Training Design

`FlowMol.forward()` will construct losses for:

- `a`
- `c`
- `s`
- `e`
- `se`

Initial recommended weights:

- `a: 0.4`
- `c: 1.0`
- `s: 0.5`
- `e: 2.0`
- `se: 1.0`

These weights bias learning toward edges while still making scaffold edge supervision explicit.

## Sampling Design

At sampling time:

- `s_0` and `se_0` are sampled from the same masked CTMC prior style already used for `a/c/e`
- `integrate()` and `step()` update `s_t` and `se_t` alongside the other categorical states

This preserves train/sample consistency.

## Compatibility

- Existing processed datasets must be regenerated because they do not contain scaffold masks.
- Old checkpoints can only be warm-started with `strict=False`; new scaffold heads will initialize randomly.
- Existing analysis code should remain usable because molecule building still depends on `a/c/e`, not `s/se`.

## Risks

- `se` is highly imbalanced because most atom pairs are not scaffold bonds.
- Many `QM9` molecules have empty Murcko scaffolds, making the scaffold signal weak on that dataset.
- The main correctness risk is mapping real-bond scaffold labels onto the doubled pairwise edge representation.

## Verification

The change is complete only if:

- preprocessing stores scaffold masks for a known ring-containing molecule
- chain molecules can still featurize with all-zero scaffold masks
- dataset graphs expose `s_1_true`, `se_1_true`, `s_0`, and `se_0`
- a forward pass produces losses for all five categorical features
- sampling prior and CTMC stepping handle `s` and `se` without special-casing failures
