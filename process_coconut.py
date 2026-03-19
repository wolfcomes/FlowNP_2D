import argparse
import atexit
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import List
import warnings

import torch
import numpy as np
import tqdm
import yaml
from rdkit import Chem
from rdkit import RDLogger
import pandas as pd

# 抑制RDKit警告
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)

from src.data_processing.geom import MoleculeFeaturizer
from src.utils.dataset_stats import compute_p_c_given_a


def resolve_coconut_sdf_file(raw_dir: Path) -> Path:
    preferred = raw_dir / "coconut_edge.sdf"
    if preferred.exists():
        return preferred

    sdf_candidates = sorted(raw_dir.glob("*.sdf"))
    if len(sdf_candidates) == 1:
        return sdf_candidates[0]

    for candidate in sdf_candidates:
        if "coconut" in candidate.name.lower():
            return candidate

    if sdf_candidates:
        return sdf_candidates[0]

    raise FileNotFoundError(f"No SDF file found in {raw_dir}")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process COCONUT dataset geometry')
    p.add_argument('--config', type=Path, help='config file path')
    p.add_argument('--chunk_size', type=int, default=1000, help='number of molecules to process at once')
    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')
    p.add_argument('--max_molecules', type=int, default=None, help='maximum number of molecules to process (for debugging)')
    p.add_argument('--max_atoms', type=int, default=1000, help='maximum number of atoms per molecule')
    p.add_argument('--explicit_aromaticity', action='store_true', help='use explicit aromaticity instead of kekulization')
    
    args = p.parse_args()
    return args

def process_split(split_indices, split_name, args, dataset_config, all_molecules, all_smiles):
    """Process a data split (train/val/test) and save the processed data."""
    
    # get processed data directory and create it if it doesn't exist
    output_dir = Path(config['dataset']['processed_data_dir'])
    output_dir.mkdir(exist_ok=True) 

    # Filter molecules for this split
    split_molecules = [all_molecules[i] for i in split_indices]
    split_smiles = [all_smiles[i] for i in split_indices]

    print(f"Processing {split_name} split with {len(split_molecules)} molecules")
    print(f"Using {'explicit aromaticity' if args.explicit_aromaticity else 'kekulization'} for bond representation")

    successful_slices = []
    all_positions = []
    all_atom_types = []
    all_atom_charges = []
    all_bond_types = []
    all_bond_idxs = []
    all_scaffold_atom_masks = []
    all_scaffold_bond_masks = []
    n_bond_orders = 5 if args.explicit_aromaticity else 4
    all_bond_order_counts = torch.zeros(n_bond_orders, dtype=torch.int64)

    mol_featurizer = MoleculeFeaturizer(config['dataset']['atom_map'], n_cpus=args.n_cpus, max_atoms=args.max_atoms, explicit_aromaticity=args.explicit_aromaticity)


    # Create chunks of molecules for processing
    chunk_iterator = chunks(split_molecules, args.chunk_size)
    n_chunks = len(split_molecules) // args.chunk_size + 1

    tqdm_iterator = tqdm.tqdm(chunk_iterator, desc=f'Featurizing {split_name} molecules', total=n_chunks)
    failed_molecules_bar = tqdm.tqdm(desc=f"Failed {split_name} Molecules", unit="molecules")
    total_molecules_bar = tqdm.tqdm(desc=f"Total {split_name} Molecules", unit="molecules", total=len(split_molecules))

    failed_molecules = 0
    chunk_start_idx = 0
    for molecule_chunk in tqdm_iterator:

        (
            positions,
            atom_types,
            atom_charges,
            bond_types,
            bond_idxs,
            scaffold_atom_masks,
            scaffold_bond_masks,
            failed_idx,
            bond_order_counts,
        ) = mol_featurizer.featurize_molecules(molecule_chunk)

        failed_molecules += len(failed_idx)
        failed_molecules_bar.update(len(failed_idx))
        total_molecules_bar.update(len(molecule_chunk))

        all_positions.extend(positions)
        all_atom_types.extend(atom_types)
        all_atom_charges.extend(atom_charges)
        all_bond_types.extend(bond_types)
        all_bond_idxs.extend(bond_idxs)
        all_scaffold_atom_masks.extend(scaffold_atom_masks)
        all_scaffold_bond_masks.extend(scaffold_bond_masks)
        all_bond_order_counts += bond_order_counts

        successful_indices_in_chunk = [i for i in range(len(molecule_chunk)) if i not in failed_idx]
        
        # 获取成功的SMILES
        for idx_in_chunk in successful_indices_in_chunk:
            global_idx = chunk_start_idx + idx_in_chunk
            successful_slices.append(global_idx)
        chunk_start_idx += len(molecule_chunk)

    split_smiles = [split_smiles[i] for i in successful_slices]

    # get number of atoms in every data point
    n_atoms_list = [ x.shape[0] for x in all_positions ]
    n_bonds_list = [ x.shape[0] for x in all_bond_idxs ]

    # convert n_atoms_list and n_bonds_list to tensors
    n_atoms_list = torch.tensor(n_atoms_list)
    n_bonds_list = torch.tensor(n_bonds_list)

    # concatenate all arrays
    all_positions = torch.concatenate(all_positions, dim=0)
    all_atom_types = torch.concatenate(all_atom_types, dim=0)
    all_atom_charges = torch.concatenate(all_atom_charges, dim=0)
    all_bond_types = torch.concatenate(all_bond_types, dim=0)
    all_bond_idxs = torch.concatenate(all_bond_idxs, dim=0)
    all_scaffold_atom_masks = torch.concatenate(all_scaffold_atom_masks, dim=0)
    all_scaffold_bond_masks = torch.concatenate(all_scaffold_bond_masks, dim=0)

    # create an array of indices to keep track of the start_idx and end_idx of each molecule's node features
    node_idx_array = torch.zeros((len(n_atoms_list), 2), dtype=torch.int32)
    node_idx_array[:, 1] = torch.cumsum(n_atoms_list, dim=0)
    node_idx_array[1:, 0] = node_idx_array[:-1, 1]

    # create an array of indices to keep track of the start_idx and end_idx of each molecule's edge features
    edge_idx_array = torch.zeros((len(n_bonds_list), 2), dtype=torch.int32)
    edge_idx_array[:, 1] = torch.cumsum(n_bonds_list, dim=0)
    edge_idx_array[1:, 0] = edge_idx_array[:-1, 1]

    all_positions = all_positions.type(torch.float32)
    all_atom_charges = all_atom_charges.type(torch.int32)
    all_bond_idxs = all_bond_idxs.type(torch.int32)

    # create a dictionary to store all the data
    data_dict = {
        'smiles': split_smiles,
        'positions': all_positions,
        'atom_types': all_atom_types,
        'atom_charges': all_atom_charges,
        'bond_types': all_bond_types,
        'bond_idxs': all_bond_idxs,
        'scaffold_atom_mask': all_scaffold_atom_masks,
        'scaffold_bond_mask': all_scaffold_bond_masks,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
    }

    # determine output file name and save the data_dict there
    output_file = output_dir / f'{split_name}_processed.pt'
    torch.save(data_dict, output_file)

    # create histogram of number of atoms
    n_atoms, counts = torch.unique(n_atoms_list, return_counts=True)
    histogram_file = output_dir / f'{split_name}_n_atoms_histogram.pt'
    torch.save((n_atoms, counts), histogram_file)

    # 计算原子类型的边际分布, p(a)
    p_a = all_atom_types.sum(dim=0)
    p_a = p_a / p_a.sum()

    # 计算键类型的边际分布, p(e)
    p_e = all_bond_order_counts / all_bond_order_counts.sum()

    # 计算电荷的边际分布, p(c)
    charge_vals, charge_counts = torch.unique(all_atom_charges, return_counts=True)
    p_c = torch.zeros(6, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        p_c[c_val+2] = c_count
    p_c = p_c / p_c.sum()

    # 计算给定原子类型的电荷的条件分布, p(c|a)
    p_c_given_a = compute_p_c_given_a(all_atom_charges, all_atom_types, dataset_config['atom_map'])

    marginal_dists_file = output_dir / f'{split_name}_marginal_dists.pt'
    torch.save((p_a, p_c, p_e, p_c_given_a), marginal_dists_file)

    # write smiles to its own file
    smiles_file = output_dir / f'{split_name}_smiles.pkl'
    with open(smiles_file, 'wb') as f:
        pickle.dump(split_smiles, f)

    print(f"Completed processing {split_name} split: {len(split_smiles)} molecules, {failed_molecules} failed")

def load_coconut_molecules(sdf_file, max_molecules=None):
    """Load molecules from COCONUT SDF file."""
    print(f"Loading molecules from {sdf_file}")
    
    all_molecules = []
    all_smiles = []
    failed_count = 0
    stereo_warnings = 0
    # 使用更宽松的分子读取参数
    mol_reader = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
    
    with tqdm.tqdm(desc="Loading molecules") as pbar:
        for mol_idx, mol in enumerate(mol_reader):
            pbar.update(1)
            
            if mol is None:
                failed_count += 1
                continue

            try:
                # 手动进行更宽松的sanitization
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except:
                    # 如果完整的sanitization失败，尝试基本的sanitization
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                    except:
                        failed_count += 1
                        continue
                
                # 生成SMILES，先尝试isomeric，如果失败则使用canonical
                smiles = None
                try:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                except:
                    stereo_warnings += 1
                    try:
                        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                    except:
                        failed_count += 1
                        continue
                
                if smiles is None:
                    failed_count += 1
                    continue

                # mol = Chem.AddHs(mol)

                all_molecules.append(mol)
                all_smiles.append(smiles)
                
                # Break if we've reached the maximum number of molecules
                if max_molecules and len(all_molecules) >= max_molecules:
                    break
                    
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {e}")
                failed_count += 1
                continue

    print(f"Loaded {len(all_molecules)} molecules successfully")
    print(f"Failed molecules: {failed_count}")
    print(f"Stereochemistry issues resolved: {stereo_warnings}")
    return all_molecules, all_smiles

if __name__ == "__main__":

    # parse command-line args
    args = parse_args()

    # load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = config['dataset']
    if dataset_config['dataset_name'] != 'coconut':
        raise ValueError('This script only works with the coconut dataset')

    # get COCONUT SDF file path
    raw_dir = Path(dataset_config['raw_data_dir']) 
    sdf_file = resolve_coconut_sdf_file(raw_dir)
    

    print(f"Using SDF file: {sdf_file}")

    # Load all molecules from COCONUT
    all_molecules, all_smiles = load_coconut_molecules(sdf_file, max_molecules=args.max_molecules)
    
    n_samples = len(all_molecules)
    
    # Define splits - using standard proportions for large datasets
    n_train = int(0.8 * n_samples)  # 80% for training
    n_val = int(0.1 * n_samples)   # 10% for validation  
    n_test = n_samples - (n_train + n_val)  # 10% for testing

    # print the number of samples in each split
    print(f"Total samples: {n_samples}")
    print(f"Number of samples in train split: {n_train}")
    print(f"Number of samples in val split: {n_val}")
    print(f"Number of samples in test split: {n_test}")

    # Create random indices for splitting
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Process each split
    split_data = [
        (train_indices, 'train_data'),
        (val_indices, 'val_data'),
        (test_indices, 'test_data')
    ]
    
    for split_indices, split_name in split_data:
        process_split(split_indices, split_name, args, dataset_config, all_molecules, all_smiles)

    print("COCONUT dataset processing completed!")
