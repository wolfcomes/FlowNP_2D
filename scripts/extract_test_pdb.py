import os
import shutil
from pathlib import Path

def copy_test_pdbs(config_path, output_pocket_dir, output_ligand_dir, num_files=100):
    """
    Copy PDB files (both pockets and ligands) from the test dataset to new directories.
    
    Args:
        config_path (str): Path to the config file used in the original script
        output_pocket_dir (str): Path to the directory where pocket PDBs should be copied
        output_ligand_dir (str): Path to the directory where ligand PDBs should be copied
        num_files (int): Number of PDB files to copy (default: 100)
    """
    # Load the config file
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Get the base directory from config
    basedir = Path(config['dataset']['raw_data_dir'])
    datadir = basedir / 'crossdocked_pocket10'
    split_path = basedir / 'split_by_name.pt'
    
    # Load the test split
    import torch
    data_splits = torch.load(split_path)
    test_pairs = data_splits['test']
    
    # Create output directories if they don't exist
    output_pocket_dir = Path(output_pocket_dir)
    output_pocket_dir.mkdir(parents=True, exist_ok=True)
    
    output_ligand_dir = Path(output_ligand_dir)
    output_ligand_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the PDB files
    copied = 0
    for pocket_fn, ligand_fn in test_pairs:
        if copied >= num_files:
            break
            
        # Process pocket file
        pdb_file = datadir / pocket_fn
        if pdb_file.exists():
            # Create the output filename (using original filename)
            output_file = output_pocket_dir / pdb_file.name
            
            # Copy the file
            shutil.copy2(pdb_file, output_file)
            
            print(f"Copied pocket {pdb_file.name} to {output_pocket_dir}")
        
        # Process ligand file
        ligand_file = datadir / ligand_fn
        if ligand_file.exists():
            # Create the output filename (using original filename)
            output_ligand_file = output_ligand_dir / ligand_file.name
            
            # Copy the file
            shutil.copy2(ligand_file, output_ligand_file)
            
            print(f"Copied ligand {ligand_file.name} to {output_ligand_dir}")
        
        if pdb_file.exists() and ligand_file.exists():
            copied += 1
    
    print(f"\nSuccessfully copied {copied} pairs of PDB files")
    print(f"Pockets copied to: {output_pocket_dir}")
    print(f"Ligands copied to: {output_ligand_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Copy PDB files from test dataset')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to the config file used in the original script')
    parser.add_argument('--output_pocket_dir', type=str, required=True,
                       help='Path to the directory where pocket PDBs should be copied')
    parser.add_argument('--output_ligand_dir', type=str, required=True,
                       help='Path to the directory where ligand PDBs should be copied')
    parser.add_argument('--num_files', type=int, default=100,
                       help='Number of PDB files to copy (default: 100)')
    
    args = parser.parse_args()
    
    copy_test_pdbs(
        config_path=args.config,
        output_pocket_dir=args.output_pocket_dir,
        output_ligand_dir=args.output_ligand_dir,
        num_files=args.num_files
    )