"""
=============================================================================
PREPARE DATASETS — Convert downloaded data to PyTorch format
=============================================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prepare_datasets")


# ═══════════════════════════════════════════════════════════════════════
# CICY DATASET
# ═══════════════════════════════════════════════════════════════════════

class CICYDataset(Dataset):
    """
    PyTorch Dataset for Calabi-Yau configurations.
    
    Each sample contains:
      - config_matrix: (max_rows, max_cols) tensor
      - hodge_numbers: (2,) tensor [h^{1,1}, h^{2,1}]
      - euler_char: scalar tensor
      - is_real: boolean flag
    """
    
    def __init__(
        self,
        data_file: str,
        max_rows: int = 8,
        max_cols: int = 5,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        with open(data_file) as f:
            self.data = json.load(f)
        
        self.max_rows = max_rows
        self.max_cols = max_cols
        
        # Split data
        n = len(self.data)
        indices = list(range(n))
        
        # Deterministic shuffle
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]
        
        logger.info(f"CICYDataset ({split}): {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def _config_to_tensor(self, config: List[List[int]]) -> torch.Tensor:
        """Convert configuration matrix to padded tensor."""
        tensor = torch.zeros(self.max_rows, self.max_cols)
        
        for i, row in enumerate(config[:self.max_rows]):
            for j, val in enumerate(row[:self.max_cols]):
                tensor[i, j] = float(val)
        
        return tensor
    
    def __getitem__(self, idx):
        item = self.data[self.indices[idx]]
        
        config_tensor = self._config_to_tensor(item["config_matrix"])
        hodge = torch.tensor([item["h11"], item["h21"]], dtype=torch.float32)
        euler = torch.tensor(item["euler_char"], dtype=torch.float32)
        is_real = torch.tensor(float(item.get("is_real", False)))
        
        return {
            "config_matrix": config_tensor,
            "hodge_numbers": hodge,
            "euler_char": euler,
            "is_real": is_real,
        }


# ═══════════════════════════════════════════════════════════════════════
# PDB DATASET
# ═══════════════════════════════════════════════════════════════════════

class PDBDataset(Dataset):
    """
    PyTorch Dataset for protein structures.
    
    Each sample contains:
      - sequence: (L,) long tensor of amino acid indices
      - ca_coords: (L, 3) float tensor of Cα coordinates
      - length: scalar tensor
      - is_real: boolean flag
    """
    
    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    
    def __init__(
        self,
        data_file: str,
        max_length: int = 512,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        with open(data_file) as f:
            self.data = json.load(f)
        
        self.max_length = max_length
        
        # Split data
        n = len(self.data)
        indices = list(range(n))
        
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]
        
        logger.info(f"PDBDataset ({split}): {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def _sequence_to_tensor(self, seq: str) -> torch.Tensor:
        """Convert amino acid sequence to index tensor."""
        indices = [self.AA_TO_IDX.get(aa, 20) for aa in seq]  # 20 = unknown
        return torch.tensor(indices, dtype=torch.long)
    
    def _coords_to_tensor(self, coords: List[List[float]]) -> torch.Tensor:
        """Convert coordinates to tensor."""
        return torch.tensor(coords, dtype=torch.float32)
    
    def __getitem__(self, idx):
        item = self.data[self.indices[idx]]
        
        sequence = item["sequence"][:self.max_length]
        coords = item["ca_coords"][:self.max_length]
        
        seq_tensor = self._sequence_to_tensor(sequence)
        coord_tensor = self._coords_to_tensor(coords)
        
        # Pad if necessary
        L = len(sequence)
        if L < self.max_length:
            pad_len = self.max_length - L
            seq_tensor = torch.cat([
                seq_tensor,
                torch.full((pad_len,), 20, dtype=torch.long)
            ])
            coord_tensor = torch.cat([
                coord_tensor,
                torch.zeros(pad_len, 3)
            ])
        
        return {
            "sequence": seq_tensor,
            "ca_coords": coord_tensor,
            "length": torch.tensor(L, dtype=torch.long),
            "is_real": torch.tensor(float(item.get("is_real", False))),
        }


# ═══════════════════════════════════════════════════════════════════════
# DATALOADER BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_cicy_loaders(
    data_file: str,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for CICY data."""
    train_ds = CICYDataset(data_file, split="train")
    val_ds = CICYDataset(data_file, split="val")
    test_ds = CICYDataset(data_file, split="test")
    
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_dl, val_dl, test_dl


def build_pdb_loaders(
    data_file: str,
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for PDB data."""
    train_ds = PDBDataset(data_file, split="train")
    val_ds = PDBDataset(data_file, split="val")
    test_ds = PDBDataset(data_file, split="test")
    
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_dl, val_dl, test_dl


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("=" * 72)
    print(" Preparing Datasets")
    print("=" * 72)
    
    # CICY
    cicy_file = data_dir / "cicy" / "cicy_data.json"
    if cicy_file.exists():
        print("\n[D1] CICY Dataset")
        train_dl, val_dl, test_dl = build_cicy_loaders(str(cicy_file), args.batch_size)
        print(f"    Train: {len(train_dl.dataset)} samples")
        print(f"    Val:   {len(val_dl.dataset)} samples")
        print(f"    Test:  {len(test_dl.dataset)} samples")
        
        # Test one batch
        batch = next(iter(train_dl))
        print(f"    Batch shapes:")
        print(f"      config_matrix: {tuple(batch['config_matrix'].shape)}")
        print(f"      hodge_numbers: {tuple(batch['hodge_numbers'].shape)}")
    else:
        print(f"\n[D1] CICY data not found at {cicy_file}")
        print("     Run: python download_real_data.py --cicy-only")
    
    # PDB
    pdb_file = data_dir / "pdb" / "pdb_data.json"
    if pdb_file.exists():
        print("\n[D2] PDB Dataset")
        train_dl, val_dl, test_dl = build_pdb_loaders(str(pdb_file), args.batch_size)
        print(f"    Train: {len(train_dl.dataset)} samples")
        print(f"    Val:   {len(val_dl.dataset)} samples")
        print(f"    Test:  {len(test_dl.dataset)} samples")
        
        # Test one batch
        batch = next(iter(train_dl))
        print(f"    Batch shapes:")
        print(f"      sequence:  {tuple(batch['sequence'].shape)}")
        print(f"      ca_coords: {tuple(batch['ca_coords'].shape)}")
    else:
        print(f"\n[D2] PDB data not found at {pdb_file}")
        print("     Run: python download_real_data.py --pdb-only")
    
    print("\n" + "=" * 72)
    print(" Dataset preparation complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()
