"""
=============================================================================
TRAIN DDP — Multi-GPU Distributed Training
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

USAGE:
  # Single GPU
  python train_ddp.py --model cy --data ./data/cicy/cicy_data.json
  
  # Multi-GPU with torchrun
  torchrun --nproc_per_node=4 train_ddp.py --model cy --data ./data/cicy/cicy_data.json
  
  # Multi-node (advanced)
  torchrun --nnodes=2 --nproc_per_node=4 \
           --rdzv_id=100 --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:29400 \
           train_ddp.py --model cy --data ./data/cicy/cicy_data.json

SUPPORTED MODELS:
  --model cy       : SESICalabiYauSystem (CICY data)
  --model protein  : ProteinFoldingOrchestrator (PDB data)
  --model biomedical: SESIBiomedicalIntegration (combined)

=============================================================================
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s"
)
logger = logging.getLogger("train_ddp")


# ═══════════════════════════════════════════════════════════════════════
# DISTRIBUTED SETUP
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed(backend: str = "nccl") -> tuple:
    """
    Initialize distributed training.
    
    Returns:
        (device, rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend=backend)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        
        logger.info(f"Initialized: rank={rank}, world_size={world_size}, "
                   f"local_rank={local_rank}, device={device}")
        return device, rank, world_size, local_rank
    else:
        # Single GPU / CPU mode
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Single device mode: {device}")
        return device, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# ═══════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════

class DDPTrainer:
    """
    Distributed Data Parallel trainer.
    
    Supports:
      • Multi-GPU training via DDP
      • Mixed precision (AMP)
      • Gradient accumulation
      • Learning rate scheduling
      • Checkpointing
      • Logging and metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        
        # Wrap model with DDP
        self.model = model.to(device)
        if dist.is_available() and dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=cfg.get("find_unused_parameters", False),
                gradient_as_bucket_view=True,
            )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.get("lr", 3e-4),
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.get("epochs", 100),
            eta_min=cfg.get("min_lr", 1e-6),
        )
        
        # Mixed precision
        self.use_amp = cfg.get("use_amp", True) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.grad_accum_steps = cfg.get("grad_accum_steps", 1)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Output directory
        self.output_dir = Path(cfg.get("output_dir", "./checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _raw_model(self) -> nn.Module:
        """Get the raw model (unwrap DDP if needed)."""
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Set sampler epoch for proper shuffling
        if hasattr(self.train_loader, "sampler") and \
           isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self._compute_loss(batch)
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.get("grad_clip", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1
            
            # Logging
            if is_main_process() and batch_idx % self.cfg.get("log_interval", 50) == 0:
                avg_loss = total_loss / n_batches
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        return {"loss": total_loss / max(n_batches, 1)}
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Override this method for custom models.
        """
        raise NotImplementedError("Subclass must implement _compute_loss")
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.val_loader:
            batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self._compute_loss(batch)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Reduce across processes
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = reduce_mean(loss_tensor).item()
        
        return {"val_loss": avg_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if not is_main_process():
            return
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self._raw_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }
        
        path = self.output_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self._raw_model().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def fit(self, epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            if is_main_process():
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}"
                )
            
            # Checkpointing
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            
            if is_main_process():
                self.save_checkpoint(is_best=is_best)
        
        logger.info("Training complete!")


# ═══════════════════════════════════════════════════════════════════════
# MODEL-SPECIFIC TRAINERS
# ═══════════════════════════════════════════════════════════════════════

class CalabiYauTrainer(DDPTrainer):
    """Trainer for Calabi-Yau system."""
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(
            config_matrix=batch["config_matrix"],
            hodge_numbers=batch["hodge_numbers"],
        )
        
        # Euler characteristic prediction loss
        euler_pred = out.get("euler_pred", torch.zeros(1, device=self.device))
        euler_target = batch["euler_char"]
        
        loss = nn.functional.mse_loss(euler_pred, euler_target)
        
        return loss


class ProteinTrainer(DDPTrainer):
    """Trainer for protein folding system."""
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # This would call the protein folding model
        # For now, return a placeholder
        return torch.tensor(0.0, device=self.device, requires_grad=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--model", type=str, default="cy",
                        choices=["cy", "protein", "biomedical"],
                        help="Model to train")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    args = parser.parse_args()
    
    # Setup distributed
    device, rank, world_size, local_rank = setup_distributed()
    
    # Config
    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
        "use_amp": not args.no_amp,
        "grad_accum_steps": 1,
        "grad_clip": 1.0,
        "log_interval": 50,
    }
    
    if is_main_process():
        logger.info("=" * 72)
        logger.info(" SESI DDP Training")
        logger.info("=" * 72)
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
    
    # Load data
    from prepare_datasets import build_cicy_loaders, build_pdb_loaders
    
    if args.model == "cy":
        train_dl, val_dl, test_dl = build_cicy_loaders(
            args.data, batch_size=args.batch_size
        )
    else:
        train_dl, val_dl, test_dl = build_pdb_loaders(
            args.data, batch_size=args.batch_size
        )
    
    # Wrap with DistributedSampler
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(
            train_dl.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_dl = DataLoader(
            train_dl.dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
        )
    
    # Build model
    if args.model == "cy":
        # Import and build CY model
        # from SESI_CALABI_YAU_AGI_ONE_INTEGRATION import SESICalabiYauSystem, CalabiYauConfig
        # cy_cfg = CalabiYauConfig()
        # model = SESICalabiYauSystem(cy_cfg)
        # trainer = CalabiYauTrainer(model, train_dl, val_dl, cfg, device)
        logger.info("Calabi-Yau model training not fully implemented yet")
        logger.info("Please integrate with SESI_CALABI_YAU_AGI_ONE_INTEGRATION.py")
        return
    else:
        logger.info(f"Model {args.model} not implemented yet")
        return
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.fit(args.epochs)
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
