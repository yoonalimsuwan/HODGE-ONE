"""
=============================================================================
structural_gno_hodge.py  —  Structural GNO-Hodge Operator (Production v1)
=============================================================================
Developer    : Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
Organization : MSPS NETWORK
ORCID        : 0009-0008-2374-0788
GitHub       : yoonalimsuwan
License      : MIT
Year         : 2026

AI Co-Developers (architecture, numerical methods, production hardening):
  - Claude  (Anthropic) — production refactor, EMA checkpointing,
                          multi-loss weighting, physics-informed losses,
                          LR scheduling, gradient monitoring, full docstrings
  - GPT     (OpenAI)    — early architecture exploration, message-passing
                          design, phase-field surrogate concept
  - Gemini  (Google)    — v2 unified discrete/continuous extension,
                          one-shot phase evolution framing

=============================================================================
Overview
--------
StructuralGNOHodge is a differentiable AI surrogate that learns to arrange
a set of 1-D particles on a manifold so that their differentiable period map
evaluates to a prescribed Hodge class vector.

The model is trained jointly with the DifferentiablePeriodComputer from
HODGE ONE via a multi-component loss:

    L_total = w_period  * L_period            # MSE to target Hodge vector
            + w_spacing * L_spacing           # particle spread regularizer
            + w_entropy * L_entropy           # cycle diversity (soft entropy)
            + w_smooth  * L_smooth            # positional smoothness

Training features
-----------------
  * AdamW optimizer with cosine annealing + linear warm-up
  * Exponential Moving Average (EMA) of model weights
  * Gradient clipping + per-step gradient norm monitoring
  * Automatic mixed precision (AMP) for CUDA
  * Checkpoint save / resume (state_dict + EMA + scheduler)
  * WandB-compatible scalar logging (dict return per step)
  * Full type-annotated API

Usage
-----
  python structural_gno_hodge.py --mode train --epochs 500 --device cuda
  python structural_gno_hodge.py --mode eval  --checkpoint best.pt
  python structural_gno_hodge.py --mode demo
=============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import hodge_one

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SGNO_HODGE")


# =============================================================================
# Utilities
# =============================================================================

def get_device(preferred: str = "cpu") -> torch.device:
    """Return best available device matching *preferred* ('cuda'|'mps'|'cpu')."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across Python / NumPy / PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SGNOHodgeConfig:
    """
    Central configuration for the StructuralGNOHodge model and trainer.

    Model hyperparameters
    ---------------------
    node_in_dim : int
        Dimension of per-particle input features.
        Default 3 = [position, local_spacing, local_density].
    period_dim : int
        Dimension of the target Hodge class / period vector.
    hidden_dim : int
        Width of all hidden layers.
    num_layers : int
        Number of FiLM-modulated convolution blocks.
    dropout : float
        Dropout probability inside each block.

    Loss weights
    ------------
    w_period  : weight for MSE(computed_periods, target_vector).
    w_spacing : weight for anti-collapse spacing regularizer.
    w_entropy : weight for soft-entropy diversity term.
    w_smooth  : weight for second-difference smoothness penalty.

    Training hyperparameters
    ------------------------
    lr           : peak learning rate for AdamW.
    weight_decay : L2 regularization coefficient.
    grad_clip    : maximum gradient norm (0 = disabled).
    warmup_steps : linear warm-up duration in steps.
    ema_decay    : EMA weight (0 = disabled).
    batch_size   : particles sampled per training step.
    epochs       : total training epochs.
    save_dir     : directory for checkpoints and logs.
    amp          : enable Automatic Mixed Precision on CUDA.
    log_interval : logging frequency in epochs.
    """
    # ---- model -----------------------------------------------------------
    node_in_dim:  int   = 3
    period_dim:   int   = 5
    hidden_dim:   int   = 128
    num_layers:   int   = 6
    dropout:      float = 0.10

    # ---- loss weights ----------------------------------------------------
    w_period:     float = 1.00
    w_spacing:    float = 0.10
    w_entropy:    float = 0.05
    w_smooth:     float = 0.02

    # ---- optimiser -------------------------------------------------------
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    warmup_steps: int   = 100
    ema_decay:    float = 0.999

    # ---- training --------------------------------------------------------
    batch_size:   int   = 4
    epochs:       int   = 500
    save_dir:     str   = "checkpoints_sgno_hodge"
    amp:          bool  = True
    log_interval: int   = 20


# =============================================================================
# Building Blocks
# =============================================================================

class LocalDensityEstimator(nn.Module):
    """
    Differentiable 1-D local particle density via Gaussian kernel density.

    For a sorted particle array ``x`` of shape ``(B, N)``, returns a density
    estimate ``rho`` of the same shape, where ``rho[b, i]`` reflects how
    many neighbours particle ``i`` has within bandwidth ``h``.

    Parameters
    ----------
    bandwidth : float
        KDE bandwidth. If ``None`` it is chosen adaptively as the median
        inter-particle spacing (computed at runtime, not differentiable w.r.t.
        positions but numerically stable).
    """

    def __init__(self, bandwidth: Optional[float] = None):
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, N)  — sorted particle positions

        Returns
        -------
        rho : Tensor (B, N) — local density at each particle position
        """
        h = self.bandwidth
        if h is None:
            # Adaptive bandwidth: median inter-particle spacing, detached
            spacings = (x[:, 1:] - x[:, :-1]).abs()          # (B, N-1)
            h = spacings.median(dim=1).values.detach()        # (B,)
            h = (h + 1e-6).unsqueeze(1)                       # (B, 1)

        # Pairwise distances: (B, N, N)
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        kernel = torch.exp(-0.5 * (diff / h) ** 2)
        rho = kernel.sum(dim=2) / (h * math.sqrt(2.0 * math.pi) * x.size(1))
        return rho                                             # (B, N)


class StructuralContextEncoder(nn.Module):
    """
    Encodes the FiLM conditioning context from scalar ``sigma`` (SOC stress)
    and the target Hodge period vector into a richer latent context.

    This replaces the bare concatenation used in v0, giving the FiLM
    modulators a non-linear view of the physics before conditioning.

    Parameters
    ----------
    in_dim  : raw context dimension = 1 + period_dim
    out_dim : output context dimension (equals ``hidden_dim`` for convenience)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sigma: torch.Tensor, target_hodge: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sigma        : (B, 1)          SOC criticality scalar
        target_hodge : (B, period_dim) target Hodge class vector

        Returns
        -------
        ctx : (B, out_dim)
        """
        raw = torch.cat([sigma, target_hodge], dim=-1)
        return self.net(raw)


class HodgeFiLMBlock(nn.Module):
    """
    One stage of the Structural GNO: a 1-D circular convolution block
    modulated by a FiLM layer conditioned on the structural context.

    Architecture per block
    ----------------------
    1. FiLM scale (gamma) and shift (beta) from context → applied to x
    2. Two-layer circular Conv1d with GELU activation and dropout
    3. Pre-norm residual: LayerNorm( x_input + conv_output )

    Parameters
    ----------
    dim         : feature width (= hidden_dim)
    context_dim : dimension of the encoded context vector
    dropout     : dropout probability
    """

    def __init__(self, dim: int, context_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size=1),          # pointwise reduce
        )
        # FiLM affine parameters: context → (gamma, beta) per feature channel
        self.film_gamma = nn.Linear(context_dim, dim)
        self.film_beta  = nn.Linear(context_dim, dim)
        self.norm       = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, dim, N)  — particle feature map
        ctx : (B, context_dim)

        Returns
        -------
        out : (B, dim, N)
        """
        gamma = self.film_gamma(ctx).unsqueeze(-1)   # (B, dim, 1) → broadcast
        beta  = self.film_beta(ctx).unsqueeze(-1)

        x_mod = gamma * x + beta
        conv_out = self.conv(x_mod)                  # (B, dim, N)

        # Residual + LayerNorm (channel-last for LayerNorm)
        residual = (x + conv_out).permute(0, 2, 1)  # (B, N, dim)
        return self.norm(residual).permute(0, 2, 1)  # (B, dim, N)


# =============================================================================
# Main Model
# =============================================================================

class StructuralGNOHodge(nn.Module):
    """
    Structural Graph Neural Operator for Hodge Class Targeting.

    The model receives an unordered set of 1-D particle positions, sorts them,
    computes local topological features, and predicts a displacement field
    (topological drift) that moves particles toward an algebraic cycle
    whose period map matches the target Hodge class vector.

    Architecture
    ------------
    1. Sort particles; compute local spacing and local KDE density.
    2. Node embedding: Linear(node_in_dim, hidden_dim) + LayerNorm.
    3. Context encoding: StructuralContextEncoder(1 + period_dim, hidden_dim).
    4. ``num_layers`` × HodgeFiLMBlock(hidden_dim, hidden_dim).
    5. Drift head: MLP(hidden_dim → hidden_dim//2 → 1) per particle.
    6. Output: sorted_x + drift  (positions on the manifold).

    Parameters
    ----------
    cfg : SGNOHodgeConfig
    """

    def __init__(self, cfg: SGNOHodgeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # ---- sub-modules ------------------------------------------------
        self.kde = LocalDensityEstimator()

        self.node_embed = nn.Sequential(
            nn.Linear(cfg.node_in_dim, d),
            nn.LayerNorm(d),
        )

        self.ctx_encoder = StructuralContextEncoder(
            in_dim  = 1 + cfg.period_dim,
            out_dim = d,
        )

        self.layers = nn.ModuleList([
            HodgeFiLMBlock(d, d, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.drift_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # ---- weight initialisation --------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming / Xavier initialisation for Linear and Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_pos:        torch.Tensor,
        target_hodge: torch.Tensor,
        sigma:        torch.Tensor,
    ) -> torch.Tensor:
        """
        One-shot topological drift prediction.

        Parameters
        ----------
        x_pos        : (B, N)          Unsorted particle positions on manifold
        target_hodge : (B, period_dim) Target Hodge class vector
        sigma        : (B, 1)          SOC criticality scalar

        Returns
        -------
        x_optimal : (B, N)  Rearranged particle positions (sorted order)
        """
        # 1. Sort particles for consistent topology
        sorted_x, _ = torch.sort(x_pos, dim=1)                        # (B, N)

        # 2. Local features
        spacings = torch.zeros_like(sorted_x)
        spacings[:, 1:] = sorted_x[:, 1:] - sorted_x[:, :-1]         # ≥ 0

        density = self.kde(sorted_x)                                   # (B, N)

        # 3. Node embedding: [position, spacing, density]
        nodes = torch.stack([sorted_x, spacings, density], dim=-1)    # (B, N, 3)
        h = self.node_embed(nodes).permute(0, 2, 1)                   # (B, d, N)

        # 4. Structural context
        ctx = self.ctx_encoder(sigma, target_hodge)                   # (B, d)

        # 5. FiLM-modulated operator blocks
        for layer in self.layers:
            h = layer(h, ctx)

        # 6. Per-particle drift
        h_t = h.permute(0, 2, 1)                                      # (B, N, d)
        drift = self.drift_head(h_t).squeeze(-1)                      # (B, N)

        return sorted_x + drift


# =============================================================================
# Loss Functions
# =============================================================================

def loss_period(
    computed_periods: torch.Tensor,
    target_vector:    torch.Tensor,
) -> torch.Tensor:
    """
    Hodge period loss: MSE between computed period vector and target.

    Parameters
    ----------
    computed_periods : (B, period_dim)
    target_vector    : (B, period_dim)

    Returns
    -------
    scalar loss
    """
    return F.mse_loss(computed_periods, target_vector)


def loss_spacing(x_sorted: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Anti-collapse regularizer: penalise particles crowding together.

    L_spacing = mean( 1 / (Δx_i + eps) )

    Parameters
    ----------
    x_sorted : (B, N) sorted particle positions
    eps      : numerical floor to avoid division by zero

    Returns
    -------
    scalar loss
    """
    gaps = x_sorted[:, 1:] - x_sorted[:, :-1]          # (B, N-1), ≥ 0
    return torch.mean(1.0 / (gaps + eps))


def loss_entropy(x_sorted: torch.Tensor, n_bins: int = 32) -> torch.Tensor:
    """
    Soft-entropy diversity term.  Encourages particles to spread evenly
    rather than cluster into a few modes.

    Approximated by differentiable soft-histogram entropy.

    Parameters
    ----------
    x_sorted : (B, N)
    n_bins   : number of histogram bins

    Returns
    -------
    scalar loss (negative entropy, i.e. minimising → maximising diversity)
    """
    x_min = x_sorted.min(dim=1, keepdim=True).values
    x_max = x_sorted.max(dim=1, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-4)

    # Normalise to [0, n_bins]
    x_norm = (x_sorted - x_min) / x_range * n_bins    # (B, N)

    # Bin centres
    centres = torch.arange(n_bins, dtype=x_sorted.dtype,
                           device=x_sorted.device) + 0.5   # (n_bins,)

    # Soft histogram: Gaussian assignment to bins
    sigma_bin = 1.0
    diff = x_norm.unsqueeze(2) - centres.unsqueeze(0).unsqueeze(0)    # (B,N,n_bins)
    weights = torch.exp(-0.5 * (diff / sigma_bin) ** 2)
    hist = weights.sum(dim=1)                                          # (B, n_bins)
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-12)             # normalise

    # Shannon entropy (negative → maximise)
    ent = -(hist * (hist + 1e-12).log()).sum(dim=1).mean()
    return -ent   # we minimise this, so -entropy → minimise loss → maximise entropy


def loss_smooth(x_sorted: torch.Tensor) -> torch.Tensor:
    """
    Positional smoothness: penalise sharp second-order differences in the
    arrangement (analogous to a membrane energy).

    L_smooth = mean( (Δ²x_i)² )  where Δ²x_i = x_{i+2} - 2x_{i+1} + x_i

    Parameters
    ----------
    x_sorted : (B, N)

    Returns
    -------
    scalar loss
    """
    d2 = x_sorted[:, 2:] - 2.0 * x_sorted[:, 1:-1] + x_sorted[:, :-2]
    return (d2 ** 2).mean()


def compute_total_loss(
    x_optimal:        torch.Tensor,
    computed_periods: torch.Tensor,
    target_vector:    torch.Tensor,
    cfg:              SGNOHodgeConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted multi-component loss and return a scalar + log dict.

    Parameters
    ----------
    x_optimal        : (B, N)          model output positions (sorted)
    computed_periods : (B, period_dim) from DifferentiablePeriodComputer
    target_vector    : (B, period_dim)
    cfg              : SGNOHodgeConfig (holds loss weights)

    Returns
    -------
    total_loss : scalar Tensor
    log_dict   : {'loss_period': float, 'loss_spacing': float, ...}
    """
    lp = loss_period(computed_periods, target_vector)
    ls = loss_spacing(x_optimal)
    le = loss_entropy(x_optimal)
    lm = loss_smooth(x_optimal)

    total = (cfg.w_period  * lp
           + cfg.w_spacing * ls
           + cfg.w_entropy * le
           + cfg.w_smooth  * lm)

    log_dict = {
        "loss_period":  lp.item(),
        "loss_spacing": ls.item(),
        "loss_entropy": le.item(),
        "loss_smooth":  lm.item(),
        "loss_total":   total.item(),
    }
    return total, log_dict


# =============================================================================
# Exponential Moving Average
# =============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.

    Usage::

        ema = EMA(model, decay=0.999)
        # after every optimiser step:
        ema.update()
        # to evaluate with EMA weights:
        with ema.average_parameters():
            val_loss = evaluate(model, ...)

    Parameters
    ----------
    model : nn.Module
    decay : float — EMA momentum (0.999 recommended for production)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Call after every ``optimizer.step()``."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Swap EMA weights into the model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original (non-EMA) weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()},
                "decay":  self.decay}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}


# =============================================================================
# Cosine Annealing with Linear Warm-up
# =============================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warm-up for ``warmup_steps`` steps, then cosine annealing to
    ``min_lr_fraction * peak_lr`` over the remaining ``total_steps`` steps.

    Parameters
    ----------
    optimizer       : torch.optim.Optimizer
    warmup_steps    : int
    total_steps     : int
    min_lr_fraction : fraction of peak LR at end of cosine (default 0.05)
    """

    def __init__(
        self,
        optimizer:       torch.optim.Optimizer,
        warmup_steps:    int,
        total_steps:     int,
        min_lr_fraction: float = 0.05,
    ):
        self.warmup_steps    = warmup_steps
        self.total_steps     = total_steps
        self.min_lr_fraction = min_lr_fraction

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

        super().__init__(optimizer, lr_lambda=lr_lambda)


# =============================================================================
# Unified Trainer
# =============================================================================

class UnifiedHodgeOperatorTrainer:
    """
    Orchestrates joint training of:

    - ``StructuralGNOHodge``       — learns topological drift
    - ``DifferentiablePeriodComputer`` — differentiable period map (HODGE ONE)

    Training loop per epoch
    -----------------------
    1. Sample random initial particle positions (uniform on [XMIN, XMAX]).
    2. Compute SOC sigma from ``LearnableSOCKernel``.
    3. Forward pass through ``StructuralGNOHodge`` → x_optimal.
    4. Evaluate ``DifferentiablePeriodComputer(x_optimal)`` → computed_periods.
    5. Compute multi-component loss (period + spacing + entropy + smooth).
    6. Backward + AdamW step + LR scheduler step + EMA update.
    7. Gradient norm monitoring; early stop on NaN.

    Parameters
    ----------
    operator_model  : StructuralGNOHodge
    period_computer : hodge_one.DifferentiablePeriodComputer
    soc_kernel      : hodge_one.LearnableSOCKernel
    cfg             : SGNOHodgeConfig
    device          : torch.device
    """

    def __init__(
        self,
        operator_model:  StructuralGNOHodge,
        period_computer: hodge_one.DifferentiablePeriodComputer,
        soc_kernel:      hodge_one.LearnableSOCKernel,
        cfg:             SGNOHodgeConfig,
        device:          torch.device,
    ):
        self.model          = operator_model.to(device)
        self.period_computer = period_computer.to(device)
        self.soc_kernel     = soc_kernel.to(device)
        self.cfg            = cfg
        self.device         = device

        # Collect all trainable parameters
        params = (
            list(self.model.parameters())
            + list(self.period_computer.parameters())
            + list(self.soc_kernel.parameters())
        )

        self.optimizer = torch.optim.AdamW(
            params,
            lr           = cfg.lr,
            weight_decay = cfg.weight_decay,
        )

        self.total_steps = cfg.epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps = cfg.warmup_steps,
            total_steps  = self.total_steps,
        )

        # EMA
        self.ema = EMA(self.model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

        # AMP scaler (CUDA only)
        self.use_amp = cfg.amp and (device.type == "cuda")
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Checkpoint directory
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step: int   = 0
        self.best_loss:   float = float("inf")

    # ------------------------------------------------------------------
    def _sigma_from_kernel(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Compute SOC criticality scalar from initial particle dispersion.

        Parameters
        ----------
        x_init : (B, N)

        Returns
        -------
        sigma : (B, 1)
        """
        r = torch.std(x_init, dim=1, keepdim=True)   # (B, 1)
        return self.soc_kernel(r)                     # (B, 1)

    # ------------------------------------------------------------------
    def train_step(
        self,
        x_init:       torch.Tensor,
        target_class: hodge_one.HodgeClass,
    ) -> Dict[str, float]:
        """
        Single training step.

        Parameters
        ----------
        x_init       : (B, N)  initial particle positions
        target_class : HodgeClass

        Returns
        -------
        log_dict : dict of scalar metrics (compatible with WandB / TensorBoard)
        """
        self.model.train()
        self.period_computer.train()
        self.optimizer.zero_grad(set_to_none=True)

        B = x_init.size(0)
        target_vector = (
            target_class.vector
            .unsqueeze(0)
            .expand(B, -1)
            .to(self.device)
        )

        with autocast(enabled=self.use_amp):
            sigma    = self._sigma_from_kernel(x_init)
            x_opt    = self.model(x_init, target_vector, sigma)
            periods  = self.period_computer(x_opt)
            total_loss, log_dict = compute_total_loss(
                x_opt, periods, target_vector, self.cfg
            )

        # Backward
        self.scaler.scale(total_loss).backward()

        # Gradient clipping + monitoring
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            log_dict["grad_norm"] = grad_norm.item()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        if self.ema is not None:
            self.ema.update()

        log_dict["lr"] = self.scheduler.get_last_lr()[0]
        self.global_step += 1
        return log_dict

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        x_eval:       torch.Tensor,
        target_class: hodge_one.HodgeClass,
        use_ema:      bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate with (optionally) EMA weights.

        Parameters
        ----------
        x_eval       : (B, N)
        target_class : HodgeClass
        use_ema      : bool — swap in EMA weights if available

        Returns
        -------
        log_dict : dict of scalar metrics
        """
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        self.period_computer.eval()

        B = x_eval.size(0)
        target_vector = (
            target_class.vector
            .unsqueeze(0)
            .expand(B, -1)
            .to(self.device)
        )

        sigma   = self._sigma_from_kernel(x_eval)
        x_opt   = self.model(x_eval, target_vector, sigma)
        periods = self.period_computer(x_opt)
        _, log_dict = compute_total_loss(
            x_opt, periods, target_vector, self.cfg
        )

        if use_ema and self.ema is not None:
            self.ema.restore()

        return log_dict

    # ------------------------------------------------------------------
    def train(
        self,
        target_class: hodge_one.HodgeClass,
        xmin: float = -5.0,
        xmax: float =  5.0,
        n_particles: int = 200,
    ) -> None:
        """
        Full training loop.

        Parameters
        ----------
        target_class : HodgeClass — the Hodge period vector to target
        xmin, xmax   : float — particle position domain
        n_particles  : int   — number of particles per sample
        """
        logger.info("=" * 65)
        logger.info("  SGNO-HODGE  |  Production Training")
        logger.info(f"  Epochs      : {self.cfg.epochs}")
        logger.info(f"  Batch size  : {self.cfg.batch_size}")
        logger.info(f"  Hidden dim  : {self.cfg.hidden_dim}")
        logger.info(f"  Layers      : {self.cfg.num_layers}")
        logger.info(f"  AMP         : {self.use_amp}")
        logger.info(f"  EMA decay   : {self.cfg.ema_decay}")
        logger.info(f"  Device      : {self.device}")
        logger.info("=" * 65)

        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            # Sample fresh batch each epoch
            x_init = (
                torch.rand(self.cfg.batch_size, n_particles, device=self.device)
                * (xmax - xmin) + xmin
            )

            log = self.train_step(x_init, target_class)

            # Validation with EMA weights (same batch, no grad)
            val_log = self.evaluate(x_init.clone(), target_class, use_ema=True)

            if epoch % self.cfg.log_interval == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:4d}/{self.cfg.epochs} | "
                    f"train={log['loss_total']:.5f} | "
                    f"val={val_log['loss_total']:.5f} | "
                    f"period={log['loss_period']:.5f} | "
                    f"grad={log.get('grad_norm', 0):.3f} | "
                    f"lr={log['lr']:.2e} | "
                    f"t={elapsed:.1f}s"
                )

                # Log SOC kernel state
                soc = self.soc_kernel
                logger.info(
                    f"  SOC kernel — Cs={soc.Cs.item():.4f}  "
                    f"λ={soc.lambd.item():.4f}  "
                    f"α={soc.alpha.item():.4f}  "
                    f"τ={soc.tau.item():.4f}"
                )

            # NaN guard
            if math.isnan(log["loss_total"]):
                logger.error("NaN loss detected — aborting training.")
                break

            # Checkpoint: save on improvement
            if val_log["loss_total"] < self.best_loss:
                self.best_loss = val_log["loss_total"]
                self._save_checkpoint(epoch, val_log["loss_total"], tag="best")

        # Final checkpoint
        self._save_checkpoint(self.cfg.epochs, self.best_loss, tag="final")
        logger.info(f"Training complete. Best loss: {self.best_loss:.6f}")

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, loss: float, tag: str = "ckpt") -> None:
        """Save model, EMA, optimizer, and scheduler state."""
        ckpt = {
            "epoch":            epoch,
            "loss":             loss,
            "model_state":      self.model.state_dict(),
            "period_state":     self.period_computer.state_dict(),
            "soc_state":        self.soc_kernel.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.scheduler.state_dict(),
            "cfg":              asdict(self.cfg),
        }
        if self.ema is not None:
            ckpt["ema_state"] = self.ema.state_dict()

        path = self.save_dir / f"sgno_hodge_{tag}.pt"
        torch.save(ckpt, path)
        logger.info(f"  ✓ Checkpoint saved → {path}  (loss={loss:.6f})")

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load a checkpoint into the trainer (model, EMA, optimizer, scheduler).

        Parameters
        ----------
        path : str — path to ``.pt`` checkpoint file

        Returns
        -------
        ckpt dict (for inspection)
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.period_computer.load_state_dict(ckpt["period_state"])
        self.soc_kernel.load_state_dict(ckpt["soc_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.ema is not None and "ema_state" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state"], self.device)
        logger.info(
            f"Loaded checkpoint from '{path}'  "
            f"(epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.6f})"
        )
        return ckpt


# =============================================================================
# Factory / Builder
# =============================================================================

def build_system(
    cfg:         SGNOHodgeConfig,
    n_particles: int,
    xmin:        float,
    xmax:        float,
    device:      torch.device,
) -> Tuple[StructuralGNOHodge,
           hodge_one.DifferentiablePeriodComputer,
           hodge_one.LearnableSOCKernel]:
    """
    Convenience factory that builds all three components with matching
    hyperparameters.

    Parameters
    ----------
    cfg         : SGNOHodgeConfig
    n_particles : int
    xmin, xmax  : float — domain bounds
    device      : torch.device

    Returns
    -------
    (model, period_computer, soc_kernel)
    """
    model = StructuralGNOHodge(cfg).to(device)

    period_computer = hodge_one.DifferentiablePeriodComputer(
        N_particles = n_particles,
        period_dim  = cfg.period_dim,
        XMIN        = xmin,
        XMAX        = xmax,
        device      = str(device),
    ).to(device)

    soc_kernel = hodge_one.LearnableSOCKernel(device=str(device)).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"StructuralGNOHodge built: {n_params:,} trainable parameters")

    return model, period_computer, soc_kernel


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="structural_gno_hodge.py — Production SGNO-Hodge Trainer"
    )
    p.add_argument("--mode",       default="train",
                   choices=["train", "eval", "demo", "info"])
    p.add_argument("--device",     default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--N",          type=int,   default=200,
                   help="Number of particles")
    p.add_argument("--XMIN",       type=float, default=-5.0)
    p.add_argument("--XMAX",       type=float, default=5.0)
    p.add_argument("--period-dim", type=int,   default=5,
                   help="Dimension of target Hodge class")
    p.add_argument("--hidden-dim", type=int,   default=128)
    p.add_argument("--num-layers", type=int,   default=6)
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save-dir",   type=str,   default="checkpoints_sgno_hodge")
    p.add_argument("--checkpoint", type=str,   default=None,
                   help="Path to .pt checkpoint (for --mode eval)")
    p.add_argument("--no-amp",     action="store_true",
                   help="Disable AMP even on CUDA")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    cfg = SGNOHodgeConfig(
        period_dim  = args.period_dim,
        hidden_dim  = args.hidden_dim,
        num_layers  = args.num_layers,
        lr          = args.lr,
        batch_size  = args.batch_size,
        epochs      = args.epochs,
        save_dir    = args.save_dir,
        amp         = not args.no_amp,
    )

    if args.mode == "info":
        print(__doc__)
        print(json.dumps(asdict(cfg), indent=2))
        return

    model, period_computer, soc_kernel = build_system(
        cfg, args.N, args.XMIN, args.XMAX, device
    )
    target = hodge_one.HodgeClass.random(args.period_dim, device=str(device))
    logger.info(f"Target Hodge vector: {target.vector.cpu().numpy()}")

    trainer = UnifiedHodgeOperatorTrainer(
        operator_model  = model,
        period_computer = period_computer,
        soc_kernel      = soc_kernel,
        cfg             = cfg,
        device          = device,
    )

    if args.mode == "train":
        trainer.train(
            target_class = target,
            xmin         = args.XMIN,
            xmax         = args.XMAX,
            n_particles  = args.N,
        )

    elif args.mode == "eval":
        if args.checkpoint is None:
            logger.error("--checkpoint is required for --mode eval")
            return
        trainer.load_checkpoint(args.checkpoint)
        x_eval = (
            torch.rand(args.batch_size, args.N, device=device)
            * (args.XMAX - args.XMIN) + args.XMIN
        )
        log = trainer.evaluate(x_eval, target, use_ema=True)
        logger.info("Evaluation results (EMA weights):")
        for k, v in log.items():
            logger.info(f"  {k}: {v:.6f}")

    elif args.mode == "demo":
        logger.info("Running quick demo (10 steps, no checkpoint save)…")
        cfg_demo     = SGNOHodgeConfig(epochs=10, log_interval=1, save_dir="demo_ckpt")
        m, pc, soc   = build_system(cfg_demo, args.N, args.XMIN, args.XMAX, device)
        t_demo       = hodge_one.HodgeClass.random(cfg_demo.period_dim, device=str(device))
        demo_trainer = UnifiedHodgeOperatorTrainer(m, pc, soc, cfg_demo, device)
        demo_trainer.train(t_demo, args.XMIN, args.XMAX, args.N)
        logger.info("Demo complete.")


if __name__ == "__main__":
    main()
