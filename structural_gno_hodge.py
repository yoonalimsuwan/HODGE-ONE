"""
=============================================================================
structural_gno_hodge.py  —  Structural GNO-Hodge Operator (Production v2)
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
                          LR scheduling, gradient monitoring, full docstrings,
                          v2 migration to hodge_one's ComplexTorusLattice /
                          K3LatticeHodgeStructure / CycleClassMap API
  - GPT     (OpenAI)    — early architecture exploration, message-passing
                          design, phase-field surrogate concept
  - Gemini  (Google)    — v2 unified discrete/continuous extension,
                          one-shot phase evolution framing

=============================================================================
WHAT CHANGED IN THIS REVISION (v1 -> v2, 2026-06)
=============================================================================
hodge_one.py was rebuilt (v2) around two concrete, selectable families of
polarized Hodge structures instead of a 1-D toy line:

  * VarietyClass.ABELIAN      — particles live on a complex torus
                                 C^g / (Z^g + tau Z^g), stored as a real
                                 tensor (B, N, g, 2) of lattice-basis
                                 coordinates in [0,1); periods are computed
                                 by hodge_one.CycleClassMap as genuine period
                                 integrals against the torus's harmonic
                                 (1,1)-forms (hodge_one.ComplexTorusLattice).
  * VarietyClass.K3_ABSTRACT  — no particle cloud at all. The "cycle" is a
                                 vector in R^22 (the K3 lattice rank),
                                 projected into H^{1,1}_R by
                                 hodge_one.CycleClassMap before being read
                                 out as the candidate class.

This file's job is to adapt StructuralGNOHodge (a learned operator that
used to predict a 1-D drift field on a single real line, x_pos: (B,N)) to
both of these, per explicit design decision:

  1. ABELIAN — keep the 1-D drift network *exactly as it was* (it still
     receives x_pos: (B,N) and predicts a drift), but call it once per real
     coordinate slice: for g complex factors there are 2g real coordinates
     (Re, Im per factor), so the model is applied independently g*2 times
     -- one slice per (factor, re/im) pair -- with a *shared* set of
     weights (a single StructuralGNOHodge instance reused across slices,
     conditioned each time on which slice it is via a small slice-index
     embedding folded into the context). Slices are NOT coupled to each
     other inside the network (no cross-factor message passing) -- this is
     the explicitly chosen faster-but-uncoupled adapter design, not an
     oversight. Periodicity (the torus wraps at [0,1)) is enforced by
     applying `remainder(1.0)` to the network's output before it is
     consumed by hodge_one.CycleClassMap, since the underlying 1-D drift
     head was designed for an unbounded line and otherwise would not
     respect the torus topology.

  2. K3_ABSTRACT — there is no particle cloud, so the GNO's role changes
     qualitatively: a new lightweight head, StructuralGNOHodgeK3, consumes
     a noise vector plus the same (sigma, target_hodge) context used in the
     ABELIAN path and maps it *directly* to a raw_vec in R^22, i.e. it
     generates the candidate-class pre-image itself rather than moving
     particles around. This raw_vec is handed to hodge_one.CycleClassMap's
     K3 pathway in place of that module's own internal nn.Parameter
     (CycleClassMap.raw_vec), so the trainable degrees of freedom for the
     K3 case live in the GNO model, not in a bare parameter sitting inside
     hodge_one. (hodge_one.K3LatticeHodgeStructure's own (2,0)-projection
     logic is reused unchanged -- only the *source* of the pre-projection
     vector moves from a free nn.Parameter to a network output.)

Both paths are dispatched from one VarietyConfig, exactly mirroring how
hodge_one v2 itself is variety-agnostic at the top level. Everything not
related to this adaptation (EMA, AdamW + warmup/cosine schedule, AMP,
checkpointing, gradient clipping/monitoring, multi-term loss weighting) is
carried over unchanged from v1.

Compatibility note: hodge_one.DifferentiablePeriodComputer, hodge_one.
HodgeClass, and the old (N_particles, period_dim, XMIN, XMAX, device)
constructor signature no longer exist in hodge_one v2. Anywhere this file
used those, it now uses hodge_one.VarietyConfig / build_variety /
CycleClassMap / HodgeClass(vector=...) instead. hodge_one.LearnableSOCKernel
is unchanged and is reused as-is.

Usage
-----
  python structural_gno_hodge.py --mode train --variety abelian --epochs 500
  python structural_gno_hodge.py --mode train --variety k3_abstract --epochs 500
  python structural_gno_hodge.py --mode eval  --checkpoint best.pt --variety abelian
  python structural_gno_hodge.py --mode demo  --variety abelian
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

    Variety selection
    -----------------
    variety : "abelian" | "k3_abstract"
        Which hodge_one.VarietyClass to target. Determines both the
        geometric backend (hodge_one.build_variety) and which GNO head
        (1-D slice-adapter vs. R^22 generator) is active.
    g : int
        Number of elliptic-curve factors (ABELIAN only). period_dim is
        then g*g (set automatically from the built variety, not by hand).
    k3_seed : int
        RNG seed for sampling the K3 period point (K3_ABSTRACT only).

    Model hyperparameters
    ---------------------
    node_in_dim : int
        Dimension of per-particle input features.
        Default 3 = [position, local_spacing, local_density].
    period_dim : int
        Dimension of the target Hodge class / period vector. Overwritten
        at build time from the chosen variety's `period_dim` (g*g for
        ABELIAN, 22 for K3_ABSTRACT) -- the field is kept here only so a
        fully-constructed config can be serialized/inspected as a whole.
    hidden_dim : int
        Width of all hidden layers.
    num_layers : int
        Number of FiLM-modulated convolution blocks (ABELIAN slice
        network) / residual MLP blocks (K3 generator).
    dropout : float
        Dropout probability inside each block.
    noise_dim : int
        Dimension of the input noise vector for the K3_ABSTRACT generator
        head (unused in the ABELIAN path).

    Loss weights
    ------------
    w_period   : weight for MSE(computed_periods, target_vector).
    w_spacing  : weight for anti-collapse spacing regularizer (ABELIAN only).
    w_entropy  : weight for soft-entropy diversity term (ABELIAN only).
    w_smooth   : weight for second-difference smoothness penalty (ABELIAN only).
    w_hr       : weight for hodge_one.HodgeRiemannLoss (both varieties).
    w_int      : weight for hodge_one.IntegralityModule soft penalty (both).

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
    # ---- variety -----------------------------------------------------------
    variety:      str   = "abelian"          # "abelian" | "k3_abstract"
    g:            int   = 2
    k3_seed:      int   = 0

    # ---- model -----------------------------------------------------------
    node_in_dim:  int   = 3
    period_dim:   int   = 4                  # overwritten at build time
    hidden_dim:   int   = 128
    num_layers:   int   = 6
    dropout:      float = 0.10
    noise_dim:    int   = 16

    # ---- loss weights ----------------------------------------------------
    w_period:     float = 1.00
    w_spacing:    float = 0.10
    w_entropy:    float = 0.05
    w_smooth:     float = 0.02
    w_hr:         float = 0.10
    w_int:        float = 0.05

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
# Building Blocks (unchanged from v1)
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
    Encodes the FiLM conditioning context from scalar ``sigma`` (SOC stress),
    the target Hodge period vector, and (new in v2) an optional slice-index
    embedding, into a richer latent context.

    The slice-index embedding is what lets a *single shared* StructuralGNOHodge
    instance be reused across all 2*g real-coordinate slices of an ABELIAN
    torus (factor index x {Re, Im}) while still letting the network tell
    which slice it is currently predicting drift for -- without it, every
    slice would receive an identical context and necessarily produce an
    identical drift field, which is not desired (the Re/Im=f_i=tau_i*e_i
    role of each slice differs, and different factors generally have
    different tau_i).

    Parameters
    ----------
    in_dim     : raw context dimension = 1 + period_dim (+ slice_embed_dim if used)
    out_dim    : output context dimension (equals ``hidden_dim`` for convenience)
    n_slices   : number of distinct slice indices to embed (0 disables the
                 slice embedding entirely, reproducing the v1 behaviour
                 exactly -- used by the K3 path's reuse of this encoder, and
                 by ABELIAN when g*2 == 1, an edge case that cannot occur
                 since g >= 1 implies at least 2 slices, but kept general).
    slice_embed_dim : dimension of the slice embedding, folded into in_dim.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_slices: int = 0, slice_embed_dim: int = 8):
        super().__init__()
        self.n_slices = n_slices
        self.slice_embed_dim = slice_embed_dim if n_slices > 0 else 0
        total_in = in_dim + self.slice_embed_dim
        if n_slices > 0:
            self.slice_embed = nn.Embedding(n_slices, slice_embed_dim)
        else:
            self.slice_embed = None
        self.net = nn.Sequential(
            nn.Linear(total_in, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sigma: torch.Tensor, target_hodge: torch.Tensor,
                slice_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        sigma        : (B, 1)          SOC criticality scalar
        target_hodge : (B, period_dim) target Hodge class vector
        slice_idx    : (B,) long tensor of slice indices in [0, n_slices),
                       required iff this encoder was built with n_slices > 0.

        Returns
        -------
        ctx : (B, out_dim)
        """
        raw = torch.cat([sigma, target_hodge], dim=-1)
        if self.slice_embed is not None:
            if slice_idx is None:
                raise ValueError(
                    "StructuralContextEncoder was built with n_slices > 0 "
                    "but forward() was called without slice_idx."
                )
            raw = torch.cat([raw, self.slice_embed(slice_idx)], dim=-1)
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
# Main Model — ABELIAN slice network (unchanged core, slice-aware context)
# =============================================================================

class StructuralGNOHodge(nn.Module):
    """
    Structural Graph Neural Operator for Hodge Class Targeting (ABELIAN path).

    The model receives an unordered set of 1-D particle positions, sorts them,
    computes local topological features, and predicts a displacement field
    (topological drift) that moves particles toward an algebraic cycle
    whose period map matches the target Hodge class vector.

    v2 note: this is the *same* network as v1 (sort + local KDE features +
    FiLM-conditioned circular convolutions + drift head). What changed is
    only how it is *called*: under hodge_one v2's ABELIAN variety, a single
    particle is a point on a complex torus C^g, stored as g*2 real
    coordinates, and per the chosen "adapter" design this 1-D network is
    applied independently to each of the g*2 real-coordinate slices, sharing
    weights across slices but distinguishing them via a slice-index
    embedding inside ``StructuralContextEncoder`` (see ``slice_idx`` below).
    Periodic wraparound onto [0,1) is applied by the *caller*
    (StructuralGNOHodgeAbelianAdapter), not inside this class, so this
    network remains exactly as reusable / domain-agnostic as it was in v1.

    Architecture
    ------------
    1. Sort particles; compute local spacing and local KDE density.
    2. Node embedding: Linear(node_in_dim, hidden_dim) + LayerNorm.
    3. Context encoding: StructuralContextEncoder(1 + period_dim [+ slice
       embedding], hidden_dim).
    4. ``num_layers`` × HodgeFiLMBlock(hidden_dim, hidden_dim).
    5. Drift head: MLP(hidden_dim → hidden_dim//2 → 1) per particle.
    6. Output: sorted_x + drift  (positions on the manifold).

    Parameters
    ----------
    cfg : SGNOHodgeConfig
    n_slices : number of (factor, re/im) slices this instance will be
               shared across (0 disables slice embedding, reproducing v1
               behaviour for callers that only ever pass a single slice).
    """

    def __init__(self, cfg: SGNOHodgeConfig, n_slices: int = 0):
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
            in_dim   = 1 + cfg.period_dim,
            out_dim  = d,
            n_slices = n_slices,
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
        slice_idx:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-shot topological drift prediction.

        Parameters
        ----------
        x_pos        : (B, N)          Unsorted particle positions on manifold
        target_hodge : (B, period_dim) Target Hodge class vector
        sigma        : (B, 1)          SOC criticality scalar
        slice_idx    : (B,) long       Which (factor, re/im) slice this call
                                        is for; required iff the model was
                                        built with n_slices > 0.

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
        ctx = self.ctx_encoder(sigma, target_hodge, slice_idx)        # (B, d)

        # 5. FiLM-modulated operator blocks
        for layer in self.layers:
            h = layer(h, ctx)

        # 6. Per-particle drift
        h_t = h.permute(0, 2, 1)                                      # (B, N, d)
        drift = self.drift_head(h_t).squeeze(-1)                      # (B, N)

        return sorted_x + drift


# =============================================================================
# ABELIAN Adapter — applies the shared 1-D slice network across g*2 slices
# =============================================================================

class StructuralGNOHodgeAbelianAdapter(nn.Module):
    """
    Wraps a single shared ``StructuralGNOHodge`` instance so it can act on
    hodge_one v2's ABELIAN particle tensor z: (B, N, g, 2) (g complex-torus
    factors, each with a Re/Im pair of real lattice-basis coordinates in
    [0,1)).

    Per the chosen adapter design (fast, slices not cross-coupled inside the
    network): for each of the g*2 real-coordinate slices, flatten that slice
    to (B, N), run the shared StructuralGNOHodge forward with a slice-index
    embedding distinguishing it from the others, then apply periodic
    wraparound (``remainder(1.0)``) since the underlying drift head has no
    intrinsic notion that its domain is a circle rather than a line. The g*2
    results are stacked back into (B, N, g, 2) for hodge_one.CycleClassMap.

    Any cross-factor coupling that the SOC kernel itself provides comes only
    from the fact that ``StructuralGNOHodge``'s weights -- not its
    input/output values -- are shared across slices (so e.g. the model
    learns a single drift "policy" that generalizes across factors), not
    from explicit communication between slices during a forward pass. This
    matches the explicitly requested trade-off (speed over cross-factor
    coupling).

    Parameters
    ----------
    cfg : SGNOHodgeConfig  (cfg.g determines the number of factors -> 2*g slices)
    """

    def __init__(self, cfg: SGNOHodgeConfig):
        super().__init__()
        self.cfg = cfg
        self.g = cfg.g
        self.n_slices = 2 * cfg.g
        self.net = StructuralGNOHodge(cfg, n_slices=self.n_slices)

        # Fixed slice-index lookup table: slice k corresponds to
        # (factor = k // 2, component = k % 2 [0=Re, 1=Im]).
        self.register_buffer(
            "_slice_ids", torch.arange(self.n_slices, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        z_init:       torch.Tensor,
        target_hodge: torch.Tensor,
        sigma:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_init       : (B, N, g, 2)   initial particle lattice-basis coords,
                                       valued in [0,1) (see
                                       hodge_one.ComplexSSCSimulator.initial_uniform)
        target_hodge : (B, period_dim) target Hodge class vector (period_dim = g*g)
        sigma        : (B, 1)          SOC criticality scalar

        Returns
        -------
        z_opt : (B, N, g, 2)  drifted particle positions, wrapped to [0,1)
        """
        B, N, g, _ = z_init.shape
        assert g == self.g, f"adapter built for g={self.g}, got g={g}"

        out_slices: List[torch.Tensor] = []
        for k in range(self.n_slices):
            factor, comp = divmod(k, 2)
            x_slice = z_init[:, :, factor, comp]                       # (B, N)
            slice_idx = self._slice_ids[k].expand(B)                    # (B,)
            x_drifted = self.net(x_slice, target_hodge, sigma, slice_idx)
            out_slices.append(x_drifted.remainder(1.0))                 # torus wraparound

        # Re-assemble (B, N, g, 2) from the g*2 (B, N) slices.
        z_opt = torch.stack(out_slices, dim=-1).reshape(B, N, g, 2)
        return z_opt


# =============================================================================
# K3_ABSTRACT Generator — produces raw_vec in R^22 directly (no particles)
# =============================================================================

class StructuralGNOHodgeK3(nn.Module):
    """
    Generator head for hodge_one v2's K3_ABSTRACT variety.

    There is no particle cloud to drift in the K3_ABSTRACT case (no
    point-set variety is simulated -- see hodge_one.K3LatticeHodgeStructure),
    so per the chosen design, this model takes over the role that
    hodge_one.CycleClassMap.raw_vec (a bare nn.Parameter in hodge_one v2)
    used to play on its own: it consumes a noise vector together with the
    same (sigma, target_hodge) context used by the ABELIAN path, and maps
    that directly to a candidate pre-image vector in R^{period_dim}
    (period_dim = 22 for K3). This raw_vec is then handed to
    hodge_one.K3LatticeHodgeStructure's own (2,0)/(0,2)-removal projection
    (replicated here via ``project_to_h11`` so this module has no hidden
    coupling to hodge_one internals beyond reading its buffers) to produce
    the final candidate (1,1)-class -- exactly mirroring what
    CycleClassMap._k3_class() does internally, but with the GNO as the
    source of the vector instead of a free parameter.

    Architecture
    ------------
    1. Context encoding: StructuralContextEncoder(1 + period_dim, hidden_dim)
       (no slice embedding -- there is only ever one "slice" here).
    2. Concatenate noise + context, then ``num_layers`` pre-norm residual
       MLP blocks (the K3 analogue of HodgeFiLMBlock, without the
       convolutional/particle-indexed structure since there is no particle
       axis to convolve over).
    3. Output head: Linear(hidden_dim, period_dim).

    Parameters
    ----------
    cfg : SGNOHodgeConfig
    """

    def __init__(self, cfg: SGNOHodgeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.ctx_encoder = StructuralContextEncoder(
            in_dim=1 + cfg.period_dim, out_dim=d, n_slices=0,
        )
        self.noise_proj = nn.Linear(cfg.noise_dim, d)
        self.input_norm = nn.LayerNorm(d)

        blocks = []
        for _ in range(cfg.num_layers):
            blocks.append(nn.Sequential(
                nn.Linear(d, d),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, d),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.block_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.num_layers)])

        self.out_head = nn.Linear(d, cfg.period_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        noise:        torch.Tensor,
        target_hodge: torch.Tensor,
        sigma:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        noise        : (B, noise_dim)   input noise vector
        target_hodge : (B, period_dim)  target Hodge class vector (period_dim=22)
        sigma        : (B, 1)           SOC criticality scalar (kept for API
                                         symmetry with the ABELIAN path; the
                                         K3 case has no particle dispersion
                                         to derive sigma from, so the caller
                                         typically passes a constant or a
                                         value derived from `noise`'s norm).

        Returns
        -------
        raw_vec : (B, period_dim)  candidate pre-image vector (NOT yet
                                    projected into H^{1,1}_R -- see
                                    project_to_h11_batch for that step,
                                    applied by the trainer before the vector
                                    is scored).
        """
        ctx = self.ctx_encoder(sigma, target_hodge)                  # (B, d)
        h = self.input_norm(self.noise_proj(noise) + ctx)            # (B, d)
        for block, norm in zip(self.blocks, self.block_norms):
            h = norm(h + block(h))
        return self.out_head(h)                                      # (B, period_dim)


def project_to_h11_batch(k3: "hodge_one.K3LatticeHodgeStructure",
                          v: torch.Tensor) -> torch.Tensor:
    """
    Batched version of hodge_one.CycleClassMap._k3_class()'s projection
    step: removes the (sigma, conj(sigma)) component from each row of v so
    the result lies in H^{1,1}_R, exactly mirroring hodge_one's own
    (un-batched) implementation so the two stay numerically identical.

    Parameters
    ----------
    k3 : hodge_one.K3LatticeHodgeStructure  (holds gram, sigma_re, sigma_im)
    v  : (B, period_dim) candidate vectors, period_dim == k3.rank

    Returns
    -------
    v_11 : (B, period_dim) projected into H^{1,1}_R
    """
    def _proj_coeff(basis_vec: torch.Tensor) -> torch.Tensor:
        # k3.bilinear already broadcasts a batched left argument against an
        # unbatched right argument via its '...i,ij,...j->...' einsum, so
        # this reproduces hodge_one.K3LatticeHodgeStructure's own
        # (un-batched) projection formula exactly, just with v carrying a
        # leading batch dimension.
        denom = k3.bilinear(basis_vec, basis_vec) + 1e-12
        num = k3.bilinear(v, basis_vec.to(v.dtype))
        return num / denom

    c_re = _proj_coeff(k3.sigma_re.to(v.dtype))
    c_im = _proj_coeff(k3.sigma_im.to(v.dtype))
    v_11 = v - c_re.unsqueeze(-1) * k3.sigma_re.to(v.dtype) \
             - c_im.unsqueeze(-1) * k3.sigma_im.to(v.dtype)
    return v_11


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
    gaps = x_sorted[:, 1:] - x_sorted[:, :-1]          # (B, N-1)
    return torch.mean(1.0 / (gaps.abs() + eps))


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


def compute_total_loss_abelian(
    z_optimal:         torch.Tensor,
    computed_periods:  torch.Tensor,
    target_vector:     torch.Tensor,
    cfg:                SGNOHodgeConfig,
    hr_loss_fn:         "hodge_one.HodgeRiemannLoss",
    integrality:        "hodge_one.IntegralityModule",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted multi-component loss for the ABELIAN path and return a
    scalar + log dict.

    Parameters
    ----------
    z_optimal        : (B, N, g, 2)    model output particle positions
    computed_periods : (B, period_dim) from hodge_one.CycleClassMap
    target_vector    : (B, period_dim)
    cfg              : SGNOHodgeConfig (holds loss weights)
    hr_loss_fn       : hodge_one.HodgeRiemannLoss bound to the active variety
    integrality      : hodge_one.IntegralityModule bound to the active variety

    Returns
    -------
    total_loss : scalar Tensor
    log_dict   : {'loss_period': float, 'loss_spacing': float, ...}
    """
    B, N, g, _ = z_optimal.shape
    # Spacing/entropy/smoothness are defined per real-coordinate slice and
    # then averaged across slices, since each (factor, re/im) coordinate is
    # itself a 1-D arrangement of N particles on [0,1).
    ls_total = 0.0
    le_total = 0.0
    lm_total = 0.0
    n_slices = g * 2
    for factor in range(g):
        for comp in range(2):
            x_slice, _ = torch.sort(z_optimal[:, :, factor, comp], dim=1)
            ls_total = ls_total + loss_spacing(x_slice)
            le_total = le_total + loss_entropy(x_slice)
            lm_total = lm_total + loss_smooth(x_slice)
    ls = ls_total / n_slices
    le = le_total / n_slices
    lm = lm_total / n_slices

    lp = loss_period(computed_periods, target_vector)
    lhr = hr_loss_fn(computed_periods)
    lint = integrality.soft_penalty(computed_periods)

    total = (cfg.w_period  * lp
           + cfg.w_spacing * ls
           + cfg.w_entropy * le
           + cfg.w_smooth  * lm
           + cfg.w_hr      * lhr
           + cfg.w_int     * lint)

    log_dict = {
        "loss_period":      lp.item(),
        "loss_spacing":      ls.item() if torch.is_tensor(ls) else float(ls),
        "loss_entropy":      le.item() if torch.is_tensor(le) else float(le),
        "loss_smooth":       lm.item() if torch.is_tensor(lm) else float(lm),
        "loss_hodge_riemann": lhr.item(),
        "loss_integrality":   lint.item(),
        "loss_total":         total.item(),
    }
    return total, log_dict


def compute_total_loss_k3(
    computed_periods:  torch.Tensor,
    target_vector:      torch.Tensor,
    cfg:                 SGNOHodgeConfig,
    hr_loss_fn:          "hodge_one.HodgeRiemannLoss",
    integrality:         "hodge_one.IntegralityModule",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted loss for the K3_ABSTRACT path. There is no particle
    cloud, so the spacing/entropy/smoothness regularizers (which are
    defined on a 1-D arrangement of particles) do not apply -- the loss is
    purely period-fit + Hodge-Riemann + integrality, matching how
    hodge_one.HodgeSSCTrainer itself scores the K3 case.

    Parameters
    ----------
    computed_periods : (B, period_dim)  already projected into H^{1,1}_R
    target_vector    : (B, period_dim)
    cfg              : SGNOHodgeConfig
    hr_loss_fn       : hodge_one.HodgeRiemannLoss bound to the active variety
    integrality      : hodge_one.IntegralityModule bound to the active variety

    Returns
    -------
    total_loss : scalar Tensor
    log_dict   : dict of scalar metrics
    """
    lp = loss_period(computed_periods, target_vector)
    lhr = hr_loss_fn(computed_periods)
    lint = integrality.soft_penalty(computed_periods)

    total = cfg.w_period * lp + cfg.w_hr * lhr + cfg.w_int * lint

    log_dict = {
        "loss_period":        lp.item(),
        "loss_spacing":       0.0,
        "loss_entropy":       0.0,
        "loss_smooth":        0.0,
        "loss_hodge_riemann": lhr.item(),
        "loss_integrality":   lint.item(),
        "loss_total":         total.item(),
    }
    return total, log_dict


# =============================================================================
# Exponential Moving Average (unchanged from v1)
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
# Cosine Annealing with Linear Warm-up (unchanged from v1)
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
    Orchestrates joint training of the GNO operator (ABELIAN adapter or K3
    generator, chosen by ``built.kind``) with hodge_one v2's
    ``CycleClassMap``, ``HodgeRiemannLoss``, and ``IntegralityModule``.

    Training loop per epoch (ABELIAN)
    ----------------------------------
    1. Sample initial particle positions on the torus (uniform on [0,1)^{2g}).
    2. Compute SOC sigma from ``LearnableSOCKernel`` (per-slice dispersion,
       averaged across slices into one scalar per batch element).
    3. Forward pass through ``StructuralGNOHodgeAbelianAdapter`` -> z_optimal.
    4. Evaluate ``CycleClassMap(z_optimal)`` -> computed_periods.
    5. Compute multi-component loss (period + spacing + entropy + smooth +
       Hodge-Riemann + integrality-soft).
    6. Backward + AdamW step + LR scheduler step + EMA update.
    7. Gradient norm monitoring; early stop on NaN.

    Training loop per epoch (K3_ABSTRACT)
    --------------------------------------
    1. Sample a fresh noise batch.
    2. Forward pass through ``StructuralGNOHodgeK3`` -> raw_vec.
    3. Project raw_vec into H^{1,1}_R (project_to_h11_batch) -> computed_periods.
    4. Compute loss (period + Hodge-Riemann + integrality-soft; no
       spacing/entropy/smooth -- there are no particles).
    5. Backward + AdamW step + LR scheduler step + EMA update.

    In both cases, after training finishes, ``final_projected_class``
    applies hodge_one.IntegralityModule's HARD lattice projection exactly
    once, matching hodge_one v2's own two-stage integrality design (soft
    during training, hard at the end).

    Parameters
    ----------
    cfg     : SGNOHodgeConfig
    device  : torch.device
    """

    def __init__(self, cfg: SGNOHodgeConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # ---- build the hodge_one v2 variety -----------------------------
        variety_enum = hodge_one.VarietyClass(cfg.variety)
        vcfg = hodge_one.VarietyConfig(
            variety=variety_enum, g=cfg.g, k3_seed=cfg.k3_seed, device=str(device),
        )
        self.built = hodge_one.build_variety(vcfg)
        cfg.period_dim = self.built.period_dim
        logger.info(f"Built hodge_one variety: {self.built.kind.value}, "
                    f"period_dim={self.built.period_dim}")

        # NOTE: for K3_ABSTRACT, hodge_one.CycleClassMap allocates its own
        # internal raw_vec nn.Parameter (see hodge_one.CycleClassMap.__init__).
        # We keep the instance around for API symmetry (e.g. so callers can
        # still do trainer.cycle_map(None) to inspect hodge_one's own
        # un-trained baseline), but that parameter is deliberately excluded
        # from the optimizer below -- StructuralGNOHodgeK3 generates the
        # actual trained raw_vec instead, and project_to_h11_batch is called
        # directly on self.built.k3 rather than through self.cycle_map.
        self.cycle_map = hodge_one.CycleClassMap(self.built, N_particles=1, device=str(device)).to(device)
        self.hr_loss_fn = hodge_one.HodgeRiemannLoss(self.built)
        if self.built.kind == hodge_one.VarietyClass.ABELIAN:
            basis_dual = self.built.torus.integral_basis_dual()
        else:
            basis_dual = torch.eye(self.built.k3.rank, dtype=torch.float64, device=device)
        self.integrality = hodge_one.IntegralityModule(basis_dual)

        self.soc_kernel = hodge_one.LearnableSOCKernel(device=str(device)).to(device)

        if self.built.kind == hodge_one.VarietyClass.ABELIAN:
            self.model: nn.Module = StructuralGNOHodgeAbelianAdapter(cfg).to(device)
        else:
            self.model = StructuralGNOHodgeK3(cfg).to(device)

        # Collect all trainable parameters (model + soc kernel; the
        # cycle_map's own raw_vec parameter, if any, is intentionally
        # excluded for K3_ABSTRACT since the GNO generates that vector
        # instead -- see StructuralGNOHodgeK3's docstring).
        params = list(self.model.parameters()) + list(self.soc_kernel.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        self.total_steps = cfg.epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps=cfg.warmup_steps, total_steps=self.total_steps,
        )

        self.ema = EMA(self.model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

        self.use_amp = cfg.amp and (device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.global_step: int = 0
        self.best_loss: float = float("inf")

    # ------------------------------------------------------------------
    def _sigma_from_dispersion(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B,1) a dispersion-like scalar -> sigma: (B,1) via the SOC kernel."""
        return self.soc_kernel(r)

    # ------------------------------------------------------------------
    def _abelian_step(
        self, target_vector: torch.Tensor, train: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        cfg = self.cfg
        B = target_vector.size(0)
        N = self._n_particles
        g = self.built.torus.g

        z_init = torch.rand(B, N, g, 2, device=self.device)
        r = z_init.std(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)   # (B,1) dispersion proxy
        sigma = self._sigma_from_dispersion(r)

        z_opt = self.model(z_init, target_vector, sigma)
        periods = self.cycle_map(z_opt)
        total_loss, log_dict = compute_total_loss_abelian(
            z_opt, periods, target_vector, cfg, self.hr_loss_fn, self.integrality,
        )
        return total_loss, log_dict, z_opt

    # ------------------------------------------------------------------
    def _k3_step(
        self, target_vector: torch.Tensor, train: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        cfg = self.cfg
        B = target_vector.size(0)

        noise = torch.randn(B, cfg.noise_dim, device=self.device)
        r = noise.std(dim=1, keepdim=True)  # (B,1) dispersion proxy for SOC sigma
        sigma = self._sigma_from_dispersion(r)

        raw_vec = self.model(noise, target_vector, sigma)
        periods = project_to_h11_batch(self.built.k3, raw_vec)
        total_loss, log_dict = compute_total_loss_k3(
            periods, target_vector, cfg, self.hr_loss_fn, self.integrality,
        )
        return total_loss, log_dict, None

    # ------------------------------------------------------------------
    def train_step(
        self, target_class: "hodge_one.HodgeClass", n_particles: int = 200,
    ) -> Dict[str, float]:
        """
        Single training step. Dispatches to the ABELIAN or K3_ABSTRACT path
        based on ``self.built.kind``.

        Parameters
        ----------
        target_class : hodge_one.HodgeClass
        n_particles  : int  (ABELIAN only; ignored for K3_ABSTRACT)

        Returns
        -------
        log_dict : dict of scalar metrics (compatible with WandB / TensorBoard)
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._n_particles = n_particles

        B = self.cfg.batch_size
        target_vector = target_class.vector.to(self.device).unsqueeze(0).expand(B, -1)

        with autocast(enabled=self.use_amp):
            if self.built.kind == hodge_one.VarietyClass.ABELIAN:
                total_loss, log_dict, _ = self._abelian_step(target_vector, train=True)
            else:
                total_loss, log_dict, _ = self._k3_step(target_vector, train=True)

        self.scaler.scale(total_loss).backward()

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
        self, target_class: "hodge_one.HodgeClass", n_particles: int = 200,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate with (optionally) EMA weights.

        Parameters
        ----------
        target_class : hodge_one.HodgeClass
        n_particles  : int  (ABELIAN only)
        use_ema      : bool — swap in EMA weights if available

        Returns
        -------
        log_dict : dict of scalar metrics
        """
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        self._n_particles = n_particles

        B = self.cfg.batch_size
        target_vector = target_class.vector.to(self.device).unsqueeze(0).expand(B, -1)

        if self.built.kind == hodge_one.VarietyClass.ABELIAN:
            _, log_dict, _ = self._abelian_step(target_vector, train=False)
        else:
            _, log_dict, _ = self._k3_step(target_vector, train=False)

        if use_ema and self.ema is not None:
            self.ema.restore()

        return log_dict

    # ------------------------------------------------------------------
    def train(
        self, target_class: "hodge_one.HodgeClass", n_particles: int = 200,
    ) -> None:
        """
        Full training loop.

        Parameters
        ----------
        target_class : hodge_one.HodgeClass — the Hodge period vector to target
        n_particles  : int — particles per torus slice (ABELIAN only)
        """
        logger.info("=" * 65)
        logger.info("  SGNO-HODGE v2  |  Production Training")
        logger.info(f"  Variety     : {self.built.kind.value}")
        logger.info(f"  Period dim  : {self.built.period_dim}")
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
            log = self.train_step(target_class, n_particles)
            val_log = self.evaluate(target_class, n_particles, use_ema=True)

            if epoch % self.cfg.log_interval == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:4d}/{self.cfg.epochs} | "
                    f"train={log['loss_total']:.5f} | "
                    f"val={val_log['loss_total']:.5f} | "
                    f"period={log['loss_period']:.5f} | "
                    f"HR={log['loss_hodge_riemann']:.5f} | "
                    f"int={log['loss_integrality']:.5f} | "
                    f"grad={log.get('grad_norm', 0):.3f} | "
                    f"lr={log['lr']:.2e} | "
                    f"t={elapsed:.1f}s"
                )
                soc = self.soc_kernel
                logger.info(
                    f"  SOC kernel — Cs={soc.Cs.item():.4f}  "
                    f"λ={soc.lambd.item():.4f}  "
                    f"α={soc.alpha.item():.4f}  "
                    f"τ={soc.tau.item():.4f}"
                )

            if math.isnan(log["loss_total"]):
                logger.error("NaN loss detected — aborting training.")
                break

            if val_log["loss_total"] < self.best_loss:
                self.best_loss = val_log["loss_total"]
                self._save_checkpoint(epoch, val_log["loss_total"], tag="best")

        self._save_checkpoint(self.cfg.epochs, self.best_loss, tag="final")
        logger.info(f"Training complete. Best loss: {self.best_loss:.6f}")

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, loss: float, tag: str = "ckpt") -> None:
        """Save model, EMA, optimizer, and scheduler state."""
        ckpt = {
            "epoch":            epoch,
            "loss":             loss,
            "variety":          self.built.kind.value,
            "model_state":      self.model.state_dict(),
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
        if ckpt.get("variety") not in (None, self.built.kind.value):
            logger.warning(
                f"Checkpoint was trained on variety='{ckpt.get('variety')}' "
                f"but trainer is configured for '{self.built.kind.value}'."
            )
        self.model.load_state_dict(ckpt["model_state"])
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

    # ------------------------------------------------------------------
    @torch.no_grad()
    def final_projected_class(
        self, target_class: "hodge_one.HodgeClass", n_particles: int = 200,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the model once more (no grad, EMA weights if available) and
        apply hodge_one.IntegralityModule's HARD lattice projection exactly
        once, as the final post-processing step -- matching hodge_one v2's
        explicit two-stage integrality design.

        Returns
        -------
        dict with 'raw', 'projected', 'raw_integrality_gap',
        'hodge_riemann_residual' (same keys as
        hodge_one.HodgeSSCTrainer.final_projected_class for drop-in
        comparability).
        """
        if self.ema is not None:
            self.ema.apply_shadow()
        self.model.eval()
        self._n_particles = n_particles

        B = self.cfg.batch_size
        target_vector = target_class.vector.to(self.device).unsqueeze(0).expand(B, -1)

        if self.built.kind == hodge_one.VarietyClass.ABELIAN:
            _, _, z_opt = self._abelian_step(target_vector, train=False)
            raw = self.cycle_map(z_opt)
        else:
            noise = torch.randn(B, self.cfg.noise_dim, device=self.device)
            r = noise.std(dim=1, keepdim=True)
            sigma = self._sigma_from_dispersion(r)
            raw_vec = self.model(noise, target_vector, sigma)
            raw = project_to_h11_batch(self.built.k3, raw_vec)

        projected = self.integrality.project_to_lattice(raw)

        if self.ema is not None:
            self.ema.restore()

        return {
            "raw": raw,
            "projected": projected,
            "raw_integrality_gap": self.integrality.integrality_gap(raw),
            "hodge_riemann_residual": self.hr_loss_fn(projected),
        }


# =============================================================================
# Factory / Builder
# =============================================================================

def build_system(cfg: SGNOHodgeConfig, device: torch.device) -> UnifiedHodgeOperatorTrainer:
    """
    Convenience factory that builds the hodge_one v2 variety, the
    appropriate GNO head (ABELIAN adapter or K3 generator), and wraps
    everything in a ``UnifiedHodgeOperatorTrainer``.

    Parameters
    ----------
    cfg    : SGNOHodgeConfig
    device : torch.device

    Returns
    -------
    trainer : UnifiedHodgeOperatorTrainer
    """
    trainer = UnifiedHodgeOperatorTrainer(cfg, device)
    n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"{type(trainer.model).__name__} built: {n_params:,} trainable parameters")
    return trainer


def make_target(cfg: SGNOHodgeConfig, built: "hodge_one.BuiltVariety",
                 mode: str, device: torch.device) -> "hodge_one.HodgeClass":
    """
    Construct a target hodge_one.HodgeClass appropriate to the active
    variety.

    Parameters
    ----------
    cfg    : SGNOHodgeConfig
    built  : hodge_one.BuiltVariety (already constructed by the trainer)
    mode   : "known" | "random"
             "known" uses hodge_one's guaranteed-algebraic theta-divisor
             class E_00 for ABELIAN (see
             hodge_one.ComplexTorusLattice.known_algebraic_class), or a
             random vector for K3_ABSTRACT (no closed-form "known
             algebraic" K3 class is modeled in hodge_one v2).
    device : torch.device

    Returns
    -------
    hodge_one.HodgeClass
    """
    if built.kind == hodge_one.VarietyClass.ABELIAN and mode == "known":
        return hodge_one.HodgeClass(
            built.torus.known_algebraic_class(0, 0), normalize=False
        )
    return hodge_one.HodgeClass.random(built.period_dim, device=str(device))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="structural_gno_hodge.py — Production SGNO-Hodge Trainer (v2)"
    )
    p.add_argument("--mode",       default="train",
                   choices=["train", "eval", "demo", "info"])
    p.add_argument("--variety",    default="abelian",
                   choices=["abelian", "k3_abstract"],
                   help="Which hodge_one v2 variety to target")
    p.add_argument("--g",          type=int,   default=2,
                   help="Number of elliptic-curve factors (ABELIAN only)")
    p.add_argument("--device",     default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--N",          type=int,   default=200,
                   help="Number of particles per torus slice (ABELIAN only)")
    p.add_argument("--hidden-dim", type=int,   default=128)
    p.add_argument("--num-layers", type=int,   default=6)
    p.add_argument("--noise-dim",  type=int,   default=16,
                   help="Noise vector dimension (K3_ABSTRACT only)")
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--target",     default="known", choices=["known", "random"])
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
        variety     = args.variety,
        g           = args.g,
        hidden_dim  = args.hidden_dim,
        num_layers  = args.num_layers,
        noise_dim   = args.noise_dim,
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

    trainer = build_system(cfg, device)
    target = make_target(cfg, trainer.built, args.target, device)
    logger.info(f"Target Hodge vector: {target.vector.cpu().numpy()}")

    if args.mode == "train":
        trainer.train(target_class=target, n_particles=args.N)

    elif args.mode == "eval":
        if args.checkpoint is None:
            logger.error("--checkpoint is required for --mode eval")
            return
        trainer.load_checkpoint(args.checkpoint)
        log = trainer.evaluate(target, n_particles=args.N, use_ema=True)
        logger.info("Evaluation results (EMA weights):")
        for k, v in log.items():
            logger.info(f"  {k}: {v:.6f}")

    elif args.mode == "demo":
        logger.info("Running quick demo (10 steps, no checkpoint save)…")
        cfg_demo = SGNOHodgeConfig(
            variety=args.variety, g=args.g, epochs=10, log_interval=1,
            save_dir="demo_ckpt", hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, noise_dim=args.noise_dim,
        )
        demo_trainer = build_system(cfg_demo, device)
        t_demo = make_target(cfg_demo, demo_trainer.built, args.target, device)
        demo_trainer.train(target_class=t_demo, n_particles=args.N)
        logger.info("Demo complete.")


if __name__ == "__main__":
    main()
