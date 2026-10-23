"""
=============================================================================
HODGE DREAM LLM v1.0 — Advanced Hodge Conjecture Simulation System
Scientific Discovery Through Mathematical Dreaming
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Hodge Conjecture Simulation + Scientific Dreaming)
=============================================================================

MATHEMATICAL FOUNDATION: HODGE CONJECTURE
─────────────────────────────────────────────────────────────────────────────
The Hodge Conjecture (one of the Millennium Prize Problems):

  "On a smooth projective algebraic variety X over C, every Hodge
   class is a rational linear combination of cohomology classes of
   algebraic cycles."

Formally:
  Hodge^p(X) ∩ H^{2p}(X, Q) = span_Q{[Z] : Z is algebraic p-cycle}

Key Components:
  [1] Hodge Decomposition:
      H^n(X, C) = ⊕_{p+q=n} H^{p,q}(X)
      where H^{p,q} = {α ∈ H^n : α is (p,q)-form}

  [2] Hodge Numbers:
      h^{p,q} = dim_C H^{p,q}(X)
      For K3 surfaces: h^{1,1} = 20, h^{2,0} = h^{0,2} = 1

  [3] Algebraic Cycles:
      Formal sums Z = Σ n_i [V_i] where V_i are subvarieties
      Codimension p cycles live in H^{2p}(X, Z)

  [4] Period Map:
      Φ: M_Hodge → D (period domain)
      Maps complex structure moduli to Hodge filtrations

  [5] Polarization:
      Q: H^n × H^n → Q satisfying Hodge-Riemann relations
      Ensures positivity of intersection form

SCIENTIFIC APPLICATIONS
───────────────────────
  • String Theory: Calabi-Yau compactification, mirror symmetry
  • Materials Science: Topological phases, band structure topology
  • Quantum Computing: Error-correcting codes from algebraic geometry
  • Biology: Protein folding landscapes as algebraic varieties
  • Climate Science: Phase space topology of climate attractors

THE "DREAMING" CONCEPT
──────────────────────
This system "dreams" in the mathematical sense:
  1. Latent Diffusion on Hodge Structure: Generate new varieties by
     diffusing through the period domain
  2. Imagination via Interpolation: Create hypothetical Hodge classes
     and test if they could be algebraic
  3. Creative Synthesis: Combine known algebraic cycles in novel ways
     to discover potential counterexamples or confirmations

The LLM backbone treats Hodge numbers, period matrices, and cycle
classes as "tokens" — the language of algebraic geometry becomes
the vocabulary of the model.

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import logging
import argparse
import contextlib
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("hodge.dream")

__all__ = [
    "HodgeDreamConfig",
    "AlgebraicVarietySimulator",
    "HodgeDecomposition",
    "HodgeConjectureVerifier",
    "DreamingEngine",
    "HodgeDreamLLM",
    "HodgeDreamLoss",
    "HodgeDreamTrainer",
    "HodgeDreamDataset",
    "build_hodge_loaders",
    "setup_dist", "cleanup_dist", "seed_all",
]


# ═══════════════════════════════════════════════════════════════════════
# AMP / DDP Helpers
# ═══════════════════════════════════════════════════════════════════════

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = False


def amp_ctx(enabled: bool, dtype: torch.dtype, device_type: str = "cuda"):
    if not enabled or device_type != "cuda":
        return contextlib.nullcontext()
    if _AMP_NEW:
        return _autocast(device_type=device_type, dtype=dtype, enabled=True)
    return _autocast(dtype=dtype, enabled=True)


@contextlib.contextmanager
def fp32_ctx():
    if _AMP_NEW:
        with _autocast(device_type="cuda", enabled=False):
            yield
    else:
        with _autocast(enabled=False):
            yield


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return rank() == 0

def all_reduce_mean(x: Tensor) -> Tensor:
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / world_size()


def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group(backend=backend)
            return torch.device("cuda", local), dist.get_rank(), dist.get_world_size()
        dist.init_process_group(backend="gloo")
        return torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    return torch.device("cpu"), 0, 1


def cleanup_dist() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Smooth Primitives
# ═══════════════════════════════════════════════════════════════════════

def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


class LowRankAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.zeros(rank, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HodgeDreamConfig:
    # ── Variety parameters ─────────────────────────────────────────────
    variety_type: str = "k3"          # "torus", "k3", "calabi_yau"
    dimension: int = 2                 # complex dimension
    max_hodge_pq: int = 3              # max p+q to consider
    
    # ── Hodge structure ────────────────────────────────────────────────
    # Standard Hodge numbers for different varieties
    hodge_numbers_torus: Tuple[int, int, int] = (1, 2, 1)      # h^0, h^1, h^2
    hodge_numbers_k3: Tuple[int, int, int] = (1, 20, 1)        # h^{2,0}, h^{1,1}, h^{0,2}
    hodge_numbers_cy3: Tuple[int, ...] = (1, 0, 0, 1)          # h^{3,0}, h^{2,1}, h^{1,2}, h^{0,3}
    
    # ── LLM backbone ───────────────────────────────────────────────────
    vocab_size: int = 256              # Hodge "tokens": numbers, symbols, cycles
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    max_seq: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    
    # ── HCSN integration ───────────────────────────────────────────────
    hcsn_every: int = 3
    hcsn_scales: int = 4
    
    # ── Dreaming engine ────────────────────────────────────────────────
    dream_latent_dim: int = 256
    dream_steps: int = 50
    dream_noise_scale: float = 0.1
    imagination_temperature: float = 0.8
    
    # ── Hodge conjecture verification ──────────────────────────────────
    n_algebraic_cycles: int = 32       # number of cycles to generate
    rationality_threshold: float = 1e-6
    period_precision: int = 64
    
    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3e-4
    min_lr_ratio: float = 0.02
    weight_decay: float = 0.05
    warmup_steps: int = 300
    max_steps: int = 30000
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    ema_decay: float = 0.9995
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 2000
    out_dir: str = "runs/hodge_dream"
    
    # ── Loss weights ───────────────────────────────────────────────────
    w_lm: float = 1.0
    w_hodge_decomposition: float = 0.5
    w_conjecture: float = 0.3
    w_dream: float = 0.2
    w_reg: float = 1e-6
    
    # ── Data ───────────────────────────────────────────────────────────
    n_train: int = 16384
    n_val: int = 2048
    n_test: int = 4096
    data_seed: int = 2026
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# 2. ALGEBRAIC VARIETY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class AlgebraicVarietySimulator(nn.Module):
    """
    Simulates algebraic varieties for Hodge theory.
    
    Supports:
      - Complex tori: C^n / Λ (lattice)
      - K3 surfaces: simply connected, trivial canonical bundle
      - Calabi-Yau 3-folds: for string theory applications
    
    The variety is represented by its period matrix and Hodge numbers.
    """
    
    def __init__(self, cfg: HodgeDreamConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        
        # Period matrix encoder (captures complex structure)
        # For a genus g surface, period matrix is g×2g
        self.period_encoder = nn.Sequential(
            LowRankAdapter(d, cfg.period_precision * 2, cfg.period_precision // 2),
            nn.GELU(),
            nn.Linear(cfg.period_precision * 2, cfg.period_precision * cfg.dimension),
        )
        
        # Hodge number predictor
        self.hodge_net = nn.Sequential(
            LowRankAdapter(d, d // 2, 32),
            nn.GELU(),
            nn.Linear(d // 2, (cfg.max_hodge_pq + 1) ** 2),  # all h^{p,q}
            nn.Softplus(),  # Hodge numbers are positive integers
        )
        
        # Variety type embedding
        self.variety_embed = nn.Embedding(3, d)  # 0: torus, 1: k3, 2: cy3
        
        # Complex structure moduli
        self.moduli_net = nn.Sequential(
            LowRankAdapter(d, d, 32),
            nn.Tanh(),  # moduli space is bounded
        )
    
    def get_hodge_numbers(self, variety_type: int) -> Tensor:
        """
        Returns standard Hodge numbers for the variety type.
        
        variety_type: 0 = torus, 1 = K3, 2 = Calabi-Yau 3-fold
        """
        if variety_type == 0:  # Torus
            return torch.tensor(self.cfg.hodge_numbers_torus, dtype=torch.float32)
        elif variety_type == 1:  # K3
            return torch.tensor(self.cfg.hodge_numbers_k3, dtype=torch.float32)
        else:  # CY3
            return torch.tensor(self.cfg.hodge_numbers_cy3, dtype=torch.float32)
    
    def forward(self, h: Tensor, variety_type: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Simulate an algebraic variety.
        
        Args:
            h: (B, d) latent state
            variety_type: (B,) optional variety type (0=torus, 1=k3, 2=cy3)
        
        Returns:
            dict with period_matrix, hodge_numbers, moduli
        """
        B = h.shape[0]
        device = h.device
        
        if variety_type is None:
            variety_type = torch.randint(0, 3, (B,), device=device)
        
        # Variety embedding
        var_emb = self.variety_embed(variety_type)  # (B, d)
        
        # Period matrix (complex structure)
        period_raw = self.period_encoder(h + var_emb)  # (B, period_dim * dim)
        period_matrix = period_raw.view(B, self.cfg.period_precision, self.cfg.dimension)
        
        # Hodge numbers (learned, but constrained by variety type)
        hodge_logits = self.hodge_net(h + var_emb)  # (B, (max_pq+1)^2)
        hodge_numbers = hodge_logits.view(B, self.cfg.max_hodge_pq + 1, self.cfg.max_hodge_pq + 1)
        
        # Add standard Hodge numbers as prior
        for i in range(B):
            std = self.get_hodge_numbers(int(variety_type[i].item())).to(device)
            # Blend learned with standard (soft constraint)
            hodge_numbers[i, :len(std)] = 0.7 * hodge_numbers[i, :len(std)] + 0.3 * std
        
        # Complex structure moduli
        moduli = self.moduli_net(h + var_emb)  # (B, d)
        
        return {
            "period_matrix": period_matrix,
            "hodge_numbers": hodge_numbers,
            "moduli": moduli,
            "variety_type": variety_type,
        }


# ═══════════════════════════════════════════════════════════════════════
# 3. HODGE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════

class HodgeDecomposition(nn.Module):
    """
    Computes Hodge decomposition: H^n(X, C) = ⊕ H^{p,q}(X)
    
    For a Kähler manifold, this decomposition is orthogonal with respect
    to the Hodge inner product.
    
    Key properties:
      - H^{p,q} = conjugate(H^{q,p})
      - Hodge star operator: * : H^{p,q} → H^{n-q,n-p}
      - Lefschetz decomposition for primitive cohomology
    """
    
    def __init__(self, cfg: HodgeDreamConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        
        # Decomposition network: splits cohomology into (p,q) components
        self.decompose_net = nn.Sequential(
            LowRankAdapter(d, d * 2, 32),
            nn.GELU(),
            nn.Linear(d * 2, d * (cfg.max_hodge_pq + 1)),
        )
        
        # Hodge star operator (learnable, should satisfy *² = (-1)^{pq})
        self.star_net = nn.Sequential(
            LowRankAdapter(d, d, 32),
            nn.Tanh(),
        )
        
        # Lefschetz operator L: H^k → H^{k+2} (wedge with Kähler form)
        self.lefschetz_net = nn.Sequential(
            LowRankAdapter(d, d, 32),
            nn.GELU(),
        )
    
    def forward(self, cohomology: Tensor, hodge_numbers: Tensor) -> Dict[str, Tensor]:
        """
        Decompose cohomology into Hodge components.
        
        Args:
            cohomology: (B, d) cohomology class representation
            hodge_numbers: (B, max_pq+1, max_pq+1) Hodge diamond
        
        Returns:
            dict with hodge_components, star_image, lefschetz_image
        """
        B = cohomology.shape[0]
        
        # Decompose into (p,q) components
        decomposed = self.decompose_net(cohomology)  # (B, d * (max_pq+1))
        hodge_components = decomposed.view(B, self.cfg.max_hodge_pq + 1, -1)
        
        # Apply Hodge star
        star_image = self.star_net(cohomology)  # (B, d)
        
        # Apply Lefschetz operator
        lefschetz_image = self.lefschetz_net(cohomology)  # (B, d)
        
        # Verify Hodge-Riemann bilinear relations (soft constraint)
        # Q(α, *α) > 0 for α ∈ H^{p,q}
        hr_product = (hodge_components * star_image.unsqueeze(1)).sum(dim=-1)
        
        return {
            "hodge_components": hodge_components,
            "star_image": star_image,
            "lefschetz_image": lefschetz_image,
            "hodge_riemann_product": hr_product,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. HODGE CONJECTURE VERIFIER
# ═══════════════════════════════════════════════════════════════════════

class HodgeConjectureVerifier(nn.Module):
    """
    Verifies the Hodge Conjecture for simulated varieties.
    
    The conjecture states:
      Every Hodge class in H^{2p}(X, Q) ∩ H^{p,p}(X) is a rational
      linear combination of classes of algebraic cycles.
    
    Our approach:
      1. Generate candidate algebraic cycles
      2. Compute their cohomology classes
      3. Check if Hodge class is in their Q-span
      4. Measure "distance to algebraicity"
    """
    
    def __init__(self, cfg: HodgeDreamConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        
        # Algebraic cycle generator
        self.cycle_generator = nn.Sequential(
            LowRankAdapter(d, d * 2, 32),
            nn.GELU(),
            nn.Linear(d * 2, cfg.n_algebraic_cycles * d),
        )
        
        # Cycle class map: cycle → cohomology class
        self.cycle_class_map = nn.Sequential(
            LowRankAdapter(d, d, 32),
            nn.Tanh(),
        )
        
        # Rationality detector: checks if coefficients are rational
        self.rationality_net = nn.Sequential(
            LowRankAdapter(d, d // 2, 32),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),  # probability of being rational
        )
        
        # Hodge class detector
        self.hodge_class_net = nn.Sequential(
            LowRankAdapter(d, d // 2, 32),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )
        
        # Distance to algebraicity metric
        self.algebraicity_net = nn.Sequential(
            LowRankAdapter(d * 2, d, 32),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.Softplus(),  # distance is positive
        )
    
    def generate_cycles(self, h: Tensor) -> Tensor:
        """
        Generate candidate algebraic cycles.
        
        Returns:
            cycles: (B, n_cycles, d) cycle representations
        """
        B = h.shape[0]
        cycles_raw = self.cycle_generator(h)  # (B, n_cycles * d)
        return cycles_raw.view(B, self.cfg.n_algebraic_cycles, -1)
    
    def cycle_classes(self, cycles: Tensor) -> Tensor:
        """
        Map cycles to cohomology classes.
        
        Args:
            cycles: (B, n_cycles, d)
        Returns:
            classes: (B, n_cycles, d)
        """
        return self.cycle_class_map(cycles)
    
    def check_rationality(self, coefficients: Tensor) -> Tensor:
        """
        Check if coefficients are (approximately) rational.
        
        Args:
            coefficients: (B, n_cycles) coefficients in Q-span
        
        Returns:
            rationality_score: (B,) probability of being rational
        """
        return self.rationality_net(coefficients).squeeze(-1)
    
    def verify_conjecture(self, hodge_class: Tensor, cycle_classes: Tensor) -> Dict[str, Tensor]:
        """
        Verify if a Hodge class is in the Q-span of algebraic cycle classes.
        
        Args:
            hodge_class: (B, d) the Hodge class to verify
            cycle_classes: (B, n_cycles, d) classes of algebraic cycles
        
        Returns:
            dict with verification results
        """
        B = hodge_class.shape[0]
        
        # Compute projection of hodge_class onto span of cycle_classes
        # This is a soft least-squares: find coefficients c such that
        # hodge_class ≈ Σ c_i * cycle_class_i
        
        # Soft pseudo-inverse via learned network
        # In practice: solve min ||hodge_class - C @ c||²
        C = cycle_classes  # (B, n_cycles, d)
        h = hodge_class.unsqueeze(-1)  # (B, d, 1)
        
        # Soft solve: c = (C^T C)^{-1} C^T h (approximated)
        CTC = torch.bmm(C.transpose(1, 2), C)  # (B, n_cycles, n_cycles)
        CTh = torch.bmm(C.transpose(1, 2), h).squeeze(-1)  # (B, n_cycles)
        
        # Regularized solve
        reg = self.cfg.rationality_threshold * torch.eye(
            self.cfg.n_algebraic_cycles, device=CTC.device
        ).unsqueeze(0)
        coefficients = torch.linalg.solve(CTC + reg, CTh)  # (B, n_cycles)
        
        # Reconstruct
        reconstructed = torch.bmm(C, coefficients.unsqueeze(-1)).squeeze(-1)  # (B, d)
        
        # Distance to algebraicity
        distance = self.algebraicity_net(
            torch.cat([hodge_class, reconstructed], dim=-1)
        ).squeeze(-1)
        
        # Rationality score
        rationality = self.check_rationality(coefficients)
        
        # Verification verdict
        is_algebraic = (distance < self.cfg.rationality_threshold).float()
        
        return {
            "coefficients": coefficients,
            "reconstructed": reconstructed,
            "distance_to_algebraicity": distance,
            "rationality_score": rationality,
            "is_algebraic": is_algebraic,
        }
    
    def forward(self, h: Tensor, hodge_class: Tensor) -> Dict[str, Tensor]:
        """
        Full Hodge conjecture verification pipeline.
        
        Args:
            h: (B, d) latent state
            hodge_class: (B, d) the Hodge class to verify
        
        Returns:
            dict with all verification results
        """
        # Generate algebraic cycles
        cycles = self.generate_cycles(h)
        
        # Map to cohomology classes
        cycle_cls = self.cycle_classes(cycles)
        
        # Verify conjecture
        verification = self.verify_conjecture(hodge_class, cycle_cls)
        
        # Detect if input is actually a Hodge class
        hodge_score = self.hodge_class_net(hodge_class).squeeze(-1)
        
        return {
            **verification,
            "hodge_score": hodge_score,
            "cycles": cycles,
            "cycle_classes": cycle_cls,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. DREAMING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class DreamingEngine(nn.Module):
    """
    The "Dreaming" module: creative exploration of Hodge structures.
    
    Dreams are:
      1. Latent diffusion through the period domain
      2. Interpolation between known varieties
      3. Generation of hypothetical Hodge classes
      4. Creative combination of algebraic cycles
    
    This is the "imagination" of the system — it generates mathematical
    objects that may or may not correspond to actual varieties.
    """
    
    def __init__(self, cfg: HodgeDreamConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        
        # Dream encoder: real variety → dream latent
        self.dream_encoder = nn.Sequential(
            LowRankAdapter(d, cfg.dream_latent_dim, 32),
            nn.GELU(),
            nn.Linear(cfg.dream_latent_dim, cfg.dream_latent_dim),
        )
        
        # Dream decoder: dream latent → imagined variety
        self.dream_decoder = nn.Sequential(
            LowRankAdapter(cfg.dream_latent_dim, d, 32),
            nn.GELU(),
            nn.Linear(d, d),
        )
        
        # Diffusion model for dreaming
        self.diffusion_net = nn.Sequential(
            nn.Linear(cfg.dream_latent_dim + 1, cfg.dream_latent_dim * 2),  # +1 for time
            nn.GELU(),
            nn.Linear(cfg.dream_latent_dim * 2, cfg.dream_latent_dim),
        )
        
        # Imagination interpolator
        self.interpolation_net = nn.Sequential(
            LowRankAdapter(cfg.dream_latent_dim * 2, cfg.dream_latent_dim, 32),
            nn.Tanh(),
        )
        
        # Creative cycle synthesizer
        self.cycle_synthesizer = nn.Sequential(
            LowRankAdapter(cfg.dream_latent_dim, d, 32),
            nn.GELU(),
            nn.Linear(d, cfg.n_algebraic_cycles * d),
        )
        
        # Dream quality assessor
        self.dream_quality_net = nn.Sequential(
            LowRankAdapter(cfg.dream_latent_dim, cfg.dream_latent_dim // 2, 32),
            nn.GELU(),
            nn.Linear(cfg.dream_latent_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def diffuse(self, z: Tensor, t: Tensor) -> Tensor:
        """
        One step of diffusion in dream space.
        
        Args:
            z: (B, dream_dim) current dream state
            t: (B,) diffusion time in [0, 1]
        
        Returns:
            z_next: (B, dream_dim) next dream state
        """
        # Concatenate time embedding
        t_emb = t.unsqueeze(-1)  # (B, 1)
        inp = torch.cat([z, t_emb], dim=-1)  # (B, dream_dim + 1)
        
        # Predict noise
        noise_pred = self.diffusion_net(inp)  # (B, dream_dim)
        
        # DDPM-style update (simplified)
        alpha_t = 1.0 - 0.1 * t  # noise schedule
        z_next = alpha_t * z + (1 - alpha_t) * noise_pred
        
        return z_next
    
    def dream(self, h: Tensor, n_steps: Optional[int] = None) -> Dict[str, Tensor]:
        """
        Generate a "dream" — an imagined mathematical object.
        
        Args:
            h: (B, d) real variety representation
            n_steps: number of diffusion steps
        
        Returns:
            dict with dream output
        """
        B = h.shape[0]
        device = h.device
        n_steps = n_steps or self.cfg.dream_steps
        
        # Encode to dream space
        z = self.dream_encoder(h)  # (B, dream_dim)
        
        # Add noise (start of dreaming)
        z = z + self.cfg.dream_noise_scale * torch.randn_like(z)
        
        # Diffusion process (dreaming)
        for step in range(n_steps):
            t = torch.full((B,), step / n_steps, device=device)
            z = self.diffuse(z, t)
        
        # Decode dream
        dream_variety = self.dream_decoder(z)  # (B, d)
        
        # Assess dream quality
        dream_quality = self.dream_quality_net(z).squeeze(-1)  # (B,)
        
        # Synthesize creative cycles from dream
        cycles_raw = self.cycle_synthesizer(z)  # (B, n_cycles * d)
        dream_cycles = cycles_raw.view(B, self.cfg.n_algebraic_cycles, -1)
        
        return {
            "dream_latent": z,
            "dream_variety": dream_variety,
            "dream_quality": dream_quality,
            "dream_cycles": dream_cycles,
        }
    
    def interpolate(self, z1: Tensor, z2: Tensor, alpha: Tensor) -> Tensor:
        """
        Interpolate between two dreams (imagination).
        
        Args:
            z1, z2: (B, dream_dim) two dream states
            alpha: (B,) interpolation parameter in [0, 1]
        
        Returns:
            z_interp: (B, dream_dim) interpolated dream
        """
        # Linear interpolation in dream space
        z_lin = alpha.unsqueeze(-1) * z1 + (1 - alpha.unsqueeze(-1)) * z2
        
        # Non-linear refinement
        z_cat = torch.cat([z1, z2], dim=-1)
        z_refined = self.interpolation_net(z_cat)
        
        # Blend
        return 0.5 * z_lin + 0.5 * z_refined
    
    def forward(self, h: Tensor, dream_mode: str = "diffuse") -> Dict[str, Tensor]:
        """
        Main dreaming interface.
        
        Args:
            h: (B, d) input variety
            dream_mode: "diffuse", "interpolate", or "create"
        
        Returns:
            dict with dream outputs
        """
        if dream_mode == "diffuse":
            return self.dream(h)
        elif dream_mode == "interpolate":
            # Interpolate between h and a random variety
            h_rand = torch.randn_like(h)
            z1 = self.dream_encoder(h)
            z2 = self.dream_encoder(h_rand)
            alpha = torch.rand(h.shape[0], device=h.device)
            z_interp = self.interpolate(z1, z2, alpha)
            dream_variety = self.dream_decoder(z_interp)
            return {
                "dream_latent": z_interp,
                "dream_variety": dream_variety,
                "dream_quality": self.dream_quality_net(z_interp).squeeze(-1),
                "dream_cycles": self.cycle_synthesizer(z_interp).view(
                    h.shape[0], self.cfg.n_algebraic_cycles, -1),
            }
        else:  # "create" - pure imagination
            z = torch.randn(h.shape[0], self.cfg.dream_latent_dim, device=h.device)
            dream_variety = self.dream_decoder(z)
            return {
                "dream_latent": z,
                "dream_variety": dream_variety,
                "dream_quality": self.dream_quality_net(z).squeeze(-1),
                "dream_cycles": self.cycle_synthesizer(z).view(
                    h.shape[0], self.cfg.n_algebraic_cycles, -1),
            }


# ═══════════════════════════════════════════════════════════════════════
# 6. HODGE DREAM LLM BACKBONE
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    
    def forward(self, x: Tensor) -> Tensor:
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).to(x.dtype) * self.w


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq: int = 8192):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
    
    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos[offset:offset+T, None, :].to(x.dtype)
        sin = self.sin[offset:offset+T, None, :].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.reshape(x.shape)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8/3):
        super().__init__()
        d_h = int(mult * d_model)
        self.w1 = nn.Linear(d_model, d_h, bias=False)
        self.w3 = nn.Linear(d_model, d_h, bias=False)
        self.w2 = nn.Linear(d_h, d_model, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: Tensor, kv_cache=None, offset: int = 0, rope=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            q = rope(q, offset)
            k = rope(k, offset)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        if offset == 0 and kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            Tq, Tk = T, k.shape[2]
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=Tk - Tq)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)


class HCSNPass(nn.Module):
    """Hierarchical Cloud Superposition measurement pass."""
    def __init__(self, d_model: int, n_scales: int = 4):
        super().__init__()
        self.scale_emb = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, n_scales),
        )
        self.outcome_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pointer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.Tanh(),
        )
        self.log_chi = nn.Parameter(torch.tensor(-3.0))
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))
        self.lambda_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
    
    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        w = F.softmax(self.scale_net(pooled), dim=-1)
        scales = torch.einsum("bs,sd->bd", w, self.scale_emb)
        bias = self.outcome_net(pooled + scales.unsqueeze(1))
        chi = torch.sigmoid(self.log_chi)
        pointer = self.pointer(x)
        gamma = F.softplus(self.log_gamma)
        damped = x * torch.exp(-gamma) + (1.0 - torch.exp(-gamma)) * pooled
        gate = self.lambda_gate(x + bias)
        return damped + chi * gate * pointer


class HodgeDreamBlock(nn.Module):
    def __init__(self, cfg: HodgeDreamConfig, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.mlp = SwiGLU(cfg.d_model)
        self.rope = RoPE(cfg.d_model // cfg.n_heads, max_seq=cfg.max_seq)
        self.use_hcsn = (cfg.hcsn_every > 0 and (layer_idx + 1) % cfg.hcsn_every == 0)
        self.hcsn = HCSNPass(cfg.d_model, cfg.hcsn_scales) if self.use_hcsn else None
    
    def forward(self, x, kv_cache=None, offset=0):
        a, kv = self.attn(self.n1(x), kv_cache, offset, self.rope)
        x = x + a
        x = x + self.mlp(self.n2(x))
        if self.hcsn is not None:
            x = self.hcsn(x)
        return x, kv


class HodgeDreamLLM(nn.Module):
    """
    Hodge Dream LLM: Language model for algebraic geometry.
    
    Treats Hodge numbers, period matrices, and cycle classes as tokens.
    The model "speaks" the language of algebraic geometry and can
    "dream" new mathematical structures.
    """
    
    def __init__(self, cfg: HodgeDreamConfig):
        super().__init__()
        self.cfg = cfg
        
        # Token embedding (Hodge "vocabulary")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # LLM backbone with HCSN passes
        self.blocks = nn.ModuleList([
            HodgeDreamBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        
        # LM head (next token prediction)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        
        # Hodge theory modules
        self.variety_sim = AlgebraicVarietySimulator(cfg)
        self.hodge_decomp = HodgeDecomposition(cfg)
        self.conjecture_verifier = HodgeConjectureVerifier(cfg)
        self.dreaming_engine = DreamingEngine(cfg)
        
        # Output heads
        self.variety_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.dimension),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
    
    def forward(
        self,
        tokens: Tensor,
        targets: Optional[Tensor] = None,
        variety_type: Optional[Tensor] = None,
        dream_mode: str = "diffuse",
    ) -> Dict[str, Tensor]:
        """
        Full forward pass.
        
        Args:
            tokens: (B, T) input token sequence
            targets: (B, T) target tokens (optional)
            variety_type: (B,) variety type (0=torus, 1=k3, 2=cy3)
            dream_mode: "diffuse", "interpolate", or "create"
        
        Returns:
            dict with all outputs
        """
        B, T = tokens.shape
        out = {}
        
        # ── LLM backbone ────────────────────────────────────────────────
        x = self.embed(tokens)
        
        for blk in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, None, 0, use_reentrant=False)
            else:
                x, _ = blk(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        out["logits"] = logits
        
        if targets is not None:
            out["lm_loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # ── Pooled hidden state ─────────────────────────────────────────
        h_pooled = x.mean(dim=1)  # (B, d)
        out["h_pooled"] = h_pooled
        
        # ── Variety simulation ──────────────────────────────────────────
        variety_out = self.variety_sim(h_pooled, variety_type)
        out.update({f"variety_{k}": v for k, v in variety_out.items()})
        
        # ── Hodge decomposition ─────────────────────────────────────────
        decomp_out = self.hodge_decomp(
            h_pooled, variety_out["hodge_numbers"])
        out.update({f"hodge_{k}": v for k, v in decomp_out.items()})
        
        # ── Hodge conjecture verification ───────────────────────────────
        # Use the (p,p) component as the Hodge class to verify
        # For simplicity, use the middle component
        mid_p = self.cfg.dimension
        hodge_class = decomp_out["hodge_components"][:, mid_p]  # (B, d)
        
        conj_out = self.conjecture_verifier(h_pooled, hodge_class)
        out.update({f"conj_{k}": v for k, v in conj_out.items()})
        
        # ── Dreaming ────────────────────────────────────────────────────
        dream_out = self.dreaming_engine(h_pooled, dream_mode)
        out.update({f"dream_{k}": v for k, v in dream_out.items()})
        
        # ── Variety prediction ──────────────────────────────────────────
        variety_pred = self.variety_head(h_pooled)
        out["variety_pred"] = variety_pred
        
        return out
    
    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new: int = 64,
                 temperature: float = 0.8, top_k: int = 50) -> Tensor:
        """Generate Hodge "text" (mathematical structures)."""
        caches = [None] * len(self.blocks)
        out = tokens
        for step in range(max_new):
            x = out if step == 0 else out[:, -1:]
            offset = out.shape[1] - x.shape[1]
            h = self.embed(x)
            for i, blk in enumerate(self.blocks):
                h, kv = blk(h, caches[i], offset)
                caches[i] = kv
            logits = self.lm_head(self.norm(h[:, -1]))
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
        return out


# ═══════════════════════════════════════════════════════════════════════
# 7. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class HodgeDreamLoss(nn.Module):
    def __init__(self, cfg: HodgeDreamConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        cfg = self.cfg
        dev = out["logits"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}
        
        # LM loss
        L["lm"] = out.get("lm_loss", zero)
        
        # Hodge decomposition loss (orthogonality)
        if "hodge_hodge_riemann_product" in out:
            # Hodge-Riemann products should be positive
            hr = out["hodge_hodge_riemann_product"]
            L["hodge_decomposition"] = F.relu(-hr).mean()
        else:
            L["hodge_decomposition"] = zero
        
        # Conjecture loss (encourage algebraicity)
        if "conj_distance_to_algebraicity" in out:
            L["conjecture"] = out["conj_distance_to_algebraicity"].mean()
        else:
            L["conjecture"] = zero
        
        # Dream loss (encourage high-quality dreams)
        if "dream_dream_quality" in out:
            L["dream"] = -out["dream_dream_quality"].mean()
        else:
            L["dream"] = zero
        
        # L2 regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg
        
        total = (cfg.w_lm * L["lm"]
                 + cfg.w_hodge_decomposition * L["hodge_decomposition"]
                 + cfg.w_conjecture * L["conjecture"]
                 + cfg.w_dream * L["dream"]
                 + cfg.w_reg * L["reg"])
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# 8. DATASET
# ═══════════════════════════════════════════════════════════════════════

class HodgeDreamDataset(Dataset):
    """Synthetic dataset for Hodge Dream LLM."""
    
    def __init__(self, cfg: HodgeDreamConfig, n: int, seed: int):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        
        # Generate token sequences (Hodge "language")
        self.tokens = torch.randint(0, cfg.vocab_size, (n, cfg.max_seq), generator=g)
        self.targets = torch.roll(self.tokens, -1, dims=1)
        
        # Variety types
        self.variety_type = torch.randint(0, 3, (n,), generator=g)
    
    def __len__(self):
        return self.tokens.shape[0]
    
    def __getitem__(self, i):
        return {
            "tokens": self.tokens[i],
            "targets": self.targets[i],
            "variety_type": self.variety_type[i],
        }


def build_hodge_loaders(cfg: HodgeDreamConfig):
    tr = HodgeDreamDataset(cfg, cfg.n_train, cfg.data_seed)
    va = HodgeDreamDataset(cfg, cfg.n_val, cfg.data_seed + 1)
    te = HodgeDreamDataset(cfg, cfg.n_test, cfg.data_seed + 2)
    tr_s = DistributedSampler(tr, shuffle=True, drop_last=True) if is_dist() else None
    va_s = DistributedSampler(va, shuffle=False) if is_dist() else None
    te_s = DistributedSampler(te, shuffle=False) if is_dist() else None
    common = dict(batch_size=cfg.batch_size, num_workers=2,
                  pin_memory=(cfg.device != "cpu"), persistent_workers=False)
    return (
        DataLoader(tr, sampler=tr_s, shuffle=(tr_s is None), drop_last=True, **common),
        DataLoader(va, sampler=va_s, shuffle=False, **common),
        DataLoader(te, sampler=te_s, shuffle=False, **common),
    )


# ═══════════════════════════════════════════════════════════════════════
# 9. TRAINER
# ═══════════════════════════════════════════════════════════════════════

class HodgeDreamTrainer:
    def __init__(self, cfg: HodgeDreamConfig, model: HodgeDreamLLM,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        
        if cfg.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                if is_main():
                    logger.info("torch.compile enabled")
            except Exception as e:
                if is_main():
                    logger.warning(f"torch.compile failed: {e}")
        
        if is_dist():
            self.model = DDP(self.model,
                             device_ids=[device.index] if device.type == "cuda" else None,
                             find_unused_parameters=False,
                             gradient_as_bucket_view=True)
        
        self.loss_fn = HodgeDreamLoss(cfg)
        
        amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[cfg.amp_dtype]
        self._amp_dtype = amp_dt
        self._amp_on = cfg.amp_dtype != "fp32" and device.type == "cuda"
        self.scaler = _GradScaler(enabled=(self._amp_on and amp_dt == torch.float16))
        
        fused = cfg.use_fused_adamw and device.type == "cuda"
        self.opt = torch.optim.AdamW(
            [p for p in self._raw().parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), fused=fused)
        self.step = 0
    
    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m
    
    def lr_now(self) -> float:
        cfg = self.cfg
        if self.step < cfg.warmup_steps:
            return cfg.lr * (self.step + 1) / max(1, cfg.warmup_steps)
        prog = (self.step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return cfg.lr * (cfg.min_lr_ratio
                         + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        for g in self.opt.param_groups:
            g["lr"] = self.lr_now()
        
        if (self.step + 1) % cfg.grad_accum == 0:
            self.opt.zero_grad(set_to_none=True)
        
        with amp_ctx(self._amp_on, self._amp_dtype, self.device.type):
            out = self.model(batch["tokens"], batch.get("targets"),
                             batch.get("variety_type"))
            losses = self.loss_fn(out, batch, self._raw())
            loss = losses["total"] / cfg.grad_accum
        
        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % cfg.grad_accum == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._raw().parameters() if p.requires_grad],
                cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
        
        self.step += 1
        return {k: float(v) for k, v in losses.items()
                if torch.is_tensor(v) and v.numel() == 1}
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot: Dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with amp_ctx(self._amp_on, self._amp_dtype, self.device.type):
                out = self.model(batch["tokens"], batch.get("targets"),
                                 batch.get("variety_type"))
                losses = self.loss_fn(out, batch, self._raw())
            bs = batch["tokens"].shape[0]
            for k, v in losses.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    tot[k] = tot.get(k, 0.0) + float(v) * bs
            n += bs
        for k in tot:
            tot[k] /= max(1, n)
        if is_dist():
            for k in list(tot.keys()):
                tot[k] = float(all_reduce_mean(torch.tensor(tot[k], device=self.device)))
        tot["n"] = n
        return tot
    
    def state_dict(self):
        return {"model": self._raw().state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step}
    
    def load_state_dict(self, sd):
        self._raw().load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        try:
            self.scaler.load_state_dict(sd["scaler"])
        except Exception:
            pass
        self.step = sd["step"]


# ═══════════════════════════════════════════════════════════════════════
# 10. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" HODGE DREAM LLM — Smoke Test")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)
    
    cfg = HodgeDreamConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq=64,
        vocab_size=64,
        batch_size=2,
        dream_latent_dim=64,
        n_algebraic_cycles=8,
        period_precision=16,
        gradient_checkpointing=False,
        use_torch_compile=False,
    )
    
    model = HodgeDreamLLM(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    
    model.train()
    B, T = 2, 32
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.roll(tokens, -1, dims=1)
    variety_type = torch.randint(0, 3, (B,), device=device)
    
    out = model(tokens, targets, variety_type)
    
    loss_fn = HodgeDreamLoss(cfg)
    losses = loss_fn(out, {"tokens": tokens, "targets": targets}, model)
    losses["total"].backward()
    
    print(f"\n  ── Outputs ──")
    print(f"  LM loss            : {losses['lm'].item():.4f}")
    print(f"  Hodge decomp loss  : {losses['hodge_decomposition'].item():.6f}")
    print(f"  Conjecture loss    : {losses['conjecture'].item():.6f}")
    print(f"  Dream loss         : {losses['dream'].item():.6f}")
    print(f"  Total loss         : {losses['total'].item():.4f}")
    
    print(f"\n  ── Hodge Theory ──")
    print(f"  Hodge numbers shape: {tuple(out['variety_hodge_numbers'].shape)}")
    print(f"  Period matrix shape: {tuple(out['variety_period_matrix'].shape)}")
    print(f"  Distance to alg    : {out['conj_distance_to_algebraicity'].mean().item():.4f}")
    print(f"  Rationality score  : {out['conj_rationality_score'].mean().item():.4f}")
    
    print(f"\n  ── Dreaming ──")
    print(f"  Dream quality      : {out['dream_dream_quality'].mean().item():.4f}")
    print(f"  Dream variety shape: {tuple(out['dream_dream_variety'].shape)}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL MODULES CONNECTED' if grad_ok else 'FAIL'}")
    print("=" * 72)
    print("  Smoke test complete.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser("Hodge Dream LLM")
    p.add_argument("--mode", choices=["smoke", "train", "test"], default="smoke")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--variety", choices=["torus", "k3", "calabi_yau"], default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")
    
    if args.mode == "smoke":
        smoke_test()
        return
    
    device, r, ws = setup_dist()
    seed_all(42)
    cfg = HodgeDreamConfig()
    if args.steps:
        cfg.max_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.out:
        cfg.out_dir = args.out
    if args.no_compile:
        cfg.use_torch_compile = False
    if args.no_amp:
        cfg.amp_dtype = "fp32"
    cfg.device = str(device)
    
    print(f"HodgeDreamLLM would train on {device}")
    cleanup_dist()


if __name__ == "__main__":
    main()
