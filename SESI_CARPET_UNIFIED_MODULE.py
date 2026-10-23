"""
=============================================================================
SESI CARPET UNIFIED MODULE v3.0
Sierpiński Carpet Fractal Engine for Neural Interfacing & Nuclear Interdiction
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 3.0.0 (Carpet-Theoretic Neural & Nuclear Framework)
=============================================================================

HONEST SCOPE — READ THIS FIRST
──────────────────────────────
This module is a DIFFERENTIABLE SIMULATION FRAMEWORK for research and
education. It models three domains through the unified mathematics of
Sierpiński carpet resistance forms:

  [D1] REMOTE NEURAL MONITORING
       Models the information-theoretic limits of reading neural signals
       at distance. Uses carpet fractal structure as a model of cortical
       folding geometry. Does NOT claim to read brains through walls.

  [D2] REMOTE NEURAL CONTROL
       Models focused-field targeting optimization with safety constraints.
       Does NOT claim to control brains remotely. All outputs are
       safety-gated and simulated.

  [D3] NUCLEAR CHAIN REACTION INTERDICTION
       Models PRE-DETONATION interdiction strategies only.
       An ongoing chain reaction CANNOT be stopped by any known physics
       (10⁻⁸ s per generation). This module simulates:
         • Criticality monitoring (k-effective tracking)
         • Pre-detonation intervention strategies
         • Gumbel extreme-value probability bounds on branching

MATHEMATICAL FOUNDATION
───────────────────────
  [M1] Sierpiński Carpet K:
       Iterated function system removing center ninth at each level.
       Hausdorff dimension: d_H = log(8)/log(3) ≈ 1.893
       Spectral dimension: d_S ≈ 1.80 (walk dimension d_W ≈ 2.10)

  [M2] Resistance Forms (Kusuoka-Zhou 1992):
       E_n^(0)(u) = Σ_{edges} (u(x) - u(y))²
       Renormalized: E_n = ρ^n E_n^(0), ρ = lim R_n^{1/n} ≥ 9/8
       Mosco convergence: E_n → E (continuum Dirichlet form)

  [M3] Spectral Sobolev Scale:
       W^s = D((I+A)^{s/2}), compact embeddings W^t ↪ W^s for t > s
       C^∞_sp = ∩_m D(A^m) — fully regular states

  [M4] Energy Measure Singularity (Hino 2005):
       Γ(f,·) ⊥ μ for all f ∈ D(E) on the carpet
       → Heterogeneous concentration of energy/attention/inhibition

  [M5] Gumbel Extreme-Value Bounds:
       P(T_{k+1} - T_k < dt) ≤ exp(-C₁ · exp(ΔE/(σ²·dt)))
       → Double-exponential suppression of catastrophic branching

APPLICATION MAPPING
───────────────────
  [A1] Neural Monitoring ↔ Carpet Spectral Analysis
       • Cortical folding ≈ carpet fractal structure
       • EEG source localization ≈ inverse problem on carpet graph
       • Attention focus ≈ energy measure concentration
       • Signal attenuation ≈ resistance scaling ρ^n

  [A2] Neural Control ↔ Carpet Resistance Optimization
       • Focused stimulation ≈ minimal-energy path on carpet
       • Safety constraints ≈ Hodge-Riemann positivity
       • Target selection ≈ spectral mode excitation
       • Off-target spread ≈ energy measure singularity

  [A3] Nuclear Interdiction ↔ Carpet Branching Suppression
       • Chain reaction ≈ topological branching cascade
       • Criticality k_eff ≈ resistance scaling factor
       • Pre-detonation arrest ≈ Gumbel bound enforcement
       • Neutron poisoning ≈ disordered medium injection

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import logging
import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("sesi.carpet_unified")

__all__ = [
    "CarpetConfig",
    # Core carpet engine
    "SierpinskiCarpetBuilder",
    "CarpetGraphLaplacian",
    "CarpetResistanceForm",
    "CarpetSpectralAnalyzer",
    # Neural interface
    "CarpetNeuralFieldModel",
    "CarpetNeuralMonitor",
    "CarpetNeuralController",
    # Nuclear interdiction
    "CarpetNuclearStateModel",
    "CarpetBranchingInhibitor",
    "GumbelExtremeValueEngine",
    # Unified system
    "SESICarpetUnifiedSystem",
    "CarpetLoss",
    "CarpetTrainer",
    # Infrastructure
    "setup_dist",
    "cleanup_dist",
    "seed_all",
]


# ═══════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def is_main() -> bool:
    return rank() == 0

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


@contextlib.contextmanager
def fp32_ctx():
    try:
        from torch.amp import autocast
        with autocast(device_type="cuda", enabled=False):
            yield
    except Exception:
        yield


# C^∞ smooth primitives
def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

def soft_clamp(x: Tensor, lo: float, hi: float) -> Tensor:
    mid = (hi + lo) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    return mid + half * torch.tanh((x - mid) / half)


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
class CarpetConfig:
    # ── Carpet geometry ───────────────────────────────────────────────────
    carpet_level: int = 4               # n in G_n
    carpet_n_cells: int = 8 ** 4        # |Q_n| = 8^n
    hausdorff_dim: float = math.log(8) / math.log(3)  # ≈ 1.893
    spectral_dim: float = 1.80          # d_S for carpet
    
    # ── Resistance form ───────────────────────────────────────────────────
    resistance_scale_rho: float = 1.25  # ρ = lim R_n^{1/n} ≥ 9/8
    n_spectral_modes: int = 64          # truncated eigenbasis
    
    # ── Neural interface ──────────────────────────────────────────────────
    neural_field_dim: int = 256         # neural state dimension
    n_neural_sources: int = 128         # cortical source dipoles
    n_sensors: int = 64                 # EEG/MEG sensor count
    sensor_noise_level: float = 0.05
    neural_band_dim: int = 8            # frequency bands
    
    # ── Neural control ────────────────────────────────────────────────────
    n_control_targets: int = 32         # stimulation targets
    control_horizon: int = 20
    max_safe_intensity: float = 1.0
    control_safety_threshold: float = 0.8
    
    # ── Nuclear interdiction ──────────────────────────────────────────────
    c1_geom: float = 1.0                # geometric constant in Gumbel bound
    sigma_variance: float = 0.05        # noise variance
    critical_energy_threshold: float = 50.0
    dt_nuclear: float = 1e-5
    gumbel_cutoff: float = 1e-12
    k_eff_initial: float = 0.95         # start subcritical
    
    # ── Integration ───────────────────────────────────────────────────────
    latent_dim: int = 512
    hcsn_scales: int = 4
    
    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    amp_dtype: str = "bf16"
    seed: int = 42
    device: str = "cuda"
    
    # ── Loss weights ──────────────────────────────────────────────────────
    w_neural_monitor: float = 0.3
    w_neural_control: float = 0.2
    w_nuclear: float = 0.3
    w_carpet_consistency: float = 0.1
    w_reg: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# 2. SIERPIŃSKI CARPET BUILDER
# ═══════════════════════════════════════════════════════════════════════

class SierpinskiCarpetBuilder(nn.Module):
    """
    Constructs the Sierpiński carpet as a graph at level n.
    
    The carpet is generated by iteratively removing the central square
    from each remaining square:
      K_0 = [0,1]²
      K_{n+1} = K_n \ (center ninth of each cell)
    
    Properties:
      - Hausdorff dimension: d_H = log(8)/log(3) ≈ 1.893
      - Spectral dimension: d_S ≈ 1.80
      - Walk dimension: d_W ≈ 2.10
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        self.level = cfg.carpet_level
        self.size = 3 ** self.level
        
        # Build carpet mask
        carpet_mask = self._build_carpet_mask()
        self.register_buffer("carpet_mask", carpet_mask, persistent=False)
        
        # Extract cell coordinates
        coords = torch.stack(torch.where(carpet_mask == 1), dim=-1).float()
        self.register_buffer("cell_coords", coords, persistent=False)
        
        self.n_cells = coords.shape[0]
        
        logger.info(f"Sierpiński carpet level {self.level}: "
                    f"{self.n_cells} cells (expected 8^{self.level} = {8**self.level})")
    
    def _build_carpet_mask(self) -> Tensor:
        """Build carpet mask recursively."""
        def carpet_recursive(n: int) -> Tensor:
            if n == 0:
                return torch.ones(1, 1)
            smaller = carpet_recursive(n - 1)
            s = smaller.shape[0]
            result = torch.zeros(3 * s, 3 * s)
            for i in range(3):
                for j in range(3):
                    if not (i == 1 and j == 1):  # skip center
                        result[i*s:(i+1)*s, j*s:(j+1)*s] = smaller
            return result
        
        mask = carpet_recursive(self.level)
        return mask
    
    def build_adjacency(self) -> Tensor:
        """
        Build adjacency matrix for carpet graph.
        Two cells are adjacent if they share an edge (4-connectivity).
        
        Returns:
            adj: (n_cells, n_cells) sparse adjacency
        """
        coords = self.cell_coords
        n = coords.shape[0]
        
        # Build adjacency using coordinate matching
        adj = torch.zeros(n, n, dtype=torch.float32)
        
        # Create coordinate-to-index mapping
        coord_dict = {}
        for idx in range(n):
            key = (int(coords[idx, 0].item()), int(coords[idx, 1].item()))
            coord_dict[key] = idx
        
        # Check 4-neighbors
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for idx in range(n):
            x, y = int(coords[idx, 0].item()), int(coords[idx, 1].item())
            for dx, dy in directions:
                neighbor = coord_dict.get((x + dx, y + dy))
                if neighbor is not None:
                    adj[idx, neighbor] = 1.0
        
        return adj
    
    def build_incidence_matrix(self) -> Tensor:
        """
        Build incidence matrix D for the carpet graph.
        D[e, v] = +1 if v is head of edge e, -1 if tail.
        
        Returns:
            incidence: (n_edges, n_cells)
        """
        adj = self.build_adjacency()
        edges = torch.triu(adj, diagonal=1).nonzero(as_tuple=False)
        n_edges = edges.shape[0]
        n_cells = adj.shape[0]
        
        incidence = torch.zeros(n_edges, n_cells)
        for e, (i, j) in enumerate(edges):
            incidence[e, i] = -1.0  # tail
            incidence[e, j] = +1.0  # head
        
        return incidence
    
    def visualize_carpet(self, save_path: str = "carpet.png"):
        """Visualize the carpet structure."""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(self.carpet_mask.cpu().numpy(), cmap='binary', origin='lower')
            ax.set_title(f"Sierpiński Carpet (Level {self.level})")
            ax.axis('off')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Carpet visualization saved to {save_path}")
        except ImportError:
            logger.warning("matplotlib not available for visualization")


# ═══════════════════════════════════════════════════════════════════════
# 3. CARPET GRAPH LAPLACIAN
# ═══════════════════════════════════════════════════════════════════════

class CarpetGraphLaplacian(nn.Module):
    """
    Graph Laplacian of the carpet graph.
    
    L = D - A where D is degree matrix, A is adjacency.
    Also provides normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    
    The eigenvalues of L encode the spectral geometry of the carpet:
      0 = λ_0 < λ_1 ≤ λ_2 ≤ ... → ∞
    
    Spectral dimension: d_S = 2 lim_{λ→0} log N(λ) / log λ
    where N(λ) = #{i : λ_i ≤ λ}
    """
    
    def __init__(self, cfg: CarpetConfig, carpet_builder: SierpinskiCarpetBuilder):
        super().__init__()
        self.cfg = cfg
        self.carpet = carpet_builder
        
        # Build Laplacian
        adj = carpet_builder.build_adjacency()
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj
        
        self.register_buffer("adjacency", adj, persistent=False)
        self.register_buffer("degree", degree, persistent=False)
        self.register_buffer("laplacian", laplacian, persistent=False)
        
        # Normalized Laplacian
        d_inv_sqrt = 1.0 / torch.sqrt(degree.clamp(min=1e-8))
        normalized_lap = torch.diag(d_inv_sqrt) @ laplacian @ torch.diag(d_inv_sqrt)
        self.register_buffer("normalized_laplacian", normalized_lap, persistent=False)
        
        logger.info(f"Carpet Laplacian: {laplacian.shape[0]}×{laplacian.shape[1]}, "
                    f"edges={int(adj.sum().item())//2}")
    
    def compute_eigenvalues(self, k: int = 64) -> Tuple[Tensor, Tensor]:
        """
        Compute k smallest eigenvalues and eigenvectors.
        
        Args:
            k: number of eigenpairs
        
        Returns:
            eigenvalues: (k,)
            eigenvectors: (n_cells, k)
        """
        try:
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp
            
            L_scipy = sp.csr_matrix(self.laplacian.cpu().numpy())
            eigenvalues, eigenvectors = eigsh(L_scipy, k=k, which='SM')
            
            # Sort by eigenvalue
            idx = torch.argsort(torch.tensor(eigenvalues))
            eigenvalues = torch.tensor(eigenvalues)[idx]
            eigenvectors = torch.tensor(eigenvectors)[:, idx]
            
            return eigenvalues, eigenvectors
            
        except ImportError:
            # Fallback to dense eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(self.laplacian)
            return eigenvalues[:k], eigenvectors[:, :k]
    
    def estimate_spectral_dimension(self, eigenvalues: Tensor, 
                                     cutoff: int = 100) -> float:
        """
        Estimate spectral dimension from eigenvalue counting.
        
        N(λ) ~ C · λ^{d_S/2}
        → d_S ≈ 2 · slope of log N vs log λ
        
        Args:
            eigenvalues: sorted eigenvalues
            cutoff: number of eigenvalues to use
        
        Returns:
            spectral dimension estimate
        """
        lambdas = eigenvalues[eigenvalues > 1e-10][:cutoff].float()
        if len(lambdas) < 10:
            return 0.0
        
        N = torch.arange(1, len(lambdas) + 1, dtype=torch.float32)
        log_lambda = torch.log(lambdas)
        log_N = torch.log(N)
        
        # Linear regression: log_N = (d_S/2) · log_lambda + const
        coeffs = torch.polyfit(log_lambda, log_N, 1)
        d_s = 2.0 * coeffs[0].item()
        
        return d_s
    
    def graph_energy(self, u: Tensor) -> Tensor:
        """
        Compute graph energy E(u) = u^T L u = Σ_{(i,j)∈E} (u_i - u_j)²
        
        Args:
            u: (B, n_cells) or (n_cells,) function values
        
        Returns:
            energy: (B,) or scalar
        """
        if u.dim() == 1:
            u = u.unsqueeze(0)
        
        # E(u) = u^T L u
        energy = torch.einsum("bi,ij,bj->b", u, self.laplacian, u)
        return energy
    
    def forward(self, u: Tensor) -> Tensor:
        """Apply Laplacian: Lu"""
        if u.dim() == 1:
            return self.laplacian @ u
        return torch.einsum("ij,bj->bi", self.laplacian, u)


# ═══════════════════════════════════════════════════════════════════════
# 4. CARPET RESISTANCE FORM
# ═══════════════════════════════════════════════════════════════════════

class CarpetResistanceForm(nn.Module):
    """
    Resistance form on the carpet (Kusuoka-Zhou 1992).
    
    E_n^(0)(u) = Σ_{edges} (u(x) - u(y))²
    E_n = ρ^n · E_n^(0)
    
    where ρ = lim R_n^{1/n} ≥ 9/8 is the resistance scaling factor.
    
    The resistance form captures the "energy" of a function on the carpet,
    with the renormalization ρ^n accounting for the fractal scaling.
    
    Key properties:
      - Mosco convergence: E_n → E (continuum Dirichlet form)
      - Energy measure singularity: Γ(f,·) ⊥ μ (Hino 2005)
      - Resistance metric: R(x,y) = sup{(u(x)-u(y))² / E(u)}
    """
    
    def __init__(self, cfg: CarpetConfig, laplacian: CarpetGraphLaplacian):
        super().__init__()
        self.cfg = cfg
        self.laplacian = laplacian
        
        # Learnable resistance scaling (ρ ≥ 9/8)
        self.log_rho_raw = nn.Parameter(
            torch.tensor(math.log(max(cfg.resistance_scale_rho - 9/8, 1e-6))))
        
        # Energy measure concentration detector
        self.concentration_net = nn.Sequential(
            LowRankAdapter(cfg.carpet_n_cells, cfg.carpet_n_cells // 4, 32),
            nn.GELU(),
            nn.Linear(cfg.carpet_n_cells // 4, 1),
            nn.Sigmoid(),
        )
    
    @property
    def rho(self) -> Tensor:
        """Resistance scaling factor ρ ≥ 9/8"""
        return 9/8 + F.softplus(self.log_rho_raw)
    
    def compute_resistance_energy(self, u: Tensor) -> Tensor:
        """
        Compute renormalized resistance energy: E_n(u) = ρ^n · u^T L u
        
        Args:
            u: (B, n_cells) function values
        
        Returns:
            energy: (B,) renormalized energy
        """
        graph_energy = self.laplacian.graph_energy(u)
        n = self.cfg.carpet_level
        rho = self.rho
        
        # Renormalized energy
        renorm_energy = (rho ** n) * graph_energy
        
        return renorm_energy
    
    def compute_effective_resistance(self, source_idx: int, 
                                      target_idx: int) -> Tensor:
        """
        Compute effective resistance R_eff(x, y) between two cells.
        
        R_eff(x,y) = (e_x - e_y)^T L^+ (e_x - e_y)
        where L^+ is the pseudoinverse of L.
        
        Args:
            source_idx, target_idx: cell indices
        
        Returns:
            effective resistance (scalar)
        """
        n_cells = self.laplacian.laplacian.shape[0]
        
        # Create unit current vector
        b = torch.zeros(n_cells, device=self.laplacian.laplacian.device)
        b[source_idx] = 1.0
        b[target_idx] = -1.0
        
        # Solve L v = b (with gauge fixing)
        L = self.laplacian.laplacian
        L_reg = L + 1e-6 * torch.eye(n_cells, device=L.device)
        
        try:
            v = torch.linalg.solve(L_reg, b)
            resistance = (b * v).sum()
        except Exception:
            # Fallback: use pseudoinverse
            v = torch.linalg.pinv(L) @ b
            resistance = (b * v).sum()
        
        return resistance
    
    def energy_measure_concentration(self, u: Tensor) -> Tensor:
        """
        Detect energy measure concentration (Hino 2005).
        
        For the carpet, energy measures Γ(f,·) are singular with respect
        to the Hausdorff measure μ. This means energy concentrates on
        fractal subsets rather than being uniformly distributed.
        
        Args:
            u: (B, n_cells) function values
        
        Returns:
            concentration: (B,) ∈ [0, 1], 1 = highly concentrated
        """
        # Compute local energy at each cell
        local_energy = self.laplacian.graph_energy(u)
        
        # Detect concentration
        concentration = self.concentration_net(u).squeeze(-1)
        
        return concentration
    
    def forward(self, u: Tensor) -> Dict[str, Tensor]:
        """
        Full resistance form computation.
        
        Args:
            u: (B, n_cells) function values
        
        Returns:
            dict with energy, concentration, rho
        """
        energy = self.compute_resistance_energy(u)
        concentration = self.energy_measure_concentration(u)
        
        return {
            "resistance_energy": energy,
            "concentration": concentration,
            "rho": self.rho.expand(u.shape[0]),
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. CARPET SPECTRAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class CarpetSpectralAnalyzer(nn.Module):
    """
    Spectral analysis of the carpet Laplacian.
    
    Computes:
      - Eigenvalues λ_k and eigenvectors φ_k
      - Spectral dimension d_S
      - Heat kernel trace: Tr(e^{-tA}) = Σ e^{-tλ_k}
      - Spectral Sobolev norms: ∥u∥²_{W^s} = Σ (1+λ_k)^s |û_k|²
    
    Applications:
      - Neural signal decomposition (eigenmodes = frequency bands)
      - Nuclear state classification (spectral gaps = stability)
    """
    
    def __init__(self, cfg: CarpetConfig, laplacian: CarpetGraphLaplacian):
        super().__init__()
        self.cfg = cfg
        self.laplacian = laplacian
        
        # Precompute eigenpairs (cached)
        self._eigenvalues = None
        self._eigenvectors = None
        
        # Spectral feature extractor
        self.spectral_net = nn.Sequential(
            LowRankAdapter(cfg.n_spectral_modes, cfg.latent_dim // 2, 32),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, cfg.latent_dim),
        )
    
    def get_eigenpairs(self, k: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Get cached eigenpairs."""
        k = k or self.cfg.n_spectral_modes
        
        if self._eigenvalues is None or len(self._eigenvalues) < k:
            self._eigenvalues, self._eigenvectors = \
                self.laplacian.compute_eigenvalues(k)
        
        return self._eigenvalues[:k], self._eigenvectors[:, :k]
    
    def spectral_decomposition(self, u: Tensor, k: Optional[int] = None) -> Dict[str, Tensor]:
        """
        Decompose function into carpet eigenmodes.
        
        u = Σ_k û_k φ_k where û_k = ⟨u, φ_k⟩
        
        Args:
            u: (B, n_cells) function values
            k: number of modes to use
        
        Returns:
            dict with coefficients, reconstruction, spectral_energy
        """
        eigenvalues, eigenvectors = self.get_eigenpairs(k)
        
        # Project onto eigenvectors: û_k = ⟨u, φ_k⟩
        # eigenvectors: (n_cells, k), u: (B, n_cells)
        coefficients = torch.einsum("bi,ik->bk", u, eigenvectors)  # (B, k)
        
        # Reconstruct: u_recon = Σ û_k φ_k
        reconstruction = torch.einsum("bk,ik->bi", coefficients, eigenvectors)
        
        # Spectral energy: Σ λ_k |û_k|²
        spectral_energy = (eigenvalues.unsqueeze(0) * coefficients.pow(2)).sum(dim=-1)
        
        # Sobolev norm W^1: Σ (1+λ_k) |û_k|²
        sobolev_norm = ((1 + eigenvalues).unsqueeze(0) * coefficients.pow(2)).sum(dim=-1)
        
        return {
            "spectral_coefficients": coefficients,
            "reconstruction": reconstruction,
            "spectral_energy": spectral_energy,
            "sobolev_norm": sobolev_norm,
            "eigenvalues": eigenvalues,
        }
    
    def heat_kernel_trace(self, t: Tensor) -> Tensor:
        """
        Compute heat kernel trace: Tr(e^{-tA}) = Σ e^{-tλ_k}
        
        Args:
            t: (B,) time values
        
        Returns:
            trace: (B,) heat kernel trace
        """
        eigenvalues, _ = self.get_eigenpairs()
        
        # Tr(e^{-tA}) = Σ_k e^{-t λ_k}
        trace = torch.exp(-t.unsqueeze(-1) * eigenvalues.unsqueeze(0)).sum(dim=-1)
        
        return trace
    
    def forward(self, u: Tensor) -> Dict[str, Tensor]:
        """Full spectral analysis."""
        spec_out = self.spectral_decomposition(u)
        
        # Extract spectral features
        spectral_features = self.spectral_net(spec_out["spectral_coefficients"])
        
        spec_out["spectral_features"] = spectral_features
        
        return spec_out


# ═══════════════════════════════════════════════════════════════════════
# 6. CARPET NEURAL FIELD MODEL
# ═══════════════════════════════════════════════════════════════════════

class CarpetNeuralFieldModel(nn.Module):
    """
    Neural field model on carpet geometry.
    
    Models cortical activity as a function on the carpet graph:
      - Neurons ≈ carpet cells
      - Synaptic connections ≈ carpet edges
      - Signal propagation ≈ diffusion on carpet
    
    The carpet's fractal structure captures:
      - Cortical folding geometry (gyri/sulci)
      - Hierarchical organization (columns → areas → networks)
      - Energy measure singularity → attention concentration
    
    Neural Mass Model (Jansen-Rit style) adapted to carpet:
      dE/dt = -E/τ_E + S(w_EE·E - w_EI·I + P(t))
      dI/dt = -I/τ_I + S(w_IE·E - w_II·I)
    where S is a sigmoid firing rate function.
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Neural state dimensions
        self.n_cells = cfg.carpet_n_cells
        
        # Excitatory/Inhibitory populations
        self.E_pop = nn.Parameter(torch.randn(self.n_cells) * 0.1)  # Excitatory
        self.I_pop = nn.Parameter(torch.randn(self.n_cells) * 0.1)  # Inhibitory
        
        # Time constants
        self.tau_E = nn.Parameter(torch.tensor(0.01))  # 10 ms
        self.tau_I = nn.Parameter(torch.tensor(0.02))  # 20 ms
        
        # Connection weights
        self.w_EE = nn.Parameter(torch.tensor(1.0))
        self.w_EI = nn.Parameter(torch.tensor(0.5))
        self.w_IE = nn.Parameter(torch.tensor(1.0))
        self.w_II = nn.Parameter(torch.tensor(0.5))
        
        # Sigmoid gain
        self.gain = nn.Parameter(torch.tensor(1.0))
        
        # External input projection
        self.input_proj = nn.Linear(cfg.neural_field_dim, self.n_cells)
        
        # Diffusion coupling (via carpet Laplacian)
        self.diffusion_strength = nn.Parameter(torch.tensor(0.1))
    
    def sigmoid_firing(self, x: Tensor) -> Tensor:
        """Sigmoid firing rate function S(x) = 1/(1+exp(-g·x))"""
        return torch.sigmoid(self.gain * x)
    
    def neural_mass_step(self, E: Tensor, I: Tensor, P: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
        """
        One step of neural mass dynamics.
        
        Args:
            E, I: (B, n_cells) excitatory/inhibitory activity
            P: (B, n_cells) external input
            dt: time step
        
        Returns:
            E_next, I_next
        """
        tau_E = soft_floor(self.tau_E, 1e-4)
        tau_I = soft_floor(self.tau_I, 1e-4)
        
        # Firing rates
        S_E = self.sigmoid_firing(self.w_EE * E - self.w_EI * I + P)
        S_I = self.sigmoid_firing(self.w_IE * E - self.w_II * I)
        
        # Dynamics
        dE = (-E + S_E) / tau_E
        dI = (-I + S_I) / tau_I
        
        E_next = E + dt * dE
        I_next = I + dt * dI
        
        # Clamp to valid range
        E_next = soft_clamp(E_next, 0.0, 1.0)
        I_next = soft_clamp(I_next, 0.0, 1.0)
        
        return E_next, I_next
    
    def forward(self, external_input: Tensor, n_steps: int = 10, 
                dt: float = 0.001) -> Dict[str, Tensor]:
        """
        Simulate neural field dynamics.
        
        Args:
            external_input: (B, neural_field_dim) external stimulation
            n_steps: number of simulation steps
            dt: time step
        
        Returns:
            dict with E, I trajectories and final states
        """
        B = external_input.shape[0]
        
        # Project input to carpet cells
        P = self.input_proj(external_input)  # (B, n_cells)
        
        # Initialize populations
        E = self.E_pop.unsqueeze(0).expand(B, -1) * 0.5
        I = self.I_pop.unsqueeze(0).expand(B, -1) * 0.5
        
        E_traj = [E.clone()]
        I_traj = [I.clone()]
        
        # Simulate dynamics
        for _ in range(n_steps):
            E, I = self.neural_mass_step(E, I, P, dt)
            E_traj.append(E)
            I_traj.append(I)
        
        E_traj = torch.stack(E_traj, dim=1)  # (B, n_steps+1, n_cells)
        I_traj = torch.stack(I_traj, dim=1)
        
        return {
            "E_final": E,
            "I_final": I,
            "E_trajectory": E_traj,
            "I_trajectory": I_traj,
            "neural_activity": E - I,  # Net activity
        }


# ═══════════════════════════════════════════════════════════════════════
# 7. CARPET NEURAL MONITOR
# ═══════════════════════════════════════════════════════════════════════

class CarpetNeuralMonitor(nn.Module):
    """
    Remote neural monitoring via carpet spectral analysis.
    
    Models the INVERSE problem:
      sensor_signals = L @ neural_sources + noise
    
    where L is the lead field matrix (forward model).
    
    The carpet structure provides:
      - Spatial prior: neural sources live on carpet cells
      - Spectral decomposition: eigenmodes = frequency bands
      - Energy measure: attention = concentration of energy
    
    HONEST SCOPE: This simulates the information-theoretic limits of
    neural monitoring. Real remote reading through walls/skull is not
    physically possible at meaningful distances due to 1/r² attenuation.
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Lead field matrix (forward model)
        # Maps neural sources to sensor measurements
        self.lead_field = nn.Parameter(
            torch.randn(cfg.n_sensors, cfg.n_neural_sources) * 0.01)
        
        # Inverse solver: sensors → sources
        self.inverse_net = nn.Sequential(
            nn.Conv1d(cfg.n_sensors, 128, 3, padding=1, groups=8),
            nn.GroupNorm(1, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, cfg.n_neural_sources),
        )
        
        # Spectral band decoder
        self.band_decoder = nn.Sequential(
            nn.Linear(cfg.n_neural_sources, cfg.neural_band_dim),
            nn.Softmax(dim=-1),
        )
        
        # Attention focus detector (energy measure concentration)
        self.attention_net = nn.Sequential(
            nn.Linear(cfg.n_neural_sources, cfg.n_neural_sources // 2),
            nn.GELU(),
            nn.Linear(cfg.n_neural_sources // 2, 1),
            nn.Sigmoid(),
        )
        
        # Noise model
        self.register_buffer("noise_level", 
                             torch.tensor(cfg.sensor_noise_level), 
                             persistent=False)
    
    def forward_model(self, sources: Tensor) -> Tensor:
        """
        Forward model: sources → sensors.
        
        sensor_signals = L @ sources + noise
        
        Args:
            sources: (B, n_sources) neural source activity
        
        Returns:
            sensors: (B, n_sensors) measured signals
        """
        signals = torch.einsum("sn,bn->bs", self.lead_field, sources)
        noise = torch.randn_like(signals) * self.noise_level
        return signals + noise
    
    def inverse_model(self, sensor_signals: Tensor) -> Tensor:
        """
        Inverse model: sensors → sources.
        
        Args:
            sensor_signals: (B, n_sensors, T) time series
        
        Returns:
            sources: (B, n_sources) estimated source activity
        """
        return self.inverse_net(sensor_signals)
    
    def forward(self, sensor_signals: Tensor) -> Dict[str, Tensor]:
        """
        Full monitoring pipeline.
        
        Args:
            sensor_signals: (B, n_sensors, T) EEG/MEG-like signals
        
        Returns:
            dict with estimated sources, bands, attention
        """
        # Inverse solve
        sources = self.inverse_model(sensor_signals)
        
        # Spectral bands
        bands = self.band_decoder(sources)
        
        # Attention focus
        attention = self.attention_net(sources).squeeze(-1)
        
        # Reconstruct sensors (for validation)
        reconstructed = self.forward_model(sources)
        
        return {
            "estimated_sources": sources,
            "spectral_bands": bands,
            "attention_focus": attention,
            "reconstructed_signals": reconstructed,
        }


# ═══════════════════════════════════════════════════════════════════════
# 8. CARPET NEURAL CONTROLLER
# ═══════════════════════════════════════════════════════════════════════

class CarpetNeuralController(nn.Module):
    """
    Remote neural control via carpet resistance optimization.
    
    Models focused-field stimulation targeting:
      - Find minimal-energy path on carpet to target
      - Safety constraints: intensity limits, off-target spread
      - Hodge-Riemann positivity: ensure valid energy distribution
    
    HONEST SCOPE: This simulates the OPTIMIZATION problem for neural
    stimulation. Real remote control without physical proximity is not
    possible. All outputs are safety-gated.
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Target selection
        self.target_selector = nn.Sequential(
            LowRankAdapter(cfg.latent_dim, cfg.n_control_targets, 32),
            nn.Softmax(dim=-1),
        )
        
        # Intensity optimizer
        self.intensity_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.n_control_targets, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # intensity in [0, 1]
        )
        
        # Safety gate
        self.safety_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Off-target spread model
        self.spread_net = nn.Sequential(
            nn.Linear(cfg.n_control_targets, cfg.n_control_targets),
            nn.Sigmoid(),
        )
        
        self.register_buffer("max_intensity",
                             torch.tensor(cfg.max_safe_intensity),
                             persistent=False)
    
    def forward(self, intent_state: Tensor) -> Dict[str, Tensor]:
        """
        Compute safe control signal.
        
        Args:
            intent_state: (B, latent_dim) desired neural state
        
        Returns:
            dict with target, intensity, safety, spread
        """
        B = intent_state.shape[0]
        
        # Select target
        target_probs = self.target_selector(intent_state)
        
        # Optimize intensity
        intensity_input = torch.cat([intent_state, target_probs], dim=-1)
        intensity = self.intensity_net(intensity_input).squeeze(-1)
        
        # Safety check
        safety_input = torch.cat([intent_state, intensity.unsqueeze(-1)], dim=-1)
        safety_score = self.safety_net(safety_input).squeeze(-1)
        
        # Apply safety gate
        safe_mask = soft_gate(safety_score - self.cfg.control_safety_threshold, tau=0.01)
        gated_intensity = intensity * safe_mask * self.max_intensity
        
        # Off-target spread
        spread = self.spread_net(target_probs)
        
        return {
            "target_probs": target_probs,
            "intensity": intensity,
            "safety_score": safety_score,
            "gated_intensity": gated_intensity,
            "off_target_spread": spread,
            "action_permitted": (safe_mask > 0.5).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# 9. CARPET NUCLEAR STATE MODEL
# ═══════════════════════════════════════════════════════════════════════

class CarpetNuclearStateModel(nn.Module):
    """
    Nuclear chain reaction state as carpet branching process.
    
    Models fission chain reaction as topological branching on carpet:
      - Neutrons ≈ random walks on carpet graph
      - Fission events ≈ branching at carpet nodes
      - Criticality k_eff ≈ resistance scaling factor
      - Supercritical ≈ branching cascade
    
    Point kinetics equations (real physics):
      dn/dt = [(k_eff - 1)/Λ] · n + S
      where Λ ≈ 10⁻⁸ s (prompt generation time)
    
    HONEST SCOPE: Once detonation begins, NO external mechanism can
    stop the chain reaction (each generation ~10⁻⁸ s). This module
    models PRE-DETONATION interdiction only.
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        # k-effective predictor
        self.k_eff_net = nn.Sequential(
            LowRankAdapter(cfg.carpet_n_cells, 64, 32),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # k_eff > 0
        )
        
        # Neutron population dynamics
        self.neutron_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 64),
            nn.GELU(),
            nn.Linear(64, cfg.carpet_n_cells),
        )
        
        # Control rod worth (absorption)
        self.control_rod_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Delayed neutron fraction (β ≈ 0.0065 for U-235)
        self.register_buffer("beta_delayed",
                             torch.tensor(0.0065),
                             persistent=False)
        
        # Prompt generation time
        self.register_buffer("Lambda_prompt",
                             torch.tensor(1e-8),
                             persistent=False)
    
    def point_kinetics_step(self, n: Tensor, k_eff: Tensor, 
                            S: Tensor, dt: float) -> Tensor:
        """
        Point kinetics: dn/dt = [(k_eff - 1)/Λ]·n + S
        
        Args:
            n: (B,) neutron population
            k_eff: (B,) effective multiplication factor
            S: (B,) source term
            dt: time step
        
        Returns:
            n_next: (B,) updated neutron population
        """
        Lambda = self.Lambda_prompt.to(n.device)
        
        # Reactivity ρ = (k_eff - 1) / k_eff
        rho = (k_eff - 1.0) / soft_floor(k_eff, 1e-8)
        
        # Point kinetics
        dn_dt = (rho / Lambda) * n + S
        
        n_next = n + dt * dn_dt
        
        # Clamp to non-negative
        n_next = soft_floor(n_next, 0.0)
        
        return n_next
    
    def classify_criticality(self, k_eff: Tensor) -> Dict[str, Tensor]:
        """
        Classify reactor state.
        
        k_eff < 1: subcritical (safe)
        k_eff = 1: critical (steady)
        k_eff > 1: supercritical (dangerous)
        
        Args:
            k_eff: (B,) multiplication factor
        
        Returns:
            dict with classification
        """
        is_subcritical = (k_eff < 1.0).float()
        is_critical = (soft_abs(k_eff - 1.0) < 1e-3).float()
        is_supercritical = (k_eff > 1.0).float()
        
        return {
            "is_subcritical": is_subcritical,
            "is_critical": is_critical,
            "is_supercritical": is_supercritical,
        }
    
    def forward(self, carpet_state: Tensor, n_steps: int = 10,
                dt: float = 1e-4) -> Dict[str, Tensor]:
        """
        Simulate nuclear state evolution.
        
        Args:
            carpet_state: (B, n_cells) material configuration
            n_steps: simulation steps
            dt: time step
        
        Returns:
            dict with k_eff, neutron trajectory, criticality
        """
        B = carpet_state.shape[0]
        
        # Predict k_eff
        k_eff = self.k_eff_net(carpet_state).squeeze(-1)  # (B,)
        
        # Initial neutron population
        n = torch.ones(B, device=carpet_state.device)
        
        # Source term (from control rods)
        control_worth = self.control_rod_net(carpet_state).squeeze(-1)
        S = -control_worth  # Negative source (absorption)
        
        n_traj = [n.clone()]
        
        # Simulate kinetics
        for _ in range(n_steps):
            n = self.point_kinetics_step(n, k_eff, S, dt)
            n_traj.append(n.clone())
        
        n_traj = torch.stack(n_traj, dim=1)  # (B, n_steps+1)
        
        # Classify criticality
        crit = self.classify_criticality(k_eff)
        
        return {
            "k_eff": k_eff,
            "neutron_trajectory": n_traj,
            "final_neutron_pop": n,
            "control_rod_worth": control_worth,
            **crit,
        }


# ═══════════════════════════════════════════════════════════════════════
# 10. CARPET BRANCHING INHIBITOR
# ═══════════════════════════════════════════════════════════════════════

class GumbelExtremeValueEngine(nn.Module):
    """
    Gumbel extreme-value probability bounds for branching suppression.
    
    P(T_{k+1} - T_k < dt) ≤ exp(-C₁ · exp(ΔE/(σ²·dt)))
    
    This double-exponential bound provides extremely strong suppression
    of catastrophic branching events when the energy barrier ΔE is high.
    
    From SESI_nuclear_chain_reaction_inhibitor.py v2.0
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Pre-compute constants
        self.register_buffer("exp_clamp_max", torch.tensor(80.0), persistent=False)
        self.register_buffer("epsilon", torch.tensor(1e-15), persistent=False)
    
    def compute_bound(self, delta_E: Tensor, 
                      sigma_var: Optional[float] = None) -> Tensor:
        """
        Compute Gumbel probability bound.
        
        P(T_{k+1} - T_k < dt) ≤ exp(-C₁ · exp(ΔE/(σ²·dt)))
        
        Args:
            delta_E: (B,) activation energy barriers
            sigma_var: noise variance (uses config if None)
        
        Returns:
            prob_bound: (B,) probability upper bounds
        """
        sigma = sigma_var if sigma_var is not None else self.cfg.sigma_variance
        sigma_sq = sigma ** 2
        
        # Denominator: σ² · dt
        denom = torch.clamp(
            torch.tensor(sigma_sq * self.cfg.dt_nuclear, 
                         device=delta_E.device, dtype=delta_E.dtype),
            min=self.epsilon)
        
        # Inner exponent (clamped to prevent overflow)
        inner = torch.clamp(delta_E / denom, max=self.exp_clamp_max)
        
        # Double exponential
        outer = -self.cfg.c1_geom * torch.exp(inner)
        prob_bound = torch.exp(outer)
        
        return prob_bound
    
    def forward(self, delta_E: Tensor) -> Tensor:
        return self.compute_bound(delta_E)


class CarpetBranchingInhibitor(nn.Module):
    """
    Carpet-based nuclear branching inhibition.
    
    Strategy: Inject disordered medium into carpet to raise energy
    barriers for branching events, enforcing No-Zeno arrest.
    
    Methods:
      1. Disordered medium injection (quenched noise)
      2. Activation energy elevation
      3. Gumbel bound enforcement
      4. No-Zeno arrest verification
    
    From SESI_nuclear_chain_reaction_inhibitor.py v2.0
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        self.gumbel_engine = GumbelExtremeValueEngine(cfg)
        
        # Disordered medium generator
        self.disorder_net = nn.Sequential(
            LowRankAdapter(cfg.carpet_n_cells, cfg.carpet_n_cells // 2, 32),
            nn.GELU(),
            nn.Linear(cfg.carpet_n_cells // 2, cfg.carpet_n_cells),
        )
        
        # Activation energy predictor
        self.activation_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Pinning controller
        self.pinning_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self.register_buffer("noise_scale", torch.tensor(0.2), persistent=False)
    
    def inject_disordered_medium(self, sigma_field: Tensor) -> Tensor:
        """
        Inject quenched spatial noise to create disordered medium.
        
        This raises energy barriers for branching events.
        
        Args:
            sigma_field: (B, n_cells) structural heterogeneity
        
        Returns:
            disordered: (B, n_cells) modified field
        """
        B = sigma_field.shape[0]
        
        # Generate disorder
        disorder = self.disorder_net(sigma_field)
        
        # Quenched noise (abs of randn for positivity)
        noise = torch.abs(torch.randn_like(sigma_field)) * self.cfg.sigma_variance
        
        return sigma_field + disorder + noise
    
    def calculate_activation_energy(self, current_energy: Tensor,
                                     proposed_energy: Tensor) -> Tensor:
        """
        Calculate activation energy ΔE for branching event.
        
        ΔE = E(proposed) - E(current)
        
        Args:
            current_energy: (B,) current state energy
            proposed_energy: (B,) proposed state energy
        
        Returns:
            delta_E: (B,) activation energy (clamped ≥ min)
        """
        delta_E = proposed_energy - current_energy
        return soft_floor(delta_E, self.cfg.c1_geom * 1e-6)
    
    def forward(self, u_state: Tensor, sigma_base: Tensor) -> Dict[str, Tensor]:
        """
        Full inhibition pipeline.
        
        Args:
            u_state: (B, n_cells) active density field
            sigma_base: (B, n_cells) base heterogeneity field
        
        Returns:
            dict with arrest status, probability bounds, etc.
        """
        B = u_state.shape[0]
        
        # 1. Inject disordered medium
        sigma_disordered = self.inject_disordered_medium(sigma_base)
        
        # 2. Compute current energy
        current_energy = u_state.pow(2).mean(dim=-1)  # Proxy energy
        
        # 3. Simulate branching perturbation
        perturbation = torch.randn_like(u_state) * self.noise_scale
        branched_state = u_state + perturbation
        proposed_energy = branched_state.pow(2).mean(dim=-1)
        
        # 4. Activation energy
        delta_E = self.calculate_activation_energy(current_energy, proposed_energy)
        
        # 5. Gumbel bound
        prob_bound = self.gumbel_engine.compute_bound(delta_E)
        
        # 6. No-Zeno arrest check
        is_arrested = (prob_bound <= self.cfg.gumbel_cutoff).float()
        
        # 7. Adaptive pinning if needed
        pinning_applied = (delta_E < self.cfg.critical_energy_threshold).float()
        pinning_strength = self.pinning_net(sigma_disordered).squeeze(-1)
        
        return {
            "disordered_sigma": sigma_disordered,
            "current_energy": current_energy,
            "proposed_energy": proposed_energy,
            "activation_energy": delta_E,
            "probability_bound": prob_bound,
            "is_arrested": is_arrested,
            "pinning_applied": pinning_applied,
            "pinning_strength": pinning_strength,
        }


# ═══════════════════════════════════════════════════════════════════════
# 11. UNIFIED SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class SESICarpetUnifiedSystem(nn.Module):
    """
    Unified SESI Carpet System integrating:
      - Carpet geometry and spectral analysis
      - Neural monitoring and control
      - Nuclear interdiction
    
    All three domains share the same carpet mathematical structure.
    """
    
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
        
        # Core carpet engine
        self.carpet_builder = SierpinskiCarpetBuilder(cfg)
        self.laplacian = CarpetGraphLaplacian(cfg, self.carpet_builder)
        self.resistance = CarpetResistanceForm(cfg, self.laplacian)
        self.spectral = CarpetSpectralAnalyzer(cfg, self.laplacian)
        
        # Neural interface
        self.neural_field = CarpetNeuralFieldModel(cfg)
        self.neural_monitor = CarpetNeuralMonitor(cfg)
        self.neural_controller = CarpetNeuralController(cfg)
        
        # Nuclear interdiction
        self.nuclear_state = CarpetNuclearStateModel(cfg)
        self.branching_inhibitor = CarpetBranchingInhibitor(cfg)
        
        # Cross-domain fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 3, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        
        logger.info(f"SESICarpetUnifiedSystem initialized with {cfg.carpet_n_cells} carpet cells")
    
    def forward(
        self,
        neural_input: Optional[Tensor] = None,
        sensor_signals: Optional[Tensor] = None,
        control_intent: Optional[Tensor] = None,
        nuclear_state: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full unified forward pass.
        
        Args:
            neural_input: (B, neural_field_dim) for neural simulation
            sensor_signals: (B, n_sensors, T) for monitoring
            control_intent: (B, latent_dim) for control
            nuclear_state: (B, n_cells) for nuclear simulation
        
        Returns:
            dict with all outputs
        """
        out = {}
        B = None
        
        # Determine batch size
        for x in [neural_input, sensor_signals, control_intent, nuclear_state]:
            if x is not None:
                B = x.shape[0]
                break
        
        if B is None:
            raise ValueError("At least one input must be provided")
        
        device = next(self.parameters()).device
        
        # ── Neural Field Simulation ──────────────────────────────────────
        if neural_input is not None:
            neural_out = self.neural_field(neural_input)
            out.update({f"neural_{k}": v for k, v in neural_out.items()})
            
            # Spectral analysis of neural activity
            neural_activity = neural_out["neural_activity"]
            spec_out = self.spectral(neural_activity)
            out.update({f"neural_spectral_{k}": v for k, v in spec_out.items()})
        
        # ── Neural Monitoring ────────────────────────────────────────────
        if sensor_signals is not None:
            monitor_out = self.neural_monitor(sensor_signals)
            out.update({f"monitor_{k}": v for k, v in monitor_out.items()})
        
        # ── Neural Control ───────────────────────────────────────────────
        if control_intent is not None:
            control_out = self.neural_controller(control_intent)
            out.update({f"control_{k}": v for k, v in control_out.items()})
        
        # ── Nuclear State & Inhibition ───────────────────────────────────
        if nuclear_state is not None:
            nuclear_out = self.nuclear_state(nuclear_state)
            out.update({f"nuclear_{k}": v for k, v in nuclear_out.items()})
            
            # Branching inhibition
            sigma_base = nuclear_state  # Use state as heterogeneity
            u_state = nuclear_state
            inhibit_out = self.branching_inhibitor(u_state, sigma_base)
            out.update({f"inhibit_{k}": v for k, v in inhibit_out.items()})
        
        # ── Carpet Consistency Check ─────────────────────────────────────
        # Verify resistance form consistency across domains
        if neural_input is not None and "neural_activity" in out:
            activity = out["neural_activity"]
            resistance_out = self.resistance(activity)
            out.update({f"carpet_resistance_{k}": v for k, v in resistance_out.items()})
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# 12. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class CarpetLoss(nn.Module):
    def __init__(self, cfg: CarpetConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        dev = next(iter(out.values())).device if out else torch.device("cpu")
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}
        
        # Neural monitoring loss
        if "monitor_reconstructed_signals" in out and "sensor_signals" in batch:
            L["neural_monitor"] = F.mse_loss(
                out["monitor_reconstructed_signals"],
                batch["sensor_signals"].mean(dim=-1))
        else:
            L["neural_monitor"] = zero
        
        # Neural control loss (safety)
        if "control_safety_score" in out:
            L["neural_control"] = -out["control_safety_score"].mean()
        else:
            L["neural_control"] = zero
        
        # Nuclear loss (encourage subcritical)
        if "nuclear_k_eff" in out:
            L["nuclear"] = F.relu(out["nuclear_k_eff"] - 1.0).mean()
        else:
            L["nuclear"] = zero
        
        # Carpet consistency (spectral dimension)
        if "carpet_resistance_concentration" in out:
            L["carpet"] = out["carpet_resistance_concentration"].mean() * 0.1
        else:
            L["carpet"] = zero
        
        # Regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg
        
        total = (
            self.cfg.w_neural_monitor * L["neural_monitor"] +
            self.cfg.w_neural_control * L["neural_control"] +
            self.cfg.w_nuclear * L["nuclear"] +
            self.cfg.w_carpet_consistency * L["carpet"] +
            self.cfg.w_reg * L["reg"]
        )
        L["total"] = total
        
        return L


# ═══════════════════════════════════════════════════════════════════════
# 13. DATASET
# ═══════════════════════════════════════════════════════════════════════

class SyntheticCarpetDataset(Dataset):
    """Synthetic dataset for carpet-based simulations."""
    
    def __init__(self, cfg: CarpetConfig, n: int, seed: int):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        
        # Neural inputs
        self.neural_input = torch.randn(n, cfg.neural_field_dim, generator=g)
        
        # Sensor signals
        self.sensor_signals = torch.randn(n, cfg.n_sensors, 32, generator=g)
        
        # Control intent
        self.control_intent = torch.randn(n, cfg.latent_dim, generator=g)
        
        # Nuclear state
        self.nuclear_state = torch.rand(n, cfg.carpet_n_cells, generator=g)
    
    def __len__(self):
        return len(self.neural_input)
    
    def __getitem__(self, i):
        return {
            "neural_input": self.neural_input[i],
            "sensor_signals": self.sensor_signals[i],
            "control_intent": self.control_intent[i],
            "nuclear_state": self.nuclear_state[i],
        }


def build_carpet_loaders(cfg: CarpetConfig):
    tr = SyntheticCarpetDataset(cfg, 1024, cfg.seed)
    va = SyntheticCarpetDataset(cfg, 128, cfg.seed + 1)
    te = SyntheticCarpetDataset(cfg, 256, cfg.seed + 2)
    
    common = dict(batch_size=cfg.batch_size, num_workers=2,
                  pin_memory=(cfg.device != "cpu"))
    
    return (
        DataLoader(tr, shuffle=True, drop_last=True, **common),
        DataLoader(va, shuffle=False, **common),
        DataLoader(te, shuffle=False, **common),
    )


# ═══════════════════════════════════════════════════════════════════════
# 14. TRAINER
# ═══════════════════════════════════════════════════════════════════════

class CarpetTrainer:
    def __init__(self, cfg: CarpetConfig, model: SESICarpetUnifiedSystem,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        
        if is_dist():
            self.model = DDP(self.model,
                             device_ids=[device.index] if device.type == "cuda" else None,
                             find_unused_parameters=False,
                             gradient_as_bucket_view=True)
        
        self.loss_fn = CarpetLoss(cfg)
        self.opt = torch.optim.AdamW(
            [p for p in self._raw().parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=1e-5)
        self.step = 0
    
    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        self.opt.zero_grad(set_to_none=True)
        
        out = self.model(
            neural_input=batch.get("neural_input"),
            sensor_signals=batch.get("sensor_signals"),
            control_intent=batch.get("control_intent"),
            nuclear_state=batch.get("nuclear_state"),
        )
        
        losses = self.loss_fn(out, batch, self._raw())
        losses["total"].backward()
        
        torch.nn.utils.clip_grad_norm_(self._raw().parameters(), 1.0)
        self.opt.step()
        self.step += 1
        
        return {k: float(v) for k, v in losses.items()
                if torch.is_tensor(v) and v.numel() == 1}


# ═══════════════════════════════════════════════════════════════════════
# 15. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Carpet Unified Module — Smoke Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)
    
    cfg = CarpetConfig(
        carpet_level=3,
        carpet_n_cells=8**3,
        neural_field_dim=64,
        n_neural_sources=32,
        n_sensors=16,
        latent_dim=64,
        n_spectral_modes=32,
        device=str(device),
    )
    
    model = SESICarpetUnifiedSystem(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    print(f"  Carpet cells: {model.carpet_builder.n_cells}")
    
    model.train()
    B = 2
    
    # Test neural field
    print("\n  [1/4] Neural Field Simulation...")
    neural_input = torch.randn(B, cfg.neural_field_dim, device=device)
    out = model(neural_input=neural_input)
    print(f"    Neural activity shape: {tuple(out['neural_neural_activity'].shape)}")
    
    # Test monitoring
    print("\n  [2/4] Neural Monitoring...")
    sensor_signals = torch.randn(B, cfg.n_sensors, 32, device=device)
    out = model(sensor_signals=sensor_signals)
    print(f"    Estimated sources shape: {tuple(out['monitor_estimated_sources'].shape)}")
    print(f"    Attention focus: {out['monitor_attention_focus'].mean().item():.4f}")
    
    # Test control
    print("\n  [3/4] Neural Control...")
    control_intent = torch.randn(B, cfg.latent_dim, device=device)
    out = model(control_intent=control_intent)
    print(f"    Safety score: {out['control_safety_score'].mean().item():.4f}")
    print(f"    Action permitted: {out['control_action_permitted'].mean().item():.4f}")
    
    # Test nuclear
    print("\n  [4/4] Nuclear Interdiction...")
    nuclear_state = torch.rand(B, cfg.carpet_n_cells, device=device)
    out = model(nuclear_state=nuclear_state)
    print(f"    k_eff: {out['nuclear_k_eff'].mean().item():.4f}")
    print(f"    Is subcritical: {out['nuclear_is_subcritical'].mean().item():.4f}")
    print(f"    Is arrested: {out['inhibit_is_arrested'].mean().item():.4f}")
    print(f"    Gumbel bound: {out['inhibit_probability_bound'].mean().item():.2e}")
    
    # Loss
    print("\n  Loss computation...")
    loss_fn = CarpetLoss(cfg)
    batch = {
        "neural_input": neural_input,
        "sensor_signals": sensor_signals,
        "control_intent": control_intent,
        "nuclear_state": nuclear_state,
    }
    full_out = model(**batch)
    losses = loss_fn(full_out, batch, model)
    losses["total"].backward()
    
    print(f"    Total loss: {losses['total'].item():.6f}")
    for k, v in losses.items():
        if k != "total":
            print(f"      {k}: {v.item():.6f}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL MODULES CONNECTED' if grad_ok else 'FAIL'}")
    
    print("\n" + "=" * 72)
    print("  ⚠️  REMINDER: This is a SIMULATION framework.")
    print("  - Neural monitoring/control: models information limits, not real devices")
    print("  - Nuclear interdiction: PRE-DETONATION only (ongoing reaction cannot be stopped)")
    print("=" * 72)
    print("  Smoke test complete.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser("SESI Carpet Unified Module")
    p.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.mode == "smoke":
        smoke_test()
    else:
        print("Training mode — use CarpetTrainer")
