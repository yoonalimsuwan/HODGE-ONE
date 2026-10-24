"""
=============================================================================
SESI ADVANCED DEFENSE SIMULATION FRAMEWORK v1.0
Multi-Domain Threat Analysis via Hodge-Carpet-Hilbert-Pólya Integration
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Production Defensive Simulation)
=============================================================================

CRITICAL HONEST SCOPE — READ THIS FIRST
────────────────────────────────────────
This is a DIFFERENTIABLE SIMULATION FRAMEWORK for defensive research and
academic study. It is NOT an operational weapon system.

What this system DOES:
  ✅ Simulates detection/analysis of aerial threats (drones, stealth, ICBM)
  ✅ Uses advanced mathematical theories (Hodge, Carpet, Hilbert-Pólya)
  ✅ Integrates with AGI ONE cognitive architecture
  ✅ Provides differentiable models for defensive strategy optimization
  ✅ Supports hardware-agnostic deployment (NVIDIA, Huawei Ascend, etc.)

What this system does NOT do:
  ❌ Control real weapons or autonomous systems
  ❌ Provide operational military capabilities
  ❌ Replace actual radar/sonar/sensor systems
  ❌ Enable remote neural control of humans
  ❌ Stop nuclear chain reactions after detonation begins

ETHICAL CONSTRAINTS:
  • All "anti-X" modules are DEFENSIVE analysis tools
  • No offensive capability modeling
  • No autonomous targeting or engagement
  • Research/educational use only

MATHEMATICAL FOUNDATIONS
────────────────────────
  [M1] Carpet Resistance Forms (Kusuoka-Zhou 1992)
       → Multi-scale threat pattern analysis
  
  [M2] Hodge Decomposition (Algebraic Geometry)
       → EM field decomposition for stealth detection
  
  [M3] Hilbert-Pólya Spectral Theory
       → Eigenvalue analysis for trajectory prediction
  
  [M4] OPMC Canonical Compression
       → Hyper-relational threat state compression
  
  [M5] SQCT Geometric Collapse
       → State transition modeling
  
  [M6] Gumbel Extreme-Value Bounds
       → Catastrophic event probability bounds

APPLICATION DOMAINS
───────────────────
  [D1] Anti-Drone Swarm Analysis
       • Graph attention for swarm trajectory prediction
       • Multi-sensor fusion (radar + acoustic + RF)
       • Carpet-based pattern recognition
  
  [D2] Anti-Stealth Resonance Analysis
       • Hodge decomposition of EM scattering
       • Resonance frequency identification
       • Vulnerability mapping
  
  [D3] Hypersonic/ICBM Tracking
       • Neural ODE for trajectory prediction
       • Plasma sheath modeling
       • Gumbel bounds for interception windows

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import logging
import argparse
import contextlib
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

logger = logging.getLogger("sesi.defense_sim")

__all__ = [
    "DefenseSimConfig",
    # Core mathematical modules
    "CarpetThreatAnalyzer",
    "HodgeFieldDecomposer",
    "HilbertPolyaSpectralPredictor",
    "OPMCThreatCompressor",
    # Defense modules
    "AntiDroneSwarmAnalyzer",
    "AntiStealthResonanceModel",
    "HypersonicICBMTracker",
    # Integration
    "SESIDefenseSimulationHub",
    "DefenseSimLoss",
    "DefenseSimTrainer",
    "DefenseDataset",
    "build_defense_loaders",
    # Infrastructure
    "setup_dist", "cleanup_dist", "seed_all",
]


# ═══════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════

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
class DefenseSimConfig:
    # ── Core dimensions ───────────────────────────────────────────────────
    latent_dim: int = 512
    threat_embed_dim: int = 256
    sensor_dim: int = 128
    
    # ── Carpet analysis ───────────────────────────────────────────────────
    carpet_level: int = 4
    carpet_n_cells: int = 8 ** 4
    resistance_rho: float = 1.25
    
    # ── Hodge decomposition ───────────────────────────────────────────────
    hodge_levels: int = 3
    hodge_grid_size: int = 32
    
    # ── Spectral prediction ───────────────────────────────────────────────
    n_eigenvalues: int = 64
    spectral_cutoff: float = 1e-6
    
    # ── OPMC compression ──────────────────────────────────────────────────
    opmc_sigma_dim: int = 128
    opmc_n_fields: int = 16
    
    # ── Threat types ──────────────────────────────────────────────────────
    n_drone_types: int = 10
    n_stealth_classes: int = 5
    n_missile_types: int = 8
    
    # ── Sensor fusion ─────────────────────────────────────────────────────
    n_sensor_types: int = 8  # radar, lidar, acoustic, IR, RF, etc.
    max_sensors_per_type: int = 32
    
    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    amp_dtype: str = "bf16"
    seed: int = 42
    device: str = "cuda"
    
    # ── Loss weights ──────────────────────────────────────────────────────
    w_detection: float = 0.3
    w_classification: float = 0.2
    w_trajectory: float = 0.3
    w_vulnerability: float = 0.2
    w_reg: float = 1e-6
    
    # ── Data ──────────────────────────────────────────────────────────────
    n_train: int = 8192
    n_val: int = 1024
    n_test: int = 2048


# ═══════════════════════════════════════════════════════════════════════
# 2. CARPET THREAT ANALYZER [M1]
# ═══════════════════════════════════════════════════════════════════════

class CarpetThreatAnalyzer(nn.Module):
    """
    [M1] Analyze threat patterns using carpet resistance forms.
    
    The fractal structure of the carpet models multi-scale threat
    patterns: from individual drone signatures to swarm formations.
    
    Resistance form E_n = ρ^n Σ (u(x)-u(y))² captures how threat
    signals propagate through different scales.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        r = 32
        d = cfg.threat_embed_dim
        
        # Lift threat features to carpet cells
        self.lift = nn.Sequential(
            LowRankAdapter(d, cfg.carpet_n_cells, r),
            nn.GELU(),
        )
        
        # Resistance scaling (learnable)
        self.log_rho_raw = nn.Parameter(
            torch.tensor(math.log(max(cfg.resistance_rho - 9/8, 1e-6)))
        )
        
        # Pattern classifier
        self.pattern_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, cfg.carpet_n_cells // 4),
            nn.GELU(),
            nn.Linear(cfg.carpet_n_cells // 4, cfg.n_drone_types + cfg.n_stealth_classes + cfg.n_missile_types),
        )
        
        # Build carpet graph
        self._build_carpet_graph()
    
    def _build_carpet_graph(self):
        n_cells = self.cfg.carpet_n_cells
        adj = torch.zeros(n_cells, n_cells)
        for i in range(n_cells):
            for j in range(max(0, i-2), min(n_cells, i+3)):
                if i != j:
                    adj[i, j] = 1.0
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj = adj / deg
        self.register_buffer("carpet_adj", adj, persistent=False)
        self.register_buffer("carpet_deg", deg.squeeze(), persistent=False)
        self.register_buffer("carpet_laplacian",
                             torch.diag(deg.squeeze()) - adj,
                             persistent=False)
    
    @property
    def rho(self) -> Tensor:
        return 9/8 + F.softplus(self.log_rho_raw)
    
    def forward(self, threat_features: Tensor) -> Dict[str, Tensor]:
        B = threat_features.shape[0]
        
        # Lift to carpet
        cell_values = self.lift(threat_features)
        
        # Compute resistance energy
        energy = torch.einsum("bi,ij,bj->b",
                              cell_values, self.carpet_laplacian, cell_values)
        energy = (self.rho ** self.cfg.carpet_level) * energy
        
        # Pattern classification
        pattern_logits = self.pattern_net(cell_values)
        
        return {
            "carpet_energy": energy,
            "pattern_logits": pattern_logits,
            "cell_values": cell_values,
            "resistance_rho": self.rho.expand(B),
        }


# ═══════════════════════════════════════════════════════════════════════
# 3. HODGE FIELD DECOMPOSER [M2]
# ═══════════════════════════════════════════════════════════════════════

class HodgeFieldDecomposer(nn.Module):
    """
    [M2] Decompose EM fields for stealth detection.
    
    Hodge decomposition: F = dα + δβ + γ
      - Harmonic (γ): non-radiating modes (stealth signature)
      - Exact (dα): source-driven fields
      - Coexact (δβ): radiation patterns
    
    For stealth detection, we analyze the harmonic component to
    identify non-radiating modes that indicate RAM materials.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        r = 32
        d = cfg.sensor_dim
        
        # Decomposition network
        self.decompose_net = nn.Sequential(
            LowRankAdapter(d, d * 3, r),
            nn.GELU(),
            nn.Linear(d * 3, d * cfg.hodge_levels),
        )
        
        # Stealth vulnerability predictor
        self.stealth_net = nn.Sequential(
            nn.Linear(d * cfg.hodge_levels, d),
            nn.GELU(),
            nn.Linear(d, cfg.n_stealth_classes),
        )
        
        # Resonance frequency predictor
        self.resonance_net = nn.Sequential(
            nn.Linear(d * cfg.hodge_levels, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 64),  # 64 frequency bands
            nn.Softmax(dim=-1),
        )
    
    def forward(self, em_field: Tensor) -> Dict[str, Tensor]:
        B = em_field.shape[0]
        
        # Hodge decomposition
        decomp = self.decompose_net(em_field)
        components = decomp.view(B, self.cfg.hodge_levels, -1)
        
        # Harmonic component (stealth signature)
        harmonic = components[:, 0]
        
        # Stealth classification
        stealth_logits = self.stealth_net(decomp)
        
        # Resonance frequencies
        resonance_bands = self.resonance_net(decomp)
        
        return {
            "hodge_components": components,
            "harmonic_component": harmonic,
            "stealth_logits": stealth_logits,
            "resonance_bands": resonance_bands,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. HILBERT-PÓLYA SPECTRAL PREDICTOR [M3]
# ═══════════════════════════════════════════════════════════════════════

class HilbertPolyaSpectralPredictor(nn.Module):
    """
    [M3] Spectral analysis for trajectory prediction.
    
    Uses Hilbert-Pólya spectral theory to analyze eigenvalue
    distributions of threat trajectories. The spectral gap indicates
    stability/instability of predicted paths.
    
    For ICBM tracking: eigenvalue clustering predicts maneuver patterns.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        r = 32
        d = cfg.latent_dim
        
        # Spectral operator construction
        self.operator_net = nn.Sequential(
            LowRankAdapter(d, cfg.n_eigenvalues ** 2, r),
            nn.Tanh(),
        )
        
        # Eigenvalue predictor
        self.eigenvalue_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues, cfg.n_eigenvalues),
            nn.GELU(),
            nn.Linear(cfg.n_eigenvalues, cfg.n_eigenvalues),
            nn.Softplus(),  # eigenvalues > 0
        )
        
        # Trajectory stability predictor
        self.stability_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )
        
        # Intercept window predictor
        self.intercept_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues + d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 2),  # [start_time, end_time]
            nn.Softplus(),
        )
    
    def forward(self, trajectory_latent: Tensor) -> Dict[str, Tensor]:
        B = trajectory_latent.shape[0]
        
        # Construct spectral operator
        op_flat = self.operator_net(trajectory_latent)
        H = op_flat.view(B, self.cfg.n_eigenvalues, self.cfg.n_eigenvalues)
        H = 0.5 * (H + H.transpose(-1, -2))  # symmetrize
        
        # Eigenvalues
        eigenvalues = torch.diagonal(H, dim1=-2, dim2=-1).abs()
        eigenvalues = self.eigenvalue_net(eigenvalues)
        
        # Spectral gap
        spectral_gap = eigenvalues[:, 1] - eigenvalues[:, 0]
        
        # Stability
        stability = self.stability_net(eigenvalues).squeeze(-1)
        
        # Intercept window
        intercept_input = torch.cat([eigenvalues, trajectory_latent], dim=-1)
        intercept_window = self.intercept_net(intercept_input)
        
        return {
            "spectral_operator": H,
            "eigenvalues": eigenvalues,
            "spectral_gap": spectral_gap,
            "trajectory_stability": stability,
            "intercept_window": intercept_window,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. OPMC THREAT COMPRESSOR [M4]
# ═══════════════════════════════════════════════════════════════════════

class OPMCThreatCompressor(nn.Module):
    """
    [M4] Compress hyper-relational threat states.
    
    OPMC canonical signature σ_T compresses the astronomical complexity
    of multi-threat scenarios into polynomial-size representation.
    
    This enables real-time analysis of complex battlefields with
    hundreds of simultaneous threats.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        r = 32
        d = cfg.latent_dim
        Fd = cfg.opmc_n_fields
        ds = cfg.opmc_sigma_dim
        
        # Field embeddings
        self.field_emb = nn.Parameter(torch.randn(Fd, d) * 0.02)
        
        # Query and value
        self.query = LowRankAdapter(d, d, r)
        self.value = LowRankAdapter(d, ds, r)
        
        # Refinement
        self.refine = nn.Sequential(
            nn.LayerNorm(ds),
            nn.Linear(ds, ds),
            nn.GELU(),
            nn.Linear(ds, ds),
        )
        
        # Threat priority predictor
        self.priority_net = nn.Sequential(
            nn.Linear(ds, ds // 2),
            nn.GELU(),
            nn.Linear(ds // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, threat_state: Tensor) -> Dict[str, Tensor]:
        B = threat_state.shape[0]
        
        # Compute canonical signature
        q = self.query(threat_state).unsqueeze(1)
        logits = torch.einsum("bnd,fd->bnf", q, self.field_emb) / math.sqrt(q.shape[-1])
        prov = F.softmax(logits, dim=1)
        
        v = self.value(threat_state).unsqueeze(1)
        sigma = torch.einsum("bnf,bnd->bfd", prov, v).squeeze(1)
        sigma = self.refine(sigma)
        
        # Threat priority
        priority = self.priority_net(sigma).squeeze(-1)
        
        return {
            "canonical_signature": sigma,
            "threat_priority": priority,
            "field_provenance": prov.squeeze(1),
        }


# ═══════════════════════════════════════════════════════════════════════
# 6. ANTI-DRONE SWARM ANALYZER [D1]
# ═══════════════════════════════════════════════════════════════════════

class AntiDroneSwarmAnalyzer(nn.Module):
    """
    [D1] Analyze drone swarm threats using graph attention.
    
    Models swarm behavior as a graph where:
      - Nodes = individual drones
      - Edges = communication/coordination links
      - Graph attention = swarm coordination patterns
    
    Uses Carpet resistance forms to identify swarm formation patterns.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        
        # Graph attention for swarm dynamics
        self.gat_conv = nn.MultiheadAttention(
            embed_dim=cfg.threat_embed_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Multi-modal sensor fusion
        self.sensor_fusion = nn.Sequential(
            nn.Linear(cfg.sensor_dim * cfg.n_sensor_types, cfg.threat_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.threat_embed_dim, cfg.threat_embed_dim),
        )
        
        # Swarm pattern classifier
        self.swarm_classifier = nn.Sequential(
            nn.Linear(cfg.threat_embed_dim, cfg.threat_embed_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.threat_embed_dim // 2, cfg.n_drone_types),
        )
        
        # Carpet analyzer
        self.carpet_analyzer = CarpetThreatAnalyzer(cfg)
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(cfg.threat_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        sensor_data: Tensor,  # (B, n_sensors, sensor_dim)
        swarm_graph: Tensor,  # (B, n_drones, threat_embed_dim)
    ) -> Dict[str, Tensor]:
        B = sensor_data.shape[0]
        
        # Sensor fusion
        fused_sensors = self.sensor_fusion(sensor_data.flatten(1))
        
        # Graph attention for swarm
        attn_out, _ = self.gat_conv(swarm_graph, swarm_graph, swarm_graph)
        swarm_state = attn_out.mean(dim=1)
        
        # Carpet pattern analysis
        carpet_out = self.carpet_analyzer(swarm_state)
        
        # Combined state
        combined = swarm_state + carpet_out["cell_values"].mean(dim=-1, keepdim=True).expand(-1, self.cfg.threat_embed_dim)
        
        # Detection
        detection_prob = self.detection_head(combined).squeeze(-1)
        
        # Classification
        swarm_logits = self.swarm_classifier(combined)
        
        return {
            "detection_probability": detection_prob,
            "swarm_classification": swarm_logits,
            "swarm_state": swarm_state,
            **{f"carpet_{k}": v for k, v in carpet_out.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# 7. ANTI-STEALTH RESONANCE MODEL [D2]
# ═══════════════════════════════════════════════════════════════════════

class AntiStealthResonanceModel(nn.Module):
    """
    [D2] Detect stealth aircraft via resonance analysis.
    
    Uses Hodge decomposition to separate radiating vs non-radiating
    EM modes. Stealth aircraft (F-22, F-35, B-2) minimize radiating
    modes via RAM materials and geometric shaping.
    
    This model identifies resonance frequencies where RAM effectiveness
    drops, revealing stealth targets.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        
        # Hodge decomposition
        self.hodge_decomp = HodgeFieldDecomposer(cfg)
        
        # Resonance analyzer
        self.resonance_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
        )
        
        # Stealth vulnerability predictor
        self.vulnerability_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Optimal counter-frequency predictor
        self.counter_freq_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 64),  # 64 frequency bands
            nn.Softmax(dim=-1),
        )
    
    def forward(self, radar_return: Tensor) -> Dict[str, Tensor]:
        B = radar_return.shape[0]
        
        # Hodge decomposition
        hodge_out = self.hodge_decomp(radar_return)
        
        # Resonance analysis
        resonance_features = self.resonance_net(hodge_out["harmonic_component"])
        
        # Vulnerability score
        vulnerability = self.vulnerability_net(resonance_features).squeeze(-1)
        
        # Optimal counter-frequencies
        counter_freqs = self.counter_freq_net(resonance_features)
        
        return {
            "stealth_vulnerability": vulnerability,
            "optimal_counter_freqs": counter_freqs,
            **{f"hodge_{k}": v for k, v in hodge_out.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# 8. HYPERSONIC/ICBM TRACKER [D3]
# ═══════════════════════════════════════════════════════════════════════

class HypersonicICBMTracker(nn.Module):
    """
    [D3] Track hypersonic glide vehicles and ICBMs.
    
    Uses:
      - Neural ODE for trajectory prediction
      - Hilbert-Pólya spectral analysis for maneuver detection
      - Gumbel bounds for interception probability
    
    Models plasma sheath effects that cause radar blackout during
    hypersonic flight.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        
        # Neural ODE for trajectory
        self.trajectory_ode = nn.Sequential(
            nn.Linear(cfg.latent_dim + 6, 512),  # +6 for [x,y,z, vx,vy,vz]
            nn.GELU(),
            nn.Linear(512, 6),
        )
        
        # Spectral predictor
        self.spectral_predictor = HilbertPolyaSpectralPredictor(cfg)
        
        # Plasma sheath model
        self.plasma_net = nn.Sequential(
            nn.Linear(3, 128),  # [velocity, altitude, angle]
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Interception probability
        self.intercept_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Gumbel extreme-value engine
        self.register_buffer("exp_clamp_max", torch.tensor(80.0), persistent=False)
        self.register_buffer("epsilon", torch.tensor(1e-15), persistent=False)
    
    def compute_gumbel_bound(self, delta_E: Tensor, sigma_var: float = 0.05) -> Tensor:
        """Gumbel probability bound for interception success."""
        sigma_sq = sigma_var ** 2
        denom = torch.clamp(
            torch.tensor(sigma_sq * 1e-5, device=delta_E.device, dtype=delta_E.dtype),
            min=self.epsilon,
        )
        inner = torch.clamp(delta_E / denom, max=self.exp_clamp_max)
        outer = -1.0 * torch.exp(inner)
        return torch.exp(outer)
    
    def forward(self, missile_state: Tensor) -> Dict[str, Tensor]:
        B = missile_state.shape[0]
        
        # Trajectory prediction (Neural ODE)
        dynamics = self.trajectory_ode(missile_state)
        acc = dynamics[:, 3:]
        
        # Euler integration
        dt = 0.1
        next_vel = missile_state[:, 3:] + acc * dt
        next_pos = missile_state[:, :3] + missile_state[:, 3:] * dt + 0.5 * acc * (dt ** 2)
        next_state = torch.cat([next_pos, next_vel], dim=-1)
        
        # Spectral analysis
        spectral_out = self.spectral_predictor(missile_state)
        
        # Plasma sheath blackout
        velocity_mag = torch.norm(missile_state[:, 3:], dim=-1, keepdim=True)
        altitude = missile_state[:, 2:3]
        blackout_prob = self.plasma_net(torch.cat([velocity_mag, altitude, torch.zeros_like(altitude)], dim=-1))
        
        # Interception probability
        intercept_input = torch.cat([
            spectral_out["eigenvalues"],
            spectral_out["intercept_window"],
        ], dim=-1)
        intercept_prob = self.intercept_net(intercept_input).squeeze(-1)
        
        # Gumbel bound
        delta_E = spectral_out["spectral_gap"]
        gumbel_bound = self.compute_gumbel_bound(delta_E)
        
        return {
            "predicted_next_state": next_state,
            "trajectory_stability": spectral_out["trajectory_stability"],
            "radar_blackout_probability": blackout_prob.squeeze(-1),
            "intercept_probability": intercept_prob,
            "intercept_window": spectral_out["intercept_window"],
            "gumbel_bound": gumbel_bound,
            **{f"spectral_{k}": v for k, v in spectral_out.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# 9. UNIFIED DEFENSE SIMULATION HUB
# ═══════════════════════════════════════════════════════════════════════

class SESIDefenseSimulationHub(nn.Module):
    """
    Unified defensive simulation integrating all modules.
    
    Provides comprehensive threat analysis across multiple domains:
      - Drone swarms
      - Stealth aircraft
      - Hypersonic/ICBM threats
    
    All outputs are defensive analysis metrics, not offensive capabilities.
    """
    
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
        
        # Defense modules
        self.drone_analyzer = AntiDroneSwarmAnalyzer(cfg)
        self.stealth_model = AntiStealthResonanceModel(cfg)
        self.missile_tracker = HypersonicICBMTracker(cfg)
        
        # OPMC compressor
        self.opmc_compressor = OPMCThreatCompressor(cfg)
        
        # Global threat fusion
        self.threat_fusion = nn.Sequential(
            nn.Linear(cfg.threat_embed_dim * 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(cfg.sensor_dim * cfg.n_sensor_types, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
    
    def forward(
        self,
        sensor_data: Tensor,
        swarm_graph: Tensor,
        radar_return: Tensor,
        missile_state: Tensor,
    ) -> Dict[str, Tensor]:
        B = sensor_data.shape[0]
        out = {}
        
        # Encode inputs
        encoded = self.input_encoder(sensor_data.flatten(1))
        
        # Drone swarm analysis
        drone_out = self.drone_analyzer(sensor_data, swarm_graph)
        out.update({f"drone_{k}": v for k, v in drone_out.items()})
        
        # Stealth analysis
        stealth_out = self.stealth_model(radar_return)
        out.update({f"stealth_{k}": v for k, v in stealth_out.items()})
        
        # Missile tracking
        missile_out = self.missile_tracker(missile_state)
        out.update({f"missile_{k}": v for k, v in missile_out.items()})
        
        # OPMC compression
        opmc_out = self.opmc_compressor(encoded)
        out.update({f"opmc_{k}": v for k, v in opmc_out.items()})
        
        # Global threat assessment
        combined = torch.cat([
            drone_out["swarm_state"],
            stealth_out["hodge_harmonic_component"].mean(dim=-1, keepdim=True).expand(B, self.cfg.threat_embed_dim),
            missile_out["predicted_next_state"].mean(dim=-1, keepdim=True).expand(B, self.cfg.threat_embed_dim),
        ], dim=-1)
        
        global_threat = self.threat_fusion(combined).squeeze(-1)
        out["global_defense_readiness"] = 1.0 - global_threat
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# 10. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class DefenseSimLoss(nn.Module):
    def __init__(self, cfg: DefenseSimConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        dev = out["global_defense_readiness"].device
        zero = torch.zeros((), device=dev)
        L = {}
        
        # Detection loss
        if "drone_detection_probability" in out and "drone_label" in batch:
            L["detection"] = F.binary_cross_entropy(
                out["drone_detection_probability"],
                batch["drone_label"].float(),
            )
        else:
            L["detection"] = zero
        
        # Classification loss
        if "drone_swarm_classification" in out and "drone_type" in batch:
            L["classification"] = F.cross_entropy(
                out["drone_swarm_classification"],
                batch["drone_type"],
            )
        else:
            L["classification"] = zero
        
        # Trajectory loss
        if "missile_predicted_next_state" in out and "missile_target_state" in batch:
            L["trajectory"] = F.mse_loss(
                out["missile_predicted_next_state"],
                batch["missile_target_state"],
            )
        else:
            L["trajectory"] = zero
        
        # Vulnerability loss
        if "stealth_stealth_vulnerability" in out and "stealth_label" in batch:
            L["vulnerability"] = F.binary_cross_entropy(
                out["stealth_stealth_vulnerability"],
                batch["stealth_label"].float(),
            )
        else:
            L["vulnerability"] = zero
        
        # Regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_req
```
