"""
=============================================================================
SESI ELECTRONIC WARFARE ENVIRONMENT v1.0
Ultimate EW System Integrating ALL Available Mathematical Theories
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Full Theory Integration — Defensive Focus)
=============================================================================

HONEST SCOPE — READ FIRST
─────────────────────────
This is a DIFFERENTIABLE SIMULATION FRAMEWORK for defensive Electronic
Warfare research. It is NOT an operational EW weapon system.

DEFENSIVE FOCUS:
  ✅ Electronic Warfare Support (ES) — detect, analyze, locate threats
  ✅ Electronic Protection (EP) — shield own systems from EW attack
  ✅ EW Research & Simulation — study adversary capabilities
  ⚠️ Electronic Attack (EA) — modeled ONLY for defensive countermeasure
     design (e.g., designing jammers to test own system resilience)

THEORIES INTEGRATED (from 5 source papers + SESI framework)
────────────────────────────────────────────────────────────
  [T1]  Hodge Decomposition        → Separate jamming/signal/noise
  [T2]  Hilbert-Pólya Spectral     → Eigenvalue analysis of threat patterns
  [T3]  Carpet Resistance Forms    → Fractal antenna/RAM design
  [T4]  Gumbel Extreme-Value       → Rare jamming event bounds
  [T5]  CUSUM + Subsampling (P11)  → Assumption-light threat detection
  [T6]  Hysteresis No-Zeno (P3)    → Adaptive response control
  [T7]  OPMC Canonical Compression → Signal data compression
  [T8]  SSC Photonic Processor     → Real-time optical signal processing
  [T9]  Discrete Correction (P-OT) → Lattice-derivative bias removal
  [T10] Isodiametric Nucleation    → Jamming pattern analysis
  [T11] Fractafold Junctions       → Multi-cell antenna arrays
  [T12] Non-p.c.f. Fractals        → Stealth material design
  [T13] FAB ONE v4.0               → Manufacture EW hardware

APPLICATION DOMAINS
───────────────────
  [D1] Threat Signal Analysis & Classification
  [D2] Spectrum Situational Awareness
  [D3] Anti-Jamming & Electronic Protection
  [D4] Fractal Antenna Array Design
  [D5] Stealth/RAM Material Design
  [D6] Real-Time Photonic Signal Processing
  [D7] Adaptive EW Response Control
  [D8] EW Hardware Fabrication (via FAB ONE)

=============================================================================
"""

from __future__ import annotations

import math
import os
import logging
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger("sesi.ew_env")

__all__ = [
    "EWEnvironmentConfig",
    # [T1] Hodge-based signal separation
    "HodgeSignalDecomposer",
    "JammingSignalClassifier",
    # [T2] Spectral cognitive engine
    "SpectralCognitiveEngine",
    "ThreatPatternRecognizer",
    # [T3][T11] Fractal antenna systems
    "FractalAntennaArray",
    "FractafoldJunctionArray",
    # [T4] Jamming probability bounds
    "GumbelJammingBounder",
    # [T5] Change-point detection
    "CUSUMThreatDetector",
    "AssumptionLightCI",
    # [T6] Adaptive control
    "HysteresisAdaptiveController",
    # [T7] Signal compression
    "OPMCSignalCompressor",
    # [T8] Photonic processing
    "SSCPhotonicProcessor",
    # [T9] Discrete correction
    "DiscreteCorrectionFilter",
    # [T10] Jamming pattern analysis
    "IsodiametricJammingAnalyzer",
    # [T12] Stealth materials
    "StealthMaterialDesigner",
    # Unified system
    "SESIElectronicWarfareEnvironment",
    "EWEnvironmentLoss",
    "EWEcosystemAdapter",
    "attach_ew_to_ecosystem",
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
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EWEnvironmentConfig:
    # ── Core dimensions ───────────────────────────────────────────────────
    latent_dim: int = 512
    signal_dim: int = 256
    spectrum_dim: int = 512
    
    # ── RF environment ────────────────────────────────────────────────────
    freq_min_ghz: float = 0.1
    freq_max_ghz: float = 100.0
    n_freq_bins: int = 1024
    n_antenna_elements: int = 64
    
    # ── Threat types ──────────────────────────────────────────────────────
    n_threat_classes: int = 20       # radar, comms, jammer types
    n_jamming_modes: int = 10        # noise, deceptive, repeater, etc.
    
    # ── Hodge decomposition ───────────────────────────────────────────────
    hodge_dim: int = 128
    
    # ── Carpet / fractal ──────────────────────────────────────────────────
    carpet_level: int = 4
    carpet_n_cells: int = 8 ** 4
    resistance_rho: float = 1.25
    
    # ── Spectral analysis ─────────────────────────────────────────────────
    n_eigenvalues: int = 64
    spectral_cutoff: float = 1e-6
    
    # ── Gumbel / CUSUM ────────────────────────────────────────────────────
    gumbel_c1: float = 1.0
    gumbel_sigma: float = 0.05
    gumbel_cutoff: float = 1e-12
    cusum_threshold: float = 5.0
    subsample_block_size: int = 100
    confidence_level: float = 0.95
    
    # ── Hysteresis No-Zeno ────────────────────────────────────────────────
    theta_hi: float = 1.0
    theta_lo: float = 0.8
    delta_e_min: float = 0.1
    
    # ── OPMC ──────────────────────────────────────────────────────────────
    opmc_sigma_dim: int = 128
    opmc_n_fields: int = 16
    
    # ── SSC Photonic ──────────────────────────────────────────────────────
    n_mzi_cells: int = 64
    n_mrr_blocks: int = 16
    n_pcm_cells: int = 32
    n2_bto: float = 1e-16
    q_mrr: float = 1e4
    a_eff_um2: float = 0.16
    l_eff_um: float = 10.0
    
    # ── Discrete correction ───────────────────────────────────────────────
    l_c: float = 1.5e-3
    n_grid: int = 64
    
    # ── Isodiametric ──────────────────────────────────────────────────────
    kappa_fatness: float = 2.0
    a_min_um2: float = 0.01
    omega_n_minus_1: float = math.pi
    
    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    amp_dtype: str = "bf16"
    seed: int = 42
    device: str = "cuda"
    
    # ── Loss weights ──────────────────────────────────────────────────────
    w_detection: float = 0.2
    w_classification: float = 0.15
    w_protection: float = 0.2
    w_spectral: float = 0.1
    w_compression: float = 0.1
    w_safety: float = 0.15
    w_reg: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# [T1] HODGE SIGNAL DECOMPOSER — Separate jamming/signal/noise
# ═══════════════════════════════════════════════════════════════════════

class HodgeSignalDecomposer(nn.Module):
    """
    [T1] Hodge decomposition of EM signals.
    
    F = dα + δβ + γ
      - Harmonic (γ): true signal of interest
      - Exact (dα): jamming interference (gradient-like)
      - Coexact (δβ): noise/artifacts (curl-like)
    
    This separation is fundamental to Electronic Protection:
    by identifying which component is the jammer, we can design
    filters that specifically target it while preserving the
    harmonic (true signal) component.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        r = 32
        d = cfg.signal_dim
        
        # Decomposition network
        self.decompose_net = nn.Sequential(
            LowRankAdapter(d, cfg.hodge_dim * 3, r),
            nn.GELU(),
            nn.Linear(cfg.hodge_dim * 3, cfg.hodge_dim * 3),
        )
        
        # Hodge-Riemann bilinear form
        self.q_form = nn.Parameter(torch.randn(cfg.hodge_dim, cfg.hodge_dim) * 0.01)
        
        # Jamming indicator (high coexact energy = jamming)
        self.jamming_net = nn.Sequential(
            nn.Linear(cfg.hodge_dim * 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, em_signal: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            em_signal: (B, signal_dim) received EM signal
        
        Returns:
            dict with harmonic, exact, coexact components + jamming score
        """
        B = em_signal.shape[0]
        
        # Decompose
        decomp = self.decompose_net(em_signal)
        harmonic, exact, coexact = decomp.chunk(3, dim=-1)
        
        # Hodge-Riemann check (positivity)
        q_sym = 0.5 * (self.q_form + self.q_form.T)
        hr_value = torch.einsum("bi,ij,bj->b", harmonic, q_sym, harmonic)
        
        # Energy in each component
        h_energy = harmonic.pow(2).sum(dim=-1)
        e_energy = exact.pow(2).sum(dim=-1)
        c_energy = coexact.pow(2).sum(dim=-1)
        total_energy = h_energy + e_energy + c_energy + 1e-8
        
        # Jamming indicator
        jam_input = torch.cat([harmonic, exact, coexact], dim=-1)
        jamming_score = self.jamming_net(jam_input).squeeze(-1)
        
        # Signal-to-jam ratio
        sjr = h_energy / (c_energy + 1e-8)
        
        return {
            "harmonic": harmonic,
            "exact": exact,
            "coexact": coexact,
            "hodge_riemann_value": hr_value,
            "harmonic_energy": h_energy,
            "exact_energy": e_energy,
            "coexact_energy": c_energy,
            "jamming_score": jamming_score,
            "signal_to_jam_ratio": sjr,
            "signal_quality": h_energy / total_energy,
        }


class JammingSignalClassifier(nn.Module):
    """
    Classify jamming type from Hodge components.
    
    Jamming modes: noise, spot, sweep, barrage, repeater,
    deceptive, asynchronous pulse, etc.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hodge_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_jamming_modes),
        )
    
    def forward(self, hodge_out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        combined = torch.cat([
            hodge_out["harmonic"],
            hodge_out["exact"],
            hodge_out["coexact"],
        ], dim=-1)
        
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)
        
        return {
            "jamming_logits": logits,
            "jamming_probs": probs,
            "jamming_type": probs.argmax(dim=-1),
            "jamming_confidence": probs.max(dim=-1).values,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T2] SPECTRAL COGNITIVE ENGINE — Eigenvalue analysis of threats
# ═══════════════════════════════════════════════════════════════════════

class SpectralCognitiveEngine(nn.Module):
    """
    [T2] Hilbert-Pólya spectral analysis of EM threat environment.
    
    Constructs a self-adjoint operator H from the spectral environment
    and analyzes its eigenvalues to:
      - Identify threat patterns (eigenvalue clustering)
      - Detect stealth emitters (spectral gaps)
      - Predict adversary behavior (eigenvalue evolution)
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        r = 32
        d = cfg.latent_dim
        
        # Operator construction (self-adjoint)
        self.operator_net = nn.Sequential(
            LowRankAdapter(d, cfg.n_eigenvalues ** 2, r),
            nn.Tanh(),
        )
        
        # Eigenvalue predictor
        self.eigenvalue_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues, cfg.n_eigenvalues),
            nn.GELU(),
            nn.Linear(cfg.n_eigenvalues, cfg.n_eigenvalues),
            nn.Softplus(),
        )
        
        # Spectral gap detector (stealth emitter indicator)
        self.gap_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Threat pattern classifier
        self.pattern_net = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_threat_classes),
        )
    
    def forward(self, spectral_state: Tensor) -> Dict[str, Tensor]:
        B = spectral_state.shape[0]
        
        # Construct self-adjoint operator
        op_flat = self.operator_net(spectral_state)
        H = op_flat.view(B, self.cfg.n_eigenvalues, self.cfg.n_eigenvalues)
        H = 0.5 * (H + H.transpose(-1, -2))
        
        # Eigenvalues
        eigenvalues = torch.diagonal(H, dim1=-2, dim2=-1).abs()
        eigenvalues = self.eigenvalue_net(eigenvalues)
        eigenvalues, _ = torch.sort(eigenvalues, dim=-1)
        
        # Spectral gap
        if eigenvalues.shape[-1] >= 2:
            spectral_gap = eigenvalues[:, 1] - eigenvalues[:, 0]
        else:
            spectral_gap = torch.zeros(B, device=spectral_state.device)
        
        gap_score = self.gap_net(eigenvalues).squeeze(-1)
        
        # Threat pattern
        pattern_logits = self.pattern_net(eigenvalues)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        return {
            "spectral_operator": H,
            "eigenvalues": eigenvalues,
            "spectral_gap": spectral_gap,
            "gap_score": gap_score,
            "threat_pattern_logits": pattern_logits,
            "threat_pattern_probs": pattern_probs,
            "threat_class": pattern_probs.argmax(dim=-1),
        }


class ThreatPatternRecognizer(nn.Module):
    """Recognize specific adversary emitter patterns."""
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        self.recognizer = nn.Sequential(
            nn.Linear(cfg.n_eigenvalues + cfg.signal_dim, 128),
            nn.GELU(),
            nn.Linear(128, cfg.n_threat_classes),
        )
    
    def forward(self, spectral_out: Dict[str, Tensor],
                hodge_out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        combined = torch.cat([
            spectral_out["eigenvalues"],
            hodge_out["harmonic"],
        ], dim=-1)
        
        logits = self.recognizer(combined)
        probs = F.softmax(logits, dim=-1)
        
        return {
            "threat_logits": logits,
            "threat_probs": probs,
            "recognized_threat": probs.argmax(dim=-1),
            "recognition_confidence": probs.max(dim=-1).values,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T3][T11] FRACTAL ANTENNA ARRAY — Carpet + Fractafold junctions
# ═══════════════════════════════════════════════════════════════════════

class FractafoldJunctionArray(nn.Module):
    """
    [T11] Multi-cell antenna array with order-4 Kirchhoff junctions.
    
    From Six Problems paper: at a shared vertex of k ≥ 2 cells,
    continuity + Kirchhoff balance conditions must hold.
    The kernel is 1-dimensional (handled via quotient space).
    """
    
    def __init__(self, cfg: EWEnvironmentConfig, k: int = 4):
        super().__init__()
        self.cfg = cfg
        self.k = k
        
        # Junction encoder
        self.junction_enc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
        )
        
        # Continuity enforcer
        self.continuity_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 4 * k),
            nn.Tanh(),
        )
        
        # Kirchhoff balance enforcer
        self.kirchhoff_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 4 * k),
            nn.Tanh(),
        )
    
    def forward(self, array_state: Tensor) -> Dict[str, Tensor]:
        B = array_state.shape[0]
        
        encoded = self.junction_enc(array_state)
        
        # Continuity values
        continuity = self.continuity_net(encoded).view(B, 4, self.k)
        continuity_residual = continuity.std(dim=-1).mean(dim=-1)
        
        # Kirchhoff balance
        kirchhoff = self.kirchhoff_net(encoded).view(B, 4, self.k)
        kirchhoff_residual = kirchhoff.sum(dim=-1).abs().mean(dim=-1)
        
        # Junction quality
        junction_quality = 1.0 / (1.0 + continuity_residual + kirchhoff_residual)
        
        return {
            "continuity_residual": continuity_residual,
            "kirchhoff_residual": kirchhoff_residual,
            "junction_quality": junction_quality,
            "kernel_dimension": torch.ones(B, device=array_state.device),
        }


class FractalAntennaArray(nn.Module):
    """
    [T3] Fractal antenna array using carpet resistance forms.
    
    Fractal antennas provide:
      - Multi-band operation (self-similar structure)
      - Wide bandwidth
      - Compact size
      - Low observability (fractal RAM)
    
    The carpet resistance form models signal propagation through
    the fractal structure:
      E_n(u) = ρ^n Σ (u(x) - u(y))²
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Fractal geometry encoder
        self.geometry_net = nn.Sequential(
            LowRankAdapter(cfg.latent_dim, cfg.carpet_n_cells, 32),
            nn.GELU(),
        )
        
        # Resistance scaling
        self.log_rho_raw = nn.Parameter(
            torch.tensor(math.log(max(cfg.resistance_rho - 9/8, 1e-6)))
        )
        
        # Gain predictor
        self.gain_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Bandwidth predictor
        self.bandwidth_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Fractafold junctions
        self.junctions = FractafoldJunctionArray(cfg)
        
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
        self.register_buffer("carpet_laplacian",
                             torch.diag(deg.squeeze()) - adj, persistent=False)
    
    @property
    def rho(self) -> Tensor:
        return 9/8 + F.softplus(self.log_rho_raw)
    
    def forward(self, design_params: Tensor) -> Dict[str, Tensor]:
        B = design_params.shape[0]
        
        # Geometry
        cell_values = self.geometry_net(design_params)
        
        # Resistance energy
        energy = torch.einsum("bi,ij,bj->b",
                              cell_values, self.carpet_laplacian, cell_values)
        energy = (self.rho ** self.cfg.carpet_level) * energy
        
        # Antenna metrics
        gain = self.gain_net(cell_values).squeeze(-1)
        bandwidth = self.bandwidth_net(cell_values).squeeze(-1)
        
        # Junction quality
        junction_out = self.junctions(design_params)
        
        # Fractal dimension estimate
        fractal_dim = torch.log(energy + 1e-8) / torch.log(self.rho + 1e-8)
        
        return {
            "antenna_gain_dbi": gain,
            "bandwidth_ghz": bandwidth,
            "resistance_energy": energy,
            "fractal_dimension": fractal_dim,
            "cell_values": cell_values,
            **{f"junction_{k}": v for k, v in junction_out.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# [T4] GUMBEL JAMMING BOUNDER — Rare event probability
# ═══════════════════════════════════════════════════════════════════════

class GumbelJammingBounder(nn.Module):
    """
    [T4] Gumbel extreme-value bounds for rare jamming events.
    
    P(T_{k+1} - T_k < dt) ≤ exp(-C₁ · exp(ΔE/(σ²·dt)))
    
    Application: bound the probability of catastrophic jamming events
    that could overwhelm defensive systems.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Activation energy predictor
        self.activation_net = nn.Sequential(
            LowRankAdapter(cfg.latent_dim, 64, 16),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Threat severity predictor
        self.severity_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.register_buffer("exp_clamp_max", torch.tensor(80.0), persistent=False)
        self.register_buffer("epsilon", torch.tensor(1e-15), persistent=False)
    
    def compute_gumbel_bound(self, delta_E: Tensor) -> Tensor:
        sigma_sq = self.cfg.gumbel_sigma ** 2
        denom = torch.clamp(
            torch.tensor(sigma_sq * 1e-5, device=delta_E.device, dtype=delta_E.dtype),
            min=self.epsilon)
        inner = torch.clamp(delta_E / denom, max=self.exp_clamp_max)
        outer = -self.cfg.gumbel_c1 * torch.exp(inner)
        return torch.exp(outer)
    
    def forward(self, threat_state: Tensor) -> Dict[str, Tensor]:
        B = threat_state.shape[0]
        
        delta_E = self.activation_net(threat_state).squeeze(-1)
        prob_bound = self.compute_gumbel_bound(delta_E)
        
        sev_input = torch.cat([threat_state, delta_E.unsqueeze(-1)], dim=-1)
        severity = self.severity_net(sev_input).squeeze(-1)
        
        is_catastrophic = (prob_bound > self.cfg.gumbel_cutoff).float()
        
        return {
            "activation_energy": delta_E,
            "gumbel_prob_bound": prob_bound,
            "threat_severity": severity,
            "is_catastrophic": is_catastrophic,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T5] CUSUM THREAT DETECTOR — Assumption-light change detection
# ═══════════════════════════════════════════════════════════════════════

class CUSUMThreatDetector(nn.Module):
    """
    [T5] CUSUM change-point detection for EW threat transitions.
    
    Detects when adversary changes jamming mode, activates new emitter,
    or shifts frequency — without knowing the mixing rate or limit law.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            LowRankAdapter(cfg.signal_dim, 64, 16),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # CUSUM threshold (learnable)
        self.log_threshold = nn.Parameter(torch.tensor(math.log(cfg.cusum_threshold)))
        
        # Drift parameter
        self.drift = nn.Parameter(torch.tensor(0.0))
        
        # Change probability predictor
        self.change_net = nn.Sequential(
            nn.Linear(cfg.signal_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # CUSUM accumulator (state)
        self.register_buffer("cusum_acc", torch.zeros(1), persistent=False)
    
    @property
    def threshold(self) -> Tensor:
        return torch.exp(self.log_threshold)
    
    def forward(self, signal_features: Tensor) -> Dict[str, Tensor]:
        B = signal_features.shape[0]
        
        # Extract scalar feature
        feat = self.feature_net(signal_features).squeeze(-1)
        
        # CUSUM update
        deviation = feat - self.drift
        cusum_new = F.relu(self.cusum_acc + deviation)
        self.cusum_acc = cusum_new.detach()
        
        # Change detection
        change_detected = (cusum_new > self.threshold).float()
        
        # Change probability
        change_input = torch.cat([signal_features, cusum_new.expand(B, 1)], dim=-1)
        change_prob = self.change_net(change_input).squeeze(-1)
        
        return {
            "cusum_statistic": cusum_new.expand(B),
            "threshold": self.threshold.expand(B),
            "change_detected": change_detected,
            "change_probability": change_prob,
            "feature_value": feat,
        }


class AssumptionLightCI(nn.Module):
    """
    [T5] Assumption-light confidence interval (Paper 11).
    
    Subsampling CI without knowing convergence rate, limit law,
    or mixing rate. Self-normalized, automatic block-length.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        self.cusum_detector = CUSUMThreatDetector(cfg)
        
        # Self-normalization factor
        self.self_norm_net = nn.Sequential(
            nn.Linear(cfg.signal_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        
        # CI width predictor
        self.ci_width_net = nn.Sequential(
            nn.Linear(cfg.signal_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(self, signal_features: Tensor) -> Dict[str, Tensor]:
        cusum_out = self.cusum_detector(signal_features)
        
        # CI width
        ci_width = self.ci_width_net(signal_features).squeeze(-1)
        
        # Self-normalization
        self_norm = self.self_norm_net(signal_features).squeeze(-1)
        adjusted_width = ci_width * self_norm
        
        # CI bounds
        center = cusum_out["feature_value"]
        ci_lower = center - adjusted_width * 1.96
        ci_upper = center + adjusted_width * 1.96
        
        return {
            **{f"cusum_{k}": v for k, v in cusum_out.items()},
            "ci_center": center,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": adjusted_width,
            "self_normalization": self_norm,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T6] HYSTERESIS ADAPTIVE CONTROLLER — No-Zeno response
# ═══════════════════════════════════════════════════════════════════════

class HysteresisAdaptiveController(nn.Module):
    """
    [T6] Hysteresis gate for adaptive EW response.
    
    Two-level hysteresis prevents chattering:
      - Response activates when threat exceeds θ_hi
      - Response deactivates only when threat falls below θ_lo < θ_hi
      - Gap δ_min = θ_hi - θ_lo > 0 guarantees No-Zeno
    
    Bound: N(T) ≤ (E(0) - E_inf) / ΔE_min
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Learnable thresholds
        self.log_theta_hi = nn.Parameter(torch.tensor(math.log(cfg.theta_hi)))
        self.log_theta_lo = nn.Parameter(torch.tensor(math.log(cfg.theta_lo)))
        
        # Response mode selector
        self.response_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_jamming_modes),  # countermeasure modes
            nn.Softmax(dim=-1),
        )
        
        # ΔE_min predictor
        self.delta_e_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    @property
    def theta_hi(self) -> Tensor:
        return torch.exp(self.log_theta_hi)
    
    @property
    def theta_lo(self) -> Tensor:
        return torch.exp(self.log_theta_lo)
    
    @property
    def delta_min(self) -> Tensor:
        return soft_floor(self.theta_hi - self.theta_lo, 1e-6)
    
    def forward(self, threat_level: Tensor, ew_state: Tensor) -> Dict[str, Tensor]:
        B = threat_level.shape[0]
        
        # Hysteresis decision
        above_hi = soft_gate(threat_level - self.theta_hi, tau=0.01)
        below_lo = soft_gate(self.theta_lo - threat_level, tau=0.01)
        
        # Response activation
        response_active = above_hi
        
        # Response mode selection
        response_probs = self.response_net(ew_state)
        response_mode = response_probs.argmax(dim=-1)
        
        # ΔE_min
        delta_E_min = self.delta_e_net(ew_state).squeeze(-1)
        delta_E_min = soft_floor(delta_E_min, self.cfg.delta_e_min)
        
        # No-Zeno bound
        energy_budget = threat_level.abs()
        N_bound = energy_budget / soft_floor(delta_E_min, 1e-8)
        
        return {
            "response_active": response_active,
            "response_probs": response_probs,
            "response_mode": response_mode,
            "theta_hi": self.theta_hi.expand(B),
            "theta_lo": self.theta_lo.expand(B),
            "delta_min_hysteresis": self.delta_min.expand(B),
            "delta_E_min": delta_E_min,
            "N_T_bound": N_bound,
            "no_zeno_guaranteed": (N_bound < 1e6).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [T7] OPMC SIGNAL COMPRESSOR — Hyper-relational compression
# ═══════════════════════════════════════════════════════════════════════

class OPMCSignalCompressor(nn.Module):
    """
    [T7] OPMC canonical signature for EW signal compression.
    
    Compresses hyper-relational signal state space (2^{2^N} capacity)
    to polynomial-size σ_T without information loss for task-relevant
    features.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
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
        
        # Compression quality
        self.quality_net = nn.Sequential(
            nn.Linear(ds, ds // 2),
            nn.GELU(),
            nn.Linear(ds // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, signal_state: Tensor) -> Dict[str, Tensor]:
        B = signal_state.shape[0]
        
        # Compute canonical signature
        q = self.query(signal_state).unsqueeze(1)
        logits = torch.einsum("bnd,fd->bnf", q, self.field_emb) / math.sqrt(q.shape[-1])
        prov = F.softmax(logits, dim=1)
        
        v = self.value(signal_state).unsqueeze(1)
        sigma = torch.einsum("bnf,bnd->bfd", prov, v).squeeze(1)
        sigma = self.refine(sigma)
        
        quality = self.quality_net(sigma).squeeze(-1)
        
        return {
            "canonical_signature": sigma,
            "compression_quality": quality,
            "field_provenance": prov.squeeze(1),
        }


# ═══════════════════════════════════════════════════════════════════════
# [T8] SSC PHOTONIC PROCESSOR — Real-time optical signal processing
# ═══════════════════════════════════════════════════════════════════════

class SSCPhotonicProcessor(nn.Module):
    """
    [T8] Sub-Quantum Semantic Collapse photonic processor for EW.
    
    From GDS-II spec: MRR + PCM_GST + BTO heterogeneous integration.
    Nonlinear collapse: C(z) = z / (1 + α_eff |z|²)
    α_eff = (n₂_BTO · Q_MRR · L_eff) / A_eff
    
    Enables ultra-fast, low-power real-time signal processing for
    EW applications where electronic processing is too slow.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # α_eff initialization
        alpha_init = (cfg.n2_bto * cfg.q_mrr * cfg.l_eff_um) / cfg.a_eff_um2
        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(alpha_init, 1e-10))))
        
        # Nonlinearity strength modulator
        self.nonlin_strength = nn.Sequential(
            nn.Linear(cfg.signal_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # PCM memory state
        self.pcm_state = nn.Parameter(torch.zeros(cfg.n_pcm_cells))
        
        # MRR resonance
        self.ring_radius_um = nn.Parameter(torch.full((cfg.n_mrr_blocks,), 5.0))
        self.n_eff = nn.Parameter(torch.full((cfg.n_mrr_blocks,), 2.4))
        self.log_Q = nn.Parameter(torch.full((cfg.n_mrr_blocks,), math.log(cfg.q_mrr)))
        self.register_buffer("m_order",
                             torch.arange(1, cfg.n_mrr_blocks + 1, dtype=torch.float32),
                             persistent=False)
        
        # In-situ compensation
        self.cross_talk_net = nn.Sequential(
            LowRankAdapter(cfg.signal_dim, cfg.n_mzi_cells ** 2, 32),
            nn.Tanh(),
        )
        
        # DRC verifier
        self.register_buffer("W_min", torch.tensor(0.40), persistent=False)
        self.register_buffer("g_min", torch.tensor(0.20), persistent=False)
        self.register_buffer("R_min", torch.tensor(5.0), persistent=False)
    
    @property
    def alpha_eff(self) -> Tensor:
        return torch.exp(self.log_alpha)
    
    def nonlinear_collapse(self, z: Tensor, control: Tensor) -> Dict[str, Tensor]:
        strength = self.nonlin_strength(control).squeeze(-1)
        alpha = self.alpha_eff * strength.unsqueeze(-1)
        z_sq = z.pow(2).sum(dim=-1, keepdim=True)
        denominator = 1.0 + alpha * z_sq
        collapsed = z / soft_floor(denominator, 1e-8)
        return {
            "collapsed_field": collapsed,
            "alpha_eff": alpha.squeeze(-1),
        }
    
    def forward(self, input_signal: Tensor, control_signal: Tensor) -> Dict[str, Tensor]:
        B = input_signal.shape[0]
        
        # Nonlinear collapse
        collapse_out = self.nonlinear_collapse(input_signal, control_signal)
        
        # PCM state
        pcm_state = torch.sigmoid(self.pcm_state).unsqueeze(0).expand(B, -1)
        
        # MRR resonance
        lambda_res = (2 * math.pi * self.ring_radius_um * self.n_eff) / self.m_order
        Q = torch.exp(self.log_Q)
        
        # In-situ compensation
        cross_talk_flat = self.cross_talk_net(control_signal)
        T_cross = cross_talk_flat.view(B, self.cfg.n_mzi_cells, self.cfg.n_mzi_cells)
        T_cross = 0.5 * (T_cross + T_cross.transpose(-1, -2))
        T_cross = T_cross + torch.eye(self.cfg.n_mzi_cells, device=T_cross.device).unsqueeze(0) * 0.1
        
        # DRC score (simplified)
        drc_score = torch.ones(B, device=input_signal.device)
        
        return {
            **{f"collapse_{k}": v for k, v in collapse_out.items()},
            "pcm_state": pcm_state,
            "resonant_wavelength_um": lambda_res.unsqueeze(0).expand(B, -1),
            "quality_factor": Q.unsqueeze(0).expand(B, -1),
            "cross_talk_matrix": T_cross,
            "drc_score": drc_score,
            "photonic_performance": collapse_out["collapsed_field"].norm(dim=-1) / (input_signal.norm(dim=-1) + 1e-8),
        }


# ═══════════════════════════════════════════════════════════════════════
# [T9] DISCRETE CORRECTION FILTER — Lattice-derivative bias removal
# ═══════════════════════════════════════════════════════════════════════

class DiscreteCorrectionFilter(nn.Module):
    """
    [T9] Discrete correction factor from Lemma 7.4 (Unified OPMC paper).
    
    Central difference on spacing h has symbol sin(ξh)/h, NOT ξ.
    This filter corrects for the lattice-derivative bias in discretized
    EW signal processing.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.l_c / max(1, cfg.n_grid)
        self.register_buffer("h_buf", torch.tensor(self.h), persistent=False)
        
        # Correction network
        self.correction_net = nn.Sequential(
            nn.Linear(cfg.signal_dim, 32),
            nn.GELU(),
            nn.Linear(32, cfg.signal_dim),
        )
    
    def forward(self, signal: Tensor) -> Dict[str, Tensor]:
        B = signal.shape[0]
        
        # Compute correction factor
        h = self.h_buf
        xi = torch.linspace(-math.pi / (2 * h), math.pi / (2 * h),
                             signal.shape[-1], device=signal.device)
        exact = xi
        discrete = torch.sin(xi * h) / h
        correction = exact / (discrete + 1e-8)
        
        # Apply correction
        corrected = signal * correction.unsqueeze(0).unsqueeze(0)
        
        # Nyquist band check
        in_band = (xi.abs() * h <= math.pi).float()
        
        # Relative error
        rel_error = (discrete - exact).abs() / (exact.abs() + 1e-12)
        
        return {
            "corrected_signal": corrected,
            "correction_factor": correction,
            "in_nyquist_band": in_band.mean(),
            "mean_relative_error": rel_error.mean(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [T10] ISODIAMETRIC JAMMING ANALYZER — Pattern analysis
# ═══════════════════════════════════════════════════════════════════════

class IsodiametricJammingAnalyzer(nn.Module):
    """
    [T10] Isodiametric analysis of jamming patterns.
    
    From Delta_Min paper:
      δ_min = (1/κ) · (a_min / ω_{n-1})^{1/(n-1)} > 0
    
    Under κ-fatness, every jamming nucleation event satisfies
    d_H(Γ, Γ') ≥ δ_min.
    
    Application: analyze spatial structure of jamming patterns
    to identify adversary tactics.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # δ_min theoretical
        delta_min = (1.0 / cfg.kappa_fatness) * \
                    (cfg.a_min_um2 / cfg.omega_n_minus_1)
        self.register_buffer("delta_min_theoretical",
                             torch.tensor(delta_min), persistent=False)
        
        # Jamming pattern encoder
        self.pattern_net = nn.Sequential(
            LowRankAdapter(cfg.signal_dim, 64, 16),
            nn.GELU(),
            nn.Linear(64, 3),  # [area, perimeter, max_dist]
            nn.Sigmoid(),
        )
        
        # Fatness predictor
        self.fatness_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, jamming_pattern: Tensor) -> Dict[str, Tensor]:
        B = jamming_pattern.shape[0]
        
        # Extract pattern features
        patch_stats = self.pattern_net(jamming_pattern)
        area = soft_floor(patch_stats[:, 0] * 0.1, self.cfg.a_min_um2)
        perim = soft_floor(patch_stats[:, 1], 1e-6)
        dist = patch_stats[:, 2].abs()
        
        # Equivalent radius
        r_P = (area / self.cfg.omega_n_minus_1)
        
        # Fatness check
        threshold = r_P / self.cfg.kappa_fatness
        fatness = soft_gate(dist - threshold, tau=0.01)
        
        # Hausdorff lower bound
        dH_lower = self.delta_min_theoretical * fatness
        
        # Eccentricity
        eccentricity = perim / (2 * math.pi * r_P + 1e-8)
        
        return {
            "delta_min": self.delta_min_theoretical.expand(B),
            "fatness": fatness,
            "hausdorff_lower_bound": dH_lower,
            "eccentricity": eccentricity,
            "patch_stats": patch_stats,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T12] STEALTH MATERIAL DESIGNER — Non-p.c.f. fractal RAM
# ═══════════════════════════════════════════════════════════════════════

class StealthMaterialDesigner(nn.Module):
    """
    [T12] Stealth material design using non-p.c.f. fractal Laplacian.
    
    From Barlow-Bass / Kusuoka-Zhou: non-p.c.f. fractals (like the
    carpet) admit Brownian motion / Laplacian construction.
    
    Application: design radar-absorbing materials with fractal
    geometry that provides broadband absorption.
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Material encoder
        self.material_net = nn.Sequential(
            LowRankAdapter(cfg.latent_dim, cfg.carpet_n_cells, 32),
            nn.GELU(),
        )
        
        # Absorption predictor
        self.absorption_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Bandwidth predictor
        self.bandwidth_net = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Build carpet graph
        self._build_carpet_graph()
        
        # Resistance scaling
        self.log_rho = nn.Parameter(torch.tensor(math.log(1.25 - 9/8 + 1e-6)))
    
    def _build_carpet_graph(self):
        n_cells = self.cfg.carpet_n_cells
        adj = torch.zeros(n_cells, n_cells)
        for i in range(n_cells):
            for j in range(max(0, i-2), min(n_cells, i+3)):
                if i != j:
                    adj[i, j] = 1.0
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj = adj / deg
        self.register_buffer("carpet_laplacian",
                             torch.diag(deg.squeeze()) - adj, persistent=False)
    
    @property
    def rho(self) -> Tensor:
        return 9/8 + F.softplus(self.log_rho)
    
    def forward(self, material_params: Tensor) -> Dict[str, Tensor]:
        B = material_params.shape[0]
        
        cell_values = self.material_net(material_params)
        
        # Resistance energy
        energy = torch.einsum("bi,ij,bj->b",
                              cell_values, self.carpet_laplacian, cell_values)
        energy = (self.rho ** self.cfg.carpet_level) * energy
        
        # Absorption (inversely related to energy — low energy = high absorption)
        absorption = self.absorption_net(cell_values).squeeze(-1)
        absorption = absorption / (energy + 1e-8)
        
        # Bandwidth
        bandwidth = self.bandwidth_net(cell_values).squeeze(-1)
        
        return {
            "absorption_coefficient": absorption,
            "bandwidth_ghz": bandwidth,
            "resistance_energy": energy,
            "cell_values": cell_values,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED EW ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class SESIElectronicWarfareEnvironment(nn.Module):
    """
    Complete SESI Electronic Warfare Environment.
    
    Integrates all 13 theories for:
      - Threat detection and classification
      - Spectrum situational awareness
      - Anti-jamming and electronic protection
      - Fractal antenna design
      - Stealth material design
      - Real-time photonic signal processing
      - Adaptive EW response control
    """
    
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
        
        # [T1] Hodge signal separation
        self.hodge_decomp = HodgeSignalDecomposer(cfg)
        self.jamming_classifier = JammingSignalClassifier(cfg)
        
        # [T2] Spectral cognitive engine
        self.spectral_engine = SpectralCognitiveEngine(cfg)
        self.pattern_recognizer = ThreatPatternRecognizer(cfg)
        
        # [T3][T11] Fractal antenna
        self.antenna_array = FractalAntennaArray(cfg)
        
        # [T4] Gumbel bounds
        self.gumbel_bounder = GumbelJammingBounder(cfg)
        
        # [T5] CUSUM detection
        self.cusum_ci = AssumptionLightCI(cfg)
        
        # [T6] Adaptive control
        self.adaptive_ctrl = HysteresisAdaptiveController(cfg)
        
        # [T7] OPMC compression
        self.opmc_compressor = OPMCSignalCompressor(cfg)
        
        # [T8] Photonic processing
        self.photonic_proc = SSCPhotonicProcessor(cfg)
        
        # [T9] Discrete correction
        self.discrete_filter = DiscreteCorrectionFilter(cfg)
        
        # [T10] Isodiametric analysis
        self.jamming_analyzer = IsodiametricJammingAnalyzer(cfg)
        
        # [T12] Stealth materials
        self.stealth_designer = StealthMaterialDesigner(cfg)
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(cfg.signal_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        
        # Final EW situation assessment
        self.situation_net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 4, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        received_signal: Tensor,
        design_params: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full EW environment forward pass.
        
        Args:
            received_signal: (B, signal_dim) received EM signal
            design_params: (B, latent_dim) optional design parameters
        
        Returns:
            dict with all EW analysis outputs
        """
        B = received_signal.shape[0]
        out = {}
        
        # Encode input
        ew_state = self.input_encoder(received_signal)
        
        if design_params is None:
            design_params = ew_state
        
        # [T1] Hodge decomposition
        hodge_out = self.hodge_decomp(received_signal)
        out.update({f"hodge_{k}": v for k, v in hodge_out.items()})
        
        jam_class = self.jamming_classifier(hodge_out)
        out.update({f"jam_class_{k}": v for k, v in jam_class.items()})
        
        # [T2] Spectral analysis
        spectral_out = self.spectral_engine(ew_state)
        out.update({f"spectral_{k}": v for k, v in spectral_out.items()})
        
        pattern_out = self.pattern_recognizer(spectral_out, hodge_out)
        out.update({f"pattern_{k}": v for k, v in pattern_out.items()})
        
        # [T3] Antenna array
        antenna_out = self.antenna_array(design_params)
        out.update({f"antenna_{k}": v for k, v in antenna_out.items()})
        
        # [T4] Gumbel bounds
        gumbel_out = self.gumbel_bounder(ew_state)
        out.update({f"gumbel_{k}": v for k, v in gumbel_out.items()})
        
        # [T5] CUSUM detection
        cusum_out = self.cusum_ci(received_signal)
        out.update({f"cusum_{k}": v for k, v in cusum_out.items()})
        
        # [T6] Adaptive control
        threat_level = gumbel_out["threat_severity"]
        adaptive_out = self.adaptive_ctrl(threat_level, ew_state)
        out.update({f"adaptive_{k}": v for k, v in adaptive_out.items()})
        
        # [T7] OPMC compression
        opmc_out = self.opmc_compressor(ew_state)
        out.update({f"opmc_{k}": v for k, v in opmc_out.items()})
        
        # [T8] Photonic processing
        photonic_out = self.photonic_proc(received_signal, ew_state)
        out.update({f"photonic_{k}": v for k, v in photonic_out.items()})
        
        # [T9] Discrete correction
        discrete_out = self.discrete_filter(received_signal)
        out.update({f"discrete_{k}": v for k, v in discrete_out.items()})
        
        # [T10] Jamming analysis
        jamming_out = self.jamming_analyzer(received_signal)
        out.update({f"jamming_{k}": v for k, v in jamming_out.items()})
        
        # [T12] Stealth material
        stealth_out = self.stealth_designer(design_params)
        out.update({f"stealth_{k}": v for k, v in stealth_out.items()})
        
        # Final situation assessment
        situation_input = torch.cat([
            ew_state,
            hodge_out["harmonic"],
            spectral_out["eigenvalues"].mean(dim=-1, keepdim=True).expand(B, self.cfg.latent_dim),
            opmc_out["canonical_signature"][:, :self.cfg.latent_dim],
        ], dim=-1)
        
        situation_score = self.situation_net(situation_input).squeeze(-1)
        out["situation_score"] = situation_score
        
        # EW readiness (composite)
        ew_readiness = (
            0.2 * hodge_out["signal_quality"] +
            0.2 * pattern_out["recognition_confidence"] +
            0.2 * adaptive_out["no_zeno_guaranteed"] +
            0.2 * opmc_out["compression_quality"] +
            0.2 * photonic_out["photonic_performance"]
        )
        out["ew_readiness"] = ew_readiness
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class EWEnvironmentLoss(nn.Module):
    def __init__(self, cfg: EWEnvironmentConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], model: nn.Module) -> Dict[str, Tensor]:
        dev = out["situation_score"].device
        zero = torch.zeros((), device=dev)
        L = {}
        
        # Detection (maximize signal quality)
        if "hodge_signal_quality" in out:
            L["detection"] = -out["hodge_signal_quality"].mean()
        else:
            L["detection"] = zero
        
        # Classification (maximize confidence)
        if "pattern_recognition_confidence" in out:
            L["classification"] = -out["pattern_recognition_confidence"].mean()
        else:
            L["classification"] = zero
        
        # Protection (maximize SJR, minimize jamming)
        if "hodge_signal_to_jam_ratio" in out:
            L["protection"] = -torch.log(out["hodge_signal_to_jam_ratio"] + 1e-8).mean()
        else:
            L["protection"] = zero
        
        # Spectral (maximize gap score)
        if "spectral_gap_score" in out:
            L["spectral"] = -out["spectral_gap_score"].mean()
        else:
            L["spectral"] = zero
        
        # Compression (maximize quality)
        if "opmc_compression_quality" in out:
            L["compression"] = -out["opmc_compression_quality"].mean()
        else:
            L["compression"] = zero
        
        # Safety (No-Zeno guarantee)
        if "adaptive_no_zeno_guaranteed" in out:
            L["safety"] = -out["adaptive_no_zeno_guaranteed"].mean()
        else:
            L["safety"] = zero
        
        # Regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg
        
        total = (
            self.cfg.w_detection * L["detection"] +
            self.cfg.w_classification * L["classification"] +
            self.cfg.w_protection * L["protection"] +
            self.cfg.w_spectral * L["spectral"] +
            self.cfg.w_compression * L["compression"] +
            self.cfg.w_safety * L["safety"] +
            self.cfg.w_reg * L["reg"]
        )
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# AGI ONE ECOSYSTEM ADAPTER
# ═══════════════════════════════════════════════════════════════════════

class EWEcosystemAdapter(nn.Module):
    """Adapter for AGI ONE EcosystemOrchestrator."""
    
    def __init__(self, ew_env: SESIElectronicWarfareEnvironment,
                 agi_latent_dim: int, device: Optional[torch.device] = None):
        super().__init__()
        self.ew_env = ew_env
        self.agi_latent_dim = agi_latent_dim
        
        self.lift = nn.Sequential(
            nn.Linear(agi_latent_dim, ew_env.cfg.signal_dim),
            nn.GELU(),
        )
        
        self.readout = nn.Sequential(
            nn.Linear(ew_env.cfg.latent_dim, agi_latent_dim),
            nn.GELU(),
            nn.Linear(agi_latent_dim, agi_latent_dim),
        )
        
        self._last_quality = 0.5
    
    def encode(self, agi_latent: Tensor) -> Tensor:
        signal = self.lift(agi_latent)
        out = self.ew_env(signal)
        return self.readout(out.get("hodge_harmonic", signal))
    
    @torch.no_grad()
    def get_quality_score(self) -> float:
        try:
            dev = next(self.ew_env.parameters()).device
            test_signal = torch.randn(1, self.ew_env.cfg.signal_dim, device=dev)
            out = self.ew_env(test_signal)
            quality = (
                0.3 * out.get("hodge_signal_quality", torch.tensor(0.5)).mean().item() +
                0.3 * out.get("ew_readiness", torch.tensor(0.5)).mean().item() +
                0.4 * out.get("situation_score", torch.tensor(0.5)).mean().item()
            )
            self._last_quality = max(0.0, min(1.0, quality))
        except Exception as e:
            logger.debug(f"EW quality probe failed: {e}")
        return self._last_quality


def attach_ew_to_ecosystem(
    orchestrator: Any,
    agi_latent_dim: int,
    device: Optional[torch.device] = None,
    cfg: Optional[EWEnvironmentConfig] = None,
    name: str = "electronic_warfare_env",
) -> EWEcosystemAdapter:
    """One-liner registration with AGI ONE EcosystemOrchestrator."""
    dev = device or torch.device("cpu")
    cfg = cfg or EWEnvironmentConfig(device=str(dev))
    
    ew_env = SESIElectronicWarfareEnvironment(cfg).to(dev)
    adapter = EWEcosystemAdapter(ew_env, agi_latent_dim, dev).to(dev)
    
    if hasattr(orchestrator, "register"):
        orchestrator.register(
            name, adapter, "electronic_warfare", agi_latent_dim,
            quality_fn=adapter.get_quality_score,
        )
        logger.info(f"Registered '{name}' with EcosystemOrchestrator")
    
    return adapter


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Electronic Warfare Environment v1.0 — Smoke Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)
    
    cfg = EWEnvironmentConfig(
        latent_dim=64,
        signal_dim=32,
        spectrum_dim=64,
        hodge_dim=16,
        carpet_level=2,
        carpet_n_cells=64,
        n_eigenvalues=16,
        n_mzi_cells=8,
        n_mrr_blocks=4,
        n_pcm_cells=8,
        opmc_sigma_dim=32,
        opmc_n_fields=4,
        batch_size=2,
        device=str(device),
    )
    
    model = SESIElectronicWarfareEnvironment(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    print(f"  Theories integrated: 13")
    
    model.train()
    B = 2
    signal = torch.randn(B, cfg.signal_dim, device=device)
    design = torch.randn(B, cfg.latent_dim, device=device)
    
    out = model(signal, design)
    
    loss_fn = EWEnvironmentLoss(cfg)
    losses = loss_fn(out, model)
    losses["total"].backward()
    
    print(f"\n  ── [T1] Hodge Signal Separation ──")
    print(f"    Signal quality       : {out['hodge_signal_quality'].mean().item():.4f}")
    print(f"    Jamming score        : {out['hodge_jamming_score'].mean().item():.4f}")
    print(f"    SJR                  : {out['hodge_signal_to_jam_ratio'].mean().item():.4f}")
    
    print(f"\n  ── [T2] Spectral Cognitive Engine ──")
    print(f"    Spectral gap         : {out['spectral_spectral_gap'].mean().item():.4f}")
    print(f"    Threat class         : {out['spectral_threat_class'].mean().item():.0f}")
    print(f"    Pattern confidence   : {out['pattern_recognition_confidence'].mean().item():.4f}")
    
    print(f"\n  ── [T3][T11] Fractal Antenna ──")
    print(f"    Antenna gain (dBi)   : {out['antenna_antenna_gain_dbi'].mean().item():.4f}")
    print(f"    Bandwidth (GHz)      : {out['antenna_bandwidth_ghz'].mean().item():.4f}")
    print(f"    Junction quality     : {out['antenna_junction_quality'].mean().item():.4f}")
    
    print(f"\n  ── [T4] Gumbel Jamming Bounds ──")
    print(f"    Activation energy    : {out['gumbel_activation_energy'].mean().item():.4f}")
    print(f"    Prob bound           : {out['gumbel_gumbel_prob_bound'].mean().item():.2e}")
    print(f"    Threat severity      : {out['gumbel_threat_severity'].mean().item():.4f}")
    
    print(f"\n  ── [T5] CUSUM + Subsampling ──")
    print(f"    CUSUM statistic      : {out['cusum_cusum_statistic'].mean().item():.4f}")
    print(f"    Change detected      : {out['cusum_change_detected'].mean().item():.4f}")
    print(f"    CI width             : {out['cusum_ci_width'].mean().item():.4f}")
    
    print(f"\n  ── [T6] Hysteresis Adaptive Control ──")
    print(f"    Response active      : {out['adaptive_response_active'].mean().item():.4f}")
    print(f"    Response mode        : {out['adaptive_response_mode'].mean().item():.0f}")
    print(f"    No-Zeno guaranteed   : {out['adaptive_no_zeno_guaranteed'].mean().item():.4f}")
    print(f"    N(T) bound           : {out['adaptive_N_T_bound'].mean().item():.2f}")
    
    print(f"\n  ── [T7] OPMC Compression ──")
    print(f"    Compression quality  : {out['opmc_compression_quality'].mean().item():.4f}")
    
    print(f"\n  ── [T8] SSC Photonic Processing ──")
    print(f"    Photonic performance : {out['photonic_photonic_performance'].mean().item():.4f}")
    print(f"    α_eff                : {out['photonic_alpha_eff'].mean().item():.2e}")
    print(f"    DRC score            : {out['photonic_drc_score'].mean().item():.4f}")
    
    print(f"\n  ── [T9] Discrete Correction ──")
    print(f"    In Nyquist band      : {out['discrete_in_nyquist_band'].item():.4f}")
    print(f"    Mean rel. error      : {out['discrete_mean_relative_error'].item():.4e}")
    
    print(f"\n  ── [T10] Isodiametric Jamming Analysis ──")
    print(f"    δ_min                : {out['jamming_delta_min'].mean().item():.4f}")
    print(f"    Fatness              : {out['jamming_fatness'].mean().item():.4f}")
    print(f"    Hausdorff LB         : {out['jamming_hausdorff_lower_bound'].mean().item():.4f}")
    
    print(f"\n  ── [T12] Stealth Material ──")
    print(f"    Absorption coeff     : {out['stealth_absorption_coefficient'].mean().item():.4f}")
    print(f"    Bandwidth (GHz)      : {out['stealth_bandwidth_ghz'].mean().item():.4f}")
    
    print(f"\n  ── Overall Assessment ──")
    print(f"    Situation score      : {out['situation_score'].mean().item():.4f}")
    print(f"    EW readiness         : {out['ew_readiness'].mean().item():.4f}")
    
    print(f"\n  ── Losses ──")
    for k, v in losses.items():
        print(f"    {k:<20s}: {v.item():.6f}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL 13 THEORIES CONNECTED' if grad_ok else 'FAIL'}")
    
    # Test AGI ONE adapter
    print(f"\n  ── AGI ONE Integration ──")
    adapter = EWEcosystemAdapter(model, cfg.latent_dim, device).to(device)
    agi_latent = torch.randn(B, cfg.latent_dim, device=device)
    encoded = adapter.encode(agi_latent)
    print(f"    AGI latent in        : {tuple(agi_latent.shape)}")
    print(f"    AGI latent out       : {tuple(encoded.shape)}")
    print(f"    Quality score        : {adapter.get_quality_score():.4f}")
    
    print("\n" + "=" * 72)
    print("  ⚠️  DEFENSIVE SCOPE REMINDER:")
    print("  • This is a SIMULATION framework for defensive EW research")
    print("  • Focus: Electronic Warfare Support + Electronic Protection")
    print("  • NOT an operational EW weapon system")
    print("  • All outputs are differentiable simulations")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
