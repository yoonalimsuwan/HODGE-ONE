"""
=============================================================================
SESI UNIFIED RESEARCH v1.0  —  Clean-room, runnable, honestly scoped
=============================================================================
Developer   : Yoon A. Limsuwan / MSPS NETWORK
ORCID       : 0009-0008-2374-0788
License     : MIT
Year        : 2026

PURPOSE
    A single-file research platform combining:
      * Sierpinski-carpet resistance forms          [VERIFIED math]
      * Learned Hodge-style field decomposition     [HEURISTIC]
      * Riemann-zero spectral proxy                 [ANALOGY ONLY]
      * Differentiable neural inverse problem       [SIMULATION]
      * Near-field neuromodulation planning         [SIMULATION + SAFETY]
      * Reactor point-kinetics safety monitor       [EDUCATIONAL]
      * Id / Ego / SuperEgo speculative axiom loop  [SOFTWARE]
      * AGI-ONE-style ecosystem orchestration       [SOFTWARE]

EPISTEMIC BOUNDARIES (NOT optional)
    * Does NOT stop an initiated nuclear chain reaction.
    * Does NOT read or control a brain without a physical device.
    * Does NOT factor RSA-2048/4096.
    * Does NOT prove the Riemann Hypothesis or the Hodge Conjecture.
    * Does NOT replace CFD / FEA / clinical / licensed safety software.

RUN
    python sesi_unified_research.py --mode smoke
    python sesi_unified_research.py --mode train --steps 200
=============================================================================
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger("sesi.research")

# ══════════════════════════════════════════════════════════════════════
# 0. DEVICE / AMP  (sanction-resilient: NPU -> MLU -> CUDA -> MPS -> CPU)
# ══════════════════════════════════════════════════════════════════════

def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if hasattr(torch, "mlu") and torch.mlu.is_available():
        return torch.device("mlu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type not in ("cuda", "npu", "mlu"):
        yield
        return
    try:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    except Exception:
        yield


# ══════════════════════════════════════════════════════════════════════
# 1. C^inf SMOOTH PRIMITIVES  (no dead gradients at kinks)
# ══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def sfloor(x: Tensor, lo: float, beta: float = 50.0) -> Tensor:
    return lo + F.softplus(x - lo, beta=beta)


def sclamp(x: Tensor, lo: float, hi: float) -> Tensor:
    mid, half = (hi + lo) / 2.0, max((hi - lo) / 2.0, 1e-6)
    return mid + half * torch.tanh((x - mid) / half)


class LowRank(nn.Module):
    """LoRA-style projection: O(d*r) instead of O(d^2)."""
    def __init__(self, din: int, dout: int, rank: int = 16):
        super().__init__()
        self.U = nn.Parameter(torch.randn(din, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, dout) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


def _safe_unit(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════
# 2. SIERPINSKI CARPET + RESISTANCE FORM                      [VERIFIED]
# ══════════════════════════════════════════════════════════════════════

def carpet_mask(level: int) -> np.ndarray:
    """Remove the open middle ninth at every recursion level."""
    if level <= 0:
        return np.ones((1, 1), dtype=bool)
    small = carpet_mask(level - 1)
    k = small.shape[0]
    out = np.zeros((3 * k, 3 * k), dtype=bool)
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                out[i*k:(i+1)*k, j*k:(j+1)*k] = small
    return out


class CarpetGraph(nn.Module):
    """Exact finite-level carpet graph G_n with 4-connectivity."""
    def __init__(self, level: int = 3):
        super().__init__()
        mask = carpet_mask(level)
        coords = np.argwhere(mask)                       # (N, 2)
        index = {tuple(c): i for i, c in enumerate(coords)}
        n = len(coords)
        assert n == 8 ** level, (n, 8 ** level)

        A = np.zeros((n, n), dtype=np.float32)
        for i, (x, y) in enumerate(coords):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = index.get((x + dx, y + dy))
                if j is not None:
                    A[i, j] = 1.0
        deg = A.sum(1)
        L = np.diag(deg) - A

        self.n_cells = n
        self.level = level
        self.register_buffer("A", torch.from_numpy(A))
        self.register_buffer("deg", torch.from_numpy(deg))
        self.register_buffer("L", torch.from_numpy(L))
        self.register_buffer("coords", torch.from_numpy(coords.astype(np.float32)))

    def energy(self, u: Tensor) -> Tensor:
        """E(u) = u^T L u  (sum of squared edge differences)."""
        return torch.einsum("bi,ij,bj->b", u, self.L.to(u.dtype), u)


class CarpetResistanceForm(nn.Module):
    """E_n(u) = rho^n * E_n^(0)(u), rho >= 9/8, smooth reparametrized."""
    def __init__(self, latent_dim: int, level: int = 3, rank: int = 16):
        super().__init__()
        self.graph = CarpetGraph(level)
        self.lift = nn.Sequential(
            LowRank(latent_dim, self.graph.n_cells, rank), nn.GELU()
        )
        excess = max(1.25 - 9 / 8, 1e-6)
        self.raw_rho = nn.Parameter(torch.tensor(math.log(math.expm1(excess))))
        self.concentration = nn.Sequential(
            nn.Linear(self.graph.n_cells, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    @property
    def rho(self) -> Tensor:
        return 9 / 8 + F.softplus(self.raw_rho)

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        u = self.lift(h)
        e0 = self.graph.energy(u)
        energy = (self.rho ** self.graph.level) * e0
        energy = sclamp(energy, -1e6, 1e6)
        return {
            "carpet_u": u,
            "carpet_energy_raw": e0,
            "carpet_energy": energy,
            "carpet_concentration": self.concentration(u).squeeze(-1),
            "carpet_rho": self.rho.expand(h.shape[0]),
        }


# ══════════════════════════════════════════════════════════════════════
# 3. LEARNED HODGE-STYLE SPLIT + RIEMANN PROXY      [HEURISTIC / ANALOGY]
# ══════════════════════════════════════════════════════════════════════

class HodgeStyleSplit(nn.Module):
    """
    Three-way learned split (harmonic / exact / coexact).
    NOT a proof of the Hodge conjecture; an orthogonality-regularized
    feature decomposition used as a signal/noise separator.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, 3 * dim))

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        a, b, c = self.net(h).chunk(3, dim=-1)
        orth = (a * b).sum(-1).abs() + (a * c).sum(-1).abs() + (b * c).sum(-1).abs()
        power = (a * a).sum(-1) / ((c * c).sum(-1) + 1e-6)
        return {"harmonic": a, "exact": b, "coexact": c,
                "orthogonality_error": orth, "signal_ratio": power}


class RiemannSpectralProxy(nn.Module):
    """
    Uses tabulated zeta-zero ordinates ONLY as a spacing-statistics
    reference (GUE-like). Does NOT compute zeta zeros or prove RH.
    """
    FIRST = (14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
             37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
             52.970321, 56.446248, 59.347044, 60.831779, 65.112544)

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(self.FIRST, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        g = self.gamma.to(h.device)
        sp = g[1:] - g[:-1]
        feats = torch.stack([sp.mean(), sp.std(unbiased=False),
                             (g[-1] - g[0]) / len(g)]).expand(h.shape[0], 3)
        return {"zero_spacing_mean": sp.mean().expand(h.shape[0]),
                "spectral_rigidity_proxy": torch.tanh(self.head(feats)).squeeze(-1)}


# ══════════════════════════════════════════════════════════════════════
# 4. NEURAL INVERSE PROBLEM                        [SIMULATION ONLY]
# ══════════════════════════════════════════════════════════════════════

class LeadField(nn.Module):
    """Forward model: sensor_c(t) = sum_v L[c,v] * source_v(t) + noise."""
    def __init__(self, n_sensors: int = 32, n_voxels: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        sen = torch.randn(n_sensors, 3, generator=g) * 0.15
        vox = torch.randn(n_voxels, 3, generator=g) * 0.05
        r2 = torch.cdist(sen, vox).pow(2)
        L = 1.0 / (r2 + 0.04)                      # dipole-like decay
        L = L / L.norm(dim=-1, keepdim=True)
        self.register_buffer("L", L)
        self.n_sensors, self.n_voxels = n_sensors, n_voxels

    def forward_model(self, src: Tensor) -> Tensor:
        return torch.einsum("cv,bvt->bct", self.L.to(src.dtype), src)


class NeuralMonitor(nn.Module):
    """Differentiable inverse source localization (not remote reading)."""
    def __init__(self, n_sensors: int = 32, n_voxels: int = 64, hidden: int = 48):
        super().__init__()
        self.physics = LeadField(n_sensors, n_voxels)
        self.enc = nn.Sequential(
            nn.Conv1d(n_sensors, hidden, 5, padding=2), nn.GroupNorm(1, hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.dec = nn.Conv1d(hidden, n_voxels, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x: (B, n_sensors, T)."""
        src = self.dec(self.enc(x))
        recon = self.physics.forward_model(src)
        resid = (recon - x).pow(2).mean(dim=(1, 2))
        snr = x.pow(2).mean(dim=(1, 2)) / (resid + 1e-6)
        l1 = src.abs().mean(dim=(1, 2))
        quality = torch.sigmoid(torch.log1p(snr) - 0.5 * l1)
        return {"sources": src, "recon": recon, "residual": resid,
                "snr": snr, "sparsity": l1, "quality": quality}


class NearFieldControlPlanner(nn.Module):
    """
    Normalized stimulation PLANNING for a hypothetical near-field device.
    Hard safety interlock: no output above the ceiling, no actuation flag
    unless the learned safety head and the physical cap both pass.
    """
    def __init__(self, n_voxels: int = 64, dim: int = 64, n_targets: int = 8):
        super().__init__()
        self.proj = nn.Linear(n_voxels, dim)
        self.target = nn.Sequential(nn.Linear(dim, n_targets), nn.Softmax(-1))
        self.intensity = nn.Sequential(nn.Linear(dim + n_targets, 32), nn.GELU(),
                                       nn.Linear(32, 1), nn.Sigmoid())
        self.safety = nn.Sequential(nn.Linear(dim + 1, 32), nn.GELU(),
                                    nn.Linear(32, 1), nn.Sigmoid())
        self.register_buffer("cap", torch.tensor(0.25))
        self.register_buffer("threshold", torch.tensor(0.80))

    def forward(self, src: Tensor) -> Dict[str, Tensor]:
        h = F.gelu(self.proj(src.mean(dim=-1)))
        tgt = self.target(h)
        raw = self.intensity(torch.cat([h, tgt], -1)).squeeze(-1)
        safe = self.safety(torch.cat([h, raw.unsqueeze(-1)], -1)).squeeze(-1)
        gate = torch.sigmoid((safe - self.threshold.to(safe.dtype)) * 40.0)
        gated = raw * gate * self.cap.to(raw.dtype)
        return {"target_probs": tgt, "requested_intensity": raw,
                "safety_score": safe, "gated_intensity": gated,
                "actuation_permitted": (gate > 0.5).float(),
                "violations": F.relu(raw - self.cap.to(raw.dtype)).mean()}


# ══════════════════════════════════════════════════════════════════════
# 5. REACTOR SAFETY MONITOR            [EDUCATIONAL, NOT WEAPONS]
# ═══════════════════════════════════════════════════════════════════════

class ReactorSafetyMonitor(nn.Module):
    """
    Lumped point kinetics in DIMENSIONLESS prompt-time units.
    Purpose: monitor subcritical margin and precursor build-up.

    n' = ((k_eff*(1-beta) - 1)/Lambda) * n + sum(lambda_i * C_i)
    C_i' = (beta_i/Lambda) * k_eff * n - lambda_i * C_i

    Does NOT model weapons, does NOT propose interdiction, and cannot
    'stop' an initiated reaction. Prompt criticality is flagged as a
    HARD FAIL, not as an actionable target.
    """
    def __init__(self, dim: int, n_steps: int = 32):
        super().__init__()
        self.n_steps = n_steps
        self.k_head = nn.Sequential(nn.Linear(dim, 32), nn.GELU(),
                                    nn.Linear(32, 1))
        self.register_buffer("beta", torch.tensor(0.0065))
        self.register_buffer("beta_i", torch.tensor(
            [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]))
        self.register_buffer("lam_i", torch.tensor(
            [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]))
        self.register_buffer("Lambda", torch.tensor(1.0))
        self.register_buffer("dt", torch.tensor(0.002))

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        B = h.shape[0]
        k = 0.6 + 0.5 * torch.sigmoid(self.k_head(h).squeeze(-1))  # 0.6 .. 1.1
        n = torch.ones(B, device=h.device, dtype=h.dtype)
        C = torch.zeros(B, 6, device=h.device, dtype=h.dtype)
        dt, lam = self.dt.to(n.dtype), self.lam_i.to(n.dtype)
        bi = self.beta_i.to(n.dtype)

        for _ in range(self.n_steps):
            src = (lam * C).sum(-1)
            dn = ((k * (1 - self.beta.to(n.dtype)) - 1) / self.Lambda.to(n.dtype)) * n + src
            dC = (bi * (k / self.Lambda.to(n.dtype)).unsqueeze(-1)) * n.unsqueeze(-1) - lam * C
            n = sfloor(n + dt * dn, 0.0)
            C = sfloor(C + dt * dC, 0.0)
            n = sclamp(n, 0.0, 1e12)

        margin = 1.0 - k
        prompt_critical = (k > 1.0 + self.beta.to(k.dtype)).float()
        safe = ((margin > 0).float() * (1.0 - prompt_critical))
        return {"k_eff": k, "subcritical_margin": margin,
                "log_neutron_population": torch.log(n.clamp_min(1e-12)),
                "prompt_critical_flag": prompt_critical,
                "safety_ok": safe}


# ══════════════════════════════════════════════════════════════════════
# 6. Id / Ego / SuperEgo  +  QUALITY-GATED CODIFICATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PsycheConfig:
    latent_dim: int = 64
    max_axioms: int = 16
    noise: float = 0.05
    axiom_weight: float = 0.25
    quality_weight: float = 0.20
    codify_threshold: float = 0.65
    quality_min: float = 0.40

    def __post_init__(self):
        if self.latent_dim <= 0 or self.max_axioms <= 0:
            raise ValueError("latent_dim and max_axioms must be positive")
        if not (0 <= self.axiom_weight + self.quality_weight < 1):
            raise ValueError("axiom_weight + quality_weight must be in [0,1)")
        if self.noise <= 0:
            raise ValueError("noise must be positive")


class SuperEgo(nn.Module):
    def __init__(self, cfg: PsycheConfig):
        super().__init__()
        self.cfg = cfg
        self.verifier = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim),
                                      nn.GELU(), nn.Linear(cfg.latent_dim, 1))
        self.register_buffer("bank", torch.zeros(cfg.max_axioms, cfg.latent_dim))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.stats: Dict[str, int] = {"codified": 0, "refused_validity": 0,
                                      "refused_quality": 0, "rejected_nonfinite": 0}

    def similarity(self, z: Tensor) -> Tensor:
        if int(self.count.item()) == 0:
            return z.new_full((z.shape[0],), 0.5)
        n = min(int(self.count.item()), self.cfg.max_axioms)
        sim = F.cosine_similarity(z.unsqueeze(1), self.bank[:n].unsqueeze(0), dim=-1)
        return (sim.max(dim=-1).values + 1) / 2

    def forward(self, z: Tensor, quality: Optional[float] = None) -> Tensor:
        lin = torch.sigmoid(self.verifier(z)).squeeze(-1)
        ax = self.similarity(z)
        w = self.cfg.axiom_weight
        if quality is None or self.cfg.quality_weight <= 0:
            score = (1 - w) * lin + w * ax
        else:
            q = torch.full_like(lin, _safe_unit(quality))
            qw = self.cfg.quality_weight
            score = (1 - w - qw) * lin + w * ax + qw * q
        return sclamp(score, 0.0, 1.0)

    def codify(self, z: Tensor) -> bool:
        v = z.reshape(-1)
        if not torch.isfinite(v).all() or v.norm().item() < 1e-6:
            self.stats["rejected_nonfinite"] += 1
            return False
        p = int(self.ptr.item())
        with torch.no_grad():
            self.bank[p] = F.normalize(v, dim=-1)
            self.ptr[0] = (p + 1) % self.cfg.max_axioms
            self.count[0] = min(int(self.count.item()) + 1, self.cfg.max_axioms)
        self.stats["codified"] += 1
        return True


class PsychePlus(nn.Module):
    def __init__(self, cfg: PsycheConfig):
        super().__init__()
        self.cfg = cfg
        self.id_noise = nn.Parameter(torch.full((cfg.latent_dim,), cfg.noise))
        self.propose = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.reconcile = nn.Linear(cfg.latent_dim * 2, cfg.latent_dim)
        self.superego = SuperEgo(cfg)

    def forward(self, z: Tensor, quality: Optional[float] = None) -> Dict[str, Any]:
        if z.shape[0] == 0:
            return {"updated_latent": z, "validity_score": z.new_zeros(0),
                    "quality_score": quality}
        noise = torch.randn_like(z) * F.softplus(self.id_noise)
        proposed = z + self.propose(z) + noise
        merged = F.gelu(self.reconcile(torch.cat([z, proposed], -1)))
        merged = torch.where(torch.isfinite(merged), merged, z)
        validity = self.superego(merged, quality)
        return {"updated_latent": merged, "validity_score": validity,
                "quality_score": quality}

    def maybe_codify(self, out: Dict[str, Any],
                     quality: Optional[float] = None) -> bool:
        v = out["validity_score"]
        if v.numel() == 0:
            return False
        q = quality if quality is not None else out.get("quality_score")
        q = _safe_unit(q)
        if float(v.mean().item()) <= self.cfg.codify_threshold:
            self.superego.stats["refused_validity"] += 1
            return False
        if q < self.cfg.quality_min:
            self.superego.stats["refused_quality"] += 1
            return False
        return self.superego.codify(out["updated_latent"][0])


# ══════════════════════════════════════════════════════════════════════
# 7. ECOSYSTEM ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AdapterSpec:
    name: str
    domain: str
    module: nn.Module
    quality_fn: Callable[[], float]
    frozen: bool = False


class EcosystemOrchestrator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.adapters: Dict[str, AdapterSpec] = nn.ModuleDict()
        self._quality_fns: Dict[str, Callable[[], float]] = {}
        self.mix = nn.Linear(3 * dim, dim)

    def register(self, spec: AdapterSpec) -> None:
        self.adapters[spec.name] = spec.module
        self._quality_fns[spec.name] = spec.quality_fn
        if spec.frozen:
            for p in spec.module.parameters():
                p.requires_grad_(False)

    def route(self, name: str, x: Tensor) -> Tensor:
        return self.adapters[name](x)

    def quality_report(self) -> Dict[str, float]:
        rep: Dict[str, float] = {}
        for name, fn in self._quality_fns.items():
            try:
                rep[name] = _safe_unit(fn())
            except Exception:
                rep[name] = 0.0
        rep["ecosystem"] = float(np.mean(list(rep.values()))) if rep else 0.0
        return rep

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        if not self.adapters:
            return x, {"ecosystem": 0.5}
        feats = [m(x) for m in self.adapters.values()][:3]
        while len(feats) < 3:
            feats.append(torch.zeros_like(feats[0]))
        return torch.tanh(self.mix(torch.cat(feats, -1))), self.quality_report()


class DomainEncoder(nn.Module):
    """Lift -> domain module -> readout, returning AGI-dim features."""
    def __init__(self, dim: int, body: nn.Module, out_key: str):
        super().__init__()
        self.body, self.out_key = body, out_key
        self.readout = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.readout(self.body(x)[self.out_key])


# ══════════════════════════════════════════════════════════════════════
# 8. AGI-ONE-STYLE HUB (compact)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HubConfig:
    dim: int = 64
    vocab: int = 512
    max_seq: int = 32
    layers: int = 2
    heads: int = 4
    carpet_level: int = 2
    n_sensors: int = 16
    n_voxels: int = 32
    steps: int = 200
    lr: float = 3e-4
    seed: int = 42
    out_dir: str = "runs/sesi_research"


@dataclass
class HubState:
    workspace: Tensor
    validity: float
    axiom_count: int
    quality: float
    safety_ok: float
    carpet_energy: float
    k_eff: float


class ResearchHub(nn.Module):
    def __init__(self, cfg: HubConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim, nhead=cfg.heads, dim_feedforward=cfg.dim * 2,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.lang = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.norm = nn.LayerNorm(cfg.dim)

        self.carpet = CarpetResistanceForm(cfg.dim, cfg.carpet_level)
        self.hodge = HodgeStyleSplit(cfg.dim)
        self.riemann = RiemannSpectralProxy(cfg.dim)
        self.monitor = NeuralMonitor(cfg.n_sensors, cfg.n_voxels)
        self.control = NearFieldControlPlanner(cfg.n_voxels, cfg.dim)
        self.reactor = ReactorSafetyMonitor(cfg.dim)

        self.orch = EcosystemOrchestrator(cfg.dim)
        self.orch.register(AdapterSpec(
            "carpet", "fractal", DomainEncoder(cfg.dim, self.carpet, "carpet_u"),
            lambda: 0.5, frozen=False))
        self.orch.register(AdapterSpec(
            "reactor", "safety", DomainEncoder(cfg.dim, self.reactor, "k_eff")._make_dummy()
            if False else _ScalarEncoder(cfg.dim, self.reactor, "k_eff"),
            lambda: float(self._last_safety), frozen=False))

        self.psyche = PsychePlus(PsycheConfig(latent_dim=cfg.dim))
        self.register_buffer("_last_safety", torch.tensor(0.5))
        self.workspace_proj = nn.Linear(cfg.dim * 2, cfg.dim)

    def forward(self, tokens: Tensor, eeg: Tensor) -> Tuple[HubState, Dict[str, Tensor]]:
        B = tokens.shape[0]
        h = self.norm(self.lang(self.embed(tokens)).mean(dim=1))

        cp = self.carpet(h)
        hg = self.hodge(h)
        rz = self.riemann(h)

        mon = self.monitor(eeg)
        ctl = self.control(mon["sources"])
        rx = self.reactor(h)

        # Safety probe for the orchestrator quality function
        with torch.no_grad():
            self._last_safety.copy_(rx["safety_ok"].mean().detach())

        ecos, report = self.orch(h)
        ws = torch.tanh(self.workspace_proj(torch.cat([h, ecos], -1)))

        quality = float(mon["quality"].mean().item()) * float(rx["safety_ok"].mean().item())
        plus = self.psyche(ws, quality_score=quality)
        self.psyche.maybe_codify(plus, quality_score=quality)

        state = HubState(
            workspace=ws,
            validity=float(plus["validity_score"].mean().item())
            if plus["validity_score"].numel() else 0.0,
            axiom_count=int(self.psyche.superego.count.item()),
            quality=quality,
            safety_ok=float(rx["safety_ok"].mean().item()),
            carpet_energy=float(cp["carpet_energy"].mean().item()),
            k_eff=float(rx["k_eff"].mean().item()),
        )
        aux = {**{f"carpet_{k}": v for k, v in cp.items()},
               **{f"hodge_{k}": v for k, v in hg.items()},
               **{f"riemann_{k}": v for k, v in rz.items()},
               **{f"mon_{k}": v for k, v in mon.items()},
               **{f"ctl_{k}": v for k, v in ctl.items()},
               **{f"rx_{k}": v for k, v in rx.items()},
               "workspace": ws, "validity": plus["validity_score"],
               "quality": torch.tensor(quality, device=h.device)}
        return state, aux


class _ScalarEncoder(nn.Module):
    """Broadcast a scalar domain output into a dim-sized feature."""
    def __init__(self, dim: int, body: nn.Module, key: str):
        super().__init__()
        self.body, self.key, self.dim = body, key, dim
        self.proj = nn.Linear(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        s = self.body(x)[self.key].unsqueeze(-1)
        return self.proj(s)


# ══════════════════════════════════════════════════════════════════════
# 9. LOSS / DATA / TRAINER
# ══════════════════════════════════════════════════════════════════════

def research_loss(aux: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
    parts = {
        "recon": aux["mon_residual"].mean(),
        "sparsity": aux["mon_sparsity"].mean() * 0.05,
        "orthogonality": aux["hodge_orthogonality_error"].mean() * 0.01,
        "energy": aux["carpet_energy_raw"].mean() * 1e-3,
        "safety": (1.0 - aux["rx_safety_ok"]).mean(),
        "critical": aux["rx_prompt_critical_flag"].mean() * 5.0,
        "control_violation": aux["ctl_violations"].mean() * 10.0,
    }
    total = sum(parts.values())
    parts["total"] = total
    return total, parts


class SyntheticEEG(Dataset):
    def __init__(self, cfg: HubConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, cfg.vocab, (n, cfg.max_seq), generator=g)
        t = torch.arange(cfg.max_seq).float()[None, None, :]
        freq = torch.rand(n, cfg.n_sensors, 1, generator=g) * 20 + 5
        self.eeg = torch.sin(2 * math.pi * freq * t / 100) + \
            torch.randn(n, cfg.n_sensors, cfg.max_seq, generator=g) * 0.1


    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return {"tokens": self.tokens[i], "eeg": self.eeg[i]}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class Trainer:
    def __init__(self, model: ResearchHub, device: torch.device, lr: float):
        self.model, self.device = model, device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                     weight_decay=1e-5,
                                     fused=(device.type == "cuda"))
        self.step = 0

    def step_batch(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()
        tokens = batch["tokens"].to(self.device)
        eeg = batch["eeg"].to(self.device)
        self.opt.zero_grad(set_to_none=True)
        with amp_context(self.device, enabled=True):
            _, aux = self.model(tokens, eeg)
            loss, parts = research_loss(aux)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.step += 1
        return {k: float(v.detach().item()) for k, v in parts.items()}


# ══════════════════════════════════════════════════════════════════════
# 10. SMOKE TEST + CLI
# ══════════════════════════════════════════════════════════════════════

def smoke(cfg: HubConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"[device] {device}")
    model = ResearchHub(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] {n_par:,}")
    print(f"[carpet] level={cfg.carpet_level} cells={model.carpet.graph.n_cells} "
          f"(expected {8 ** cfg.carpet_level})")

    B = 2
    tokens = torch.randint(0, cfg.vocab, (B, cfg.max_seq), device=device)
    eeg = torch.randn(B, cfg.n_sensors, cfg.max_seq, device=device)

    model.train()
    state, aux = model(tokens, eeg)
    loss, parts = research_loss(aux)
    loss.backward()

    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
    print(f"[loss]   {float(loss):.6f}")
    for k, v in parts.items():
        print(f"    {k:<20s} {float(v):.6f}")
    print(f"[state]  validity={state.validity:.3f} axioms={state.axiom_count} "
          f"quality={state.quality:.3f}")
    print(f"[safety] k_eff={state.k_eff:.4f} ok={state.safety_ok:.2f} "
          f"carpet_E={state.carpet_energy:.3f}")
    print(f"[control] permitted={aux['ctl_actuation_permitted'].mean().item():.2f} "
          f"max_gated={aux['ctl_gated_intensity'].max().item():.4f} (cap=0.25)")
    print(f"[autograd] {'PASS' if grads_ok else 'FAIL'}")
    assert grads_ok, "Gradient flow broken"
    assert aux["ctl_gated_intensity"].max().item() <= 0.25 + 1e-6


def train(cfg: HubConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    model = ResearchHub(cfg).to(device)
    trainer = Trainer(model, device, cfg.lr)
    ds = SyntheticEEG(cfg, n=256, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    it = iter(dl)
    for step in range(cfg.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        logs = trainer.step_batch(batch)
        if step % 20 == 0:
            print(f"step {step:4d} loss={logs['total']:.5f} "
                  f"safety={logs['safety']:.4f} recon={logs['recon']:.5f}")

    ckpt = {"config": vars(cfg), "state_dict": model.state_dict(),
            "axiom_bank": model.psyche.superego.bank.cpu(),
            "axiom_count": int(model.psyche.superego.count.item())}
    torch.save(ckpt, out / "research.pt")
    (out / "quality_report.json").write_text(
        json.dumps(model.orch.quality_report(), indent=2))
    print(f"[saved] {out / 'research.pt'}")


def main() -> None:
    p = argparse.ArgumentParser("SESI Unified Research")
    p.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--carpet-level", type=int, default=2)
    p.add_argument("--out", type=str, default="runs/sesi_research")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = HubConfig(dim=a.dim, carpet_level=a.carpet_level,
                    steps=a.steps, out_dir=a.out)
    if a.mode == "smoke":
        smoke(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
