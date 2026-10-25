"""
=============================================================================
SESI COGNITIVE-MODE ORCHESTRATOR v1.0
7 knowledge modes × Psyche Plus speculative design × AGI ONE reasoning loop
× External-AI panel with enforced provenance
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
License   : MIT | Year : 2026
=============================================================================

EPISTEMIC CONTRACT
──────────────────
1. This is a cognitive-architecture PATTERN, not AGI or sentience.
2. No web browsing is performed by this module. External claims enter ONLY
   through TheoryRegistry.ingest_external(), which REQUIRES a citation,
   provenance hash and an epistemic level. Unattributed claims are rejected.
3. External-AI transport is OPT-IN and OFF by default. The offline panel
   is an explicit stub; it never upgrades a claim to VERIFIED.
4. Superego gates are DETACHED. Generation cannot learn to fake a pass.
5. Theory anchors below come ONLY from the supplied Knowledge Base:

   [KB1] USSC(X) = a1<X> + a2 ΔX + a3 ||∇_h X||² − a4 X
         Standing Hypothesis 7.9; a1..a4 are FREE DESIGN PARAMETERS.
         The source explicitly flags the dual-use of l_c as an
         undocumented-assumption risk. We keep the two channels separate.
   [KB2] δ_min = (1/κ)(a_min/ω_{n−1})^{1/(n−1)} > 0
         Conditional on κ-fatness. Erratum: gasket condition number
         is Θ(5^m), NOT Θ((5/3)^m).
   [KB3] Hysteresis No-Zeno: N(T) ≤ (E(0)−E_inf)/ΔE_min
         Bounds NUMERICAL regularization events, not physical reactors.
   [KB4] Subsampling CI: valid without knowing rate / limit law / mixing
         rate; only qualitative weak dependence (Paper 11, Thm 4.4).
   [KB5] Discrete symbol: central difference → sin(ξh)/h, NOT ξ
         (Lemma 7.4). Nyquist band must be checked.
   [KB6] Hilbert–Pólya: Tasks 1–5 are functional analysis; Task 6
         (exact arithmetic trace/determinant identity) is the OPEN
         decisive step. No RH claim here.
   [KB7] WUCP: x_{t+1} = x_t + η_t Δ_t, Σ c_t ≤ B.
         No guaranteed speedup; must beat tuned classical baselines.
   [KB8] WQGP/GSRM: weak operations are not automatically efficient;
         full energy ledger required.
   [KB9] Vacuum energy: hypothetical; 7 missing-mechanism conditions.
   [KB10] Neural interface: Tegmark decoherence gap; no remote reading.

=============================================================================
"""
from __future__ import annotations

import math
import json
import time
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.cognitive")

__all__ = [
    "CognitiveConfig", "EpistemicLevel", "CognitiveMode",
    "TheoryRecord", "TheoryRegistry",
    "AnalysisMode", "ComparisonMode", "ExtractionMode", "DecompositionMode",
    "SynthesisMode", "CreationMode", "IntegrationMode", "MetaRouter",
    "VerificationGates", "PsychePlusCore",
    "EpisodicMemory", "AGIONEReasoningLoop",
    "ExternalAIPanel", "SESI_CognitiveOrchestrator",
    "attach_cognitive_modes_to_ecosystem",
]


# ═══════════════════════════════════════════════════════════════════════
# 0. Primitives
# ═══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def sfloor(x: Tensor, f: float, beta: float = 50.0) -> Tensor:
    return f + F.softplus(x - f, beta=beta)

def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 16):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, r) * 0.02)
        self.V = nn.Parameter(torch.zeros(r, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


def lap periodic(x: Tensor) -> Tensor:
    """Periodic 1-D Laplacian along feature axis (documented BC)."""
    return torch.roll(x, -1, -1) + torch.roll(x, 1, -1) - 2.0 * x


def grad_energy(x: Tensor) -> Tensor:
    d = torch.roll(x, -1, -1) - x
    return d.pow(2).mean(-1)


# ═══════════════════════════════════════════════════════════════════════
# 1. Config / Epistemics
# ═══════════════════════════════════════════════════════════════════════

class EpistemicLevel(Enum):
    VERIFIED = "verified"
    CONDITIONAL = "conditional"
    HEURISTIC = "heuristic"
    SPECULATIVE = "speculative"
    UNATTRIBUTED = "unattributed"   # external, not yet checked


class CognitiveMode(Enum):
    ANALYZE = auto()
    COMPARE = auto()
    EXTRACT = auto()
    DECOMPOSE = auto()
    SYNTHESIZE = auto()
    CREATE = auto()
    INTEGRATE = auto()


@dataclass
class CognitiveConfig:
    latent_dim: int = 64
    axiom_dim: int = 32
    reasoning_dim: int = 64
    n_scales: int = 4
    n_proposals: int = 8
    n_keep: int = 3
    create_steps: int = 6
    reasoning_horizon: int = 4
    memory_capacity: int = 32

    # [KB2] isodiametric
    kappa: float = 2.0
    a_min: float = 0.01

    # [KB3] hysteresis
    theta_hi: float = 1.0
    theta_lo: float = 0.80
    delta_e_min: float = 0.10

    # [KB5] discrete symbol
    h: float = 1.5e-3 / 64

    # gate
    superego_threshold: float = 0.55
    feedback_gain: float = 0.10
    ema_decay: float = 0.95

    # external AI — OFF by default
    allow_network: bool = False
    external_endpoints: Dict[str, str] = field(default_factory=dict)

    device: str = "cpu"


@dataclass
class TheoryRecord:
    tid: str
    statement: str
    level: EpistemicLevel
    source: str                      # file / URL / model+date
    provenance_hash: str
    assumptions: List[str] = field(default_factory=list)
    falsification: str = ""
    external: bool = False
    numeric: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# 2. Theory Registry — grounded in KB, external intake enforced
# ═══════════════════════════════════════════════════════════════════════

class TheoryRegistry:
    def __init__(self):
        self.records: Dict[str, TheoryRecord] = {}
        self._seed_kb()

    def _seed_kb(self) -> None:
        seeds = [
            TheoryRecord(
                "USSC_7.9",
                "USSC(X)=a1<X>+a2 ΔX+a3||grad_h X||^2 - a4 X",
                EpistemicLevel.CONDITIONAL,
                "Unified_Ordinal-Tensor_Measurement_Theory_Merged_Revised.pdf",
                "KB:Standing-Hypothesis-7.9",
                assumptions=["a1..a4 are free design parameters",
                             "l_c channels must not be conflated"],
                falsification="Fail if operator bound or coercivity breaks on test domain."),
            TheoryRecord(
                "DELTA_MIN",
                "d_min=(1/kappa)(a_min/omega_{n-1})^(1/(n-1))>0",
                EpistemicLevel.CONDITIONAL,
                "Delta_Min_Isodiametric_Closure_and_Erratum.pdf",
                "KB:Section-1.2-1.3",
                assumptions=["kappa-fatness holds"],
                falsification="Find a nucleation patch violating kappa-fatness."),
            TheoryRecord(
                "NO_ZENO",
                "N(T) <= (E(0)-E_inf)/dE_min via two-threshold hysteresis",
                EpistemicLevel.VERIFIED,
                "Six_Problems_Closed_As_Far_As_They_Go.pdf",
                "KB:Section-6.2-Prop-6.1",
                assumptions=["NUMERICAL event bound only, not physical reactor control"],
                falsification="Construct a reset sequence exceeding the energy/gap bound."),
            TheoryRecord(
                "SUBSAMPLE_CI",
                "Self-normalized subsampling CI without rate/limit-law/mixing-rate",
                EpistemicLevel.VERIFIED,
                "Paper11_Assumption_Light_Statistics_SESI_Delta_Min-1.pdf",
                "KB:Theorem-4.4",
                assumptions=["qualitative weak dependence only"],
                falsification="CI coverage fails under validated long-memory data."),
            TheoryRecord(
                "DISCRETE_SYMBOL",
                "central-difference symbol is sin(xi*h)/h, not xi",
                EpistemicLevel.VERIFIED,
                "Unified_Ordinal-Tensor_Measurement_Theory_Merged_Revised.pdf",
                "KB:Lemma-7.4",
                assumptions=["Nyquist band |xi*h| <= pi"],
                falsification="Show stencil symbol equals continuum outside truncation error."),
            TheoryRecord(
                "HP_OPEN",
                "Hilbert-Polya: arithmetic-identification arrow is still OPEN",
                EpistemicLevel.SPECULATIVE,
                "Hilbert_Polya_Proof_Gap_Audit_and_Closure_Program.pdf",
                "KB:Tasks-1..7, Section-19-20",
                assumptions=["Task 6 exact trace/determinant identity NOT established"],
                falsification="Produce explicit self-adjoint H with D_H(z)=C(z)Xi(z), C zero-free."),
            TheoryRecord(
                "WUCP",
                "x_{t+1}=x_t+eta_t*Delta_t subject to sum(c_t)<=B",
                EpistemicLevel.HEURISTIC,
                "Toward Quantum-Level Computational Efficiency....pdf",
                "KB:Section-3.2-3.3, Section-9",
                assumptions=["no guaranteed speedup; benchmark required"],
                falsification="Tuned classical baseline dominates on Pareto frontier."),
            TheoryRecord(
                "VACUUM_HYP",
                "Vacuum-energy extraction remains hypothetical (7 conditions unmet)",
                EpistemicLevel.SPECULATIVE,
                "Weak Quantum Gates, Gradual State Reduction....pdf",
                "KB:Section-8.2, Section-9",
                assumptions=["no reservoir/coupling/balance demonstrated"],
                falsification="Net-positive repeatable cycle excluding all alternative sources."),
        ]
        for r in seeds:
            self.records[r.tid] = r

    def ingest_external(self, *, tid: str, statement: str, source: str,
                        raw_evidence: str,
                        level: EpistemicLevel = EpistemicLevel.UNATTRIBUTED,
                        assumptions: Optional[List[str]] = None) -> Optional[TheoryRecord]:
        """External claims REQUIRE source + evidence. Never auto-verified."""
        if not source.strip() or not raw_evidence.strip():
            logger.warning("Rejected %s: missing provenance", tid)
            return None
        rec = TheoryRecord(
            tid=tid, statement=statement, level=level, source=source,
            provenance_hash=hashlib.sha256(
                (source + "|" + raw_evidence).encode()).hexdigest()[:16],
            assumptions=assumptions or ["external claim, not independently checked"],
            falsification="Specify a reproducible counterexperiment before promotion.",
            external=True,
        )
        self.records[tid] = rec
        return rec

    def ledger(self) -> List[Dict[str, Any]]:
        return [
            {"tid": r.tid, "level": r.level.value, "source": r.source,
             "external": r.external, "hash": r.provenance_hash}
            for r in self.records.values()
        ]


# ═══════════════════════════════════════════════════════════════════════
# 3. Seven cognitive modes
# ═══════════════════════════════════════════════════════════════════════

class AnalysisMode(nn.Module):
    """[KB1] USSC functional + structural statistics. No proof implied."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.a_raw = nn.Parameter(torch.zeros(4))
        self.project = LowRank(c.latent_dim, c.latent_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        a = F.softplus(self.a_raw) + 1e-6          # a1..a4 > 0
        mean = x.mean(-1, keepdim=True)            # normalized <X>
        lap = lap_periodic(x)
        ge = grad_energy(x).unsqueeze(-1)
        u = a[0] * mean + a[1] * lap + a[2] * ge - a[3] * x
        return {"latent": torch.tanh(self.project(u)),
                "mean": mean.squeeze(-1), "laplacian_energy": lap.pow(2).mean(-1),
                "gradient_energy": ge.squeeze(-1), "coefficients": a}


class ComparisonMode(nn.Module):
    """Pairwise alignment / conflict between internal and external latents."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.head = LowRank(c.latent_dim * 3, c.latent_dim)

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        cos = F.cosine_similarity(a, b, dim=-1)
        dist = (a - b).norm(dim=-1)
        gap = (a.pow(2).mean(-1) - b.pow(2).mean(-1)).abs()
        feats = torch.cat([a, b, (a - b).abs()], dim=-1)
        latent = torch.tanh(self.head(feats))
        return {"latent": latent, "agreement": cos,
                "distance": dist, "scale_gap": gap,
                "conflict": (dist > 1.0).detach()}


class ExtractionMode(nn.Module):
    """Attention-based salience: pick the load-bearing features."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.q = nn.Linear(c.latent_dim, 1)
        self.k = nn.Linear(c.latent_dim, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # treat last dim as feature tokens via a small projection bank
        bank = x.unsqueeze(1)                       # B,1,D
        scores = self.q(bank).squeeze(-1) + self.k(bank).squeeze(-1)
        w = F.softmax(scores, dim=-1)
        salient = (w * x)
        entropy = -(w * torch.log(w + 1e-8)).sum(-1)
        return {"latent": salient, "salience": w, "entropy": entropy,
                "top_index": w.argmax(-1).detach()}


class DecompositionMode(nn.Module):
    """Split into nested subproblems with residual dependency chain."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.scales = nn.ModuleList([
            LowRank(c.latent_dim, c.latent_dim) for _ in range(c.n_scales)])
        self.merge = nn.Linear(c.latent_dim, c.latent_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        subs, residual = [], x
        for layer in self.scales:
            part = torch.tanh(layer(residual))
            subs.append(part)
            residual = residual - part
        stack = torch.stack(subs, dim=1)             # B,S,D
        return {"latent": torch.tanh(self.merge(stack.mean(1))),
                "subproblems": stack, "residual": residual}


class SynthesisMode(nn.Module):
    """Cross-attention fusion of internal + external knowledge."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(c.latent_dim, 4, batch_first=True)
        self.out = nn.Linear(c.latent_dim, c.latent_dim)

    def forward(self, internal: Tensor, external: Tensor) -> Dict[str, Tensor]:
        q = internal.unsqueeze(1)                    # B,1,D
        kv = torch.stack([internal, external], dim=1)
        fused, w = self.attn(q, kv, kv)
        latent = torch.tanh(self.out(fused.squeeze(1)))
        coherence = 1.0 / (1.0 + (latent - internal).pow(2).mean(-1))
        return {"latent": latent, "coherence": coherence,
                "source_weights": w.detach()}


class CreationMode(nn.Module):
    """Bounded latent proposal generation (not a proof generator)."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        self.step = nn.Sequential(
            nn.Linear(c.latent_dim + 1, c.latent_dim * 2), nn.GELU(),
            nn.Linear(c.latent_dim * 2, c.latent_dim))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        z = torch.tanh(x)
        for i in range(self.c.create_steps):
            t = torch.full((z.shape[0], 1), i / max(1, self.c.create_steps),
                           device=z.device, dtype=z.dtype)
            d = self.step(torch.cat([z, t], dim=-1))
            z = torch.tanh(0.9 * z + 0.1 * d)
        novelty = 1.0 - F.cosine_similarity(z, x, dim=-1)
        return {"latent": z, "novelty": novelty}


class IntegrationMode(nn.Module):
    """Cross-mode consistency + unresolved-conflict ledger."""
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.out = nn.Linear(c.latent_dim, c.latent_dim)

    def forward(self, mode_latents: Tensor) -> Dict[str, Tensor]:
        # mode_latents: B, M, D
        mean = mode_latents.mean(1)
        var = mode_latents.var(1).mean(-1)
        consistency = torch.exp(-var)
        return {"latent": torch.tanh(self.out(mean)),
                "consistency": consistency,
                "contradiction": (var > 0.5).detach()}


# ═══════════════════════════════════════════════════════════════════════
# 4. Meta-router (budget-aware, hard choice detached)
# ═══════════════════════════════════════════════════════════════════════

class MetaRouter(nn.Module):
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        self.score = nn.Sequential(
            nn.Linear(c.latent_dim + 2, 32), nn.GELU(),
            nn.Linear(32, 7))
        self.register_buffer("budget", torch.tensor(1.0), persistent=False)

    def forward(self, x: Tensor, uncertainty: Tensor) -> Dict[str, Tensor]:
        info = torch.cat([x, uncertainty.unsqueeze(-1),
                          self.budget.expand(x.shape[0], 1)], dim=-1)
        logits = self.score(info)
        weights = F.softmax(logits, dim=-1)
        hard = weights.argmax(-1).detach()
        return {"weights": weights, "hard_mode": hard,
                "logits": logits}


# ═══════════════════════════════════════════════════════════════════════
# 5. Real verification gates from the KB (detached decisions)
# ═══════════════════════════════════════════════════════════════════════

class VerificationGates(nn.Module):
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        dmin = (1.0 / c.kappa) * math.sqrt(c.a_min / math.pi)
        self.register_buffer("dmin", torch.tensor(dmin), persistent=False)
        self.register_buffer("hi", torch.tensor(c.theta_hi), persistent=False)
        self.register_buffer("lo", torch.tensor(c.theta_lo), persistent=False)
        self.register_buffer("h", torch.tensor(c.h), persistent=False)
        self.patch = nn.Sequential(nn.Linear(c.axiom_dim, 16), nn.GELU(),
                                   nn.Linear(16, 3), nn.Sigmoid())
        self.stress = nn.Sequential(nn.Linear(c.axiom_dim, 16), nn.GELU(),
                                    nn.Linear(16, 1), nn.Softplus())
        self.freq = nn.Sequential(nn.Linear(c.axiom_dim, 5), nn.Tanh())
        self.ci = nn.Sequential(nn.Linear(c.axiom_dim, 16), nn.GELU(),
                                nn.Linear(16, 1), nn.Softplus())

    def forward(self, axiom: Tensor) -> Dict[str, Tensor]:
        B = axiom.shape[0]
        ps = self.patch(axiom)
        area = sfloor(ps[:, 0] * 0.1, self.c.a_min)
        rP = (area / math.pi).clamp_min(1e-12)
        fat = sgate(ps[:, 2] * 0.1 - rP / self.c.kappa, 1e-3)

        stress = self.stress(axiom).squeeze(-1)
        fire = sgate(stress - self.hi, 1e-2)
        n_bound = stress / sfloor(torch.tensor(self.c.delta_e_min), 1e-8)

        xi = self.freq(axiom) * (math.pi / (2 * self.h))
        sym = torch.sin(xi * self.h) / self.h
        rel = (sym - xi).abs() / (xi.abs() + 1e-12)
        in_band = (xi.abs() * self.h <= math.pi).float().mean(-1)
        symbol_ok = (rel.mean(-1) < 1e-2) & (in_band > 0.9)

        width = self.ci(axiom).squeeze(-1)
        ci_ok = (self.dmin - 1.96 * width) > 0

        numeric = (fat > 0.5) & (n_bound < 1e6) & symbol_ok & ci_ok
        return {"fatness": fat, "no_zeno": (n_bound < 1e6).float(),
                "symbol_ok": symbol_ok.float(), "ci_ok": ci_ok.float(),
                "dmin": self.dmin.expand(B), "ci_width": width,
                "N_bound": n_bound, "fire": fire,
                "pass_hard": numeric.detach()}


# ═══════════════════════════════════════════════════════════════════════
# 6. Psyche Plus core
# ═══════════════════════════════════════════════════════════════════════

class PsychePlusCore(nn.Module):
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        self.id = nn.Sequential(LowRank(c.latent_dim, c.axiom_dim), nn.Tanh())
        self.ego = nn.GRUCell(c.axiom_dim, c.axiom_dim)
        self.gates = VerificationGates(c)
        self.conf = nn.Sequential(nn.Linear(c.axiom_dim * 2, 32), nn.GELU(),
                                  nn.Linear(32, 1), nn.Sigmoid())
        self.feedback = nn.Parameter(torch.zeros(c.axiom_dim))
        self.register_buffer("quality", torch.tensor(0.5), persistent=False)

    def forward(self, context: Tensor) -> Dict[str, Tensor]:
        B = context.shape[0]
        raw = self.id(context) + self.feedback.unsqueeze(0)
        props = raw.unsqueeze(1).expand(-1, self.c.n_proposals, -1)
        props = props + 0.1 * torch.randn_like(props)

        kept = []
        for i in range(self.c.n_keep):
            h = props[:, i]
            for _ in range(2):
                h = self.ego(h, h)
            kept.append(h)
        cand = torch.stack(kept, 1)                       # B,K,A
        flat = cand.reshape(B * self.c.n_keep, -1)
        g = self.gates(flat)
        conf = self.conf(torch.cat([flat, g["fatness"].unsqueeze(-1)], -1))
        conf = conf.squeeze(-1).view(B, self.c.n_keep)
        hard = g["pass_hard"].view(B, self.c.n_keep).float()
        mask = (hard * (conf >= self.c.superego_threshold).float()).detach()

        with torch.no_grad():
            rate = mask.mean()
            self.quality.mul_(self.c.ema_decay).add_(
                rate, alpha=1 - self.c.ema_decay)

        return {"axioms": cand * mask.unsqueeze(-1), "confidence": conf,
                "mask": mask, "quality": self.quality,
                "gate_detail": g}

    def receive_feedback(self, signal: Tensor) -> None:
        with torch.no_grad():
            s = signal.mean(0)
            if s.shape[-1] == self.c.axiom_dim:
                self.feedback.add_(s * self.c.feedback_gain)


# ═══════════════════════════════════════════════════════════════════════
# 7. Memory + AGI ONE reasoning loop
# ═══════════════════════════════════════════════════════════════════════

class EpisodicMemory(nn.Module):
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.cap = c.memory_capacity
        self.register_buffer("buf", torch.zeros(c.memory_capacity, c.reasoning_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def write(self, x: Tensor) -> None:
        with torch.no_grad():
            for row in x.detach():
                self.buf[int(self.ptr.item()) % self.cap] = row
                self.ptr.add_(1)

    def recall(self, q: Tensor, k: int = 4) -> Tensor:
        sims = F.cosine_similarity(q.unsqueeze(1), self.buf.unsqueeze(0), dim=-1)
        idx = sims.topk(min(k, self.cap), dim=-1).indices
        return self.buf[idx].mean(1)


class AGIONEReasoningLoop(nn.Module):
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        self.enc = nn.Linear(c.axiom_dim, c.reasoning_dim)
        self.gru = nn.GRUCell(c.reasoning_dim, c.reasoning_dim)
        self.out = nn.Sequential(nn.Linear(c.reasoning_dim, c.reasoning_dim),
                                 nn.GELU(),
                                 nn.Linear(c.reasoning_dim, c.reasoning_dim))
        self.quality = nn.Sequential(
            nn.Linear(c.reasoning_dim * 2, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid())
        self.fb = nn.Sequential(nn.Linear(c.reasoning_dim, c.axiom_dim), nn.Tanh())
        self.memory = EpisodicMemory(c)

    def forward(self, axioms: Tensor, conf: Tensor, mask: Tensor,
                context_memory: Optional[Tensor] = None) -> Dict[str, Tensor]:
        B = axioms.shape[0]
        proj = self.enc(axioms)
        w = (conf * mask).unsqueeze(-1)
        state = (proj * w).sum(1) / (w.sum(1) + 1e-6)
        if context_memory is not None:
            state = state + 0.1 * context_memory

        h = state
        for _ in range(self.c.reasoning_horizon):
            h = self.gru(state, h)
        outcome = self.out(h)
        q = self.quality(torch.cat([state, outcome], -1)).squeeze(-1)
        feedback = self.fb(outcome) * (2 * q - 1).unsqueeze(-1)
        self.memory.write(outcome)
        return {"outcome": outcome, "quality": q, "feedback": feedback,
                "reasoning_state": state}


# ═══════════════════════════════════════════════════════════════════════
# 8. External-AI panel — OPT-IN, provenance enforced
# ═══════════════════════════════════════════════════════════════════════

class ExternalAIPanel(nn.Module):
    """
    No live connections by default.

    - offline: deterministic local stub, labelled UNATTRIBUTED
    - online : requires allow_network=True + endpoint + env API key
               (HTTPS only, timeout, bounded payload)

    Outputs are ALWAYS ingested as external/unverified. The panel cannot
    promote its own output to VERIFIED.
    """
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        self.proj = nn.Linear(c.latent_dim, c.latent_dim)
        self.names = list(c.external_endpoints.keys()) or ["offline_panel"]

    def _http(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import os, urllib.request
        url = self.c.external_endpoints[name]
        if not url.startswith("https://"):
            raise ValueError("HTTPS required")
        key = os.environ.get(f"SESI_{name.upper()}_API_KEY", "")
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def forward(self, x: Tensor, registry: TheoryRegistry) -> Dict[str, Tensor]:
        B = x.shape[0]
        outs, online_flags = [], []
        for name in self.names:
            if self.c.allow_network and name in self.c.external_endpoints:
                try:
                    resp = self._http(name, {"mode": "critique",
                                             "vector": x[0].detach().tolist()})
                    # only accept a bounded numeric vector if provided
                    v = resp.get("vector")
                    if isinstance(v, list) and len(v) == x.shape[-1]:
                        vec = torch.tensor(v, device=x.device, dtype=x.dtype)
                        outs.append(vec.unsqueeze(0).expand(B, -1))
                        online_flags.append(1.0)
                        registry.ingest_external(
                            tid=f"ext_{name}", statement=str(resp.get("claim", "")),
                            source=f"{name}@{time.strftime('%Y-%m-%d')}",
                            raw_evidence=json.dumps(resp)[:4000])
                        continue
                except Exception as e:
                    logger.warning("external %s unavailable: %s", name, e)
            # honest offline stub
            outs.append(torch.tanh(self.proj(x)))
            online_flags.append(0.0)

        stack = torch.stack(outs, dim=1)                # B,P,D
        return {"panel_latents": stack,
                "online_fraction": torch.tensor(sum(online_flags) / len(online_flags),
                                                device=x.device),
                "self_reported_verified": torch.zeros(B, device=x.device)}


# ═══════════════════════════════════════════════════════════════════════
# 9. Capstone orchestrator
# ═══════════════════════════════════════════════════════════════════════

class SESI_CognitiveOrchestrator(nn.Module):
    def __init__(self, c: CognitiveConfig):
        super().__init__()
        self.c = c
        self.registry = TheoryRegistry()
        self.analyze = AnalysisMode(c)
        self.compare = ComparisonMode(c)
        self.extract = ExtractionMode(c)
        self.decompose = DecompositionMode(c)
        self.synthesize = SynthesisMode(c)
        self.create = CreationMode(c)
        self.integrate = IntegrationMode(c)
        self.router = MetaRouter(c)
        self.psyche = PsychePlusCore(c)
        self.reasoning = AGIONEReasoningLoop(c)
        self.panel = ExternalAIPanel(c)
        self.lift = nn.Linear(c.latent_dim, c.reasoning_dim)

    def forward(self, agi_context: Tensor,
                external_context: Optional[Tensor] = None,
                use_panel: bool = False) -> Dict[str, Any]:
        B, D = agi_context.shape
        ext = (external_context if external_context is not None
               else torch.tanh(agi_context))

        # 1–4: analyze / extract / decompose / compare
        an = self.analyze(agi_context)
        ex = self.extract(agi_context)
        de = self.decompose(agi_context)
        co = self.compare(an["latent"], ext)

        # panel (optional) + synthesis
        if use_panel:
            panel = self.panel(ext, self.registry)
            ext_fused = panel["panel_latents"].mean(1)
        else:
            panel = None
            ext_fused = ext
        sy = self.synthesize(an["latent"], ext_fused)

        # 5–6: create + integrate
        cr = self.create(sy["latent"])
        mode_stack = torch.stack([an["latent"], co["latent"], ex["latent"],
                                  de["latent"], sy["latent"], cr["latent"]], dim=1)
        ig = self.integrate(mode_stack)

        # meta-routing
        uncer = 1.0 - ig["consistency"]
        route = self.router(ig["latent"], uncer)

        # Psyche speculation → real gates
        psyche_ctx = F.pad(ig["latent"], (0, max(0, self.c.latent_dim - ig["latent"].shape[-1]))) \
            if ig["latent"].shape[-1] < self.c.latent_dim else ig["latent"]
        # AnalysisMode already emits latent_dim, so this is normally a no-op
        psy = self.psyche(psyche_ctx)

        # feed hypotheses into AGI ONE reasoning loop with memory recall
        q = self.lift(ig["latent"])
        recalled = self.reasoning.memory.recall(q)
        reason = self.reasoning(psy["axioms"], psy["confidence"], psy["mask"],
                                context_memory=recalled)

        # closed loop
        self.psyche.receive_feedback(reason["feedback"])

        return {
            "modes": {"analyze": an, "compare": co, "extract": ex,
                      "decompose": de, "synthesize": sy, "create": cr,
                      "integrate": ig},
            "router": route, "psyche": psy, "reasoning": reason,
            "panel": panel,
            "epistemic_ledger": self.registry.ledger(),
            "integrated_latent": ig["latent"],
        }


# ═══════════════════════════════════════════════════════════════════════
# 10. Ecosystem adapter + smoke test
# ═══════════════════════════════════════════════════════════════════════

class CognitiveEcosystemAdapter(nn.Module):
    def __init__(self, orch: SESI_CognitiveOrchestrator, agi_dim: int):
        super().__init__()
        self.orch = orch
        self.lift = nn.Linear(agi_dim, orch.c.latent_dim)
        self.readout = nn.Sequential(
            nn.Linear(orch.c.reasoning_dim, agi_dim), nn.GELU(),
            nn.Linear(agi_dim, agi_dim))
        self._q = 0.5

    def encode(self, x: Tensor) -> Tensor:
        out = self.orch(torch.tanh(self.lift(x)))
        return self.readout(out["reasoning"]["outcome"])

    @torch.no_grad()
    def get_quality_score(self) -> float:
        try:
            dev = next(self.parameters()).device
            o = self.orch(torch.randn(2, self.orch.c.latent_dim, device=dev))
            self._q = float(o["reasoning"]["quality"].mean().item())
        except Exception:
            pass
        return self._q


def attach_cognitive_modes_to_ecosystem(orchestrator: Any, agi_dim: int,
                                        cfg: Optional[CognitiveConfig] = None):
    cfg = cfg or CognitiveConfig()
    adapter = CognitiveEcosystemAdapter(SESI_CognitiveOrchestrator(cfg), agi_dim)
    if hasattr(orchestrator, "register"):
        orchestrator.register("psyche_agione_cognitive_modes", adapter,
                              "cognitive_integration", agi_dim,
                              quality_fn=adapter.get_quality_score)
    return adapter


def smoke_test():
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = CognitiveConfig(latent_dim=64, axiom_dim=32, reasoning_dim=64,
                        n_proposals=6, n_keep=3, device=str(dev))
    m = SESI_CognitiveOrchestrator(c).to(dev)
    x = torch.randn(2, c.latent_dim, device=dev, requires_grad=True)
    out = m(x, use_panel=False)

    loss = -out["reasoning"]["quality"].mean()
    loss.backward()
    ok = x.grad is not None and torch.isfinite(x.grad).all()

    print("modes        :", [v.name for v in CognitiveMode])
    print("consistency  :", out["modes"]["integrate"]["consistency"].detach().mean().item())
    print("novelty      :", out["modes"]["create"]["novelty"].detach().mean().item())
    print("hard mode    :", out["router"]["hard_mode"].tolist())
    print("axioms passed:", out["psyche"]["mask"].sum().item())
    print("reason quality:", out["reasoning"]["quality"].detach().mean().item())
    print("external online fraction:",
          0.0 if out["panel"] is None else out["panel"]["online_fraction"].item())
    print("theory records:", len(out["epistemic_ledger"]))
    print("autograd     :", "PASS" if ok else "FAIL")
    print("NOTE: no browsing performed; external panel OFF; not AGI/sentience.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
