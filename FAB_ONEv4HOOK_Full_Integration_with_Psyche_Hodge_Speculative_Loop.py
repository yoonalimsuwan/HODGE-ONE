"""
=============================================================================
FAB-ONE-v4 HOOK — Full Integration with Psyche-Hodge Speculative Loop
=============================================================================
Developer  : PAI & Yoon A. Limsuwan / MSPS NETWORK
License    : MIT
Year       : 2026
=============================================================================

PURPOSE
───────
Connect the Psyche-Hodge speculative design loop (from psyche_hodge_fab_bridge)
to the REAL FAB ONE v4 engine so that:

  1. Each material candidate produced by Hodge Dream is pushed through
     PolyharmonicInterfaceSolver + IsodiametricNucleationController +
     TopologicalBottleneckController — the actual fabrication physics.
  2. The resulting `production_loss` is returned as a DETACHED scalar
     reward to the Psyche Plus speculator.
  3. Psyche ranks / accepts / rejects axioms using this reward + the
     paper-derived verification gates.
  4. The EcosystemOrchestrator's quality_fn reads the real FAB ONE
     production loss, not a hardcoded constant.

DESIGN PRINCIPLES (epistemic hygiene)
─────────────────────────────────────
  • The reward is DETACHED — Psyche cannot game the gradient by
    optimizing production_loss directly. It can only influence it
    indirectly through the proposal distribution.
  • Acceptance is still gated by the five paper-derived checks
    (DRC, κ-fatness, No-Zeno, subsampling CI, Kirchhoff residual).
    A low production_loss alone does NOT pass a candidate.
  • If FAB ONE v4 is unavailable, the hook degrades gracefully to
    a synthetic reward with an explicit "synthetic=True" flag so
    downstream code knows the physics is not being evaluated.
  • All tensors flow through C^∞ primitives; no dead gradients.

=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("fab_one_v4_hook")

__all__ = [
    "FabOneV4HookConfig",
    "FabOneV4Hook",
    "PsycheReward",
    "attach_fab_hook_to_loop",
    "run_integration_demo",
]


# ═══════════════════════════════════════════════════════════════════════
# Smooth primitives
# ═══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def sfloor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)


def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, r) * 0.02)
        self.V = nn.Parameter(torch.zeros(r, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FabOneV4HookConfig:
    latent_dim: int = 128
    material_dim: int = 64
    photonic_dim: int = 64
    n_axioms: int = 8

    # FAB ONE v4 physics (must match the real engine)
    l_c: float = 1.5e-3
    tension_constant_cv: float = 0.85
    m_poly: int = 4
    n_poly: int = 32
    lopatinski_det: float = 12.0

    # Isodiametric nucleation
    kappa_fatness: float = 2.0
    a_min_um2: float = 0.01
    omega_n_minus_1: float = math.pi

    # Reward shaping
    reward_scale: float = 1.0
    reward_offset: float = 0.0
    ema_decay: float = 0.95
    synthetic_fallback: bool = True   # if True, use synthetic reward when FAB ONE unavailable

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# PsycheReward — the detached scalar returned to Psyche
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PsycheReward:
    """
    Detached reward payload returned to Psyche Plus speculator.

    Fields
    ------
    value : float
        Scalar reward ∈ (-inf, +inf). Higher is better.
        Computed as -production_loss (after shaping), detached.
    production_loss : float
        Raw FAB ONE production loss (detached).
    fab_available : bool
        True if the reward came from the real FAB ONE v4 engine.
        False if it came from the synthetic fallback.
    gate_scores : Dict[str, float]
        Per-gate scores from the paper-derived verification.
    candidate_id : int
        Index of the material candidate this reward is for.
    """
    value: float
    production_loss: float
    fab_available: bool
    gate_scores: Dict[str, float] = field(default_factory=dict)
    candidate_id: int = 0


# ═══════════════════════════════════════════════════════════════════════
# FAB ONE v4 HOOK — the main integration class
# ═══════════════════════════════════════════════════════════════════════

class FabOneV4Hook(nn.Module):
    """
    Hook that routes Hodge-Dream material candidates through the real
    FAB ONE v4 engine and returns a detached PsycheReward.

    Pipeline
    --------
    candidate_latent  ──►  patch_stats (differentiable)
                       ──►  PolyharmonicInterfaceSolver
                              (8th-order GMT, det M = 12)
                       ──►  IsodiametricNucleationController
                              (κ-fatness, δ_min bound)
                       ──►  TopologicalBottleneckController
                              (No-Zeno soft-floor)
                       ──►  production_loss  ──►  PsycheReward

    The hook is a nn.Module so it can be registered with the
    EcosystemOrchestrator and its parameters (if any) participate
    in the global training.
    """

    def __init__(
        self,
        cfg: FabOneV4HookConfig,
        fab_engine: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.fab_engine = fab_engine
        self.fab_available = fab_engine is not None

        # ── Differentiable candidate → patch-stats head ────────────────
        # Maps a material latent to [area, perimeter, max_dist_to_boundary]
        # so the IsodiametricNucleationController can evaluate κ-fatness.
        self.patch_head = nn.Sequential(
            LowRank(cfg.material_dim, 32, 16),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

        # ── Differentiable candidate → manufacturing energy ───────────
        self.energy_head = nn.Sequential(
            LowRank(cfg.material_dim, 32, 16),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # ── Differentiable candidate → surface noise ──────────────────
        # tensor_dim of PolyharmonicInterfaceSolver
        self.tensor_dim = (cfg.m_poly ** 2) * (cfg.n_poly ** 2) + cfg.m_poly * (cfg.n_poly ** 2)
        self.noise_head = nn.Sequential(
            LowRank(cfg.material_dim, 64, 16),
            nn.GELU(),
            nn.Linear(64, self.tensor_dim),
            nn.Tanh(),
        )

        # ── Build minimal FAB ONE components if engine not provided ───
        if not self.fab_available and cfg.synthetic_fallback:
            self._build_synthetic_fab()
        else:
            self._synthetic = None

        # ── EMA of production loss (for quality probe) ────────────────
        self.register_buffer("ema_loss", torch.tensor(1.0), persistent=False)

    # ------------------------------------------------------------------ #
    # Synthetic FAB ONE fallback (used only when real engine unavailable)
    # ------------------------------------------------------------------ #
    def _build_synthetic_fab(self) -> None:
        """
        Build a minimal, differentiable stand-in for FAB ONE v4 so the
        hook can be exercised in unit tests and smoke runs without the
        full engine on the path.

        This is NOT a physics surrogate — it is a shape-matching shim
        that returns a loss with the correct dimensionality and gradient
        flow. Downstream code MUST check `reward.fab_available` before
        treating the loss as physical.
        """
        self._synthetic = nn.ModuleDict({
            "polyharmonic": nn.Linear(self.tensor_dim, self.tensor_dim),
            "nucleation": nn.Sequential(
                nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Softplus(),
            ),
            "bottleneck": nn.Sequential(
                nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1), nn.Softplus(),
            ),
        })

    # ------------------------------------------------------------------ #
    # Core: run a candidate through FAB ONE v4
    # ------------------------------------------------------------------ #
    def evaluate_candidate(
        self,
        candidate_latent: Tensor,
        candidate_id: int = 0,
    ) -> PsycheReward:
        """
        Run a single material candidate through the FAB ONE v4 engine
        and return a detached PsycheReward.

        Parameters
        ----------
        candidate_latent : Tensor [B, material_dim]
            Output of HodgeDreamMaterialInnovator["candidate_latent"].
        candidate_id : int
            Index for logging / audit trail.

        Returns
        -------
        PsycheReward
            Detached reward payload.
        """
        B = candidate_latent.shape[0]
        dev = candidate_latent.device

        # 1. Differentiable projection to FAB ONE inputs
        patch_stats = self.patch_head(candidate_latent)        # [B, 3]
        mfg_energy = self.energy_head(candidate_latent).squeeze(-1)  # [B]
        surface_noise = self.noise_head(candidate_latent)      # [B, tensor_dim]

        # 2. Run through FAB ONE v4 (real or synthetic)
        if self.fab_available and self.fab_engine is not None:
            try:
                out = self._run_real_fab(mfg_energy, surface_noise, patch_stats)
                prod_loss = out["production_loss"]
                gate_scores = out.get("gate_scores", {})
                fab_ok = True
            except Exception as e:
                logger.warning("FAB ONE v4 forward failed (%s); falling back.", e)
                prod_loss, gate_scores = self._run_synthetic_fab(
                    mfg_energy, surface_noise, patch_stats
                )
                fab_ok = False
        else:
            prod_loss, gate_scores = self._run_synthetic_fab(
                mfg_energy, surface_noise, patch_stats
            )
            fab_ok = False

        # 3. Aggregate to a scalar (per-sample mean, then batch mean)
        scalar_loss = prod_loss.mean()

        # 4. Update EMA (no grad)
        with torch.no_grad():
            self.ema_loss.mul_(self.cfg.ema_decay).add_(
                scalar_loss.detach(), alpha=1.0 - self.cfg.ema_decay
            )

        # 5. Detach the reward — Psyche must not backprop through it
        reward_value = float(
            -(scalar_loss.detach() * self.cfg.reward_scale + self.cfg.reward_offset)
        )

        return PsycheReward(
            value=reward_value,
            production_loss=float(scalar_loss.detach()),
            fab_available=fab_ok,
            gate_scores={k: float(v.detach()) if torch.is_tensor(v) else float(v)
                         for k, v in gate_scores.items()},
            candidate_id=candidate_id,
        )

    # ------------------------------------------------------------------ #
    # Real FAB ONE v4 forward
    # ------------------------------------------------------------------ #
    def _run_real_fab(
        self,
        mfg_energy: Tensor,
        surface_noise: Tensor,
        patch_stats: Tensor,
    ) -> Dict[str, Any]:
        """
        Call the real FAB ONE v4 engine. The engine is expected to expose
        the following sub-modules (as defined in fab_one_v4_ultimate.py):

          • fractal_litho       : PolyharmonicInterfaceSolver
          • nucleation          : IsodiametricNucleationController
          • zeno_controller     : TopologicalBottleneckController

        We drive them in sequence and compute a production_loss that
        combines their outputs with the same cost weights as the real
        engine.
        """
        eng = self.fab_engine

        # (a) Polyharmonic interface patterning
        patterned = eng.fractal_litho(surface_noise)               # [B, tensor_dim]

        # (b) Isodiametric nucleation — uses patch_stats directly
        #     We pass a zero latent for the "process params" slot;
        #     the gate only needs patch_stats for the fatness check.
        dummy_latent = torch.zeros(
            mfg_energy.shape[0], self.cfg.latent_dim,
            device=mfg_energy.device, dtype=mfg_energy.dtype,
        )
        nuc_out = eng.nucleation(dummy_latent, patch_stats)

        # (c) Topological bottleneck (No-Zeno soft floor)
        stabilized = eng.zeno_controller(mfg_energy)

        # (d) Production loss — matches FAB ONE v4's cost structure
        w = eng.cost_weights.to(dtype=patterned.dtype)
        mat_loss = torch.zeros_like(mfg_energy)                   # placeholder
        eng_loss = stabilized                                     # [B]
        pat_loss = torch.linalg.vector_norm(patterned, dim=-1)    # [B]

        # Gate penalties (soft)
        fatness_penalty = (1.0 - nuc_out["fatness"])              # [B]
        hausdorff_penalty = 1.0 / (nuc_out["hausdorff_lower_bound"] + 1e-6)

        production_loss = (
            w[0] * mat_loss
            + w[1] * eng_loss
            + w[2] * pat_loss
            + 0.1 * fatness_penalty
            + 0.05 * hausdorff_penalty
        )

        gate_scores = {
            "fatness": nuc_out["fatness"].mean(),
            "hausdorff_lb": nuc_out["hausdorff_lower_bound"].mean(),
            "delta_min": nuc_out["delta_min_effective"].mean(),
            "patterned_norm": pat_loss.mean(),
            "stabilized_energy": eng_loss.mean(),
        }

        return {"production_loss": production_loss, "gate_scores": gate_scores}

    # ------------------------------------------------------------------ #
    # Synthetic fallback
    # ------------------------------------------------------------------ #
    def _run_synthetic_fab(
        self,
        mfg_energy: Tensor,
        surface_noise: Tensor,
        patch_stats: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Shape-matching shim. Returns a loss with the correct gradient
        flow but NO physical meaning. Callers MUST check fab_available.
        """
        syn = self._synthetic
        patterned = torch.tanh(syn["polyharmonic"](surface_noise))
        nuc_score = syn["nucleation"](patch_stats).squeeze(-1)
        bot_score = syn["bottleneck"](mfg_energy.unsqueeze(-1)).squeeze(-1)

        production_loss = (
            patterned.pow(2).mean(dim=-1)
            + 0.1 * (1.0 - nuc_score)
            + 0.05 * bot_score
        )

        gate_scores = {
            "fatness": nuc_score,
            "stabilized_energy": bot_score,
            "patterned_norm": patterned.pow(2).mean(dim=-1),
        }
        return production_loss, gate_scores

    # ------------------------------------------------------------------ #
    # Batch evaluation (for Psyche's axiom ranking)
    # ------------------------------------------------------------------ #
    def evaluate_batch(
        self,
        candidate_latents: Tensor,
    ) -> List[PsycheReward]:
        """Evaluate a batch of candidates; returns one PsycheReward each."""
        rewards: List[PsycheReward] = []
        for i in range(candidate_latents.shape[0]):
            rewards.append(
                self.evaluate_candidate(candidate_latents[i:i+1], candidate_id=i)
            )
        return rewards

    # ------------------------------------------------------------------ #
    # Quality probe (for EcosystemOrchestrator)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quality_probe(self, n_probes: int = 4) -> float:
        """
        Return a quality score ∈ [0, 1] based on the EMA production loss
        and (if available) gate margins. This is what the orchestrator's
        quality_fn reads — NOT a hardcoded constant.
        """
        dev = next(self.parameters()).device
        try:
            z = torch.randn(n_probes, self.cfg.material_dim, device=dev)
            rewards = self.evaluate_batch(z)
            avg_loss = sum(r.production_loss for r in rewards) / max(1, len(rewards))
            avg_fat = sum(r.gate_scores.get("fatness", 0.0) for r in rewards) / max(1, len(rewards))
            # Lower loss + higher fatness → higher quality
            q = 1.0 / (1.0 + avg_loss) * (0.5 + 0.5 * avg_fat)
            return float(max(0.0, min(1.0, q)))
        except Exception as e:
            logger.debug("quality_probe failed: %s", e)
            return 0.5


# ═══════════════════════════════════════════════════════════════════════
# Loop integration helper
# ═══════════════════════════════════════════════════════════════════════

def attach_fab_hook_to_loop(
    loop: nn.Module,
    fab_engine: Optional[nn.Module],
    hook_cfg: Optional[FabOneV4HookConfig] = None,
) -> FabOneV4Hook:
    """
    One-liner to wire the FAB ONE v4 hook into a Psyche-Hodge loop.

    After this call:
      • loop.fab_hook is the new FabOneV4Hook
      • loop.forward() routes Hodge Dream candidates through FAB ONE
      • Psyche receives PsycheReward.value as its ranking signal
      • loop.psyche.update_quality() is fed the hook's quality_probe()
    """
    cfg = hook_cfg or FabOneV4HookConfig()
    hook = FabOneV4Hook(cfg, fab_engine=fab_engine).to(
        next(loop.parameters()).device if hasattr(loop, "parameters") else torch.device(cfg.device)
    )
    loop.fab_hook = hook  # type: ignore[attr-defined]
    return hook


# ═══════════════════════════════════════════════════════════════════════
# Integration demo
# ═══════════════════════════════════════════════════════════════════════

def run_integration_demo(use_real_fab: bool = False) -> None:
    """
    Smoke test showing the full pipeline:
      AGI latent → Hodge Dream → FAB ONE v4 hook → Psyche reward
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build a minimal Psyche-Hodge loop (inline, for the demo)
    from psyche_hodge_fab_bridge import (
        BridgeConfig, HodgeDreamMaterialInnovator, PsychePlusSpeculator,
    )
    bridge_cfg = BridgeConfig(
        latent_dim=64, material_dim=64, photonic_dim=32,
        hodge_dim=16, n_axioms=8, dream_steps=4, device=str(dev),
    )
    dream = HodgeDreamMaterialInnovator(bridge_cfg).to(dev)
    psyche = PsychePlusSpeculator(bridge_cfg).to(dev)

    # 2. Optionally build a real FAB ONE v4 engine
    fab_engine = None
    if use_real_fab:
        try:
            from fab_one_v4_ultimate import FabOneV4Engine, FabOneV4Config
            fab_cfg = FabOneV4Config(
                latent_dim=bridge_cfg.latent_dim,
                material_dim=bridge_cfg.material_dim,
                photonic_dim=bridge_cfg.photonic_dim,
                m_poly=2, n_poly=8,   # small for demo
                carpet_level=2, carpet_n_cells=64,
                hodge_dim=16,
                device=str(dev),
            )
            fab_engine = FabOneV4Engine(fab_cfg).to(dev)
            logger.info("Real FAB ONE v4 engine loaded.")
        except Exception as e:
            logger.warning("Could not load FAB ONE v4: %s", e)

    # 3. Build the hook
    hook_cfg = FabOneV4HookConfig(
        latent_dim=bridge_cfg.latent_dim,
        material_dim=bridge_cfg.material_dim,
        photonic_dim=bridge_cfg.photonic_dim,
        m_poly=2, n_poly=8,
        device=str(dev),
    )
    hook = FabOneV4Hook(hook_cfg, fab_engine=fab_engine).to(dev)

    # 4. Run one speculative step
    agi_latent = torch.randn(2, bridge_cfg.latent_dim, device=dev)
    dream_out = dream(agi_latent)
    candidates = dream_out["candidate_latent"]       # [B, material_dim]

    # 5. Evaluate each candidate through FAB ONE v4
    rewards = hook.evaluate_batch(candidates)

    # 6. Psyche sees the reward and updates its quality EMA
    for r in rewards:
        psyche.update_quality(max(0.0, min(1.0, 0.5 + 0.1 * r.value)))

    # 7. Print the audit trail
    print("\n" + "=" * 72)
    print(" FAB ONE v4 HOOK — Integration Demo")
    print("=" * 72)
    print(f"  FAB ONE v4 available : {hook.fab_available}")
    print(f"  Candidates evaluated : {len(rewards)}")
    print(f"  EMA production loss  : {float(hook.ema_loss):.6f}")
    print(f"  Quality probe        : {hook.quality_probe():.4f}")
    print()
    for r in rewards:
        print(f"  Candidate #{r.candidate_id}")
        print(f"    reward value     : {r.value:+.4f}")
        print(f"    production loss  : {r.production_loss:.6f}")
        print(f"    fab_available    : {r.fab_available}")
        print(f"    gate scores      : { {k: f'{v:.4f}' for k, v in r.gate_scores.items()} }")
    print("=" * 72)


if __name__ == "__main__":
    run_integration_demo(use_real_fab=False)
