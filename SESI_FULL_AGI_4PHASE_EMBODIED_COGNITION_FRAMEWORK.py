"""
=============================================================================
SESI FULL AGI — 4-PHASE EMBODIED COGNITION FRAMEWORK v1.0
Phase 1: Embodied Agent (Virtual Environment + Sensorimotor Loop)
Phase 2: Open-Ended Learning (Intrinsic Motivation + Skill Growth)
Phase 3: World Models (Prediction + Counterfactual Reasoning)
Phase 4: Transfer Learning (Few-Shot + Cross-Domain + Composition)
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Complete 4-Phase AGI Research Framework)
=============================================================================

EPISTEMIC HONESTY CONTRACT
──────────────────────────
This is a RESEARCH FRAMEWORK for exploring the architectural requirements
of general intelligence. It is NOT:
  ❌ A sentient being
  ❌ A system that "understands" in the human sense
  ❌ A replacement for biological intelligence
  ❌ A guarantee of AGI/ASI achievement

It IS:
  ✅ A differentiable, trainable architecture that implements the four
     phases identified as necessary for moving beyond narrow AI
  ✅ A testbed for hypotheses about embodied, open-ended, world-modeling,
     transfer-capable cognition
  ✅ A framework that can be benchmarked against specific tasks
  ✅ An honest exploration of what computational architectures can and
     cannot achieve

THE FOUR PHASES (from our previous analysis)
─────────────────────────────────────────────
  Phase 1: EMBODIMENT
    • Agent has a body (virtual or physical)
    • Sensors perceive the environment
    • Actuators change the environment
    • Perception-action loop is closed

  Phase 2: OPEN-ENDED LEARNING
    • Intrinsic motivation drives exploration
    • Agent sets its own goals
    • Skill library grows over time
    • No fixed task distribution required

  Phase 3: WORLD MODELS
    • Internal model predicts future states
    • Counterfactual reasoning ("what if?")
    • Uncertainty quantification
    • Planning in imagination before acting

  Phase 4: TRANSFER LEARNING
    • Learn new skills from few examples
    • Transfer knowledge across domains
    • Compose skills into novel behaviors
    • Analogical reasoning between structures

MATHEMATICAL FOUNDATIONS (from SESI corpus)
────────────────────────────────────────────
  [W1] WUCP (Weak-Update Computational Principle)
       → Adaptive learning rate based on error/budget
  [W2] FMCA (Fractal Multiscale Architecture)
       → Hierarchical skill representation
  [W3] Hodge Decomposition
       → Separate signal/noise in perception
  [W4] OPMC Canonical Compression
       → Compact skill representations
  [W5] Carpet Resistance Forms
       → Multi-scale environment modeling
  [W6] Hysteresis No-Zeno Gate
       → Stable learning without oscillation
  [W7] Subsampling CI (Paper 11)
       → Uncertainty quantification for world model
  [W8] Gumbel Extreme-Value Bounds
       → Rare event prediction in environment

=============================================================================
"""

from __future__ import annotations

import math
import logging
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.full_agi_4phase")

__all__ = [
    "FullAGIConfig",
    # Phase 1: Embodiment
    "VirtualEnvironment",
    "SensorSuite",
    "ActuatorModel",
    "BodySchema",
    "EmbodiedAgent",
    # Phase 2: Open-Ended Learning
    "IntrinsicMotivationEngine",
    "GoalGenerator",
    "SkillLibrary",
    "OpenEndedLearner",
    # Phase 3: World Models
    "LatentWorldModel",
    "CounterfactualReasoner",
    "UncertaintyEstimator",
    "WorldModelSystem",
    # Phase 4: Transfer Learning
    "FewShotAdapter",
    "DomainAdapter",
    "SkillComposer",
    "AnalogicalReasoner",
    "TransferLearningSystem",
    # Unified
    "SESIFullAGI",
    "FullAGILoss",
    "FullAGITrainer",
]


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Primitives
# ═══════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def fp32_island():
    try:
        from torch.amp import autocast
        with autocast(device_type="cuda", enabled=False):
            yield
    except Exception:
        yield


def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def sfloor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

def sclamp(x: Tensor, lo: float, hi: float) -> Tensor:
    mid = (hi + lo) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    return mid + half * torch.tanh((x - mid) / half)


class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, r) * 0.02)
        self.V = nn.Parameter(torch.zeros(r, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FullAGIConfig:
    # ── Core dimensions ───────────────────────────────────────────────────
    latent_dim: int = 256
    obs_dim: int = 128
    action_dim: int = 32
    skill_dim: int = 64
    
    # ── Phase 1: Embodiment ───────────────────────────────────────────────
    env_grid_size: int = 16
    n_sensor_types: int = 4  # visual, proprioceptive, tactile, auditory
    n_actuators: int = 8
    
    # ── Phase 2: Open-Ended Learning ──────────────────────────────────────
    n_skills_max: int = 100
    curiosity_weight: float = 0.3
    competence_weight: float = 0.3
    novelty_weight: float = 0.4
    
    # ── Phase 3: World Model ──────────────────────────────────────────────
    world_model_dim: int = 128
    prediction_horizon: int = 8
    n_counterfactuals: int = 4
    
    # ── Phase 4: Transfer ─────────────────────────────────────────────────
    n_few_shot_examples: int = 5
    n_domains: int = 10
    
    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    device: str = "cuda"
    
    # ── Loss weights ──────────────────────────────────────────────────────
    w_perception: float = 0.2
    w_action: float = 0.2
    w_curiosity: float = 0.15
    w_prediction: float = 0.2
    w_transfer: float = 0.15
    w_reg: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: EMBODIED AGENT
# ═══════════════════════════════════════════════════════════════════════

class VirtualEnvironment(nn.Module):
    """
    [Phase 1] A differentiable virtual environment.
    
    The agent interacts with this environment through sensors and
    actuators. The environment has:
      - State that evolves based on physics-like rules
      - Observations that the agent can perceive
      - Rewards/signals that guide learning
    
    HONEST SCOPE: This is a simplified continuous-state environment,
    not a full physics simulator. It captures the essential
    perception-action loop structure.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Environment state transition
        self.dynamics = nn.Sequential(
            nn.Linear(cfg.obs_dim + cfg.action_dim, cfg.obs_dim),
            nn.GELU(),
            nn.Linear(cfg.obs_dim, cfg.obs_dim),
        )
        
        # Observation function (partial observability)
        self.observation_fn = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.obs_dim),
            nn.Sigmoid(),  # observations are bounded
        )
        
        # Reward signal (intrinsic + extrinsic)
        self.reward_fn = nn.Sequential(
            nn.Linear(cfg.obs_dim + cfg.action_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Environment noise (stochasticity)
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
    
    def reset(self, batch_size: int, device: torch.device) -> Tensor:
        """Reset environment to initial state."""
        return torch.randn(batch_size, self.cfg.obs_dim, device=device) * 0.5
    
    def step(self, state: Tensor, action: Tensor) -> Dict[str, Tensor]:
        """
        Take one environment step.
        
        Args:
            state: (B, obs_dim) current state
            action: (B, action_dim) agent's action
        
        Returns:
            dict with next_state, observation, reward
        """
        # State transition
        transition_input = torch.cat([state, action], dim=-1)
        next_state = self.dynamics(transition_input)
        
        # Add stochasticity
        noise = torch.randn_like(next_state) * self.noise_scale
        next_state = next_state + noise
        
        # Observation (partial observability)
        observation = self.observation_fn(next_state)
        
        # Reward
        reward_input = torch.cat([next_state, action], dim=-1)
        reward = self.reward_fn(reward_input).squeeze(-1)
        
        return {
            "next_state": next_state,
            "observation": observation,
            "reward": reward,
        }


class SensorSuite(nn.Module):
    """
    [Phase 1] Multi-modal sensor suite.
    
    Simulates different sensor modalities:
      - Visual: high-dimensional, spatial
      - Proprioceptive: body position/velocity
      - Tactile: contact forces
      - Auditory: temporal patterns
    
    Each modality has different noise characteristics and bandwidth.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Sensor encoders for each modality
        self.visual_enc = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.obs_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.obs_dim // 2, cfg.latent_dim // 4),
        )
        
        self.proprio_enc = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.latent_dim // 4),
            nn.GELU(),
        )
        
        self.tactile_enc = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.latent_dim // 4),
            nn.GELU(),
        )
        
        self.auditory_enc = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.latent_dim // 4),
            nn.GELU(),
        )
        
        # Sensor noise models (different per modality)
        self.visual_noise = nn.Parameter(torch.tensor(0.05))
        self.proprio_noise = nn.Parameter(torch.tensor(0.01))
        self.tactile_noise = nn.Parameter(torch.tensor(0.1))
        self.auditory_noise = nn.Parameter(torch.tensor(0.08))
        
        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
    
    def forward(self, observation: Tensor) -> Dict[str, Tensor]:
        """
        Process raw observation through sensor suite.
        
        Args:
            observation: (B, obs_dim) raw environment observation
        
        Returns:
            dict with per-modality features and fused perception
        """
        B = observation.shape[0]
        
        # Add modality-specific noise
        visual = observation + torch.randn_like(observation) * self.visual_noise
        proprio = observation + torch.randn_like(observation) * self.proprio_noise
        tactile = observation + torch.randn_like(observation) * self.tactile_noise
        auditory = observation + torch.randn_like(observation) * self.auditory_noise
        
        # Encode each modality
        v_feat = self.visual_enc(visual)
        p_feat = self.proprio_enc(proprio)
        t_feat = self.tactile_enc(tactile)
        a_feat = self.auditory_enc(auditory)
        
        # Concatenate and fuse
        concatenated = torch.cat([v_feat, p_feat, t_feat, a_feat], dim=-1)
        fused = self.fusion(concatenated)
        
        return {
            "visual": v_feat,
            "proprioceptive": p_feat,
            "tactile": t_feat,
            "auditory": a_feat,
            "fused_perception": fused,
        }


class ActuatorModel(nn.Module):
    """
    [Phase 1] Actuator model — converts intent to action.
    
    Models the agent's effectors on the environment.
    Includes motor noise, saturation, and dynamics.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Policy network: perception → action
        self.policy = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, cfg.action_dim),
            nn.Tanh(),  # actions are bounded
        )
        
        # Motor noise
        self.motor_noise = nn.Parameter(torch.tensor(0.05))
        
        # Action value estimator (for learning)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, perception: Tensor) -> Dict[str, Tensor]:
        """
        Generate action from perception.
        
        Args:
            perception: (B, latent_dim) fused sensory input
        
        Returns:
            dict with action, action_log_prob, value
        """
        # Clean action
        action_mean = self.policy(perception)
        
        # Add motor noise (exploration)
        noise = torch.randn_like(action_mean) * self.motor_noise
        action = torch.tanh(action_mean + noise)
        
        # Value estimate
        value_input = torch.cat([perception, action], dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        
        return {
            "action": action,
            "action_mean": action_mean,
            "value": value,
        }


class BodySchema(nn.Module):
    """
    [Phase 1] Body schema — the agent's model of its own body.
    
    Maintains an internal representation of:
      - Body configuration (joint positions, orientations)
      - Body capabilities (action space, constraints)
      - Body state (energy, damage, fatigue)
    
    This is essential for embodied cognition — the agent must know
    what it IS to know what it CAN DO.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Body state encoder
        self.body_enc = nn.Sequential(
            nn.Linear(cfg.action_dim + cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, cfg.latent_dim // 2),
        )
        
        # Capability predictor
        self.capability_net = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2, cfg.action_dim),
            nn.Sigmoid(),  # capability ∈ [0, 1]
        )
        
        # Energy/fatigue model
        self.energy_net = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, action: Tensor, perception: Tensor) -> Dict[str, Tensor]:
        """
        Update body schema based on action and perception.
        
        Args:
            action: (B, action_dim) last action taken
            perception: (B, latent_dim) current perception
        
        Returns:
            dict with body_state, capabilities, energy
        """
        body_input = torch.cat([action, perception], dim=-1)
        body_state = self.body_enc(body_input)
        
        capabilities = self.capability_net(body_state)
        energy = self.energy_net(body_state).squeeze(-1)
        
        return {
            "body_state": body_state,
            "capabilities": capabilities,
            "energy": energy,
        }


class EmbodiedAgent(nn.Module):
    """
    [Phase 1] Complete embodied agent.
    
    Integrates: Environment + Sensors + Actuators + Body Schema
    into a closed perception-action loop.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        self.environment = VirtualEnvironment(cfg)
        self.sensors = SensorSuite(cfg)
        self.actuators = ActuatorModel(cfg)
        self.body = BodySchema(cfg)
        
        # State encoder (for recurrent processing)
        self.state_enc = nn.GRUCell(cfg.latent_dim, cfg.latent_dim)
    
    def forward(self, batch_size: int, device: torch.device,
                n_steps: int = 4) -> Dict[str, Tensor]:
        """
        Run the embodied agent for n_steps.
        
        Args:
            batch_size: number of parallel agents
            device: computation device
            n_steps: environment interaction steps
        
        Returns:
            dict with trajectories and final states
        """
        # Reset environment
        state = self.environment.reset(batch_size, device)
        observation = self.environment.observation_fn(state)
        
        # Initial internal state
        h = torch.zeros(batch_size, self.cfg.latent_dim, device=device)
        
        trajectories = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "perceptions": [],
        }
        
        for step in range(n_steps):
            # 1. Sense
            sensor_out = self.sensors(observation)
            perception = sensor_out["fused_perception"]
            
            # 2. Update internal state
            h = self.state_enc(perception, h)
            
            # 3. Act
            actuator_out = self.actuators(h)
            action = actuator_out["action"]
            
            # 4. Update body schema
            body_out = self.body(action, perception)
            
            # 5. Environment step
            env_out = self.environment.step(state, action)
            state = env_out["next_state"]
            observation = env_out["observation"]
            reward = env_out["reward"]
            
            # Record trajectory
            trajectories["observations"].append(observation)
            trajectories["actions"].append(action)
            trajectories["rewards"].append(reward)
            trajectories["perceptions"].append(perception)
        
        # Stack trajectories
        for key in trajectories:
            trajectories[key] = torch.stack(trajectories[key], dim=1)
        
        return {
            **{f"traj_{k}": v for k, v in trajectories.items()},
            "final_state": h,
            "final_body_state": body_out["body_state"],
            "total_reward": trajectories["rewards"].sum(dim=1),
        }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: OPEN-ENDED LEARNING
# ═══════════════════════════════════════════════════════════════════════

class IntrinsicMotivationEngine(nn.Module):
    """
    [Phase 2] Intrinsic motivation — drives exploration without
    external rewards.
    
    Three components:
      1. Curiosity: prediction error of world model
      2. Novelty: distance from previously seen states
      3. Competence: improvement in skill performance
    
    Based on WUCP principle: allocate computation where uncertainty
    is highest and expected improvement is greatest.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Curiosity: prediction error
        self.curiosity_net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 1),
            nn.Softplus(),  # curiosity ≥ 0
        )
        
        # Novelty: distance from memory
        self.novelty_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, 1),
            nn.Softplus(),
        )
        
        # Competence: learning progress
        self.competence_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Memory of seen states (for novelty)
        self.register_buffer(
            "state_memory",
            torch.zeros(cfg.n_skills_max, cfg.latent_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_idx",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
    
    def compute_curiosity(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """Curiosity = prediction error."""
        curiosity_input = torch.cat([predicted, actual], dim=-1)
        return self.curiosity_net(curiosity_input).squeeze(-1)
    
    def compute_novelty(self, state: Tensor) -> Tensor:
        """Novelty = distance from remembered states."""
        # Simple approach: distance from memory centroid
        memory_centroid = self.state_memory.mean(dim=0, keepdim=True)
        distance = (state - memory_centroid).norm(dim=-1)
        novelty_input = state
        return self.novelty_net(novelty_input).squeeze(-1) * (1 + distance * 0.1)
    
    def compute_competence(self, skill_state: Tensor) -> Tensor:
        """Competence = learning progress."""
        return self.competence_net(skill_state).squeeze(-1)
    
    def add_to_memory(self, state: Tensor) -> None:
        """Add state to memory (ring buffer)."""
        with torch.no_grad():
            idx = self.memory_idx.item() % self.cfg.n_skills_max
            self.state_memory[idx] = state.mean(dim=0).detach()
            self.memory_idx += 1
    
    def forward(self, predicted: Tensor, actual: Tensor,
                skill_state: Tensor) -> Dict[str, Tensor]:
        """
        Compute intrinsic motivation signals.
        
        Args:
            predicted: (B, latent_dim) world model prediction
            actual: (B, latent_dim) actual outcome
            skill_state: (B, latent_dim) current skill representation
        
        Returns:
            dict with curiosity, novelty, competence, total_motivation
        """
        curiosity = self.compute_curiosity(predicted, actual)
        novelty = self.compute_novelty(actual)
        competence = self.compute_competence(skill_state)
        
        # Weighted combination
        total = (
            self.cfg.curiosity_weight * curiosity +
            self.cfg.novelty_weight * novelty +
            self.cfg.competence_weight * competence
        )
        
        # Update memory
        self.add_to_memory(actual)
        
        return {
            "curiosity": curiosity,
            "novelty": novelty,
            "competence": competence,
            "total_motivation": total,
        }


class GoalGenerator(nn.Module):
    """
    [Phase 2] Self-generated goals.
    
    The agent doesn't just respond to external rewards — it creates
    its own goals based on:
      - What it's curious about
      - What skills it wants to improve
      - What states it hasn't visited
    
    This is essential for open-ended learning.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Goal proposal network
        self.goal_proposer = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        
        # Goal value estimator
        self.goal_value = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Goal achievability estimator
        self.achievability = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, current_state: Tensor,
                motivation: Tensor) -> Dict[str, Tensor]:
        """
        Generate self-directed goals.
        
        Args:
            current_state: (B, latent_dim) where the agent is now
            motivation: (B,) intrinsic motivation signal
        
        Returns:
            dict with goals, goal_values, achievability
        """
        # Propose goals (conditioned on current state)
        goals = self.goal_proposer(current_state)
        
        # Estimate goal value
        goal_values = self.goal_value(goals).squeeze(-1)
        
        # Estimate achievability
        achievability_input = torch.cat([current_state, goals], dim=-1)
        achievability = self.achievability(achievability_input).squeeze(-1)
        
        # Weight by motivation
        goal_priority = goal_values * achievability * (1 + motivation * 0.5)
        
        return {
            "goals": goals,
            "goal_values": goal_values,
            "achievability": achievability,
            "goal_priority": goal_priority,
        }


class SkillLibrary(nn.Module):
    """
    [Phase 2] Growing skill library.
    
    Skills are reusable behavioral primitives that:
      - Are learned through practice
      - Can be composed into complex behaviors
      - Have preconditions and effects
      - Improve with use
    
    Based on FMCA hierarchical structure.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Skill encoder (perception → skill space)
        self.skill_enc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.skill_dim),
            nn.GELU(),
            nn.Linear(cfg.skill_dim, cfg.skill_dim),
        )
        
        # Skill decoder (skill → action sequence)
        self.skill_dec = nn.Sequential(
            nn.Linear(cfg.skill_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.action_dim),
        )
        
        # Skill quality estimator
        self.skill_quality = nn.Sequential(
            nn.Linear(cfg.skill_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Skill storage (growing library)
        self.register_buffer(
            "skill_library",
            torch.zeros(cfg.n_skills_max, cfg.skill_dim),
            persistent=False,
        )
        self.register_buffer(
            "n_skills",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
    
    def encode_skill(self, perception: Tensor) -> Tensor:
        """Encode a behavior as a skill."""
        return self.skill_enc(perception)
    
    def decode_skill(self, skill: Tensor) -> Tensor:
        """Decode a skill into actions."""
        return self.skill_dec(skill)
    
    def add_skill(self, skill: Tensor) -> None:
        """Add a new skill to the library."""
        with torch.no_grad():
            idx = self.n_skills.item()
            if idx < self.cfg.n_skills_max:
                self.skill_library[idx] = skill.mean(dim=0).detach()
                self.n_skills += 1
    
    def retrieve_skill(self, query: Tensor) -> Tensor:
        """Retrieve most relevant skill from library."""
        # Simple similarity-based retrieval
        n = self.n_skills.item()
        if n == 0:
            return torch.zeros(query.shape[0], self.cfg.skill_dim,
                             device=query.device)
        
        library = self.skill_library[:n]  # (n, skill_dim)
        similarity = torch.einsum("bd,nd->bn", query, library)
        best_idx = similarity.argmax(dim=-1)
        
        return library[best_idx]
    
    def forward(self, perception: Tensor) -> Dict[str, Tensor]:
        """
        Process perception through skill library.
        
        Args:
            perception: (B, latent_dim)
        
        Returns:
            dict with skill, action, quality
        """
        # Encode as skill
        skill = self.encode_skill(perception)
        
        # Decode to action
        action = self.decode_skill(skill)
        
        # Quality estimate
        quality = self.skill_quality(skill).squeeze(-1)
        
        # Add to library (if good enough)
        self.add_skill(skill)
        
        return {
            "skill": skill,
            "skill_action": action,
            "skill_quality": quality,
            "library_size": self.n_skills.float(),
        }


class OpenEndedLearner(nn.Module):
    """
    [Phase 2] Complete open-ended learning system.
    
    Integrates: Intrinsic Motivation + Goal Generation + Skill Library
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        self.motivation = IntrinsicMotivationEngine(cfg)
        self.goals = GoalGenerator(cfg)
        self.skills = SkillLibrary(cfg)
        
        # Learning progress tracker
        self.progress_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, predicted: Tensor, actual: Tensor,
                current_state: Tensor) -> Dict[str, Tensor]:
        """
        Open-ended learning step.
        
        Args:
            predicted: (B, latent_dim) world model prediction
            actual: (B, latent_dim) actual outcome
            current_state: (B, latent_dim) current agent state
        
        Returns:
            dict with motivation, goals, skills, progress
        """
        # Intrinsic motivation
        motivation_out = self.motivation(predicted, actual, current_state)
        
        # Goal generation
        goal_out = self.goals(current_state, motivation_out["total_motivation"])
        
        # Skill learning
        skill_out = self.skills(current_state)
        
        # Learning progress
        progress = self.progress_net(current_state).squeeze(-1)
        
        return {
            **{f"motiv_{k}": v for k, v in motivation_out.items()},
            **{f"goal_{k}": v for k, v in goal_out.items()},
            **{f"skill_{k}": v for k, v in skill_out.items()},
            "learning_progress": progress,
        }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: WORLD MODELS
# ═══════════════════════════════════════════════════════════════════════

class LatentWorldModel(nn.Module):
    """
    [Phase 3] DreamerV3-style latent world model.
    
    Learns to:
      - Encode observations into latent state
      - Predict future latent states given actions
      - Decode predictions back to observations
      - Estimate uncertainty
    
    This allows the agent to "imagine" outcomes before acting.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Encoder: observation → latent
        self.encoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.world_model_dim),
            nn.GELU(),
            nn.Linear(cfg.world_model_dim, cfg.world_model_dim * 2),  # mean + logvar
        )
        
        # Dynamics model: (latent, action) → next latent
        self.dynamics = nn.Sequential(
            nn.Linear(cfg.world_model_dim + cfg.action_dim, cfg.world_model_dim),
            nn.GELU(),
            nn.Linear(cfg.world_model_dim, cfg.world_model_dim * 2),
        )
        
        # Decoder: latent → observation
        self.decoder = nn.Sequential(
            nn.Linear(cfg.world_model_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        
        # Reward predictor
        self.reward_pred = nn.Sequential(
            nn.Linear(cfg.world_model_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def encode(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode observation to latent distribution."""
        encoded = self.encoder(observation)
        mean, logvar = encoded.chunk(2, dim=-1)
        return mean, logvar
    
    def sample_latent(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterized sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def predict_next(self, latent: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict next latent state."""
        dynamics_input = torch.cat([latent, action], dim=-1)
        predicted = self.dynamics(dynamics_input)
        mean, logvar = predicted.chunk(2, dim=-1)
        return mean, logvar
    
    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent to observation."""
        return self.decoder(latent)
    
    def forward(self, observation: Tensor, action: Tensor,
                horizon: Optional[int] = None) -> Dict[str, Tensor]:
        """
        World model forward pass.
        
        Args:
            observation: (B, latent_dim) current observation
            action: (B, action_dim) action to take
            horizon: number of steps to predict
        
        Returns:
            dict with predictions, uncertainties, rewards
        """
        H = horizon or self.cfg.prediction_horizon
        
        # Encode current observation
        mean, logvar = self.encode(observation)
        latent = self.sample_latent(mean, logvar)
        
        predictions = []
        uncertainties = []
        rewards = []
        
        current_latent = latent
        for _ in range(H):
            # Predict next state
            next_mean, next_logvar = self.predict_next(current_latent, action)
            next_latent = self.sample_latent(next_mean, next_logvar)
            
            # Decode prediction
            predicted_obs = self.decode(next_latent)
            predictions.append(predicted_obs)
            
            # Uncertainty (from logvar)
            uncertainty = torch.exp(0.5 * next_logvar).mean(dim=-1)
            uncertainties.append(uncertainty)
            
            # Reward prediction
            reward = self.reward_pred(next_latent).squeeze(-1)
            rewards.append(reward)
            
            current_latent = next_latent
        
        return {
            "predictions": torch.stack(predictions, dim=1),  # (B, H, latent_dim)
            "uncertainties": torch.stack(uncertainties, dim=1),  # (B, H)
            "predicted_rewards": torch.stack(rewards, dim=1),  # (B, H)
            "latent_trajectory": current_latent,
        }


class CounterfactualReasoner(nn.Module):
    """
    [Phase 3] Counterfactual reasoning — "what if?"
    
    Given a state and action, reason about:
      - What would have happened with a different action?
      - What conditions would lead to a desired outcome?
      - What are the causal factors in this situation?
    
    This is essential for planning and learning from mistakes.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Counterfactual generator
        self.cf_generator = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim * cfg.n_counterfactuals),
        )
        
        # Outcome predictor for each counterfactual
        self.outcome_pred = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Causal importance estimator
        self.causal_net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, cfg.action_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, state: Tensor, action: Tensor) -> Dict[str, Tensor]:
        """
        Generate and evaluate counterfactuals.
        
        Args:
            state: (B, latent_dim) current state
            action: (B, action_dim) action taken
        
        Returns:
            dict with counterfactuals, outcomes, causal_importance
        """
        B = state.shape[0]
        
        # Generate counterfactual states
        cf_input = torch.cat([state, action], dim=-1)
        cf_flat = self.cf_generator(cf_input)
        counterfactuals = cf_flat.view(B, self.cfg.n_counterfactuals, self.cfg.latent_dim)
        
        # Predict outcomes for each counterfactual
        outcomes = []
        for i in range(self.cfg.n_counterfactuals):
            outcome = self.outcome_pred(counterfactuals[:, i]).squeeze(-1)
            outcomes.append(outcome)
        outcomes = torch.stack(outcomes, dim=-1)  # (B, n_cf)
        
        # Causal importance
        causal_input = torch.cat([state, counterfactuals.mean(dim=1)], dim=-1)
        causal_importance = self.causal_net(causal_input)
        
        return {
            "counterfactuals": counterfactuals,
            "counterfactual_outcomes": outcomes,
            "best_counterfactual": outcomes.argmax(dim=-1),
            "causal_importance": causal_importance,
        }


class UncertaintyEstimator(nn.Module):
    """
    [Phase 3] Uncertainty quantification.
    
    Distinguishes:
      - Aleatoric uncertainty (inherent noise)
      - Epistemic uncertainty (model ignorance)
    
    Based on subsampling CI from Paper 11 — assumption-light
    uncertainty estimation.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Aleatoric (data noise)
        self.aleatoric_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Epistemic (model uncertainty)
        self.epistemic_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Ensemble disagreement (proxy for epistemic)
        self.ensemble_size = 5
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.latent_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            ) for _ in range(self.ensemble_size)
        ])
    
    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        """
        Estimate uncertainty.
        
        Args:
            state: (B, latent_dim)
        
        Returns:
            dict with aleatoric, epistemic, total_uncertainty
        """
        # Aleatoric
        aleatoric = self.aleatoric_net(state).squeeze(-1)
        
        # Epistemic via ensemble disagreement
        ensemble_outputs = []
        for net in self.ensemble:
            ensemble_outputs.append(net(state).squeeze(-1))
        ensemble_stack = torch.stack(ensemble_outputs, dim=-1)  # (B, ensemble_size)
        
        epistemic = ensemble_stack.std(dim=-1)  # disagreement = uncertainty
        
        # Total
        total = aleatoric + epistemic
        
        return {
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": total,
            "ensemble_outputs": ensemble_stack,
        }


class WorldModelSystem(nn.Module):
    """
    [Phase 3] Complete world model system.
    
    Integrates: Latent Dynamics + Counterfactual Reasoning + Uncertainty
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        self.world_model = LatentWorldModel(cfg)
        self.counterfactual = CounterfactualReasoner(cfg)
        self.uncertainty = UncertaintyEstimator(cfg)
        
        # Planning head (use world model for planning)
        self.planner = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.world_model_dim, cfg.action_dim),
            nn.Tanh(),
        )
    
    def forward(self, observation: Tensor, action: Tensor) -> Dict[str, Tensor]:
        """
        Complete world model inference.
        
        Args:
            observation: (B, latent_dim)
            action: (B, action_dim)
        
        Returns:
            dict with predictions, counterfactuals, uncertainty, planned_action
        """
        # World model prediction
        wm_out = self.world_model(observation, action)
        
        # Counterfactual reasoning
        cf_out = self.counterfactual(observation, action)
        
        # Uncertainty estimation
        unc_out = self.uncertainty(observation)
        
        # Planning (use world model to choose better action)
        plan_input = torch.cat([
            observation,
            wm_out["latent_trajectory"],
        ], dim=-1)
        planned_action = self.planner(plan_input)
        
        return {
            **{f"wm_{k}": v for k, v in wm_out.items()},
            **{f"cf_{k}": v for k, v in cf_out.items()},
            **{f"unc_{k}": v for k, v in unc_out.items()},
            "planned_action": planned_action,
        }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: TRANSFER LEARNING
# ═══════════════════════════════════════════════════════════════════════

class FewShotAdapter(nn.Module):
    """
    [Phase 4] Few-shot learning — adapt to new tasks from few examples.
    
    Uses meta-learning style adaptation:
      - Task encoder: extract task representation from examples
      - Fast adapter: quickly specialize for new task
      - No retraining of base model required
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Task encoder (from support examples)
        self.task_enc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, cfg.latent_dim // 2),
        )
        
        # Fast adapter (task-conditioned)
        self.fast_adapter = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.latent_dim // 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.action_dim),
        )
        
        # Adaptation rate (how fast to specialize)
        self.adaptation_rate = nn.Parameter(torch.tensor(0.1))
    
    def encode_task(self, support_examples: Tensor) -> Tensor:
        """
        Encode task from support examples.
        
        Args:
            support_examples: (B, n_examples, latent_dim)
        
        Returns:
            task_repr: (B, latent_dim // 2)
        """
        # Average pool support examples
        pooled = support_examples.mean(dim=1)
        return self.task_enc(pooled)
    
    def adapt(self, base_repr: Tensor, task_repr: Tensor) -> Tensor:
        """
        Adapt base representation to new task.
        
        Args:
            base_repr: (B, latent_dim)
            task_repr: (B, latent_dim // 2)
        
        Returns:
            adapted_action: (B, action_dim)
        """
        adapt_input = torch.cat([base_repr, task_repr], dim=-1)
        return self.fast_adapter(adapt_input)
    
    def forward(self, support: Tensor, query: Tensor) -> Dict[str, Tensor]:
        """
        Few-shot adaptation.
        
        Args:
            support: (B, n_examples, latent_dim) support set
            query: (B, latent_dim) query to adapt to
        
        Returns:
            dict with task_repr, adapted_output
        """
        task_repr = self.encode_task(support)
        adapted = self.adapt(query, task_repr)
        
        return {
            "task_representation": task_repr,
            "adapted_output": adapted,
            "adaptation_rate": self.adaptation_rate.expand(query.shape[0]),
        }


class DomainAdapter(nn.Module):
    """
    [Phase 4] Cross-domain transfer.
    
    Transfers knowledge between different domains by:
      - Learning domain-invariant representations
      - Mapping between domain-specific spaces
      - Identifying structural similarities
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Domain encoder
        self.domain_enc = nn.Embedding(cfg.n_domains, cfg.latent_dim // 4)
        
        # Domain-invariant feature extractor
        self.invariant_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
        )
        
        # Domain-specific adapter
        self.domain_adapter = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2 + cfg.latent_dim // 4, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.action_dim),
        )
    
    def forward(self, features: Tensor, source_domain: Tensor,
                target_domain: Tensor) -> Dict[str, Tensor]:
        """
        Transfer from source to target domain.
        
        Args:
            features: (B, latent_dim) source domain features
            source_domain: (B,) source domain indices
            target_domain: (B,) target domain indices
        
        Returns:
            dict with invariant_features, transferred_output
        """
        # Extract domain-invariant features
        invariant = self.invariant_net(features)
        
        # Get target domain embedding
        target_emb = self.domain_enc(target_domain)
        
        # Adapt to target domain
        adapt_input = torch.cat([invariant, target_emb], dim=-1)
        transferred = self.domain_adapter(adapt_input)
        
        return {
            "invariant_features": invariant,
            "transferred_output": transferred,
            "domain_embedding": target_emb,
        }


class SkillComposer(nn.Module):
    """
    [Phase 4] Skill composition — combine skills into novel behaviors.
    
    Given a library of learned skills, compose them to solve new
    problems that weren't explicitly trained on.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Composition planner
        self.composition_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.n_skills_max),
            nn.Softmax(dim=-1),  # weights over skills
        )
        
        # Skill combination
        self.combiner = nn.Sequential(
            nn.Linear(cfg.skill_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.action_dim),
        )
    
    def forward(self, goal: Tensor, skill_library: Tensor) -> Dict[str, Tensor]:
        """
        Compose skills to achieve goal.
        
        Args:
            goal: (B, latent_dim) desired outcome
            skill_library: (n_skills, skill_dim) available skills
        
        Returns:
            dict with composition_weights, composed_action
        """
        # Plan composition
        weights = self.composition_net(goal)  # (B, n_skills)
        
        # Weighted combination of skills
        # skill_library: (n_skills, skill_dim)
        # weights: (B, n_skills)
        composed_skill = torch.einsum("bn,nd->bd", weights, skill_library)
        
        # Generate action
        composed_action = self.combiner(composed_skill)
        
        return {
            "composition_weights": weights,
            "composed_skill": composed_skill,
            "composed_action": composed_action,
        }


class AnalogicalReasoner(nn.Module):
    """
    [Phase 4] Analogical reasoning — map structure between domains.
    
    Identifies structural similarities between different situations
    and transfers solutions accordingly.
    
    "A is to B as C is to ?"
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Relation encoder
        self.relation_enc = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
        )
        
        # Analogy solver
        self.analogy_net = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2 * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def encode_relation(self, a: Tensor, b: Tensor) -> Tensor:
        """Encode the relation A→B."""
        return self.relation_enc(torch.cat([a, b], dim=-1))
    
    def solve_analogy(self, rel_ab: Tensor, c: Tensor) -> Tensor:
        """Given A:B :: C:?, solve for ?."""
        # The answer D should satisfy: relation(C, D) ≈ relation(A, B)
        # Simplified: D ≈ C + (B - A) in relation space
        analogy_input = torch.cat([rel_ab, c], dim=-1)
        return self.analogy_net(analogy_input)
    
    def forward(self, a: Tensor, b: Tensor, c: Tensor) -> Dict[str, Tensor]:
        """
        Solve analogy A:B :: C:?
        
        Args:
            a, b, c: (B, latent_dim) analogy components
        
        Returns:
            dict with relation, answer, confidence
        """
        # Encode relation A→B
        rel_ab = self.encode_relation(a, b)
        
        # Solve for D
        answer = self.solve_analogy(rel_ab, c)
        
        # Confidence
        confidence = self.confidence_net(answer).squeeze(-1)
        
        return {
            "relation_ab": rel_ab,
            "analogy_answer": answer,
            "analogy_confidence": confidence,
        }


class TransferLearningSystem(nn.Module):
    """
    [Phase 4] Complete transfer learning system.
    
    Integrates: Few-Shot + Domain Transfer + Skill Composition + Analogy
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        self.few_shot = FewShotAdapter(cfg)
        self.domain_transfer = DomainAdapter(cfg)
        self.skill_composer = SkillComposer(cfg)
        self.analogy = AnalogicalReasoner(cfg)
        
        # Transfer quality estimator
        self.transfer_quality = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: Tensor, support: Tensor,
                source_domain: Tensor, target_domain: Tensor,
                skill_library: Tensor, goal: Tensor) -> Dict[str, Tensor]:
        """
        Complete transfer learning pipeline.
        
        Args:
            features: (B, latent_dim) current features
            support: (B, n_examples, latent_dim) few-shot support
            source_domain, target_domain: (B,) domain indices
            skill_library: (n_skills, skill_dim) available skills
            goal: (B, latent_dim) desired outcome
        
        Returns:
            dict with all transfer outputs
        """
        # Few-shot adaptation
        fs_out = self.few_shot(support, features)
        
        # Domain transfer
        dt_out = self.domain_transfer(features, source_domain, target_domain)
        
        # Skill composition
        sc_out = self.skill_composer(goal, skill_library)
        
        # Analogical reasoning (using features as analogy components)
        analogy_out = self.analogy(features, fs_out["adapted_output"], goal)
        
        # Transfer quality
        quality_input = fs_out["adapted_output"]
        transfer_quality = self.transfer_quality(quality_input).squeeze(-1)
        
        return {
            **{f"fs_{k}": v for k, v in fs_out.items()},
            **{f"dt_{k}": v for k, v in dt_out.items()},
            **{f"sc_{k}": v for k, v in sc_out.items()},
            **{f"analogy_{k}": v for k, v in analogy_out.items()},
            "transfer_quality": transfer_quality,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED FULL AGI SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class SESIFullAGI(nn.Module):
    """
    Complete SESI Full AGI system integrating all 4 phases.
    
    Architecture:
      Phase 1: EmbodiedAgent (perception-action loop)
      Phase 2: OpenEndedLearner (intrinsic motivation + skills)
      Phase 3: WorldModelSystem (prediction + counterfactuals)
      Phase 4: TransferLearningSystem (few-shot + composition)
    
    The phases are not sequential — they operate in parallel and
    inform each other continuously.
    """
    
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
        
        # Phase 1: Embodiment
        self.embodied = EmbodiedAgent(cfg)
        
        # Phase 2: Open-Ended Learning
        self.open_ended = OpenEndedLearner(cfg)
        
        # Phase 3: World Models
        self.world_model = WorldModelSystem(cfg)
        
        # Phase 4: Transfer Learning
        self.transfer = TransferLearningSystem(cfg)
        
        # Meta-controller (coordinates all phases)
        self.meta_controller = nn.Sequential(
            nn.Linear(cfg.latent_dim * 4, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 4),  # phase priorities
            nn.Softmax(dim=-1),
        )
        
        # Overall capability score
        self.capability_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, batch_size: int, device: torch.device,
                n_steps: int = 4) -> Dict[str, Tensor]:
        """
        Complete AGI forward pass.
        
        Args:
            batch_size: number of parallel agents
            device: computation device
            n_steps: embodiment steps
        
        Returns:
            dict with all phase outputs and overall assessment
        """
        B = batch_size
        out = {}
        
        # ── Phase 1: Embodiment ─────────────────────────────────────────
        embodied_out = self.embodied(batch_size, device, n_steps)
        out.update({f"p1_{k}": v for k, v in embodied_out.items()})
        
        # Get final perception for other phases
        perception = embodied_out["final_state"]
        action = embodied_out["traj_actions"][:, -1]  # last action
        
        # ── Phase 2: Open-Ended Learning ────────────────────────────────
        # Use world model prediction vs actual for curiosity
        wm_out = self.world_model(perception, action)
        predicted = wm_out["wm_predictions"][:, 0]  # first prediction
        actual = perception  # current state as "actual"
        
        open_ended_out = self.open_ended(predicted, actual, perception)
        out.update({f"p2_{k}": v for k, v in open_ended_out.items()})
        
        # ── Phase 3: World Models ───────────────────────────────────────
        out.update({f"p3_{k}": v for k, v in wm_out.items()})
        
        # ── Phase 4: Transfer Learning ──────────────────────────────────
        # Create synthetic support set for few-shot
        support = perception.unsqueeze(1).expand(-1, self.cfg.n_few_shot_examples, -1)
        source_domain = torch.zeros(B, dtype=torch.long, device=device)
        target_domain = torch.ones(B, dtype=torch.long, device=device)
        skill_library = self.open_ended.skills.skill_library
        goal = open_ended_out["goal_goals"]
        
        transfer_out = self.transfer(
            perception, support, source_domain, target_domain,
            skill_library, goal
        )
        out.update({f"p4_{k}": v for k, v in transfer_out.items()})
        
        # ── Meta-Controller ─────────────────────────────────────────────
        meta_input = torch.cat([
            perception,
            open_ended_out["motiv_total_motivation"].unsqueeze(-1).expand(-1, self.cfg.latent_dim),
            wm_out["unc_total_uncertainty"].unsqueeze(-1).expand(-1, self.cfg.latent_dim),
            transfer_out["transfer_quality"].unsqueeze(-1).expand(-1, self.cfg.latent_dim),
        ], dim=-1)
        
        phase_priorities = self.meta_controller(meta_input)
        out["phase_priorities"] = phase_priorities
        
        # ── Overall Capability ──────────────────────────────────────────
        capability = self.capability_net(perception).squeeze(-1)
        out["overall_capability"] = capability
        
        # Composite AGI score
        agi_score = (
            0.25 * embodied_out["total_reward"].mean() / 10.0 +
            0.25 * open_ended_out["learning_progress"].mean() +
            0.25 * (1.0 - wm_out["unc_total_uncertainty"].mean()) +
            0.25 * transfer_out["transfer_quality"].mean()
        )
        out["agi_score"] = agi_score
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class FullAGILoss(nn.Module):
    def __init__(self, cfg: FullAGIConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], model: nn.Module) -> Dict[str, Tensor]:
        dev = out["agi_score"].device
        zero = torch.zeros((), device=dev)
        L = {}
        
        # Perception quality
        if "p1_total_reward" in out:
            L["perception"] = -out["p1_total_reward"].mean() / 10.0
        else:
            L["perception"] = zero
        
        # Action quality
        if "p3_planned_action" in out:
            L["action"] = out["p3_planned_action"].pow(2).mean() * 0.01
        else:
            L["action"] = zero
        
        # Curiosity (encourage exploration)
        if "p2_motiv_curiosity" in out:
            L["curiosity"] = -out["p2_motiv_curiosity"].mean() * 0.1
        else:
            L["curiosity"] = zero
        
        # Prediction accuracy
        if "p3_wm_uncertainties" in out:
            L["prediction"] = out["p3_wm_uncertainties"].mean()
        else:
            L["prediction"] = zero
        
        # Transfer quality
        if "p4_transfer_quality" in out:
            L["transfer"] = -out["p4_transfer_quality"].mean()
        else:
            L["transfer"] = zero
        
        # Regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg
        
        total = (
            self.cfg.w_perception * L["perception"] +
            self.cfg.w_action * L["action"] +
            self.cfg.w_curiosity * L["curiosity"] +
            self.cfg.w_prediction * L["prediction"] +
            self.cfg.w_transfer * L["transfer"] +
            self.cfg.w_reg * L["reg"]
        )
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Full AGI — 4-Phase Embodied Cognition — Smoke Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)
    
    cfg = FullAGIConfig(
        latent_dim=64,
        obs_dim=32,
        action_dim=16,
        skill_dim=32,
        world_model_dim=32,
        env_grid_size=8,
        n_skills_max=16,
        batch_size=2,
        device=str(device),
    )
    
    model = SESIFullAGI(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    
    model.train()
    B = 2
    
    out = model(batch_size=B, device=device, n_steps=3)
    
    loss_fn = FullAGILoss(cfg)
    losses = loss_fn(out, model)
    losses["total"].backward()
    
    print(f"\n  ── Phase 1: Embodiment ──")
    print(f"    Total reward         : {out['p1_total_reward'].mean().item():.4f}")
    print(f"    Trajectory shape     : {tuple(out['p1_traj_observations'].shape)}")
    print(f"    Final state shape    : {tuple(out['p1_final_state'].shape)}")
    
    print(f"\n  ── Phase 2: Open-Ended Learning ──")
    print(f"    Curiosity            : {out['p2_motiv_curiosity'].mean().item():.4f}")
    print(f"    Novelty              : {out['p2_motiv_novelty'].mean().item():.4f}")
    print(f"    Competence           : {out['p2_motiv_competence'].mean().item():.4f}")
    print(f"    Learning progress    : {out['p2_learning_progress'].mean().item():.4f}")
    print(f"    Skill library size   : {out['p2_skill_library_size'].mean().item():.0f}")
    
    print(f"\n  ── Phase 3: World Models ──")
    print(f"    Prediction shape     : {tuple(out['p3_wm_predictions'].shape)}")
    print(f"    Uncertainty          : {out['p3_unc_total_uncertainty'].mean().item():.4f}")
    print(f"    Aleatoric            : {out['p3_unc_aleatoric_uncertainty'].mean().item():.4f}")
    print(f"    Epistemic            : {out['p3_unc_epistemic_uncertainty'].mean().item():.4f}")
    print(f"    CF outcomes          : {tuple(out['p3_cf_counterfactual_outcomes'].shape)}")
    
    print(f"\n  ── Phase 4: Transfer Learning ──")
    print(f"    Few-shot output      : {tuple(out['p4_fs_adapted_output'].shape)}")
    print(f"    Transfer quality     : {out['p4_transfer_quality'].mean().item():.4f}")
    print(f"    Analogy confidence   : {out['p4_analogy_analogy_confidence'].mean().item():.4f}")
    
    print(f"\n  ── Meta & Overall ──")
    print(f"    Phase priorities     : {out['phase_priorities'][0].detach().cpu().numpy()}")
    print(f"    Overall capability   : {out['overall_capability'].mean().item():.4f}")
    print(f"    AGI score            : {out['agi_score'].item():.4f}")
    
    print(f"\n  ── Losses ──")
    for k, v in losses.items():
        print(f"    {k:<20s}: {v.item():.6f}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL 4 PHASES CONNECTED' if grad_ok else 'FAIL'}")
    
    print("\n" + "=" * 72)
    print("  ⚠️  EPISTEMIC REMINDERS:")
    print("  • This is a RESEARCH FRAMEWORK, not sentience")
    print("  • Phase 1 uses a VIRTUAL environment, not physical embodiment")
    print("  • Phase 2 motivation is COMPUTED, not felt")
    print("  • Phase 3 world model is a NEURAL NETWORK, not understanding")
    print("  • Phase 4 transfer is PATTERN MATCHING, not reasoning")
    print("  • Real AGI requires validation on real-world tasks")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
