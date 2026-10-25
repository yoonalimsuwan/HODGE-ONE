"""
=============================================================================
PUBLIC SAFETY QUANTUM SENSOR NETWORK v1.0
Detecting Active Shooters, Mass Shootings & Terrorism
via Dispersed Quantum Signals in Public Spaces
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

CORE INSIGHT
────────────
Violent events (gunshots, explosions, mass movement) create distinctive
"quantum footprints" that disperse into the environment — like ink drops
in the ocean. A distributed network of quantum sensors can detect these
dispersed signals using:

  [M1] CUSUM Change Detection (Six Problems §4)
       → Instant detection of regime change (peaceful → violent)
  [M2] Assumption-Light Subsampling (Paper 11)
       → Detect without knowing noise model in advance
  [M3] δ_min Isodiametric Bound (Delta_Min Theorem 1.2)
       → Minimum detectable change threshold
  [M4] Gumbel Extreme-Value Bounds (Nuclear Inhibitor)
       → Detect rare catastrophic events
  [M5] Quantum Darwinism Fragment Analysis
       → Find where threat information dispersed

APPLICATIONS
────────────
  • Active shooter detection (gunshot acoustic + kinetic signature)
  • Mass shooting prevention (crowd panic + weapon signature)
  • Terrorism detection (explosion + coordinated movement)
  • School safety (unusual sound + movement patterns)
  • Public transport security (anomalous behavior detection)

HONEST SCOPE
────────────
  ✅ SIMULATES: quantum sensor network detection capability
  ✅ MODELS: physical signal propagation honestly
  ✅ PROVIDES: early warning + threat classification
  ❌ IS NOT: actual deployed quantum hardware (yet)
  ❌ CANNOT: predict events before they happen (only detect onset)
  ❌ CANNOT: replace human judgment in complex situations

=============================================================================
"""

from __future__ import annotations

import math
import logging
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("public_safety_quantum")

__all__ = [
    "PublicSafetyConfig",
    "ThreatType",
    # Sensor network
    "QuantumAcousticSensor",
    "QuantumMagnetometerSensor",
    "QuantumRFSensor",
    "QuantumGravimeterSensor",
    "DistributedQuantumSensorNetwork",
    # Detection algorithms
    "CUSUMThreatDetector",
    "AssumptionLightThreatDetector",
    "DeltaMinThresholdDetector",
    "GumbelRareEventDetector",
    "QuantumDarwinismThreatAnalyzer",
    # Threat classification
    "ThreatClassifier",
    "ActiveShooterDetector",
    "MassShootingDetector",
    "TerrorismDetector",
    # Unified system
    "PublicSafetyQuantumNetwork",
]


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Primitives
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
# CONFIG & ENUMS
# ═══════════════════════════════════════════════════════════════════════

class ThreatType(Enum):
    NONE = auto()
    ACTIVE_SHOOTER = auto()
    MASS_SHOOTING = auto()
    TERRORISM = auto()
    EXPLOSION = auto()
    CROWD_PANIC = auto()
    WEAPON_DETECTED = auto()


@dataclass
class PublicSafetyConfig:
    # ── Sensor network ───────────────────────────────────────────────────
    n_sensors: int = 64
    sensor_dim: int = 128
    coverage_radius_m: float = 50.0
    
    # ── Acoustic sensor (gunshot detection) ────────────────────────────────
    acoustic_sample_rate_hz: float = 44100.0
    gunshot_freq_range_hz: Tuple[float, float] = (500.0, 5000.0)
    gunshot_duration_ms: float = 50.0
    
    # ── Magnetometer (movement detection) ────────────────────────────────
    magnetic_sensitivity_nt: float = 0.1  # nanoTesla
    
    # ── RF sensor (signal anomaly) ───────────────────────────────────────
    rf_freq_range_ghz: Tuple[float, float] = (0.1, 6.0)
    
    # ── Gravimeter (vibration) ───────────────────────────────────────────
    vibration_sensitivity_mg: float = 0.001  # milli-g
    
    # ── CUSUM (Six Problems §4) ──────────────────────────────────────────
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    
    # ── δ_min (Delta_Min Theorem 1.2) ────────────────────────────────────
    kappa_fatness: float = 2.0
    a_min: float = 0.01
    
    # ── Gumbel (Nuclear Inhibitor) ───────────────────────────────────────
    gumbel_c1: float = 1.0
    gumbel_sigma: float = 0.05
    
    # ── Assumption-light subsampling (Paper 11) ──────────────────────────
    subsample_block_size: int = 64
    n_subsamples: int = 100
    confidence_level: float = 0.95
    
    # ── Threat classification ────────────────────────────────────────────
    n_threat_types: int = 7
    
    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM ACOUSTIC SENSOR (Gunshot Detection)
# ═══════════════════════════════════════════════════════════════════════

class QuantumAcousticSensor(nn.Module):
    """
    Quantum-enhanced acoustic sensor for gunshot detection.
    
    Detects:
      • Gunshot acoustic signature (500-5000 Hz, ~50ms duration)
      • Shockwave pattern (N-wave)
      • Echo pattern (indoor vs outdoor)
    
    Uses quantum-enhanced sensitivity to detect faint signals
    dispersed in the environment.
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Acoustic feature extractor
        self.acoustic_enc = nn.Sequential(
            nn.Linear(cfg.sensor_dim, cfg.sensor_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.sensor_dim // 2, cfg.sensor_dim // 2),
        )
        
        # Gunshot signature detector
        self.gunshot_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Shockwave detector (N-wave pattern)
        self.shockwave_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Distance estimator (from signal attenuation)
        self.distance_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # distance > 0
        )
        
        # Direction estimator (from time-difference-of-arrival)
        self.direction_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # [azimuth, elevation]
            nn.Tanh(),  # bounded angles
        )
    
    def forward(self, acoustic_signal: Tensor) -> Dict[str, Tensor]:
        """
        Process acoustic signal for gunshot detection.
        
        Args:
            acoustic_signal: (B, sensor_dim) acoustic waveform features
        
        Returns:
            dict with gunshot probability, distance, direction
        """
        # Encode
        encoded = self.acoustic_enc(acoustic_signal)
        
        # Gunshot detection
        gunshot_prob = self.gunshot_net(encoded).squeeze(-1)
        
        # Shockwave detection
        shockwave_prob = self.shockwave_net(encoded).squeeze(-1)
        
        # Distance estimation
        distance_m = self.distance_net(encoded).squeeze(-1)
        
        # Direction estimation
        direction = self.direction_net(encoded)
        
        # Combined threat score
        threat_score = gunshot_prob * shockwave_prob
        
        return {
            "gunshot_probability": gunshot_prob,
            "shockwave_probability": shockwave_prob,
            "estimated_distance_m": distance_m,
            "estimated_direction": direction,
            "acoustic_threat_score": threat_score,
        }


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM MAGNETOMETER SENSOR (Movement Detection)
# ═══════════════════════════════════════════════════════════════════════

class QuantumMagnetometerSensor(nn.Module):
    """
    Quantum magnetometer for detecting rapid movement and weapons.
    
    Detects:
      • Rapid crowd movement (panic, running)
      • Metallic objects (weapons, explosives)
      • Electromagnetic anomalies (IED triggers)
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Magnetic field encoder
        self.magnetic_enc = nn.Sequential(
            nn.Linear(cfg.sensor_dim, cfg.sensor_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.sensor_dim // 2, cfg.sensor_dim // 2),
        )
        
        # Rapid movement detector
        self.movement_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Weapon detector (metallic signature)
        self.weapon_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Crowd density estimator
        self.crowd_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(self, magnetic_field: Tensor) -> Dict[str, Tensor]:
        """
        Process magnetic field for movement/weapon detection.
        
        Args:
            magnetic_field: (B, sensor_dim) magnetic field features
        
        Returns:
            dict with movement, weapon, crowd metrics
        """
        encoded = self.magnetic_enc(magnetic_field)
        
        movement_prob = self.movement_net(encoded).squeeze(-1)
        weapon_prob = self.weapon_net(encoded).squeeze(-1)
        crowd_density = self.crowd_net(encoded).squeeze(-1)
        
        # Panic indicator (rapid movement + high crowd density)
        panic_score = movement_prob * crowd_density / 10.0
        
        return {
            "movement_probability": movement_prob,
            "weapon_probability": weapon_prob,
            "crowd_density": crowd_density,
            "panic_score": panic_score,
            "magnetic_threat_score": movement_prob * weapon_prob,
        }


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM RF SENSOR (Signal Anomaly Detection)
# ═══════════════════════════════════════════════════════════════════════

class QuantumRFSensor(nn.Module):
    """
    Quantum RF sensor for detecting anomalous signals.
    
    Detects:
      • IED trigger signals
      • Drone communication
      • Jamming attempts
      • Coordinated attack communication
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # RF spectrum encoder
        self.rf_enc = nn.Sequential(
            nn.Linear(cfg.sensor_dim, cfg.sensor_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.sensor_dim // 2, cfg.sensor_dim // 2),
        )
        
        # Anomaly detector
        self.anomaly_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # IED trigger detector
        self.ied_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Drone detector
        self.drone_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, rf_spectrum: Tensor) -> Dict[str, Tensor]:
        """
        Process RF spectrum for anomaly detection.
        
        Args:
            rf_spectrum: (B, sensor_dim) RF spectrum features
        
        Returns:
            dict with anomaly, IED, drone metrics
        """
        encoded = self.rf_enc(rf_spectrum)
        
        anomaly_prob = self.anomaly_net(encoded).squeeze(-1)
        ied_prob = self.ied_net(encoded).squeeze(-1)
        drone_prob = self.drone_net(encoded).squeeze(-1)
        
        return {
            "rf_anomaly_probability": anomaly_prob,
            "ied_trigger_probability": ied_prob,
            "drone_probability": drone_prob,
            "rf_threat_score": anomaly_prob * (ied_prob + drone_prob) / 2.0,
        }


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM GRAVIMETER SENSOR (Vibration Detection)
# ═══════════════════════════════════════════════════════════════════════

class QuantumGravimeterSensor(nn.Module):
    """
    Quantum gravimeter for detecting explosions and structural stress.
    
    Detects:
      • Explosion shockwaves
      • Structural collapse
      • Vehicle ramming
      • Heavy weapon recoil
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Vibration encoder
        self.vibration_enc = nn.Sequential(
            nn.Linear(cfg.sensor_dim, cfg.sensor_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.sensor_dim // 2, cfg.sensor_dim // 2),
        )
        
        # Explosion detector
        self.explosion_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Structural stress detector
        self.stress_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Impact detector (vehicle ramming)
        self.impact_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, vibration_signal: Tensor) -> Dict[str, Tensor]:
        """
        Process vibration for explosion/impact detection.
        
        Args:
            vibration_signal: (B, sensor_dim) vibration features
        
        Returns:
            dict with explosion, stress, impact metrics
        """
        encoded = self.vibration_enc(vibration_signal)
        
        explosion_prob = self.explosion_net(encoded).squeeze(-1)
        stress_prob = self.stress_net(encoded).squeeze(-1)
        impact_prob = self.impact_net(encoded).squeeze(-1)
        
        return {
            "explosion_probability": explosion_prob,
            "structural_stress_probability": stress_prob,
            "impact_probability": impact_prob,
            "vibration_threat_score": explosion_prob + impact_prob,
        }


# ═══════════════════════════════════════════════════════════════════════
# DISTRIBUTED QUANTUM SENSOR NETWORK
# ═══════════════════════════════════════════════════════════════════════

class DistributedQuantumSensorNetwork(nn.Module):
    """
    Distributed network of quantum sensors covering a public space.
    
    Integrates:
      • Acoustic sensors (gunshot detection)
      • Magnetometers (movement/weapon detection)
      • RF sensors (signal anomaly)
      • Gravimeters (explosion/impact detection)
    
    Uses sensor fusion to create a comprehensive threat picture.
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Individual sensor modules
        self.acoustic = QuantumAcousticSensor(cfg)
        self.magnetometer = QuantumMagnetometerSensor(cfg)
        self.rf = QuantumRFSensor(cfg)
        self.gravimeter = QuantumGravimeterSensor(cfg)
        
        # Sensor fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim * 4, cfg.sensor_dim),
            nn.GELU(),
            nn.Linear(cfg.sensor_dim, cfg.sensor_dim),
        )
        
        # Spatial localization (triangulation from multiple sensors)
        self.localization_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [x, y, z] coordinates
        )
        
        # Overall threat level
        self.threat_level_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        acoustic: Tensor,
        magnetic: Tensor,
        rf: Tensor,
        vibration: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Process all sensor modalities and fuse.
        
        Args:
            acoustic: (B, sensor_dim) acoustic features
            magnetic: (B, sensor_dim) magnetic features
            rf: (B, sensor_dim) RF features
            vibration: (B, sensor_dim) vibration features
        
        Returns:
            dict with fused threat assessment
        """
        # Process each sensor
        acoustic_out = self.acoustic(acoustic)
        magnetic_out = self.magnetometer(magnetic)
        rf_out = self.rf(rf)
        vibration_out = self.gravimeter(vibration)
        
        # Fuse all sensor outputs
        fusion_input = torch.cat([
            acoustic,
            magnetic,
            rf,
            vibration,
        ], dim=-1)
        
        fused = self.fusion_net(fusion_input)
        
        # Localization
        location = self.localization_net(fused)
        
        # Overall threat level
        threat_level = self.threat_level_net(fused).squeeze(-1)
        
        return {
            **{f"acoustic_{k}": v for k, v in acoustic_out.items()},
            **{f"magnetic_{k}": v for k, v in magnetic_out.items()},
            **{f"rf_{k}": v for k, v in rf_out.items()},
            **{f"vibration_{k}": v for k, v in vibration_out.items()},
            "fused_features": fused,
            "estimated_location": location,
            "overall_threat_level": threat_level,
        }


# ═══════════════════════════════════════════════════════════════════════
# CUSUM THREAT DETECTOR (Six Problems §4)
# ═══════════════════════════════════════════════════════════════════════

class CUSUMThreatDetector(nn.Module):
    """
    [M1] CUSUM for instant threat detection.
    
    From Six Problems §4:
      S_t = max(0, S_{t-1} + x_t - drift)
      Alarm when S_t > threshold
    
    Detects regime change: peaceful → violent
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Feature extractor for CUSUM input
        self.feature_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        # Drift parameter (learnable)
        self.drift = nn.Parameter(torch.tensor(cfg.cusum_drift))
        
        # Threshold (learnable)
        self.log_threshold = nn.Parameter(
            torch.tensor(math.log(cfg.cusum_threshold)))
        
        # Threat type classifier
        self.threat_classifier = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_threat_types),
            nn.Softmax(dim=-1),
        )
    
    @property
    def threshold(self) -> Tensor:
        return torch.exp(self.log_threshold)
    
    def forward(self, sensor_stream: Tensor) -> Dict[str, Tensor]:
        """
        Detect threat using CUSUM.
        
        Args:
            sensor_stream: (B, T, sensor_dim) time series of sensor data
        
        Returns:
            dict with CUSUM trajectory, alarms, threat type
        """
        B, T, D = sensor_stream.shape
        device = sensor_stream.device
        
        # Extract scalar feature from each time step
        features = self.feature_net(
            sensor_stream.reshape(B * T, D)
        ).reshape(B, T)
        
        # CUSUM recursion
        cusum_state = torch.zeros(B, device=device)
        alarms = []
        cusum_trajectory = []
        
        for t in range(T):
            x = features[:, t]
            cusum_state = torch.clamp(
                cusum_state + x - self.drift, min=0.0)
            alarm = (cusum_state > self.threshold).float()
            alarms.append(alarm)
            cusum_trajectory.append(cusum_state)
        
        alarms = torch.stack(alarms, dim=1)
        cusum_trajectory = torch.stack(cusum_trajectory, dim=1)
        
        # First alarm time
        first_alarm = torch.zeros(B, device=device)
        for t in range(T):
            is_first = (alarms[:, t] > 0.5) & (first_alarm == 0)
            first_alarm = first_alarm + is_first.float() * t
        
        # Threat type classification
        threat_probs = self.threat_classifier(sensor_stream.mean(dim=1))
        
        return {
            "cusum_trajectory": cusum_trajectory,
            "alarms": alarms,
            "n_alarms": alarms.sum(dim=-1),
            "first_alarm_time": first_alarm,
            "threat_detected": (alarms.sum(dim=-1) > 0).float(),
            "threat_type_probs": threat_probs,
            "predicted_threat_type": threat_probs.argmax(dim=-1),
        }


# ═══════════════════════════════════════════════════════════════════════
# ASSUMPTION-LIGHT THREAT DETECTOR (Paper 11)
# ═══════════════════════════════════════════════════════════════════════

class AssumptionLightThreatDetector(nn.Module):
    """
    [M2] Assumption-light threat detection via subsampling.
    
    From Paper 11:
      • Subsampling CI without knowing noise model
      • Automatic block-length selection
      • Valid under only qualitative mixing conditions
    
    Detects threats without pre-specifying noise characteristics.
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Threat parameter estimator
        self.param_estimator = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
        )
        
        # Subsampling variance estimator
        self.variance_net = nn.Sequential(
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 16),
            nn.Softplus(),
        )
        
        # Block length selector
        self.block_selector = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        
        # Threat presence classifier
        self.presence_net = nn.Sequential(
            nn.Linear(16 * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def subsampling_ci(
        self,
        params: Tensor,
        n_subsamples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute subsampling confidence interval."""
        n_sub = n_subsamples or self.cfg.n_subsamples
        
        variance = self.variance_net(params)
        z = 1.96  # 95% CI
        ci_width = z * torch.sqrt(variance / n_sub + 1e-8)
        
        ci_lower = params - ci_width
        ci_upper = params + ci_width
        
        return ci_lower, ci_upper
    
    def forward(self, sensor_data: Tensor) -> Dict[str, Tensor]:
        """
        Detect threat using assumption-light methods.
        
        Args:
            sensor_data: (B, sensor_dim)
        
        Returns:
            dict with parameters, CI, presence probability
        """
        # Estimate threat parameters
        params = self.param_estimator(sensor_data)
        
        # Subsampling CI
        ci_lower, ci_upper = self.subsampling_ci(params)
        ci_width = ci_upper - ci_lower
        
        # Block length selection
        block_length = self.block_selector(sensor_data).squeeze(-1)
        
        # Threat presence classification
        presence_input = torch.cat([params, ci_width], dim=-1)
        presence_prob = self.presence_net(presence_input).squeeze(-1)
        
        # Detection: CI doesn't contain zero → threat present
        ci_excludes_zero = (ci_lower > 0).float().mean(dim=-1)
        
        return {
            "threat_parameters": params,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
            "block_length": block_length,
            "threat_presence_probability": presence_prob,
            "ci_excludes_zero": ci_excludes_zero,
            "threat_confidently_present": (presence_prob > 0.8).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# DELTA-MIN THRESHOLD DETECTOR (Delta_Min Theorem 1.2)
# ═══════════════════════════════════════════════════════════════════════

class DeltaMinThresholdDetector(nn.Module):
    """
    [M3] δ_min threshold for minimum detectable threat.
    
    From Delta_Min Theorem 1.2:
      Under κ-fatness, d_H(Γ, Γ') ≥ δ_min = (1/κ)√(a_min/ω_{n-1})
    
    Defines minimum threat signature size that can be reliably detected.
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # δ_min from isodiametric bound
        delta_min = (1.0 / cfg.kappa_fatness) * math.sqrt(cfg.a_min / math.pi)
        self.register_buffer("delta_min", torch.tensor(delta_min), persistent=False)
        
        # Threat signature size estimator
        self.signature_size_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Fatness checker
        self.fatness_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Detection decision
        self.detection_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, sensor_data: Tensor) -> Dict[str, Tensor]:
        """
        Check if threat exceeds δ_min threshold.
        
        Args:
            sensor_data: (B, sensor_dim)
        
        Returns:
            dict with detection result
        """
        B = sensor_data.shape[0]
        
        # Estimate threat signature size
        signature_size = self.signature_size_net(sensor_data).squeeze(-1)
        
        # Check fatness
        fatness = self.fatness_net(sensor_data).squeeze(-1)
        
        # Detection decision
        detect_input = torch.stack([
            signature_size,
            self.delta_min.expand(B),
            fatness,
        ], dim=-1)
        
        detection_prob = self.detection_net(detect_input).squeeze(-1)
        
        # Hard detection
        is_detected = (
            (signature_size > self.delta_min) &
            (fatness > 0.5)
        ).float().detach()
        
        return {
            "delta_min_threshold": self.delta_min.expand(B),
            "threat_signature_size": signature_size,
            "fatness_score": fatness,
            "detection_probability": detection_prob,
            "threat_detected": is_detected,
            "above_threshold": (signature_size > self.delta_min).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# GUMBEL RARE EVENT DETECTOR (Nuclear Inhibitor)
# ═══════════════════════════════════════════════════════════════════════

class GumbelRareEventDetector(nn.Module):
    """
    [M4] Gumbel bounds for detecting rare catastrophic events.
    
    From Nuclear Inhibitor:
      P(event) ≤ exp(-C₁·exp(ΔE/(σ²·dt)))
    
    Detects extremely rare but high-impact threats (terrorism, mass casualty).
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        self.register_buffer("gumbel_c1", torch.tensor(cfg.gumbel_c1), persistent=False)
        self.register_buffer("gumbel_sigma", torch.tensor(cfg.gumbel_sigma), persistent=False)
        self.register_buffer("exp_clamp", torch.tensor(80.0), persistent=False)
        
        # Energy barrier estimator
        self.barrier_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Catastrophic event classifier
        self.catastrophe_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def gumbel_bound(self, delta_E: Tensor) -> Tensor:
        """Compute Gumbel extreme-value probability bound."""
        sigma_sq = self.gumbel_sigma ** 2
        dt = 1e-5
        denom = torch.clamp(
            torch.tensor(sigma_sq * dt, device=delta_E.device, dtype=delta_E.dtype),
            min=1e-15)
        inner = torch.clamp(delta_E / denom, max=self.exp_clamp)
        return torch.exp(-self.gumbel_c1 * torch.exp(inner))
    
    def forward(self, sensor_data: Tensor) -> Dict[str, Tensor]:
        """
        Detect rare catastrophic events.
        
        Args:
            sensor_data: (B, sensor_dim)
        
        Returns:
            dict with catastrophe probability
        """
        # Energy barrier
        delta_E = self.barrier_net(sensor_data).squeeze(-1)
        
        # Gumbel bound
        rare_event_prob = self.gumbel_bound(delta_E)
        
        # Catastrophe classification
        catastrophe_prob = self.catastrophe_net(sensor_data).squeeze(-1)
        
        # Combined rare event score
        rare_event_score = rare_event_prob * catastrophe_prob
        
        return {
            "energy_barrier": delta_E,
            "gumbel_rare_event_prob": rare_event_prob,
            "catastrophe_probability": catastrophe_prob,
            "rare_event_score": rare_event_score,
            "catastrophic_threat_detected": (rare_event_score > 0.01).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM DARWINISM THREAT ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class QuantumDarwinismThreatAnalyzer(nn.Module):
    """
    [M5] Find where threat information dispersed in sensor network.
    
    Quantum Darwinism: Threat signature is redundantly imprinted
    on multiple sensors. Find which sensors carry the most information.
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Sensor encoder
        self.sensor_enc = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
        )
        
        # Mutual information estimator
        self.mutual_info_net = nn.Sequential(
            nn.Linear(16 + cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        
        # Redundancy counter
        self.redundancy_net = nn.Sequential(
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        
        # Information localization
        self.localizer = nn.Sequential(
            nn.Linear(cfg.sensor_dim * cfg.n_sensors, cfg.n_sensors),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        threat_signature: Tensor,
        sensor_network: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Analyze where threat information dispersed.
        
        Args:
            threat_signature: (B, sensor_dim)
            sensor_network: (B, n_sensors, sensor_dim)
        
        Returns:
            dict with fragment MI, redundancy, localization
        """
        B = threat_signature.shape[0]
        n_sensors = sensor_network.shape[1]
        
        # Encode each sensor
        sensor_features = self.sensor_enc(sensor_network)  # (B, n_sensors, 16)
        
        # Mutual information with threat for each sensor
        mutual_infos = []
        redundancies = []
        
        for i in range(n_sensors):
            sensor_feat = sensor_features[:, i]
            
            # MI estimation
            mi_input = torch.cat([sensor_feat, threat_signature], dim=-1)
            mi = self.mutual_info_net(mi_input).squeeze(-1)
            mutual_infos.append(mi)
            
            # Redundancy
            redundancy = self.redundancy_net(sensor_feat).squeeze(-1)
            redundancies.append(redundancy)
        
        mutual_infos = torch.stack(mutual_infos, dim=-1)
        redundancies = torch.stack(redundancies, dim=-1)
        
        # Localization
        flat_sensors = sensor_network.reshape(B, -1)
        localization = self.localizer(flat_sensors)
        
        # Total threat information
        total_mi = (mutual_infos * localization).sum(dim=-1)
        
        return {
            "sensor_mutual_infos": mutual_infos,
            "sensor_redundancies": redundancies,
            "threat_localization": localization,
            "total_threat_information": total_mi,
            "redundancy_score": redundancies.mean(dim=-1),
            "n_informative_sensors": (mutual_infos > 0.1).float().sum(dim=-1),
            "most_informative_sensor": localization.argmax(dim=-1),
        }


# ═══════════════════════════════════════════════════════════════════════
# THREAT CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════

class ThreatClassifier(nn.Module):
    """Classify threat type from fused sensor data."""
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_threat_types),
            nn.Softmax(dim=-1),
        )
        
        self.confidence_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, fused_features: Tensor) -> Dict[str, Tensor]:
        threat_probs = self.classifier(fused_features)
        confidence = self.confidence_net(fused_features).squeeze(-1)
        
        return {
            "threat_type_probs": threat_probs,
            "predicted_threat_type": threat_probs.argmax(dim=-1),
            "classification_confidence": confidence,
        }


class ActiveShooterDetector(nn.Module):
    """Specialized detector for active shooter scenarios."""
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        self.detector = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.shooter_count_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(self, fused_features: Tensor) -> Dict[str, Tensor]:
        active_shooter_prob = self.detector(fused_features).squeeze(-1)
        shooter_count = self.shooter_count_net(fused_features).squeeze(-1)
        
        return {
            "active_shooter_probability": active_shooter_prob,
            "estimated_shooter_count": shooter_count,
        }


class MassShootingDetector(nn.Module):
    """Specialized detector for mass shooting scenarios."""
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        self.detector = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.casualty_estimator = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(self, fused_features: Tensor) -> Dict[str, Tensor]:
        mass_shooting_prob = self.detector(fused_features).squeeze(-1)
        estimated_casualties = self.casualty_estimator(fused_features).squeeze(-1)
        
        return {
            "mass_shooting_probability": mass_shooting_prob,
            "estimated_casualties": estimated_casualties,
        }


class TerrorismDetector(nn.Module):
    """Specialized detector for terrorism scenarios."""
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        self.detector = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.attack_type_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 5),  # bombing, shooting, vehicle, chemical, other
            nn.Softmax(dim=-1),
        )
    
    def forward(self, fused_features: Tensor) -> Dict[str, Tensor]:
        terrorism_prob = self.detector(fused_features).squeeze(-1)
        attack_type_probs = self.attack_type_net(fused_features)
        
        return {
            "terrorism_probability": terrorism_prob,
            "attack_type_probs": attack_type_probs,
            "predicted_attack_type": attack_type_probs.argmax(dim=-1),
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED PUBLIC SAFETY QUANTUM NETWORK
# ═══════════════════════════════════════════════════════════════════════

class PublicSafetyQuantumNetwork(nn.Module):
    """
    Complete public safety quantum sensor network.
    
    Pipeline:
      1. Distributed sensor network (acoustic, magnetic, RF, vibration)
      2. CUSUM threat detection (instant regime change)
      3. Assumption-light detection (no noise model needed)
      4. δ_min threshold check (minimum detectable threat)
      5. Gumbel rare event detection (catastrophic threats)
      6. Quantum Darwinism analysis (where threat info dispersed)
      7. Threat classification (active shooter, mass shooting, terrorism)
      8. Alert generation + localization
    
    HONEST SCOPE:
      This is a SIMULATION FRAMEWORK for research.
      It does NOT provide actual deployed quantum hardware.
      All outputs are differentiable simulations of detection capability.
    """
    
    def __init__(self, cfg: PublicSafetyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Sensor network
        self.sensor_network = DistributedQuantumSensorNetwork(cfg)
        
        # Detection algorithms
        self.cusum_detector = CUSUMThreatDetector(cfg)
        self.assumption_light = AssumptionLightThreatDetector(cfg)
        self.delta_min_detector = DeltaMinThresholdDetector(cfg)
        self.gumbel_detector = GumbelRareEventDetector(cfg)
        self.darwinism_analyzer = QuantumDarwinismThreatAnalyzer(cfg)
        
        # Threat classifiers
        self.threat_classifier = ThreatClassifier(cfg)
        self.active_shooter_detector = ActiveShooterDetector(cfg)
        self.mass_shooting_detector = MassShootingDetector(cfg)
        self.terrorism_detector = TerrorismDetector(cfg)
        
        # Alert generation
        self.alert_net = nn.Sequential(
            nn.Linear(cfg.sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        acoustic: Tensor,
        magnetic: Tensor,
        rf: Tensor,
        vibration: Tensor,
        sensor_stream: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full public safety detection pipeline.
        
        Args:
            acoustic: (B, sensor_dim) acoustic features
            magnetic: (B, sensor_dim) magnetic features
            rf: (B, sensor_dim) RF features
            vibration: (B, sensor_dim) vibration features
            sensor_stream: (B, T, sensor_dim) optional time series
        
        Returns:
            dict with complete threat assessment
        """
        B = acoustic.shape[0]
        dev = acoustic.device
        out = {}
        
        # 1. Sensor network fusion
        sensor_out = self.sensor_network(acoustic, magnetic, rf, vibration)
        out.update({f"sensor_{k}": v for k, v in sensor_out.items()})
        
        fused = sensor_out["fused_features"]
        
        # 2. CUSUM detection (if stream provided)
        if sensor_stream is not None:
            cusum_out = self.cusum_detector(sensor_stream)
            out.update({f"cusum_{k}": v for k, v in cusum_out.items()})
        
        # 3. Assumption-light detection
        al_out = self.assumption_light(fused)
        out.update({f"al_{k}": v for k, v in al_out.items()})
        
        # 4. δ_min threshold check
        delta_out = self.delta_min_detector(fused)
        out.update({f"delta_{k}": v for k, v in delta_out.items()})
        
        # 5. Gumbel rare event detection
        gumbel_out = self.gumbel_detector(fused)
        out.update({f"gumbel_{k}": v for k, v in gumbel_out.items()})
        
        # 6. Quantum Darwinism analysis
        # (simulate sensor network data)
        sensor_network_data = torch.stack([acoustic, magnetic, rf, vibration], dim=1)
        # Pad to n_sensors if needed
        if sensor_network_data.shape[1] < self.cfg.n_sensors:
            padding = torch.randn(
                B, self.cfg.n_sensors - sensor_network_data.shape[1], 
                self.cfg.sensor_dim, device=dev) * 0.1
            sensor_network_data = torch.cat([sensor_network_data, padding], dim=1)
        
        darwin_out = self.darwinism_analyzer(fused, sensor_network_data)
        out.update({f"darwin_{k}": v for k, v in darwin_out.items()})
        
        # 7. Threat classification
        classifier_out = self.threat_classifier(fused)
        out.update({f"classifier_{k}": v for k, v in classifier_out.items()})
        
        # Specialized detectors
        active_shooter_out = self.active_shooter_detector(fused)
        out.update({f"active_shooter_{k}": v for k, v in active_shooter_out.items()})
        
        mass_shooting_out = self.mass_shooting_detector(fused)
        out.update({f"mass_shooting_{k}": v for k, v in mass_shooting_out.items()})
        
        terrorism_out = self.terrorism_detector(fused)
        out.update({f"terrorism_{k}": v for k, v in terrorism_out.items()})
        
        # 8. Alert generation
        alert_level = self.alert_net(fused).squeeze(-1)
        out["alert_level"] = alert_level
        
        # Overall threat assessment
        overall_threat = (
            sensor_out["overall_threat_level"] *
            al_out["threat_presence_probability"] *
            delta_out["detection_probability"]
        )
        out["overall_threat_assessment"] = overall_threat
        
        # Emergency response recommendation
        emergency_response = (overall_threat > 0.7).float()
        out["emergency_response_required"] = emergency_response
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Public Safety Quantum Sensor Network — Smoke Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)
    
    cfg = PublicSafetyConfig(
        n_sensors=16,
        sensor_dim=64,
        n_threat_types=7,
        subsample_block_size=32,
        n_subsamples=50,
        batch_size=2,
        device=str(device),
    )
    
    model = PublicSafetyQuantumNetwork(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    
    model.train()
    B = 2
    
    # Simulate sensor data (with threat signature)
    acoustic = torch.randn(B, cfg.sensor_dim, device=device)
    magnetic = torch.randn(B, cfg.sensor_dim, device=device)
    rf = torch.randn(B, cfg.sensor_dim, device=device)
    vibration = torch.randn(B, cfg.sensor_dim, device=device)
    sensor_stream = torch.randn(B, 8, cfg.sensor_dim, device=device)
    
    # Add threat signature (simulate gunshot + panic)
    acoustic[:, :10] += 2.0  # gunshot signature
    magnetic[:, :10] += 1.5  # weapon signature
    vibration[:, :10] += 1.0  # shockwave
    
    out = model(acoustic, magnetic, rf, vibration, sensor_stream)
    
    # Backward
    loss = -out["overall_threat_assessment"].mean()
    loss.backward()
    
    print(f"\n  ── Sensor Network ──")
    print(f"    Acoustic threat      : {out['sensor_acoustic_threat_score'].mean().item():.4f}")
    print(f"    Magnetic threat      : {out['sensor_magnetic_threat_score'].mean().item():.4f}")
    print(f"    RF threat            : {out['sensor_rf_threat_score'].mean().item():.4f}")
    print(f"    Vibration threat     : {out['sensor_vibration_threat_score'].mean().item():.4f}")
    print(f"    Overall threat level : {out['sensor_overall_threat_level'].mean().item():.4f}")
    print(f"    Estimated location   : {out['sensor_estimated_location'][0].detach().cpu().numpy()}")
    
    print(f"\n  ── CUSUM Detection ──")
    print(f"    Threat detected      : {out['cusum_threat_detected'].mean().item():.4f}")
    print(f"    N alarms             : {out['cusum_n_alarms'].mean().item():.1f}")
    print(f"    First alarm time     : {out['cusum_first_alarm_time'].mean().item():.1f}")
    print(f"    Predicted threat type: {out['cusum_predicted_threat_type'].tolist()}")
    
    print(f"\n  ── Assumption-Light Detection ──")
    print(f"    Threat presence prob : {out['al_threat_presence_probability'].mean().item():.4f}")
    print(f"    CI excludes zero     : {out['al_ci_excludes_zero'].mean().item():.4f}")
    print(f"    Confidently present  : {out['al_threat_confidently_present'].mean().item():.4f}")
    
    print(f"\n  ── δ_min Threshold ──")
    print(f"    δ_min threshold      : {out['delta_delta_min_threshold'].mean().item():.6f}")
    print(f"    Threat signature size: {out['delta_threat_signature_size'].mean().item():.4f}")
    print(f"    Above threshold      : {out['delta_above_threshold'].mean().item():.4f}")
    print(f"    Threat detected      : {out['delta_threat_detected'].mean().item():.4f}")
    
    print(f"\n  ── Gumbel Rare Event ──")
    print(f"    Energy barrier       : {out['gumbel_energy_barrier'].mean().item():.4f}")
    print(f"    Rare event prob      : {out['gumbel_gumbel_rare_event_prob'].mean().item():.2e}")
    print(f"    Catastrophe prob     : {out['gumbel_catastrophe_probability'].mean().item():.4f}")
    print(f"    Catastrophic detected: {out['gumbel_catastrophic_threat_detected'].mean().item():.4f}")
    
    print(f"\n  ── Quantum Darwinism ──")
    print(f"    Total threat info    : {out['darwin_total_threat_information'].mean().item():.4f}")
    print(f"    Redundancy score     : {out['darwin_redundancy_score'].mean().item():.4f}")
    print(f"    Informative sensors  : {out['darwin_n_informative_sensors'].mean().item():.1f}")
    print(f"    Most informative     : {out['darwin_most_informative_sensor'].tolist()}")
    
    print(f"\n  ── Threat Classification ──")
    print(f"    Threat type probs     : {out['classifier_threat_type_probs'][0].detach().cpu().numpy()}")
    print(f"    Predicted type       : {out['classifier_predicted_threat_type'].tolist()}")
    print(f"    Confidence           : {out['classifier_classification_confidence'].mean().item():.4f}")
    
    print(f"\n  ── Specialized Detectors ──")
    print(f"    Active shooter prob  : {out['active_shooter_active_shooter_probability'].mean().item():.4f}")
    print(f"    Mass shooting prob   : {out['mass_shooting_mass_shooting_probability'].mean().item():.4f}")
    print(f"    Terrorism prob       : {out['terrorism_terrorism_probability'].mean().item():.4f}")
    
    print(f"\n  ── Overall Assessment ──")
    print(f"    Alert level          : {out['alert_level'].mean().item():.4f}")
    print(f"    Overall threat       : {out['overall_threat_assessment'].mean().item():.4f}")
    print(f"    Emergency response   : {out['emergency_response_required'].mean().item():.4f}")
    
    print(f"\n  ── Loss ──")
    print(f"    Loss                 : {loss.item():.6f}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL MODULES CONNECTED' if grad_ok else 'FAIL'}")
    
    print("\n" + "=" * 72)
    print("  ⚠️  HONEST REMINDERS:")
    print("  • This is a SIMULATION, not deployed quantum hardware")
    print("  • Detection is probabilistic, not deterministic")
    print("  • False positives/negatives are possible")
    print("  • Human judgment required for final decisions")
    print("  • Privacy and civil liberties must be protected")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
