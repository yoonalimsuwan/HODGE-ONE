"""
=============================================================================
TRANSCRANIAL NEURAL SIGNAL LIMIT ANALYZER v1.0
Honest Physics Simulation of Non-Invasive Brain Reading Limits
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

CRITICAL HONEST SCOPE
─────────────────────
This module SIMULATES THE PHYSICAL LIMITS of transcranial (non-invasive)
neural signal reading. It demonstrates WHY "shooting enough quantum
signal into the brain" CANNOT replace implanted devices.

The physics:
  [P1] Tissue scattering: μ_s ≈ 100-200 cm⁻¹ in brain tissue
  [P2] Absorption: μ_a ≈ 0.1-1 cm⁻¹ (wavelength-dependent)
  [P3] Penetration depth: ~1-2 mm practical limit for optical
  [P4] SAR limit: < 4 W/kg (IEEE safety standard)
  [P5] Thermal damage threshold: ~1°C temperature rise max
  [P6] Inverse problem: scattering destroys spatial information

This module computes HONEST upper bounds on what's achievable,
showing the fundamental gap between:
  • Implanted devices (near neurons, high SNR, single-cell)
  • Transcranial methods (through skull, low SNR, population-level)

=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("transcranial_limits")

__all__ = [
    "TranscranialConfig",
    "TissueOpticalProperties",
    "TranscranialSignalModel",
    "SafePowerLimitCalculator",
    "ImplantVsTranscranialComparison",
    "TranscranialReadingLimit",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — Physical Constants
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TranscranialConfig:
    # ── Tissue optical properties (published values) ──────────────────────
    # Scattering coefficient μ_s (cm⁻¹) for brain tissue
    mu_s_brain: float = 150.0           # 100-200 cm⁻¹
    mu_s_skull: float = 100.0           # slightly lower for bone
    
    # Absorption coefficient μ_a (cm⁻¹)
    mu_a_brain: float = 0.2             # wavelength-dependent
    mu_a_skull: float = 0.1
    
    # ── Geometry ──────────────────────────────────────────────────────────
    skull_thickness_cm: float = 0.7     # ~7 mm human skull
    scalp_thickness_cm: float = 0.5     # ~5 mm scalp
    cortex_depth_cm: float = 0.3        # ~3 mm cortical target
    
    # ── Safety limits (IEEE C95.1) ────────────────────────────────────────
    sar_limit_w_per_kg: float = 4.0     # local SAR limit
    max_temp_rise_c: float = 1.0        # maximum allowed temperature rise
    brain_mass_kg: float = 1.4          # average adult brain mass
    
    # ── Signal properties ─────────────────────────────────────────────────
    neuron_signal_mv: float = 100.0     # action potential amplitude
    eeg_signal_uv: float = 10.0         # EEG signal at scalp
    noise_floor_uv: float = 1.0         # typical EEG noise
    
    # ── Comparison parameters ─────────────────────────────────────────────
    implant_channels: int = 1024        # e.g., Neuralink
    eeg_channels: int = 64              # typical EEG cap
    
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# [P1] TISSUE OPTICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════════

class TissueOpticalProperties(nn.Module):
    """
    Models light propagation through brain tissue.
    
    Key physics:
      • Scattering: photons change direction randomly
      • Absorption: photons are absorbed by chromophores
      • Effective attenuation: combination of both
    
    Beer-Lambert law with scattering:
      I(d) = I_0 · exp(-μ_eff · d)
    
    where μ_eff = √(3·μ_a·(μ_a + μ_s')) is the effective attenuation
    coefficient and μ_s' = μ_s·(1-g) is the reduced scattering coefficient.
    """
    
    def __init__(self, cfg: TranscranialConfig):
        super().__init__()
        self.cfg = cfg
        
        # Anisotropy factor g (typical for biological tissue)
        self.register_buffer("g", torch.tensor(0.9), persistent=False)
        
        # Reduced scattering coefficient: μ_s' = μ_s·(1-g)
        mu_s_prime_brain = cfg.mu_s_brain * (1 - 0.9)  # ~15 cm⁻¹
        mu_s_prime_skull = cfg.mu_s_skull * (1 - 0.9)
        
        self.register_buffer("mu_s_prime_brain",
                             torch.tensor(mu_s_prime_brain), persistent=False)
        self.register_buffer("mu_s_prime_skull",
                             torch.tensor(mu_s_prime_skull), persistent=False)
        
        # Effective attenuation coefficients
        self.register_buffer(
            "mu_eff_brain",
            torch.tensor(math.sqrt(3 * cfg.mu_a_brain *
                                    (cfg.mu_a_brain + mu_s_prime_brain))),
            persistent=False)
        self.register_buffer(
            "mu_eff_skull",
            torch.tensor(math.sqrt(3 * cfg.mu_a_skull *
                                    (cfg.mu_a_skull + mu_s_prime_skull))),
            persistent=False)
    
    def compute_transmission(self, depth_cm: Tensor) -> Tensor:
        """
        Compute transmitted intensity fraction at depth d.
        
        I(d)/I_0 = exp(-μ_eff · d)
        
        Args:
            depth_cm: (B,) or scalar, depth in cm
        
        Returns:
            transmission_fraction: (B,) ∈ [0, 1]
        """
        # Combined effective attenuation (simplified: use brain value)
        mu_eff = self.mu_eff_brain
        
        transmission = torch.exp(-mu_eff * depth_cm)
        return transmission
    
    def compute_penetration_depth(self, threshold: float = 0.01) -> Tensor:
        """
        Compute depth at which intensity drops to threshold (default 1%).
        
        d_penetration = -ln(threshold) / μ_eff
        
        Args:
            threshold: minimum useful transmission fraction
        
        Returns:
            penetration_depth_cm: scalar
        """
        return -math.log(threshold) / self.mu_eff_brain


# ═══════════════════════════════════════════════════════════════════════
# [P2] TRANSCRANIAL SIGNAL MODEL
# ═══════════════════════════════════════════════════════════════════════

class TranscranialSignalModel(nn.Module):
    """
    Models signal attenuation through skull + scalp + tissue.
    
    The signal path for optical methods:
      1. Light enters scalp
      2. Light traverses skull (bone, highly scattering)
      3. Light reaches cortical target
      4. Reflected/transmitted signal returns
      5. Signal must be detected through same path
    
    Total round-trip attenuation is the SQUARE of one-way attenuation.
    """
    
    def __init__(self, cfg: TranscranialConfig):
        super().__init__()
        self.cfg = cfg
        self.optics = TissueOpticalProperties(cfg)
        
        # Total path length (round-trip)
        total_depth = (cfg.scalp_thickness_cm + 
                       cfg.skull_thickness_cm + 
                       cfg.cortex_depth_cm)
        self.register_buffer("total_depth_cm",
                             torch.tensor(total_depth), persistent=False)
    
    def forward(self, input_power_w: Tensor) -> Dict[str, Tensor]:
        """
        Compute signal attenuation through skull.
        
        Args:
            input_power_w: (B,) input optical power in watts
        
        Returns:
            dict with transmitted power, attenuation, SNR
        """
        B = input_power_w.shape[0]
        
        # One-way transmission through full depth
        one_way_transmission = self.optics.compute_transmission(self.total_depth_cm)
        
        # Round-trip (signal must return through same path)
        round_trip_transmission = one_way_transmission ** 2
        
        # Transmitted power
        transmitted_power = input_power_w * round_trip_transmission
        
        # Total attenuation in dB
        attenuation_db = -10 * torch.log10(round_trip_transmission + 1e-12)
        
        # Signal at target (assuming neural signal modulates light)
        # This is a very optimistic assumption — real neural signals
        # modulate light weakly
        neural_modulation_depth = 0.01  # 1% modulation (optimistic)
        signal_power = transmitted_power * neural_modulation_depth
        
        return {
            "input_power_w": input_power_w,
            "one_way_transmission": one_way_transmission.expand(B),
            "round_trip_transmission": round_trip_transmission.expand(B),
            "transmitted_power_w": transmitted_power,
            "attenuation_db": attenuation_db,
            "neural_signal_power_w": signal_power,
            "penetration_depth_cm": self.optics.compute_penetration_depth(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [P3] SAFE POWER LIMIT CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class SafePowerLimitCalculator(nn.Module):
    """
    Computes maximum safe input power based on safety limits.
    
    Constraints:
      • SAR limit: Specific Absorption Rate < 4 W/kg (IEEE)
      • Thermal limit: Temperature rise < 1°C
      • Photochemical damage: wavelength-dependent
    
    Key insight: "shooting more power" hits a HARD safety ceiling.
    You cannot simply increase power to overcome attenuation.
    """
    
    def __init__(self, cfg: TranscranialConfig):
        super().__init__()
        self.cfg = cfg
        
        # Heat capacity of brain tissue (J/kg·°C)
        self.register_buffer("heat_capacity",
                             torch.tensor(3600.0), persistent=False)
    
    def compute_sar_limit_power(self) -> Tensor:
        """
        Maximum power from SAR limit.
        
        SAR = Power / Mass
        → Power_max = SAR_limit × Mass
        
        For local exposure, mass ≈ 10 g (IEEE local SAR averaging)
        """
        local_mass_kg = 0.01  # 10 g averaging volume
        sar_limit = self.cfg.sar_limit_w_per_kg
        
        max_power = sar_limit * local_mass_kg
        return torch.tensor(max_power)
    
    def compute_thermal_limit_power(self, exposure_time_s: float = 1.0) -> Tensor:
        """
        Maximum power from thermal limit.
        
        ΔT = (P · t) / (m · c)
        → P_max = ΔT_max · m · c / t
        """
        delta_t_max = self.cfg.max_temp_rise_c
        mass = self.cfg.brain_mass_kg
        c = self.heat_capacity
        
        max_power = delta_t_max * mass * c / exposure_time_s
        return max_power
    
    def forward(self, exposure_time_s: float = 1.0) -> Dict[str, Tensor]:
        """
        Compute safe power limits.
        
        Args:
            exposure_time_s: exposure duration
        
        Returns:
            dict with SAR limit, thermal limit, effective limit
        """
        sar_limit = self.compute_sar_limit_power()
        thermal_limit = self.compute_thermal_limit_power(exposure_time_s)
        
        # Effective limit is the MINIMUM of all constraints
        effective_limit = torch.min(sar_limit, thermal_limit)
        
        return {
            "sar_limit_w": sar_limit,
            "thermal_limit_w": thermal_limit,
            "effective_safe_power_w": effective_limit,
            "HONEST_NOTE": "Cannot exceed these limits without tissue damage",
        }


# ═══════════════════════════════════════════════════════════════════════
# [P4] IMPLANT vs TRANSCRANIAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

class ImplantVsTranscranialComparison(nn.Module):
    """
    Quantitative comparison of implanted vs transcranial methods.
    
    Shows the FUNDAMENTAL gap between:
      • Implanted: near neurons, high SNR, single-cell resolution
      • Transcranial: through skull, low SNR, population-level only
    
    This demonstrates why "shooting enough signal" cannot replace implants.
    """
    
    def __init__(self, cfg: TranscranialConfig):
        super().__init__()
        self.cfg = cfg
        self.transcranial = TranscranialSignalModel(cfg)
        self.safety = SafePowerLimitCalculator(cfg)
    
    def forward(self) -> Dict[str, Tensor]:
        """
        Compare implanted vs transcranial approaches.
        
        Returns:
            dict with comparison metrics
        """
        # Safe power limit
        safety_out = self.safety()
        max_safe_power = safety_out["effective_safe_power_w"]
        
        # Transcranial signal at max safe power
        transcranial_out = self.transcranial(max_safe_power.unsqueeze(0))
        
        # Implanted signal (assumed: direct neural signal)
        # No attenuation, direct measurement
        implant_signal_uv = self.cfg.neuron_signal_mv * 1000  # 100 mV = 100,000 μV
        # But measured extracellularly, typically ~100 μV
        implant_signal_uv = torch.tensor(100.0)
        
        # Transcranial signal (heavily attenuated)
        transcranial_signal_uv = self.cfg.eeg_signal_uv  # ~10 μV at scalp
        
        # SNR comparison
        noise_uv = self.cfg.noise_floor_uv
        implant_snr = implant_signal_uv / noise_uv
        transcranial_snr = transcranial_signal_uv / noise_uv
        
        # Resolution comparison
        implant_resolution_um = 50.0    # ~50 μm (single-cell)
        transcranial_resolution_mm = 10.0  # ~10 mm (population)
        
        # Channels
        implant_channels = self.cfg.implant_channels
        transcranial_channels = self.cfg.eeg_channels
        
        return {
            # Signal strength
            "implant_signal_uv": implant_signal_uv,
            "transcranial_signal_uv": transcranial_signal_uv,
            "signal_ratio": implant_signal_uv / transcranial_signal_uv,
            
            # SNR
            "implant_snr": implant_snr,
            "transcranial_snr": transcranial_snr,
            "snr_ratio": implant_snr / transcranial_snr,
            
            # Resolution
            "implant_resolution_um": torch.tensor(implant_resolution_um),
            "transcranial_resolution_mm": torch.tensor(transcranial_resolution_mm),
            "resolution_ratio": torch.tensor(
                transcranial_resolution_mm * 1000 / implant_resolution_um),
            
            # Channels
            "implant_channels": torch.tensor(implant_channels),
            "transcranial_channels": torch.tensor(transcranial_channels),
            "channel_ratio": torch.tensor(
                implant_channels / transcranial_channels),
            
            # Power analysis
            "max_safe_power_w": max_safe_power,
            "transcranial_transmitted_power_w":
                transcranial_out["transmitted_power_w"].squeeze(0),
            "transcranial_attenuation_db":
                transcranial_out["attenuation_db"].squeeze(0),
            
            # Conclusion
            "can_replace_implant": torch.tensor(False),
            "HONEST_NOTE": (
                "Transcranial methods CANNOT replace implants for single-cell "
                "recording. Physics (scattering, absorption, safety limits) "
                "prevents it with current knowledge."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [P5] TRANSCRANIAL READING LIMIT — The Honest Answer
# ═══════════════════════════════════════════════════════════════════════

class TranscranialReadingLimit(nn.Module):
    """
    Computes the THEORETICAL LIMIT of what transcranial methods can achieve.
    
    This is the honest answer to: "Can shooting enough quantum signal
    into the brain replace implanted devices?"
    
    Answer: NO, due to fundamental physics:
      1. Scattering destroys spatial information
      2. Safety limits cap input power
      3. Inverse problem is ill-posed
      4. Signal modulation by single neurons is too weak
    
    What IS achievable:
      • Population-level activity (EEG, fNIRS)
      • Hemodynamic signals (slow, indirect)
      • Maybe, with future technology, better spatial resolution
        but still not single-cell
    """
    
    def __init__(self, cfg: TranscranialConfig):
        super().__init__()
        self.cfg = cfg
        self.comparison = ImplantVsTranscranialComparison(cfg)
    
    def forward(self) -> Dict[str, Tensor]:
        """
        Compute theoretical limits.
        
        Returns:
            dict with limits and conclusions
        """
        comparison_out = self.comparison()
        
        # Theoretical best-case SNR (assuming perfect detectors, no noise)
        # Even with perfect detection, scattering limits information content
        theoretical_best_snr = comparison_out["transcranial_snr"] * 10  # optimistic
        
        # Maximum achievable spatial resolution (diffraction + scattering)
        # Optical diffraction limit: ~λ/2 ≈ 0.5 μm
        # But scattering limits to ~1 mm in practice
        max_resolution_mm = 1.0  # realistic limit
        
        # Maximum channels (limited by independence of measurements)
        # With scattering, channels are not independent
        max_independent_channels = 100  # optimistic
        
        return {
            **comparison_out,
            "theoretical_best_snr": theoretical_best_snr,
            "max_spatial_resolution_mm": torch.tensor(max_resolution_mm),
            "max_independent_channels": torch.tensor(max_independent_channels),
            
            # Final verdict
            "can_read_single_neurons": torch.tensor(False),
            "can_read_population_activity": torch.tensor(True),
            "can_replace_implants": torch.tensor(False),
            
            "VERDICT": (
                "TRANSCRANIAL methods can read POPULATION-LEVEL activity "
                "(EEG, fNIRS). They CANNOT read SINGLE NEURONS or replace "
                "IMPLANTED devices. This is a FUNDAMENTAL PHYSICS LIMIT, "
                "not a technology gap. Shooting more power does not help "
                "because safety limits and scattering prevent it."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Transcranial Neural Signal Limit Analyzer — Smoke Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    cfg = TranscranialConfig(device=str(device))
    analyzer = TranscranialReadingLimit(cfg).to(device)
    
    out = analyzer()
    
    print(f"\n  ── Tissue Optical Properties ──")
    print(f"    μ_s (brain)            : {cfg.mu_s_brain} cm⁻¹")
    print(f"    μ_a (brain)            : {cfg.mu_a_brain} cm⁻¹")
    print(f"    Penetration depth      : {out['penetration_depth_cm'].item():.2f} cm")
    
    print(f"\n  ── Safety Limits ──")
    print(f"    SAR limit power        : {out['max_safe_power_w'].item():.4f} W")
    print(f"    NOTE: {out['HONEST_NOTE']}")
    
    print(f"\n  ── Transcranial Attenuation ──")
    print(f"    Attenuation            : {out['transcranial_attenuation_db'].item():.1f} dB")
    print(f"    Transmitted power      : {out['transcranial_transmitted_power_w'].item():.2e} W")
    
    print(f"\n  ── Implant vs Transcranial Comparison ──")
    print(f"    Implant signal         : {out['implant_signal_uv'].item():.0f} μV")
    print(f"    Transcranial signal    : {out['transcranial_signal_uv'].item():.0f} μV")
    print(f"    Signal ratio           : {out['signal_ratio'].item():.0f}×")
    print(f"    SNR ratio              : {out['snr_ratio'].item():.0f}×")
    print(f"    Resolution ratio       : {out['resolution_ratio'].item():.0f}×")
    print(f"    Channel ratio          : {out['channel_ratio'].item():.0f}×")
    
    print(f"\n  ── Theoretical Limits ──")
    print(f"    Best-case SNR          : {out['theoretical_best_snr'].item():.1f}")
    print(f"    Max resolution         : {out['max_spatial_resolution_mm'].item():.1f} mm")
    print(f"    Max channels           : {out['max_independent_channels'].item()}")
    
    print(f"\n  ── VERDICT ──")
    print(f"    Can read single neurons: {out['can_read_single_neurons'].item()}")
    print(f"    Can read population    : {out['can_read_population_activity'].item()}")
    print(f"    Can replace implants   : {out['can_replace_implants'].item()}")
    print(f"\n    {out['VERDICT']}")
    
    print("\n" + "=" * 72)
    print("  CONCLUSION:")
    print("  • Transcranial ≠ Implanted (physics prevents equivalence)")
    print("  • 'Shooting more power' hits safety ceiling")
    print("  • Scattering destroys single-cell information")
    print("  • Current best: population-level (EEG, fNIRS)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
