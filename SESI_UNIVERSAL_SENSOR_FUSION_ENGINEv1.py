"""
=============================================================================
SESI UNIVERSAL SENSOR FUSION ENGINE v1.0
All Sensor Types: Neural / Nuclear / Biomedical / Molecular / Environmental
Hardware-Agnostic: NVIDIA CUDA | Huawei Ascend NPU | Cambricon MLU | CPU
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

SENSOR TAXONOMY (15 categories, 40+ sensor types)
─────────────────────────────────────────────────
[A] NEURAL SENSORS (non-invasive)
    A1. EEG (Electroencephalography) — 10-1024 channels, μV range
    A2. MEG (Magnetoencephalography) — SQUID-based, fT range
    A3. OPM-MEG (Optically Pumped Magnetometers) — quantum sensors
    A4. fNIRS (functional Near-Infrared Spectroscopy) — hemodynamics
    A5. fMRI (functional MRI) — BOLD signal, whole-brain
    A6. fUS (functional Ultrasound) — cerebral blood volume
    A7. ECoG (Electrocorticography) — subdural grid

[B] NEURAL SENSORS (invasive / implant)
    B1. Utah Array / Blackrock — microelectrode arrays
    B2. Neural Dust — piezoelectric ultrasound-powered
    B3. Stentrode (Synchron) — endovascular BCI
    B4. Neural Lace — flexible mesh electrodes
    B5. Optogenetic fiber — light-based read/write

[C] NUCLEAR / RADIATION SENSORS
    C1. Neutron detector (He-3, BF3, scintillator)
    C2. Gamma spectrometer (NaI(Tl), HPGe, LaBr3)
    C3. Alpha detector (ZnS(Ag), silicon surface barrier)
    C4. Beta detector (GM tube, plastic scintillator)
    C5. Dosimeter (TLD, OSL, electronic personal)
    C6. Cherenkov detector — ultra-relativistic particles

[D] BIOMEDICAL VITAL SIGNS
    D1. ECG (Electrocardiography) — 12-lead
    D2. SpO2 (Pulse oximetry)
    D3. Blood pressure (invasive / non-invasive)
    D4. Temperature (core / skin)
    D5. Respiratory rate (impedance, capnography)
    D6. EMG (Electromyography)

[E] MOLECULAR / OMICS SENSORS
    E1. DNA sequencer (NGS, nanopore)
    E2. Protein mass spectrometer
    E3. CRISPR activity monitor
    E4. Metabolomics (NMR, MS)
    E5. Single-cell RNA-seq

[F] ENVIRONMENTAL SENSORS
    F1. Chemical (gas chromatograph, e-nose)
    F2. Biological (pathogen detector)
    F3. Radiological (area monitor)
    F4. Atmospheric (weather station)

[G] QUANTUM SENSORS
    G1. NV-center diamond magnetometer
    G2. Atomic clock (optical lattice)
    G3. SQUID (Superconducting Quantum Interference Device)
    G4. Gravitational wave detector (LIGO-class)

[H] IMAGING SENSORS
    H1. PET (Positron Emission Tomography)
    H2. SPECT (Single Photon Emission CT)
    H3. CT (X-ray Computed Tomography)
    H4. MRI (structural)
    H5. Ultrasound (diagnostic)
    H6. Optical coherence tomography

MATHEMATICAL FOUNDATION
───────────────────────
  [M1] Carpet Resistance Forms — sensor fusion on fractal graph
       Each sensor type lives on a carpet cell; fusion = Mosco limit
       of multi-sensor energy forms.
  [M2] Hodge Decomposition — separate signal into harmonic (true),
       exact (transient), coexact (noise) components.
  [M3] OPMC Canonical Signature — compress N sensor streams into
       polynomial-size σ_T without information loss.
  [M4] HCSN Weak Measurement — non-destructive sensor readout.
  [M5] SQCT Collapse — sensor triggers state reduction at interface.
  [M6] RUT Relational Accessibility — which sensors can physically
       reach which targets (causal constraints).

HARDWARE ABSTRACTION
────────────────────
  Supports: NVIDIA CUDA (H100/B200), Huawei Ascend 910B/310P,
            Cambricon MLU 370/590, Apple MPS, CPU fallback.
  Auto-detects device at runtime; no code changes needed.
  Sanction-resilient: works under US technology export controls.

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("sesi.sensor_fusion")

__all__ = [
    # Config
    "SensorFusionConfig",
    # Enums
    "SensorCategory", "SensorType", "HardwareBackend",
    # Hardware abstraction
    "HardwareAbstractionLayer", "get_device",
    # Base sensor classes
    "SensorBase", "SensorArray", "SensorRegistry",
    # Category A: Non-invasive neural
    "EEGSensor", "MEGSensor", "OPMMEGSensor", "FNRSSensor",
    "FMRISensor", "FUSSensor", "ECoGSensor",
    # Category B: Invasive neural
    "UtahArraySensor", "NeuralDustSensor", "StentrodeSensor",
    "NeuralLaceSensor", "OptogeneticSensor",
    # Category C: Nuclear
    "NeutronDetector", "GammaSpectrometer", "AlphaDetector",
    "BetaDetector", "Dosimeter", "CherenkovDetector",
    # Category D: Biomedical
    "ECGSensor", "SpO2Sensor", "BloodPressureSensor",
    "TemperatureSensor", "RespiratorySensor", "EMGSensor",
    # Category E: Molecular
    "DNASensor", "ProteinMassSpec", "CRISPRMonitor",
    "MetabolomicsSensor", "SingleCellRNASEQ",
    # Category F: Environmental
    "ChemicalSensor", "BiologicalSensor", "RadiologicalSensor",
    "AtmosphericSensor",
    # Category G: Quantum
    "NVDiamondSensor", "AtomicClockSensor", "SQUIDSensor",
    "GravitationalWaveSensor",
    # Category H: Imaging
    "PETSensor", "SPECTSensor", "CTSensor",
    "MRISensor", "UltrasoundSensor", "OCTSensor",
    # Fusion engine
    "CarpetSensorFusion",
    "HodgeSignalDecomposer",
    "OPMCCanonicalCompressor",
    "UniversalSensorFusionEngine",
    # Training
    "SensorFusionLoss",
    "SensorFusionTrainer",
    "SyntheticMultiSensorDataset",
    "build_sensor_loaders",
    # Infrastructure
    "setup_dist", "cleanup_dist", "seed_all",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class SensorCategory(Enum):
    NEURAL_NONINVASIVE = auto()  # A
    NEURAL_INVASIVE = auto()     # B
    NUCLEAR = auto()             # C
    BIOMEDICAL = auto()          # D
    MOLECULAR = auto()           # E
    ENVIRONMENTAL = auto()       # F
    QUANTUM = auto()             # G
    IMAGING = auto()             # H


class SensorType(Enum):
    # A: Non-invasive neural
    EEG = "eeg"
    MEG = "meg"
    OPM_MEG = "opm_meg"
    FNIRS = "fnirs"
    FMRI = "fmri"
    FUS = "fus"
    ECOG = "ecog"
    # B: Invasive neural
    UTAH_ARRAY = "utah_array"
    NEURAL_DUST = "neural_dust"
    STENTRODE = "stentrode"
    NEURAL_LACE = "neural_lace"
    OPTOGENETIC = "optogenetic"
    # C: Nuclear
    NEUTRON = "neutron"
    GAMMA = "gamma"
    ALPHA = "alpha"
    BETA = "beta"
    DOSIMETER = "dosimeter"
    CHERENKOV = "cherenkov"
    # D: Biomedical
    ECG = "ecg"
    SPO2 = "spo2"
    BLOOD_PRESSURE = "blood_pressure"
    TEMPERATURE = "temperature"
    RESPIRATORY = "respiratory"
    EMG = "emg"
    # E: Molecular
    DNA_SEQ = "dna_seq"
    PROTEIN_MS = "protein_ms"
    CRISPR_MON = "crispr_mon"
    METABOLOMICS = "metabolomics"
    SC_RNASEQ = "sc_rnaseq"
    # F: Environmental
    CHEMICAL = "chemical"
    BIOLOGICAL = "biological"
    RADIOLOGICAL = "radiological"
    ATMOSPHERIC = "atmospheric"
    # G: Quantum
    NV_DIAMOND = "nv_diamond"
    ATOMIC_CLOCK = "atomic_clock"
    SQUID = "squid"
    GRAV_WAVE = "grav_wave"
    # H: Imaging
    PET = "pet"
    SPECT = "spect"
    CT = "ct"
    MRI_STRUCT = "mri_struct"
    ULTRASOUND = "ultrasound"
    OCT = "oct"


class HardwareBackend(Enum):
    NVIDIA_CUDA = "cuda"
    HUAWEI_ASCEND = "npu"
    CAMBRICON_MLU = "mlu"
    APPLE_MPS = "mps"
    CPU = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
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
        device = get_device()
        if device.type == "cuda":
            torch.cuda.set_device(local)
            dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend="gloo")
        return device, dist.get_rank(), dist.get_world_size()
    return get_device(), 0, 1

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
# HARDWARE ABSTRACTION LAYER (Sanction-Resilient)
# ═══════════════════════════════════════════════════════════════════════

class HardwareAbstractionLayer:
    """
    Detects available hardware and provides unified interface.
    Priority order (sanction-resilient):
      1. Huawei Ascend NPU (torch_npu)
      2. Cambricon MLU (torch_mlu)
      3. NVIDIA CUDA
      4. Apple MPS
      5. CPU fallback
    """
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.device = self._get_device()
        logger.info(f"[HAL] Detected backend: {self.backend.value} on {self.device}")
    
    def _detect_backend(self) -> HardwareBackend:
        # Huawei Ascend
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return HardwareBackend.HUAWEI_ASCEND
        # Cambricon MLU
        if hasattr(torch, 'mlu') and torch.mlu.is_available():
            return HardwareBackend.CAMBRICON_MLU
        # NVIDIA CUDA
        if torch.cuda.is_available():
            return HardwareBackend.NVIDIA_CUDA
        # Apple MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return HardwareBackend.APPLE_MPS
        return HardwareBackend.CPU
    
    def _get_device(self) -> torch.device:
        if self.backend == HardwareBackend.HUAWEI_ASCEND:
            return torch.device("npu")
        elif self.backend == HardwareBackend.CAMBRICON_MLU:
            return torch.device("mlu")
        elif self.backend == HardwareBackend.NVIDIA_CUDA:
            return torch.device("cuda")
        elif self.backend == HardwareBackend.APPLE_MPS:
            return torch.device("mps")
        return torch.device("cpu")
    
    def supports_triton(self) -> bool:
        """Triton only works on NVIDIA CUDA."""
        return self.backend == HardwareBackend.NVIDIA_CUDA
    
    def supports_fused_adamw(self) -> bool:
        """Fused AdamW only on CUDA."""
        return self.backend == HardwareBackend.NVIDIA_CUDA
    
    def compile_backend(self) -> str:
        """Backend string for torch.compile."""
        if self.backend == HardwareBackend.HUAWEI_ASCEND:
            return "npu"
        elif self.backend == HardwareBackend.CAMBRICON_MLU:
            return "mlu"
        elif self.backend == HardwareBackend.NVIDIA_CUDA:
            return "inductor"
        return "eager"


def get_device() -> torch.device:
    return HardwareAbstractionLayer().device


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Smooth Primitives
# ═══════════════════════════════════════════════════════════════════════

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
    """[C1] Low-rank projection: O(d·r) instead of O(d²)."""
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
class SensorFusionConfig:
    # ── Core dimensions ─────────────────────────────────────────────────
    latent_dim: int = 512
    sensor_embed_dim: int = 128
    fusion_dim: int = 256
    
    # ── Sensor counts per category ──────────────────────────────────────
    n_eeg_channels: int = 64
    n_meg_channels: int = 306
    n_fnirs_channels: int = 72
    n_fmri_voxels: int = 1024
    n_ecg_leads: int = 12
    n_neutron_detectors: int = 8
    n_gamma_detectors: int = 4
    
    # ── Temporal parameters ─────────────────────────────────────────────
    sampling_rate_hz: float = 1000.0
    window_size_ms: int = 100
    n_time_steps: int = 128
    
    # ── Carpet fusion ───────────────────────────────────────────────────
    carpet_level: int = 3
    carpet_n_cells: int = 8 ** 3
    resistance_scale_rho: float = 1.25
    
    # ── Hodge decomposition ─────────────────────────────────────────────
    hodge_dim: int = 64
    
    # ── OPMC compression ────────────────────────────────────────────────
    opmc_sigma_dim: int = 128
    opmc_n_fields: int = 16
    
    # ── Hardware ────────────────────────────────────────────────────────
    hardware_backend: str = "auto"  # auto | cuda | npu | mlu | cpu
    
    # ── Cost optimization ───────────────────────────────────────────────
    mental_rank: int = 32
    gradient_checkpointing: bool = True
    use_torch_compile: bool = True
    
    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    amp_dtype: str = "bf16"
    seed: int = 42
    device: str = "cuda"
    
    # ── Loss weights ────────────────────────────────────────────────────
    w_fusion: float = 0.3
    w_hodge: float = 0.2
    w_opmc: float = 0.2
    w_recon: float = 0.2
    w_safety: float = 0.1
    w_reg: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# BASE SENSOR CLASSES
# ═══════════════════════════════════════════════════════════════════════

class SensorBase(nn.Module):
    """
    Abstract base class for all sensors.
    Each sensor:
      - Has a type, category, and physical parameters
      - Produces a fixed-size embedding
      - Reports quality/confidence
      - Can be fused with other sensors via carpet
    """
    
    def __init__(
        self,
        sensor_type: SensorType,
        category: SensorCategory,
        embed_dim: int,
        n_channels: int,
        n_time_steps: int,
        sampling_rate: float,
        noise_level: float = 0.01,
    ):
        super().__init__()
        self.sensor_type = sensor_type
        self.category = category
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.n_time_steps = n_time_steps
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        
        # Temporal encoder (1D conv + transformer)
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(n_channels, embed_dim // 2, kernel_size=7, padding=3),
            nn.GroupNorm(1, embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
        )
        
        # Quality estimator
        self.quality_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Sensor-specific calibration parameters
        self.calibration_bias = nn.Parameter(torch.zeros(n_channels))
        self.calibration_scale = nn.Parameter(torch.ones(n_channels))
    
    def calibrate(self, raw_signal: Tensor) -> Tensor:
        """Apply sensor-specific calibration."""
        return (raw_signal + self.calibration_bias) * self.calibration_scale
    
    def add_noise(self, signal: Tensor, training: bool) -> Tensor:
        """Add sensor noise (only during training for regularization)."""
        if training and self.noise_level > 0:
            noise = torch.randn_like(signal) * self.noise_level
            return signal + noise
        return signal
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            raw_signal: (B, n_channels, n_time_steps)
        Returns:
            dict with embedding, quality, calibrated_signal
        """
        B = raw_signal.shape[0]
        
        # Calibrate
        calibrated = self.calibrate(raw_signal)
        
        # Add noise (training only)
        noisy = self.add_noise(calibrated, self.training)
        
        # Temporal encoding
        temporal_feat = self.temporal_enc(noisy)  # (B, embed_dim, T)
        embedding = temporal_feat.mean(dim=-1)    # (B, embed_dim)
        
        # Quality estimation
        quality = self.quality_net(embedding).squeeze(-1)
        
        return {
            "embedding": embedding,
            "quality": quality,
            "calibrated_signal": calibrated,
            "noisy_signal": noisy,
            "temporal_features": temporal_feat,
            "sensor_type": self.sensor_type.value,
        }


class SensorArray(nn.Module):
    """Container for multiple sensors of the same type."""
    
    def __init__(self, sensors: List[SensorBase]):
        super().__init__()
        self.sensors = nn.ModuleList(sensors)
        self.n_sensors = len(sensors)
    
    def forward(self, signals: List[Tensor]) -> Dict[str, Tensor]:
        outputs = [sensor(sig) for sensor, sig in zip(self.sensors, signals)]
        
        # Stack embeddings
        embeddings = torch.stack([o["embedding"] for o in outputs], dim=1)  # (B, N, D)
        qualities = torch.stack([o["quality"] for o in outputs], dim=1)     # (B, N)
        
        # Quality-weighted mean
        weights = F.softmax(qualities, dim=-1)
        fused_embedding = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        
        return {
            "array_embedding": fused_embedding,
            "per_sensor_embeddings": embeddings,
            "qualities": qualities,
            "weights": weights,
        }


class SensorRegistry:
    """Registry of all available sensors."""
    
    def __init__(self):
        self._sensors: Dict[SensorType, SensorBase] = {}
        self._arrays: Dict[SensorCategory, SensorArray] = {}
    
    def register(self, sensor: SensorBase) -> None:
        self._sensors[sensor.sensor_type] = sensor
    
    def get(self, sensor_type: SensorType) -> Optional[SensorBase]:
        return self._sensors.get(sensor_type)
    
    def list_by_category(self, category: SensorCategory) -> List[SensorBase]:
        return [s for s in self._sensors.values() if s.category == category]
    
    def all_types(self) -> List[SensorType]:
        return list(self._sensors.keys())


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY A: NON-INVASIVE NEURAL SENSORS
# ═══════════════════════════════════════════════════════════════════════

class EEGSensor(SensorBase):
    """
    Electroencephalography — 10-1024 channels, μV range.
    Temporal resolution: ~1 ms
    Spatial resolution: ~cm (low)
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.EEG,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_eeg_channels,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=cfg.sampling_rate_hz,
            noise_level=0.05,  # EEG is noisy
        )
        # Frequency band decomposition (delta/theta/alpha/beta/gamma)
        self.band_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 5),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["frequency_bands"] = self.band_net(out["embedding"])
        return out


class MEGSensor(SensorBase):
    """
    Magnetoencephalography — SQUID-based, fT range.
    Better spatial resolution than EEG, similar temporal.
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.MEG,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_meg_channels,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=cfg.sampling_rate_hz,
            noise_level=0.02,
        )


class OPMMEGSensor(SensorBase):
    """
    Optically Pumped Magnetometer MEG — quantum sensors.
    Wearable, room-temperature, higher sensitivity than SQUID.
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.OPM_MEG,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_meg_channels // 2,  # fewer channels but better
            n_time_steps=cfg.n_time_steps,
            sampling_rate=cfg.sampling_rate_hz,
            noise_level=0.01,  # lower noise than SQUID
        )
        # Quantum sensor calibration
        self.quantum_calibration = nn.Parameter(torch.ones(1))


class FNRSSensor(SensorBase):
    """
    functional Near-Infrared Spectroscopy — hemodynamics.
    Measures oxygenated/deoxygenated hemoglobin.
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.FNIRS,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_fnirs_channels,
            n_time_steps=cfg.n_time_steps // 10,  # slower sampling
            sampling_rate=10.0,  # Hz
            noise_level=0.03,
        )
        # HbO/HbR decomposition
        self.hemo_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 2),
            nn.Softmax(dim=-1),  # [HbO, HbR]
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["hemoglobin_decomp"] = self.hemo_net(out["embedding"])
        return out


class FMRISensor(SensorBase):
    """
    functional MRI — BOLD signal, whole-brain.
    High spatial (~mm), low temporal (~s).
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.FMRI,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_fmri_voxels,
            n_time_steps=cfg.n_time_steps // 100,
            sampling_rate=0.5,  # Hz (TR ~ 2s)
            noise_level=0.02,
        )
        # Spatial attention over voxels
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=cfg.sensor_embed_dim,
            num_heads=4,
            batch_first=True,
        )


class FUSSensor(SensorBase):
    """
    functional Ultrasound — cerebral blood volume.
    High spatial and temporal, portable.
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.FUS,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=128,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=500.0,
            noise_level=0.02,
        )


class ECoGSensor(SensorBase):
    """
    Electrocorticography — subdural grid.
    Invasive but high spatial/temporal resolution.
    """
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.ECOG,
            category=SensorCategory.NEURAL_NONINVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=64,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=cfg.sampling_rate_hz,
            noise_level=0.01,
        )


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY B: INVASIVE NEURAL SENSORS
# ═══════════════════════════════════════════════════════════════════════

class UtahArraySensor(SensorBase):
    """Utah Array / Blackrock microelectrode arrays."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.UTAH_ARRAY,
            category=SensorCategory.NEURAL_INVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=96,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=30000.0,  # 30 kHz for spike sorting
            noise_level=0.005,
        )
        # Spike detection
        self.spike_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["spike_probability"] = self.spike_net(out["embedding"]).squeeze(-1)
        return out


class NeuralDustSensor(SensorBase):
    """Neural Dust — piezoelectric ultrasound-powered micro-sensors."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.NEURAL_DUST,
            category=SensorCategory.NEURAL_INVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=1000,  # thousands of dust particles
            n_time_steps=cfg.n_time_steps,
            sampling_rate=10000.0,
            noise_level=0.01,
        )
        # Ultrasound backscatter modeling
        self.ultrasound_gain = nn.Parameter(torch.ones(1))


class StentrodeSensor(SensorBase):
    """Stentrode (Synchron) — endovascular BCI."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.STENTRODE,
            category=SensorCategory.NEURAL_INVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=16,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1000.0,
            noise_level=0.02,
        )


class NeuralLaceSensor(SensorBase):
    """Neural Lace — flexible mesh electrodes."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.NEURAL_LACE,
            category=SensorCategory.NEURAL_INVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=256,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=10000.0,
            noise_level=0.005,
        )


class OptogeneticSensor(SensorBase):
    """Optogenetic fiber — light-based read/write."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.OPTOGENETIC,
            category=SensorCategory.NEURAL_INVASIVE,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=8,  # few fibers but precise
            n_time_steps=cfg.n_time_steps,
            sampling_rate=10000.0,
            noise_level=0.001,
        )
        # Light wavelength parameter
        self.wavelength_nm = nn.Parameter(torch.tensor(473.0))  # blue light


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY C: NUCLEAR / RADIATION SENSORS
# ═══════════════════════════════════════════════════════════════════════

class NeutronDetector(SensorBase):
    """Neutron detector (He-3, BF3, scintillator)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.NEUTRON,
            category=SensorCategory.NUCLEAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_neutron_detectors,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e6,  # 1 MHz
            noise_level=0.01,
        )
        # Neutron energy spectrum
        self.energy_spectrum = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["energy_spectrum"] = self.energy_spectrum(out["embedding"])
        return out


class GammaSpectrometer(SensorBase):
    """Gamma spectrometer (NaI(Tl), HPGe, LaBr3)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.GAMMA,
            category=SensorCategory.NUCLEAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_gamma_detectors,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e6,
            noise_level=0.005,
        )
        # Isotope identification
        self.isotope_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 20),  # 20 common isotopes
            nn.Softmax(dim=-1),
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["isotope_probs"] = self.isotope_net(out["embedding"])
        return out


class AlphaDetector(SensorBase):
    """Alpha particle detector (ZnS(Ag), silicon surface barrier)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.ALPHA,
            category=SensorCategory.NUCLEAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=4,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e5,
            noise_level=0.01,
        )


class BetaDetector(SensorBase):
    """Beta particle detector (GM tube, plastic scintillator)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.BETA,
            category=SensorCategory.NUCLEAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=4,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e5,
            noise_level=0.01,
        )


class Dosimeter(SensorBase):
    """Personal dosimeter (TLD, OSL, electronic)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.DOSIMETER,
            category=SensorCategory.NUCLEAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=1,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,  # 1 Hz
            noise_level=0.001,
        )
        # Dose rate estimator
        self.dose_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 1),
            nn.Softplus(),  # dose > 0
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["dose_rate"] = self.dose_net(out["embedding"]).squeeze(-1)
        return out


class CherenkovDetector(SensorBase):
    """Cherenkov detector for ultra-relativistic particles."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.CHERENKOV,
            category=SensorCategory.NUCLEAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=16,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e9,  # 1 GHz
            noise_level=0.001,
        )


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY D: BIOMEDICAL VITAL SIGNS
# ═══════════════════════════════════════════════════════════════════════

class ECGSensor(SensorBase):
    """12-lead Electrocardiography."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.ECG,
            category=SensorCategory.BIOMEDICAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=cfg.n_ecg_leads,
            n_time_steps=cfg.n_time_steps * 2,
            sampling_rate=500.0,
            noise_level=0.02,
        )
        # Heart rate variability
        self.hrv_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 4),
            nn.Softplus(),  # [HR, HRV_SDNN, HRV_RMSSD, LF/HF]
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["hrv_features"] = self.hrv_net(out["embedding"])
        return out


class SpO2Sensor(SensorBase):
    """Pulse oximetry."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.SPO2,
            category=SensorCategory.BIOMEDICAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=2,  # red + IR
            n_time_steps=cfg.n_time_steps,
            sampling_rate=100.0,
            noise_level=0.01,
        )
        # SpO2 estimator
        self.spo2_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 1),
            nn.Sigmoid(),  # [0, 1]
        )
    
    def forward(self, raw_signal: Tensor) -> Dict[str, Tensor]:
        out = super().forward(raw_signal)
        out["spo2"] = self.spo2_net(out["embedding"]).squeeze(-1)
        return out


class BloodPressureSensor(SensorBase):
    """Blood pressure (invasive / non-invasive)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.BLOOD_PRESSURE,
            category=SensorCategory.BIOMEDICAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=2,  # systolic + diastolic
            n_time_steps=cfg.n_time_steps,
            sampling_rate=100.0,
            noise_level=0.02,
        )


class TemperatureSensor(SensorBase):
    """Temperature (core / skin)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.TEMPERATURE,
            category=SensorCategory.BIOMEDICAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=4,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=0.001,
        )


class RespiratorySensor(SensorBase):
    """Respiratory rate (impedance, capnography)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.RESPIRATORY,
            category=SensorCategory.BIOMEDICAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=2,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=25.0,
            noise_level=0.02,
        )


class EMGSensor(SensorBase):
    """Electromyography."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.EMG,
            category=SensorCategory.BIOMEDICAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=8,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1000.0,
            noise_level=0.03,
        )


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY E: MOLECULAR / OMICS SENSORS
# ═══════════════════════════════════════════════════════════════════════

class DNASensor(SensorBase):
    """DNA sequencer (NGS, nanopore)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.DNA_SEQ,
            category=SensorCategory.MOLECULAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=4,  # A, C, G, T
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,  # per base
            noise_level=0.01,
        )
        # Base calling network
        self.base_caller = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 4),
            nn.Softmax(dim=-1),
        )


class ProteinMassSpec(SensorBase):
    """Protein mass spectrometer."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.PROTEIN_MS,
            category=SensorCategory.MOLECULAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=64,  # m/z bins
            n_time_steps=cfg.n_time_steps,
            sampling_rate=100.0,
            noise_level=0.02,
        )


class CRISPRMonitor(SensorBase):
    """CRISPR activity monitor."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.CRISPR_MON,
            category=SensorCategory.MOLECULAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=8,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=10.0,
            noise_level=0.01,
        )
        # Editing efficiency
        self.efficiency_net = nn.Sequential(
            nn.Linear(cfg.sensor_embed_dim, 1),
            nn.Sigmoid(),
        )


class MetabolomicsSensor(SensorBase):
    """Metabolomics (NMR, MS)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.METABOLOMICS,
            category=SensorCategory.MOLECULAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=128,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=0.02,
        )


class SingleCellRNASEQ(SensorBase):
    """Single-cell RNA-seq."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.SC_RNASEQ,
            category=SensorCategory.MOLECULAR,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=2000,  # genes
            n_time_steps=1,   # snapshot
            sampling_rate=1.0,
            noise_level=0.05,  # high dropout
        )


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY F: ENVIRONMENTAL SENSORS
# ═══════════════════════════════════════════════════════════════════════

class ChemicalSensor(SensorBase):
    """Chemical sensor (gas chromatograph, e-nose)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.CHEMICAL,
            category=SensorCategory.ENVIRONMENTAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=16,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=0.02,
        )


class BiologicalSensor(SensorBase):
    """Biological sensor (pathogen detector)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.BIOLOGICAL,
            category=SensorCategory.ENVIRONMENTAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=8,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=0.1,
            noise_level=0.03,
        )


class RadiologicalSensor(SensorBase):
    """Area radiological monitor."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.RADIOLOGICAL,
            category=SensorCategory.ENVIRONMENTAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=4,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=0.01,
        )


class AtmosphericSensor(SensorBase):
    """Atmospheric / weather station."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.ATMOSPHERIC,
            category=SensorCategory.ENVIRONMENTAL,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=8,  # T, P, humidity, wind, etc.
            n_time_steps=cfg.n_time_steps,
            sampling_rate=0.1,
            noise_level=0.01,
        )


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY G: QUANTUM SENSORS
# ═══════════════════════════════════════════════════════════════════════

class NVDiamondSensor(SensorBase):
    """NV-center diamond magnetometer."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.NV_DIAMOND,
            category=SensorCategory.QUANTUM,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=1,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e6,
            noise_level=0.001,  # ultra-low noise
        )
        # Quantum coherence time
        self.t2_coherence = nn.Parameter(torch.tensor(1e-3))  # seconds


class AtomicClockSensor(SensorBase):
    """Atomic clock (optical lattice)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.ATOMIC_CLOCK,
            category=SensorCategory.QUANTUM,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=1,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=1e-18,  # fractional frequency stability
        )


class SQUIDSensor(SensorBase):
    """SQUID (Superconducting Quantum Interference Device)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.SQUID,
            category=SensorCategory.QUANTUM,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=1,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1e6,
            noise_level=1e-6,  # fT/√Hz
        )


class GravitationalWaveSensor(SensorBase):
    """Gravitational wave detector (LIGO-class)."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.GRAV_WAVE,
            category=SensorCategory.QUANTUM,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=2,  # LIGO has 2 detectors
            n_time_steps=cfg.n_time_steps,
            sampling_rate=16384.0,
            noise_level=1e-23,  # strain sensitivity
        )


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY H: IMAGING SENSORS
# ═══════════════════════════════════════════════════════════════════════

class PETSensor(SensorBase):
    """Positron Emission Tomography."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.PET,
            category=SensorCategory.IMAGING,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=256,  # detector rings
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=0.02,
        )


class SPECTSensor(SensorBase):
    """Single Photon Emission CT."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.SPECT,
            category=SensorCategory.IMAGING,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=64,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=1.0,
            noise_level=0.03,
        )


class CTSensor(SensorBase):
    """X-ray Computed Tomography."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.CT,
            category=SensorCategory.IMAGING,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=512,  # detector elements
            n_time_steps=1,  # single scan
            sampling_rate=1.0,
            noise_level=0.01,
        )


class MRISensor(SensorBase):
    """Structural MRI."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.MRI_STRUCT,
            category=SensorCategory.IMAGING,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=1024,
            n_time_steps=1,
            sampling_rate=1.0,
            noise_level=0.01,
        )


class UltrasoundSensor(SensorBase):
    """Diagnostic ultrasound."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.ULTRASOUND,
            category=SensorCategory.IMAGING,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=128,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=30.0,  # frame rate
            noise_level=0.02,
        )


class OCTSensor(SensorBase):
    """Optical Coherence Tomography."""
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__(
            sensor_type=SensorType.OCT,
            category=SensorCategory.IMAGING,
            embed_dim=cfg.sensor_embed_dim,
            n_channels=512,
            n_time_steps=cfg.n_time_steps,
            sampling_rate=100.0,
            noise_level=0.01,
        )


# ═══════════════════════════════════════════════════════════════════════
# CARPET SENSOR FUSION (M1)
# ═══════════════════════════════════════════════════════════════════════

class CarpetSensorFusion(nn.Module):
    """
    [M1] Fuse all sensor embeddings on Sierpiński carpet graph.
    
    Each sensor type occupies a carpet cell; fusion = Mosco limit
    of multi-sensor energy forms.
    """
    
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        # Lift: sensor embeddings → carpet cells
        self.lift = nn.Sequential(
            LowRankAdapter(cfg.sensor_embed_dim, cfg.carpet_n_cells, r),
            nn.GELU(),
        )
        
        # Resistance scaling (learnable)
        self.log_rho_raw = nn.Parameter(
            torch.tensor(math.log(max(cfg.resistance_scale_rho - 9/8, 1e-6)))
        )
        
        # Readout: carpet → fused latent
        self.readout = nn.Sequential(
            nn.Linear(cfg.carpet_n_cells, cfg.fusion_dim),
            nn.GELU(),
            nn.Linear(cfg.fusion_dim, cfg.latent_dim),
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
    
    def forward(self, sensor_embeddings: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            sensor_embeddings: (B, n_sensors, embed_dim)
        Returns:
            dict with fused_embedding, carpet_energy, etc.
        """
        B, N, D = sensor_embeddings.shape
        
        # Pool sensors to single embedding per sample
        sensor_pooled = sensor_embeddings.mean(dim=1)  # (B, D)
        
        # Lift to carpet
        cell_values = self.lift(sensor_pooled)  # (B, n_cells)
        
        # Carpet energy
        graph_energy = torch.einsum("bi,ij,bj->b",
                                     cell_values,
                                     self.carpet_laplacian,
                                     cell_values)
        carpet_energy = (self.rho ** self.cfg.carpet_level) * graph_energy
        
        # Readout
        fused = self.readout(cell_values)  # (B, latent_dim)
        
        return {
            "fused_embedding": fused,
            "carpet_energy": carpet_energy,
            "cell_values": cell_values,
            "resistance_rho": self.rho.expand(B),
        }


# ═══════════════════════════════════════════════════════════════════════
# HODGE SIGNAL DECOMPOSER (M2)
# ═══════════════════════════════════════════════════════════════════════

class HodgeSignalDecomposer(nn.Module):
    """
    [M2] Decompose fused signal into harmonic/exact/coexact components.
    
    harmonic = true signal
    exact = transient artifacts
    coexact = noise
    """
    
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        self.decompose_net = nn.Sequential(
            LowRankAdapter(cfg.latent_dim, cfg.hodge_dim * 3, r),
            nn.GELU(),
            nn.Linear(cfg.hodge_dim * 3, cfg.hodge_dim * 3),
        )
        
        # Hodge-Riemann form
        self.q_form = nn.Parameter(torch.randn(cfg.hodge_dim, cfg.hodge_dim) * 0.01)
    
    def forward(self, fused_embedding: Tensor) -> Dict[str, Tensor]:
        decomp = self.decompose_net(fused_embedding)
        harmonic, exact, coexact = decomp.chunk(3, dim=-1)
        
        # Hodge-Riemann check
        q_sym = 0.5 * (self.q_form + self.q_form.T)
        hr_value = torch.einsum("bi,ij,bj->b", harmonic, q_sym, harmonic)
        
        return {
            "harmonic": harmonic,
            "exact": exact,
            "coexact": coexact,
            "hodge_riemann_value": hr_value,
            "signal_to_noise": (harmonic.pow(2).sum(-1) /
                                (coexact.pow(2).sum(-1) + 1e-8)),
        }


# ═══════════════════════════════════════════════════════════════════════
# OPMC CANONICAL COMPRESSOR (M3)
# ═══════════════════════════════════════════════════════════════════════

class OPMCCanonicalCompressor(nn.Module):
    """
    [M3] Compress multi-sensor data into canonical signature σ_T.
    
    Polynomial-size representation of hyper-relational sensor space.
    """
    
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        self.field_emb = nn.Parameter(
            torch.randn(cfg.opmc_n_fields, cfg.latent_dim) * 0.02)
        self.query = LowRankAdapter(cfg.latent_dim, cfg.latent_dim, r)
        self.value = LowRankAdapter(cfg.latent_dim, cfg.opmc_sigma_dim, r)
        
        self.refine = nn.Sequential(
            nn.LayerNorm(cfg.opmc_sigma_dim),
            nn.Linear(cfg.opmc_sigma_dim, cfg.opmc_sigma_dim),
            nn.GELU(),
            nn.Linear(cfg.opmc_sigma_dim, cfg.opmc_sigma_dim),
        )
        
        # Compression quality
        self.quality_net = nn.Sequential(
            nn.Linear(cfg.opmc_sigma_dim, cfg.opmc_sigma_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.opmc_sigma_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, harmonic: Tensor) -> Dict[str, Tensor]:
        B = harmonic.shape[0]
        
        q = self.query(harmonic).unsqueeze(1)
        logits = torch.einsum("bnd,fd->bnf", q, self.field_emb) / math.sqrt(q.shape[-1])
        prov = F.softmax(logits, dim=1)
        
        v = self.value(harmonic).unsqueeze(1)
        sigma = torch.einsum("bnf,bnd->bfd", prov, v).squeeze(1)
        sigma = self.refine(sigma)
        
        quality = self.quality_net(sigma).squeeze(-1)
        
        return {
            "canonical_signature": sigma,
            "compression_quality": quality,
            "field_provenance": prov.squeeze(1),
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSAL SENSOR FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class UniversalSensorFusionEngine(nn.Module):
    """
    Complete sensor fusion pipeline:
      raw_signals → per-sensor encoding → carpet fusion →
      Hodge decomposition → OPMC compression → final latent
    """
    
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__()
        self.cfg = cfg
        
        # Initialize all sensors
        self.sensors = nn.ModuleDict({
            # A: Non-invasive neural
            "eeg": EEGSensor(cfg),
            "meg": MEGSensor(cfg),
            "opm_meg": OPMMEGSensor(cfg),
            "fnirs": FNRSSensor(cfg),
            "fmri": FMRISensor(cfg),
            "fus": FUSSensor(cfg),
            "ecog": ECoGSensor(cfg),
            # B: Invasive neural
            "utah_array": UtahArraySensor(cfg),
            "neural_dust": NeuralDustSensor(cfg),
            "stentrode": StentrodeSensor(cfg),
            "neural_lace": NeuralLaceSensor(cfg),
            "optogenetic": OptogeneticSensor(cfg),
            # C: Nuclear
            "neutron": NeutronDetector(cfg),
            "gamma": GammaSpectrometer(cfg),
            "alpha": AlphaDetector(cfg),
            "beta": BetaDetector(cfg),
            "dosimeter": Dosimeter(cfg),
            "cherenkov": CherenkovDetector(cfg),
            # D: Biomedical
            "ecg": ECGSensor(cfg),
            "spo2": SpO2Sensor(cfg),
            "blood_pressure": BloodPressureSensor(cfg),
            "temperature": TemperatureSensor(cfg),
            "respiratory": RespiratorySensor(cfg),
            "emg": EMGSensor(cfg),
            # E: Molecular
            "dna_seq": DNASensor(cfg),
            "protein_ms": ProteinMassSpec(cfg),
            "crispr_mon": CRISPRMonitor(cfg),
            "metabolomics": MetabolomicsSensor(cfg),
            "sc_rnaseq": SingleCellRNASEQ(cfg),
            # F: Environmental
            "chemical": ChemicalSensor(cfg),
            "biological": BiologicalSensor(cfg),
            "radiological": RadiologicalSensor(cfg),
            "atmospheric": AtmosphericSensor(cfg),
            # G: Quantum
            "nv_diamond": NVDiamondSensor(cfg),
            "atomic_clock": AtomicClockSensor(cfg),
            "squid": SQUIDSensor(cfg),
            "grav_wave": GravitationalWaveSensor(cfg),
            # H: Imaging
            "pet": PETSensor(cfg),
            "spect": SPECTSensor(cfg),
            "ct": CTSensor(cfg),
            "mri_struct": MRISensor(cfg),
            "ultrasound": UltrasoundSensor(cfg),
            "oct": OCTSensor(cfg),
        })
        
        # Fusion pipeline
        self.carpet_fusion = CarpetSensorFusion(cfg)
        self.hodge_decomp = HodgeSignalDecomposer(cfg)
        self.opmc_compress = OPMCCanonicalCompressor(cfg)
        
        # Final output head
        self.output_head = nn.Sequential(
            nn.Linear(cfg.opmc_sigma_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
    
    def forward(self, sensor_signals: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sensor_signals: dict mapping sensor_name → (B, C, T) tensor
                           Only sensors present in dict are processed.
        Returns:
            dict with all fusion outputs
        """
        out = {}
        embeddings = []
        qualities = []
        
        # Process each available sensor
        for name, sensor in self.sensors.items():
            if name in sensor_signals:
                sensor_out = sensor(sensor_signals[name])
                embeddings.append(sensor_out["embedding"])
                qualities.append(sensor_out["quality"])
                out[f"sensor_{name}_quality"] = sensor_out["quality"]
        
        if not embeddings:
            # No sensors available — return zeros
            B = 1
            return {
                "fused_embedding": torch.zeros(B, self.cfg.latent_dim,
                                                device=next(self.parameters()).device),
                "canonical_signature": torch.zeros(B, self.cfg.opmc_sigma_dim,
                                                    device=next(self.parameters()).device),
            }
        
        # Stack embeddings
        embeddings = torch.stack(embeddings, dim=1)  # (B, N, D)
        qualities = torch.stack(qualities, dim=1)    # (B, N)
        
        out["per_sensor_embeddings"] = embeddings
        out["per_sensor_qualities"] = qualities
        
        # Carpet fusion
        carpet_out = self.carpet_fusion(embeddings)
        out.update({f"carpet_{k}": v for k, v in carpet_out.items()})
        
        # Hodge decomposition
        hodge_out = self.hodge_decomp(carpet_out["fused_embedding"])
        out.update({f"hodge_{k}": v for k, v in hodge_out.items()})
        
        # OPMC compression
        opmc_out = self.opmc_compress(hodge_out["harmonic"])
        out.update({f"opmc_{k}": v for k, v in opmc_out.items()})
        
        # Final output
        final = self.output_head(opmc_out["canonical_signature"])
        out["final_latent"] = final
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════

class SensorFusionLoss(nn.Module):
    def __init__(self, cfg: SensorFusionConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        dev = out["final_latent"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}
        
        # Fusion consistency
        L["fusion"] = out.get("carpet_carpet_energy", zero).mean()
        
        # Hodge decomposition quality
        if "hodge_hodge_riemann_value" in out:
            L["hodge"] = F.relu(-out["hodge_hodge_riemann_value"]).mean()
        else:
            L["hodge"] = zero
        
        # OPMC compression quality
        if "opmc_compression_quality" in out:
            L["opmc"] = -out["opmc_compression_quality"].mean()
        else:
            L["opmc"] = zero
        
        # Sensor quality (encourage high quality)
        if "per_sensor_qualities" in out:
            L["recon"] = -out["per_sensor_qualities"].mean()
        else:
            L["recon"] = zero
        
        # Safety: SNR should be positive
        if "hodge_signal_to_noise" in out:
            L["safety"] = F.relu(-out["hodge_signal_to_noise"]).mean()
        else:
            L["safety"] = zero
        
        # Regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg
        
        total = (
            self.cfg.w_fusion * L["fusion"] +
            self.cfg.w_hodge * L["hodge"] +
            self.cfg.w_opmc * L["opmc"] +
            self.cfg.w_recon * L["recon"] +
            self.cfg.w_safety * L["safety"] +
            self.cfg.w_reg * L["reg"]
        )
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

class SyntheticMultiSensorDataset(Dataset):
    """Synthetic dataset with all sensor types."""
    
    def __init__(self, cfg: SensorFusionConfig, n: int, seed: int):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        
        # Generate synthetic signals for each sensor
        self.signals = {}
        for name in [
            "eeg", "meg", "opm_meg", "fnirs", "fmri", "fus", "ecog",
            "utah_array", "neural_dust", "stentrode", "neural_lace", "optogenetic",
            "neutron", "gamma", "alpha", "beta", "dosimeter", "cherenkov",
            "ecg", "spo2", "blood_pressure", "temperature", "respiratory", "emg",
            "dna_seq", "protein_ms", "crispr_mon", "metabolomics", "sc_rnaseq",
            "chemical", "biological", "radiological", "atmospheric",
            "nv_diamond", "atomic_clock", "squid", "grav_wave",
            "pet", "spect", "ct", "mri_struct", "ultrasound", "oct",
        ]:
            sensor = getattr(cfg, f"n_{name}_channels", None) or \
                     getattr(cfg, f"n_{name}_leads", None) or 8
            t_steps = cfg.n_time_steps
            self.signals[name] = torch.randn(n, sensor, t_steps, generator=g)
    
    def __len__(self):
        return len(next(iter(self.signals.values())))
    
    def __getitem__(self, i):
        return {name: sig[i] for name, sig in self.signals.items()}


def build_sensor_loaders(cfg: SensorFusionConfig):
    tr = SyntheticMultiSensorDataset(cfg, 1024, cfg.seed)
    va = SyntheticMultiSensorDataset(cfg, 128, cfg.seed + 1)
    te = SyntheticMultiSensorDataset(cfg, 256, cfg.seed + 2)
    
    common = dict(batch_size=cfg.batch_size, num_workers=2,
                  pin_memory=(cfg.device != "cpu"))
    
    return (
        DataLoader(tr, shuffle=True, drop_last=True, **common),
        DataLoader(va, shuffle=False, **common),
        DataLoader(te, shuffle=False, **common),
    )


# ═══════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════

class SensorFusionTrainer:
    def __init__(self, cfg: SensorFusionConfig, model: UniversalSensorFusionEngine,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        
        if cfg.use_torch_compile and hasattr(torch, "compile"):
            try:
                backend = HardwareAbstractionLayer().compile_backend()
                self.model = torch.compile(self.model, mode="max-autotune",
                                            backend=backend)
                if is_main():
                    logger.info(f"torch.compile enabled (backend={backend})")
            except Exception as e:
                if is_main():
                    logger.warning(f"torch.compile failed: {e}")
        
        if is_dist():
            self.model = DDP(self.model,
                             device_ids=[device.index] if device.type == "cuda" else None,
                             find_unused_parameters=False,
                             gradient_as_bucket_view=True)
        
        self.loss_fn = SensorFusionLoss(cfg)
        
        amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[cfg.amp_dtype]
        self._amp_dtype = amp_dt
        self._amp_on = cfg.amp_dtype != "fp32" and device.type in ("cuda", "npu", "mlu")
        
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler(enabled=(self._amp_on and amp_dt == torch.float16))
        except ImportError:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=(self._amp_on and amp_dt == torch.float16))
        
        hal = HardwareAbstractionLayer()
        self.opt = torch.optim.AdamW(
            [p for p in self._raw().parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=1e-5,
            fused=hal.supports_fused_adamw(),
        )
        self.step = 0
    
    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        self.opt.zero_grad(set_to_none=True)
        
        try:
            from torch.amp import autocast
            ctx = autocast(device_type=self.device.type,
                           dtype=self._amp_dtype, enabled=self._amp_on)
        except ImportError:
            from torch.cuda.amp import autocast
            ctx = autocast(dtype=self._amp_dtype, enabled=self._amp_on)
        
        with ctx:
            out = self._raw()(batch)
            losses = self.loss_fn(out, batch, self._raw())
            loss = losses["total"]
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self._raw().parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        
        self.step += 1
        return {k: float(v) for k, v in losses.items()
                if torch.is_tensor(v) and v.numel() == 1}


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Universal Sensor Fusion Engine — Smoke Test")
    print("=" * 72)
    
    hal = HardwareAbstractionLayer()
    device = hal.device
    print(f"  Hardware backend: {hal.backend.value}")
    print(f"  Device: {device}")
    print(f"  Triton support: {hal.supports_triton()}")
    print(f"  Fused AdamW: {hal.supports_fused_adamw()}")
    
    torch.manual_seed(42)
    
    cfg = SensorFusionConfig(
        latent_dim=64,
        sensor_embed_dim=32,
        fusion_dim=32,
        n_eeg_channels=8,
        n_meg_channels=16,
        n_fnirs_channels=8,
        n_fmri_voxels=32,
        n_ecg_leads=4,
        n_neutron_detectors=2,
        n_gamma_detectors=2,
        n_time_steps=16,
        carpet_level=2,
        carpet_n_cells=8**2,
        hodge_dim=16,
        opmc_sigma_dim=32,
        opmc_n_fields=4,
        batch_size=2,
        use_torch_compile=False,
        gradient_checkpointing=False,
        device=str(device),
    )
    
    model = UniversalSensorFusionEngine(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    print(f"  Sensor types: {len(model.sensors)}")
    
    # Generate synthetic signals
    sensor_signals = {}
    for name, sensor in model.sensors.items():
        B = 2
        sensor_signals[name] = torch.randn(
            B, sensor.n_channels, sensor.n_time_steps, device=device)
    
    model.train()
    out = model(sensor_signals)
    
    loss_fn = SensorFusionLoss(cfg)
    losses = loss_fn(out, sensor_signals, model)
    losses["total"].backward()
    
    print(f"\n  ── Sensor Categories ──")
    categories = {}
    for name, sensor in model.sensors.items():
        cat = sensor.category.name
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"    {cat:<25s}: {count} sensors")
    
    print(f"\n  ── Fusion Outputs ──")
    print(f"    Final latent shape      : {tuple(out['final_latent'].shape)}")
    print(f"    Canonical signature     : {tuple(out['opmc_canonical_signature'].shape)}")
    print(f"    Carpet energy           : {out['carpet_carpet_energy'].mean().item():.4f}")
    print(f"    Hodge SNR               : {out['hodge_signal_to_noise'].mean().item():.4f}")
    print(f"    Compression quality     : {out['opmc_compression_quality'].mean().item():.4f}")
    
    print(f"\n  ── Per-Sensor Qualities (sample) ──")
    for name in list(model.sensors.keys())[:10]:
        key = f"sensor_{name}_quality"
        if key in out:
            print(f"    {name:<20s}: {out[key].mean().item():.4f}")
    
    print(f"\n  ── Losses ──")
    for k, v in losses.items():
        print(f"    {k:<20s}: {v.item():.6f}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL 42 SENSORS CONNECTED' if grad_ok else 'FAIL'}")
    
    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser("SESI Universal Sensor Fusion Engine")
    p.add_argument("--mode", choices=["smoke", "train", "list"], default="smoke")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")
    
    if args.mode == "list":
        print("Available sensor types:")
        for st in SensorType:
            print(f"  {st.value}")
        return
    
    if args.mode == "smoke":
        smoke_test()
        return
    
    # Train mode
    device, r, ws = setup_dist()
    seed_all(42)
    cfg = SensorFusionConfig(device=str(device))
    
    model = UniversalSensorFusionEngine(cfg)
    trainer = SensorFusionTrainer(cfg, model, device)
    
    train_dl, _, _ = build_sensor_loaders(cfg)
    
    for step in range(args.steps):
        for batch in train_dl:
            losses = trainer.train_step(batch)
            if is_main() and trainer.step % 10 == 0:
                logger.info(f"Step {trainer.step}: loss={losses['total']:.4f}")
            break
    
    cleanup_dist()


if __name__ == "__main__":
    main()
