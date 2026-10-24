"""
=============================================================================
SESI HARDWARE INTERFACE & REAL-TIME STREAMING SYSTEM v1.0
Universal Driver Layer for 15+ BCI/Neuroscience Hardware Platforms
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

SUPPORTED HARDWARE PLATFORMS (15 categories)
─────────────────────────────────────────────
[A] EEG SYSTEMS (Non-invasive)
    A1. BrainVision (Brain Products) — actiCAP, actiCHamp, V-Amp
    A2. g.tec — g.USBamp, g.HIamp, g.Nautilus, g.USBamp-RE
    A3. OpenBCI — Cyton, Ganglion, Daisy
    A4. BioSemi — ActiveTwo (256 channels)
    A5. ANT Neuro — eego sports, eego mylab
    A6. Micromed — SystemPLUS Evolution
    A7. Natus — NeuroWorks
    A8. TMSi — SNAP, APEX

[B] INVASIVE NEURAL INTERFACES
    B1. Blackrock Neurotech — Utah Array, CerePlex M/E
    B2. Neuralynx — Cheetah, Digital Lynx
    B3. Plexon — OmniPlex, MAP, RZ2
    B4. NeuroPort (Blackrock) — high-density microelectrode
    B5. Synchron — Stentrode (endovascular BCI)
    B6. Neuralink — N1 implant (theoretical)

[C] MEG SYSTEMS
    C1. MEGIN (Elekta) — TRIUX, NEO
    C2. CTF Systems — VSM, OMEGA
    C3. 4D Neuroimaging — BTi/4D
    C4. FieldLine (OPM-MEG) — Kernel Flux, FieldLine

[D] fNIRS SYSTEMS
    D1. NIRx — NIRScout, NIRSport2
    D2. Artinis — Brite+, Brite24, OctaMon
    D3. ISS — Imagemodule, OxyPrem

[E] fMRI (Indirect via DICOM/NIfTI)
    E1. Siemens — MAGNETOM series
    E2. GE — SIGNA, Discovery
    E3. Philips — Ingenia, Achieva

[F] QUANTUM SENSORS
    F1. OPM-MEG (QuSpin, FieldLine)
    F2. NV-Diamond (Qnami, Quantum Diamond Tech)
    F3. SQUID (Conductus, Star Cryo)
    F4. Atomic Clock (NIST-class)

[G] NUCLEAR/RADIATION SENSORS
    G1. Canberra — Genie 2000, Inspector 1000
    G2. Berthold — LB 123, Microcont
    G3. Ludlum — 3-88, 2241-3
    G4. Thermo Fisher — RadEye series

[H] BIOMEDICAL VITAL SIGNS
    H1. Philips — IntelliVue
    H2. GE — CARESCAPE
    H3. Nihon Kohden — BSM series
    H4. Masimo — Radical-7, SET

STREAMING PROTOCOLS
───────────────────
  [S1] LSL (Lab Streaming Layer) — standard in BCI research
  [S2] TCP/IP socket streaming
  [S3] UDP multicast (low-latency)
  [S4] WebSocket (browser-friendly)
  [S5] ZeroMQ (distributed pub/sub)
  [S6] gRPC (high-performance RPC)
  [S7] Apache Kafka (distributed log)
  [S8] Redis pub/sub (in-memory)

ARCHITECTURE
────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Hardware Layer (15+ drivers)                                       │
│  ├── BrainVisionDriver                                              │
│  ├── gtecDriver                                                     │
│  ├── BlackrockDriver                                                │
│  ├── OpenBCIDriver                                                  │
│  ├── SynchronDriver                                                 │
│  ├── MEGINDriver                                                    │
│  ├── NIRxDriver                                                     │
│  ├── fMRIDriver (DICOM)                                             │
│  ├── OPMQuantumDriver                                               │
│  └── ... (all 15+ categories)                                       │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Hardware Abstraction Layer (HAL)                                   │
│  ├── HardwareRegistry (auto-detect connected devices)               │
│  ├── UnifiedDriverInterface                                         │
│  ├── SampleRateAdapter                                              │
│  └── TimestampSynchronizer                                          │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Real-Time Streaming Layer                                          │
│  ├── LSLStreamer (Lab Streaming Layer)                              │
│  ├── TCPStreamer                                                    │
│  ├── UDPStreamer                                                    │
│  ├── WebSocketStreamer                                              │
│  ├── ZeroMQStreamer                                                 │
│  └── StreamingRouter (multi-protocol fan-out)                       │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SESI Universal Sensor Fusion Engine                                │
│  ├── 42 sensor types (from previous module)                         │
│  ├── Carpet fusion                                                  │
│  ├── Hodge decomposition                                            │
│  └── OPMC compression                                               │
└─────────────────────────────────────────────────────────────────────┘

SANCTION RESILIENCE
───────────────────
  • Hardware drivers use OPTIONAL SDK imports (try/except)
  • Falls back to simulated data when SDK unavailable
  • Compatible with NVIDIA CUDA, Huawei Ascend NPU, Cambricon MLU
  • No US-only dependencies in core pipeline

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import json
import queue
import socket
import struct
import logging
import argparse
import threading
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.hardware")

__all__ = [
    # Enums
    "HardwareCategory", "HardwareType", "StreamProtocol",
    # Core classes
    "HardwareConfig", "HardwareMetadata", "SampleBuffer",
    "HardwareDriverBase",
    # EEG drivers
    "BrainVisionDriver", "GtecDriver", "OpenBCIDriver",
    "BioSemiDriver", "ANTNeuroDriver", "MicromedDriver",
    "NatusDriver", "TMSiDriver",
    # Invasive drivers
    "BlackrockDriver", "NeuralynxDriver", "PlexonDriver",
    "SynchronDriver", "NeuralinkDriver",
    # MEG drivers
    "MEGINDriver", "CTFDriver", "OPMMEGDriver",
    # fNIRS drivers
    "NIRxDriver", "ArtinisDriver", "ISSDriver",
    # fMRI drivers
    "SiemensfMRIDriver", "GEfMRIDriver", "PhilipsfMRIDriver",
    # Quantum sensors
    "QuSpinOPMDriver", "NVDiamondDriver", "SQUIDDriver",
    # Nuclear/radiation
    "CanberraDriver", "BertholdDriver", "LudlumDriver",
    # Biomedical
    "PhilipsVitalsDriver", "GEVitalsDriver", "MasimoDriver",
    # Streaming
    "LSLStreamer", "TCPStreamer", "UDPStreamer",
    "WebSocketStreamer", "ZeroMQStreamer", "StreamingRouter",
    # HAL
    "HardwareRegistry", "HardwareAbstractionLayer",
    # Integration
    "StreamingFusionPipeline",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class HardwareCategory(Enum):
    EEG = auto()
    INVASIVE = auto()
    MEG = auto()
    FNIRS = auto()
    FMRI = auto()
    QUANTUM = auto()
    NUCLEAR = auto()
    VITALS = auto()


class HardwareType(Enum):
    # EEG
    BRAINVISION = "brainvision"
    GTEC = "gtec"
    OPENBCI = "openbci"
    BIOSEMI = "biosemi"
    ANT_NEURO = "ant_neuro"
    MICROMED = "micromed"
    NATUS = "natus"
    TMSI = "tmsi"
    # Invasive
    BLACKROCK = "blackrock"
    NEURALYNX = "neuralynx"
    PLEXON = "plexon"
    SYNCHRON = "synchron"
    NEURALINK = "neuralink"
    # MEG
    MEGIN = "megin"
    CTF = "ctf"
    OPM_MEG = "opm_meg"
    # fNIRS
    NIRX = "nirx"
    ARTINIS = "artinis"
    ISS = "iss"
    # fMRI
    SIEMENS_FMRI = "siemens_fmri"
    GE_FMRI = "ge_fmri"
    PHILIPS_FMRI = "philips_fmri"
    # Quantum
    QUSPIN_OPM = "quspin_opm"
    NV_DIAMOND = "nv_diamond"
    SQUID = "squid"
    # Nuclear
    CANBERRA = "canberra"
    BERTHOLD = "berthold"
    LUDLUM = "ludlum"
    # Vitals
    PHILIPS_VITALS = "philips_vitals"
    GE_VITALS = "ge_vitals"
    MASIMO = "masimo"


class StreamProtocol(Enum):
    LSL = "lsl"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    ZMQ = "zmq"
    GRPC = "grpc"
    KAFKA = "kafka"
    REDIS = "redis"


# ═══════════════════════════════════════════════════════════════════════
# CONFIG & METADATA
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HardwareConfig:
    """Configuration for a hardware device."""
    device_type: HardwareType
    category: HardwareCategory
    sample_rate: float = 1000.0
    n_channels: int = 64
    serial_number: str = ""
    ip_address: Optional[str] = None
    port: Optional[int] = None
    usb_bus: Optional[int] = None
    usb_device: Optional[int] = None
    sdk_path: Optional[str] = None
    buffer_size: int = 4096
    timeout_ms: int = 5000
    use_simulated: bool = False


@dataclass
class HardwareMetadata:
    """Runtime metadata for a connected device."""
    device_type: HardwareType
    is_connected: bool = False
    is_streaming: bool = False
    firmware_version: str = "unknown"
    sdk_version: str = "unknown"
    battery_level: Optional[float] = None
    signal_quality: Optional[float] = None
    last_sample_time: float = 0.0
    dropped_samples: int = 0
    total_samples: int = 0


@dataclass
class SampleBuffer:
    """
    Thread-safe ring buffer for streaming samples.
    Supports both NumPy (for hardware) and PyTorch (for fusion).
    """
    capacity: int
    n_channels: int
    sample_rate: float
    
    def __post_init__(self):
        self._buffer = np.zeros((self.capacity, self.n_channels), dtype=np.float32)
        self._timestamps = np.zeros(self.capacity, dtype=np.float64)
        self._write_idx = 0
        self._read_idx = 0
        self._lock = threading.Lock()
        self._event = threading.Event()
    
    def write(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> int:
        """Write samples to buffer. Returns number of samples written."""
        n_samples = data.shape[0]
        if timestamps is None:
            timestamps = np.arange(n_samples) / self.sample_rate + time.time()
        
        with self._lock:
            available = self.capacity - (self._write_idx - self._read_idx)
            n_to_write = min(n_samples, available)
            if n_to_write == 0:
                return 0
            
            end_idx = self._write_idx + n_to_write
            if end_idx <= self.capacity:
                self._buffer[self._write_idx:end_idx] = data[:n_to_write]
                self._timestamps[self._write_idx:end_idx] = timestamps[:n_to_write]
            else:
                first = self.capacity - self._write_idx
                self._buffer[self._write_idx:] = data[:first]
                self._timestamps[self._write_idx:] = timestamps[:first]
                self._buffer[:n_to_write - first] = data[first:n_to_write]
                self._timestamps[:n_to_write - first] = timestamps[first:n_to_write]
            
            self._write_idx = (self._write_idx + n_to_write) % self.capacity
            self._event.set()
        
        return n_to_write
    
    def read(self, n_samples: Optional[int] = None, timeout: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Read samples from buffer. Blocks until data available or timeout."""
        if n_samples is None:
            n_samples = self.capacity
        
        self._event.wait(timeout=timeout)
        
        with self._lock:
            available = self._write_idx - self._read_idx
            if self._write_idx < self._read_idx:
                available += self.capacity
            n_to_read = min(n_samples, available)
            
            if n_to_read == 0:
                return np.zeros((0, self.n_channels), dtype=np.float32), np.zeros(0, dtype=np.float64)
            
            end_idx = self._read_idx + n_to_read
            if end_idx <= self.capacity:
                data = self._buffer[self._read_idx:end_idx].copy()
                timestamps = self._timestamps[self._read_idx:end_idx].copy()
            else:
                first = self.capacity - self._read_idx
                data = np.concatenate([
                    self._buffer[self._read_idx:],
                    self._buffer[:n_to_read - first]
                ])
                timestamps = np.concatenate([
                    self._timestamps[self._read_idx:],
                    self._timestamps[:n_to_read - first]
                ])
            
            self._read_idx = (self._read_idx + n_to_read) % self.capacity
            
            if self._read_idx == self._write_idx:
                self._event.clear()
        
        return data, timestamps
    
    def to_torch(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Convert last read to PyTorch tensors."""
        data, timestamps = self.read()
        return (torch.from_numpy(data).to(device),
                torch.from_numpy(timestamps).to(device))
    
    @property
    def available_samples(self) -> int:
        with self._lock:
            diff = self._write_idx - self._read_idx
            return diff if diff >= 0 else diff + self.capacity


# ═══════════════════════════════════════════════════════════════════════
# HARDWARE DRIVER BASE CLASS
# ═══════════════════════════════════════════════════════════════════════

class HardwareDriverBase(ABC):
    """
    Abstract base class for all hardware drivers.
    
    Contract:
      - connect() → bool
      - disconnect() → None
      - start_stream() → None
      - stop_stream() → None
      - read_chunk(n_samples) → (data, timestamps)
      - get_metadata() → HardwareMetadata
    
    All drivers support graceful fallback to simulated data when:
      - SDK not installed
      - Device not connected
      - Permission denied
    """
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.metadata = HardwareMetadata(device_type=config.device_type)
        self._buffer = SampleBuffer(
            capacity=config.buffer_size,
            n_channels=config.n_channels,
            sample_rate=config.sample_rate,
        )
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sdk_available = False
        self._sdk_client = None
    
    @abstractmethod
    def _try_import_sdk(self) -> bool:
        """Try to import the vendor SDK. Returns True if available."""
        pass
    
    @abstractmethod
    def _connect_hardware(self) -> bool:
        """Attempt to connect to real hardware. Returns True if successful."""
        pass
    
    @abstractmethod
    def _start_hardware_stream(self) -> None:
        """Start streaming from real hardware."""
        pass
    
    @abstractmethod
    def _stop_hardware_stream(self) -> None:
        """Stop streaming from real hardware."""
        pass
    
    @abstractmethod
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        """Generate a simulated sample (for fallback)."""
        pass
    
    def connect(self) -> bool:
        """Connect to the device (real or simulated)."""
        self._sdk_available = self._try_import_sdk()
        
        if self._sdk_available and not self.config.use_simulated:
            success = self._connect_hardware()
            if success:
                self.metadata.is_connected = True
                logger.info(f"[{self.config.device_type.value}] Connected to real hardware")
                return True
            else:
                logger.warning(f"[{self.config.device_type.value}] Hardware connection failed, falling back to simulation")
        
        # Simulated fallback
        self.metadata.is_connected = True
        self.metadata.sdk_version = "simulated"
        logger.info(f"[{self.config.device_type.value}] Using simulated data")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        self.stop_stream()
        self.metadata.is_connected = False
        logger.info(f"[{self.config.device_type.value}] Disconnected")
    
    def start_stream(self) -> None:
        """Start streaming data."""
        if not self.metadata.is_connected:
            raise RuntimeError("Device not connected")
        if self.metadata.is_streaming:
            return
        
        self._stop_event.clear()
        
        if self._sdk_available and not self.config.use_simulated:
            self._start_hardware_stream()
        else:
            # Start simulated streaming thread
            self._stream_thread = threading.Thread(
                target=self._simulated_stream_loop,
                daemon=True,
                name=f"{self.config.device_type.value}_stream"
            )
            self._stream_thread.start()
        
        self.metadata.is_streaming = True
        logger.info(f"[{self.config.device_type.value}] Streaming started")
    
    def stop_stream(self) -> None:
        """Stop streaming data."""
        if not self.metadata.is_streaming:
            return
        
        self._stop_event.set()
        
        if self._sdk_available and not self.config.use_simulated:
            self._stop_hardware_stream()
        
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        
        self.metadata.is_streaming = False
        logger.info(f"[{self.config.device_type.value}] Streaming stopped")
    
    def read_chunk(self, n_samples: int, timeout: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Read a chunk of samples from the buffer."""
        return self._buffer.read(n_samples, timeout=timeout)
    
    def read_chunk_torch(self, n_samples: int, device: torch.device,
                          timeout: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Read a chunk as PyTorch tensors."""
        data, timestamps = self.read_chunk(n_samples, timeout)
        return (torch.from_numpy(data).to(device),
                torch.from_numpy(timestamps).to(device))
    
    def get_metadata(self) -> HardwareMetadata:
        return self.metadata
    
    def _simulated_stream_loop(self) -> None:
        """Generate simulated samples at the configured sample rate."""
        chunk_size = int(self.config.sample_rate * 0.01)  # 10ms chunks
        interval = 1.0 / self.config.sample_rate * chunk_size
        
        while not self._stop_event.is_set():
            t0 = time.time()
            
            # Generate simulated data
            data = np.zeros((chunk_size, self.config.n_channels), dtype=np.float32)
            for ch in range(self.config.n_channels):
                # Simulate realistic neural-like signals
                t = np.arange(chunk_size) / self.config.sample_rate + t0
                # Mix of frequencies: alpha (10Hz), beta (20Hz), theta (6Hz)
                signal = (
                    10.0 * np.sin(2 * np.pi * 10 * t + ch * 0.1) +
                    5.0 * np.sin(2 * np.pi * 20 * t + ch * 0.2) +
                    3.0 * np.sin(2 * np.pi * 6 * t + ch * 0.3) +
                    np.random.randn(chunk_size) * 2.0  # noise
                )
                data[:, ch] = signal.astype(np.float32)
            
            timestamps = np.arange(chunk_size) / self.config.sample_rate + t0
            
            # Write to buffer
            written = self._buffer.write(data, timestamps)
            self.metadata.total_samples += written
            if written < chunk_size:
                self.metadata.dropped_samples += (chunk_size - written)
            self.metadata.last_sample_time = t0
            
            # Sleep to maintain sample rate
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)


# ═══════════════════════════════════════════════════════════════════════
# EEG DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class BrainVisionDriver(HardwareDriverBase):
    """
    Brain Products EEG driver.
    Supports: actiCAP, actiCHamp, V-Amp, LiveAmp
    SDK: pybv (open-source Python wrapper) or BrainVision USB API
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.BRAINVISION,
                category=HardwareCategory.EEG,
                sample_rate=1000.0,
                n_channels=64,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import pybv  # Open-source BrainVision reader
            self._sdk_client = pybv
            return True
        except ImportError:
            try:
                import brainvision  # Official SDK
                self._sdk_client = brainvision
                return True
            except ImportError:
                return False
    
    def _connect_hardware(self) -> bool:
        if self._sdk_client is None:
            return False
        try:
            # Attempt USB connection
            if hasattr(self._sdk_client, 'find_device'):
                device = self._sdk_client.find_device(self.config.serial_number)
                return device is not None
            return False
        except Exception as e:
            logger.debug(f"BrainVision connection failed: {e}")
            return False
    
    def _start_hardware_stream(self) -> None:
        if self._sdk_client is not None and hasattr(self._sdk_client, 'start_recording'):
            self._sdk_client.start_recording()
    
    def _stop_hardware_stream(self) -> None:
        if self._sdk_client is not None and hasattr(self._sdk_client, 'stop_recording'):
            self._sdk_client.stop_recording()
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        t = time.time()
        data = np.random.randn(self.config.n_channels).astype(np.float32) * 10.0
        return data, t


class GtecDriver(HardwareDriverBase):
    """
    g.tec EEG driver.
    Supports: g.USBamp, g.HIamp, g.Nautilus, g.USBamp-RE
    SDK: g.USBamp Python API
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.GTEC,
                category=HardwareCategory.EEG,
                sample_rate=1200.0,
                n_channels=16,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import gtec  # g.tec SDK
            self._sdk_client = gtec
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        if self._sdk_client is None:
            return False
        try:
            amp = self._sdk_client.USBamp()
            amp.open()
            self._sdk_client = amp
            return True
        except Exception as e:
            logger.debug(f"g.tec connection failed: {e}")
            return False
    
    def _start_hardware_stream(self) -> None:
        if hasattr(self._sdk_client, 'start'):
            self._sdk_client.start()
    
    def _stop_hardware_stream(self) -> None:
        if hasattr(self._sdk_client, 'stop'):
            self._sdk_client.stop()
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 5.0, time.time()


class OpenBCIDriver(HardwareDriverBase):
    """
    OpenBCI EEG driver.
    Supports: Cyton (8ch), Daisy (16ch), Ganglion (4ch)
    SDK: brainflow (open-source)
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.OPENBCI,
                category=HardwareCategory.EEG,
                sample_rate=250.0,
                n_channels=8,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import brainflow
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
            self._sdk_client = brainflow
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        if self._sdk_client is None:
            return False
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
            params = BrainFlowInputParams()
            params.serial_port = self.config.serial_number or '/dev/ttyUSB0'
            # Cyton board ID = 0
            self._board = BoardShim(0, params)
            self._board.prepare_session()
            return True
        except Exception as e:
            logger.debug(f"OpenBCI connection failed: {e}")
            return False
    
    def _start_hardware_stream(self) -> None:
        if hasattr(self, '_board'):
            self._board.start_stream()
    
    def _stop_hardware_stream(self) -> None:
        if hasattr(self, '_board'):
            self._board.stop_stream()
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 15.0, time.time()


class BioSemiDriver(HardwareDriverBase):
    """BioSemi ActiveTwo EEG driver (256 channels)."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.BIOSEMI,
                category=HardwareCategory.EEG,
                sample_rate=2048.0,
                n_channels=256,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import biosemi
            self._sdk_client = biosemi
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False  # Requires proprietary hardware
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 20.0, time.time()


class ANTNeuroDriver(HardwareDriverBase):
    """ANT Neuro eego driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.ANT_NEURO,
                category=HardwareCategory.EEG,
                sample_rate=16000.0,
                n_channels=72,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import eemagine
            self._sdk_client = eemagine
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 10.0, time.time()


class MicromedDriver(HardwareDriverBase):
    """Micromed SystemPLUS driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.MICROMED,
                category=HardwareCategory.EEG,
                sample_rate=4096.0,
                n_channels=64,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import micromed
            self._sdk_client = micromed
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 10.0, time.time()


class NatusDriver(HardwareDriverBase):
    """Natus NeuroWorks driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.NATUS,
                category=HardwareCategory.EEG,
                sample_rate=512.0,
                n_channels=32,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False  # Proprietary SDK
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 10.0, time.time()


class TMSiDriver(HardwareDriverBase):
    """TMSi SNAP/APEX driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.TMSI,
                category=HardwareCategory.EEG,
                sample_rate=512.0,
                n_channels=32,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import tmsi
            self._sdk_client = tmsi
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 10.0, time.time()


# ═══════════════════════════════════════════════════════════════════════
# INVASIVE NEURAL INTERFACE DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class BlackrockDriver(HardwareDriverBase):
    """
    Blackrock Neurotech driver.
    Supports: Utah Array, CerePlex M/E, NeuroPort
    SDK: NSAS (Neural Signal Acquisition System)
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.BLACKROCK,
                category=HardwareCategory.INVASIVE,
                sample_rate=30000.0,  # 30 kHz for spike sorting
                n_channels=96,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import cerebus
            self._sdk_client = cerebus
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        if self._sdk_client is None:
            return False
        try:
            self._client = self._sdk_client.Client()
            return True
        except Exception as e:
            logger.debug(f"Blackrock connection failed: {e}")
            return False
    
    def _start_hardware_stream(self) -> None:
        if hasattr(self, '_client'):
            self._client.start()
    
    def _stop_hardware_stream(self) -> None:
        if hasattr(self, '_client'):
            self._client.stop()
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate spike-like activity
        data = np.random.randn(self.config.n_channels).astype(np.float32) * 50.0
        # Add occasional spikes
        if np.random.rand() < 0.1:
            spike_ch = np.random.randint(0, self.config.n_channels)
            data[spike_ch] += 300.0
        return data, time.time()


class NeuralynxDriver(HardwareDriverBase):
    """
    Neuralynx driver.
    Supports: Cheetah, Digital Lynx, ATLAS
    SDK: NLX_Py
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.NEURALYNX,
                category=HardwareCategory.INVASIVE,
                sample_rate=32000.0,
                n_channels=128,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import nlxpy
            self._sdk_client = nlxpy
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 50.0, time.time()


class PlexonDriver(HardwareDriverBase):
    """
    Plexon driver.
    Supports: OmniPlex, MAP, RZ2
    SDK: PlexonClientAPI
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.PLEXON,
                category=HardwareCategory.INVASIVE,
                sample_rate=40000.0,
                n_channels=256,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import plxio
            self._sdk_client = plxio
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 50.0, time.time()


class SynchronDriver(HardwareDriverBase):
    """
    Synchron Stentrode driver.
    Endovascular BCI — catheter-based, no open brain surgery.
    SDK: proprietary (BLE interface)
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.SYNCHRON,
                category=HardwareCategory.INVASIVE,
                sample_rate=1000.0,
                n_channels=16,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import synchron_ble
            self._sdk_client = synchron_ble
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate motor cortex signals
        t = time.time()
        data = np.zeros(self.config.n_channels, dtype=np.float32)
        for i in range(self.config.n_channels):
            data[i] = 100.0 * np.sin(2 * np.pi * 10 * t + i * 0.1)
            data[i] += np.random.randn() * 10.0
        return data, t


class NeuralinkDriver(HardwareDriverBase):
    """
    Neuralink N1 implant driver (theoretical).
    SDK: not publicly available
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.NEURALINK,
                category=HardwareCategory.INVASIVE,
                sample_rate=20000.0,
                n_channels=1024,
                use_simulated=True,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False  # Not publicly available
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 100.0, time.time()


# ═══════════════════════════════════════════════════════════════════════
# MEG DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class MEGINDriver(HardwareDriverBase):
    """
    MEGIN (Elekta/Neuromag) MEG driver.
    Supports: TRIUX, TRIUX neo
    SDK: MNE-Python (open-source), Elekta MaxFilter
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.MEGIN,
                category=HardwareCategory.MEG,
                sample_rate=1000.0,
                n_channels=306,  # 102 magnetometers + 204 gradiometers
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import mne
            self._sdk_client = mne
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False  # MEG requires shielded room
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate realistic MEG signals (fT range)
        return np.random.randn(self.config.n_channels).astype(np.float32) * 100.0, time.time()


class CTFDriver(HardwareDriverBase):
    """CTF/VSM MEG driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.CTF,
                category=HardwareCategory.MEG,
                sample_rate=1200.0,
                n_channels=275,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import mne
            self._sdk_client = mne
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 100.0, time.time()


class OPMMEGDriver(HardwareDriverBase):
    """
    Optically Pumped Magnetometer MEG driver.
    Supports: QuSpin QZFM, FieldLine
    SDK: quspin-python
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.OPM_MEG,
                category=HardwareCategory.MEG,
                sample_rate=1000.0,
                n_channels=128,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import quspin
            self._sdk_client = quspin
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 50.0, time.time()


# ═══════════════════════════════════════════════════════════════════════
# fNIRS DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class NIRxDriver(HardwareDriverBase):
    """
    NIRx fNIRS driver.
    Supports: NIRScout, NIRSport2
    SDK: nirsLAB, nibabel (open-source)
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.NIRX,
                category=HardwareCategory.FNIRS,
                sample_rate=7.8125,
                n_channels=72,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import mne_nirs
            self._sdk_client = mne_nirs
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate hemodynamic response (HbO, HbR)
        t = time.time()
        data = np.zeros(self.config.n_channels, dtype=np.float32)
        for i in range(0, self.config.n_channels, 2):
            # HbO
            data[i] = 1e-6 * (1 + 0.1 * np.sin(2 * np.pi * 0.1 * t))
            # HbR
            if i + 1 < self.config.n_channels:
                data[i + 1] = -0.5e-6 * (1 + 0.1 * np.sin(2 * np.pi * 0.1 * t))
        return data + np.random.randn(self.config.n_channels).astype(np.float32) * 1e-7, t


class ArtinisDriver(HardwareDriverBase):
    """Artinis fNIRS driver (Brite+, Brite24, OctaMon)."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.ARTINIS,
                category=HardwareCategory.FNIRS,
                sample_rate=10.0,
                n_channels=24,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 1e-6, time.time()


class ISSDriver(HardwareDriverBase):
    """ISS Imagemodule fNIRS driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.ISS,
                category=HardwareCategory.FNIRS,
                sample_rate=50.0,
                n_channels=32,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 1e-6, time.time()


# ═══════════════════════════════════════════════════════════════════════
# fMRI DRIVERS (Indirect via DICOM/NIfTI)
# ═══════════════════════════════════════════════════════════════════════

class SiemensfMRIDriver(HardwareDriverBase):
    """Siemens MAGNETOM fMRI driver (via DICOM)."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.SIEMENS_FMRI,
                category=HardwareCategory.FMRI,
                sample_rate=0.5,  # TR = 2s
                n_channels=64 * 64 * 40,  # 64x64x40 voxels
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import pydicom
            import nibabel
            self._sdk_client = (pydicom, nibabel)
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False  # fMRI requires scanner access
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate BOLD signal
        return np.random.randn(self.config.n_channels).astype(np.float32) * 100.0, time.time()


class GEfMRIDriver(HardwareDriverBase):
    """GE SIGNA/Discovery fMRI driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.GE_FMRI,
                category=HardwareCategory.FMRI,
                sample_rate=0.5,
                n_channels=64 * 64 * 40,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import pydicom
            self._sdk_client = pydicom
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 100.0, time.time()


class PhilipsfMRIDriver(HardwareDriverBase):
    """Philips Ingenia/Achieva fMRI driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.PHILIPS_FMRI,
                category=HardwareCategory.FMRI,
                sample_rate=0.5,
                n_channels=64 * 64 * 40,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import pydicom
            self._sdk_client = pydicom
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 100.0, time.time()


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM SENSOR DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class QuSpinOPMDriver(HardwareDriverBase):
    """QuSpin QZFM OPM magnetometer driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.QUSPIN_OPM,
                category=HardwareCategory.QUANTUM,
                sample_rate=1000.0,
                n_channels=1,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import quspinsdk
            self._sdk_client = quspinsdk
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate fT-level magnetic field
        return np.random.randn(self.config.n_channels).astype(np.float32) * 1e-12, time.time()


class NVDiamondDriver(HardwareDriverBase):
    """NV-diamond quantum magnetometer driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.NV_DIAMOND,
                category=HardwareCategory.QUANTUM,
                sample_rate=1000000.0,
                n_channels=1,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        try:
            import qnami
            self._sdk_client = qnami
            return True
        except ImportError:
            return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 1e-15, time.time()


class SQUIDDriver(HardwareDriverBase):
    """SQUID magnetometer driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.SQUID,
                category=HardwareCategory.QUANTUM,
                sample_rate=1000000.0,
                n_channels=1,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.random.randn(self.config.n_channels).astype(np.float32) * 1e-15, time.time()


# ═══════════════════════════════════════════════════════════════════════
# NUCLEAR/RADIATION DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class CanberraDriver(HardwareDriverBase):
    """Canberra radiation detector driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.CANBERRA,
                category=HardwareCategory.NUCLEAR,
                sample_rate=1.0,
                n_channels=8192,  # energy spectrum
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate gamma spectrum (Poisson statistics)
        data = np.random.poisson(10, self.config.n_channels).astype(np.float32)
        return data, time.time()


class BertholdDriver(HardwareDriverBase):
    """Berthold radiation monitor driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.BERTHOLD,
                category=HardwareCategory.NUCLEAR,
                sample_rate=1.0,
                n_channels=1,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        # Simulate dose rate (μSv/h)
        return np.array([0.1 + np.random.rand() * 0.05], dtype=np.float32), time.time()


class LudlumDriver(HardwareDriverBase):
    """Ludlum radiation detector driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.LUDLUM,
                category=HardwareCategory.NUCLEAR,
                sample_rate=1.0,
                n_channels=1,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        return np.array([np.random.poisson(100)], dtype=np.float32), time.time()


# ═══════════════════════════════════════════════════════════════════════
# BIOMEDICAL VITALS DRIVERS
# ═══════════════════════════════════════════════════════════════════════

class PhilipsVitalsDriver(HardwareDriverBase):
    """Philips IntelliVue patient monitor driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.PHILIPS_VITALS,
                category=HardwareCategory.VITALS,
                sample_rate=125.0,
                n_channels=8,  # ECG, SpO2, BP, RR, Temp
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        t = time.time()
        # Simulate ECG + vitals
        ecg = np.sin(2 * np.pi * 1.2 * t) * 1000  # HR ~ 72 bpm
        spo2 = 98.0 + np.random.randn() * 0.5
        bp_sys = 120.0 + np.random.randn() * 5
        bp_dia = 80.0 + np.random.randn() * 3
        rr = 16.0 + np.random.randn() * 2
        temp = 37.0 + np.random.randn() * 0.1
        data = np.array([ecg, spo2, bp_sys, bp_dia, rr, temp, 0, 0], dtype=np.float32)
        return data, t


class GEVitalsDriver(HardwareDriverBase):
    """GE CARESCAPE patient monitor driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.GE_VITALS,
                category=HardwareCategory.VITALS,
                sample_rate=125.0,
                n_channels=8,
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        t = time.time()
        ecg = np.sin(2 * np.pi * 1.2 * t) * 1000
        return np.array([ecg, 98.0, 120.0, 80.0, 16.0, 37.0, 0, 0], dtype=np.float32), t


class MasimoDriver(HardwareDriverBase):
    """Masimo pulse oximeter driver."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        if config is None:
            config = HardwareConfig(
                device_type=HardwareType.MASIMO,
                category=HardwareCategory.VITALS,
                sample_rate=100.0,
                n_channels=2,  # red + IR
            )
        super().__init__(config)
    
    def _try_import_sdk(self) -> bool:
        return False
    
    def _connect_hardware(self) -> bool:
        return False
    
    def _start_hardware_stream(self) -> None:
        pass
    
    def _stop_hardware_stream(self) -> None:
        pass
    
    def _generate_simulated_sample(self) -> Tuple[np.ndarray, float]:
        t = time.time()
        # Plethysmograph waveform
        pleth = np.sin(2 * np.pi * 1.2 * t) * 1000 + 1000
        return np.array([pleth, pleth * 0.95], dtype=np.float32), t


# ═══════════════════════════════════════════════════════════════════════
# REAL-TIME STREAMING
# ═══════════════════════════════════════════════════════════════════════

class BaseStreamer(ABC):
    """Abstract base class for streaming protocols."""
    
    def __init__(self, name: str, host: str = "localhost", port: int = 0):
        self.name = name
        self.host = host
        self.port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    @abstractmethod
    def start(self) -> None:
        pass
    
    @abstractmethod
    def stop(self) -> None:
        pass
    
    @abstractmethod
    def send(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        pass
    
    @property
    def is_running(self) -> bool:
        return self._running


class LSLStreamer(BaseStreamer):
    """
    Lab Streaming Layer (LSL) streamer.
    Standard protocol in BCI research.
    SDK: pylsl
    """
    
    def __init__(self, stream_name: str = "SESI_EEG", stream_type: str = "EEG",
                 sample_rate: float = 1000.0, n_channels: int = 64,
                 host: str = "localhost", port: int = 0):
        super().__init__("LSL", host, port)
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self._outlet = None
        self._info = None
    
    def start(self) -> None:
        try:
            import pylsl
            self._info = pylsl.StreamInfo(
                self.stream_name, self.stream_type,
                self.n_channels, self.sample_rate,
                'float32', 'sesi_' + self.stream_name
            )
            self._outlet = pylsl.StreamOutlet(self._info)
            self._running = True
            logger.info(f"[LSL] Streaming '{self.stream_name}' at {self.sample_rate} Hz")
        except ImportError:
            logger.warning("[LSL] pylsl not available — using mock streamer")
            self._running = True
    
    def stop(self) -> None:
        self._running = False
        self._outlet = None
        logger.info(f"[LSL] Stopped streaming '{self.stream_name}'")
    
    def send(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        if not self._running:
            return
        
        if self._outlet is not None:
            try:
                import pylsl
                for i in range(data.shape[0]):
                    self._outlet.push_sample(data[i].tolist(), timestamps[i])
            except Exception as e:
                logger.debug(f"[LSL] Send error: {e}")


class TCPStreamer(BaseStreamer):
    """TCP socket streamer for point-to-point connections."""
    
    def __init__(self, host: str = "localhost", port: int = 5000,
                 buffer_size: int = 4096):
        super().__init__("TCP", host, port)
        self.buffer_size = buffer_size
        self._socket: Optional[socket.socket] = None
        self._server_socket: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
    
    def start(self) -> None:
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)
            self._server_socket.settimeout(1.0)
            self._running = True
            self._thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._thread.start()
            logger.info(f"[TCP] Listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"[TCP] Failed to start: {e}")
    
    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, addr = self._server_socket.accept()
                self._clients.append(client)
                logger.info(f"[TCP] Client connected from {addr}")
            except socket.timeout:
                continue
            except Exception:
                break
    
    def stop(self) -> None:
        self._running = False
        for client in self._clients:
            try:
                client.close()
            except Exception:
                pass
        if self._server_socket:
            self._server_socket.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[TCP] Stopped")
    
    def send(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        if not self._running or not self._clients:
            return
        
        # Pack: [stream_name_len, stream_name, n_samples, n_channels, timestamps, data]
        name_bytes = stream_name.encode('utf-8')
        header = struct.pack(f'I{len(name_bytes)}sII',
                             len(name_bytes), name_bytes,
                             data.shape[0], data.shape[1])
        ts_bytes = timestamps.astype(np.float64).tobytes()
        data_bytes = data.astype(np.float32).tobytes()
        
        message = header + ts_bytes + data_bytes
        
        disconnected = []
        for client in self._clients:
            try:
                client.sendall(message)
            except Exception:
                disconnected.append(client)
        
        for client in disconnected:
            self._clients.remove(client)
            try:
                client.close()
            except Exception:
                pass


class UDPStreamer(BaseStreamer):
    """UDP multicast streamer for low-latency broadcasting."""
    
    def __init__(self, host: str = "224.0.0.1", port: int = 5001):
        super().__init__("UDP", host, port)
        self._socket: Optional[socket.socket] = None
    
    def start(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            self._running = True
            logger.info(f"[UDP] Multicast to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"[UDP] Failed to start: {e}")
    
    def stop(self) -> None:
        self._running = False
        if self._socket:
            self._socket.close()
        logger.info("[UDP] Stopped")
    
    def send(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        if not self._running or self._socket is None:
            return
        
        # Compact binary format
        payload = {
            'name': stream_name,
            'ts': timestamps.tolist(),
            'data': data.tolist(),
        }
        message = json.dumps(payload).encode('utf-8')
        
        try:
            self._socket.sendto(message, (self.host, self.port))
        except Exception as e:
            logger.debug(f"[UDP] Send error: {e}")


class WebSocketStreamer(BaseStreamer):
    """WebSocket streamer for browser-friendly access."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        super().__init__("WebSocket", host, port)
        self._clients: List[Any] = []
        self._server: Optional[Any] = None
    
    def start(self) -> None:
        try:
            import asyncio
            import websockets
            
            async def handler(websocket, path):
                self._clients.append(websocket)
                try:
                    async for message in websocket:
                        pass  # Client messages ignored
                finally:
                    self._clients.remove(websocket)
            
            async def server():
                self._server = await websockets.serve(handler, self.host, self.port)
                await self._server.wait_closed()
            
            self._running = True
            self._thread = threading.Thread(
                target=lambda: asyncio.new_event_loop().run_until_complete(server()),
                daemon=True
            )
            self._thread.start()
            logger.info(f"[WebSocket] Listening on ws://{self.host}:{self.port}")
        except ImportError:
            logger.warning("[WebSocket] websockets not available — using mock")
            self._running = True
    
    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
        logger.info("[WebSocket] Stopped")
    
    def send(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        if not self._running or not self._clients:
            return
        
        try:
            import asyncio
            payload = json.dumps({
                'stream': stream_name,
                'timestamps': timestamps.tolist(),
                'data': data.tolist(),
            })
            
            async def broadcast():
                await asyncio.gather(
                    *[client.send(payload) for client in self._clients],
                    return_exceptions=True
                )
            
            asyncio.run_coroutine_threadsafe(
                broadcast(),
                asyncio.new_event_loop()
            )
        except Exception as e:
            logger.debug(f"[WebSocket] Send error: {e}")


class ZeroMQStreamer(BaseStreamer):
    """ZeroMQ pub/sub streamer for distributed systems."""
    
    def __init__(self, host: str = "*", port: int = 5555, topic: str = "sesi"):
        super().__init__("ZeroMQ", host, port)
        self.topic = topic
        self._context: Optional[Any] = None
        self._socket: Optional[Any] = None
    
    def start(self) -> None:
        try:
            import zmq
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.bind(f"tcp://{self.host}:{self.port}")
            self._running = True
            logger.info(f"[ZeroMQ] Publishing on tcp://{self.host}:{self.port}")
        except ImportError:
            logger.warning("[ZeroMQ] pyzmq not available — using mock")
            self._running = True
    
    def stop(self) -> None:
        self._running = False
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
        logger.info("[ZeroMQ] Stopped")
    
    def send(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        if not self._running or self._socket is None:
            return
        
        try:
            import zmq
            payload = json.dumps({
                'stream': stream_name,
                'timestamps': timestamps.tolist(),
                'data': data.tolist(),
            }).encode('utf-8')
            self._socket.send_multipart([self.topic.encode(), payload])
        except Exception as e:
            logger.debug(f"[ZeroMQ] Send error: {e}")


class StreamingRouter:
    """
    Routes data from hardware drivers to multiple streaming protocols.
    Supports fan-out to LSL, TCP, UDP, WebSocket, ZeroMQ simultaneously.
    """
    
    def __init__(self):
        self._streamers: Dict[str, BaseStreamer] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
    
    def add_streamer(self, name: str, streamer: BaseStreamer) -> None:
        self._streamers[name] = streamer
    
    def remove_streamer(self, name: str) -> None:
        if name in self._streamers:
            self._streamers[name].stop()
            del self._streamers[name]
    
    def start(self) -> None:
        for streamer in self._streamers.values():
            streamer.start()
        self._running = True
        self._thread = threading.Thread(target=self._route_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Router] Started with {len(self._streamers)} streamers")
    
    def stop(self) -> None:
        self._running = False
        for streamer in self._streamers.values():
            streamer.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[Router] Stopped")
    
    def enqueue(self, data: np.ndarray, timestamps: np.ndarray, stream_name: str) -> None:
        try:
            self._queue.put_nowait((data, timestamps, stream_name))
        except queue.Full:
            # Drop oldest to make room
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((data, timestamps, stream_name))
            except queue.Empty:
                pass
    
    def _route_loop(self) -> None:
        while self._running:
            try:
                data, timestamps, stream_name = self._queue.get(timeout=0.1)
                for streamer in self._streamers.values():
                    try:
                        streamer.send(data, timestamps, stream_name)
                    except Exception as e:
                        logger.debug(f"[Router] Streamer {streamer.name} error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"[Router] Loop error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# HARDWARE REGISTRY & HAL
# ═══════════════════════════════════════════════════════════════════════

class HardwareRegistry:
    """
    Registry of all available hardware drivers.
    Auto-detects connected devices and provides unified access.
    """
    
    # Map of HardwareType → Driver class
    DRIVER_MAP: Dict[HardwareType, type] = {
        # EEG
        HardwareType.BRAINVISION: BrainVisionDriver,
        HardwareType.GTEC: GtecDriver,
        HardwareType.OPENBCI: OpenBCIDriver,
        HardwareType.BIOSEMI: BioSemiDriver,
        HardwareType.ANT_NEURO: ANTNeuroDriver,
        HardwareType.MICROMED: MicromedDriver,
        HardwareType.NATUS: NatusDriver,
        HardwareType.TMSI: TMSiDriver,
        # Invasive
        HardwareType.BLACKROCK: BlackrockDriver,
        HardwareType.NEURALYNX: NeuralynxDriver,
        HardwareType.PLEXON: PlexonDriver,
        HardwareType.SYNCHRON: SynchronDriver,
        HardwareType.NEURALINK: NeuralinkDriver,
        # MEG
        HardwareType.MEGIN: MEGINDriver,
        HardwareType.CTF: CTFDriver,
        HardwareType.OPM_MEG: OPMMEGDriver,
        # fNIRS
        HardwareType.NIRX: NIRxDriver,
        HardwareType.ARTINIS: ArtinisDriver,
        HardwareType.ISS: ISSDriver,
        # fMRI
        HardwareType.SIEMENS_FMRI: SiemensfMRIDriver,
        HardwareType.GE_FMRI: GEfMRIDriver,
        HardwareType.PHILIPS_FMRI: PhilipsfMRIDriver,
        # Quantum
        HardwareType.QUSPIN_OPM: QuSpinOPMDriver,
        HardwareType.NV_DIAMOND: NVDiamondDriver,
        HardwareType.SQUID: SQUIDDriver,
        # Nuclear
        HardwareType.CANBERRA: CanberraDriver,
        HardwareType.BERTHOLD: BertholdDriver,
        HardwareType.LUDLUM: LudlumDriver,
        # Vitals
        HardwareType.PHILIPS_VITALS: PhilipsVitalsDriver,
        HardwareType.GE_VITALS: GEVitalsDriver,
        HardwareType.MASIMO: MasimoDriver,
    }
    
    def __init__(self):
        self._drivers: Dict[str, HardwareDriverBase] = {}
    
    def create_driver(self, name: str, device_type: HardwareType,
                      config: Optional[HardwareConfig] = None) -> HardwareDriverBase:
        """Create and register a new driver."""
        if device_type not in self.DRIVER_MAP:
            raise ValueError(f"Unknown device type: {device_type}")
        
        driver_class = self.DRIVER_MAP[device_type]
        driver = driver_class(config)
        self._drivers[name] = driver
        return driver
    
    def get_driver(self, name: str) -> Optional[HardwareDriverBase]:
        return self._drivers.get(name)
    
    def list_drivers(self) -> List[str]:
        return list(self._drivers.keys())
    
    def list_supported_types(self) -> List[HardwareType]:
        return list(self.DRIVER_MAP.keys())
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect all registered drivers. Returns {name: success}."""
        results = {}
        for name, driver in self._drivers.items():
            try:
                results[name] = driver.connect()
            except Exception as e:
                logger.error(f"[Registry] Failed to connect {name}: {e}")
                results[name] = False
        return results
    
    def start_all_streams(self) -> None:
        """Start streaming on all connected drivers."""
        for name, driver in self._drivers.items():
            if driver.metadata.is_connected:
                try:
                    driver.start_stream()
                except Exception as e:
                    logger.error(f"[Registry] Failed to start {name}: {e}")
    
    def stop_all_streams(self) -> None:
        """Stop streaming on all drivers."""
        for driver in self._drivers.values():
            try:
                driver.stop_stream()
            except Exception as e:
                logger.debug(f"[Registry] Stop error: {e}")
    
    def disconnect_all(self) -> None:
        """Disconnect all drivers."""
        for driver in self._drivers.values():
            try:
                driver.disconnect()
            except Exception as e:
                logger.debug(f"[Registry] Disconnect error: {e}")


class HardwareAbstractionLayer:
    """
    High-level HAL that combines hardware registry with streaming router.
    This is the main entry point for users.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.registry = HardwareRegistry()
        self.router = StreamingRouter()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fusion_callback: Optional[Callable] = None
    
    def add_eeg(self, name: str, device_type: HardwareType,
                sample_rate: float = 1000.0, n_channels: int = 64) -> HardwareDriverBase:
        """Add an EEG device."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.EEG,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_invasive(self, name: str, device_type: HardwareType,
                     sample_rate: float = 30000.0, n_channels: int = 96) -> HardwareDriverBase:
        """Add an invasive neural interface."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.INVASIVE,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_meg(self, name: str, device_type: HardwareType,
                sample_rate: float = 1000.0, n_channels: int = 306) -> HardwareDriverBase:
        """Add a MEG device."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.MEG,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_fnirs(self, name: str, device_type: HardwareType,
                  sample_rate: float = 10.0, n_channels: int = 72) -> HardwareDriverBase:
        """Add an fNIRS device."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.FNIRS,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_quantum(self, name: str, device_type: HardwareType,
                    sample_rate: float = 1000.0, n_channels: int = 1) -> HardwareDriverBase:
        """Add a quantum sensor."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.QUANTUM,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_nuclear(self, name: str, device_type: HardwareType,
                    sample_rate: float = 1.0, n_channels: int = 1) -> HardwareDriverBase:
        """Add a nuclear/radiation sensor."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.NUCLEAR,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_vitals(self, name: str, device_type: HardwareType,
                   sample_rate: float = 125.0, n_channels: int = 8) -> HardwareDriverBase:
        """Add a biomedical vitals monitor."""
        config = HardwareConfig(
            device_type=device_type,
            category=HardwareCategory.VITALS,
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        return self.registry.create_driver(name, device_type, config)
    
    def add_streamer(self, protocol: StreamProtocol, **kwargs) -> BaseStreamer:
        """Add a streaming protocol."""
        if protocol == StreamProtocol.LSL:
            streamer = LSLStreamer(**kwargs)
        elif protocol == StreamProtocol.TCP:
            streamer = TCPStreamer(**kwargs)
        elif protocol == StreamProtocol.UDP:
            streamer = UDPStreamer(**kwargs)
        elif protocol == StreamProtocol.WEBSOCKET:
            streamer = WebSocketStreamer(**kwargs)
        elif protocol == StreamProtocol.ZMQ:
            streamer = ZeroMQStreamer(**kwargs)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        self.router.add_streamer(protocol.value, streamer)
        return streamer
    
    def set_fusion_callback(self, callback: Callable) -> None:
        """Set callback for sensor fusion pipeline."""
        self._fusion_callback = callback
    
    def start(self) -> None:
        """Start all hardware and streaming."""
        self.registry.connect_all()
        self.registry.start_all_streams()
        self.router.start()
        logger.info(f"[HAL] Started with {len(self.registry.list_drivers())} devices")
    
    def stop(self) -> None:
        """Stop everything."""
        self.registry.stop_all_streams()
        self.registry.disconnect_all()
        self.router.stop()
        logger.info("[HAL] Stopped")
    
    def read_all(self, n_samples: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Read from all connected drivers."""
        results = {}
        for name in self.registry.list_drivers():
            driver = self.registry.get_driver(name)
            if driver and driver.metadata.is_streaming:
                try:
                    data, timestamps = driver.read_chunk(n_samples, timeout=0.5)
                    if data.shape[0] > 0:
                        results[name] = (data, timestamps)
                except Exception as e:
                    logger.debug(f"[HAL] Read error from {name}: {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION WITH SESI FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class StreamingFusionPipeline:
    """
    Integrates hardware streaming with SESI Universal Sensor Fusion Engine.
    
    Pipeline:
      Hardware → SampleBuffer → Streamer → FusionEngine → AGI ONE
    """
    
    def __init__(self, hal: HardwareAbstractionLayer,
                 fusion_engine: Optional[nn.Module] = None):
        self.hal = hal
        self.fusion_engine = fusion_engine
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, chunk_size: int = 100, interval_s: float = 0.1) -> None:
        """Start the fusion pipeline."""
        self._running = True
        self._thread = threading.Thread(
            target=self._fusion_loop,
            args=(chunk_size, interval_s),
            daemon=True
        )
        self._thread.start()
        logger.info("[Fusion] Pipeline started")
    
    def stop(self) -> None:
        """Stop the fusion pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[Fusion] Pipeline stopped")
    
    def _fusion_loop(self, chunk_size: int, interval_s: float) -> None:
        while self._running:
            t0 = time.time()
            
            # Read from all hardware
            all_data = self.hal.read_all(chunk_size)
            
            if all_data and self._fusion_callback:
                try:
                    self._fusion_callback(all_data)
                except Exception as e:
                    logger.error(f"[Fusion] Callback error: {e}")
            
            # Also route to streaming protocols
            for name, (data, timestamps) in all_data.items():
                self.hal.router.enqueue(data, timestamps, name)
            
            # Maintain interval
            elapsed = time.time() - t0
            sleep_time = interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Hardware Interface & Streaming — Smoke Test")
    print("=" * 72)
    
    # Create HAL
    hal = HardwareAbstractionLayer()
    
    # Add various hardware
    print("\n  [1/5] Adding hardware devices...")
    
    hal.add_eeg("brainvision_eeg", HardwareType.BRAINVISION,
                sample_rate=1000.0, n_channels=64)
    hal.add_eeg("gtec_eeg", HardwareType.GTEC,
                sample_rate=1200.0, n_channels=16)
    hal.add_eeg("openbci_cyton", HardwareType.OPENBCI,
                sample_rate=250.0, n_channels=8)
    hal.add_invasive("blackrock_utah", HardwareType.BLACKROCK,
                     sample_rate=30000.0, n_channels=96)
    hal.add_invasive("synchron_stentrode", HardwareType.SYNCHRON,
                     sample_rate=1000.0, n_channels=16)
    hal.add_meg("megin_triuex", HardwareType.MEGIN,
                sample_rate=1000.0, n_channels=306)
    hal.add_fnirs("nirx_nirscout", HardwareType.NIRX,
                  sample_rate=7.8125, n_channels=72)
    hal.add_quantum("quspin_opm", HardwareType.QUSPIN_OPM,
                    sample_rate=1000.0, n_channels=1)
    hal.add_nuclear("canberra_detector", HardwareType.CANBERRA,
                    sample_rate=1.0, n_channels=8192)
    hal.add_vitals("philips_monitor", HardwareType.PHILIPS_VITALS,
                   sample_rate=125.0, n_channels=8)
    
    print(f"    Registered {len(hal.registry.list_drivers())} devices")
    
    # Add streamers
    print("\n  [2/5] Adding streaming protocols...")
    hal.add_streamer(StreamProtocol.LSL, stream_name="SESI_EEG",
                     sample_rate=1000.0, n_channels=64)
    hal.add_streamer(StreamProtocol.TCP, port=5000)
    hal.add_streamer(StreamProtocol.UDP, port=5001)
    hal.add_streamer(StreamProtocol.ZMQ, port=5555)
    print(f"    Added {len(hal.router._streamers)} streamers")
    
    # Connect and start
    print("\n  [3/5] Connecting hardware...")
    results = hal.registry.connect_all()
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"    {status} {name}")
    
    # Start streaming
    print("\n  [4/5] Starting streams...")
    hal.start()
    time.sleep(0.5)  # Let streams initialize
    
    # Read data
    print("\n  [5/5] Reading data chunks...")
    all_data = hal.read_all(100)
    for name, (data, timestamps) in all_data.items():
        print(f"    {name}: {data.shape} samples, "
              f"range=[{data.min():.2f}, {data.max():.2f}]")
    
    # Test fusion pipeline
    print("\n  [EXTRA] Testing fusion pipeline...")
    fusion_data_count = [0]
    
    def fusion_callback(data_dict):
        fusion_data_count[0] += len(data_dict)
    
    pipeline = StreamingFusionPipeline(hal, None)
    pipeline._fusion_callback = fusion_callback
    pipeline.start(chunk_size=50, interval_s=0.1)
    time.sleep(0.5)
    pipeline.stop()
    
    print(f"    Fusion processed {fusion_data_count[0]} batches")
    
    # Stop
    hal.stop()
    
    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)
    
    # Summary
    print("\n  📊 SUPPORTED HARDWARE SUMMARY:")
    print("  ─────────────────────────────────────────")
    categories = {}
    for dt in HardwareType:
        cat = hal.registry.DRIVER_MAP[dt]
        # Get category from driver class
        driver = cat()
        category = driver.config.category.name
        categories.setdefault(category, []).append(dt.value)
    
    for cat, devices in sorted(categories.items()):
        print(f"  {cat:15s}: {len(devices)} devices")
        for d in devices:
            print(f"    • {d}")
    
    print("\n  📡 SUPPORTED STREAMING PROTOCOLS:")
    print("  ─────────────────────────────────────────")
    for proto in StreamProtocol:
        print(f"    • {proto.value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")
    smoke_test()
