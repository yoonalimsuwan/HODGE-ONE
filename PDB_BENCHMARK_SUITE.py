"""
PDB BENCHMARK SUITE v1.0
Real-World Protein Structure Validation for PROTEIN_FOLDING_ORCHESTRATOR
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
AI Co-Developers: Claude (Anthropic), GPT (OpenAI), Gemini (Google)
=============================================================================
PURPOSE
-------
ทดสอบ PROTEIN_FOLDING_ORCHESTRATOR กับโครงสร้างโปรตีนจริงจาก PDB
วัดคุณภาพด้วย RMSD (Kabsch), TM-score, GDT-TS และวิเคราะห์ Pareto frontier

HONEST SCOPE & CRITICAL DISCLAIMERS
------------------------------------
• นี่คือ BENCHMARK FRAMEWORK ไม่ใช่ production structure predictor
• AlphaFold2 / ESMFold / RoseTTAFold ยังคงเป็น empirical baseline ที่ควรเทียบ
• PDB fetch ต้องใช้ internet connection (มี offline fallback แบบ curated)
• โมดูลใน orchestrator (Level1-3) เป็น quantum-inspired / heuristic — 
  ไม่ได้แข่งขันกับ AlphaFold2 ในแง่ accuracy จริง
• ผลลัพธ์ใช้สำหรับ validate pipeline ต่างๆ ใน ecosystem เท่านั้น

MATHEMATICAL FOUNDATIONS
-------------------------
[M1] Kabsch RMSD:
  RMSD = sqrt( (1/N) * sum_i ||x_i - R·y_i - t||² )
  R = V · diag(1,1,det(VUᵀ)) · Uᵀ   (from SVD of H = XᵀY)
[M2] TM-score (Zhang & Skolnick 2004):
  TM = (1/L) * sum_i 1 / (1 + (d_i/d₀)²)
  d₀ = 1.24 * (L - 15)^(1/3) - 1.8
[M3] GDT-TS:
  GDT = (1/4) * (P₁ + P₂ + P₄ + P₈)
  P_c = fraction of residues with d_i < c Å
=============================================================================
"""
from __future__ import annotations
import logging
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("pdb_benchmark")

__all__ = [
    "PDBBenchmarkConfig",
    "PDBStructure",
    "PDBFetcher",
    "StructureMetrics",
    "PDBBenchmarkRunner",
    "BenchmarkReport",
    "CURATED_PDB_SET",
]

# ============================================================================
# Optional dependencies (graceful fallback)
# ============================================================================
try:
    from Bio.PDB import PDBParser, MMCIFParser, Polypeptide
    from Bio.SeqUtils import seq1
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False
    PDBParser = None
    MMCIFParser = None
    Polypeptide = None
    seq1 = None
    warnings.warn("Biopython not found — PDB parsing will use fallback REST API.")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    requests = None

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None

# ============================================================================
# 0. CONFIG
# ============================================================================
@dataclass
class PDBBenchmarkConfig:
    """Configuration for PDB benchmark suite."""
    # --- Dataset ---
    pdb_ids: List[str] = field(default_factory=lambda: list(CURATED_PDB_SET.keys()))
    cache_dir: str = "./pdb_cache"
    use_offline_fallback: bool = True  # use curated offline data if fetch fails
    
    # --- Metrics ---
    rmsd_convergence_tol: float = 1e-6
    tm_score_d0_formula: str = "zhang2004"  # or "fixed"
    gdt_cutoffs: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    
    # --- Orchestrator integration ---
    max_sequence_length: int = 1000  # truncate for smoke tests
    run_all_levels: bool = True
    
    # --- Pareto analysis ---
    reference_point: Tuple[float, float, float, float] = (1.0, 100.0, 1000.0, 100.0)
    
    # --- Device ---
    device: str = "auto"
    seed: int = 42
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 1. CURATED PDB DATASET (Offline fallback)
# ============================================================================
# คัดเลือกโปรตีนหลากหลายขนาดและ topology สำหรับ benchmark
# ข้อมูลนี้ได้มาจากการคำนวณ offline จาก PDB จริง
CURATED_PDB_SET = {
    # Small single-domain proteins (< 100 residues)
    "1CRN": {
        "name": "Crambin",
        "length": 46,
        "description": "Small hydrophobic plant protein, classic benchmark",
        "secondary_structure": "alpha+beta",
        "difficulty": "easy",
    },
    "1UBQ": {
        "name": "Ubiquitin",
        "length": 76,
        "description": "Ubiquitin — gold standard small protein benchmark",
        "secondary_structure": "mixed",
        "difficulty": "easy",
    },
    "1R69": {
        "name": "Lambda repressor headpiece",
        "length": 56,
        "description": "3-helix bundle, fast folder",
        "secondary_structure": "alpha",
        "difficulty": "easy",
    },
    # Medium proteins (100-300 residues)
    "2LZM": {
        "name": "T4 Lysozyme",
        "length": 164,
        "description": "Classic enzyme, mixed alpha/beta",
        "secondary_structure": "mixed",
        "difficulty": "medium",
    },
    "1TIM": {
        "name": "Triosephosphate isomerase (monomer)",
        "length": 248,
        "description": "TIM barrel — classic fold",
        "secondary_structure": "alpha+beta",
        "difficulty": "medium",
    },
    "3INS": {
        "name": "Insulin (chain A+B)",
        "length": 51 + 30,
        "description": "Two-chain hormone with disulfide bonds",
        "secondary_structure": "alpha",
        "difficulty": "medium",
    },
    # Larger / harder targets
    "1AKE": {
        "name": "Adenylate kinase",
        "length": 214,
        "description": "Two-domain enzyme with hinge motion",
        "secondary_structure": "mixed",
        "difficulty": "medium",
    },
    "7KZB": {
        "name": "SARS-CoV-2 RBD (partial)",
        "length": 200,  # truncated for testing
        "description": "Receptor binding domain — medically relevant",
        "secondary_structure": "beta-rich",
        "difficulty": "hard",
    },
}


# ============================================================================
# 2. PDB STRUCTURE DATA CLASS
# ============================================================================
@dataclass
class PDBStructure:
    """Container for parsed PDB structure."""
    pdb_id: str
    name: str
    sequence: str                    # one-letter amino acid sequence
    ca_coords: Optional[Tensor]      # (L, 3) Cα coordinates in Å
    length: int
    source: str                      # "biopython", "rest_api", "curated", "synthetic"
    resolution: Optional[float] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.ca_coords is not None:
            d["ca_coords_shape"] = tuple(self.ca_coords.shape)
            d["ca_coords"] = None  # don't dump full tensor
        return d


# ============================================================================
# 3. PDB FETCHER (with multiple fallbacks)
# ============================================================================
class PDBFetcher:
    """
    Fetch and parse PDB structures with multiple fallback strategies:
      1. Biopython + local file (if already downloaded)
      2. RCSB PDB REST API (download PDB file, parse with Biopython)
      3. RCSB PDB structure API (JSON endpoint — no Biopython needed)
      4. Curated offline fallback (synthetic CA trace for known PDB IDs)
    """
    RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
    RCSB_API_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    def __init__(self, cfg: PDBBenchmarkConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(self, pdb_id: str) -> PDBStructure:
        """Fetch a PDB structure with automatic fallback."""
        pdb_id = pdb_id.upper()
        logger.info(f"Fetching PDB {pdb_id}...")
        
        # Strategy 1: local cached file + Biopython
        local_path = self.cache_dir / f"{pdb_id.lower()}.pdb"
        if _HAS_BIOPYTHON and local_path.exists():
            try:
                return self._parse_with_biopython(pdb_id, local_path)
            except Exception as e:
                logger.warning(f"Biopython parse of cached file failed: {e}")
        
        # Strategy 2: download from RCSB + Biopython
        if _HAS_BIOPYTHON and _HAS_REQUESTS:
            try:
                self._download_pdb(pdb_id, local_path)
                return self._parse_with_biopython(pdb_id, local_path)
            except Exception as e:
                logger.warning(f"RCSB download failed: {e}")
        
        # Strategy 3: RCSB JSON API (no Biopython needed)
        if _HAS_REQUESTS:
            try:
                return self._fetch_from_rcsb_api(pdb_id)
            except Exception as e:
                logger.warning(f"RCSB API fetch failed: {e}")
        
        # Strategy 4: curated offline fallback
        if self.cfg.use_offline_fallback and pdb_id in CURATED_PDB_SET:
            logger.warning(f"Using OFFLINE FALLBACK for {pdb_id} (synthetic CA trace)")
            return self._synthetic_fallback(pdb_id)
        
        raise RuntimeError(
            f"Could not fetch PDB {pdb_id}. "
            f"Install 'biopython' and 'requests' for full functionality, "
            f"or enable use_offline_fallback."
        )
    
    def _download_pdb(self, pdb_id: str, dest: Path) -> None:
        url = self.RCSB_PDB_URL.format(pdb_id=pdb_id)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text)
        logger.info(f"  Downloaded {pdb_id} → {dest}")
    
    def _parse_with_biopython(self, pdb_id: str, path: Path) -> PDBStructure:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(path))
        model = structure[0]
        
        # Extract Cα coordinates and sequence
        ca_coords = []
        sequence = []
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue  # skip HETATM
                if "CA" in residue:
                    ca = residue["CA"].get_coord()
                    ca_coords.append(ca)
                    sequence.append(seq1(residue.get_resname(), "X"))
        
        if not ca_coords:
            raise ValueError(f"No Cα atoms found in {pdb_id}")
        
        # Try to get resolution
        resolution = None
        try:
            header = structure.header
            resolution = header.get("resolution")
        except Exception:
            pass
        
        return PDBStructure(
            pdb_id=pdb_id,
            name=CURATED_PDB_SET.get(pdb_id, {}).get("name", pdb_id),
            sequence="".join(sequence),
            ca_coords=torch.tensor(ca_coords, dtype=torch.float32),
            length=len(ca_coords),
            source="biopython",
            resolution=resolution,
            description=CURATED_PDB_SET.get(pdb_id, {}).get("description", ""),
        )
    
    def _fetch_from_rcsb_api(self, pdb_id: str) -> PDBStructure:
        """Use RCSB JSON API when Biopython is unavailable."""
        url = self.RCSB_API_URL.format(pdb_id=pdb_id)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract sequence
        seq_entity = data.get("entity", {}).get("1", {})
        sequence = seq_entity.get("entity_poly", {}).get("pdbx_seq_one_letter", "")
        if not sequence:
            # fallback: polymer_entity
            sequence = data.get("polymer_entity", [{}])[0].get(
                "entity_poly", {}
            ).get("pdbx_seq_one_letter", "")
        
        # We cannot easily get Cα coords from JSON API without further parsing
        # Fall back to synthetic
        logger.warning(f"RCSB API returned sequence but no coords — using synthetic CA trace for {pdb_id}")
        struct = self._synthetic_fallback(pdb_id)
        struct.sequence = sequence  # override with real sequence
        struct.source = "rcsb_api_sequence_only"
        return struct
    
    def _synthetic_fallback(self, pdb_id: str) -> PDBStructure:
        """Generate a synthetic Cα trace for testing purposes."""
        info = CURATED_PDB_SET.get(pdb_id, {})
        length = info.get("length", 100)
        
        torch.manual_seed(self.cfg.seed + hash(pdb_id) % 10000)
        # Random-walk Cα trace with realistic bond lengths (~3.8 Å)
        steps = torch.randn(length, 3) * 0.3
        steps = steps / steps.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 3.8
        coords = torch.cumsum(steps, dim=0)
        
        # Synthetic sequence (just use alternating hydrophobic/polar)
        aa_pool = "ACDEFGHIKLMNPQRSTVWY"
        sequence = "".join(aa_pool[i % 20] for i in range(length))
        
        return PDBStructure(
            pdb_id=pdb_id,
            name=info.get("name", pdb_id),
            sequence=sequence,
            ca_coords=coords,
            length=length,
            source="synthetic",
            description=f"SYNTHETIC FALLBACK — {info.get('description', '')}",
        )
    
    def fetch_all(self) -> List[PDBStructure]:
        """Fetch all PDB IDs in the benchmark set."""
        results = []
        for pdb_id in self.cfg.pdb_ids:
            try:
                struct = self.fetch(pdb_id)
                if struct.length > self.cfg.max_sequence_length:
                    logger.info(f"  Truncating {pdb_id} from {struct.length} to "
                                f"{self.cfg.max_sequence_length} residues")
                    struct.sequence = struct.sequence[: self.cfg.max_sequence_length]
                    if struct.ca_coords is not None:
                        struct.ca_coords = struct.ca_coords[: self.cfg.max_sequence_length]
                    struct.length = self.cfg.max_sequence_length
                results.append(struct)
            except Exception as e:
                logger.error(f"Failed to fetch {pdb_id}: {e}")
        return results


# ============================================================================
# 4. STRUCTURE METRICS (RMSD, TM-score, GDT-TS)
# ============================================================================
class StructureMetrics:
    """
    Compute structure similarity metrics between predicted and reference.
    All implementations are pure PyTorch for autograd compatibility.
    """
    
    @staticmethod
    def kabsch_align(P: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor, float]:
        """
        Kabsch algorithm: find rotation R and translation t that aligns
        P to Q minimizing RMSD.
        
        Args:
            P: (N, 3) predicted coordinates
            Q: (N, 3) reference coordinates
        
        Returns:
            P_aligned: (N, 3) aligned predicted coords
            R: (3, 3) rotation matrix
            rmsd: float RMSD after alignment
        """
        assert P.shape == Q.shape and P.shape[1] == 3, \
            f"Shape mismatch: P={P.shape}, Q={Q.shape}"
        
        # Center both
        p_mean = P.mean(dim=0)
        q_mean = Q.mean(dim=0)
        P_c = P - p_mean
        Q_c = Q - q_mean
        
        # Cross-covariance matrix
        H = P_c.T @ Q_c  # (3, 3)
        
        # SVD
        U, S, Vh = torch.linalg.svd(H)
        
        # Handle reflection
        d = torch.det(Vh.T @ U.T)
        sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=P.device))
        
        # Optimal rotation
        R = Vh.T @ sign_matrix @ U.T
        
        # Apply rotation and translation
        P_aligned = (P_c @ R.T) + q_mean
        
        # RMSD
        rmsd = torch.sqrt(((P_aligned - Q) ** 2).sum(dim=-1).mean()).item()
        
        return P_aligned, R, rmsd
    
    @staticmethod
    def tm_score(P: Tensor, Q: Tensor, L: Optional[int] = None) -> float:
        """
        TM-score (Zhang & Skolnick 2004).
        Length-independent measure of structural similarity, ∈ (0, 1].
        TM > 0.5 generally indicates same fold.
        """
        L = L or P.shape[0]
        # d0 parameter
        d0 = 1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)  # floor for very short proteins
        
        # Align first
        P_aligned, _, _ = StructureMetrics.kabsch_align(P, Q)
        
        # Per-residue distances
        d = torch.norm(P_aligned - Q, dim=-1)
        
        # TM-score formula
        tm = (1.0 / L) * torch.sum(1.0 / (1.0 + (d / d0) ** 2))
        return tm.item()
    
    @staticmethod
    def gdt_ts(
        P: Tensor, Q: Tensor,
        cutoffs: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    ) -> float:
        """
        GDT-TS (Global Distance Test — Total Score).
        Average fraction of residues within 1, 2, 4, 8 Å.
        ∈ [0, 1], higher is better.
        """
        P_aligned, _, _ = StructureMetrics.kabsch_align(P, Q)
        d = torch.norm(P_aligned - Q, dim=-1)
        
        fractions = []
        for c in cutoffs:
            frac = (d < c).float().mean().item()
            fractions.append(frac)
        
        return sum(fractions) / len(fractions)
    
    @staticmethod
    def compute_all(
        predicted: Tensor, reference: Tensor,
        cfg: Optional[PDBBenchmarkConfig] = None,
    ) -> Dict[str, float]:
        """Compute all metrics at once."""
        cfg = cfg or PDBBenchmarkConfig()
        P_aligned, R, rmsd = StructureMetrics.kabsch_align(predicted, reference)
        tm = StructureMetrics.tm_score(P_aligned, reference)
        gdt = StructureMetrics.gdt_ts(P_aligned, reference, cfg.gdt_cutoffs)
        
        return {
            "rmsd_angstrom": rmsd,
            "tm_score": tm,
            "gdt_ts": gdt,
            "rotation_matrix": R.detach().cpu(),
        }


# ============================================================================
# 5. BENCHMARK REPORT
# ============================================================================
@dataclass
class BenchmarkReport:
    """Report for a single PDB benchmark run."""
    pdb_id: str
    name: str
    sequence_length: int
    source: str  # biopython / synthetic / etc.
    
    # Metrics
    rmsd: float
    tm_score: float
    gdt_ts: float
    
    # Orchestrator outputs
    levels_used: int
    runtime_s: float
    memory_mb: float
    energy_j: float
    verified: bool
    epistemic_levels: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def pareto_point(self) -> Tuple[float, float, float, float]:
        """(error, runtime, memory, energy) for Pareto analysis."""
        return (1.0 - self.tm_score, self.runtime_s, self.memory_mb, self.energy_j)


# ============================================================================
# 6. BENCHMARK RUNNER
# ============================================================================
class PDBBenchmarkRunner:
    """
    Runs PROTEIN_FOLDING_ORCHESTRATOR against real PDB structures.
    Collects metrics and produces Pareto reports.
    """
    
    def __init__(self, cfg: PDBBenchmarkConfig):
        self.cfg = cfg
        self.fetcher = PDBFetcher(cfg)
        self.device = torch.device(cfg.device)
        self.reports: List[BenchmarkReport] = []
    
    def run_all(self, orchestrator: nn.Module) -> List[BenchmarkReport]:
        """
        Run benchmark on all configured PDB IDs.
        
        Args:
            orchestrator: ProteinFoldingOrchestrator instance (from
                          PROTEIN_FOLDING_ORCHESTRATOR.py)
        """
        from PROTEIN_FOLDING_ORCHESTRATOR import FoldingTask, TaskType
        
        structures = self.fetcher.fetch_all()
        logger.info(f"Running benchmark on {len(structures)} PDB structures...")
        
        self.reports = []
        for struct in structures:
            report = self._run_single(orchestrator, struct, FoldingTask, TaskType)
            self.reports.append(report)
            self._log_report(report)
        
        return self.reports
    
    def _run_single(
        self,
        orchestrator: nn.Module,
        struct: PDBStructure,
        FoldingTask,
        TaskType,
    ) -> BenchmarkReport:
        """Run orchestrator on a single PDB structure."""
        # Build FoldingTask
        task = FoldingTask(
            task_id=f"pdb_{struct.pdb_id}",
            task_type=TaskType.STRUCTURE_PREDICTION,
            sequence_length=struct.length,
            accuracy_target=0.8,
            time_budget_s=60.0,
            memory_budget_mb=2000.0,
            energy_budget_j=500.0,
            requires_quantum=False,
        )
        
        # Encode sequence as indices
        AA = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(AA)}
        seq_indices = [aa_to_idx.get(aa, 20) for aa in struct.sequence.upper()]
        sequence_tensor = torch.tensor([seq_indices], dtype=torch.long, device=self.device)
        
        # Run orchestrator
        t0 = time.time()
        orchestrator.eval()
        with torch.no_grad():
            out = orchestrator(sequence_tensor, task)
        runtime = time.time() - t0
        
        # Get predicted structure (from highest executed level)
        if "level3" in out.get("exec_", {}) or "exec_level3" in out:
            final = out.get("exec_level3", {}).get("quantum_features")
        elif "level2" in out.get("exec_", {}) or "exec_level2" in out:
            final = out.get("exec_level2", {}).get("refined_structure")
        else:
            final = out.get("exec_level1", {}).get("coarse_structure")
        
        # Compute metrics against reference
        if struct.ca_coords is not None and final is not None:
            # For demo: we compare a projection of final features to real coords
            # In real integration, final should be (1, L, 3) coords
            # Here we use a simple projection for demonstration
            predicted_coords = self._features_to_coords(final, struct.length)
            metrics = StructureMetrics.compute_all(
                predicted_coords,
                struct.ca_coords.to(self.device),
                self.cfg,
            )
        else:
            metrics = {"rmsd_angstrom": float("nan"), "tm_score": 0.0, "gdt_ts": 0.0}
        
        # Extract Pareto metrics
        pareto = out.get("pareto_report")
        if pareto is not None:
            runtime_s = pareto.runtime_s
            memory_mb = pareto.memory_mb
            energy_j = pareto.energy_j
            levels_used = pareto.levels_used
            verified = pareto.verified
            epistemic_levels = pareto.epistemic_levels
        else:
            runtime_s = runtime
            memory_mb = 100.0
            energy_j = 10.0
            levels_used = 1
            verified = False
            epistemic_levels = ["HEURISTIC"]
        
        return BenchmarkReport(
            pdb_id=struct.pdb_id,
            name=struct.name,
            sequence_length=struct.length,
            source=struct.source,
            rmsd=metrics["rmsd_angstrom"],
            tm_score=metrics["tm_score"],
            gdt_ts=metrics["gdt_ts"],
            levels_used=levels_used,
            runtime_s=runtime_s,
            memory_mb=memory_mb,
            energy_j=energy_j,
            verified=verified,
            epistemic_levels=epistemic_levels,
        )
    
    def _features_to_coords(self, features: Tensor, length: int) -> Tensor:
        """
        Convert high-dimensional features to (L, 3) coordinates.
        This is a placeholder — in production, the orchestrator should
        output coordinates directly.
        """
        if features.dim() == 3:
            features = features.mean(dim=0)  # collapse batch
        if features.dim() == 1:
            # Need to expand to (L, 3)
            dim = features.shape[0]
            # Use first 3*length dims if available, else project
            if dim >= 3 * length:
                return features[: 3 * length].reshape(length, 3)
            else:
                # Random projection for demonstration
                torch.manual_seed(self.cfg.seed)
                proj = torch.randn(dim, 3 * length, device=features.device) * 0.1
                coords = (features @ proj).reshape(length, 3)
                return coords
        return features.reshape(length, 3)
    
    def _log_report(self, report: BenchmarkReport) -> None:
        logger.info(
            f"  {report.pdb_id} ({report.name}, L={report.sequence_length}): "
            f"RMSD={report.rmsd:.2f}Å, TM={report.tm_score:.3f}, "
            f"GDT={report.gdt_ts:.3f}, source={report.source}"
        )
    
    def summary_table(self) -> str:
        """Generate a summary table of all benchmark results."""
        lines = []
        lines.append("=" * 90)
        lines.append("PDB BENCHMARK RESULTS")
        lines.append("=" * 90)
        lines.append(
            f"{'PDB':<6} {'Name':<20} {'L':>4} {'Src':<10} "
            f"{'RMSD':>7} {'TM':>6} {'GDT':>6} {'Lev':>3} {'Ver':>3}"
        )
        lines.append("-" * 90)
        for r in self.reports:
            lines.append(
                f"{r.pdb_id:<6} {r.name[:20]:<20} {r.sequence_length:>4} "
                f"{r.source[:10]:<10} {r.rmsd:>7.2f} {r.tm_score:>6.3f} "
                f"{r.gdt_ts:>6.3f} {r.levels_used:>3} "
                f"{'Y' if r.verified else 'N':>3}"
            )
        lines.append("=" * 90)
        
        # Aggregate
        if self.reports:
            tm_scores = [r.tm_score for r in self.reports if not math.isnan(r.tm_score)]
            rmsd_vals = [r.rmsd for r in self.reports if not math.isnan(r.rmsd)]
            if tm_scores:
                lines.append(f"Mean TM-score: {sum(tm_scores)/len(tm_scores):.3f}")
            if rmsd_vals:
                lines.append(f"Mean RMSD:     {sum(rmsd_vals)/len(rmsd_vals):.2f} Å")
        
        return "\n".join(lines)
    
    def save_report(self, path: str) -> None:
        """Save full report as JSON."""
        data = {
            "config": self.cfg.to_dict(),
            "reports": [r.to_dict() for r in self.reports],
            "timestamp": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Report saved to {path}")


# ============================================================================
# 7. SMOKE TEST / SELF-TEST SUITE
# ============================================================================
def smoke_test():
    """[PASS]/[FAIL] self-test suite following ONE Ecosystem conventions."""
    print("=" * 72)
    print("  PDB BENCHMARK SUITE v1.0 — Self-Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Biopython available: {_HAS_BIOPYTHON}")
    print(f"  Requests available:  {_HAS_REQUESTS}")
    torch.manual_seed(42)
    
    cfg = PDBBenchmarkConfig(
        pdb_ids=["1UBQ", "1CRN"],  # small set for smoke test
        cache_dir="/tmp/pdb_benchmark_smoke",
        use_offline_fallback=True,
        max_sequence_length=80,
        device=str(device),
    )
    
    # ── Test 1: Config validation ──
    assert len(cfg.pdb_ids) > 0
    print("[PASS] PDBBenchmarkConfig validates")
    
    # ── Test 2: Kabsch alignment on identical structures ──
    P = torch.randn(20, 3, device=device)
    P_aligned, R, rmsd = StructureMetrics.kabsch_align(P, P)
    assert rmsd < 1e-4, f"Self-alignment RMSD should be ~0, got {rmsd}"
    assert torch.allclose(R @ R.T, torch.eye(3, device=device), atol=1e-5)
    print(f"[PASS] Kabsch self-alignment: RMSD={rmsd:.2e}")
    
    # ── Test 3: Kabsch with known rotation ──
    theta = math.pi / 4
    R_true = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0, 0, 1],
    ], device=device)
    t_true = torch.tensor([5.0, -3.0, 2.0], device=device)
    Q = torch.randn(30, 3, device=device)
    P_rot = Q @ R_true.T + t_true
    _, R_rec, rmsd = StructureMetrics.kabsch_align(P_rot, Q)
    assert rmsd < 1e-4, f"Known-rotation RMSD should be ~0, got {rmsd}"
    assert torch.allclose(R_rec, R_true, atol=1e-4)
    print(f"[PASS] Kabsch recovers known rotation: RMSD={rmsd:.2e}")
    
    # ── Test 4: TM-score of identical structures ──
    P = torch.randn(50, 3, device=device)
    tm = StructureMetrics.tm_score(P, P, L=50)
    assert tm > 0.99, f"TM-score of identical should be ~1.0, got {tm}"
    print(f"[PASS] TM-score self-comparison: {tm:.4f}")
    
    # ── Test 5: GDT-TS of identical structures ──
    gdt = StructureMetrics.gdt_ts(P, P)
    assert gdt > 0.99, f"GDT-TS of identical should be ~1.0, got {gdt}"
    print(f"[PASS] GDT-TS self-comparison: {gdt:.4f}")
    
    # ── Test 6: TM-score decreases with perturbation ──
    P_noisy = P + torch.randn_like(P) * 5.0  # large noise
    tm_noisy = StructureMetrics.tm_score(P_noisy, P, L=50)
    assert tm_noisy < tm, "TM-score should decrease with noise"
    print(f"[PASS] TM-score decreases with noise: {tm:.4f} → {tm_noisy:.4f}")
    
    # ── Test 7: PDBFetcher offline fallback ──
    fetcher = PDBFetcher(cfg)
    struct = fetcher.fetch("1UBQ")
    assert struct.pdb_id == "1UBQ"
    assert struct.length > 0
    assert struct.ca_coords is not None
    assert struct.ca_coords.shape == (struct.length, 3)
    print(f"[PASS] PDBFetcher fetched {struct.pdb_id}: L={struct.length}, "
          f"source={struct.source}")
    
    # ── Test 8: fetch_all with curated set ──
    all_structs = fetcher.fetch_all()
    assert len(all_structs) == len(cfg.pdb_ids)
    print(f"[PASS] fetch_all returned {len(all_structs)} structures")
    
    # ── Test 9: Integration with orchestrator ──
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ProteinFoldingOrchestrator,
            ProteinOrchestratorConfig,
        )
        orch_cfg = ProteinOrchestratorConfig(
            latent_dim=32,
            sequence_dim=16,
            structure_dim=32,
            level1_dim=16,
            level2_dim=24,
            level3_dim=32,
            batch_size=1,
            device=str(device),
        )
        orchestrator = ProteinFoldingOrchestrator(orch_cfg).to(device)
        
        runner = PDBBenchmarkRunner(cfg)
        reports = runner.run_all(orchestrator)
        
        assert len(reports) == len(cfg.pdb_ids)
        for r in reports:
            assert not math.isnan(r.rmsd), f"NaN RMSD for {r.pdb_id}"
            assert 0.0 <= r.tm_score <= 1.0, f"TM-score out of range for {r.pdb_id}"
        
        print(runner.summary_table())
        print("[PASS] End-to-end orchestrator integration")
        
        # Save report
        runner.save_report("/tmp/pdb_benchmark_smoke.json")
        print("[PASS] Report saved to JSON")
        
    except ImportError as e:
        print(f"[SKIP] Orchestrator integration: {e}")
    
    print("=" * 72)
    print("  ⚠️  HONEST SCOPE REMINDERS:")
    print("  • Offline fallback uses SYNTHETIC CA traces — not real structures")
    print("  • For real benchmarking, install biopython + requests")
    print("  • Compare against AlphaFold2 / ESMFold for meaningful accuracy")
    print("  • Orchestrator levels are HEURISTIC/CONDITIONAL — not SOTA")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
