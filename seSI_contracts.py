"""
seSI_contracts.py — Integration glue for the five SESI modules.
Purpose: make the orchestrator CALL real adapters instead of
         carrying metadata-only stand-ins.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class Epistemic(Enum):
    VERIFIED = 4
    CONDITIONAL = 3
    HEURISTIC = 2
    SPECULATIVE = 1


class Provenance(Enum):
    REAL_DATASET = "real_dataset"
    SYNTHETIC = "synthetic"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class TensorSpec:
    """Explicit I/O contract at every module boundary."""
    name: str
    shape: Tuple[int, ...]        # use -1 for batch
    dtype: str = "float32"
    semantic: str = ""            # e.g. "aa_embedding", "geometry_prior"


@dataclass
class ModuleCard:
    name: str
    module: nn.Module
    epistemic: Epistemic
    provenance: Provenance
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    cost: Dict[str, float]        # time / memory / energy (normalized)
    validate: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass
class Envelope:
    """Everything crossing a boundary carries its own provenance."""
    tensors: Dict[str, Tensor]
    epistemic: Epistemic
    provenance: Provenance
    cost: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def epistemic_floor(self) -> Epistemic:
        """Conservative: a chain is only as strong as its weakest link."""
        return self.epistemic


class ContractAdapter(nn.Module):
    """
    Normalize shape/semantic mismatches EXPLICITLY.
    No silent broadcasting, no silent dtype casting of physical barriers.
    """
    def __init__(self, src_dim: int, dst_dim: int, semantic: str):
        super().__init__()
        self.semantic = semantic
        self.proj = nn.Linear(src_dim, dst_dim)
        self.norm = nn.LayerNorm(dst_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.proj.in_features:
            raise ValueError(
                f"[{self.semantic}] expected {self.proj.in_features}, "
                f"got {x.shape[-1]}"
            )
        return self.norm(self.proj(x))


class SESIRegistry:
    """Registry of REAL module instances, not just names."""
    def __init__(self):
        self._cards: Dict[str, ModuleCard] = {}

    def register(self, card: ModuleCard) -> None:
        if card.name in self._cards:
            raise ValueError(f"Duplicate module: {card.name}")
        self._cards[card.name] = card

    def get(self, name: str) -> ModuleCard:
        return self._cards[name]

    def eligible(self, *, needs_real: bool = False) -> List[ModuleCard]:
        out = []
        for c in self._cards.values():
            if needs_real and c.provenance is not Provenance.REAL_DATASET:
                continue
            out.append(c)
        return out


class BoundHierarchicalExecutor(nn.Module):
    """
    Drop-in replacement for the orchestrator's self-contained executor.
    Levels are BOUND to real adapters from the other four files.
    """
    def __init__(self, registry: SESIRegistry):
        super().__init__()
        self.registry = registry
        self._bound: Dict[int, str] = {}
        self._adapters = nn.ModuleDict()

    def bind(self, level: int, module_name: str,
             src_dim: int, dst_dim: int) -> None:
        card = self.registry.get(module_name)
        self._bound[level] = module_name
        self._adapters[f"l{level}"] = ContractAdapter(
            src_dim, dst_dim, semantic=f"level{level}"
        )

    def _epistemic_floor(self, levels_used: List[int]) -> Epistemic:
        levels = [
            self.registry.get(self._bound[i]).epistemic
            for i in levels_used
        ]
        return min(levels, key=lambda e: e.value)

    def forward(self, task: Any, seq_features: Tensor,
                geometry_prior: Optional[Tensor] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        levels_used: List[int] = []
        provenance = Provenance.SIMULATED
        total_cost = {"time": 0.0, "memory": 0.0, "energy": 0.0}

        for level in sorted(self._bound):
            # Budget / task gating — do NOT run everything
            if level == 3 and not getattr(task, "requires_quantum", False):
                continue
            if level >= 2 and "level1" not in results:
                continue

            card = self.registry.get(self._bound[level])
            x = self._adapters[f"l{level}"](seq_features)

            # Attach geometry prior ONLY as auxiliary, never as AA sequence
            if level == 2 and geometry_prior is not None:
                x = x + 0.1 * geometry_prior.mean(dim=-1, keepdim=True) * x

            try:
                with torch.no_grad() if False else _nullcontext():
                    raw = card.module(x)
            except Exception as e:
                # Fail loudly at integration boundary; never fake a result
                raise RuntimeError(
                    f"Level {level} ({card.name}) failed: {e}"
                ) from e

            if card.validate is not None and not card.validate(raw):
                raise ValueError(f"Level {level}: contract validation failed")

            results[f"level{level}"] = raw
            levels_used.append(level)

            # Conservative provenance: synthetic anywhere => not real
            if card.provenance is Provenance.REAL_DATASET and \
               provenance is not Provenance.SYNTHETIC:
                provenance = Provenance.REAL_DATASET
            else:
                provenance = Provenance.SYNTHETIC

            for k in total_cost:
                total_cost[k] += card.cost.get(k, 1.0)

        return {
            **results,
            "levels_executed": len(levels_used),
            "total_computational_cost": total_cost["time"],
            "epistemic_floor": self._epistemic_floor(levels_used),
            "provenance": provenance,
        }


from contextlib import nullcontext as _nullcontext_ctx

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def promotion_allowed(envelope: Envelope) -> bool:
    """
    Hard rule: synthetic/simulated evidence can NEVER be promoted
    to VERIFIED, regardless of how good the loss looks.
    """
    if envelope.provenance is not Provenance.REAL_DATASET:
        return False
    return envelope.epistemic_floor.value >= Epistemic.CONDITIONAL.value
