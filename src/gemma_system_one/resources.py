"""Medición de memoria y tiempo.

La memoria del proceso (RSS) y la del driver MPS se registran por separado y no se
suman: en memoria unificada pueden solaparse. Los picos son el máximo de muestras
puntuales tomadas en los puntos instrumentados, no un pico exacto.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

from .env import sysctl

GIB = 1024**3


def _mps_ready() -> bool:
    return torch.backends.mps.is_available()


def synchronize(device: str | torch.device) -> None:
    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":  # pragma: no cover - no se usa en este proyecto
        torch.cuda.synchronize()


def memory_snapshot() -> dict[str, Any]:
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    snap: dict[str, Any] = {
        "process_rss_bytes": proc.memory_info().rss,
        "system_total_bytes": vm.total,
        "system_available_bytes": vm.available,
        "system_used_percent": vm.percent,
        "swap_used_bytes": sw.used,
        "swap_total_bytes": sw.total,
    }
    pressure = sysctl("kern.memorystatus_vm_pressure_level")
    # 1 = normal, 2 = warn, 4 = critical (niveles de memorystatus de Darwin).
    snap["vm_pressure_level"] = int(pressure) if pressure and pressure.isdigit() else None
    if _mps_ready():
        snap["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        snap["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        snap["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()
    return snap


@dataclass
class MemoryTracker:
    """Registra instantáneas etiquetadas y el máximo observado de cada métrica."""

    budget_bytes: int
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, label: str) -> dict[str, Any]:
        snap = {"label": label, **memory_snapshot()}
        self.samples.append(snap)
        exceeded = self.over_budget()
        if exceeded:
            raise RuntimeError(f"Presupuesto de memoria superado: {exceeded}")
        return snap

    def peaks(self) -> dict[str, int]:
        keys = (
            "process_rss_bytes",
            "mps_current_allocated_bytes",
            "mps_driver_allocated_bytes",
            "swap_used_bytes",
        )
        out: dict[str, int] = {}
        for k in keys:
            vals = [s[k] for s in self.samples if s.get(k) is not None]
            if vals:
                out[k] = max(vals)
        return out

    def over_budget(self) -> list[str]:
        """Métricas cuyo máximo muestreado supera el presupuesto (cada una por separado)."""
        return [
            k
            for k, v in self.peaks().items()
            if k in ("process_rss_bytes", "mps_driver_allocated_bytes") and v > self.budget_bytes
        ]

    def summary(self) -> dict[str, Any]:
        first = self.samples[0] if self.samples else {}
        last = self.samples[-1] if self.samples else {}
        return {
            "sampling": "instantáneas en puntos instrumentados; el pico real puede ser mayor",
            "budget_bytes": self.budget_bytes,
            "peaks_sampled": self.peaks(),
            "over_budget": self.over_budget(),
            "swap_delta_bytes": (last.get("swap_used_bytes", 0) - first.get("swap_used_bytes", 0))
            if self.samples
            else None,
            "samples": self.samples,
        }


@contextmanager
def timed(device: str | torch.device) -> Iterator[dict[str, float]]:
    """Cronómetro con sincronización del dispositivo antes y después."""
    result: dict[str, float] = {}
    synchronize(device)
    start = time.perf_counter()
    try:
        yield result
    finally:
        synchronize(device)
        result["seconds"] = time.perf_counter() - start
