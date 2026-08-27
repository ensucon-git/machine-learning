"""Physical + learned models of the house and the heat pump."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from .heatpump import PumpModel
from .performance import PerformanceMap, load_performance_map

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config

__all__ = ["PumpModel", "PerformanceMap", "build_pump", "load_performance_map"]


@lru_cache(maxsize=8)
def _cached_map(name: str) -> PerformanceMap:
    return load_performance_map(name)


def build_pump(cfg: "Config") -> PumpModel:
    """Assemble the pump model a config asks for.

    ``heat_pump.model`` names a bundled performance map (or a path to your own
    YAML). Leaving it empty falls back to the generic Carnot model, which is
    blind to the electric backup heater - fine for a first look, not for
    deciding how hard to preheat in February.
    """
    if not cfg.heat_pump.model:
        return PumpModel(cfg.heat_pump)
    performance = replace(
        _cached_map(cfg.heat_pump.model),
        efficiency_scale=cfg.heat_pump.efficiency_scale,
        capacity_scale=cfg.heat_pump.capacity_scale,
    )
    return PumpModel(cfg.heat_pump, performance)
