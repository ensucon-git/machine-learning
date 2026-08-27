"""Manufacturer performance map: COP, capacity and the backup heater.

A weather-compensated heat pump has two ways to make the bill worse rather than
better, and a generic Carnot model sees neither:

* **COP is not a smooth function of the outdoor temperature alone.** Pushing the
  supply temperature up to grab a cheap hour can cost more efficiency than the
  price saves, especially at low ambient temperatures where the lift is already
  large.
* **The compressor runs out of capacity.** Once the demanded heat exceeds what
  the compressor can deliver, a Daikin hydrobox makes up the difference with an
  electric backup heater at COP 1.0. An optimiser blind to that will happily
  plan a preheat that quietly burns resistive kilowatt-hours at three to four
  times the price.

The efficiency table is stored as Carnot efficiency rather than raw COP:

    COP = efficiency(T_ambient, T_supply) * (T_supply + 273.15) / (T_supply - T_ambient)

Carnot efficiency varies slowly and stays bounded, so bilinear interpolation
between rating points is stable and extrapolation off the edges stays physical.
Interpolating raw COP does neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

KELVIN = 273.15
MIN_LIFT = 1.0


def _bilinear(
    x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Bilinear lookup with clamped edges, vectorised over arbitrary shapes."""
    x = np.clip(np.asarray(x, dtype=float), x_grid[0], x_grid[-1])
    y = np.clip(np.asarray(y, dtype=float), y_grid[0], y_grid[-1])
    i = np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, len(x_grid) - 2)
    j = np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, len(y_grid) - 2)
    tx = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    ty = (y - y_grid[j]) / (y_grid[j + 1] - y_grid[j])
    return (
        values[i, j] * (1 - tx) * (1 - ty)
        + values[i + 1, j] * tx * (1 - ty)
        + values[i, j + 1] * (1 - tx) * ty
        + values[i + 1, j + 1] * tx * ty
    )


@dataclass
class PerformanceMap:
    """Efficiency, capacity and backup-heater behaviour of a specific machine.

    Pure lookup: turning a heat demand into electricity is
    :meth:`hpmpc.model.heatpump.PumpModel.deliver`, so that arithmetic exists in
    exactly one place.
    """

    name: str
    source: str
    eff_ambient: np.ndarray
    eff_supply: np.ndarray
    eff_values: np.ndarray
    cap_ambient: np.ndarray
    cap_kw: np.ndarray
    derate_supply: np.ndarray
    derate_factor: np.ndarray
    min_ambient_c: float = -25.0
    min_supply_c: float = 25.0
    max_supply_c: float = 55.0
    backup_enabled: bool = True
    backup_max_kw: float = 9.0
    backup_cop: float = 1.0
    defrost_enabled: bool = True
    defrost_peak_ambient_c: float = 1.0
    defrost_width_c: float = 5.0
    defrost_max_derate: float = 0.10
    defrost_humidity_reference_pct: float = 75.0
    defrost_humidity_sensitivity: float = 0.9
    # Fitted against the owner's own electricity meter by ``hpmpc train``.
    efficiency_scale: float = 1.0
    capacity_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ lookups

    def efficiency(self, t_ambient: np.ndarray, t_supply: np.ndarray) -> np.ndarray:
        raw = _bilinear(self.eff_ambient, self.eff_supply, self.eff_values, t_ambient, t_supply)
        return raw * self.efficiency_scale

    def defrost_factor(self, t_ambient: np.ndarray, humidity_pct: np.ndarray | None = None) -> np.ndarray:
        """Extra derate for humid air near freezing, on top of the rating points.

        EN14511 ratings already include typical defrost losses, so this only
        bites when conditions are worse than the reference humidity.
        """
        t = np.asarray(t_ambient, dtype=float)
        if not self.defrost_enabled:
            return np.ones_like(t)
        shape = np.exp(-(((t - self.defrost_peak_ambient_c) / self.defrost_width_c) ** 2))
        if humidity_pct is None:
            excess = np.zeros_like(t)
        else:
            rh = np.asarray(humidity_pct, dtype=float)
            rh = np.where(np.isfinite(rh), rh, self.defrost_humidity_reference_pct)
            excess = np.clip(
                (rh - self.defrost_humidity_reference_pct) / max(100.0 - self.defrost_humidity_reference_pct, 1.0),
                0.0,
                1.0,
            )
        return 1.0 - self.defrost_max_derate * shape * (
            self.defrost_humidity_sensitivity * excess + (1.0 - self.defrost_humidity_sensitivity) * 1.0
        )

    def cop(
        self,
        t_ambient: np.ndarray,
        t_supply: np.ndarray,
        humidity_pct: np.ndarray | None = None,
    ) -> np.ndarray:
        ts = np.asarray(t_supply, dtype=float)
        ta = np.asarray(t_ambient, dtype=float)
        lift = np.maximum(ts - ta, MIN_LIFT)
        ideal = self.efficiency(ta, ts) * (ts + KELVIN) / lift
        return np.clip(ideal * self.defrost_factor(ta, humidity_pct), 1.0, 8.0)

    def capacity_w(self, t_ambient: np.ndarray, t_supply: np.ndarray) -> np.ndarray:
        """Compressor heating capacity [W]; zero below the operating limit."""
        ta = np.asarray(t_ambient, dtype=float)
        ts = np.asarray(t_supply, dtype=float)
        base = np.interp(ta, self.cap_ambient, self.cap_kw)
        derate = np.interp(ts, self.derate_supply, self.derate_factor)
        capacity = base * derate * self.capacity_scale * 1000.0
        return np.where(ta >= self.min_ambient_c, np.maximum(capacity, 0.0), 0.0)

    def backup_capacity_w(self) -> float:
        return self.backup_max_kw * 1000.0 if self.backup_enabled else 0.0

    # ----------------------------------------------------------- plumbing

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "efficiency_scale": self.efficiency_scale,
            "capacity_scale": self.capacity_scale,
            "backup_enabled": self.backup_enabled,
            "backup_max_kw": self.backup_max_kw,
            "min_ambient_c": self.min_ambient_c,
        }

    def cop_table(
        self, ambient: list[float] | None = None, supply: list[float] | None = None
    ) -> tuple[list[float], list[float], np.ndarray]:
        """COP over a grid, for printing and eyeballing against the databook."""
        ambient = ambient if ambient is not None else [float(v) for v in self.eff_ambient]
        supply = supply if supply is not None else [float(v) for v in self.eff_supply]
        grid_a, grid_s = np.meshgrid(np.array(ambient), np.array(supply), indexing="ij")
        return ambient, supply, self.cop(grid_a, grid_s)


def load_performance_map(name_or_path: str) -> PerformanceMap:
    """Load a bundled pump model by name, or an arbitrary YAML file by path."""
    candidate = Path(name_or_path)
    if candidate.exists():
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    else:
        from importlib.resources import files

        resource = files("hpmpc") / "resources" / "pumps" / f"{name_or_path}.yaml"
        if not resource.is_file():
            available = sorted(
                p.name.removesuffix(".yaml")
                for p in (files("hpmpc") / "resources" / "pumps").iterdir()
                if p.name.endswith(".yaml")
            )
            raise ValueError(
                f"Unknown heat pump model '{name_or_path}'. Bundled models: {', '.join(available)}. "
                "You can also give a path to your own YAML file."
            )
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return performance_map_from_dict(raw)


def performance_map_from_dict(raw: dict[str, Any]) -> PerformanceMap:
    efficiency = raw["efficiency"]
    values = np.asarray(efficiency["values"], dtype=float)
    ambient = np.asarray(efficiency["ambient_c"], dtype=float)
    supply = np.asarray(efficiency["supply_c"], dtype=float)
    if values.shape != (ambient.size, supply.size):
        raise ValueError(
            f"efficiency.values must be {ambient.size} rows (ambient) x {supply.size} columns (supply), "
            f"got {values.shape}"
        )
    if np.any(np.diff(ambient) <= 0) or np.any(np.diff(supply) <= 0):
        raise ValueError("efficiency.ambient_c and efficiency.supply_c must be strictly increasing")
    if np.any(values <= 0) or np.any(values > 1.0):
        raise ValueError("efficiency values are Carnot efficiencies and must lie in (0, 1]")

    capacity = raw["capacity"]
    derate = capacity.get("supply_derate", {})
    limits = raw.get("limits", {})
    backup = raw.get("backup_heater", {})
    defrost = raw.get("defrost", {})

    return PerformanceMap(
        name=raw.get("name", "unnamed"),
        source=raw.get("source", "unknown"),
        eff_ambient=ambient,
        eff_supply=supply,
        eff_values=values,
        cap_ambient=np.asarray(capacity["ambient_c"], dtype=float),
        cap_kw=np.asarray(capacity["kw_at_w35"], dtype=float),
        derate_supply=np.asarray(derate.get("supply_c", [25.0, 55.0]), dtype=float),
        derate_factor=np.asarray(derate.get("factor", [1.0, 1.0]), dtype=float),
        min_ambient_c=float(limits.get("min_ambient_c", -25.0)),
        min_supply_c=float(limits.get("min_supply_c", 25.0)),
        max_supply_c=float(limits.get("max_supply_c", 55.0)),
        backup_enabled=bool(backup.get("enabled", True)),
        backup_max_kw=float(backup.get("max_kw", 9.0)),
        backup_cop=float(backup.get("cop", 1.0)),
        defrost_enabled=bool(defrost.get("enabled", True)),
        defrost_peak_ambient_c=float(defrost.get("peak_ambient_c", 1.0)),
        defrost_width_c=float(defrost.get("width_c", 5.0)),
        defrost_max_derate=float(defrost.get("max_derate", 0.10)),
        defrost_humidity_reference_pct=float(defrost.get("humidity_reference_pct", 75.0)),
        defrost_humidity_sensitivity=float(defrost.get("humidity_sensitivity", 0.9)),
        metadata={k: v for k, v in raw.items() if k not in {"efficiency", "capacity"}},
    )
