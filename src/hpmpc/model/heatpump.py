"""Heat pump model: heating curve, outdoor-temperature filter, COP and power.

The controller never touches the pump's own control loop. It only changes what
the pump *believes* the outdoor temperature is, and the pump then derives its
supply-temperature setpoint from its heating curve. Everything in this module
is that chain, written so it can be evaluated for a whole batch of candidate
offset schedules at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HeatPumpConfig
from .performance import PerformanceMap

KELVIN = 273.15


def outdoor_filter_alpha(tau_hours: float, dt_hours: float) -> float:
    """Discrete first-order filter coefficient for the pump's outdoor averaging."""
    tau = max(float(tau_hours), 1e-6)
    return float(np.exp(-float(dt_hours) / tau))


def filter_outdoor_series(
    perceived: np.ndarray, tau_hours: float, dt_hours: float, initial: float | np.ndarray
) -> np.ndarray:
    """Run the pump's outdoor-temperature filter over a series.

    ``perceived`` has shape ``(..., K)``; the filtered output has the same shape.
    Most pumps average the outdoor sensor over a few hours, which is exactly why
    a naive offset step does not act on the supply temperature immediately.
    """
    perceived = np.atleast_2d(np.asarray(perceived, dtype=float))
    alpha = outdoor_filter_alpha(tau_hours, dt_hours)
    out = np.empty_like(perceived)
    state = np.broadcast_to(np.asarray(initial, dtype=float), perceived.shape[:-1]).astype(float).copy()
    for k in range(perceived.shape[-1]):
        state = alpha * state + (1.0 - alpha) * perceived[..., k]
        out[..., k] = state
    return out


def supply_setpoint(t_outdoor_filtered: np.ndarray, cfg: HeatPumpConfig) -> np.ndarray:
    """Supply-temperature setpoint [degC] produced by the pump's heating curve."""
    t = np.asarray(t_outdoor_filtered, dtype=float)
    setpoint = cfg.curve_offset + cfg.curve_slope * (cfg.curve_ref - t)
    return np.clip(setpoint, cfg.supply_min, cfg.supply_max)


def heating_enabled(t_outdoor_filtered: np.ndarray, cfg: HeatPumpConfig) -> np.ndarray:
    """Boolean mask: is the pump allowed to produce heat at this filtered temp?"""
    return np.asarray(t_outdoor_filtered, dtype=float) < cfg.heat_stop_temp


def mean_water_temp(supply_temp: np.ndarray, cfg: HeatPumpConfig) -> np.ndarray:
    """Mean floor-loop water temperature, i.e. supply minus half the loop delta."""
    return np.asarray(supply_temp, dtype=float) - 0.5 * cfg.loop_delta_t


def cop(supply_temp: np.ndarray, t_outdoor_real: np.ndarray, cfg: HeatPumpConfig) -> np.ndarray:
    """Coefficient of performance from a Carnot model with a defrost penalty.

    Note the asymmetry that makes the whole scheme work: the *supply*
    temperature follows the faked outdoor temperature, but the COP is set by the
    *real* outdoor temperature, because that is the air the evaporator sees.
    """
    ts = np.asarray(supply_temp, dtype=float)
    te = np.asarray(t_outdoor_real, dtype=float)
    lift = np.maximum(ts - te, 1.0)
    ideal = cfg.carnot_efficiency * (ts + KELVIN) / lift
    # Defrost losses peak around -2 degC where air is cold and still humid.
    defrost = 1.0 - cfg.defrost_penalty * np.exp(-(((te + 2.0) / 4.0) ** 2))
    return np.clip(ideal * defrost, cfg.cop_min, cfg.cop_max)


def electric_power(
    heat_output_w: np.ndarray,
    supply_temp: np.ndarray,
    t_outdoor_real: np.ndarray,
    cfg: HeatPumpConfig,
) -> np.ndarray:
    """Electric input power [W] for a given delivered heat output."""
    q = np.maximum(np.asarray(heat_output_w, dtype=float), 0.0)
    c = cop(supply_temp, t_outdoor_real, cfg)
    return q / c + cfg.standby_power_w


def offset_for_supply_temp(target_supply: float, t_outdoor_real: float, cfg: HeatPumpConfig) -> float:
    """Inverse heating curve: which offset yields ``target_supply`` in steady state?

    Only used for diagnostics and for explaining a decision to the user.
    """
    if abs(cfg.curve_slope) < 1e-9:
        return 0.0
    target = float(np.clip(target_supply, cfg.supply_min, cfg.supply_max))
    t_needed = cfg.curve_ref - (target - cfg.curve_offset) / cfg.curve_slope
    return t_needed - float(t_outdoor_real)


@dataclass
class OperatingPoint:
    """COP and capacity already looked up for a whole horizon.

    Both depend only on the supply temperature and the real outdoor
    temperature, which are known before the house is simulated. Resolving them
    once, for the whole batch and horizon, keeps the table lookup out of the
    integration loop - about a tenfold difference in solve time.
    """

    cop: np.ndarray
    capacity: np.ndarray
    backup_capacity_w: float
    backup_cop: float
    standby_power_w: float

    def deliver_at(self, demand: np.ndarray, index: int) -> dict[str, np.ndarray]:
        demand = np.maximum(demand, 0.0)
        q_hp = np.minimum(demand, self.capacity[:, index])
        q_backup = np.clip(demand - q_hp, 0.0, self.backup_capacity_w)
        power = q_hp / self.cop[:, index] + q_backup / self.backup_cop + self.standby_power_w
        return {"q_heat": q_hp + q_backup, "q_backup": q_backup, "p_electric": power}


@dataclass
class PumpModel:
    """The pump as the controller sees it: a heating curve plus a way to turn a
    heat demand into electricity.

    Two backends. With a :class:`PerformanceMap` loaded, capacity, COP, defrost
    and the electric backup heater all come from the machine's own data. Without
    one, it falls back to the generic Carnot model and a flat capacity limit,
    which is fine for a first look but blind to the backup heater - and the
    backup heater is precisely how a well-meaning optimiser turns a cheap hour
    into a more expensive month.
    """

    cfg: HeatPumpConfig
    performance: PerformanceMap | None = None

    @classmethod
    def coerce(cls, pump: "PumpModel | HeatPumpConfig") -> "PumpModel":
        return pump if isinstance(pump, cls) else cls(pump)

    @property
    def max_supply(self) -> float:
        if self.performance is None:
            return self.cfg.supply_max
        return min(self.cfg.supply_max, self.performance.max_supply_c)

    def supply_setpoint(self, t_outdoor_filtered: np.ndarray) -> np.ndarray:
        t = np.asarray(t_outdoor_filtered, dtype=float)
        setpoint = self.cfg.curve_offset + self.cfg.curve_slope * (self.cfg.curve_ref - t)
        return np.clip(setpoint, self.cfg.supply_min, self.max_supply)

    def heating_enabled(self, t_outdoor_filtered: np.ndarray) -> np.ndarray:
        return heating_enabled(t_outdoor_filtered, self.cfg)

    def mean_water_temp(self, supply_temp: np.ndarray) -> np.ndarray:
        return mean_water_temp(supply_temp, self.cfg)

    def capacity_w(self, t_ambient: np.ndarray, t_supply: np.ndarray) -> np.ndarray:
        if self.performance is None:
            return np.full(np.broadcast(np.asarray(t_ambient), np.asarray(t_supply)).shape,
                           self.cfg.max_heat_output_w)
        return self.performance.capacity_w(t_ambient, t_supply)

    def cop(
        self, t_supply: np.ndarray, t_ambient: np.ndarray, humidity_pct: np.ndarray | None = None
    ) -> np.ndarray:
        if self.performance is None:
            return cop(t_supply, t_ambient, self.cfg)
        return self.performance.cop(t_ambient, t_supply, humidity_pct)

    def operating_point(
        self, t_supply: np.ndarray, t_ambient: np.ndarray, humidity_pct: np.ndarray | None = None
    ) -> OperatingPoint:
        """Resolve COP and capacity over a whole (batch, horizon) grid at once."""
        shape = np.broadcast_shapes(np.shape(t_supply), np.shape(t_ambient))
        ts = np.broadcast_to(np.asarray(t_supply, dtype=float), shape)
        ta = np.broadcast_to(np.asarray(t_ambient, dtype=float), shape)
        rh = None if humidity_pct is None else np.broadcast_to(np.asarray(humidity_pct, dtype=float), shape)
        if self.performance is None:
            return OperatingPoint(
                cop=np.maximum(cop(ts, ta, self.cfg), 1e-6),
                capacity=np.full(shape, float(self.cfg.max_heat_output_w)),
                backup_capacity_w=0.0,
                backup_cop=1.0,
                standby_power_w=self.cfg.standby_power_w,
            )
        return OperatingPoint(
            cop=np.maximum(self.performance.cop(ta, ts, rh), 1e-6),
            capacity=self.performance.capacity_w(ta, ts),
            backup_capacity_w=self.performance.backup_capacity_w(),
            backup_cop=max(self.performance.backup_cop, 1e-6),
            standby_power_w=self.cfg.standby_power_w,
        )

    def deliver(
        self,
        q_demand_w: np.ndarray,
        t_supply: np.ndarray,
        t_ambient: np.ndarray,
        humidity_pct: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Meet a heat demand and report what it costs in electricity."""
        demand = np.maximum(np.asarray(q_demand_w, dtype=float), 0.0)
        point = self.operating_point(t_supply, t_ambient, humidity_pct)
        q_hp = np.minimum(demand, point.capacity)
        q_backup = np.clip(demand - q_hp, 0.0, point.backup_capacity_w)
        return {
            "q_heat": q_hp + q_backup,
            "q_compressor": q_hp,
            "q_backup": q_backup,
            "p_electric": q_hp / point.cop + q_backup / point.backup_cop + point.standby_power_w,
            "cop": point.cop,
            "capacity": point.capacity,
        }

    def describe(self) -> dict[str, object]:
        if self.performance is None:
            return {"model": "generic Carnot", "carnot_efficiency": self.cfg.carnot_efficiency}
        return {"model": self.performance.name, **self.performance.to_dict()}
