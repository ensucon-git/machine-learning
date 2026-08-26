"""Heat pump model: heating curve, outdoor-temperature filter, COP and power.

The controller never touches the pump's own control loop. It only changes what
the pump *believes* the outdoor temperature is, and the pump then derives its
supply-temperature setpoint from its heating curve. Everything in this module
is that chain, written so it can be evaluated for a whole batch of candidate
offset schedules at once.
"""

from __future__ import annotations

import numpy as np

from ..config import HeatPumpConfig

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
