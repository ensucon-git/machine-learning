"""Conversion between a fake outdoor temperature and the resistance the heat
pump's outdoor sensor input should see.

This is the "digital resistor" half of the setup: Home Assistant drives a
digital potentiometer / DAC-controlled resistance in place of the pump's NTC
sensor, so the pump can be told any outdoor temperature we like.
"""

from __future__ import annotations

import numpy as np

from .config import NTCConfig, PotConfig

KELVIN = 273.15


def temperature_to_resistance(temp_c: float | np.ndarray, cfg: NTCConfig) -> np.ndarray:
    """Resistance [ohm] the sensor input should see for ``temp_c``."""
    t = np.asarray(temp_c, dtype=float)
    if cfg.model == "table":
        temps = np.asarray(cfg.table_temp_c, dtype=float)
        ohms = np.asarray(cfg.table_ohm, dtype=float)
        if temps.size < 2 or temps.size != ohms.size:
            raise ValueError("ntc.table_temp_c and ntc.table_ohm must have the same length >= 2")
        order = np.argsort(temps)
        r = np.interp(t, temps[order], ohms[order])
    elif cfg.model == "beta":
        # R = R25 * exp(B * (1/T - 1/298.15))
        r = cfg.r25 * np.exp(cfg.beta * (1.0 / (t + KELVIN) - 1.0 / (25.0 + KELVIN)))
    else:
        raise ValueError(f"Unknown ntc.model '{cfg.model}' (expected 'beta' or 'table')")
    return np.clip(r, cfg.resistance_min, cfg.resistance_max)


def resistance_to_temperature(resistance: float | np.ndarray, cfg: NTCConfig) -> np.ndarray:
    """Inverse of :func:`temperature_to_resistance`."""
    r = np.asarray(resistance, dtype=float)
    if cfg.model == "table":
        temps = np.asarray(cfg.table_temp_c, dtype=float)
        ohms = np.asarray(cfg.table_ohm, dtype=float)
        order = np.argsort(ohms)
        return np.interp(r, ohms[order], temps[order])
    if cfg.model == "beta":
        inv_t = np.log(np.maximum(r, 1e-9) / cfg.r25) / cfg.beta + 1.0 / (25.0 + KELVIN)
        return 1.0 / inv_t - KELVIN
    raise ValueError(f"Unknown ntc.model '{cfg.model}'")


def resolution_check(cfg: NTCConfig, temp_c: float, step_ohm: float) -> float:
    """Temperature resolution [K] achievable with a resistance step ``step_ohm``.

    Useful when sizing the digital potentiometer: a 100 kohm / 256-step device
    gives coarse steps at mild outdoor temperatures.
    """
    r0 = float(temperature_to_resistance(temp_c, cfg))
    t1 = float(resistance_to_temperature(r0 + step_ohm, cfg))
    return abs(t1 - temp_c)


# --------------------------------------------------------------------------
# The digital potentiometer itself
#
# The NTC curve says which resistance means which temperature. The pot says
# which resistances are actually reachable, and in what increments. Keeping the
# two apart matters: recalibrating the sensor curve must not silently change
# what the hardware can do.


def wiper_span(cfg: PotConfig) -> int:
    """Highest aggregate wiper index across all devices in series."""
    return int(cfg.devices) * (int(cfg.steps) - 1)


def wiper_to_resistance(step: float | np.ndarray, cfg: PotConfig) -> np.ndarray:
    """Resistance [ohm] presented at aggregate wiper position ``step``."""
    span = wiper_span(cfg)
    d = np.clip(np.asarray(step, dtype=float), 0.0, span)
    per_step = float(cfg.resistance_ohm) / (int(cfg.steps) - 1)
    return d * per_step + float(cfg.wiper_ohm) * int(cfg.devices) + float(cfg.series_ohm)


def resistance_to_wiper(resistance: float | np.ndarray, cfg: PotConfig) -> np.ndarray:
    """Nearest reachable wiper position for ``resistance``, as an integer.

    Clamps rather than extrapolating: outside the pot's span there is no
    position to return, and pretending otherwise is how a controller ends up
    believing it commanded something the hardware refused.
    """
    per_step = float(cfg.resistance_ohm) / (int(cfg.steps) - 1)
    fixed = float(cfg.wiper_ohm) * int(cfg.devices) + float(cfg.series_ohm)
    raw = (np.asarray(resistance, dtype=float) - fixed) / per_step
    return np.clip(np.rint(raw), 0.0, float(wiper_span(cfg)))


def reachable_temperatures(pot: PotConfig, ntc: NTCConfig) -> tuple[float, float]:
    """(coldest, warmest) temperature the pot can actually show the pump.

    Coldest comes from the highest resistance, because the sensor is an NTC.
    """
    lo = float(wiper_to_resistance(0, pot))
    hi = float(wiper_to_resistance(wiper_span(pot), pot))
    coldest = float(resistance_to_temperature(hi, ntc))
    warmest = float(resistance_to_temperature(max(lo, 1e-6), ntc))
    return coldest, warmest


def wiper_resolution(pot: PotConfig, ntc: NTCConfig, temp_c: float) -> float:
    """Temperature resolution [K] of one wiper step at ``temp_c``."""
    per_step = float(pot.resistance_ohm) / (int(pot.steps) - 1)
    return resolution_check(ntc, temp_c, per_step)
