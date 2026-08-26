"""Conversion between a fake outdoor temperature and the resistance the heat
pump's outdoor sensor input should see.

This is the "digital resistor" half of the setup: Home Assistant drives a
digital potentiometer / DAC-controlled resistance in place of the pump's NTC
sensor, so the pump can be told any outdoor temperature we like.
"""

from __future__ import annotations

import numpy as np

from .config import NTCConfig

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
