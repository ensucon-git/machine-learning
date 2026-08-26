"""Local solar geometry and a clear-sky irradiance model.

Home Assistant weather integrations usually give cloud coverage in percent but
not irradiance. Irradiance is what actually heats the house, so we reconstruct
it from solar geometry + cloud cover. Everything is computed locally with
numpy - no external service, no API key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1361.0


def solar_position(index: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Return solar ``elevation`` and ``azimuth`` in degrees (NOAA approximation)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    utc = idx.tz_convert("UTC")

    day_of_year = utc.dayofyear.to_numpy(dtype=float)
    hour = utc.hour.to_numpy(dtype=float) + utc.minute.to_numpy(dtype=float) / 60.0

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    minutes_utc = hour * 60.0
    true_solar_time = (minutes_utc + eqtime + 4.0 * longitude) % 1440.0
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)

    lat = np.radians(latitude)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)
    elevation = 90.0 - np.degrees(zenith)

    sin_az = -np.sin(hour_angle) * np.cos(decl)
    cos_az = (np.sin(decl) - np.sin(lat) * cos_zenith) / np.maximum(np.cos(lat) * np.sin(zenith), 1e-9)
    azimuth = (np.degrees(np.arctan2(sin_az, cos_az)) + 360.0) % 360.0

    return pd.DataFrame({"elevation": elevation, "azimuth": azimuth}, index=index)


def clear_sky_ghi(elevation_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance [W/m^2]."""
    sin_h = np.sin(np.radians(np.asarray(elevation_deg, dtype=float)))
    sin_h = np.clip(sin_h, 0.0, 1.0)
    with np.errstate(divide="ignore", over="ignore"):
        ghi = 1098.0 * sin_h * np.exp(-0.059 / np.maximum(sin_h, 1e-6))
    return np.where(sin_h > 0.0, ghi, 0.0)


def irradiance_from_cloud_cover(
    index: pd.DatetimeIndex,
    cloud_cover_pct: np.ndarray | pd.Series | None,
    latitude: float,
    longitude: float,
) -> pd.Series:
    """Estimate GHI [W/m^2] from cloud coverage using Kasten-Czeplak."""
    pos = solar_position(index, latitude, longitude)
    clear = clear_sky_ghi(pos["elevation"].to_numpy())
    if cloud_cover_pct is None:
        cc = np.zeros(len(index))
    else:
        cc = np.asarray(pd.Series(cloud_cover_pct).to_numpy(), dtype=float)
        cc = np.nan_to_num(np.clip(cc, 0.0, 100.0), nan=50.0)
    attenuation = 1.0 - 0.75 * (cc / 100.0) ** 3.4
    return pd.Series(clear * attenuation, index=index, name="solar_ghi")


def condition_to_cloud_cover(condition: str | None) -> float:
    """Map a Home Assistant weather ``condition`` string to cloud cover percent.

    Used as a fallback when the weather integration reports no numeric cloud
    coverage - "soligt eller molnigt" as a plain string is still usable.
    """
    if not condition:
        return 50.0
    table = {
        "sunny": 0.0,
        "clear": 0.0,
        "clear-night": 0.0,
        "windy": 15.0,
        "partlycloudy": 40.0,
        "windy-variant": 45.0,
        "cloudy": 90.0,
        "fog": 95.0,
        "hail": 95.0,
        "lightning": 85.0,
        "lightning-rainy": 95.0,
        "pouring": 100.0,
        "rainy": 95.0,
        "snowy": 100.0,
        "snowy-rainy": 100.0,
        "exceptional": 50.0,
    }
    return table.get(str(condition).strip().lower(), 50.0)
