"""Build the exogenous forecast the optimiser plans against.

Weather comes from a Home Assistant ``weather`` entity, electricity price from
whichever price integration is installed (Nordpool, ENTSO-e, Tibber, ...). Both
are parsed defensively: a missing forecast degrades to "persist the current
value" rather than crashing the controller.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .ha import HomeAssistant, to_float
from .solar import condition_to_cloud_cover, irradiance_from_cloud_cover

log = logging.getLogger(__name__)

# Attribute names used by the common Swedish/Nordic price integrations.
PRICE_LIST_ATTRIBUTES = (
    "raw_today", "raw_tomorrow",            # nordpool custom component
    "prices_today", "prices_tomorrow",      # entsoe
    "today", "tomorrow",                    # some forks (bare float lists)
    "forecast", "prices", "data",           # tibber / generic
)
PRICE_TIME_KEYS = ("start", "time", "hour", "datetime", "start_time", "from")
PRICE_VALUE_KEYS = ("value", "price", "total", "cost", "electricity_price", "price_ct_per_kwh")


def horizon_index(cfg: Config, now: datetime | None = None) -> pd.DatetimeIndex:
    """UTC timestamps at the start of each control step."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    step = cfg.control.step_minutes
    floor = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % step)
    steps = int(round(cfg.control.horizon_hours * 60 / step))
    return pd.DatetimeIndex([floor + timedelta(minutes=step * i) for i in range(steps)], tz="UTC", name="time")


def _series_from_points(points: Sequence[tuple[pd.Timestamp, float]], index: pd.DatetimeIndex,
                        method: str) -> pd.Series | None:
    if not points:
        return None
    frame = pd.DataFrame(points, columns=["time", "value"]).dropna()
    if frame.empty:
        return None
    frame = frame.drop_duplicates(subset="time", keep="last").set_index("time").sort_index()
    combined = frame["value"].reindex(frame.index.union(index))
    if method == "step":
        combined = combined.ffill().bfill()
    else:
        combined = combined.interpolate(method="time", limit_direction="both")
    return combined.reindex(index)


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_weather_forecast(forecast: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Normalise a Home Assistant weather forecast into a tidy frame."""
    rows = []
    for item in forecast or []:
        if not isinstance(item, dict):
            continue
        ts = _parse_timestamp(item.get("datetime") or item.get("time"))
        if ts is None:
            continue
        cloud = to_float(item.get("cloud_coverage"))
        if cloud is None:
            cloud = condition_to_cloud_cover(item.get("condition"))
        rows.append(
            {
                "time": ts,
                "t_outdoor": to_float(item.get("temperature") or item.get("native_temperature")),
                "wind": to_float(item.get("wind_speed") or item.get("native_wind_speed")),
                "cloud": cloud,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["time", "t_outdoor", "wind", "cloud"])
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def parse_price_attributes(attributes: dict[str, Any], fallback: float | None) -> list[tuple[pd.Timestamp, float]]:
    """Extract an hourly price series from a price entity's attributes."""
    points: list[tuple[pd.Timestamp, float]] = []
    for key in PRICE_LIST_ATTRIBUTES:
        raw = attributes.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        if all(isinstance(x, (int, float)) or x is None for x in raw):
            # Bare hourly list for a calendar day - anchor it to that day.
            day = pd.Timestamp.now(tz="UTC").normalize()
            if "tomorrow" in key:
                day += pd.Timedelta(days=1)
            for hour, value in enumerate(raw):
                if value is None:
                    continue
                points.append((day + pd.Timedelta(hours=hour), float(value)))
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            ts = None
            for tk in PRICE_TIME_KEYS:
                if tk in item:
                    ts = _parse_timestamp(item[tk])
                    if ts is not None:
                        break
            value = None
            for vk in PRICE_VALUE_KEYS:
                if vk in item:
                    value = to_float(item[vk])
                    if value is not None:
                        break
            if ts is not None and value is not None:
                points.append((ts, value))
    if not points and fallback is not None:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        points = [(now, float(fallback))]
    return points


def build_forecast(cfg: Config, ha: HomeAssistant, now: datetime | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the exogenous frame over the horizon plus provenance metadata."""
    index = horizon_index(cfg, now)
    sources: dict[str, Any] = {}
    frame = pd.DataFrame(index=index)

    # ---- weather -------------------------------------------------------
    weather = parse_weather_forecast(ha.weather_forecast(cfg.entities.weather)) if cfg.entities.weather else pd.DataFrame()
    current_outdoor = _current_value(ha, cfg.entities.outdoor_temp)
    current_wind = _current_value(ha, cfg.entities.wind_speed)
    current_cloud = _current_value(ha, cfg.entities.cloud_cover)

    if not weather.empty and weather["t_outdoor"].notna().any():
        points = [(r.time, r.t_outdoor) for r in weather.itertuples() if r.t_outdoor is not None]
        series = _series_from_points(points, index, "time")
        sources["t_outdoor"] = f"weather forecast ({cfg.entities.weather})"
    else:
        series = None
    if series is None or series.isna().all():
        if current_outdoor is None:
            raise ValueError("No outdoor temperature available - neither a forecast nor a current sensor value")
        series = pd.Series(current_outdoor, index=index)
        sources["t_outdoor"] = "persisted current sensor value (no forecast)"
    elif current_outdoor is not None:
        # Anchor the forecast to the measured value and let the bias decay over
        # 6 h; forecasts are often a degree off at the current hour.
        bias = current_outdoor - float(series.iloc[0])
        decay = np.exp(-np.arange(len(index)) * cfg.control.step_minutes / 60.0 / 6.0)
        series = series + bias * decay
        sources["t_outdoor_bias_correction_c"] = round(bias, 2)
    frame["t_outdoor"] = series.to_numpy(dtype=float)

    wind_points = [(r.time, r.wind) for r in weather.itertuples() if getattr(r, "wind", None) is not None] if not weather.empty else []
    wind = _series_from_points(wind_points, index, "time")
    if wind is None or wind.isna().all():
        wind = pd.Series(current_wind if current_wind is not None else 0.0, index=index)
        sources["wind"] = "persisted current value" if current_wind is not None else "assumed calm"
    else:
        sources["wind"] = f"weather forecast ({cfg.entities.weather})"
    frame["wind"] = np.clip(wind.to_numpy(dtype=float), 0.0, None)

    cloud_points = [(r.time, r.cloud) for r in weather.itertuples() if getattr(r, "cloud", None) is not None] if not weather.empty else []
    cloud = _series_from_points(cloud_points, index, "time")
    if cloud is None or cloud.isna().all():
        cloud = pd.Series(current_cloud if current_cloud is not None else 50.0, index=index)
        sources["cloud"] = "persisted current value" if current_cloud is not None else "assumed 50%"
    else:
        sources["cloud"] = f"weather forecast ({cfg.entities.weather})"
    frame["cloud"] = np.clip(cloud.to_numpy(dtype=float), 0.0, 100.0)

    frame["solar_ghi"] = irradiance_from_cloud_cover(
        index, frame["cloud"], cfg.site.latitude, cfg.site.longitude
    ).to_numpy(dtype=float)
    sources["solar_ghi"] = "clear-sky model attenuated by forecast cloud cover"

    # ---- price ---------------------------------------------------------
    price_state = ha.get_state(cfg.entities.price) if cfg.entities.price else None
    fallback = price_state.numeric if price_state else None
    points = parse_price_attributes(price_state.attributes, fallback) if price_state else []
    price = _series_from_points(points, index, "step")
    if price is None or price.isna().all():
        price = pd.Series(fallback if fallback is not None else 1.0, index=index)
        sources["price"] = "flat (no price forecast found)"
    else:
        horizon_end = index[-1]
        known_until = max(t for t, _ in points)
        sources["price"] = f"{cfg.entities.price}, known until {known_until.isoformat()}"
        if known_until < horizon_end:
            sources["price_extrapolated_hours"] = round((horizon_end - known_until).total_seconds() / 3600.0, 1)
    frame["price"] = price.to_numpy(dtype=float) * cfg.control.price_scale + cfg.control.price_addition
    frame["price"] = frame["price"].ffill().bfill().fillna(1.0)

    frame["price_known"] = _price_known_mask(points, index)
    return frame, sources


def _price_known_mask(points: Sequence[tuple[pd.Timestamp, float]], index: pd.DatetimeIndex) -> np.ndarray:
    if not points:
        return np.zeros(len(index), dtype=bool)
    last = max(t for t, _ in points) + pd.Timedelta(hours=1)
    return np.asarray(index <= last, dtype=bool)


def _current_value(ha: HomeAssistant, entity_id: str) -> float | None:
    if not entity_id:
        return None
    state = ha.get_state(entity_id)
    return state.numeric if state else None
