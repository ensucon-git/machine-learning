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
from .providers import PriceUnavailable, fetch_forecast, fetch_prices
from .providers._http import ProviderError
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
            # A bare list of numbers for one calendar day. The spacing is
            # whatever divides the day evenly - 24 hourly values or 96
            # quarter-hourly ones.
            day = pd.Timestamp.now(tz="UTC").normalize()
            if "tomorrow" in key:
                day += pd.Timedelta(days=1)
            # Only read the length as a resolution when it plausibly covers a
            # whole day: 24 hourly values, or 96 quarter-hourly ones. A partial
            # list means hourly, which is what every integration used before
            # Nord Pool moved to 15-minute settlement.
            spacing = (
                pd.Timedelta(days=1) / len(raw) if len(raw) in {24, 48, 96} else pd.Timedelta(hours=1)
            )
            for slot, value in enumerate(raw):
                if value is None:
                    continue
                points.append((day + spacing * slot, float(value)))
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


def weather_points(cfg: Config, ha: HomeAssistant, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Get a weather forecast from the configured source, falling back sideways.

    SMHI first when configured, then the Home Assistant weather entity, then
    nothing - and the caller persists the current sensor readings. Each step
    down is recorded in the returned metadata rather than hidden, because a
    controller planning 36 hours against a persisted constant should say so.
    """
    sources: dict[str, Any] = {}
    if cfg.forecast.weather_source == "smhi":
        try:
            frame, meta = fetch_forecast(
                cfg.site.latitude,
                cfg.site.longitude,
                cache_dir=cfg.cache_dir,
                cache_minutes=cfg.forecast.weather_cache_minutes,
                timeout=cfg.forecast.timeout,
            )
            sources["weather"] = meta
            return frame, sources
        except (ProviderError, ValueError) as exc:
            log.warning("SMHI forecast failed (%s); falling back to Home Assistant", exc)
            sources["weather_error"] = str(exc)

    if cfg.entities.weather:
        parsed = parse_weather_forecast(ha.weather_forecast(cfg.entities.weather))
        if not parsed.empty:
            frame = parsed.set_index("time")
            if "humidity" not in frame:
                frame["humidity"] = np.nan
            sources.setdefault("weather", {"source": f"Home Assistant weather entity {cfg.entities.weather}"})
            return frame, sources

    sources.setdefault("weather", {"source": "none - persisting current sensor values"})
    return pd.DataFrame(columns=["t_outdoor", "wind", "cloud", "humidity"]), sources


def price_points(cfg: Config, ha: HomeAssistant, now: datetime) -> tuple[list[tuple[pd.Timestamp, float]], dict[str, Any]]:
    """Get spot prices from the configured source, falling back to the price entity."""
    sources: dict[str, Any] = {}
    if cfg.forecast.price_source == "elprisetjustnu":
        try:
            points, meta = fetch_prices(
                area=cfg.forecast.price_area,
                now=now,
                timezone_name=cfg.site.timezone,
                cache_dir=cfg.cache_dir,
                cache_minutes=cfg.forecast.price_cache_minutes,
                timeout=cfg.forecast.timeout,
            )
            sources["price"] = meta
            return points, sources
        except (PriceUnavailable, ProviderError, ValueError) as exc:
            log.warning("Spot price fetch failed (%s); falling back to Home Assistant", exc)
            sources["price_error"] = str(exc)

    state = ha.get_state(cfg.entities.price) if cfg.entities.price else None
    fallback = state.numeric if state else None
    points = parse_price_attributes(state.attributes, fallback) if state else []
    if points:
        sources.setdefault("price", {"source": f"Home Assistant entity {cfg.entities.price}"})
    else:
        sources.setdefault("price", {"source": "flat - no price forecast found"})
    return points, sources


def marginal_price(spot: np.ndarray, cfg: Config) -> np.ndarray:
    """Turn a spot price into the marginal cost of one more kilowatt-hour.

    ``(spot * price_scale + price_addition) * (1 + vat/100)``. The addition is
    grid transfer plus energy tax; leaving it out inflates the *relative* gap
    between cheap and expensive hours and makes the optimiser chase savings
    that are proportionally smaller than they look.
    """
    c = cfg.control
    return (np.asarray(spot, dtype=float) * c.price_scale + c.price_addition) * (1.0 + c.price_vat_pct / 100.0)


def build_forecast(cfg: Config, ha: HomeAssistant, now: datetime | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the exogenous frame over the horizon plus provenance metadata."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    index = horizon_index(cfg, now)
    frame = pd.DataFrame(index=index)

    weather, sources = weather_points(cfg, ha, index)
    current_outdoor = _current_value(ha, cfg.entities.outdoor_temp)
    current_wind = _current_value(ha, cfg.entities.wind_speed)
    current_cloud = _current_value(ha, cfg.entities.cloud_cover)
    current_humidity = _current_value(ha, cfg.entities.outdoor_humidity)

    # ---- outdoor temperature -------------------------------------------
    series = _column_series(weather, "t_outdoor", index, "time")
    if series is None:
        if current_outdoor is None:
            raise ValueError("No outdoor temperature available - neither a forecast nor a current sensor value")
        series = pd.Series(current_outdoor, index=index)
        sources["t_outdoor"] = "persisted current sensor value (no forecast)"
    elif current_outdoor is not None:
        # Anchor the forecast to the measured value and let the bias decay over
        # 6 h; forecasts are routinely a degree off at the current hour, and the
        # first hour is the one the controller acts on.
        bias = current_outdoor - float(series.iloc[0])
        decay = np.exp(-np.arange(len(index)) * cfg.control.step_minutes / 60.0 / 6.0)
        series = series + bias * decay
        sources["t_outdoor_bias_correction_c"] = round(bias, 2)
    frame["t_outdoor"] = series.to_numpy(dtype=float)

    # ---- wind, cloud, humidity -----------------------------------------
    for column, current, default in (
        ("wind", current_wind, 0.0),
        ("cloud", current_cloud, 50.0),
        ("humidity", current_humidity, np.nan),
    ):
        values = _column_series(weather, column, index, "time")
        if values is None or values.isna().all():
            fallback = current if current is not None else default
            values = pd.Series(fallback, index=index)
            sources[column] = "persisted current value" if current is not None else f"assumed {default}"
        frame[column] = values.to_numpy(dtype=float)
    frame["wind"] = np.clip(frame["wind"], 0.0, None)
    frame["cloud"] = np.clip(frame["cloud"], 0.0, 100.0)

    frame["solar_ghi"] = irradiance_from_cloud_cover(
        index, frame["cloud"], cfg.site.latitude, cfg.site.longitude
    ).to_numpy(dtype=float)
    sources["solar_ghi"] = "clear-sky model attenuated by forecast cloud cover"

    # ---- price ----------------------------------------------------------
    points, price_sources = price_points(cfg, ha, now)
    sources.update(price_sources)
    spot = _series_from_points(points, index, "step")
    if spot is None or spot.isna().all():
        current_price = _current_value(ha, cfg.entities.price)
        spot = pd.Series(current_price if current_price is not None else 1.0, index=index)
        sources["price_fallback"] = "flat price"
    elif points:
        resolution = price_resolution(points)
        known_until = max(t for t, _ in points) + resolution
        sources["price_resolution_minutes"] = round(resolution.total_seconds() / 60.0)
        if known_until < index[-1]:
            sources["price_extrapolated_hours"] = round(
                (index[-1] - known_until).total_seconds() / 3600.0, 1
            )
    frame["spot_price"] = spot.ffill().bfill().fillna(1.0).to_numpy(dtype=float)
    frame["price"] = marginal_price(frame["spot_price"].to_numpy(), cfg)
    frame["price_known"] = _price_known_mask(points, index)
    return frame, sources


def _column_series(
    weather: pd.DataFrame, column: str, index: pd.DatetimeIndex, method: str
) -> pd.Series | None:
    if weather.empty or column not in weather:
        return None
    values = weather[column].dropna()
    if values.empty:
        return None
    return _series_from_points(list(values.items()), index, method)


def price_resolution(points: Sequence[tuple[pd.Timestamp, float]]) -> pd.Timedelta:
    """Spacing between published prices.

    Nord Pool settles in 15-minute periods since October 2025, so a day is 96
    prices rather than 24. Nothing downstream cares, as long as the "known
    until" boundary uses the real spacing instead of assuming an hour.
    """
    if len(points) < 2:
        return pd.Timedelta(hours=1)
    stamps = pd.DatetimeIndex([t for t, _ in points]).sort_values()
    gaps = stamps.to_series().diff().dropna()
    return pd.Timedelta(gaps.median()) if not gaps.empty else pd.Timedelta(hours=1)


def _price_known_mask(points: Sequence[tuple[pd.Timestamp, float]], index: pd.DatetimeIndex) -> np.ndarray:
    if not points:
        return np.zeros(len(index), dtype=bool)
    last = max(t for t, _ in points) + price_resolution(points)
    return np.asarray(index <= last, dtype=bool)


def _current_value(ha: HomeAssistant, entity_id: str) -> float | None:
    if not entity_id:
        return None
    state = ha.get_state(entity_id)
    return state.numeric if state else None
