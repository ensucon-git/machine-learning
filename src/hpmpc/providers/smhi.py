"""SMHI open meteorological forecast (pmp3g v2).

Free, keyless, no registration, Nordic coverage, ~10 days ahead. Hourly for the
first day and coarser after that, which the caller interpolates onto the control
grid.

Endpoint::

    https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2
        /geotype/point/lon/{lon}/lat/{lat}/data.json

Note the order: longitude first, and at most six decimals - SMHI rejects more.
The model grid is roughly 2.5 km, so street-level precision is meaningless; any
coordinate inside the same town lands in the same cell.

Parameters used here:

===========  ====================================  ====================
SMHI name    meaning                               unit
===========  ====================================  ====================
``t``        air temperature at 2 m                degrees Celsius
``ws``       wind speed at 10 m                    m/s
``tcc_mean`` total cloud cover                     octas (0-8)
``r``        relative humidity                     percent
===========  ====================================  ====================

Humidity matters more than it looks: it drives how often an air/water heat pump
has to defrost, and defrost is a pure efficiency loss.

Data source: SMHI Open Data, licensed CC BY 4.0.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pandas as pd

from ._http import ProviderError, get_json, read_cache, read_stale_cache, write_cache

log = logging.getLogger(__name__)

BASE_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2"
    "/geotype/point/lon/{lon:.6f}/lat/{lat:.6f}/data.json"
)
OCTAS_TO_PERCENT = 100.0 / 8.0


def forecast_url(latitude: float, longitude: float) -> str:
    return BASE_URL.format(lon=float(longitude), lat=float(latitude))


def parse_forecast(payload: dict[str, Any]) -> pd.DataFrame:
    """Turn an SMHI response into a tidy UTC-indexed frame."""
    series = payload.get("timeSeries") or []
    rows: list[dict[str, Any]] = []
    for entry in series:
        valid = entry.get("validTime")
        if not valid:
            continue
        values = {
            p.get("name"): (p.get("values") or [None])[0]
            for p in entry.get("parameters", [])
            if isinstance(p, dict)
        }
        cloud_octas = values.get("tcc_mean")
        rows.append(
            {
                "time": pd.Timestamp(valid).tz_convert("UTC")
                if pd.Timestamp(valid).tz is not None
                else pd.Timestamp(valid).tz_localize("UTC"),
                "t_outdoor": _number(values.get("t")),
                "wind": _number(values.get("ws")),
                "cloud": None if cloud_octas is None else float(cloud_octas) * OCTAS_TO_PERCENT,
                "humidity": _number(values.get("r")),
            }
        )
    if not rows:
        raise ProviderError("SMHI returned a forecast with no time series")
    frame = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time").set_index("time")
    frame.index.name = "time"
    return frame


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def fetch_forecast(
    latitude: float,
    longitude: float,
    cache_dir: str | None = None,
    cache_minutes: float = 30.0,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch (or reuse a cached) point forecast.

    Returns the forecast and metadata describing where it came from - including
    whether a stale cache had to be used, which the caller should surface rather
    than silently plan against yesterday's weather.
    """
    key = f"smhi_{latitude:.4f}_{longitude:.4f}"
    meta: dict[str, Any] = {"source": "SMHI open data (pmp3g v2)", "latitude": latitude, "longitude": longitude}

    if cache_dir:
        cached = read_cache(cache_dir, key, cache_minutes * 60.0)
        if cached is not None:
            meta["cache"] = "fresh"
            return parse_forecast(cached), meta

    url = forecast_url(latitude, longitude)
    try:
        payload = get_json(url, timeout=timeout, client=client)
    except ProviderError as exc:
        if cache_dir:
            stale = read_stale_cache(cache_dir, key)
            if stale is not None:
                log.warning("SMHI unreachable (%s); using a stale cached forecast", exc)
                meta["cache"] = "stale"
                meta["error"] = str(exc)
                return parse_forecast(stale), meta
        raise

    if cache_dir:
        write_cache(cache_dir, key, payload)
    meta["cache"] = "miss"
    meta["approved_time"] = payload.get("approvedTime")
    meta["reference_time"] = payload.get("referenceTime")
    return parse_forecast(payload), meta
