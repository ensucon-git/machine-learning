"""Day-ahead electricity spot prices per Swedish bidding area.

Source: elprisetjustnu.se, which republishes Nord Pool's day-ahead prices as a
free, keyless JSON file per day and area::

    https://www.elprisetjustnu.se/api/v1/prices/{YYYY}/{MM}-{DD}_{AREA}.json

Each entry looks like::

    {"SEK_per_kWh": 0.26673, "EUR_per_kWh": 0.02308, "EXR": 11.558152,
     "time_start": "2026-08-26T00:00:00+02:00",
     "time_end":   "2026-08-26T01:00:00+02:00"}

Two things the caller must get right:

* **These are raw spot prices excluding VAT, grid transfer and energy tax.**
  What actually decides whether load shifting pays is the *marginal* cost, so
  add your transfer fee and tax (``control.price_addition``) and VAT
  (``control.price_vat_pct``). Skipping that overstates the relative spread
  between cheap and expensive hours and makes the optimiser too eager.
* **Tomorrow's file does not exist until Nord Pool publishes**, shortly after
  13:00 Swedish time. Before that, asking for it returns 404 - which is normal,
  not an error, and is why the horizon quietly shortens in the morning.

Attribution requested by the service: "Elpriser tillhandahålls av
Elprisetjustnu.se".
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from ._http import ProviderError, get_json, read_cache, write_cache

log = logging.getLogger(__name__)

BASE_URL = "https://www.elprisetjustnu.se/api/v1/prices/{year:04d}/{month:02d}-{day:02d}_{area}.json"
AREAS = ("SE1", "SE2", "SE3", "SE4")
PUBLICATION_HOUR_LOCAL = 13
ATTRIBUTION = "Elpriser tillhandahålls av Elprisetjustnu.se"


class PriceUnavailable(ProviderError):
    """Raised when no price data at all could be obtained."""


def price_url(day: date, area: str) -> str:
    return BASE_URL.format(year=day.year, month=day.month, day=day.day, area=area.upper())


def parse_day(payload: Any) -> list[tuple[pd.Timestamp, float]]:
    points: list[tuple[pd.Timestamp, float]] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        start = item.get("time_start")
        value = item.get("SEK_per_kWh")
        if start is None or value is None:
            continue
        stamp = pd.Timestamp(start)
        stamp = stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
        points.append((stamp, float(value)))
    return sorted(points)


def tomorrow_is_published(now: datetime, timezone_name: str = "Europe/Stockholm") -> bool:
    """Nord Pool publishes the next day shortly after 13:00 local time."""
    local = now.astimezone(ZoneInfo(timezone_name))
    return local.hour >= PUBLICATION_HOUR_LOCAL


def fetch_prices(
    area: str = "SE3",
    now: datetime | None = None,
    timezone_name: str = "Europe/Stockholm",
    cache_dir: str | None = None,
    cache_minutes: float = 60.0,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> tuple[list[tuple[pd.Timestamp, float]], dict[str, Any]]:
    """Fetch today's and (when published) tomorrow's spot prices.

    Yesterday is included too, so a controller started just after midnight still
    has the hours behind it. Returns points in UTC with SEK/kWh excluding VAT
    and fees.
    """
    area = area.upper()
    if area not in AREAS:
        raise ValueError(f"Unknown Swedish bidding area '{area}' (expected one of {', '.join(AREAS)})")

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local_today = now.astimezone(ZoneInfo(timezone_name)).date()
    wanted = [local_today - timedelta(days=1), local_today]
    expect_tomorrow = tomorrow_is_published(now, timezone_name)
    if expect_tomorrow:
        wanted.append(local_today + timedelta(days=1))

    points: list[tuple[pd.Timestamp, float]] = []
    meta: dict[str, Any] = {
        "source": "elprisetjustnu.se (Nord Pool day-ahead)",
        "area": area,
        "attribution": ATTRIBUTION,
        "excludes": "VAT, grid transfer and energy tax",
        "days": [],
    }
    errors: list[str] = []

    for day in wanted:
        key = f"elpris_{area}_{day.isoformat()}"
        payload = read_cache(cache_dir, key, cache_minutes * 60.0) if cache_dir else None
        cached = payload is not None
        if payload is None:
            try:
                payload = get_json(price_url(day, area), timeout=timeout, client=client, allow_404=True)
            except ProviderError as exc:
                errors.append(f"{day}: {exc}")
                continue
            if payload is None:
                if day > local_today:
                    meta["tomorrow_published"] = False
                    log.info("Tomorrow's %s prices are not published yet", area)
                else:
                    errors.append(f"{day}: not found")
                continue
            if cache_dir:
                write_cache(cache_dir, key, payload)
        day_points = parse_day(payload)
        if day_points:
            points.extend(day_points)
            meta["days"].append({"date": day.isoformat(), "hours": len(day_points), "cached": cached})
            if day > local_today:
                meta["tomorrow_published"] = True

    if not points:
        raise PriceUnavailable(
            f"No spot prices for {area} could be fetched" + (f" ({'; '.join(errors)})" if errors else "")
        )
    if errors:
        meta["errors"] = errors
    points = sorted(dict(points).items())
    meta["first"] = points[0][0].isoformat()
    meta["last"] = points[-1][0].isoformat()
    if expect_tomorrow and not meta.get("tomorrow_published"):
        meta["warning"] = (
            "It is past 13:00 but tomorrow's prices are still missing; the horizon will "
            "extrapolate the last known price."
        )
    return points, meta
