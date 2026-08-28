"""Turn Home Assistant recorder history into a regular training matrix.

Recorder data is event based and irregular. Identification needs a fixed grid,
so every signal is resampled: continuous sensors are averaged and interpolated,
step-like signals (price, the commanded offset) are forward filled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .ha import HomeAssistant
from .ntc import resistance_to_temperature, wiper_to_resistance
from .solar import irradiance_from_cloud_cover

log = logging.getLogger(__name__)

# Signals that are physically continuous and may be interpolated across gaps.
CONTINUOUS = [
    "t_indoor", "t_outdoor", "wind", "cloud", "humidity", "solar_radiation",
    "t_supply", "t_return", "power", "house_l1", "house_l2", "house_l3", "house_power",
]
# Signals that hold their value until explicitly changed.
STEPWISE = ["price", "output_offset", "output_fake_temp", "output_resistance",
            "output_wiper", "pot_wiper", "ev_charging"]

REQUIRED = ["t_indoor", "t_outdoor"]


def column_map(cfg: Config) -> dict[str, str]:
    """entity_id -> dataset column name."""
    e = cfg.entities
    pairs = {
        e.indoor_temp: "t_indoor",
        e.outdoor_temp: "t_outdoor",
        e.wind_speed: "wind",
        e.cloud_cover: "cloud",
        e.solar_radiation: "solar_radiation",
        e.supply_temp: "t_supply",
        e.return_temp: "t_return",
        e.heatpump_power: "power",
        e.outdoor_humidity: "humidity",
        e.house_power_l1: "house_l1",
        e.house_power_l2: "house_l2",
        e.house_power_l3: "house_l3",
        e.house_power_total: "house_power",
        e.ev_charging: "ev_charging",
        e.price: "price",
        e.offset_output: "output_offset",
        e.fake_temperature_output: "output_fake_temp",
        e.resistance_output: "output_resistance",
        e.wiper_output: "output_wiper",
        e.pot_wiper: "pot_wiper",
    }
    return {k: v for k, v in pairs.items() if k}


def fetch_history(cfg: Config, ha: HomeAssistant, days: int | None = None) -> pd.DataFrame:
    days = int(days if days is not None else cfg.training.history_days)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    mapping = column_map(cfg)
    log.info("Fetching %d days of history for %d entities", days, len(mapping))
    return ha.history(list(mapping.keys()), start, end)


def pivot_history(long_frame: pd.DataFrame, mapping: dict[str, str], resample_minutes: int) -> pd.DataFrame:
    """Resample the long history frame onto a regular UTC grid."""
    if long_frame.empty:
        return pd.DataFrame()
    frame = long_frame.copy()
    frame["column"] = frame["entity_id"].map(mapping)
    frame = frame.dropna(subset=["column"])
    if frame.empty:
        return pd.DataFrame()

    rule = f"{int(resample_minutes)}min"
    pieces: dict[str, pd.Series] = {}
    for column, group in frame.groupby("column"):
        series = group.set_index("time")["value"].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        resampled = series.resample(rule).mean()
        if column in STEPWISE:
            # A price or an offset holds until it is changed; interpolating it
            # would invent values that never existed.
            resampled = resampled.ffill()
        else:
            resampled = resampled.interpolate(limit=8, limit_direction="both")
        pieces[str(column)] = resampled

    out = pd.DataFrame(pieces)
    out.index.name = "time"
    return out.sort_index()


def add_derived(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add solar irradiance and the actually applied offset."""
    out = frame.copy()

    if "wind" not in out:
        out["wind"] = 0.0
    out["wind"] = out["wind"].fillna(0.0).clip(lower=0.0)

    if "solar_radiation" in out and out["solar_radiation"].notna().sum() > 0.5 * len(out):
        out["solar_ghi"] = out["solar_radiation"].fillna(0.0).clip(lower=0.0)
    else:
        cloud = out["cloud"] if "cloud" in out else None
        out["solar_ghi"] = irradiance_from_cloud_cover(
            out.index, cloud, cfg.site.latitude, cfg.site.longitude
        ).to_numpy()

    out["offset"] = applied_offset(out, cfg)

    if "price" not in out:
        out["price"] = 1.0
    out["price"] = out["price"].ffill().bfill().fillna(1.0)
    return out


def applied_offset(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Reconstruct the commanded offset in kelvin from whichever output was logged.

    The kelvin entity is preferred because it needs no conversion and therefore
    cannot disagree with what the controller decided. The others are recovered by
    subtracting the outdoor temperature, which is exact for degrees and
    only as good as the NTC table for ohm.

    ``pot_wiper`` comes last but is the most honest of the lot: it is what the
    ESP32 reports it actually drove, so it includes anything the hardware
    clamped or refused. Prefer it over the commanded value only when there is
    no kelvin entity, because it also carries the NTC table's errors.
    """
    zeros = pd.Series(0.0, index=frame.index, name="offset")
    if "output_offset" in frame and frame["output_offset"].notna().any():
        value = frame["output_offset"]
    elif "output_fake_temp" in frame and frame["output_fake_temp"].notna().any():
        value = frame["output_fake_temp"] - frame["t_outdoor"]
    elif "output_resistance" in frame and frame["output_resistance"].notna().any():
        fake = pd.Series(
            resistance_to_temperature(frame["output_resistance"].to_numpy(dtype=float), cfg.ntc),
            index=frame.index,
        )
        value = fake - frame["t_outdoor"]
    else:
        wiper = next(
            (c for c in ("output_wiper", "pot_wiper") if c in frame and frame[c].notna().any()),
            None,
        )
        if wiper is None:
            return zeros
        ohm = wiper_to_resistance(frame[wiper].to_numpy(dtype=float), cfg.pot)
        fake = pd.Series(resistance_to_temperature(ohm, cfg.ntc), index=frame.index)
        value = fake - frame["t_outdoor"]
    return value.astype(float).ffill().fillna(0.0).clip(cfg.control.offset_min, cfg.control.offset_max)


def finish_dataset(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Derive the computed columns and check that the essentials are there.

    Kept separate from the fetch so the same step can be applied to history
    read back from the archive, which stores raw signals only.
    """
    frame = add_derived(frame, cfg)
    missing = [c for c in REQUIRED if c not in frame or frame[c].isna().all()]
    if missing:
        raise ValueError(f"History is missing required signals: {', '.join(missing)}")
    return frame.dropna(subset=REQUIRED)


def build_dataset(cfg: Config, ha: HomeAssistant, days: int | None = None) -> pd.DataFrame:
    long_frame = fetch_history(cfg, ha, days)
    frame = pivot_history(long_frame, column_map(cfg), cfg.training.resample_minutes)
    if frame.empty:
        raise ValueError("No usable history returned by Home Assistant - check entity ids and recorder retention")
    return finish_dataset(frame, cfg)


def save_dataset(frame: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "time"
    return frame


def segments(frame: pd.DataFrame, resample_minutes: int, max_gap_steps: int = 2) -> list[pd.DataFrame]:
    """Split the dataset where the recorder has holes, so identification never
    integrates the model across a gap it cannot see."""
    if frame.empty:
        return []
    step = pd.Timedelta(minutes=resample_minutes)
    delta = frame.index.to_series().diff()
    breaks = np.flatnonzero((delta > step * max_gap_steps).to_numpy())
    pieces = []
    start = 0
    for b in [*breaks, len(frame)]:
        piece = frame.iloc[start:b]
        if len(piece) > 4:
            pieces.append(piece)
        start = b
    return pieces


def describe(frame: pd.DataFrame) -> dict[str, object]:
    """Data-quality summary. ``offset_excitation`` matters most: without any
    variation in the commanded offset the heating-curve gain is only weakly
    identifiable, so run ``hpmpc excite`` for a week first."""
    hours = (frame.index[-1] - frame.index[0]).total_seconds() / 3600.0 if len(frame) > 1 else 0.0
    info: dict[str, object] = {
        "rows": int(len(frame)),
        "span_hours": round(hours, 1),
        "start": str(frame.index[0]) if len(frame) else None,
        "end": str(frame.index[-1]) if len(frame) else None,
        "columns": sorted(frame.columns.tolist()),
        "missing_fraction": {c: round(float(frame[c].isna().mean()), 4) for c in frame.columns},
    }
    if "offset" in frame:
        info["offset_excitation"] = {
            "std": round(float(frame["offset"].std(ddof=0)), 3),
            "min": round(float(frame["offset"].min()), 2),
            "max": round(float(frame["offset"].max()), 2),
            "distinct": int(frame["offset"].round(1).nunique()),
        }
    if "t_outdoor" in frame:
        info["outdoor_range"] = [round(float(frame["t_outdoor"].min()), 1), round(float(frame["t_outdoor"].max()), 1)]
    if "t_indoor" in frame:
        info["indoor_range"] = [round(float(frame["t_indoor"].min()), 1), round(float(frame["t_indoor"].max()), 1)]
    return info
