"""Remembering when the house was excited, so retraining never forgets it.

The excitation week is the only stretch of history where the offset moved on a
schedule that had nothing to do with price, weather or the clock - which is
what lets identification tell "the house responds strongly to the offset" from
"it got warmer outside". Ordinary retraining reads the newest
``training.history_days`` of the archive, so six weeks after the experiment it
falls out of the window, and every later fit learns the gain only from offsets
the optimiser chose itself: correlated with price and weather, and therefore
weakly determined. Re-running the experiment every 45 days in midwinter is not
a reasonable thing to ask of anyone.

So excitation runs are written down as they happen, in a small JSON file next
to the archive, and the training dataset always carries them along, however
old, for as long as the archive still holds their rows.

Installs that ran an excitation week before this existed are backfilled once
from the archive itself. That is only sound while no model has been trained:
until then the controller holds ``fallback_offset`` (zero), so a stretch of
non-zero offset can only have come from the experiment. Once a model exists the
optimiser moves the offset too, and the file is the only authority.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

LOG_NAME = "excitation_runs.json"
# A restart of the excite container, or an outage of a few hours, continues
# the same run rather than starting a new one.
JOIN_GAP = timedelta(hours=6)
# Backfill ignores anything shorter: a stray non-zero offset from a test is not
# an experiment the fit should be pinned to.
MIN_DETECTED = timedelta(hours=24)
OFFSET_EPS = 0.05


@dataclass
class Run:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def log_path(cfg: Config) -> Path:
    return Path(cfg.paths.data_dir) / LOG_NAME


def _ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def load_runs(cfg: Config) -> list[Run] | None:
    """The recorded runs, oldest first, or None if nothing was ever recorded."""
    path = log_path(cfg)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return sorted((Run(_ts(r["start"]), _ts(r["end"])) for r in raw.get("runs", [])),
                      key=lambda r: r.start)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("Could not read %s (%s); treating it as absent", path, exc)
        return None


def save_runs(cfg: Config, runs: list[Run]) -> None:
    path = log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": [r.as_dict() for r in sorted(runs, key=lambda r: r.start)]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record(cfg: Config, now: datetime) -> None:
    """Note that an excitation cycle actually ran at ``now``."""
    stamp = _ts(now)
    runs = load_runs(cfg) or []
    if runs and stamp - runs[-1].end <= JOIN_GAP and stamp >= runs[-1].start:
        runs[-1].end = max(runs[-1].end, stamp)
    else:
        runs.append(Run(stamp, stamp))
    save_runs(cfg, runs)


def detect(frame: pd.DataFrame) -> list[Run]:
    """Stretches of non-zero offset in history, for the one-time backfill."""
    if frame.empty or "offset" not in frame:
        return []
    moving = frame.index[frame["offset"].abs().fillna(0.0) > OFFSET_EPS]
    runs: list[Run] = []
    for stamp in moving:
        if runs and stamp - runs[-1].end <= JOIN_GAP:
            runs[-1].end = stamp
        else:
            runs.append(Run(stamp, stamp))
    return [r for r in runs if r.end - r.start >= MIN_DETECTED]


def runs_for(cfg: Config, full_archive: pd.DataFrame) -> list[Run]:
    """The runs to keep, backfilling from the archive if that is still safe."""
    runs = load_runs(cfg)
    if runs is not None:
        return runs
    if cfg.model_path.exists():
        # A model means the optimiser has moved the offset too; history alone
        # can no longer say which stretches were the experiment.
        return []
    from .dataset import add_derived

    found = detect(add_derived(full_archive.copy(), cfg))
    if found:
        save_runs(cfg, found)
        log.info("Recorded %d excitation run(s) found in the archive: %s", len(found),
                 ", ".join(f"{r.start:%Y-%m-%d} ({r.hours / 24:.1f} d)" for r in found))
    return found


def with_excitation(cfg: Config, window: pd.DataFrame, full_archive: pd.DataFrame) -> pd.DataFrame:
    """``window`` plus every recorded excitation run, at whatever age.

    The pieces are disjoint in time; identification splits on the gap between
    them, so nothing is integrated across it.
    """
    runs = runs_for(cfg, full_archive)
    if not runs or full_archive.empty:
        return window
    start = window.index[0] if len(window) else None
    pieces = [window]
    kept: list[str] = []
    for run in runs:
        rows = full_archive[(full_archive.index >= run.start) & (full_archive.index <= run.end)]
        if start is not None:
            rows = rows[rows.index < start]     # the part the window does not already hold
        if len(rows):
            pieces.append(rows)
            kept.append(f"{run.start:%Y-%m-%d} ({len(rows)} rows)")
    if len(pieces) == 1:
        return window
    log.info("Training also includes excitation outside the window: %s", ", ".join(kept))
    frame = pd.concat(pieces).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def describe(cfg: Config) -> list[dict[str, object]]:
    return [{"start": str(r.start), "end": str(r.end), "days": round(r.hours / 24, 1)}
            for r in (load_runs(cfg) or [])]
