"""Our own copy of the history the model trains on.

Home Assistant's recorder is a rolling window - ten days by default - purged on
a schedule that belongs to another system. Identification wants six weeks and
the power split wants a month, so the usual advice is "raise purge_keep_days to
45 and don't touch it". That makes the model depend on a setting nobody will
remember, in a database that gets restored from a backup, moved to a new
machine, or trimmed when the SD card fills up.

So the controller keeps its own archive instead. Every cycle it asks the
recorder only for what has happened since the last row it stored, and appends
that. Recorder retention then has to outlast the longest gap between two
control cycles - hours, not weeks - and the archive keeps everything it has
already seen, for as long as ``training.archive_keep_days``.

The two are never in conflict: delete the archive and it refills from whatever
the recorder still holds; shorten recorder retention and the archive keeps what
it copied before.

Only the raw resampled signals are stored. Everything derived - solar
irradiance from cloud cover, the applied offset in kelvin - is recomputed on
load, so correcting the NTC table or the site coordinates also corrects the
history that is read through them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import Config
from .dataset import column_map, finish_dataset, pivot_history
from .ha import HomeAssistant

log = logging.getLogger(__name__)

# Re-pull a little more than strictly necessary. The newest resample bucket is
# always partial when it is written, and gets averaged properly once the
# overlap covers it again.
OVERLAP = timedelta(hours=2)

_STAMP = "%Y-%m"


@dataclass
class Archive:
    """Month-sized gzipped CSV files under ``directory``, indexed by UTC time."""

    directory: Path
    resample_minutes: int = 15

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    # ------------------------------------------------------------- reading

    def files(self) -> list[Path]:
        return sorted(self.directory.glob("*.csv.gz"))

    def load(self, days: float | None = None) -> pd.DataFrame:
        """Read the archive back, newest ``days`` of it, raw columns only."""
        paths = self.files()
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=float(days))
            # A month file is kept if it can possibly contain rows after the
            # cutoff; the row filter below does the exact work.
            paths = [p for p in paths if _month_end(p) >= cutoff]
        pieces = [_read(p) for p in paths]
        pieces = [p for p in pieces if not p.empty]
        if not pieces:
            return pd.DataFrame()
        frame = pd.concat(pieces).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        if days is not None:
            frame = frame[frame.index >= pd.Timestamp(cutoff)]
        frame.index.name = "time"
        return frame

    def last_timestamp(self) -> pd.Timestamp | None:
        paths = self.files()
        if not paths:
            return None
        for path in reversed(paths):
            frame = _read(path)
            if not frame.empty:
                return frame.index[-1]
        return None

    def span_days(self) -> float:
        frame = self.load()
        if len(frame) < 2:
            return 0.0
        return float((frame.index[-1] - frame.index[0]).total_seconds() / 86400.0)

    # ------------------------------------------------------------- writing

    def merge(self, frame: pd.DataFrame) -> int:
        """Fold new rows in, newest reading wins. Returns rows actually added.

        Where the new frame has no value for a column - the entity was
        unavailable, or is not configured any more - the stored value survives.
        Rewriting a whole month costs a hundred kilobytes, and only the months
        the new rows touch are rewritten.
        """
        if frame is None or frame.empty:
            return 0
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        self.directory.mkdir(parents=True, exist_ok=True)

        added = 0
        for month, chunk in frame.groupby(frame.index.strftime(_STAMP)):
            path = self.directory / f"{month}.csv.gz"
            existing = _read(path)
            if existing.empty:
                merged, new_rows = chunk, len(chunk)
            else:
                new_rows = int(len(chunk.index.difference(existing.index)))
                merged = chunk.combine_first(existing).sort_index()
            merged.index.name = "time"
            merged.to_csv(path, index=True, float_format="%.4f")
            added += new_rows
        return added

    def prune(self, keep_days: float) -> int:
        """Drop whole month files that are entirely older than the cutoff."""
        if keep_days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=float(keep_days))
        removed = 0
        for path in self.files():
            if _month_end(path) < cutoff:
                path.unlink()
                removed += 1
        return removed

    # ------------------------------------------------------------ reporting

    def describe(self) -> dict[str, object]:
        frame = self.load()
        files = self.files()
        info: dict[str, object] = {
            "directory": str(self.directory),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "rows": int(len(frame)),
            "span_days": round(self.span_days(), 2),
            "start": str(frame.index[0]) if len(frame) else None,
            "end": str(frame.index[-1]) if len(frame) else None,
        }
        if len(frame) > 1:
            step = pd.Timedelta(minutes=self.resample_minutes)
            gaps = frame.index.to_series().diff()
            holes = gaps[gaps > step * 2]
            info["gaps"] = [
                {"after": str(t - g), "hours": round(g.total_seconds() / 3600.0, 1)}
                for t, g in holes.items()
            ][-10:]
            expected = (frame.index[-1] - frame.index[0]) / step + 1
            info["coverage"] = round(float(len(frame) / expected), 4)
        return info


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return frame
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "time"
    return frame.sort_index()


def _month_end(path: Path) -> datetime:
    period = pd.Period(path.name.removesuffix(".csv.gz"), freq="M")
    return period.end_time.tz_localize("UTC").to_pydatetime()


def open_archive(cfg: Config) -> Archive:
    return Archive(cfg.archive_dir, cfg.training.resample_minutes)


def refresh(cfg: Config, ha: HomeAssistant, archive: Archive | None = None,
            now: datetime | None = None) -> dict[str, object]:
    """Copy everything the recorder has that the archive does not.

    On an empty archive this reaches back ``training.history_days``, which is
    how a fresh install inherits whatever history Home Assistant happens to
    hold. After that each call asks for a couple of hours, because that is all
    that can have happened since the last control cycle.
    """
    archive = archive or open_archive(cfg)
    now = now or datetime.now(timezone.utc)
    last = archive.last_timestamp()
    if last is None:
        start = now - timedelta(days=float(cfg.training.history_days))
        reason = "empty archive - reaching back as far as the recorder goes"
    else:
        start = max(last.to_pydatetime() - OVERLAP, now - timedelta(days=float(cfg.training.history_days)))
        reason = "incremental"
    if start >= now:
        return {"rows_added": 0, "reason": "up to date", "pruned": 0}

    mapping = column_map(cfg)
    long_frame = ha.history(list(mapping.keys()), start, now)
    frame = pivot_history(long_frame, mapping, cfg.training.resample_minutes)
    added = archive.merge(frame)
    pruned = archive.prune(cfg.training.archive_keep_days)
    log.info("Archive: +%d rows from %s (%s)", added, start.isoformat(timespec="minutes"), reason)
    return {
        "rows_added": int(added),
        "reason": reason,
        "pulled_from": start.isoformat(),
        "pruned": int(pruned),
        "span_days": round(archive.span_days(), 2),
    }


def dataset_from_archive(cfg: Config, archive: Archive | None = None,
                         days: float | None = None) -> pd.DataFrame:
    """Build a training dataset out of the archive instead of the recorder."""
    archive = archive or open_archive(cfg)
    frame = archive.load(days if days is not None else cfg.training.history_days)
    if frame.empty:
        raise ValueError(
            f"The archive at {archive.directory} is empty - run 'hpmpc collect' while the "
            "controller has access to Home Assistant"
        )
    return finish_dataset(frame, cfg)


def build_training_frame(cfg: Config, ha: HomeAssistant,
                         days: float | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Refresh the archive and build a dataset from it, or go straight to the
    recorder if the archive is switched off."""
    if not cfg.training.archive:
        from .dataset import build_dataset

        return build_dataset(cfg, ha, int(days) if days else None), {"source": "recorder"}
    archive = open_archive(cfg)
    info = dict(refresh(cfg, ha, archive))
    info["source"] = "archive"
    return dataset_from_archive(cfg, archive, days), info
