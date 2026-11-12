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

import gzip
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config
from .dataset import CONTINUOUS, column_map, finish_dataset, pivot_history
from .ha import HomeAssistant
from .providers._http import ProviderError

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
            _write_atomic(path, merged)
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


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write a month file through a staged copy, so a full disk cannot halve it.

    gzip has no way to say "the rest is missing": a write that runs out of space
    leaves a stream that stops mid-record, and everything after the cut is gone.
    Writing a neighbour file and renaming it into place makes that impossible -
    the rename is atomic, so the old file stays whole until a complete new one
    exists. A disk that fills up then costs this cycle's rows, which the next
    cycle fetches again, instead of the month of history that was already there.

    The staged name does not match the ``*.csv.gz`` glob, so a leftover from a
    crash is never read as history.
    """
    staged = path.with_name(f"{path.name}.new")
    try:
        # compression is explicit: pandas infers it from the extension, and the
        # staged name does not end in .gz, so the file would land uncompressed
        # and be unreadable once it was renamed into place.
        frame.to_csv(staged, index=True, float_format="%.4f", compression="gzip")
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:                        # noqa: BLE001 - see _salvage
        frame = _salvage(path, exc)
    if frame.empty:
        return frame
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "time"
    return frame.sort_index()


def _salvage(path: Path, exc: Exception) -> pd.DataFrame:
    """Recover what a half-written month file still holds, then set it aside.

    A disk filling up mid-write leaves the gzip stream cut off, and pandas
    raises from deep inside its parser - which took down every caller, because
    ``_read`` is also how ``last_timestamp`` and therefore every control cycle
    reads the archive. One bad file stopped the whole controller, which is the
    opposite of what an archive that is supposed to heal itself should do.

    gzip decompresses sequentially, so everything written before the cut is
    still good: read until it dies, drop the half-written last line, keep the
    rest. Losing a month of history to a full disk is the expensive part, not
    the broken bytes at the end of it.

    The original is never deleted, only renamed out of the way - it no longer
    matches the ``*.csv.gz`` glob, so it is not read again, and it is there if
    anyone wants to look. The replacement is written to a temporary file first
    and swapped in, so a disk that is STILL full leaves the original untouched
    rather than destroying it on the way out.
    """
    rows: list[str] = []
    try:
        with gzip.open(path, "rt") as handle:
            # One line at a time, on purpose: readlines() reads to the end
            # before it raises, so the partial content would be thrown away
            # with the exception. Appending as we go keeps everything that
            # made it out of the stream before it died.
            for line in handle:
                rows.append(line)
    except Exception:                               # noqa: BLE001
        pass                    # whatever came out before the stream died
    if rows and not rows[-1].endswith("\n"):
        rows.pop()              # the write stopped in the middle of a line

    salvaged = pd.DataFrame()
    if len(rows) > 1:           # a header alone carries no history
        try:
            salvaged = pd.read_csv(io.StringIO("".join(rows)), index_col=0, parse_dates=True)
        except Exception:                           # noqa: BLE001
            salvaged = pd.DataFrame()

    aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    if salvaged.empty:
        path.rename(aside)
        log.error(
            "%s is unreadable (%s) and nothing could be recovered from it. Moved to %s; "
            "the archive refills this month from whatever the recorder still holds.",
            path.name, exc, aside.name,
        )
        return pd.DataFrame()

    staged = path.with_name(f"{path.name}.rewrite")
    try:
        # compression is explicit: pandas infers it from the extension, and the
        # staged name does not end in .gz, so the replacement would land
        # uncompressed and be unreadable the next time round.
        salvaged.to_csv(staged, index=True, float_format="%.4f", compression="gzip")
    except Exception as write_exc:                  # noqa: BLE001
        staged.unlink(missing_ok=True)
        log.error(
            "%s is damaged (%s); recovered %d rows in memory but could not rewrite it (%s). "
            "Leaving the file alone and carrying on with what was recovered.",
            path.name, exc, len(salvaged), write_exc,
        )
        return salvaged
    path.rename(aside)
    staged.rename(path)
    log.error(
        "%s was damaged (%s) - most likely a disk that filled up mid-write. Recovered %d rows "
        "(%s to %s) and rewrote the file; the original is kept as %s. Any gap left refills from "
        "the recorder on the next cycle.",
        path.name, exc, len(salvaged), salvaged.index[0], salvaged.index[-1], aside.name,
    )
    return salvaged


def _month_end(path: Path) -> datetime:
    period = pd.Period(path.name.removesuffix(".csv.gz"), freq="M")
    return period.end_time.tz_localize("UTC").to_pydatetime()


def open_archive(cfg: Config) -> Archive:
    return Archive(cfg.archive_dir, cfg.training.resample_minutes)


def refresh(cfg: Config, ha: HomeAssistant, archive: Archive | None = None,
            now: datetime | None = None, reach_back_days: float | None = None) -> dict[str, object]:
    """Copy everything the recorder has that the archive does not.

    On an empty archive this reaches back ``training.history_days``, which is
    how a fresh install inherits whatever history Home Assistant happens to
    hold. After that each call asks for a couple of hours, because that is all
    that can have happened since the last control cycle.

    ``reach_back_days`` overrides that and asks for a fixed window regardless of
    what is already stored. The incremental rule is right for a running system
    and exactly wrong for repairing a hole: once anything at all is stored after
    the hole - one cycle is enough - "since the last row" steps straight over it
    and the gap is sealed in for good. Merging is non-destructive, so asking for
    a window that is mostly already present only fills what is missing.
    """
    archive = archive or open_archive(cfg)
    now = now or datetime.now(timezone.utc)
    last = archive.last_timestamp()
    if reach_back_days is not None:
        start = now - timedelta(days=float(reach_back_days))
        reason = f"backfill - asking the recorder for the last {float(reach_back_days):g} days"
    elif last is None:
        start = now - timedelta(days=float(cfg.training.history_days))
        reason = "empty archive - reaching back as far as the recorder goes"
    else:
        start = max(last.to_pydatetime() - OVERLAP, now - timedelta(days=float(cfg.training.history_days)))
        reason = "incremental"
    if start >= now:
        return {"rows_added": 0, "reason": "up to date", "pruned": 0}

    mapping = column_map(cfg)
    # Decided before this cycle's pull: the pull itself brings in the newest
    # rows of a just-configured sensor, and afterwards it would look followed.
    known = _known_entities(archive, mapping) if last is not None else set()
    long_frame = ha.history(list(mapping.keys()), start, now)
    frame = pivot_history(long_frame, mapping, cfg.training.resample_minutes)
    added = archive.merge(frame)
    if last is None:
        _remember_entities(archive, set(mapping))     # this pull already reached back
        backfilled: list[str] = []
    else:
        backfilled = _backfill_new_entities(cfg, ha, archive, mapping, known, now)
    pruned = archive.prune(cfg.training.archive_keep_days)
    log.info("Archive: +%d rows from %s (%s)", added, start.isoformat(timespec="minutes"), reason)
    return {
        "rows_added": int(added),
        "reason": reason,
        "pulled_from": start.isoformat(),
        "pruned": int(pruned),
        "span_days": round(archive.span_days(), 2),
        **({"backfilled": backfilled} if backfilled else {}),
    }


KNOWN_ENTITIES = "entities.json"


def _known_entities(archive: Archive, mapping: dict[str, str]) -> set[str]:
    """The entities this archive has been following."""
    path = archive.directory / KNOWN_ENTITIES
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        pass
    # An archive from before the list existed: what it already holds is what
    # it was following.
    present: set[str] = set()
    for month in archive.files()[-2:]:
        present |= set(_read(month).dropna(axis=1, how="all").columns)
    return {entity for entity, column in mapping.items() if column in present}


def _remember_entities(archive: Archive, entities: set[str]) -> None:
    archive.directory.mkdir(parents=True, exist_ok=True)
    (archive.directory / KNOWN_ENTITIES).write_text(json.dumps(sorted(entities)), encoding="utf-8")


def _backfill_new_entities(cfg: Config, ha: HomeAssistant, archive: Archive,
                           mapping: dict[str, str], known: set[str], now: datetime) -> list[str]:
    """Reach back for an entity that was configured after the archive started.

    The incremental pull only asks for what happened since the last stored row,
    which is right for every signal the archive already follows and wrong for
    one that was just added: the recorder still holds its last week or so, and
    once that is purged nobody can fetch it again. The case that prompted this
    was a stove sensor configured at the end of an excitation week - without it
    the week's fires would have been archived as no fire at all.

    "New" means new to the configuration, which is why the archive keeps the
    list of entities it follows. Asking whether a column has data would be the
    wrong question twice over: the incremental pull has already brought in the
    newest two hours of a just-added sensor by the time anything could look,
    and plenty of entities that have been configured all along simply have
    nothing in the recorder.
    """
    new = {entity: column for entity, column in mapping.items() if entity not in known}
    _remember_entities(archive, known | set(mapping))
    if not new:
        return []
    start = now - timedelta(days=float(cfg.training.history_days))
    frame = pivot_history(ha.history(list(new), start, now), new, cfg.training.resample_minutes)
    if frame.empty:
        return []
    archive.merge(frame)
    found = sorted(c for c in new.values() if c in frame.columns)
    if found:
        log.info("Archive: reached back to %s for newly configured %s",
                 start.isoformat(timespec="minutes"), ", ".join(found))
    return found


def backfill_prices(cfg: Config, archive: Archive | None = None, days: float = 14.0,
                    now: datetime | None = None, client: object | None = None) -> dict[str, object]:
    """Refetch the spot price for days the archive has no price for.

    The price is the one archived signal the recorder cannot give back. Unless
    an entity is configured for it, it never passes through Home Assistant at
    all: the controller fetches it live and writes down the number it used, so
    any cycle that did not run leaves a hole that no later pull can fill. The
    day-ahead market is settled history, though, and elprisetjustnu serves any
    past day, so the hole can simply be fetched again.

    The house does not care what electricity costs and identification does not
    either - this matters for the backtest, which grades savings on exactly this
    number, and would otherwise grade those days against a flat invention.

    Only days with something missing are requested, and the refetched price is
    blanked wherever the archive already holds one, so this can never overwrite
    what the controller actually paid.
    """
    from .providers.elpris import fetch_days

    archive = archive or open_archive(cfg)
    if cfg.entities.price:
        return {"skipped": f"entities.price is set to {cfg.entities.price} - the recorder has it"}

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    step = pd.Timedelta(minutes=int(cfg.training.resample_minutes))
    start = pd.Timestamp(now - timedelta(days=float(days))).ceil(step)
    end = pd.Timestamp(now).floor(step)
    if end <= start:
        return {"filled": 0, "days": []}
    grid = pd.date_range(start, end, freq=step)

    stored = archive.load()
    have = stored["price"].reindex(grid) if "price" in stored.columns else pd.Series(np.nan, index=grid)
    gaps = have.isna()
    if not bool(gaps.any()):
        return {"filled": 0, "days": [], "reason": "every step in the window already has a price"}

    zone = ZoneInfo(cfg.site.timezone)
    wanted = sorted({stamp.tz_convert(zone).date() for stamp in grid[gaps]})
    try:
        points, unavailable = fetch_days(wanted, cfg.forecast.price_area, client=client)
    except (ProviderError, ValueError) as exc:    # PriceUnavailable is a ProviderError
        return {"filled": 0, "days": [d.isoformat() for d in wanted], "error": str(exc)}
    if not points:
        return {"filled": 0, "days": [d.isoformat() for d in wanted],
                "error": "; ".join(unavailable) or "no prices returned"}

    series = pd.Series({stamp: value for stamp, value in points}).sort_index()
    # The market publishes a step function - the price holds until the next
    # quoted start - so forward fill onto our grid rather than interpolating.
    on_grid = series.reindex(series.index.union(grid)).ffill().reindex(grid)
    on_grid = on_grid.where(gaps)          # never overwrite a price we recorded
    on_grid = on_grid.dropna()
    if on_grid.empty:
        return {"filled": 0, "days": [d.isoformat() for d in wanted],
                "error": "the fetched days did not cover any missing step"}

    archive.merge(pd.DataFrame({"price": on_grid}))
    result: dict[str, object] = {
        "filled": int(len(on_grid)),
        "days": [d.isoformat() for d in wanted],
        "from": str(on_grid.index[0]),
        "to": str(on_grid.index[-1]),
    }
    if unavailable:
        result["unavailable"] = unavailable
    log.info("Archive: refetched %d missing price steps over %d days", len(on_grid), len(wanted))
    return result


# Signals the recorder cannot supply when no entity is configured for them:
# they come from the forecast provider, live, and are gone by the next cycle
# unless something writes them down. Column name -> the entities.* field that
# would otherwise carry it.
RESOLVED_SIGNALS = {
    "t_outdoor": "outdoor_temp",
    "wind": "wind_speed",
    "cloud": "cloud_cover",
    "humidity": "outdoor_humidity",
    # The spot price comes straight from elprisetjustnu and never passes through
    # Home Assistant either. Identification does not care - the house does not
    # know what electricity costs - but a backtest replayed against a flat
    # made-up price is worthless, and that is the number it grades savings on.
    "price": "price",
}


def record_resolved(cfg: Config, values: dict[str, float | None], now: datetime,
                    archive: Archive | None = None) -> list[str]:
    """Write down what the controller actually used, for signals nothing else logs.

    Without an outdoor sensor the temperature comes from SMHI at the moment it
    is needed and is never seen by Home Assistant, so the recorder has no
    history of it and identification has nothing to fit against - the model
    needs to know what it was outside, not only what it is now. The control
    loop knows, once per cycle, so it writes it into our own archive.

    Only signals with no configured entity are recorded: where a sensor exists
    the recorder's own history is denser and closer to the house.
    """
    archive = archive or open_archive(cfg)
    row: dict[str, float] = {}
    for column, field in RESOLVED_SIGNALS.items():
        if getattr(cfg.entities, field, ""):
            continue                        # the recorder has this one covered
        value = values.get(column)
        if value is not None and np.isfinite(float(value)):
            row[column] = float(value)
    if not row:
        return []
    stamp = pd.Timestamp(now).tz_convert("UTC").floor(f"{int(cfg.training.resample_minutes)}min")
    archive.merge(pd.DataFrame(row, index=[stamp]))
    return sorted(row)


def dataset_from_archive(cfg: Config, archive: Archive | None = None,
                         days: float | None = None) -> pd.DataFrame:
    """Build a training dataset out of the archive instead of the recorder."""
    archive = archive or open_archive(cfg)
    frame = archive.load(days if days is not None else cfg.training.history_days)
    if cfg.training.keep_excitation and not frame.empty:
        # The experiment is the only history where the offset moved for no
        # reason the fit could confuse with weather; it must not age out.
        from .excitation import with_excitation

        frame = with_excitation(cfg, frame, archive.load())
    if frame.empty:
        raise ValueError(
            f"The archive at {archive.directory} is empty - run 'hpmpc collect' while the "
            "controller has access to Home Assistant"
        )
    # Signals the control loop records itself arrive once per cycle, which need
    # not line up with every grid step. Bridge the small gaps the same way the
    # recorder path does, rather than dropping otherwise complete rows.
    for column in CONTINUOUS:
        if column in frame:
            frame[column] = frame[column].interpolate(limit=8, limit_direction="both")
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
