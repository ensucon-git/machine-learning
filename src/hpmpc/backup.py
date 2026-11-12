"""Snapshots of everything hpmpc has learned, somewhere other than this machine.

    hpmpc backup --to /backup                 # one snapshot now
    hpmpc backup --to /backup --daily-at 04:15  # the sidecar's loop
    hpmpc backup --to /backup --check         # healthcheck: is the newest recent?
    hpmpc restore /backup/hpmpc-20261112-041500.tar.gz

Most of /data cannot be fetched again. The archive holds history the recorder
purged long ago (it keeps about ten days), the excitation runs are a week of
deliberate experiment that cannot be repeated mid-winter, and the model is a
month of fitting on both. A dead disk in the NUC would take all of it.

This runs as its own process - in compose, its own container - and never
inside the control loop. A backup target is typically a network share, and a
hard NFS mount that stops answering blocks the process touching it in
uninterruptible sleep. That must not be the process feeding the sensor
emulator, which falls back to its failsafe within hours of the writes stopping.

Nothing here reads the configuration: a config.yaml broken enough to stop the
controller is exactly the case where the last good copy of it matters.
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

PREFIX = "hpmpc-"
SUFFIX = ".tar.gz"
STAMP = "%Y%m%d-%H%M%S"
_NAME = re.compile(rf"^{re.escape(PREFIX)}(\d{{8}}-\d{{6}}){re.escape(SUFFIX)}$")

# Staging names the writers use for their own atomic replace. A copy caught
# halfway is worthless and the finished file is captured next to it anyway.
_SKIP_SUFFIXES = (".new", ".rewrite", ".partial", ".tmp")


class BackupError(RuntimeError):
    """A snapshot could not be written, or a restore would destroy data."""


def _skip(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = info.name.rsplit("/", 1)[-1]
    if name.endswith(_SKIP_SUFFIXES) or name.startswith(".swap_"):
        return None
    return info


def snapshots(dest: Path) -> list[tuple[datetime, Path]]:
    """Every finished snapshot in ``dest``, oldest first."""
    found = []
    if not dest.is_dir():
        return found
    for path in dest.iterdir():
        match = _NAME.match(path.name)
        if match:
            found.append((datetime.strptime(match.group(1), STAMP), path))
    return sorted(found)


def _check_writable(dest: Path) -> None:
    """Fail with a reason a person can act on, not a bare EACCES."""
    try:
        dest.mkdir(parents=True, exist_ok=True)
        probe = dest / ".hpmpc-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise BackupError(
            f"Cannot write to {dest}: {exc}. If this is an NFS share, the container "
            "writes as root, and a share exported with root_squash maps root to "
            "nobody - map root to a user that owns the directory on the NAS, or "
            "make the directory writable for that user."
        ) from exc


def snapshot(data_dir: Path, config_dir: Path, dest: Path, now: datetime | None = None,
             keep_daily: int = 14, keep_monthly: int = 12) -> Path:
    """Write one verified snapshot of ``data_dir`` and ``config_dir`` into ``dest``.

    Written under a temporary name and renamed when complete, so a snapshot
    that exists is a snapshot that finished; then read back in full before
    the rename, because a backup nobody has ever opened is a hope.
    """
    now = now or datetime.now()
    data_dir, config_dir, dest = Path(data_dir), Path(config_dir), Path(dest)
    if not data_dir.is_dir():
        raise BackupError(f"{data_dir} does not exist - nothing to back up")
    _check_writable(dest)

    final = dest / f"{PREFIX}{now:{STAMP}}{SUFFIX}"
    partial = dest / f".{final.name}.partial"
    try:
        for attempt in range(2):
            try:
                with tarfile.open(partial, "w:gz") as tar:
                    tar.add(data_dir, arcname="data", filter=_skip)
                    if config_dir.is_dir():
                        tar.add(config_dir, arcname="config", filter=_skip)
                break
            except OSError:
                # A file rewritten in place while being read. Every writer
                # that matters replaces atomically; the rest are small JSON
                # files that are whole again a moment later.
                if attempt:
                    raise
                time.sleep(5)
        members = verify(partial)
        os.replace(partial, final)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    log.info("Backup written: %s (%d files, %.1f MB)", final, members,
             final.stat().st_size / 1e6)
    removed = prune(dest, keep_daily, keep_monthly)
    if removed:
        log.info("Removed %d old snapshot(s)", len(removed))
    return final


def verify(path: Path) -> int:
    """Read every member back. Returns how many files the snapshot holds."""
    count = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if member.isfile():
                handle = tar.extractfile(member)
                while handle.read(1 << 20):
                    pass
                count += 1
    if count == 0:
        raise BackupError(f"{path} contains no files")
    return count


def prune(dest: Path, keep_daily: int = 14, keep_monthly: int = 12) -> list[Path]:
    """Keep the newest ``keep_daily`` snapshots, plus the first of each month for
    ``keep_monthly`` months - a slow corruption can take more than two weeks
    to notice, and month files are small enough that a year of them costs little."""
    found = snapshots(dest)
    keep = {path for _, path in found[-keep_daily:]} if keep_daily > 0 else set()
    first_of_month: dict[tuple[int, int], Path] = {}
    for stamp, path in found:
        first_of_month.setdefault((stamp.year, stamp.month), path)
    for key in sorted(first_of_month)[-keep_monthly:] if keep_monthly > 0 else []:
        keep.add(first_of_month[key])
    removed = []
    for _, path in found:
        if path not in keep:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def newest_age(dest: Path, now: datetime | None = None) -> timedelta | None:
    found = snapshots(Path(dest))
    if not found:
        return None
    return (now or datetime.now()) - found[-1][0]


def run_daily(data_dir: Path, config_dir: Path, dest: Path, at: str, keep_daily: int = 14,
              keep_monthly: int = 12, poll_seconds: float = 600.0) -> None:  # pragma: no cover - a loop
    """Write a snapshot once a day at ``at`` (local HH:MM), and at start if the
    newest one is more than a day old. Errors are logged and retried next
    poll; the loop itself never exits."""
    hour, minute = (int(part) for part in at.split(":"))
    while True:
        now = datetime.now()
        try:
            age = newest_age(dest, now)
            due_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            last = snapshots(Path(dest))[-1][0] if age is not None else None
            if age is None or age > timedelta(hours=26) or (now >= due_today and last < due_today):
                snapshot(data_dir, config_dir, dest, now, keep_daily, keep_monthly)
        except Exception as exc:
            log.error("Backup failed: %s", exc)
        time.sleep(poll_seconds)


def restore(archive: Path, data_dir: Path, config_dir: Path, force: bool = False) -> dict[str, int]:
    """Unpack a snapshot into ``data_dir`` and ``config_dir``.

    Refuses to write over a data directory that already holds a model or an
    archive unless ``force`` - restoring onto a running install by mistake
    would replace weeks of history with an older copy.
    """
    archive, data_dir, config_dir = Path(archive), Path(data_dir), Path(config_dir)
    if not archive.is_file():
        raise BackupError(f"No snapshot at {archive}")
    occupied = [p for p in (data_dir / "models", data_dir / "history") if p.exists() and any(p.iterdir())]
    if occupied and not force:
        raise BackupError(
            f"{data_dir} already holds {', '.join(p.name for p in occupied)}. Restoring would "
            "replace it with the snapshot. Stop the controller first and pass --force if that "
            "is what you want."
        )
    targets = {"data": data_dir.resolve(), "config": config_dir.resolve()}
    counts = {"data": 0, "config": 0}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            top, _, rest = member.name.partition("/")
            if top not in targets or not rest:
                continue
            target = (targets[top] / rest).resolve()
            if targets[top] not in target.parents:
                raise BackupError(f"Refusing a path that leaves its directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, open(target, "wb") as sink:
                    while chunk := source.read(1 << 20):
                        sink.write(chunk)
                os.utime(target, (member.mtime, member.mtime))
                counts[top] += 1
    return counts
