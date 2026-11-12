"""Snapshots of /data and the config, and getting them back."""

from __future__ import annotations

import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hpmpc import backup
from hpmpc.cli import main


@pytest.fixture
def install(tmp_path):
    """A data directory shaped like the real one, and a config next to it."""
    data = tmp_path / "data"
    (data / "history").mkdir(parents=True)
    (data / "models").mkdir()
    (data / "history" / "2026-10.csv.gz").write_bytes(b"\x1f\x8b archive bytes")
    (data / "history" / "2026-11.csv.gz.new").write_bytes(b"half-written")   # a writer's staging
    (data / "models" / "thermal_model.json").write_text('{"params": {}}', encoding="utf-8")
    (data / "excitation_runs.json").write_text("[]", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("entities: {}\n", encoding="utf-8")
    return data, config, tmp_path / "nas"


def test_a_snapshot_holds_the_archive_the_model_the_runs_and_the_config(install):
    data, config, nas = install
    path = backup.snapshot(data, config, nas, datetime(2026, 11, 12, 4, 15))
    assert path.name == "hpmpc-20261112-041500.tar.gz"
    with tarfile.open(path) as tar:
        names = set(tar.getnames())
    assert {"data/history/2026-10.csv.gz", "data/models/thermal_model.json",
            "data/excitation_runs.json", "config/config.yaml"} <= names
    # A writer's half-finished staging file is not history.
    assert "data/history/2026-11.csv.gz.new" not in names
    # Nothing left behind under a temporary name.
    assert [p.name for p in nas.iterdir()] == [path.name]


def test_restore_puts_everything_back_where_it_was(install, tmp_path):
    data, config, nas = install
    path = backup.snapshot(data, config, nas, datetime(2026, 11, 12, 4, 15))
    new_data, new_config = tmp_path / "new" / "data", tmp_path / "new" / "config"
    counts = backup.restore(path, new_data, new_config)
    assert counts == {"data": 3, "config": 1}
    assert (new_data / "history" / "2026-10.csv.gz").read_bytes() == b"\x1f\x8b archive bytes"
    assert (new_data / "models" / "thermal_model.json").read_text(encoding="utf-8") == '{"params": {}}'
    assert (new_config / "config.yaml").read_text(encoding="utf-8") == "entities: {}\n"


def test_restore_will_not_overwrite_a_live_install_without_force(install):
    """Restoring onto the running machine by mistake would swap weeks of
    history for an older copy."""
    data, config, nas = install
    path = backup.snapshot(data, config, nas, datetime(2026, 11, 12, 4, 15))
    with pytest.raises(backup.BackupError, match="--force"):
        backup.restore(path, data, config)
    assert backup.restore(path, data, config, force=True)["data"] == 3


def test_restore_refuses_a_path_that_escapes(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="data/../../outside")
    with pytest.raises(backup.BackupError, match="leaves"):
        backup.restore(evil, tmp_path / "data", tmp_path / "config")
    assert not (tmp_path / "outside").exists()


def test_pruning_keeps_two_weeks_and_the_first_of_each_month(install):
    data, config, nas = install
    start = datetime(2026, 9, 1, 4, 15)
    for day in range(75):
        backup.snapshot(data, config, nas, start + timedelta(days=day), keep_daily=14, keep_monthly=12)
    kept = [stamp for stamp, _ in backup.snapshots(nas)]
    daily = [s for s in kept if s >= start + timedelta(days=61)]
    assert len(daily) == 14
    assert datetime(2026, 9, 1, 4, 15) in kept and datetime(2026, 10, 1, 4, 15) in kept
    assert len(kept) == 14 + 2      # September and October; November's first is inside the 14


def test_an_unwritable_target_says_what_to_do(install, tmp_path):
    """An NFS export with root_squash is the usual cause, and EACCES alone
    does not say so."""
    data, config, _ = install
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(backup.BackupError, match="root_squash"):
        backup.snapshot(data, config, blocker / "nas")


def test_check_reports_whether_the_newest_snapshot_is_recent(install, monkeypatch, capsys):
    data, config, nas = install
    monkeypatch.setenv("HPMPC_DATA_DIR", str(data))
    cfg = ["--config", str(config / "config.yaml")]
    assert main([*cfg, "backup", "--to", str(nas), "--check"]) == 1     # nothing yet
    assert main([*cfg, "backup", "--to", str(nas)]) == 0
    assert main([*cfg, "backup", "--to", str(nas), "--check"]) == 0
    old = datetime.now() - timedelta(hours=40)
    for _, path in backup.snapshots(nas):
        path.rename(nas / f"{backup.PREFIX}{old:{backup.STAMP}}{backup.SUFFIX}")
    assert main([*cfg, "backup", "--to", str(nas), "--check"]) == 1
    assert "40" in capsys.readouterr().out


def test_backup_needs_no_valid_config(install, monkeypatch):
    """A config broken enough to stop the controller is exactly the one whose
    last good copy matters."""
    data, config, nas = install
    (config / "config.yaml").write_text("this: [is not: valid yaml", encoding="utf-8")
    monkeypatch.setenv("HPMPC_DATA_DIR", str(data))
    assert main(["--config", str(config / "config.yaml"), "backup", "--to", str(nas)]) == 0
    assert len(backup.snapshots(nas)) == 1


def test_restore_command_round_trip(install, tmp_path, monkeypatch):
    data, config, nas = install
    monkeypatch.setenv("HPMPC_DATA_DIR", str(data))
    path = backup.snapshot(data, config, nas, datetime(2026, 11, 12, 4, 15))
    target = tmp_path / "fresh"
    assert main(["--config", str(target / "config" / "config.yaml"), "restore", str(path),
                 "--data-dir", str(target / "data")]) == 0
    assert (target / "data" / "excitation_runs.json").exists()
    assert (target / "config" / "config.yaml").exists()
    assert Path(path).exists()
