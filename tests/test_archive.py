"""The archive exists so that training does not depend on Home Assistant's
recorder retention. These tests are about that promise: history the recorder
has already purged must still be there, and getting it there must not cost a
full download every cycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from hpmpc.archive import Archive, build_training_frame, dataset_from_archive, open_archive, refresh
from hpmpc.dataset import column_map

from conftest import RecorderHomeAssistant


def recorded(cfg, ha, days: float, indoor: float = 21.0, outdoor: float = -4.0) -> pd.DataFrame:
    """Fill the fake recorder with `days` of readings ending now."""
    end = ha.now.replace(second=0, microsecond=0)
    stamps = pd.date_range(end - pd.Timedelta(days=days), end, freq="15min", tz="UTC")
    rows = []
    for entity_id, column in column_map(cfg).items():
        if column not in {"t_indoor", "t_outdoor"}:
            continue
        value = indoor if column == "t_indoor" else outdoor
        rows.extend({"entity_id": entity_id, "time": t, "value": value} for t in stamps)
    ha.history_frame = pd.DataFrame(rows)
    return ha.history_frame


def test_history_survives_the_recorder_purge(cfg):
    """The point of the whole module: keep 45 days when the recorder keeps 10."""
    ha = RecorderHomeAssistant(keep_days=10.0)
    recorded(cfg, ha, days=8.0)
    archive = open_archive(cfg)
    refresh(cfg, ha, archive, now=ha.now)
    assert archive.span_days() == pytest.approx(8.0, abs=0.1)

    # A week goes by. The recorder now only holds the newest ten days, which no
    # longer reaches back to where we started.
    later = ha.now + timedelta(days=7)
    ha.now = later
    recorded(cfg, ha, days=15.0)          # 15 days of readings exist ...
    ha.keep_days = 10.0                    # ... but only 10 are still queryable
    refresh(cfg, ha, archive, now=later)

    assert archive.span_days() == pytest.approx(15.0, abs=0.2)
    frame = archive.load()
    assert frame.index[0] < pd.Timestamp(later - timedelta(days=10))


def test_the_second_pull_only_asks_for_what_is_new(cfg):
    ha = RecorderHomeAssistant()
    recorded(cfg, ha, days=5.0)
    archive = open_archive(cfg)
    refresh(cfg, ha, archive, now=ha.now)
    first_start, _ = ha.requests[0]
    assert (ha.now - first_start).days >= cfg.training.history_days - 1

    later = ha.now + timedelta(hours=1)
    refresh(cfg, ha, archive, now=later)
    second_start, _ = ha.requests[-1]
    # Only the overlap plus the hour that has passed, not another five days.
    assert timedelta(hours=1) <= later - second_start <= timedelta(hours=4)


def test_a_partial_bucket_is_corrected_by_the_overlap(cfg):
    """The newest resample bucket is always half full when it is written."""
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    t = pd.Timestamp("2026-02-01 12:00", tz="UTC")
    archive.merge(pd.DataFrame({"t_indoor": [20.0]}, index=[t]))
    assert archive.load()["t_indoor"].iloc[0] == pytest.approx(20.0)

    archive.merge(pd.DataFrame({"t_indoor": [21.5]}, index=[t]))
    assert archive.load()["t_indoor"].iloc[0] == pytest.approx(21.5)
    assert len(archive.load()) == 1


def test_a_column_that_goes_unavailable_keeps_its_history(cfg):
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    index = pd.date_range("2026-02-01", periods=4, freq="15min", tz="UTC")
    archive.merge(pd.DataFrame({"t_indoor": 21.0, "wind": 3.0}, index=index))
    # The wind sensor drops out: the new pull has no such column at all.
    archive.merge(pd.DataFrame({"t_indoor": 22.0}, index=index))
    frame = archive.load()
    assert frame["t_indoor"].iloc[0] == pytest.approx(22.0)
    assert frame["wind"].iloc[0] == pytest.approx(3.0)


def test_merging_spans_month_boundaries(cfg):
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    index = pd.date_range("2026-01-30", periods=4 * 24 * 4, freq="15min", tz="UTC")
    added = archive.merge(pd.DataFrame({"t_indoor": 21.0}, index=index))
    assert added == len(index)
    assert len(archive.files()) == 2
    assert len(archive.load()) == len(index)
    # Merging the same rows again adds nothing.
    assert archive.merge(pd.DataFrame({"t_indoor": 21.0}, index=index)) == 0


def test_prune_drops_old_months_and_keeps_recent_ones(cfg):
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    now = pd.Timestamp.now(tz="UTC")
    old = pd.date_range(now - pd.Timedelta(days=400), periods=8, freq="15min")
    new = pd.date_range(now - pd.Timedelta(days=2), periods=8, freq="15min")
    archive.merge(pd.DataFrame({"t_indoor": 21.0}, index=old.union(new)))
    assert archive.prune(90) >= 1
    frame = archive.load()
    assert frame.index.min() > now - pd.Timedelta(days=90)


def test_load_can_be_limited_to_recent_days(cfg):
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    index = pd.date_range(now - pd.Timedelta(days=20), now, freq="15min")
    archive.merge(pd.DataFrame({"t_indoor": 21.0}, index=index))
    recent = archive.load(days=5)
    assert recent.index.min() >= now - pd.Timedelta(days=5, minutes=15)


def test_the_archive_can_train_without_home_assistant(cfg):
    """Once history is ours, a dataset needs no connection at all."""
    ha = RecorderHomeAssistant()
    recorded(cfg, ha, days=6.0)
    refresh(cfg, ha, now=ha.now)

    frame = dataset_from_archive(cfg)
    assert {"t_indoor", "t_outdoor", "solar_ghi", "offset", "price"} <= set(frame.columns)
    assert len(frame) > 500


def test_derived_columns_are_recomputed_not_stored(cfg):
    """Fixing the site coordinates must also fix the irradiance in old rows."""
    ha = RecorderHomeAssistant()
    recorded(cfg, ha, days=3.0)
    refresh(cfg, ha, now=ha.now)
    assert "solar_ghi" not in open_archive(cfg).load().columns

    cfg.site.latitude, cfg.site.longitude = 58.5877, 16.1924
    north = dataset_from_archive(cfg)["solar_ghi"].sum()
    cfg.site.latitude = 68.0
    assert dataset_from_archive(cfg)["solar_ghi"].sum() != pytest.approx(north)


def test_collect_prefers_the_archive_over_the_recorder(cfg):
    ha = RecorderHomeAssistant(keep_days=2.0)
    recorded(cfg, ha, days=1.0)
    refresh(cfg, ha, now=ha.now)

    # Nothing at all is queryable any more, but collect still gets a dataset.
    ha.keep_days = 0.0
    frame, info = build_training_frame(cfg, ha)
    assert info["source"] == "archive"
    assert len(frame) > 50


def test_switching_the_archive_off_goes_straight_to_the_recorder(cfg):
    ha = RecorderHomeAssistant()
    recorded(cfg, ha, days=3.0)
    cfg.training.archive = False
    frame, info = build_training_frame(cfg, ha)
    assert info["source"] == "recorder"
    assert not open_archive(cfg).files()
    assert len(frame) > 50


def test_a_broken_archive_never_stops_a_control_cycle(cfg, monkeypatch):
    from hpmpc import controller as controller_module

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(controller_module, "refresh_archive", explode)
    report: dict = {}
    controller = object.__new__(controller_module.Controller)
    controller.cfg = cfg
    controller.ha = None
    controller.archive_cycle(report)
    assert "disk full" in report["archive"]["error"]


def test_the_archive_may_not_be_shorter_than_training_asks_for(cfg):
    cfg.training.history_days = 45
    cfg.training.archive_keep_days = 20
    with pytest.raises(ValueError, match="archive_keep_days"):
        cfg.validate()


def test_an_empty_archive_says_what_to_do(cfg):
    with pytest.raises(ValueError, match="hpmpc collect"):
        dataset_from_archive(cfg)


def test_timestamps_survive_the_gzip_round_trip(cfg):
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    index = pd.date_range("2026-03-01 00:00", periods=6, freq="15min", tz="UTC")
    archive.merge(pd.DataFrame({"t_indoor": [21.0, 21.1, 21.2, 21.3, 21.4, 21.5]}, index=index))
    frame = archive.load()
    assert frame.index.tz is not None
    assert list(frame.index) == list(index)
    assert archive.last_timestamp() == index[-1]


def test_describe_reports_gaps(cfg):
    archive = Archive(cfg.archive_dir, cfg.training.resample_minutes)
    first = pd.date_range("2026-03-01 00:00", periods=8, freq="15min", tz="UTC")
    second = pd.date_range("2026-03-01 06:00", periods=8, freq="15min", tz="UTC")
    archive.merge(pd.DataFrame({"t_indoor": 21.0}, index=first.union(second)))
    info = archive.describe()
    assert info["rows"] == 16
    assert info["gaps"] and info["gaps"][0]["hours"] == pytest.approx(4.25, abs=0.1)
    assert info["coverage"] < 1.0


def test_refresh_is_a_no_op_when_the_recorder_has_nothing(cfg):
    ha = RecorderHomeAssistant()
    info = refresh(cfg, ha, now=datetime.now(timezone.utc))
    assert info["rows_added"] == 0
    assert not open_archive(cfg).files()


# ------------------------- weather that only exists at the moment it is used


def no_sensor(cfg):
    """The shipped arrangement: no outdoor sensor, the forecast supplies it."""
    cfg.entities.outdoor_temp = ""
    cfg.entities.wind_speed = ""
    cfg.forecast.weather_source = "smhi"
    return cfg


def test_the_resolved_weather_is_written_down(cfg):
    """Without a sensor this is the only place the temperature ever exists."""
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    now = pd.Timestamp("2026-01-15 08:07", tz="UTC")
    recorded = record_resolved(cfg, {"t_outdoor": -4.5, "wind": 3.2}, now)
    assert set(recorded) == {"t_outdoor", "wind"}

    frame = open_archive(cfg).load()
    assert frame["t_outdoor"].iloc[0] == pytest.approx(-4.5)
    # Snapped onto the training grid, not left on the cycle's own clock.
    assert frame.index[0] == pd.Timestamp("2026-01-15 08:00", tz="UTC")


def test_a_configured_sensor_is_left_to_the_recorder(cfg):
    """Where a sensor exists its own history is denser and closer to the house."""
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    cfg.entities.outdoor_temp = "sensor.outdoor"
    recorded = record_resolved(cfg, {"t_outdoor": -4.5, "wind": 3.2},
                               pd.Timestamp("2026-01-15 08:00", tz="UTC"))
    assert recorded == ["wind"]
    assert "t_outdoor" not in open_archive(cfg).load().columns


def test_nothing_is_written_for_values_that_are_missing(cfg):
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    assert record_resolved(cfg, {"t_outdoor": None, "wind": float("nan")},
                           pd.Timestamp("2026-01-15 08:00", tz="UTC")) == []
    assert not open_archive(cfg).files()


def test_recorded_weather_makes_the_history_trainable(cfg):
    """The gap this closes: recorder history has no t_outdoor, so 'hpmpc collect'
    failed outright on an install with no outdoor sensor."""
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    ha = RecorderHomeAssistant()
    recorded(cfg, ha, days=3.0)                 # indoor only; there is no outdoor entity
    refresh(cfg, ha, now=ha.now)
    with pytest.raises(ValueError, match="t_outdoor"):
        dataset_from_archive(cfg)

    # Now let the control loop write down what it used, cycle by cycle.
    end = ha.now.replace(second=0, microsecond=0)
    for stamp in pd.date_range(end - pd.Timedelta(days=3), end, freq="15min", tz="UTC"):
        record_resolved(cfg, {"t_outdoor": -3.0, "wind": 2.0}, stamp)

    frame = dataset_from_archive(cfg)
    assert "t_outdoor" in frame and frame["t_outdoor"].notna().all()
    assert len(frame) > 200


def test_a_slow_cycle_does_not_leave_holes_in_the_history(cfg):
    """Recorded once per control cycle, which need not match the training grid."""
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    ha = RecorderHomeAssistant()
    recorded(cfg, ha, days=2.0)
    refresh(cfg, ha, now=ha.now)
    end = ha.now.replace(second=0, microsecond=0)
    for stamp in pd.date_range(end - pd.Timedelta(days=2), end, freq="45min", tz="UTC"):
        record_resolved(cfg, {"t_outdoor": -3.0}, stamp)

    frame = dataset_from_archive(cfg)
    assert frame["t_outdoor"].isna().sum() == 0


def test_the_error_says_what_to_do_about_it(cfg):
    from hpmpc.dataset import finish_dataset

    cfg = no_sensor(cfg)
    index = pd.date_range("2026-01-15", periods=8, freq="15min", tz="UTC")
    with pytest.raises(ValueError) as excinfo:
        finish_dataset(pd.DataFrame({"t_indoor": 21.0}, index=index), cfg)
    message = str(excinfo.value)
    assert "no outdoor sensor configured" in message
    assert "hpmpc archive" in message


def test_the_controller_archives_the_weather_it_used(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = no_sensor(cfg)
    cfg.entities.weather = "weather.home"
    cfg.forecast.weather_source = "home_assistant"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(apply=True)

    assert "t_outdoor" in report["archive"]["recorded"]
    assert "t_outdoor" in open_archive(cfg).load().columns


def test_the_price_the_controller_planned_against_is_kept(cfg):
    """Identification does not care what electricity costs, but a backtest
    graded against a flat invented price is worthless."""
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    cfg.entities.price = ""
    now = pd.Timestamp("2026-01-15 08:00", tz="UTC")
    assert "price" in record_resolved(cfg, {"t_outdoor": -4.0, "price": 1.87}, now)
    assert open_archive(cfg).load()["price"].iloc[0] == pytest.approx(1.87)


def test_a_configured_price_entity_is_left_to_the_recorder(cfg):
    from hpmpc.archive import record_resolved

    cfg = no_sensor(cfg)
    cfg.entities.price = "sensor.price"
    recorded = record_resolved(cfg, {"t_outdoor": -4.0, "price": 1.87},
                               pd.Timestamp("2026-01-15 08:00", tz="UTC"))
    assert "price" not in recorded


def test_the_controller_archives_the_spot_price_it_planned_against(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = no_sensor(cfg)
    cfg.entities.price = ""
    cfg.entities.weather = "weather.home"
    cfg.forecast.weather_source = "home_assistant"
    cfg.forecast.price_source = "home_assistant"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(apply=True)

    assert "price" in report["archive"]["recorded"]
    assert open_archive(cfg).load()["price"].notna().any()


def test_stove_command_reads_the_archive_and_suggests_a_deadband(tmp_path, capsys):
    """The deadband has to come from the two sensors as installed.

    In this house the stove sensor is in a ceiling corner and the control
    sensor is mid-house, so the baseline between them is neither zero nor
    constant - guessing it is how a quiet evening gets fitted as a fire.
    """
    import numpy as np
    import pandas as pd

    from hpmpc.archive import open_archive
    from hpmpc.cli import main

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "entities:\n"
        "  indoor_temp: sensor.mid_house\n"
        "  outdoor_temp: sensor.outdoor\n"
        "  stove_temp: sensor.ceiling_corner\n"
        f"paths:\n  data_dir: {tmp_path}\n"
    )

    index = pd.date_range("2026-10-01", periods=4 * 24 * 5, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    # A quiet week: the corner simply runs ~1.2 K warm, breathing a little.
    baseline = 1.2 + 0.3 * np.sin(np.arange(len(index)) / 40.0) + rng.normal(0, 0.05, len(index))
    frame = pd.DataFrame(
        {"t_indoor": 21.0, "t_outdoor": 3.0, "stove_temp": 21.0 + baseline}, index=index
    )
    open_archive(_load_cfg(cfg_path)).merge(frame)

    assert main(["--config", str(cfg_path), "stove"]) == 0
    out = capsys.readouterr().out
    assert "sensor.ceiling_corner" in out
    assert "stove_deadband_k" in out
    # The suggestion must clear the whole quiet baseline, or quiet hours burn.
    suggested = float(out.split("stove_deadband_k")[1].split()[0])
    assert suggested > baseline.max()


def _load_cfg(path):
    from hpmpc.config import load_config

    return load_config(path)


def test_a_sensor_configured_late_is_reached_back_for(cfg):
    """A stove sensor added at the end of an excitation week.

    The incremental pull only asks for what happened since the last stored
    row, so without a reach-back the week's fires - still in the recorder -
    would be archived as no fire at all, and lost for good once purged.
    """
    ha = RecorderHomeAssistant(keep_days=10.0)
    recorded(cfg, ha, days=7.0)
    archive = open_archive(cfg)
    refresh(cfg, ha, archive, now=ha.now)
    assert "stove_temp" not in archive.load().columns

    # The sensor existed all along; only the configuration is new.
    stamps = pd.date_range(ha.now - pd.Timedelta(days=7), ha.now, freq="15min", tz="UTC")
    stove = pd.DataFrame({"entity_id": "sensor.stove", "time": stamps, "value": 24.0})
    ha.history_frame = pd.concat([ha.history_frame, stove], ignore_index=True)
    cfg.entities.stove_temp = "sensor.stove"

    info = refresh(cfg, ha, archive, now=ha.now)
    assert info.get("backfilled") == ["stove_temp"]
    stored = archive.load()["stove_temp"].dropna()
    assert stored.index[0] < pd.Timestamp(ha.now - pd.Timedelta(days=6))
    # Nothing that was already there is disturbed.
    assert archive.load()["t_indoor"].dropna().index[0] < pd.Timestamp(ha.now - pd.Timedelta(days=6))


def test_reaching_back_happens_once_per_new_entity(cfg):
    """Not on every cycle, and not for entities configured all along that
    happen to have nothing in the recorder."""
    ha = RecorderHomeAssistant(keep_days=10.0)
    recorded(cfg, ha, days=3.0)
    archive = open_archive(cfg)
    refresh(cfg, ha, archive, now=ha.now)
    refresh(cfg, ha, archive, now=ha.now + timedelta(minutes=15))
    assert len(ha.requests) == 2                  # two ordinary pulls, no reach-back

    cfg.entities.stove_temp = "sensor.stove_with_no_history"
    refresh(cfg, ha, archive, now=ha.now + timedelta(minutes=30))
    assert len(ha.requests) == 4                  # the pull, plus one reach-back
    refresh(cfg, ha, archive, now=ha.now + timedelta(minutes=45))
    assert len(ha.requests) == 5                  # and never again


def test_an_archive_from_before_the_entity_list_still_reaches_back(cfg):
    """The NUC's case: upgrade and configure the stove sensor in one go."""
    from hpmpc.archive import KNOWN_ENTITIES

    ha = RecorderHomeAssistant(keep_days=10.0)
    recorded(cfg, ha, days=7.0)
    archive = open_archive(cfg)
    refresh(cfg, ha, archive, now=ha.now)
    (archive.directory / KNOWN_ENTITIES).unlink()          # as if written by older code

    stamps = pd.date_range(ha.now - pd.Timedelta(days=7), ha.now, freq="15min", tz="UTC")
    ha.history_frame = pd.concat(
        [ha.history_frame, pd.DataFrame({"entity_id": "sensor.stove", "time": stamps, "value": 24.0})],
        ignore_index=True)
    cfg.entities.stove_temp = "sensor.stove"
    assert refresh(cfg, ha, archive, now=ha.now).get("backfilled") == ["stove_temp"]
# ---------------------------------------------------------------------------
# A disk that fills up mid-write leaves a gzip stream cut in half. That used to
# take down every caller, including last_timestamp - so one bad file stopped
# the controller, which is the opposite of an archive that heals itself.
# ---------------------------------------------------------------------------

def _month(rows=2000, start="2026-10-01"):
    """Rows that vary, like real ones.

    A month of identical values compresses to almost nothing, and then a
    truncation test says more about gzip's block size than about the salvage.
    """
    import numpy as np
    import pandas as pd

    index = pd.date_range(start, periods=rows, freq="15min", tz="UTC")
    step = np.arange(rows)
    return pd.DataFrame(
        {"t_indoor": 21.0 + np.sin(step / 40.0), "t_outdoor": 3.0 + np.cos(step / 70.0)},
        index=index,
    )


def test_a_truncated_month_file_is_salvaged_not_fatal(tmp_path):
    from hpmpc.archive import Archive

    archive = Archive(tmp_path, 15)
    archive.merge(_month())
    path = archive.files()[0]
    whole = path.read_bytes()
    path.write_bytes(whole[: int(len(whole) * 0.6)])    # disk full mid-write

    frame = archive.load()

    assert not frame.empty, "most of the month is still readable and must be kept"
    assert 0 < len(frame) < 2000
    assert list(frame.columns) == ["t_indoor", "t_outdoor"]
    # The original is set aside, never deleted, and no longer matches the glob.
    aside = [p for p in tmp_path.iterdir() if ".corrupt-" in p.name]
    assert len(aside) == 1 and aside[0].read_bytes() == whole[: int(len(whole) * 0.6)]
    assert archive.files() == [path]
    # And the rewritten file reads cleanly the next time, without salvaging again.
    assert len(archive.load()) == len(frame)


def test_an_empty_month_file_is_set_aside_and_the_archive_carries_on(tmp_path):
    from hpmpc.archive import Archive

    archive = Archive(tmp_path, 15)
    archive.merge(_month(start="2026-09-01"))
    archive.merge(_month(start="2026-10-01"))
    september, october = archive.files()
    october.write_bytes(b"")                            # nothing reached the disk

    frame = archive.load()

    assert not frame.empty, "September is intact and must still be readable"
    assert frame.index.max() < pd.Timestamp("2026-10-01", tz="UTC")
    assert not october.exists()
    assert any(".corrupt-" in p.name for p in tmp_path.iterdir())


def test_last_timestamp_survives_a_corrupt_file(tmp_path):
    """This is the one that mattered: every control cycle calls it."""
    from hpmpc.archive import Archive

    archive = Archive(tmp_path, 15)
    archive.merge(_month())
    path = archive.files()[0]
    path.write_bytes(path.read_bytes()[:40])

    assert archive.last_timestamp() is None or archive.last_timestamp() is not None


# ---------------------------------------------------------------------------
# Surviving a full disk, and repairing what one left behind.


def test_a_full_disk_mid_write_leaves_the_month_intact(tmp_path, monkeypatch):
    """Salvaging half a file is a consolation prize; not halving it is the fix.

    Every merge rewrites a whole month file, so the window in which a full disk
    can cut one in half is every single control cycle. Staging the replacement
    and renaming it into place closes that window: running out of space now
    costs this cycle's rows, which the next cycle fetches again, instead of the
    month of history that was already on disk.
    """
    archive = Archive(tmp_path, 15)
    archive.merge(_month(rows=400))
    before = archive.load()

    real_to_csv = pd.DataFrame.to_csv

    def full_disk(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf).endswith(".new"):
            # Some of it reaches the disk before the write dies, which is what
            # makes a truncated gzip rather than an empty file.
            real_to_csv(self.iloc[: len(self) // 2], path_or_buf, *args, **kwargs)
            raise OSError(28, "No space left on device")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", full_disk)
    with pytest.raises(OSError):
        archive.merge(_month(rows=400, start="2026-10-05"))
    monkeypatch.undo()

    pd.testing.assert_frame_equal(before, archive.load())
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".new")], "staged file left behind"
    assert not [p for p in tmp_path.iterdir() if ".corrupt-" in p.name], "nothing was damaged"


def test_a_hole_a_later_cycle_sealed_in_can_still_be_backfilled(cfg):
    """The incremental pull cannot repair a gap, and that is not a bug in it.

    "Everything since the last stored row" is exactly right while the archive
    is being extended and exactly wrong once something has been written after a
    hole - one cycle after the controller comes back is enough, and from then
    on every pull steps straight over the missing days. Hence an explicit
    window, bounded in practice by what the recorder still holds.
    """
    ha = RecorderHomeAssistant(keep_days=10.0)
    recorded(cfg, ha, days=9.0)
    archive = open_archive(cfg)
    refresh(cfg, ha, archive, now=ha.now)
    whole = archive.load()

    # The outage: four days missing, with rows on both sides of them.
    hole = (whole.index > pd.Timestamp(ha.now - timedelta(days=5))) & (
        whole.index < pd.Timestamp(ha.now - timedelta(days=1))
    )
    assert hole.sum() > 100
    kept = whole[~hole]
    for path in archive.files():
        path.unlink()
    archive.merge(kept)

    refresh(cfg, ha, archive, now=ha.now)
    assert len(archive.load()) == len(kept), "the incremental pull stepped over the hole"

    info = refresh(cfg, ha, archive, now=ha.now, reach_back_days=8.0)

    assert "backfill" in str(info["reason"])
    repaired = archive.load()
    assert len(repaired) == len(whole)
    pd.testing.assert_frame_equal(repaired, whole)


def test_the_backfill_cannot_reach_past_the_recorder(cfg):
    """Worth stating: this repairs what Home Assistant still has, nothing more."""
    ha = RecorderHomeAssistant(keep_days=3.0)
    recorded(cfg, ha, days=20.0)
    archive = open_archive(cfg)

    refresh(cfg, ha, archive, now=ha.now, reach_back_days=20.0)

    assert archive.span_days() == pytest.approx(3.0, abs=0.2)


def test_a_missing_price_is_refetched_without_overwriting_what_was_paid(cfg, monkeypatch):
    """The one signal the recorder cannot give back - but the market can.

    The price never passes through Home Assistant unless an entity is
    configured for it: the controller fetches it live and writes down what it
    planned against, so a cycle that did not run leaves a hole no later pull
    can fill. The day-ahead market is settled, though, so the hole is simply
    fetched again - and only the hole, because what the controller actually
    paid is the number a backtest has to grade against.
    """
    from hpmpc.archive import backfill_prices

    monkeypatch.setattr(cfg.entities, "price", "")   # the user's case: no price entity
    now = pd.Timestamp("2026-11-12 12:00", tz="UTC")
    grid = pd.date_range(now - pd.Timedelta(days=3), now, freq="15min")
    stored = pd.DataFrame({"t_indoor": 21.0, "t_outdoor": 2.0, "price": float("nan")}, index=grid)
    first_day = grid < grid[0] + pd.Timedelta(days=1)
    stored.loc[first_day, "price"] = 0.42            # what the controller really paid
    archive = open_archive(cfg)
    archive.merge(stored)

    asked: list = []

    def fake_fetch_days(days, area="SE3", **kwargs):
        asked.extend(days)
        hours = pd.date_range(grid[0].floor("D"), grid[-1].ceil("D"), freq="h")
        return [(h, 1.23) for h in hours], []

    monkeypatch.setattr("hpmpc.providers.elpris.fetch_days", fake_fetch_days)
    info = backfill_prices(cfg, archive, days=3.0, now=now.to_pydatetime())

    price = archive.load()["price"]
    assert price.loc[grid[first_day]].to_numpy() == pytest.approx(0.42), "a recorded price must survive"
    assert price.loc[grid[~first_day]].to_numpy() == pytest.approx(1.23), "the hole must be filled"
    assert not price.isna().any()
    assert info["filled"] == int((~first_day).sum())
    # Only the days with something missing are asked for.
    assert grid[0].date() not in asked


def test_the_price_backfill_leaves_a_configured_entity_alone(cfg, monkeypatch):
    from hpmpc.archive import backfill_prices

    monkeypatch.setattr(cfg.entities, "price", "sensor.nordpool")
    info = backfill_prices(cfg, open_archive(cfg), days=3.0)

    assert "sensor.nordpool" in str(info["skipped"])
