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
