"""The excitation week is the one stretch of history the offset gain is learned
from. These tests are about the promise that it never ages out of training."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from hpmpc.archive import dataset_from_archive, open_archive
from hpmpc.excitation import load_runs, record, save_runs, Run
from hpmpc.model.thermal import ThermalParams, save_params

NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0, minute=0)


def fill_archive(cfg, days: float = 70.0, excite_from: float = 60.0, excite_to: float = 53.0) -> None:
    """`days` of history; between `excite_from` and `excite_to` days ago the
    offset hops every six hours, as 'hpmpc excite' does, and is zero otherwise."""
    stamps = pd.date_range(NOW - timedelta(days=days), NOW, freq="15min", tz="UTC")
    age = (pd.Timestamp(NOW) - stamps) / pd.Timedelta(days=1)
    rng = np.random.default_rng(0)
    blocks = (stamps.asi8 // (6 * 3600 * 10**9))
    hop = pd.Series(blocks).map(lambda b: rng.uniform(-4, 3)).to_numpy()
    offset = np.where((age <= excite_from) & (age >= excite_to), hop, 0.0)
    frame = pd.DataFrame(
        {"t_indoor": 21.0, "t_outdoor": -4.0, "output_offset": offset}, index=stamps
    )
    frame.index.name = "time"
    open_archive(cfg).merge(frame)


def excited_rows(frame: pd.DataFrame) -> int:
    return int((frame["offset"].abs() > 0.05).sum())


def test_a_week_older_than_the_window_is_still_trained_on(cfg):
    fill_archive(cfg)
    frame = dataset_from_archive(cfg, days=45)
    # About seven days of 15-minute rows, all from before the 45-day window.
    assert excited_rows(frame) > 6 * 96
    assert frame.index[0] < pd.Timestamp(NOW - timedelta(days=50))


def test_without_the_feature_the_week_ages_out(cfg):
    """The failure this exists for: six weeks on, the experiment is gone."""
    cfg.training.keep_excitation = False
    fill_archive(cfg)
    assert excited_rows(dataset_from_archive(cfg, days=45)) == 0


def test_the_quiet_rows_around_the_old_run_are_not_dragged_in(cfg):
    fill_archive(cfg)
    frame = dataset_from_archive(cfg, days=45)
    outside = frame[frame.index < pd.Timestamp(NOW - timedelta(days=45))]
    run = load_runs(cfg)[0]
    assert len(outside) > 0
    assert outside.index.min() >= run.start and outside.index.max() <= run.end


def test_a_run_from_before_this_existed_is_backfilled_once(cfg):
    assert load_runs(cfg) is None
    fill_archive(cfg)
    dataset_from_archive(cfg, days=45)
    runs = load_runs(cfg)
    assert len(runs) == 1
    assert runs[0].hours == pytest.approx(7 * 24, abs=7)


def test_no_backfill_once_a_model_exists(cfg):
    """After training the optimiser moves the offset as well; history alone can
    no longer say which stretches were the experiment."""
    save_params(cfg.model_path, ThermalParams(), {})
    fill_archive(cfg)
    assert excited_rows(dataset_from_archive(cfg, days=45)) == 0
    assert load_runs(cfg) is None


def test_a_recorded_run_wins_over_detection(cfg):
    save_params(cfg.model_path, ThermalParams(), {})
    fill_archive(cfg)
    start = pd.Timestamp(NOW - timedelta(days=60))
    save_runs(cfg, [Run(start, start + pd.Timedelta(days=7))])
    assert excited_rows(dataset_from_archive(cfg, days=45)) > 6 * 96


def test_cycles_extend_one_run_and_a_long_break_starts_another(cfg):
    t0 = NOW - timedelta(days=3)
    for k in range(8):
        record(cfg, t0 + timedelta(minutes=15 * k))
    record(cfg, t0 + timedelta(days=2))
    runs = load_runs(cfg)
    assert len(runs) == 2
    assert runs[0].hours == pytest.approx(1.75)


def test_excitation_cycles_are_recorded_and_paused_ones_are_not(cfg, fake_ha):
    from hpmpc.controller import Controller

    cfg.control.max_change_per_cycle = 99.0
    controller = Controller(cfg, None, fake_ha)
    controller.excite_step(now=fake_ha.now, apply=True)
    assert len(load_runs(cfg)) == 1

    fake_ha.drop("sensor.indoor")
    controller.excite_step(now=fake_ha.now + timedelta(days=2), apply=True)
    assert len(load_runs(cfg)) == 1
