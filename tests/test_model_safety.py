"""Retraining must not be able to make things worse than they were.

Found on the real house: the chronological split held the whole excitation
week out as validation, and a retrain that predicted worse than 'no change'
overwrote the only copy of the model that was steering."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hpmpc.identify import _split, make_windows
from hpmpc.model.thermal import ThermalParams, load_params, save_params
from hpmpc.simulator import make_demo_dataset


@pytest.fixture(scope="module")
def house():
    from hpmpc.config import Config

    cfg = Config()
    cfg.entities.indoor_temp = "sensor.indoor"
    cfg.entities.outdoor_temp = "sensor.outdoor"
    cfg.entities.price = "sensor.price"
    cfg.training.max_windows = 400
    cfg.validate()
    frame, _ = make_demo_dataset(cfg, days=12, seed=3)
    return cfg, frame


def test_an_excitation_at_the_end_of_the_data_mostly_trains(house):
    """As on a fresh install: weeks of offset 0, then the experiment, newest."""
    cfg, frame = house
    quiet = frame.copy()
    cut = int(len(quiet) * 0.7)
    quiet.iloc[:cut, quiet.columns.get_loc("offset")] = 0.0
    ws = make_windows(cfg=cfg, frame=quiet)
    train, val = _split(ws, 0.25)

    moving = lambda w: int((w.offset.std(axis=1) > 0.25).sum())     # noqa: E731
    assert moving(train) > 2 * moving(val) > 0
    # A plain chronological cut would have sent nearly all of it to validation.
    n_val = int(round(len(ws) * 0.25))
    plain_train = ws.subset(np.arange(len(ws) - n_val))
    assert moving(train) > moving(plain_train)


def test_without_movement_the_split_is_the_old_chronological_one(house):
    cfg, frame = house
    still = frame.copy()
    still["offset"] = 0.0
    ws = make_windows(cfg=cfg, frame=still)
    train, val = _split(ws, 0.25)
    assert len(val) == int(round(len(ws) * 0.25))
    assert train.t_indoor0[-1] == ws.t_indoor0[len(train) - 1]


def _rigged(monkeypatch, rmse, baseline):
    import hpmpc.train as train_module

    metrics = {"validation": {"rmse_c": rmse, "baseline_rmse_c": baseline}}
    monkeypatch.setattr(train_module, "fit_thermal",
                        lambda frame, cfg, initial=None: (ThermalParams(Hie=77.0), metrics))


def test_a_model_worse_than_no_change_does_not_replace_the_running_one(house, cfg, monkeypatch):
    from hpmpc.train import rejected_path, train

    _, frame = house
    cfg.training.use_residual_model = False
    save_params(cfg.model_path, ThermalParams(Hie=123.0), {"trained_at": "before"})
    _rigged(monkeypatch, rmse=0.60, baseline=0.46)

    report = train(cfg, frame)
    assert "rejected" in report
    assert load_params(cfg.model_path)[0].Hie == pytest.approx(123.0)
    assert load_params(rejected_path(cfg))[0].Hie == pytest.approx(77.0)


def test_force_installs_it_anyway(house, cfg, monkeypatch):
    from hpmpc.train import train

    _, frame = house
    cfg.training.use_residual_model = False
    save_params(cfg.model_path, ThermalParams(Hie=123.0), {})
    _rigged(monkeypatch, rmse=0.60, baseline=0.46)
    assert "rejected" not in train(cfg, frame, force=True)
    assert load_params(cfg.model_path)[0].Hie == pytest.approx(77.0)


def test_the_very_first_model_is_installed_even_if_weak(house, cfg, monkeypatch):
    """Collecting mode is not better than a weak model held to small limits,
    and there is nothing to keep instead."""
    from hpmpc.train import train

    _, frame = house
    cfg.training.use_residual_model = False
    _rigged(monkeypatch, rmse=0.60, baseline=0.46)
    assert "rejected" not in train(cfg, frame)


def test_an_accepted_model_keeps_the_one_it_replaced_and_revert_swaps_them(house, cfg, monkeypatch):
    from hpmpc.train import previous_path, revert_model, train

    _, frame = house
    cfg.training.use_residual_model = False
    save_params(cfg.model_path, ThermalParams(Hie=123.0), {"trained_at": "old"})
    _rigged(monkeypatch, rmse=0.30, baseline=0.46)
    train(cfg, frame)
    assert load_params(cfg.model_path)[0].Hie == pytest.approx(77.0)
    assert load_params(previous_path(cfg))[0].Hie == pytest.approx(123.0)

    revert_model(cfg)
    assert load_params(cfg.model_path)[0].Hie == pytest.approx(123.0)
    assert load_params(previous_path(cfg))[0].Hie == pytest.approx(77.0)
    revert_model(cfg)                                     # and back again
    assert load_params(cfg.model_path)[0].Hie == pytest.approx(77.0)
    json.loads(previous_path(cfg).read_text())            # still a readable model
