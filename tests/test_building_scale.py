"""The meter, not the prior, sets how many watts the house loses."""

from __future__ import annotations

import dataclasses
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from hpmpc.identify import calibrate_building_scale, make_windows, predict, rescale_thermal
from hpmpc.model import heatpump as hp
from hpmpc.model.thermal import ThermalParams, load_params, save_params
from hpmpc.simulator import make_demo_dataset
from hpmpc.train import train

from test_disaggregate import house_config


@pytest.fixture(scope="module")
def house():
    cfg = house_config()
    cfg.training.restarts = 1
    frame, truth = make_demo_dataset(cfg, days=30, seed=5, whole_house_power=True)
    return cfg, frame, truth


def test_indoor_temperatures_cannot_tell_a_house_from_one_three_times_its_size(house, monkeypatch):
    """The reason the scale has to come from somewhere else. Below capacity,
    scaling every heat flow and store together changes no trajectory at all."""
    cfg, frame, truth = house
    original = hp.PumpModel.operating_point

    def unlimited(self, *args, **kwargs):
        point = original(self, *args, **kwargs)
        return dataclasses.replace(point, capacity=np.full_like(point.capacity, 1e9))

    monkeypatch.setattr(hp.PumpModel, "operating_point", unlimited)
    ws = make_windows(frame.iloc[: 6 * 96], cfg)
    base = predict(truth, cfg, ws)
    for factor in (1 / 3, 3.0):
        assert np.abs(predict(truth.scaled(factor), cfg, ws) - base).max() < 1e-9
    assert truth.scaled(3.0).heat_loss_w_per_k() == pytest.approx(3 * truth.heat_loss_w_per_k())
    assert truth.scaled(3.0).k_wind == truth.k_wind      # ratios are not scaled


def test_the_meter_measures_how_far_off_the_scale_is(house):
    cfg, frame, truth = house
    found = calibrate_building_scale(frame, cfg, truth.scaled(0.4))
    assert found["accepted"], found["reason"]
    assert found["factor"] == pytest.approx(2.5, rel=0.06)
    assert found["heat_loss_after_w_per_k"] == pytest.approx(truth.heat_loss_w_per_k(), rel=0.06)
    assert found["car_charger_kw"] == pytest.approx(11.0, rel=0.1)


def test_training_from_a_prior_a_third_too_small_lands_on_the_true_heat_loss(house, tmp_path):
    """What happened on the real house: every fit stayed near its starting
    point's scale. With the meter setting it, the starting point stops mattering."""
    cfg, frame, truth = house
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, model_dir=str(tmp_path)))
    wrong = truth.scaled(0.35)
    save_params(cfg.model_path, wrong)

    report = train(cfg, frame, fit_curve=False)

    params, _ = load_params(cfg.model_path)
    assert report["building_scale"]["accepted"]
    assert params.heat_loss_w_per_k() == pytest.approx(truth.heat_loss_w_per_k(), rel=0.08)
    # The COP map was trusted, not bent to absorb the error.
    assert "efficiency_scale" not in report["pump_overrides"]
    assert report["thermal"]["validation"]["rmse_c"] < 0.15


def test_calibrating_the_efficiency_instead_bends_the_cop_to_fit_the_wrong_house(house):
    """The confound, shown from the other side: the old behaviour finds an
    efficiency as far off as the building is."""
    cfg, frame, truth = house
    from hpmpc.identify import fit_cop

    cop = fit_cop(frame, cfg, truth.scaled(0.5))
    assert cop["efficiency_scale"] == pytest.approx(0.5, rel=0.08)


def test_an_undetermined_meter_does_not_move_the_building(house):
    cfg, frame, truth = house
    noisy = frame.copy()
    rng = np.random.default_rng(1)
    for phase in ("house_l1", "house_l2", "house_l3"):
        noisy[phase] = rng.uniform(200.0, 6000.0, len(noisy))
    found = calibrate_building_scale(noisy, cfg, truth)
    assert found is not None and not found["accepted"]
    assert found["reason"]


def test_without_power_data_there_is_nothing_to_calibrate(house):
    cfg, frame, truth = house
    cfg = dataclasses.replace(cfg, power=dataclasses.replace(cfg.power, source="none"))
    assert calibrate_building_scale(frame, cfg, truth) is None


def test_a_scale_outside_the_physical_bounds_is_refused(house):
    cfg, frame, truth = house
    outcome = rescale_thermal(truth, {"validation": {"rmse_c": 0.1}}, frame, cfg, 50.0)
    assert isinstance(outcome, str) and "bounds" in outcome


def test_the_service_retrains_from_the_file_not_from_the_last_models_pump_values(tmp_path, monkeypatch, cfg, fake_ha):
    """The running config carries the last model's learned efficiency. Had the
    retrain used it, an efficiency of 0.3 from an old calibration would have
    set the very scale the meter is now supposed to measure."""
    import hpmpc.api as api_module

    save_params(cfg.model_path, ThermalParams(), {"pump_overrides": {"efficiency_scale": 0.3}})
    old = time.time() - 40 * 86400
    os.utime(cfg.model_path, (old, old))
    hour = datetime.now(ZoneInfo(cfg.site.timezone)).hour
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
entities:
  indoor_temp: sensor.indoor
  outdoor_temp: sensor.outdoor
  price: sensor.price
training:
  retrain_days: 30
  retrain_hour: {hour}
paths:
  data_dir: {cfg.paths.data_dir}
  model_dir: {cfg.paths.model_dir}
  state_file: {cfg.paths.state_file}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "HomeAssistant", lambda *a, **k: fake_ha)
    seen = {}
    monkeypatch.setattr(api_module, "build_training_frame", lambda c, ha: (seen.setdefault("frame_cfg", c), None))
    monkeypatch.setattr(api_module, "save_dataset", lambda frame, path: None)
    monkeypatch.setattr(api_module, "train", lambda c, frame: seen.setdefault("train_cfg", c) and {})

    service = api_module.ControllerService(config_path)
    assert service.cfg.heat_pump.efficiency_scale == 0.3
    service.maybe_retrain()
    assert seen["train_cfg"].heat_pump.efficiency_scale == 1.0
