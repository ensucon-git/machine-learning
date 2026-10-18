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


def _cycling_with_hot_water(frame, cfg, rng):
    """What a real pump looks like on a meter: on/off at full power rather than
    a smooth modulation, plus hot-water runs the building model knows nothing about."""
    import pandas as pd

    pump = frame["power"].to_numpy(dtype=float)
    cycling = np.zeros_like(pump)
    for start in range(0, len(pump), 4):
        energy, slot = pump[start:start + 4].sum(), 0
        while energy > 0 and slot < 4:
            cycling[start + slot] = min(3500.0, energy)
            energy -= cycling[start + slot]
            slot += 1
        rng.shuffle(cycling[start:start + 4])
    local = frame.index.tz_convert(cfg.site.timezone)
    hour = local.hour + local.minute / 60.0
    day = (local.normalize() - local.normalize()[0]).days
    for d in range(int(day.max()) + 1):
        for centre in (6.5, 21.0):
            if rng.random() < 0.8:
                begin = centre + rng.normal(0.0, 1.5)
                cycling[(day == d) & (hour >= begin) & (hour < begin + rng.uniform(0.75, 1.5))] += 3000.0
    out = frame.copy()
    out["power"] = cycling
    return pd.DataFrame(out)


def test_the_scale_survives_a_cycling_pump_and_hot_water():
    """At quarter-hour resolution a cycling pump and hot water make the split's
    R2 collapse - as on the real house - while the energy per day still says
    how far off the building is."""
    from hpmpc.simulator import add_house_electricity, excitation_offsets, simulate_house, synthetic_prices, synthetic_weather
    import pandas as pd

    cfg = house_config()
    rng = np.random.default_rng(3)
    index = pd.date_range("2026-09-05", periods=40 * 96, freq="15min", tz="UTC")
    truth = ThermalParams(Ci=2100.0, Cm=26000.0, Him=980.0, Hie=165.0, Hme=30.0, k_wind=0.045,
                          A_sol=5.5, f_sol_i=0.45, Q_int=420.0, Hfloor=1350.0)
    frame = simulate_house(cfg, truth, synthetic_weather(index, rng, mean_outdoor=9.0),
                           synthetic_prices(index, rng, cfg.site.timezone), excitation_offsets(index, rng), seed=3)
    frame = _cycling_with_hot_water(frame, cfg, rng)
    frame = add_house_electricity(frame, cfg, rng).drop(columns=["power"])

    found = calibrate_building_scale(frame, cfg, truth.scaled(0.35))
    assert found["accepted"], found["reason"]
    assert found["daily_r2"] > 0.9
    assert found["heat_loss_after_w_per_k"] == pytest.approx(truth.heat_loss_w_per_k(), rel=0.12)


def test_the_bounds_grow_with_the_house():
    """A 350 m2 house at UA 380 runs its slab coupling past bounds drawn for
    140 m2. On the real house that refused the meter's x4.4 as 'outside the
    physical bounds' although the result, UA 383, was exactly what the pump's
    sizing implies."""
    lo1, hi1 = ThermalParams.bounds()
    lo2, hi2 = ThermalParams.bounds(2.0)
    names = ThermalParams.names()
    assert hi2[names.index("Hfloor")] == 2 * hi1[names.index("Hfloor")]
    assert hi2[names.index("k_wind")] == hi1[names.index("k_wind")]      # a ratio, not a size
    assert ThermalParams().size() == pytest.approx(1.0)


def test_a_large_house_can_be_scaled_to_its_true_size(house):
    cfg, frame, truth = house
    small = truth.scaled(0.4)
    validation = rescale_thermal(small, {"validation": {"rmse_c": 1.0}}, frame, cfg, 1.0)[1]["validation"]
    outcome = rescale_thermal(small, {"validation": validation}, frame, cfg, 12.5)
    assert not isinstance(outcome, str), outcome
    scaled, _ = outcome
    assert scaled.Hfloor > 6000.0                  # past the old fixed bound
    assert scaled.heat_loss_w_per_k() == pytest.approx(5 * truth.heat_loss_w_per_k())


def test_training_can_ignore_the_weeks_before_the_heating_was_switched_on(house):
    """On the real house the metered energy stepped up ~12 kWh/day around the
    day space heating was switched on. Before it the model assumes heat the
    pump never made, which steepens the meter's slope (x4.4 against x3.0)."""
    import pandas as pd

    from hpmpc.config import Config
    from hpmpc.train import training_window

    cfg, frame, _ = house
    cut = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, ignore_before="2026-01-20"))
    kept = training_window(frame, cut)
    assert kept.index[0] == pd.Timestamp("2026-01-20", tz=cfg.site.timezone)
    assert len(kept) < len(frame)
    assert training_window(frame, cfg) is frame

    nothing_left = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, ignore_before="2030-01-01"))
    with pytest.raises(ValueError, match="ignore_before"):
        train(nothing_left, frame)

    bad = Config()
    bad.entities.indoor_temp = "sensor.indoor"
    bad.training.ignore_before = "19 september"
    with pytest.raises(ValueError, match="not a date"):
        bad.validate()
