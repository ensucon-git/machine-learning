"""Separating the heat pump out of a whole-house electricity measurement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpmpc.config import Config
from hpmpc.disaggregate import (
    base_load_design,
    disaggregate,
    ev_mask,
    house_power,
    quality_warnings,
)
from hpmpc.identify import _simulate_period, fit_cop
from hpmpc.simulator import make_demo_dataset


def house_config(**overrides) -> Config:
    cfg = Config()
    cfg.entities.indoor_temp = "sensor.indoor"
    cfg.entities.outdoor_temp = "sensor.outdoor"
    cfg.entities.house_power_l1 = "sensor.l1"
    cfg.entities.house_power_l2 = "sensor.l2"
    cfg.entities.house_power_l3 = "sensor.l3"
    cfg.entities.ev_charging = "binary_sensor.eh6nh5cd_charging"
    cfg.heat_pump.model = "daikin_erlq016caw1"
    for key, value in overrides.items():
        section, field = key.split(".")
        setattr(getattr(cfg, section), field, value)
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def house():
    cfg = house_config()
    frame, truth = make_demo_dataset(cfg, days=30, seed=5, whole_house_power=True)
    return cfg, frame, truth


def run(cfg, frame, params):
    context = _simulate_period(frame, cfg, params)
    return disaggregate(context["data"], cfg, context["q_compressor"], context["q_backup"], context["cop_base"])


# ------------------------------------------------------------- ingredients


def test_balanced_power_is_three_times_the_smallest_phase():
    frame = pd.DataFrame(
        {"house_l1": [1000.0, 500.0], "house_l2": [1200.0, 500.0], "house_l3": [900.0, 500.0]},
        index=pd.date_range("2026-01-15", periods=2, freq="15min", tz="UTC"),
    )
    total, balanced = house_power(frame, house_config())
    assert total.tolist() == [3100.0, 1500.0]
    assert balanced.tolist() == [2700.0, 1500.0]


def test_a_single_total_entity_still_works():
    frame = pd.DataFrame(
        {"house_power": [2000.0, 2500.0]},
        index=pd.date_range("2026-01-15", periods=2, freq="15min", tz="UTC"),
    )
    total, balanced = house_power(frame, house_config())
    assert total.tolist() == [2000.0, 2500.0]
    assert balanced is None


def test_no_power_data_at_all_returns_nothing():
    frame = pd.DataFrame({"t_indoor": [21.0]}, index=pd.date_range("2026-01-15", periods=1, freq="15min", tz="UTC"))
    assert house_power(frame, house_config()) == (None, None)


def test_charging_mask_is_widened_by_the_guard_band():
    index = pd.date_range("2026-01-15", periods=12, freq="15min", tz="UTC")
    charging = np.zeros(12)
    charging[6] = 1.0
    frame = pd.DataFrame({"ev_charging": charging}, index=index)
    cfg = house_config(**{"power.ev_guard_minutes": 30.0})
    mask = ev_mask(frame, cfg)
    assert mask[6]
    assert mask[4] and mask[8]        # two steps either side
    assert not mask[3] and not mask[9]


def test_base_load_design_uses_the_clock_and_nothing_else():
    index = pd.date_range("2026-01-15", periods=96, freq="15min", tz="UTC")
    design, names = base_load_design(index, "Europe/Stockholm", harmonics=4)
    assert design.shape == (96, 10)
    assert names[0] == "base_constant" and names[-1] == "base_weekend"
    assert np.allclose(design[:, 0], 1.0)


# ---------------------------------------------------------------- the fit


def test_recovers_the_efficiency_scale_from_whole_house_power(house):
    cfg, frame, truth = house
    result = run(cfg, frame, truth)
    assert result is not None
    # The synthetic house was generated at scale 1.0.
    assert result.efficiency_scale == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("true_scale", [0.85, 1.25])
def test_tracks_a_pump_that_is_worse_or_better_than_the_map(true_scale):
    cfg = house_config(**{"heat_pump.efficiency_scale": true_scale})
    frame, truth = make_demo_dataset(cfg, days=30, seed=5, whole_house_power=True)
    cfg.heat_pump.efficiency_scale = 1.0      # the fit has to find it again
    result = run(cfg, frame, truth)
    assert result.efficiency_scale == pytest.approx(true_scale, rel=0.1)


def test_the_car_charger_is_recovered_as_a_sanity_check(house):
    cfg, frame, truth = house
    result = run(cfg, frame, truth)
    assert result.ev_power_w == pytest.approx(cfg.power.ev_nominal_kw * 1000, rel=0.15)
    assert result.metrics["ev_power_ratio"] == pytest.approx(1.0, abs=0.15)


def test_charging_samples_are_excluded_from_the_fit(house):
    cfg, frame, truth = house
    result = run(cfg, frame, truth)
    charging = float(frame["ev_charging"].mean())
    assert charging > 0.02
    assert result.metrics["charging_fraction"] == pytest.approx(charging, abs=0.05)
    assert result.used_samples < len(frame) * (1 - charging)


def test_survives_a_house_full_of_large_appliances(house):
    cfg, frame, truth = house
    noisy = frame.copy()
    rng = np.random.default_rng(3)
    extra = np.zeros(len(noisy))
    for _ in range(60):
        start = int(rng.integers(0, len(noisy)))
        extra[start : start + int(rng.integers(2, 10))] += float(rng.uniform(3000, 6500))
    noisy["house_l1"] = noisy["house_l1"] + extra
    result = run(cfg, noisy, truth)
    assert result.efficiency_scale == pytest.approx(1.0, abs=0.1)


def test_total_target_also_works_just_less_precisely(house):
    cfg, frame, truth = house
    cfg = house_config(**{"power.target": "total"})
    result = run(cfg, frame, truth)
    assert result.target.startswith("total")
    assert result.efficiency_scale == pytest.approx(1.0, abs=0.12)


def test_too_little_data_is_refused(house):
    cfg, frame, truth = house
    assert run(cfg, frame.iloc[:300], truth) is None


# ------------------------------------------------------------ diagnostics


def test_short_history_is_flagged_even_when_the_fit_looks_precise():
    cfg = house_config()
    frame, truth = make_demo_dataset(cfg, days=8, seed=5, whole_house_power=True)
    result = run(cfg, frame, truth)
    warnings = quality_warnings(result)
    assert any("days of usable data" in w for w in warnings)


def test_a_misconfigured_charger_power_is_called_out(house):
    cfg, frame, truth = house
    cfg = house_config(**{"power.ev_nominal_kw": 3.0})
    result = run(cfg, frame, truth)
    assert any("car charger looks like" in w for w in quality_warnings(result))


def test_clean_data_produces_no_warnings(house):
    cfg, frame, truth = house
    assert quality_warnings(run(cfg, frame, truth)) == []


# ------------------------------------------------------- through fit_cop


def test_fit_cop_uses_the_house_meter_when_there_is_no_dedicated_one(house):
    cfg, frame, truth = house
    result = fit_cop(frame, cfg, truth)
    assert result is not None
    assert "whole-house" in result["method"]
    assert result["efficiency_scale"] == pytest.approx(1.0, abs=0.08)
    assert "disaggregation" in result


def test_a_dedicated_meter_wins_when_both_exist(house):
    cfg, frame, truth = house
    with_meter = frame.copy()
    with_meter["power"] = frame["heatpump_power_true"]
    result = fit_cop(with_meter, cfg, truth)
    assert result["method"] == "dedicated heat pump meter"
    assert result["efficiency_scale"] == pytest.approx(1.0, abs=0.05)


def test_power_source_none_disables_the_whole_thing(house):
    cfg, frame, truth = house
    cfg = house_config(**{"power.source": "none"})
    assert fit_cop(frame, cfg, truth) is None


def test_requesting_a_meter_that_does_not_exist_fails_loudly(house, caplog):
    cfg, frame, truth = house
    cfg = house_config(**{"power.source": "heatpump_meter"})
    assert fit_cop(frame, cfg, truth) is None
    assert "no heat pump power sensor" in caplog.text
