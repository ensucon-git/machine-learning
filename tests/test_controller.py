from __future__ import annotations

from datetime import timedelta

import pytest

from hpmpc.controller import Controller, ControllerState
from hpmpc.model.thermal import ThermalParams
from hpmpc.ntc import resistance_to_temperature


@pytest.fixture
def controller(cfg, fake_ha) -> Controller:
    return Controller(cfg, ThermalParams(), fake_ha)


def test_normal_cycle_writes_a_bounded_offset(controller, cfg, fake_ha):
    report = controller.step(now=fake_ha.now)
    assert report["mode"] == "mpc"
    assert report["applied"] is True
    assert fake_ha.written and fake_ha.written[-1][0] == "number.offset"
    assert cfg.control.offset_min <= report["offset"] <= cfg.control.offset_max
    assert report["mpc"]["predicted_indoor_min"] > 0


def test_dry_run_computes_everything_but_writes_nothing(controller, fake_ha):
    report = controller.step(now=fake_ha.now, apply=False)
    assert report["applied"] is False
    assert fake_ha.written == []
    assert "mpc" in report


def test_rate_limit_caps_the_change_per_cycle(controller, cfg, fake_ha):
    controller.state.last_offset = 0.0
    cfg.control.max_change_per_cycle = 0.25
    report = controller.step(now=fake_ha.now)
    assert abs(report["offset"]) <= 0.25 + 1e-9
    assert any("rate limited" in note for note in report["notes"])


def test_cold_house_triggers_maximum_heat(controller, cfg, fake_ha):
    fake_ha.set("sensor.indoor", cfg.control.hard_min - 1.0)
    cfg.control.max_change_per_cycle = 99.0
    report = controller.step(now=fake_ha.now)
    assert report["mode"] == "safety_override"
    assert report["offset"] == pytest.approx(cfg.control.offset_min)


def test_hot_house_triggers_minimum_heat(controller, cfg, fake_ha):
    fake_ha.set("sensor.indoor", cfg.control.hard_max + 1.0)
    cfg.control.max_change_per_cycle = 99.0
    report = controller.step(now=fake_ha.now)
    assert report["mode"] == "safety_override"
    assert report["offset"] == pytest.approx(cfg.control.offset_max)


def test_missing_indoor_sensor_falls_back(controller, cfg, fake_ha):
    fake_ha.drop("sensor.indoor")
    report = controller.step(now=fake_ha.now)
    assert report["mode"] == "fallback"
    assert report["offset"] == pytest.approx(cfg.control.fallback_offset)
    assert any("unavailable" in p for p in report["problems"])


def test_stale_data_falls_back(controller, cfg, fake_ha):
    fake_ha.set("sensor.indoor", 21.0, age_minutes=cfg.control.max_data_age_minutes + 10)
    report = controller.step(now=fake_ha.now)
    assert report["mode"] == "fallback"
    assert any("stale" in p for p in report["problems"])


def test_implausible_reading_falls_back(controller, fake_ha):
    fake_ha.set("sensor.indoor", 250.0)
    report = controller.step(now=fake_ha.now)
    assert report["mode"] == "fallback"
    assert any("plausible range" in p for p in report["problems"])


def test_fallback_is_reached_gradually_not_in_one_jump(controller, cfg, fake_ha):
    controller.state.last_offset = -6.0
    cfg.control.max_change_per_cycle = 1.0
    fake_ha.drop("sensor.indoor")
    report = controller.step(now=fake_ha.now)
    assert report["offset"] == pytest.approx(-5.0)


@pytest.mark.parametrize("mode", ["offset", "fake_temperature", "resistance"])
def test_output_modes_translate_the_offset_correctly(controller, cfg, mode):
    cfg.control.output_mode = mode
    value, unit = controller.output_value(-3.0, -5.0)
    if mode == "offset":
        assert (value, unit) == (-3.0, "K")
    elif mode == "fake_temperature":
        assert (value, unit) == (-8.0, "degC")
    else:
        assert unit == "ohm"
        assert float(resistance_to_temperature(value, cfg.ntc)) == pytest.approx(-8.0, abs=0.05)


def test_write_failure_is_reported_not_raised(controller, fake_ha):
    fake_ha.fail_write = True
    report = controller.step(now=fake_ha.now)
    assert report["applied"] is False
    assert any("failed to write" in note for note in report["notes"])


def test_status_entity_is_published(controller, cfg, fake_ha):
    cfg.entities.status_entity = "sensor.hpmpc_status"
    controller.step(now=fake_ha.now)
    urls = [url for url, _ in fake_ha.posted]
    assert "/api/states/sensor.hpmpc_status" in urls


def test_state_survives_a_restart(controller, cfg, fake_ha):
    controller.step(now=fake_ha.now)
    reloaded = ControllerState.load(cfg.paths.state_file)
    assert reloaded is not None
    assert reloaded.last_offset == pytest.approx(controller.state.last_offset)
    assert reloaded.t_mass == pytest.approx(controller.state.t_mass)


def test_observer_moves_the_slab_estimate_toward_reality(controller, fake_ha):
    controller.state.t_mass = 24.0
    controller.state.t_indoor = 21.0
    before = controller.state.t_mass
    # The house is measurably warmer than the model expects.
    controller.update_estimate({"t_indoor": 22.5, "t_outdoor": -5.0, "wind": 2.0}, elapsed_hours=0.25)
    assert controller.state.t_mass > before
    assert controller.state.t_indoor == pytest.approx(22.5)


def test_observer_keeps_the_slab_estimate_physical(controller):
    controller.state.t_mass = 24.0
    controller.update_estimate({"t_indoor": 21.0, "t_outdoor": -5.0}, elapsed_hours=6.0)
    assert 16.0 < controller.state.t_mass < 46.0


def test_excitation_holds_a_value_within_a_block(controller, cfg, fake_ha):
    cfg.control.max_change_per_cycle = 99.0
    first = controller.excite_step(now=fake_ha.now, hold_hours=6.0)
    second = controller.excite_step(now=fake_ha.now + timedelta(minutes=15), hold_hours=6.0)
    assert first["mode"] == "excitation"
    assert first["offset"] == pytest.approx(second["offset"])


def test_excitation_changes_between_blocks(controller, cfg, fake_ha):
    cfg.control.max_change_per_cycle = 99.0
    values = {
        round(controller.excite_step(now=fake_ha.now + timedelta(hours=6 * i), hold_hours=6.0)["offset"], 3)
        for i in range(6)
    }
    assert len(values) > 1


def test_excitation_still_respects_the_comfort_guard(controller, cfg, fake_ha):
    cfg.control.max_change_per_cycle = 99.0
    fake_ha.set("sensor.indoor", cfg.control.hard_min - 1.0)
    report = controller.excite_step(now=fake_ha.now)
    assert report["mode"] == "excitation_safety_override"
    assert report["offset"] == pytest.approx(cfg.control.offset_min)


def test_plan_table_is_hourly_and_covers_the_horizon(controller, cfg, fake_ha):
    report = controller.step(now=fake_ha.now, apply=False)
    plan = report["plan"]
    assert len(plan) == int(cfg.control.horizon_hours)
    assert {"time", "price", "offset", "t_supply", "t_indoor", "kw"} <= set(plan[0])


def test_residual_failure_does_not_break_control(controller, fake_ha):
    class Broken:
        def predict(self, index, exog):
            raise RuntimeError("model corrupted")

    controller.residual = Broken()
    report = controller.step(now=fake_ha.now, apply=False)
    assert report["mode"] == "mpc"
