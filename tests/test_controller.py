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
    assert "number.offset" in dict(fake_ha.written)
    assert cfg.control.offset_min <= report["offset"] <= cfg.control.offset_max
    assert report["mpc"]["predicted_indoor_min"] > 0


def test_dry_run_computes_everything_but_writes_nothing(controller, fake_ha):
    report = controller.step(now=fake_ha.now, apply=False)
    assert report["applied"] is False
    assert fake_ha.written == []
    assert "mpc" in report


def test_rate_limit_caps_the_change_per_cycle(controller, cfg, fake_ha):
    # Start far from anything the optimiser could want, so the limiter has to
    # engage whatever it decides - the decision itself depends on the price
    # profile, which depends on the time of day the tests happen to run.
    controller.state.last_offset = 5.0
    cfg.control.max_change_per_cycle = 0.25
    report = controller.step(now=fake_ha.now)
    assert report["offset"] >= 5.0 - 0.25 - 1e-9
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


def test_every_configured_output_gets_the_same_decision(controller, cfg):
    """Kelvin, degrees and ohm are one number in three units. Writing them all
    means the one you act on and the one you look at cannot disagree."""
    cfg.entities.resistance_output = "number.resistans"
    outputs = {o["kind"]: o for o in controller.outputs(-3.0, -5.0)}
    assert outputs["offset"]["value"] == pytest.approx(-3.0)
    assert outputs["offset"]["unit"] == "K"
    assert outputs["fake_temperature"]["value"] == pytest.approx(-8.0)
    assert outputs["resistance"]["unit"] == "ohm"
    assert float(resistance_to_temperature(outputs["resistance"]["value"], cfg.ntc)) == pytest.approx(
        -8.0, abs=0.05
    )


def test_only_configured_outputs_are_produced(controller, cfg):
    cfg.entities.fake_temperature_output = ""
    cfg.entities.resistance_output = ""
    assert [o["kind"] for o in controller.outputs(-3.0, -5.0)] == ["offset"]


def test_all_outputs_are_written_each_cycle(controller, cfg, fake_ha):
    controller.step(now=fake_ha.now)
    written = dict(fake_ha.written)
    assert set(written) == {"number.offset", "number.fake_temp"}
    # The pair is self-consistent: fake temperature is outdoor plus offset.
    assert written["number.fake_temp"] == pytest.approx(
        written["number.offset"] + float(fake_ha.get_state("sensor.outdoor").numeric), abs=0.01
    )


def test_one_failing_output_does_not_stop_the_others(controller, cfg, fake_ha):
    original = fake_ha.set_number

    def flaky(entity_id, value):
        if entity_id == "number.offset":
            from hpmpc.ha import HomeAssistantError

            raise HomeAssistantError("nope")
        original(entity_id, value)

    fake_ha.set_number = flaky
    report = controller.step(now=fake_ha.now)
    assert report["applied"] is True                       # the other one landed
    assert any("failed to write number.offset" in n for n in report["notes"])


def test_write_failure_is_reported_not_raised(controller, fake_ha):
    fake_ha.fail_write = True
    report = controller.step(now=fake_ha.now)
    assert report["applied"] is False
    assert any("failed to write" in note for note in report["notes"])


def test_the_outdoor_temperature_can_come_from_the_forecast(cfg, fake_ha):
    """No outdoor sensor means Home Assistant is not in the loop for it at all."""
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg.entities.outdoor_temp = ""
    fake_ha.drop("sensor.outdoor")
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(now=fake_ha.now, apply=False)
    assert report["mode"] == "mpc"
    assert "forecast" in report["readings"]["t_outdoor_source"]
    assert report["readings"]["t_outdoor"] is not None


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
    hold_hours = 6.0
    block = fake_ha.now.timestamp() // (hold_hours * 3600)
    # Step to another moment in the same block. Stepping forward would
    # occasionally cross a boundary depending on when the test happens to run,
    # so fall back to stepping backwards - the sensors stay fresh either way.
    other = fake_ha.now + timedelta(minutes=15)
    if other.timestamp() // (hold_hours * 3600) != block:
        other = fake_ha.now - timedelta(minutes=15)

    first = controller.excite_step(now=fake_ha.now, hold_hours=hold_hours)
    second = controller.excite_step(now=other, hold_hours=hold_hours)
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


def test_long_outage_forces_a_state_rebuild(controller, fake_ha):
    from datetime import timedelta

    controller.state.warm_started = True
    controller.state.updated_at = (fake_ha.now - timedelta(hours=30)).isoformat()
    report = controller.step(now=fake_ha.now, apply=False)
    assert any("re-warming" in note for note in report["notes"])


def test_a_corrupt_timestamp_does_not_crash_the_cycle(controller, fake_ha):
    controller.state.updated_at = "not-a-timestamp"
    assert controller.step(now=fake_ha.now, apply=False)["mode"] == "mpc"


def test_perceived_temperature_limit_caps_the_offset(controller, cfg, fake_ha):
    cfg.control.max_change_per_cycle = 99.0
    cfg.heat_pump.perceived_min_c = -8.0
    fake_ha.set("sensor.outdoor", -5.0)
    # The house is freezing, so the safety override asks for maximum heat ...
    fake_ha.set("sensor.indoor", cfg.control.hard_min - 1.0)
    report = controller.step(now=fake_ha.now)
    # ... but the pump may only ever be shown -8 degC, i.e. a -3 K offset.
    assert report["offset"] == pytest.approx(-3.0)
    assert any("degC" in note for note in report["notes"])


def test_backup_heater_energy_reaches_the_status_entity(controller, cfg, fake_ha):
    cfg.entities.status_entity = "sensor.hpmpc_status"
    cfg.heat_pump.model = "daikin_erlq016caw1"
    from hpmpc.model import build_pump
    from hpmpc.mpc import MpcSolver

    controller.pump = build_pump(cfg)
    controller.solver = MpcSolver(cfg, controller.params)
    controller.step(now=fake_ha.now)
    posted = dict(fake_ha.posted)["/api/states/sensor.hpmpc_status"]
    assert "backup_heater_kwh_horizon" in posted["attributes"]


# ------------------------------------------- closed-loop actuator check


def test_actuator_check_is_silent_without_the_entity(controller, fake_ha):
    report = controller.step(now=fake_ha.now, apply=False)
    assert "actuator" not in report


def test_actuator_check_reports_the_discrepancy(controller, cfg, fake_ha):
    cfg.entities.pump_outdoor_temp = "sensor.daikin_utetemp"
    controller.state.last_offset = -2.0
    fake_ha.set("sensor.outdoor", -5.0)
    fake_ha.set("sensor.daikin_utetemp", -7.0)          # exactly what -5 + -2 should give
    report = controller.step(now=fake_ha.now, apply=False)
    actuator = report["actuator"]
    assert actuator["commanded_c"] == pytest.approx(-7.0)
    assert actuator["error_now_c"] == pytest.approx(0.0)
    assert "warning" not in actuator


def test_a_persistent_actuator_error_eventually_warns(controller, cfg, fake_ha):
    cfg.entities.pump_outdoor_temp = "sensor.daikin_utetemp"
    cfg.control.actuator_error_smoothing = 0.5           # settle quickly for the test
    controller.state.last_offset = 0.0
    fake_ha.set("sensor.outdoor", -5.0)
    fake_ha.set("sensor.daikin_utetemp", -1.5)          # the pump is 3.5 K off

    warning = None
    for _ in range(10):
        actuator = controller.check_actuator(controller.read_sensors())
        warning = actuator.get("warning")
    assert controller.state.actuator_error_c == pytest.approx(3.5, abs=0.1)
    assert warning is not None and "does not match the sensor" in warning


def test_a_brief_excursion_does_not_warn(controller, cfg, fake_ha):
    """The pump filters its outdoor reading, so single cycles are lag, not bias.

    Run at the real smoothing constant: one wild sample must not be enough to
    accuse the calibration of being wrong.
    """
    cfg.entities.pump_outdoor_temp = "sensor.daikin_utetemp"
    controller.state.last_offset = 0.0
    fake_ha.set("sensor.outdoor", -5.0)
    fake_ha.set("sensor.daikin_utetemp", -5.0)

    settled = None
    for _ in range(80):
        settled = controller.check_actuator(controller.read_sensors())
    assert settled["settled"] is True and "warning" not in settled

    fake_ha.set("sensor.daikin_utetemp", 1.0)           # one wild sample
    actuator = controller.check_actuator(controller.read_sensors())
    assert actuator["error_now_c"] == pytest.approx(6.0)
    assert abs(actuator["error_smoothed_c"]) < 0.5
    assert "warning" not in actuator


def test_the_check_respects_the_absolute_clamp(controller, cfg, fake_ha):
    """The commanded value is what the pump was actually shown, limits included."""
    cfg.entities.pump_outdoor_temp = "sensor.daikin_utetemp"
    cfg.heat_pump.perceived_max_c = 10.0
    controller.state.last_offset = 20.0                  # would be +15, but clamped to +10
    fake_ha.set("sensor.outdoor", -5.0)
    fake_ha.set("sensor.daikin_utetemp", 10.0)
    actuator = controller.check_actuator(controller.read_sensors())
    assert actuator["commanded_c"] == pytest.approx(10.0)
    assert actuator["error_now_c"] == pytest.approx(0.0)


# --------------------------------------------- outputs hpmpc creates itself


def test_a_sensor_output_is_published_rather_than_written(cfg, fake_ha):
    """No helper has to exist first: hpmpc puts the entity into Home Assistant."""
    cfg.entities.offset_output = ""
    cfg.entities.fake_temperature_output = "sensor.hpmpc_fake_outdoor"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    controller.step(apply=True)

    assert fake_ha.written == [], "a sensor is not a number service call"
    posted = [(url, body) for url, body in fake_ha.posted if "hpmpc_fake_outdoor" in url]
    assert posted, "the entity should have been published"
    assert posted[-1][1]["attributes"]["device_class"] == "temperature"
    assert posted[-1][1]["attributes"]["unit_of_measurement"] == "°C"
    assert fake_ha.get_state("sensor.hpmpc_fake_outdoor") is not None


def test_helpers_and_published_sensors_can_be_mixed(cfg, fake_ha):
    """Drive the actuator from the durable helper, watch the rest on a dashboard."""
    cfg.entities.offset_output = "input_number.offset"
    cfg.entities.fake_temperature_output = "sensor.hpmpc_fake_outdoor"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(apply=True)

    assert [e for e, _ in fake_ha.written] == ["input_number.offset"]
    assert any("hpmpc_fake_outdoor" in url for url, _ in fake_ha.posted)
    assert all(o.get("written") for o in report["outputs"])


def test_the_published_value_is_the_same_decision(cfg, fake_ha):
    cfg.entities.offset_output = "input_number.offset"
    cfg.entities.fake_temperature_output = "sensor.hpmpc_fake_outdoor"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(apply=True)

    outputs = {o["kind"]: o["value"] for o in report["outputs"]}
    posted = [body for url, body in fake_ha.posted if "hpmpc_fake_outdoor" in url][-1]
    assert posted["state"] == outputs["fake_temperature"]
    assert outputs["fake_temperature"] == pytest.approx(
        outputs["offset"] + report["readings"]["t_outdoor"], abs=0.01
    )


def test_an_output_domain_that_cannot_be_written_is_refused(cfg):
    cfg.entities.fake_temperature_output = "climate.living_room"
    with pytest.raises(ValueError, match="fake_temperature_output"):
        cfg.validate()


# ----------------------------------------- before there is a model to run on


def untrained(cfg, fake_ha) -> Controller:
    return Controller(cfg, None, fake_ha)


def test_a_controller_without_a_model_still_runs(cfg, fake_ha):
    """A fresh install has no model, and refusing to start would be backwards:
    the pump has no other sensor and the fit needs history this loop collects."""
    controller = untrained(cfg, fake_ha)
    assert not controller.trained
    report = controller.step(apply=True)
    assert report["mode"] == "collecting"
    assert report["outputs"] and fake_ha.written


def test_collecting_holds_the_neutral_offset(cfg, fake_ha):
    """fallback_offset is zero by default, which is simply the truth."""
    controller = untrained(cfg, fake_ha)
    report = controller.step(apply=True)
    assert report["offset"] == pytest.approx(cfg.control.fallback_offset, abs=0.01)
    shown = {o["kind"]: o["value"] for o in report["outputs"]}["fake_temperature"]
    assert shown == pytest.approx(report["readings"]["t_outdoor"], abs=0.01)


def test_collecting_says_what_is_missing_and_how_to_fix_it(cfg, fake_ha):
    controller = untrained(cfg, fake_ha)
    notes = " ".join(controller.step(apply=True)["notes"])
    assert "no trained model" in notes
    assert "hpmpc train" in notes


def test_a_model_can_be_adopted_without_a_restart(cfg, fake_ha):
    controller = untrained(cfg, fake_ha)
    assert controller.step(apply=True)["mode"] == "collecting"
    controller.adopt_model(ThermalParams(), cfg)
    assert controller.trained
    assert controller.step(apply=True)["mode"] not in {"collecting"}


def test_an_unknown_outdoor_temperature_writes_nothing(cfg, fake_ha):
    """Every output but the offset is outdoor + offset. With no outdoor
    temperature, writing zero would tell a pump with no other sensor that it is
    0 C outside; holding the last value is the honest answer."""
    cfg.entities.outdoor_temp = "sensor.outdoor"
    cfg.forecast.weather_source = "home_assistant"
    cfg.entities.weather = ""
    controller = untrained(cfg, fake_ha)
    fake_ha.drop("sensor.outdoor")
    report = controller.step(apply=True)
    assert report["outputs"] == []
    assert fake_ha.written == []
    assert "holds its last value" in " ".join(report["notes"])


def test_the_untrained_message_names_the_two_commands(cfg):
    from hpmpc.train import ModelNotTrained, load_model, load_model_if_trained

    with pytest.raises(ModelNotTrained) as excinfo:
        load_model(cfg)
    assert "hpmpc collect" in str(excinfo.value)
    assert "hpmpc train" in str(excinfo.value)
    # The tolerant loader turns the same situation into a state.
    assert load_model_if_trained(cfg)[1] is None
