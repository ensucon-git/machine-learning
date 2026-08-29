"""The digital potentiometer's limits are the controller's limits.

A wrong NTC table biases the offset and is largely absorbed by the curve fit. A
pot that cannot reach the commanded resistance does something worse: it clamps,
says nothing, and the model learns a house that does not respond. These tests
are about noticing that."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpmpc.config import Config, NTCConfig, PotConfig
from hpmpc.dataset import add_derived
from hpmpc.ntc import (
    reachable_temperatures,
    resistance_to_temperature,
    resistance_to_wiper,
    temperature_to_resistance,
    wiper_resolution,
    wiper_span,
    wiper_to_resistance,
)

DAIKIN = NTCConfig(
    model="table",
    table_temp_c=[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30],
    table_ohm=[347667, 260639, 196648, 149283, 114003, 87562, 67628, 52514,
               40991, 32159, 25354, 20084, 15984],
)


def test_wiper_and_resistance_are_inverses():
    pot = PotConfig()
    for step in (0, 1, 64, 128, 255):
        ohm = wiper_to_resistance(step, pot)
        assert int(resistance_to_wiper(ohm, pot)) == step


def test_the_wiper_is_an_integer_position_not_a_continuum():
    pot = PotConfig()
    a = wiper_to_resistance(100, pot)
    b = wiper_to_resistance(101, pot)
    assert b - a == pytest.approx(pot.resistance_ohm / (pot.steps - 1))
    # Anything between the two taps rounds to one of them.
    assert int(resistance_to_wiper((a + b) / 2 - 1, pot)) == 100


def test_resistance_outside_the_span_clamps_instead_of_extrapolating():
    pot = PotConfig()
    assert int(resistance_to_wiper(1e9, pot)) == wiper_span(pot)
    assert int(resistance_to_wiper(-500, pot)) == 0


def test_one_mcp41100_cannot_show_a_swedish_winter():
    """The finding that motivated the pot: section existing at all."""
    coldest, warmest = reachable_temperatures(PotConfig(devices=1), DAIKIN)
    assert coldest > -8.0            # nowhere near -20
    assert warmest > 25.0


def test_two_in_series_buy_range_not_resolution():
    one, two = PotConfig(devices=1), PotConfig(devices=2)
    assert reachable_temperatures(two, DAIKIN)[0] < -20.0
    # Same step size in both cases: the extra device adds taps, not precision.
    assert wiper_resolution(two, DAIKIN, 0.0) == pytest.approx(
        wiper_resolution(one, DAIKIN, 0.0)
    )


def test_a_series_resistor_shifts_the_band_colder():
    plain = reachable_temperatures(PotConfig(), DAIKIN)
    shifted = reachable_temperatures(PotConfig(series_ohm=100000), DAIKIN)
    assert shifted[0] < plain[0]
    assert shifted[1] < plain[1]     # and gives up the warm end for it


def test_resolution_is_fine_where_the_range_reaches():
    pot = PotConfig()
    for temp in (-5.0, 0.0, 5.0, 10.0):
        assert wiper_resolution(pot, DAIKIN, temp) < 0.2


# --------------------------------------------------------------- the wiring


def wired(cfg: Config) -> Config:
    cfg.ntc = DAIKIN
    cfg.entities.pot_wiper = "sensor.mcp41100_wiper"
    cfg.heat_pump.perceived_min_c = -7.0
    return cfg


def test_the_wiper_readback_reconstructs_the_applied_offset(cfg):
    """History can be read back from what the ESP32 actually drove."""
    cfg = wired(cfg)
    cfg.entities.offset_output = ""
    cfg.entities.fake_temperature_output = ""
    index = pd.date_range("2026-01-15", periods=8, freq="15min", tz="UTC")
    wanted = 1.5
    outdoor = -2.0
    step = float(resistance_to_wiper(temperature_to_resistance(outdoor + wanted, DAIKIN), cfg.pot))
    frame = pd.DataFrame({"t_indoor": 21.0, "t_outdoor": outdoor, "pot_wiper": step}, index=index)
    assert add_derived(frame, cfg)["offset"].iloc[-1] == pytest.approx(wanted, abs=0.15)


def test_the_kelvin_entity_still_wins_over_the_wiper(cfg):
    cfg = wired(cfg)
    index = pd.date_range("2026-01-15", periods=4, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {"t_indoor": 21.0, "t_outdoor": -2.0, "output_offset": 2.0, "pot_wiper": 200.0},
        index=index,
    )
    assert add_derived(frame, cfg)["offset"].iloc[-1] == pytest.approx(2.0)


def test_the_actuator_check_uses_the_wiper_when_the_pump_is_silent(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = wired(cfg)
    controller = Controller(cfg, ThermalParams(), fake_ha)
    controller.state.last_offset = 1.0
    outdoor = -2.0
    step = int(resistance_to_wiper(temperature_to_resistance(outdoor + 1.0, DAIKIN), cfg.pot))
    result = controller.check_actuator({"t_outdoor": outdoor, "pot_wiper": float(step)})
    assert result["source"] == "pot_wiper"
    assert abs(result["error_now_c"]) < 0.2
    assert "warning" not in result


def test_an_end_stop_is_reported_as_a_hardware_limit(cfg, fake_ha):
    """The failure this readback exists for: the pot cannot go colder."""
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = wired(cfg)
    cfg.heat_pump.perceived_min_c = -25.0        # asking for more than the hardware has
    controller = Controller(cfg, ThermalParams(), fake_ha)
    controller.state.last_offset = -5.0
    result = controller.check_actuator({"t_outdoor": -12.0, "pot_wiper": float(wiper_span(cfg.pot))})
    assert "end stop" in result["warning"]
    assert "perceived_min_c" in result["warning"]


def test_a_wiper_that_ignores_the_command_is_reported(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = wired(cfg)
    controller = Controller(cfg, ThermalParams(), fake_ha)
    controller.state.last_offset = 0.0
    result = controller.check_actuator({"t_outdoor": 0.0, "pot_wiper": 60.0})
    assert "not arriving" in result["warning"]


def test_the_pump_reading_still_wins_when_both_exist(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = wired(cfg)
    cfg.entities.pump_outdoor_temp = "sensor.daikin_outdoor"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    result = controller.check_actuator({"t_outdoor": 0.0, "pump_outdoor": 0.2, "pot_wiper": 172.0})
    assert result["source"] == "pump"


def test_a_wiper_output_is_written_alongside_the_others(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg = wired(cfg)
    cfg.entities.wiper_output = "number.mcp41100_target"
    controller = Controller(cfg, ThermalParams(), fake_ha)
    outputs = {o["kind"]: o for o in controller.outputs(1.0, -2.0)}
    assert outputs["wiper"]["unit"] == "step"
    assert 0 <= outputs["wiper"]["value"] <= wiper_span(cfg.pot)
    # Same decision as the other outputs, just quantised.
    ohm = float(wiper_to_resistance(outputs["wiper"]["value"], cfg.pot))
    assert float(resistance_to_temperature(ohm, DAIKIN)) == pytest.approx(-1.0, abs=0.2)


def test_the_pot_must_be_physically_possible(cfg):
    cfg.pot.steps = 1
    with pytest.raises(ValueError, match="pot.steps"):
        cfg.validate()


def test_arrays_go_through_the_conversion_in_one_call():
    pot = PotConfig()
    steps = np.array([0.0, 50.0, 255.0])
    ohms = wiper_to_resistance(steps, pot)
    assert ohms.shape == steps.shape
    assert np.allclose(resistance_to_wiper(ohms, pot), steps)


# ----------------------------------------- weather the hardware cannot reach
#
# The pump has no outdoor sensor of its own: the potentiometer IS its sensor.
# So running out of range must never mean "stop commanding" - the pump would be
# left with an open circuit. It means "command the coldest thing the hardware
# has, keep the heat coming, and say what it is costing".


def controller_for(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    return Controller(wired(cfg), ThermalParams(), fake_ha)


def test_below_the_floor_it_still_writes(cfg, fake_ha):
    controller = controller_for(cfg, fake_ha)
    fake_ha.set("sensor.outdoor", -12.0)
    report = controller.step(apply=True)
    assert report["outputs"], "the pump must never be left without a value"
    assert fake_ha.written


def test_below_the_floor_it_asks_for_all_the_heat_it_can(cfg, fake_ha):
    """Pinned at the coldest the pot can present, which is maximum heat."""
    controller = controller_for(cfg, fake_ha)
    fake_ha.set("sensor.outdoor", -12.0)
    report = controller.step(apply=True)
    shown = {o["kind"]: o["value"] for o in report["outputs"]}["fake_temperature"]
    assert shown == pytest.approx(cfg.heat_pump.perceived_min_c, abs=0.05)


def test_the_shortfall_is_quantified_not_just_flagged(cfg, fake_ha):
    """Knowing "it is limited" is not much use; knowing how much heat is missing is."""
    controller = controller_for(cfg, fake_ha)
    fake_ha.set("sensor.outdoor", -12.0)
    report = controller.step(apply=True)
    shortfall = report["range_shortfall"]
    assert shortfall["gap_c"] == pytest.approx(5.0, abs=0.05)
    expected = 5.0 * cfg.heat_pump.curve_slope
    assert shortfall["supply_shortfall_c"] == pytest.approx(expected, abs=0.05)
    assert "drift cool" in shortfall["warning"]


def test_no_complaint_while_the_hardware_can_keep_up(cfg, fake_ha):
    controller = controller_for(cfg, fake_ha)
    fake_ha.set("sensor.outdoor", -3.0)
    assert controller.range_shortfall(-3.0) is None
    assert "range_shortfall" not in controller.step(apply=True)


def test_a_second_potentiometer_removes_the_shortfall(cfg, fake_ha):
    """The whole point of pot.devices: 2 - the same weather becomes controllable."""
    cfg = wired(cfg)
    cfg.pot.devices = 2
    cfg.heat_pump.perceived_min_c = -20.0
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    controller = Controller(cfg, ThermalParams(), fake_ha)
    fake_ha.set("sensor.outdoor", -12.0)
    report = controller.step(apply=True)
    assert "range_shortfall" not in report
    shown = {o["kind"]: o["value"] for o in report["outputs"]}["fake_temperature"]
    assert shown < -12.0 + 0.1        # free to ask for more heat than the truth


def test_the_shortfall_reaches_home_assistant(cfg, fake_ha):
    cfg = wired(cfg)
    cfg.entities.status_entity = "sensor.hpmpc_status"
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    controller = Controller(cfg, ThermalParams(), fake_ha)
    fake_ha.set("sensor.outdoor", -12.0)
    controller.step(apply=True)
    posted = [body for path, body in fake_ha.posted if "hpmpc_status" in path]
    assert posted and posted[-1]["attributes"]["range_shortfall_c"] == pytest.approx(5.0, abs=0.05)


def test_even_the_sensor_failure_path_keeps_the_pump_supplied(cfg, fake_ha):
    """A stale indoor sensor must not leave the pump's own sensor input dark."""
    controller = controller_for(cfg, fake_ha)
    fake_ha.set("sensor.indoor", 21.0, age_minutes=600)
    fake_ha.set("sensor.outdoor", -12.0)
    report = controller.step(apply=True)
    assert report["mode"] == "fallback"
    assert report["outputs"] and fake_ha.written


def test_the_perceived_floor_can_be_raised_at_runtime(cfg):
    """So fitting the second pot does not mean editing config.yaml."""
    from hpmpc.settings import OVERRIDABLE, apply

    assert "heat_pump.perceived_min_c" in OVERRIDABLE
    working, notes = apply(cfg, {"heat_pump.perceived_min_c": -20.0})
    assert working.heat_pump.perceived_min_c == pytest.approx(-20.0)
    assert notes


def test_the_potentiometer_geometry_is_settable_without_editing_the_file(cfg):
    """docs/HARDWARE.md tells you to run these; they have to exist."""
    from hpmpc.settings import OVERRIDABLE, coerce

    for field in ("pot.devices", "pot.resistance_ohm", "pot.wiper_ohm", "pot.series_ohm"):
        assert field in OVERRIDABLE, f"{field} must be settable with 'hpmpc set'"
    # A count stays a count rather than becoming 2.0 in the file.
    assert coerce("pot.devices", "2") == 2
    assert isinstance(coerce("pot.devices", 2.0), int)


def test_setting_two_devices_moves_the_reachable_range(cfg):
    """The whole point of the upgrade the guide describes."""
    from hpmpc.settings import apply

    cfg.ntc = DAIKIN
    before = reachable_temperatures(cfg.pot, cfg.ntc)[0]
    working, _ = apply(cfg, {"pot.devices": 2, "heat_pump.perceived_min_c": -20.0})
    after = reachable_temperatures(working.pot, working.ntc)[0]
    assert before > -8.0 and after < -20.0
