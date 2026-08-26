"""Physics-level checks: NTC, solar geometry, heat pump, thermal model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpmpc.config import HeatPumpConfig, NTCConfig
from hpmpc.model import heatpump as hp
from hpmpc.model.thermal import (
    Exogenous,
    State,
    ThermalParams,
    simulate,
    steady_state_mass_temp,
)
from hpmpc.ntc import resistance_to_temperature, resolution_check, temperature_to_resistance
from hpmpc.solar import clear_sky_ghi, irradiance_from_cloud_cover, solar_position


# ------------------------------------------------------------------ NTC


@pytest.mark.parametrize("temp", [-25.0, -10.0, 0.0, 12.5, 25.0])
def test_ntc_beta_roundtrip(temp):
    cfg = NTCConfig()
    ohm = temperature_to_resistance(temp, cfg)
    assert resistance_to_temperature(ohm, cfg) == pytest.approx(temp, abs=1e-6)


def test_ntc_is_monotonically_decreasing():
    cfg = NTCConfig()
    ohms = temperature_to_resistance(np.array([-20.0, -10.0, 0.0, 10.0, 20.0]), cfg)
    assert np.all(np.diff(ohms) < 0)


def test_ntc_table_interpolates_and_inverts():
    cfg = NTCConfig(model="table", table_temp_c=[-20, 0, 20], table_ohm=[100000, 40000, 15000])
    assert float(temperature_to_resistance(0.0, cfg)) == pytest.approx(40000)
    assert float(temperature_to_resistance(10.0, cfg)) == pytest.approx(27500)
    assert float(resistance_to_temperature(40000.0, cfg)) == pytest.approx(0.0)


def test_ntc_respects_clipping_bounds():
    cfg = NTCConfig(resistance_max=50000)
    assert float(temperature_to_resistance(-30.0, cfg)) == pytest.approx(50000)


def test_resolution_check_reports_a_positive_step():
    assert resolution_check(NTCConfig(), 0.0, 500.0) > 0


def test_bad_ntc_model_raises():
    with pytest.raises(ValueError, match="Unknown ntc.model"):
        temperature_to_resistance(0.0, NTCConfig(model="nope"))


# ---------------------------------------------------------------- solar


def test_solar_noon_elevation_matches_astronomy():
    # Stockholm at midsummer: the sun peaks a little above 53 degrees.
    index = pd.date_range("2026-06-21 00:00", periods=48, freq="30min", tz="Europe/Stockholm")
    elevation = solar_position(index, 59.33, 18.06)["elevation"]
    assert 52.0 < elevation.max() < 56.0
    # Midwinter is much lower.
    winter = pd.date_range("2026-12-21 00:00", periods=48, freq="30min", tz="Europe/Stockholm")
    assert solar_position(winter, 59.33, 18.06)["elevation"].max() < 8.0


def test_clear_sky_is_zero_below_the_horizon():
    assert float(clear_sky_ghi(np.array([-5.0]))[0]) == 0.0
    assert float(clear_sky_ghi(np.array([60.0]))[0]) > 700.0


def test_cloud_cover_reduces_irradiance_monotonically():
    index = pd.date_range("2026-06-21 12:00", periods=1, freq="h", tz="Europe/Stockholm")
    values = [
        float(irradiance_from_cloud_cover(index, np.array([cc]), 59.33, 18.06).iloc[0])
        for cc in (0, 25, 50, 75, 100)
    ]
    assert all(a > b for a, b in zip(values, values[1:]))


# ------------------------------------------------------------ heat pump


def test_heating_curve_is_decreasing_and_clipped():
    cfg = HeatPumpConfig()
    assert float(hp.supply_setpoint(np.array(0.0), cfg)) > float(hp.supply_setpoint(np.array(10.0), cfg))
    assert float(hp.supply_setpoint(np.array(-60.0), cfg)) == pytest.approx(cfg.supply_max)
    assert float(hp.supply_setpoint(np.array(60.0), cfg)) == pytest.approx(cfg.supply_min)


def test_outdoor_filter_reaches_63_percent_after_one_time_constant():
    filtered = hp.filter_outdoor_series(np.full((1, 12), -10.0), tau_hours=3.0, dt_hours=0.25, initial=0.0)
    assert float(filtered[0, -1]) == pytest.approx(-10.0 * (1 - np.exp(-1)), abs=0.3)


def test_cop_falls_with_lift_and_stays_in_bounds():
    cfg = HeatPumpConfig()
    warm = float(hp.cop(np.array(35.0), np.array(5.0), cfg))
    cold = float(hp.cop(np.array(35.0), np.array(-15.0), cfg))
    assert warm > cold
    assert cfg.cop_min <= cold <= cfg.cop_max


def test_electric_power_includes_standby_and_scales_with_heat():
    cfg = HeatPumpConfig()
    idle = float(hp.electric_power(np.array(0.0), np.array(30.0), np.array(0.0), cfg))
    load = float(hp.electric_power(np.array(4000.0), np.array(30.0), np.array(0.0), cfg))
    assert idle == pytest.approx(cfg.standby_power_w)
    assert load > idle


def test_inverse_curve_returns_the_requested_supply_temperature():
    cfg = HeatPumpConfig()
    offset = hp.offset_for_supply_temp(33.0, -5.0, cfg)
    assert float(hp.supply_setpoint(np.array(-5.0 + offset), cfg)) == pytest.approx(33.0, abs=1e-6)


# ------------------------------------------------------------- thermal


def _exog(steps: int, t_out: float = -5.0, **kwargs) -> Exogenous:
    return Exogenous(np.full(steps, t_out), kwargs.get("wind", 0.0), kwargs.get("sun", 0.0), 1.0)


def test_steady_state_holds_the_setpoint_under_the_default_curve():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 5 * 24 * 4
    result = simulate(
        params, pump, _exog(steps), np.zeros((1, steps)),
        State(21.0, float(steady_state_mass_temp(params, 21.0, -5.0)), -5.0), 0.25,
    )
    assert float(result["t_indoor"][0, -1]) == pytest.approx(21.0, abs=0.5)


def test_negative_offset_makes_the_house_warmer():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 3 * 24 * 4
    state = State(21.0, float(steady_state_mass_temp(params, 21.0, -5.0)), -5.0)
    finals = []
    for offset in (3.0, 0.0, -3.0):
        result = simulate(params, pump, _exog(steps), np.full((1, steps), offset), state, 0.25)
        finals.append(float(result["t_indoor"][0, -1]))
    assert finals[0] < finals[1] < finals[2]


def test_wind_increases_heat_loss():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 48 * 4
    state = State(21.0, float(steady_state_mass_temp(params, 21.0, -5.0)), -5.0)
    calm = simulate(params, pump, _exog(steps, wind=0.0), np.zeros((1, steps)), state, 0.25)
    windy = simulate(params, pump, _exog(steps, wind=12.0), np.zeros((1, steps)), state, 0.25)
    assert float(windy["t_indoor"][0, -1]) < float(calm["t_indoor"][0, -1])


def test_sun_warms_the_house():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 24 * 4
    state = State(21.0, float(steady_state_mass_temp(params, 21.0, -5.0)), -5.0)
    dark = simulate(params, pump, _exog(steps, sun=0.0), np.zeros((1, steps)), state, 0.25)
    bright = simulate(params, pump, _exog(steps, sun=300.0), np.zeros((1, steps)), state, 0.25)
    assert float(bright["t_indoor"][0, -1]) > float(dark["t_indoor"][0, -1])


def test_batch_and_per_window_inputs_agree_with_single_rollouts():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 32
    exog = Exogenous(np.tile(np.linspace(-10, 0, steps), (3, 1)), 1.0, 0.0, 1.0)
    state = State(np.array([20.0, 21.0, 22.0]), np.array([24.0, 25.0, 26.0]), np.full(3, -8.0))
    batch = simulate(params, pump, exog, np.zeros((3, steps)), state, 0.25)
    assert batch["t_indoor"].shape == (3, steps)
    for i in range(3):
        single = simulate(
            params, pump,
            Exogenous(np.linspace(-10, 0, steps), 1.0, 0.0, 1.0),
            np.zeros((1, steps)),
            State(20.0 + i, 24.0 + i, -8.0), 0.25,
        )
        assert float(batch["t_indoor"][i, -1]) == pytest.approx(float(single["t_indoor"][0, -1]), abs=1e-9)


def test_heat_stop_temperature_switches_the_pump_off():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 24 * 4
    warm_outside = pump.heat_stop_temp + 3.0
    result = simulate(
        params, pump, _exog(steps, t_out=warm_outside), np.zeros((1, steps)),
        State(21.0, 22.0, warm_outside), 0.25,
    )
    assert float(np.max(result["q_heat"])) == 0.0


def test_learned_bias_shifts_the_trajectory_in_the_right_direction():
    params, pump = ThermalParams(), HeatPumpConfig()
    steps = 24 * 4
    state = State(21.0, float(steady_state_mass_temp(params, 21.0, -5.0)), -5.0)
    neutral = simulate(params, pump, _exog(steps), np.zeros((1, steps)), state, 0.25)
    warmer = simulate(
        params, pump,
        Exogenous(np.full(steps, -5.0), 0.0, 0.0, 1.0, indoor_bias=np.full(steps, 0.1)),
        np.zeros((1, steps)), state, 0.25,
    )
    assert float(warmer["t_indoor"][0, -1]) > float(neutral["t_indoor"][0, -1])


def test_mismatched_horizon_lengths_are_rejected():
    params, pump = ThermalParams(), HeatPumpConfig()
    with pytest.raises(ValueError, match="steps"):
        simulate(params, pump, _exog(10), np.zeros((1, 12)), State(21.0, 24.0, -5.0), 0.25)


def test_parameter_vector_roundtrip_and_clipping():
    params = ThermalParams()
    assert ThermalParams.from_vector(params.to_vector()).to_dict() == params.to_dict()
    absurd = ThermalParams(Ci=-100.0, Cm=10**9)
    low, high = ThermalParams.bounds()
    clipped = absurd.clipped().to_vector()
    assert np.all(clipped >= low) and np.all(clipped <= high)
