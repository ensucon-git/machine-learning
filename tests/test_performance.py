"""The pump's own performance map: COP, capacity and the backup heater."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from hpmpc.config import Config, HeatPumpConfig
from hpmpc.model import build_pump
from hpmpc.model.heatpump import PumpModel
from hpmpc.model.performance import load_performance_map, performance_map_from_dict
from hpmpc.model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp

DAIKIN = "daikin_erlq016caw1"

# The rating points the bundled table is anchored on.
ANCHORS = [(7, 35, 4.4), (7, 45, 3.3), (7, 55, 2.6), (2, 35, 3.6), (-7, 35, 2.8), (-7, 55, 1.9), (-15, 35, 2.3)]


@pytest.fixture(scope="module")
def daikin():
    return load_performance_map(DAIKIN)


@pytest.mark.parametrize("ambient,supply,expected", ANCHORS)
def test_map_reproduces_its_anchor_points(daikin, ambient, supply, expected):
    got = float(daikin.cop(np.array(float(ambient)), np.array(float(supply))))
    assert got == pytest.approx(expected, abs=0.06)


def test_cop_falls_with_lift(daikin):
    assert daikin.cop(np.array(7.0), np.array(35.0)) > daikin.cop(np.array(7.0), np.array(55.0))
    assert daikin.cop(np.array(7.0), np.array(35.0)) > daikin.cop(np.array(-15.0), np.array(35.0))


def test_cop_rises_monotonically_with_ambient_and_has_no_jumps(daikin):
    # Stop below the COP ceiling, where the curve is genuinely flat by design.
    ambient = np.linspace(-25.0, 10.0, 400)
    values = daikin.cop(ambient, np.full_like(ambient, 35.0))
    steps = np.diff(values)
    assert np.all(steps > -1e-9)
    # Bilinear interpolation is piecewise linear, so check for jumps rather than
    # curvature: no grid crossing may produce a step out of scale with the rest.
    assert float(steps.max()) < 4.0 * float(steps.mean())


def test_extrapolating_off_the_table_stays_physical(daikin):
    """Efficiency is clamped at the edge; COP keeps falling because the lift
    keeps growing. That is the point of tabulating efficiency rather than COP."""
    edge_efficiency = float(daikin.efficiency(np.array(-25.0), np.array(35.0)))
    beyond_efficiency = float(daikin.efficiency(np.array(-40.0), np.array(35.0)))
    assert beyond_efficiency == pytest.approx(edge_efficiency)

    beyond = float(daikin.cop(np.array(-40.0), np.array(35.0)))
    assert 1.0 <= beyond < float(daikin.cop(np.array(-25.0), np.array(35.0)))


def test_capacity_falls_with_cold_and_with_supply_temperature(daikin):
    warm = float(daikin.capacity_w(np.array(7.0), np.array(35.0)))
    cold = float(daikin.capacity_w(np.array(-15.0), np.array(35.0)))
    hot_water = float(daikin.capacity_w(np.array(7.0), np.array(55.0)))
    assert warm == pytest.approx(16000, rel=0.05)
    assert cold < warm
    assert hot_water < warm


def test_compressor_stops_below_the_operating_limit(daikin):
    assert float(daikin.capacity_w(np.array(-30.0), np.array(35.0))) == 0.0


def test_defrost_only_bites_near_freezing_and_when_humid(daikin):
    dry_cold = float(daikin.defrost_factor(np.array(-15.0), np.array(95.0)))
    humid_near_zero = float(daikin.defrost_factor(np.array(1.0), np.array(100.0)))
    dry_near_zero = float(daikin.defrost_factor(np.array(1.0), np.array(50.0)))
    assert dry_cold == pytest.approx(1.0, abs=0.01)
    assert humid_near_zero < dry_near_zero < 1.001
    assert humid_near_zero > 1.0 - daikin.defrost_max_derate - 1e-9


def test_unknown_humidity_is_treated_as_the_reference(daikin):
    assert float(daikin.defrost_factor(np.array(1.0), None)) == pytest.approx(
        float(daikin.defrost_factor(np.array(1.0), np.array(daikin.defrost_humidity_reference_pct)))
    )


# ------------------------------------------------------------ backup heater


def test_backup_heater_covers_the_shortfall_at_a_terrible_cop(daikin):
    pump = PumpModel(HeatPumpConfig(standby_power_w=0.0), daikin)
    ambient, supply = np.array(-15.0), np.array(40.0)
    capacity = float(pump.capacity_w(ambient, supply))
    result = pump.deliver(np.array(capacity + 3000.0), supply, ambient)
    assert float(result["q_compressor"]) == pytest.approx(capacity)
    assert float(result["q_backup"]) == pytest.approx(3000.0)
    # The marginal 3 kW arrives at COP 1, so it costs 3 kW of electricity.
    without = pump.deliver(np.array(capacity), supply, ambient)
    assert float(result["p_electric"] - without["p_electric"]) == pytest.approx(3000.0, rel=0.01)


def test_demand_beyond_compressor_and_backup_is_simply_not_met(daikin):
    pump = PumpModel(HeatPumpConfig(), daikin)
    ambient, supply = np.array(-20.0), np.array(45.0)
    result = pump.deliver(np.array(100000.0), supply, ambient)
    ceiling = float(pump.capacity_w(ambient, supply)) + daikin.backup_capacity_w()
    assert float(result["q_heat"]) == pytest.approx(ceiling)


def test_disabling_the_backup_heater_removes_it(daikin):
    from dataclasses import replace

    pump = PumpModel(HeatPumpConfig(), replace(daikin, backup_enabled=False))
    ambient, supply = np.array(-20.0), np.array(45.0)
    result = pump.deliver(np.array(100000.0), supply, ambient)
    assert float(result["q_backup"]) == 0.0


# -------------------------------------------------------------- build_pump


def test_build_pump_falls_back_to_carnot_without_a_model(cfg):
    pump = build_pump(cfg)
    assert pump.performance is None
    assert "Carnot" in pump.describe()["model"]


def test_build_pump_loads_the_named_map(cfg):
    cfg.heat_pump.model = DAIKIN
    pump = build_pump(cfg)
    assert pump.performance is not None
    assert "ERLQ016CAW1" in pump.describe()["model"]


def test_efficiency_scale_from_config_reaches_the_map(cfg):
    cfg.heat_pump.model = DAIKIN
    base = float(build_pump(cfg).cop(np.array(35.0), np.array(7.0)))
    cfg.heat_pump.efficiency_scale = 0.8
    scaled = float(build_pump(cfg).cop(np.array(35.0), np.array(7.0)))
    assert scaled == pytest.approx(0.8 * base, rel=1e-6)


def test_supply_setpoint_respects_the_map_ceiling(cfg):
    cfg.heat_pump.model = DAIKIN
    cfg.heat_pump.supply_max = 38.0
    assert float(build_pump(cfg).supply_setpoint(np.array(-40.0))) == pytest.approx(38.0)


# ------------------------------------------------------------- validation


def _minimal_map() -> dict:
    return {
        "name": "test",
        "efficiency": {"ambient_c": [-10, 10], "supply_c": [30, 50], "values": [[0.4, 0.38], [0.42, 0.40]]},
        "capacity": {"ambient_c": [-10, 10], "kw_at_w35": [8.0, 12.0]},
    }


def test_a_minimal_map_loads():
    assert performance_map_from_dict(_minimal_map()).name == "test"


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d["efficiency"].__setitem__("values", [[0.4]]), "rows"),
        (lambda d: d["efficiency"].__setitem__("ambient_c", [10, -10]), "increasing"),
        (lambda d: d["efficiency"].__setitem__("values", [[4.0, 3.8], [4.2, 4.0]]), "Carnot"),
    ],
)
def test_malformed_maps_are_rejected_with_a_useful_message(mutate, message):
    data = _minimal_map()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        performance_map_from_dict(data)


def test_unknown_model_name_lists_what_is_available():
    with pytest.raises(ValueError, match="Bundled models"):
        load_performance_map("no_such_pump")


def test_a_custom_map_can_be_loaded_from_a_path(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump(_minimal_map()), encoding="utf-8")
    assert load_performance_map(str(path)).name == "test"


# --------------------------------------------------- effect on simulation


def test_simulation_reports_backup_heat_when_the_compressor_runs_out():
    cfg = Config()
    cfg.entities.indoor_temp = "sensor.a"
    cfg.entities.outdoor_temp = "sensor.b"
    cfg.heat_pump.model = DAIKIN
    cfg.validate()
    pump = build_pump(cfg)
    # A leaky house in deep cold, asked for a lot of heat.
    params = ThermalParams(Hie=400.0, Him=2000.0, Hfloor=4000.0)
    steps = 48
    exog = Exogenous(np.full(steps, -20.0), 6.0, 0.0, 1.0, humidity=90.0)
    result = simulate(
        params, pump, exog, np.full((1, steps), -6.0),
        State(21.0, float(steady_state_mass_temp(params, 21.0, -20.0)), -20.0), 0.25,
    )
    assert float(np.max(result["q_backup"])) > 0.0
    assert float(np.max(result["p_electric"])) > 5000.0


def test_perceived_temperature_is_clamped_inside_the_model():
    cfg = Config()
    cfg.entities.indoor_temp = "sensor.a"
    cfg.entities.outdoor_temp = "sensor.b"
    cfg.heat_pump.perceived_min_c = -10.0
    cfg.validate()
    pump = build_pump(cfg)
    params = ThermalParams()
    steps = 96
    exog = Exogenous(np.full(steps, -5.0), 0.0, 0.0, 1.0)
    state = State(21.0, float(steady_state_mass_temp(params, 21.0, -5.0)), -5.0)
    # -5 with an offset of -20 would be -25, but the pump may only be shown -10.
    huge = simulate(params, pump, exog, np.full((1, steps), -20.0), state, 0.25)
    exact = simulate(params, pump, exog, np.full((1, steps), -5.0), state, 0.25)
    assert float(huge["t_supply"][0, -1]) == pytest.approx(float(exact["t_supply"][0, -1]))
