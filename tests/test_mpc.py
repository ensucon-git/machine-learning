from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpmpc.model.thermal import Exogenous, State, ThermalParams, steady_state_mass_temp
from hpmpc.mpc import MpcSolver, block_layout, expand_blocks


def scenario(cfg, cheap_night: bool = True, t_out: float = -5.0):
    steps = int(cfg.control.horizon_hours * 60 / cfg.control.step_minutes)
    index = pd.date_range("2026-01-15 00:00", periods=steps, freq=f"{cfg.control.step_minutes}min", tz="UTC")
    hour = index.hour + index.minute / 60.0
    price = np.where((hour >= 7) & (hour < 10), 3.0, np.where(hour < 6, 0.3, 1.0)) if cheap_night else np.ones(steps)
    return Exogenous(np.full(steps, t_out), np.full(steps, 2.0), np.zeros(steps), price)


def start_state(params: ThermalParams, t_out: float = -5.0, setpoint: float = 21.0) -> State:
    return State(setpoint, float(steady_state_mass_temp(params, setpoint, t_out)), t_out)


def test_block_layout_covers_the_whole_horizon(cfg):
    sizes = block_layout(cfg)
    assert int(sizes.sum()) == int(cfg.control.horizon_hours * 60 / cfg.control.step_minutes)


def test_expand_blocks_repeats_each_block(cfg):
    sizes = block_layout(cfg)
    expanded = expand_blocks(np.arange(len(sizes))[None, :], sizes)
    assert expanded.shape == (1, int(sizes.sum()))
    assert expanded[0, 0] == 0 and expanded[0, -1] == len(sizes) - 1


def test_solution_respects_the_offset_limits(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    result = solver.solve(scenario(cfg), start_state(params))
    assert np.all(result.offset_blocks >= cfg.control.offset_min - 1e-9)
    assert np.all(result.offset_blocks <= cfg.control.offset_max + 1e-9)
    assert result.offset_now == pytest.approx(result.offset_blocks[0])


def test_mpc_beats_every_constant_offset_on_its_own_objective(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    exog, state = scenario(cfg), start_state(params)
    result = solver.solve(exog, state)
    grid = np.linspace(cfg.control.offset_min, cfg.control.offset_max, 21)
    totals, _, _ = solver.evaluate(np.repeat(grid[:, None], solver.n_blocks, axis=1), exog, state, 0.0)
    assert result.cost.total <= float(totals.min()) + 1e-6


def test_cheap_hours_get_more_heat_than_expensive_hours(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    exog = scenario(cfg)
    result = solver.solve(exog, start_state(params))
    price = np.asarray(exog.price)[0]
    cheap = result.offset_schedule[price < 0.5]
    dear = result.offset_schedule[price > 2.0]
    # A more negative offset means "tell the pump it is colder", i.e. more heat.
    assert cheap.mean() < dear.mean()


def test_a_flat_price_produces_a_nearly_flat_offset(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    result = solver.solve(scenario(cfg, cheap_night=False), start_state(params))
    assert result.offset_blocks.std() < 1.2


def test_stored_energy_is_credited_so_ending_cold_is_not_free(cfg):
    """The horizon-end cheat: a plan that dumps heat late looks cheap on the
    electricity meter, and must not look cheap on the net cost."""
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    exog, state = scenario(cfg), start_state(params)

    warm = np.full((1, solver.n_blocks), -3.0)
    cold = np.full((1, solver.n_blocks), 3.0)
    _, parts, _ = solver.evaluate(np.vstack([warm, cold]), exog, state, 0.0)
    warm_cost = solver._breakdown(parts, 0)
    cold_cost = solver._breakdown(parts, 1)

    assert cold_cost.energy_sek < warm_cost.energy_sek        # coasting buys less power ...
    assert cold_cost.stored_value_sek < warm_cost.stored_value_sek  # ... by emptying the slab
    gross_gap = warm_cost.energy_sek - cold_cost.energy_sek
    net_gap = warm_cost.net_cost_sek - cold_cost.net_cost_sek
    assert net_gap < gross_gap


def test_comfort_violations_are_penalised(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    exog = scenario(cfg)
    freezing = State(cfg.control.hard_min - 1.0, cfg.control.hard_min - 1.0, -5.0)
    _, parts, _ = solver.evaluate(np.zeros((1, solver.n_blocks)), exog, freezing, 0.0)
    assert parts["hard"][0] > 0.0
    assert parts["comfort"][0] > 0.0


def test_warm_start_makes_a_repeated_solve_no_worse(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    exog, state = scenario(cfg), start_state(params)
    first = solver.solve(exog, state)
    second = solver.solve(exog, state)
    assert second.cost.total <= first.cost.total + 1e-6


def test_smoothness_penalty_limits_the_first_jump(cfg):
    params = ThermalParams()
    cfg.control.weight_offset_change = 50.0
    solver = MpcSolver(cfg, params)
    result = solver.solve(scenario(cfg), start_state(params), previous_offset=0.0)
    assert abs(result.offset_now) < 2.0


def test_wrong_horizon_length_is_rejected(cfg):
    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    short = Exogenous(np.full(4, -5.0), 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="forecast has"):
        solver.solve(short, start_state(params))


def test_summary_is_json_friendly(cfg):
    import json

    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    summary = solver.solve(scenario(cfg), start_state(params)).summary()
    json.dumps(summary)
    assert {"offset_now", "baseline_matched", "predicted_saving_sek", "cost"} <= set(summary)


# ---------------------------------------------------- comfort as a schedule


def _winter(cfg, hours: float = 36.0):
    """A cold day with an expensive morning, for setback tests.

    Setback needs a real horizon and a real search: reheating a slab is a
    36-hour decision, and finding "coast, then commit" is not something four
    tiny CEM iterations will stumble on.
    """
    cfg.control.horizon_hours = hours
    cfg.control.block_hours = 2.0
    cfg.optimizer.population = 192
    cfg.optimizer.elites = 20
    cfg.optimizer.iterations = 10
    cfg.optimizer.polish = True
    steps = int(cfg.control.horizon_hours * 60 / cfg.control.step_minutes)
    index = pd.date_range("2026-02-10 00:00", periods=steps, freq=f"{cfg.control.step_minutes}min", tz="UTC")
    hour = index.hour + index.minute / 60.0
    price = np.where((hour >= 7) & (hour < 10), 2.8, np.where(hour < 6, 0.4, 1.1))
    return index, Exogenous(np.full(steps, -6.0), np.full(steps, 3.0), np.zeros(steps), price)


def test_a_time_varying_band_is_accepted_and_reported(cfg):
    from hpmpc.comfort import ComfortSchedule

    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    schedule = ComfortSchedule.flat(cfg.control, solver.steps, mode="away")
    schedule.setpoint[solver.steps // 2 :] += 3.0
    result = solver.solve(scenario(cfg), start_state(params), comfort=schedule)
    assert result.diagnostics["comfort"]["mode"] == "away"
    assert result.diagnostics["comfort"]["varies_over_horizon"] is True


def test_a_wrong_length_schedule_is_rejected(cfg):
    from hpmpc.comfort import ComfortSchedule

    params = ThermalParams()
    solver = MpcSolver(cfg, params)
    with pytest.raises(ValueError, match="comfort schedule has"):
        solver.solve(scenario(cfg), start_state(params), comfort=ComfortSchedule.flat(cfg.control, 4))


def test_setback_actually_lets_the_house_cool(cfg):
    """The heating curve is built to hold the normal setpoint, so reaching a
    deep setback needs the offset authority the holiday profile grants."""
    from hpmpc.comfort import apply_mode, build_schedule

    cfg.heat_pump.model = "daikin_erlq016caw1"
    params = ThermalParams()
    index, exog = _winter(cfg)
    state = State(21.0, float(steady_state_mass_temp(params, 21.0, -6.0)), -6.0)

    schedule = build_schedule(cfg, index, "holiday")
    result = MpcSolver(apply_mode(cfg, "holiday"), params).solve(exog, state, 0.0, comfort=schedule)

    assert result.offset_now > 3.0                                   # telling it it is warm out
    assert result.trajectory["t_indoor"][-1] < 19.0                  # and the house does cool
    assert float(np.max(result.trajectory["q_backup"])) == 0.0       # without touching the immersion heater


def test_a_return_time_makes_the_optimiser_reheat_in_advance(cfg):
    """The whole point of telling it when you are back: a ten-hour slab does not
    warm up on arrival, so the plan has to start hours earlier - and how many
    hours is something the optimiser works out, not a fixed lead time."""
    from hpmpc.comfort import apply_mode, build_schedule

    cfg.heat_pump.model = "daikin_erlq016caw1"
    params = ThermalParams()
    index, exog = _winter(cfg)
    state = State(16.0, float(steady_state_mass_temp(params, 16.0, -6.0)), -6.0)
    solver_cfg = apply_mode(cfg, "holiday")

    coasting = MpcSolver(solver_cfg, params).solve(
        exog, state, 0.0, comfort=build_schedule(cfg, index, "holiday")
    )
    return_step = int(20 * 60 / cfg.control.step_minutes)
    schedule = build_schedule(cfg, index, "holiday", index[return_step].to_pydatetime())
    returning = MpcSolver(solver_cfg, params).solve(exog, state, 0.0, comfort=schedule)

    # Heat is bought before the return, not after it.
    assert returning.offset_schedule[:return_step].mean() < coasting.offset_schedule[:return_step].mean() - 5.0
    # And the house is actually warm when you walk in.
    arrival = returning.trajectory["t_indoor"][return_step]
    assert arrival >= schedule.comfort_min[return_step] - 0.3
    assert arrival > coasting.trajectory["t_indoor"][return_step] + 2.0


def test_more_notice_means_a_later_and_cheaper_start(cfg):
    """With more slack the optimiser coasts longer before reheating."""
    from hpmpc.comfort import apply_mode, build_schedule

    cfg.heat_pump.model = "daikin_erlq016caw1"
    params = ThermalParams()
    index, exog = _winter(cfg)
    state = State(16.0, float(steady_state_mass_temp(params, 16.0, -6.0)), -6.0)
    solver_cfg = apply_mode(cfg, "holiday")

    def first_heating_step(hours: float) -> int:
        step = int(hours * 60 / cfg.control.step_minutes)
        schedule = build_schedule(cfg, index, "holiday", index[step].to_pydatetime())
        result = MpcSolver(solver_cfg, params).solve(exog, state, 0.0, comfort=schedule)
        heating = np.flatnonzero(result.offset_schedule < -2.0)
        return int(heating[0]) if heating.size else len(result.offset_schedule)

    assert first_heating_step(30.0) > first_heating_step(20.0)
