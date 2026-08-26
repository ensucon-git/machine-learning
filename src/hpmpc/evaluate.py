"""Closed-loop backtest.

Replays a stretch of history: at every control cycle the MPC re-plans against
the weather and prices that actually occurred, applies its first block, and the
identified model is stepped forward as the plant.

Important caveat, stated up front: the plant here *is* the identified model, so
this measures the value of the optimisation, not the accuracy of the model.
Model accuracy is what ``hpmpc train`` reports (validation RMSE against real
measurements); the two numbers answer different questions and both matter.
Forecasts are also perfect in this replay, so treat the saving as an upper
bound - typically the real result lands somewhat below it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .model import heatpump as hp
from .model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp
from .mpc import MpcSolver
from .residual import ResidualModel

log = logging.getLogger(__name__)


def _exog_slice(frame: pd.DataFrame, start: int, steps: int, bias: np.ndarray | None) -> Exogenous:
    sl = slice(start, start + steps)
    return Exogenous(
        frame["t_outdoor"].to_numpy(dtype=float)[sl],
        frame["wind"].to_numpy(dtype=float)[sl],
        frame["solar_ghi"].to_numpy(dtype=float)[sl],
        frame["price"].to_numpy(dtype=float)[sl],
        indoor_bias=np.zeros(steps) if bias is None else bias[sl],
    )


def _stored_energy_reference(cfg: Config, params: ThermalParams, frame: pd.DataFrame, n_steps: int) -> dict[str, float]:
    """Common yardstick for valuing whatever heat a run leaves in the building."""
    tail = max(1, int(round(6.0 / (cfg.control.step_minutes / 60.0))))
    window = slice(max(0, n_steps - tail), n_steps)
    t_end = float(frame["t_outdoor"].to_numpy(dtype=float)[window].mean())
    price_ref = float(frame["price"].to_numpy(dtype=float)[window].mean())
    supply_ref = float(hp.supply_setpoint(np.array(t_end), cfg.heat_pump))
    return {
        "t_mass_target": float(steady_state_mass_temp(params, cfg.control.setpoint, t_end)),
        "price_ref": price_ref,
        "cop_ref": float(hp.cop(np.array(supply_ref), np.array(t_end), cfg.heat_pump)),
    }


def _stored_value(cfg: Config, params: ThermalParams, ti: float, tm: float, ref: dict[str, float]) -> float:
    """Value in SEK of the heat a run leaves behind, relative to the reference.

    Without this the comparison rewards whichever run happens to end with a
    colder slab - an artefact that shrinks as the replay gets longer but never
    disappears, and that can easily swamp a few percent of genuine saving on a
    short period.
    """
    stored_wh = params.Cm * (tm - ref["t_mass_target"]) + params.Ci * (ti - cfg.control.setpoint)
    return stored_wh / 1000.0 * ref["price_ref"] / max(ref["cop_ref"], 1e-3)


def _run_policy(
    cfg: Config,
    params: ThermalParams,
    frame: pd.DataFrame,
    offsets_fn,
    bias: np.ndarray | None,
    state: State,
    cycle_steps: int,
    n_cycles: int,
    horizon_steps: int,
    reference: dict[str, float],
) -> dict[str, Any]:
    dt = cfg.control.step_minutes / 60.0
    ti, tm, tf = state.t_indoor, state.t_mass, state.t_filtered_outdoor
    indoor, power, cost, applied = [], [], [], []
    previous = 0.0

    for cycle in range(n_cycles):
        start = cycle * cycle_steps
        exog = _exog_slice(frame, start, horizon_steps, bias)
        offset = offsets_fn(exog, State(ti, tm, tf), previous, cycle)
        previous = float(offset)

        # Apply only the first cycle worth of the plan, then re-plan.
        plant_exog = _exog_slice(frame, start, cycle_steps, bias)
        result = simulate(
            params,
            cfg.heat_pump,
            plant_exog,
            np.full((1, cycle_steps), float(offset)),
            State(ti, tm, tf),
            dt,
        )
        ti = float(result["t_indoor"][0, -1])
        tm = float(result["t_mass"][0, -1])
        tf = float(result["t_filtered_outdoor"][0, -1])

        kwh = result["p_electric"][0] * dt / 1000.0
        indoor.extend(result["t_indoor"][0].tolist())
        power.extend(result["p_electric"][0].tolist())
        cost.extend((kwh * plant_exog.price[0]).tolist())
        applied.extend([float(offset)] * cycle_steps)

    return _summarise_run(
        cfg, params, np.array(indoor), np.array(power), np.array(cost), np.array(applied), dt,
        _stored_value(cfg, params, ti, tm, reference), ti, tm,
    )


def _summarise_run(
    cfg: Config,
    params: ThermalParams,
    indoor_arr: np.ndarray,
    power: np.ndarray,
    cost: np.ndarray,
    applied: np.ndarray,
    dt: float,
    stored_value_sek: float,
    final_indoor: float,
    final_mass: float,
) -> dict[str, Any]:
    c = cfg.control
    return {
        "stored_value_sek": round(float(stored_value_sek), 3),
        "net_cost_sek": round(float(np.sum(cost) - stored_value_sek), 3),
        "final_indoor": round(float(final_indoor), 3),
        "final_mass": round(float(final_mass), 3),
        "kwh": round(float(np.sum(power) * dt / 1000.0), 3),
        "cost_sek": round(float(np.sum(cost)), 3),
        "indoor_mean": round(float(indoor_arr.mean()), 3),
        "indoor_min": round(float(indoor_arr.min()), 3),
        "indoor_max": round(float(indoor_arr.max()), 3),
        "hours_below_comfort": round(float(np.sum(indoor_arr < c.comfort_min) * dt), 2),
        "hours_above_comfort": round(float(np.sum(indoor_arr > c.comfort_max) * dt), 2),
        "kelvin_hours_outside_comfort": round(
            float(
                np.sum(np.maximum(c.comfort_min - indoor_arr, 0.0) + np.maximum(indoor_arr - c.comfort_max, 0.0)) * dt
            ),
            3,
        ),
        "offset_mean": round(float(np.mean(applied)), 3),
        "series": {"indoor": indoor_arr, "offset": applied, "power": power},
    }


def _constant_offset_runs(
    cfg: Config,
    params: ThermalParams,
    frame: pd.DataFrame,
    bias: np.ndarray | None,
    state: State,
    n_steps: int,
    reference: dict[str, float],
) -> dict[float, dict[str, Any]]:
    """Evaluate every constant-offset reference in one batched rollout.

    A constant offset needs no re-planning, so the whole period is a single
    continuous simulation - which makes 25 references cost about as much as one.
    """
    dt = cfg.control.step_minutes / 60.0
    exog = _exog_slice(frame, 0, n_steps, bias)
    grid = np.round(np.linspace(cfg.control.offset_min, cfg.control.offset_max, 25), 3)
    result = simulate(
        params,
        cfg.heat_pump,
        exog,
        np.repeat(grid[:, None], n_steps, axis=1),
        state,
        dt,
    )
    price = np.asarray(exog.price, dtype=float)
    runs: dict[float, dict[str, Any]] = {}
    for i, value in enumerate(grid):
        power = result["p_electric"][i]
        cost = power * dt / 1000.0 * price[0]
        ti = float(result["t_indoor"][i, -1])
        tm = float(result["t_mass"][i, -1])
        runs[float(value)] = _summarise_run(
            cfg, params, result["t_indoor"][i], power, cost, np.full(n_steps, float(value)), dt,
            _stored_value(cfg, params, ti, tm, reference), ti, tm,
        )
    return runs


def backtest(
    cfg: Config,
    params: ThermalParams,
    frame: pd.DataFrame,
    days: float = 7.0,
    residual: ResidualModel | None = None,
) -> dict[str, Any]:
    """Compare the MPC against constant-offset control over the same period."""
    step_minutes = cfg.control.step_minutes
    if cfg.training.resample_minutes != step_minutes:
        frame = frame.resample(f"{step_minutes}min").mean().interpolate(limit_direction="both")

    horizon_steps = int(round(cfg.control.horizon_hours * 60 / step_minutes))
    cycle_steps = max(1, int(round(cfg.control.cycle_minutes / step_minutes)))
    needed = int(round(days * 24 * 60 / step_minutes)) + horizon_steps
    if len(frame) < needed:
        days = max(1.0, (len(frame) - horizon_steps) * step_minutes / 60.0 / 24.0)
        needed = len(frame)
        log.warning("Backtest shortened to %.1f days - not enough data for the full request", days)
    frame = frame.iloc[-needed:].copy()
    n_cycles = (len(frame) - horizon_steps) // cycle_steps
    if n_cycles < 1:
        raise ValueError("Not enough data for even one control cycle plus a full horizon")

    bias = None
    if residual is not None:
        bias = residual.predict(frame.index, frame)

    t0 = float(frame["t_outdoor"].iloc[0])
    ti0 = float(frame["t_indoor"].iloc[0]) if "t_indoor" in frame else cfg.control.setpoint
    initial = State(ti0, float(steady_state_mass_temp(params, ti0, t0)), t0)

    # A backtest re-solves the MPC once per control cycle, so it runs the
    # optimiser hundreds of times. Trim the search a little; the plan barely
    # changes and the replay finishes in a reasonable time on a small machine.
    solver_cfg = replace(
        cfg,
        optimizer=replace(
            cfg.optimizer,
            population=max(96, cfg.optimizer.population // 2),
            elites=max(12, cfg.optimizer.elites // 2),
            iterations=max(6, cfg.optimizer.iterations - 4),
        ),
    )
    solver = MpcSolver(solver_cfg, params)
    progress = max(1, n_cycles // 10)
    reference = _stored_energy_reference(cfg, params, frame, n_cycles * cycle_steps)

    def mpc_policy(exog: Exogenous, state: State, previous: float, cycle: int) -> float:
        if cycle % progress == 0:
            log.info("  backtest %3d%%", round(100 * cycle / n_cycles))
        return solver.solve(exog, state, previous).offset_now

    log.info("Backtesting %d control cycles over %.1f days ...", n_cycles, n_cycles * cycle_steps * step_minutes / 1440)
    mpc_run = _run_policy(
        cfg, params, frame, mpc_policy, bias, initial, cycle_steps, n_cycles, horizon_steps, reference
    )

    constants = _constant_offset_runs(
        cfg, params, frame, bias, initial, n_cycles * cycle_steps, reference
    )

    target = mpc_run["indoor_mean"]
    matched_offset = min(constants, key=lambda v: abs(constants[v]["indoor_mean"] - target))
    matched = constants[matched_offset]
    flat = constants[min(constants, key=lambda v: abs(v))]

    saving = matched["net_cost_sek"] - mpc_run["net_cost_sek"]
    gross_saving = matched["cost_sek"] - mpc_run["cost_sek"]
    return {
        "period": {
            "start": str(frame.index[0]),
            "end": str(frame.index[n_cycles * cycle_steps - 1]),
            "days": round(n_cycles * cycle_steps * step_minutes / 1440, 2),
            "cycles": int(n_cycles),
        },
        "mpc": {k: v for k, v in mpc_run.items() if k != "series"},
        "baseline_matched": {
            "offset": matched_offset,
            **{k: v for k, v in matched.items() if k != "series"},
        },
        "baseline_flat": {k: v for k, v in flat.items() if k != "series"},
        "saving_sek": round(float(saving), 3),
        "gross_saving_sek": round(float(gross_saving), 3),
        "terminal_credit_sek": round(float(saving - gross_saving), 3),
        "saving_pct": round(float(100.0 * saving / matched["net_cost_sek"]) if matched["net_cost_sek"] else 0.0, 2),
        "energy_change_pct": round(
            float(100.0 * (mpc_run["kwh"] - matched["kwh"]) / matched["kwh"]) if matched["kwh"] else 0.0, 2
        ),
        "caveats": [
            "The plant is the identified model, so this isolates the value of the optimisation, not model error.",
            "Weather and price forecasts are perfect in this replay; expect the real saving to be lower.",
            "Compare against 'baseline_matched': it holds the same average indoor temperature.",
            "The saving is net of the heat each run leaves in the slab, so ending cold is not counted as a win.",
            "On short replays the terminal term can be a large share of the saving - use --days 7 or more.",
        ],
        "series": mpc_run["series"],
    }


def format_backtest(result: dict[str, Any]) -> str:
    period = result["period"]
    mpc = result["mpc"]
    matched = result["baseline_matched"]
    lines = [
        f"Backtest over {period['days']} days ({period['cycles']} control cycles)",
        "",
        f"{'':22}{'MPC':>12}{'constant':>12}",
        f"{'offset':22}{mpc['offset_mean']:>12.2f}{matched['offset']:>12.2f}",
        f"{'electricity (kWh)':22}{mpc['kwh']:>12.1f}{matched['kwh']:>12.1f}",
        f"{'cost (SEK)':22}{mpc['cost_sek']:>12.2f}{matched['cost_sek']:>12.2f}",
        f"{'stored heat (SEK)':22}{mpc['stored_value_sek']:>12.2f}{matched['stored_value_sek']:>12.2f}",
        f"{'net cost (SEK)':22}{mpc['net_cost_sek']:>12.2f}{matched['net_cost_sek']:>12.2f}",
        f"{'mean indoor (C)':22}{mpc['indoor_mean']:>12.2f}{matched['indoor_mean']:>12.2f}",
        f"{'min indoor (C)':22}{mpc['indoor_min']:>12.2f}{matched['indoor_min']:>12.2f}",
        f"{'max indoor (C)':22}{mpc['indoor_max']:>12.2f}{matched['indoor_max']:>12.2f}",
        f"{'Kh outside comfort':22}{mpc['kelvin_hours_outside_comfort']:>12.2f}{matched['kelvin_hours_outside_comfort']:>12.2f}",
        "",
        f"Saving: {result['saving_sek']:.2f} SEK ({result['saving_pct']:.1f} %) at equal average indoor temperature",
        f"        of which {result['gross_saving_sek']:.2f} SEK on the meter "
        f"and {result['terminal_credit_sek']:.2f} SEK in heat left in the slab",
        f"Energy: {result['energy_change_pct']:+.1f} % kWh",
        "",
        "Caveats:",
        *[f"  - {c}" for c in result["caveats"]],
    ]
    return "\n".join(lines)
