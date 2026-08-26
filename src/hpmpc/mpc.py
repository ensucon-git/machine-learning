"""Model predictive control: pick the offset schedule that heats the house for
the least money while keeping it comfortable.

The decision variable is a piecewise-constant offset over the horizon (default
36 h in 3 h blocks = 12 numbers). The objective is nonsmooth - the heating
curve clips, the pump has a heat-stop temperature, the comfort band is a
one-sided penalty - so the search is a cross-entropy method over batched
rollouts, optionally polished with L-BFGS-B. On a Raspberry Pi 4 a full solve
takes on the order of a second.

Only the first block is actually applied; the rest is replanned at the next
cycle. That is what makes the scheme robust to forecast error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .model import heatpump as hp
from .model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp

log = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    total: float
    energy_sek: float
    comfort: float
    hard: float
    smoothness: float
    terminal: float
    energy_kwh: float
    stored_value_sek: float

    @property
    def net_cost_sek(self) -> float:
        """Electricity bought, minus the value of the heat left in the slab.

        Comparing raw ``energy_sek`` between plans is misleading: a plan can look
        cheap simply by ending the horizon with a cold house. This nets that out.
        """
        return self.energy_sek - self.stored_value_sek

    def to_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 4),
            "energy_sek": round(self.energy_sek, 4),
            "stored_value_sek": round(self.stored_value_sek, 4),
            "net_cost_sek": round(self.net_cost_sek, 4),
            "comfort": round(self.comfort, 4),
            "hard": round(self.hard, 4),
            "smoothness": round(self.smoothness, 4),
            "terminal": round(self.terminal, 4),
            "energy_kwh": round(self.energy_kwh, 3),
        }


@dataclass
class Baseline:
    """A constant-offset reference plan the MPC result is compared against."""

    label: str
    offset: float
    cost: CostBreakdown
    mean_indoor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "offset": round(self.offset, 2),
            "kwh": round(self.cost.energy_kwh, 2),
            "cost_sek": round(self.cost.energy_sek, 2),
            "net_cost_sek": round(self.cost.net_cost_sek, 2),
            "mean_indoor": round(self.mean_indoor, 2),
        }


@dataclass
class MpcResult:
    offset_now: float
    offset_blocks: np.ndarray
    offset_schedule: np.ndarray
    trajectory: dict[str, np.ndarray]
    cost: CostBreakdown
    mean_indoor: float
    baseline_flat: Baseline
    baseline_matched: Baseline
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_saving_sek(self) -> float:
        """Saving against the constant offset that holds the *same* average
        indoor temperature. Comparing against offset 0 would flatter the result,
        because part of any saving there is simply a colder house."""
        return self.baseline_matched.cost.net_cost_sek - self.cost.net_cost_sek

    @property
    def predicted_saving_pct(self) -> float:
        base = self.baseline_matched.cost.net_cost_sek
        return 100.0 * self.predicted_saving_sek / base if base > 1e-9 else 0.0

    def summary(self) -> dict[str, Any]:
        traj = self.trajectory
        return {
            "offset_now": round(float(self.offset_now), 3),
            "offset_blocks": [round(float(v), 2) for v in self.offset_blocks],
            "predicted_indoor_min": round(float(np.min(traj["t_indoor"])), 2),
            "predicted_indoor_max": round(float(np.max(traj["t_indoor"])), 2),
            "predicted_indoor_mean": round(float(self.mean_indoor), 2),
            "predicted_indoor_end": round(float(traj["t_indoor"][-1]), 2),
            "predicted_supply_now": round(float(traj["t_supply"][0]), 1),
            "horizon_kwh": round(float(self.cost.energy_kwh), 2),
            "horizon_cost_sek": round(float(self.cost.energy_sek), 2),
            "horizon_net_cost_sek": round(float(self.cost.net_cost_sek), 2),
            "stored_energy_value_sek": round(float(self.cost.stored_value_sek), 2),
            "baseline_flat": self.baseline_flat.to_dict(),
            "baseline_matched": self.baseline_matched.to_dict(),
            "predicted_saving_sek": round(float(self.predicted_saving_sek), 2),
            "predicted_saving_pct": round(float(self.predicted_saving_pct), 1),
            "cost": self.cost.to_dict(),
            **self.diagnostics,
        }


def block_layout(cfg: Config) -> np.ndarray:
    """Number of control steps in each decision block."""
    steps = int(round(cfg.control.horizon_hours * 60 / cfg.control.step_minutes))
    per_block = max(1, int(round(cfg.control.block_hours * 60 / cfg.control.step_minutes)))
    sizes = [per_block] * (steps // per_block)
    if steps % per_block:
        sizes.append(steps % per_block)
    return np.array(sizes, dtype=int)


def expand_blocks(blocks: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """(B, n_blocks) -> (B, K)."""
    return np.repeat(np.atleast_2d(blocks), sizes, axis=1)


class MpcSolver:
    """Stateful solver: it keeps the previous solution to warm-start the next."""

    def __init__(self, cfg: Config, params: ThermalParams) -> None:
        self.cfg = cfg
        self.params = params
        self.sizes = block_layout(cfg)
        self.n_blocks = len(self.sizes)
        self.steps = int(self.sizes.sum())
        self.dt = cfg.control.step_minutes / 60.0
        self._previous: np.ndarray | None = None
        self._rng = np.random.default_rng(cfg.optimizer.seed)

    # ------------------------------------------------------------- costing

    def evaluate(
        self, blocks: np.ndarray, exog: Exogenous, state: State, previous_offset: float
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        c = self.cfg.control
        u = expand_blocks(blocks, self.sizes)
        traj = simulate(self.params, self.cfg.heat_pump, exog, u, state, self.dt)

        ti = traj["t_indoor"]
        energy_kwh = traj["p_electric"] * self.dt / 1000.0
        energy_sek = np.sum(energy_kwh * exog.price, axis=1)

        below = np.maximum(c.comfort_min - ti, 0.0)
        above = np.maximum(ti - c.comfort_max, 0.0)
        comfort = c.weight_comfort * np.sum(below**2 + above**2, axis=1) * self.dt

        hard_below = np.maximum(c.hard_min - ti, 0.0)
        hard_above = np.maximum(ti - c.hard_max, 0.0)
        hard = c.weight_hard * np.sum(hard_below**2 + hard_above**2, axis=1) * self.dt

        blocks2d = np.atleast_2d(blocks)
        prev_column = np.full((blocks2d.shape[0], 1), float(previous_offset))
        deltas = np.diff(np.concatenate([prev_column, blocks2d], axis=1), axis=1)
        smoothness = c.weight_offset_change * np.sum(deltas**2, axis=1)

        stored_value = self._stored_energy_value(traj, exog)
        terminal = c.weight_terminal * (ti[:, -1] - c.setpoint) ** 2 - stored_value

        total = energy_sek + comfort + hard + smoothness + terminal
        parts = {
            "energy_sek": energy_sek,
            "stored_value_sek": stored_value,
            "comfort": comfort,
            "hard": hard,
            "smoothness": smoothness,
            "terminal": terminal,
            "energy_kwh": np.sum(energy_kwh, axis=1),
            "total": total,
        }
        return total, parts, traj

    def _stored_energy_value(self, traj: dict[str, np.ndarray], exog: Exogenous) -> np.ndarray:
        """Value of the heat left in the building at the end of the horizon [SEK].

        Without this the optimiser has an obvious cheat: end the horizon with a
        cold slab. That looks cheap inside the horizon and is silently undone at
        the next replan, so the open-loop saving never materialises in closed
        loop. Pricing the stored energy at what it would cost to buy removes the
        incentive entirely - and correctly rewards genuinely pre-storing heat
        while electricity is cheap.
        """
        c = self.cfg.control
        tail = max(1, int(round(6.0 / self.dt)))
        t_end = float(np.mean(np.asarray(exog.t_outdoor, dtype=float)[..., -tail:]))
        price_ref = float(np.mean(np.asarray(exog.price, dtype=float)[..., -tail:]))

        tm_target = float(steady_state_mass_temp(self.params, c.setpoint, t_end))
        stored_wh = self.params.Cm * (traj["t_mass"][:, -1] - tm_target) + self.params.Ci * (
            traj["t_indoor"][:, -1] - c.setpoint
        )
        cop_ref = np.maximum(hp.cop(traj["t_supply"][:, -1], np.full_like(stored_wh, t_end), self.cfg.heat_pump), 1e-3)
        return stored_wh / 1000.0 * price_ref / cop_ref

    def _breakdown(self, parts: dict[str, np.ndarray], i: int) -> CostBreakdown:
        return CostBreakdown(
            total=float(parts["total"][i]),
            energy_sek=float(parts["energy_sek"][i]),
            comfort=float(parts["comfort"][i]),
            hard=float(parts["hard"][i]),
            smoothness=float(parts["smoothness"][i]),
            terminal=float(parts["terminal"][i]),
            energy_kwh=float(parts["energy_kwh"][i]),
            stored_value_sek=float(parts["stored_value_sek"][i]),
        )

    # -------------------------------------------------------------- search

    def _constant_grid(
        self, exog: Exogenous, state: State, previous_offset: float
    ) -> dict[str, Any]:
        """Evaluate every constant offset once, in a single batched rollout.

        Serves three purposes at once: it seeds the search with the best
        constant plan (so the MPC can never come out worse than simply leaving
        the offset alone), it provides the comparison baselines, and it costs
        about as much as one extra candidate.
        """
        c = self.cfg.control
        grid = np.unique(np.round(np.linspace(c.offset_min, c.offset_max, 41), 4))
        candidates = np.repeat(grid[:, None], self.n_blocks, axis=1)
        totals, parts, traj = self.evaluate(candidates, exog, state, previous_offset)
        return {
            "grid": grid,
            "totals": totals,
            "parts": parts,
            "mean_indoor": traj["t_indoor"].mean(axis=1),
        }

    def _seed_population(
        self, exog: Exogenous, previous_offset: float, constants: dict[str, Any]
    ) -> np.ndarray:
        """Hand-crafted starting points: do nothing, repeat last plan, the best
        constant offset, and a price-following heuristic (preheat when cheap,
        coast when expensive)."""
        c = self.cfg.control
        seeds = [np.zeros(self.n_blocks), np.full(self.n_blocks, float(previous_offset))]
        if self._previous is not None:
            shifted = np.concatenate([self._previous[1:], self._previous[-1:]])
            seeds.append(shifted)
        best_constants = constants["grid"][np.argsort(constants["totals"])[:2]]
        seeds.extend(np.full(self.n_blocks, float(v)) for v in best_constants)
        price_steps = np.asarray(exog.price, dtype=float).reshape(-1, self.steps)[0]
        block_price = np.array([seg.mean() for seg in np.split(price_steps, np.cumsum(self.sizes)[:-1])])
        spread = block_price.std()
        if spread > 1e-6:
            z = (block_price - block_price.mean()) / spread
            for gain in (0.7, 1.5, 3.0):
                seeds.append(np.clip(gain * z, c.offset_min, c.offset_max))
        return np.clip(np.array(seeds), c.offset_min, c.offset_max)

    def solve(
        self,
        exog: Exogenous,
        state: State,
        previous_offset: float = 0.0,
        baseline_offset: float | None = None,
    ) -> MpcResult:
        if len(exog) != self.steps:
            raise ValueError(f"forecast has {len(exog)} steps, solver expects {self.steps}")
        c = self.cfg.control
        o = self.cfg.optimizer

        constants = self._constant_grid(exog, state, previous_offset)
        seeds = self._seed_population(exog, previous_offset, constants)
        mean = self._previous.copy() if self._previous is not None else np.zeros(self.n_blocks)
        sigma = np.full(self.n_blocks, max((c.offset_max - c.offset_min) / 4.0, 0.5))

        best_x = seeds[0]
        best_cost = np.inf
        history = []

        for iteration in range(o.iterations):
            sampled = self._rng.normal(mean, sigma, size=(o.population, self.n_blocks))
            candidates = np.clip(np.vstack([seeds, sampled]), c.offset_min, c.offset_max)
            totals, _, _ = self.evaluate(candidates, exog, state, previous_offset)
            order = np.argsort(totals)
            elite = candidates[order[: o.elites]]
            mean = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), o.sigma_floor)
            if totals[order[0]] < best_cost:
                best_cost = float(totals[order[0]])
                best_x = candidates[order[0]].copy()
            history.append(round(best_cost, 5))
            seeds = np.vstack([seeds[:2], best_x])

        polished = False
        if o.polish:
            candidate = self._polish(best_x, exog, state, previous_offset)
            if candidate is not None:
                totals, _, _ = self.evaluate(candidate[None, :], exog, state, previous_offset)
                if float(totals[0]) < best_cost:
                    best_cost = float(totals[0])
                    best_x = candidate
                    polished = True

        best_x = np.clip(best_x, c.offset_min, c.offset_max)
        totals, parts, traj = self.evaluate(best_x[None, :], exog, state, previous_offset)
        cost = self._breakdown(parts, 0)
        mean_indoor = float(np.mean(traj["t_indoor"][0]))

        flat_offset = float(baseline_offset if baseline_offset is not None else 0.0)
        baselines = self._baselines(constants, flat_offset, mean_indoor)

        self._previous = best_x.copy()
        return MpcResult(
            offset_now=float(best_x[0]),
            offset_blocks=best_x,
            offset_schedule=expand_blocks(best_x[None, :], self.sizes)[0],
            trajectory={k: v[0] for k, v in traj.items()},
            cost=cost,
            mean_indoor=mean_indoor,
            baseline_flat=baselines[0],
            baseline_matched=baselines[1],
            diagnostics={
                "iterations": o.iterations,
                "population": o.population,
                "polished": polished,
                "cost_history": history,
                "block_hours": self.cfg.control.block_hours,
                "horizon_hours": self.cfg.control.horizon_hours,
            },
        )

    def _baselines(
        self, constants: dict[str, Any], flat_offset: float, target_mean_indoor: float
    ) -> tuple[Baseline, Baseline]:
        """Pick the reference plans out of the constant-offset grid.

        ``matched`` is the constant offset whose average indoor temperature is
        closest to the plan's, i.e. the honest "what would a fixed setting have
        cost for the same comfort" comparison.
        """
        grid = constants["grid"]
        parts = constants["parts"]
        means = constants["mean_indoor"]
        flat_idx = int(np.argmin(np.abs(grid - flat_offset)))
        matched_idx = int(np.argmin(np.abs(means - target_mean_indoor)))
        return (
            Baseline("flat", float(grid[flat_idx]), self._breakdown(parts, flat_idx), float(means[flat_idx])),
            Baseline(
                "matched_comfort",
                float(grid[matched_idx]),
                self._breakdown(parts, matched_idx),
                float(means[matched_idx]),
            ),
        )

    def _polish(
        self, x0: np.ndarray, exog: Exogenous, state: State, previous_offset: float
    ) -> np.ndarray | None:
        """Local refinement with L-BFGS-B.

        The finite-difference gradient is computed as one batched rollout of
        ``n_blocks + 1`` perturbed schedules instead of that many sequential
        solver calls - roughly an order of magnitude faster, which is what makes
        the polish affordable on a Raspberry Pi.
        """
        from scipy.optimize import minimize

        c = self.cfg.control
        eps = 0.05
        perturbations = np.vstack([np.zeros(self.n_blocks), np.eye(self.n_blocks) * eps])

        def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            batch = np.clip(x[None, :] + perturbations, c.offset_min, c.offset_max)
            totals, _, _ = self.evaluate(batch, exog, state, previous_offset)
            base = float(totals[0])
            grad = (totals[1:] - base) / eps
            return base, grad

        try:
            result = minimize(
                value_and_grad,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(c.offset_min, c.offset_max)] * self.n_blocks,
                options={"maxiter": 60},
            )
        except Exception as exc:  # pragma: no cover - numerical safety net
            log.warning("Polish step failed: %s", exc)
            return None
        return np.clip(result.x, c.offset_min, c.offset_max)
