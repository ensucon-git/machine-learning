"""Learning the house from Home Assistant history.

Three things are learned, in order:

1. **The heating curve** (``fit_heating_curve``) - how the pump translates the
   outdoor temperature it believes into a supply-temperature setpoint, plus the
   time constant of its internal outdoor averaging. Only possible when a supply
   sensor exists; otherwise the configured curve is trusted.
2. **The building** (``fit_thermal``) - the 2R2C parameters, by minimising
   multi-step prediction error of the indoor temperature over many overlapping
   windows. This is the expensive part and the one that actually matters.
3. **The COP** (``fit_cop``) - a single Carnot efficiency scaling, fitted
   against measured electrical power when that sensor exists.

Everything runs locally on CPU in seconds to a couple of minutes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import Config
from .dataset import segments
from .model import heatpump as hp
from .model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp

log = logging.getLogger(__name__)


@dataclass
class WindowSet:
    """A batch of overlapping simulation windows carved out of the history."""

    exog: Exogenous
    offset: np.ndarray
    target: np.ndarray
    t_indoor0: np.ndarray
    t_filtered0: np.ndarray
    eval_mask: np.ndarray
    supply: np.ndarray | None
    dt_hours: float

    def __len__(self) -> int:
        return int(self.offset.shape[0])

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(
            exog=Exogenous(
                self.exog.t_outdoor[idx],
                self.exog.wind[idx],
                self.exog.solar_ghi[idx],
                self.exog.price[idx],
            ),
            offset=self.offset[idx],
            target=self.target[idx],
            t_indoor0=self.t_indoor0[idx],
            t_filtered0=self.t_filtered0[idx],
            eval_mask=self.eval_mask[idx],
            supply=None if self.supply is None else self.supply[idx],
            dt_hours=self.dt_hours,
        )


def _filtered_outdoor_for_segment(seg: pd.DataFrame, cfg: Config, dt_hours: float) -> np.ndarray:
    perceived = (seg["t_outdoor"] + seg["offset"]).to_numpy(dtype=float)
    return hp.filter_outdoor_series(
        perceived[None, :], cfg.heat_pump.outdoor_filter_hours, dt_hours, perceived[0]
    )[0]


def make_windows(
    frame: pd.DataFrame,
    cfg: Config,
    window_hours: float | None = None,
    burn_in_hours: float | None = None,
    stride_hours: float | None = None,
    max_windows: int | None = None,
    use_measured_supply: bool | None = None,
) -> WindowSet:
    """Slice the history into fixed-length simulation windows.

    Each window is preceded by a burn-in stretch that is simulated but not
    scored. The slab temperature is never measured, so the burn-in is what lets
    the model forget a wrong initial guess before the error is counted.
    """
    tr = cfg.training
    dt_hours = tr.resample_minutes / 60.0
    window_hours = tr.window_hours if window_hours is None else window_hours
    burn_in_hours = tr.burn_in_hours if burn_in_hours is None else burn_in_hours
    stride_hours = tr.window_stride_hours if stride_hours is None else stride_hours
    max_windows = tr.max_windows if max_windows is None else max_windows

    has_supply = "t_supply" in frame and frame["t_supply"].notna().mean() > 0.8
    if use_measured_supply is None:
        use_measured_supply = has_supply
    use_measured_supply = bool(use_measured_supply and has_supply)

    burn = max(1, int(round(burn_in_hours / dt_hours)))
    span = burn + max(1, int(round(window_hours / dt_hours)))
    stride = max(1, int(round(stride_hours / dt_hours)))

    cols_out, cols_wind, cols_sun, cols_price = [], [], [], []
    cols_offset, cols_target, cols_supply = [], [], []
    ti0, tf0 = [], []

    for seg in segments(frame, tr.resample_minutes):
        if len(seg) < span:
            continue
        seg = seg.copy()
        if "offset" not in seg:
            seg["offset"] = 0.0
        seg["offset"] = seg["offset"].fillna(0.0)
        filt = _filtered_outdoor_for_segment(seg, cfg, dt_hours)

        arr_out = seg["t_outdoor"].to_numpy(dtype=float)
        arr_wind = seg.get("wind", pd.Series(0.0, index=seg.index)).fillna(0.0).to_numpy(dtype=float)
        arr_sun = seg.get("solar_ghi", pd.Series(0.0, index=seg.index)).fillna(0.0).to_numpy(dtype=float)
        arr_price = seg.get("price", pd.Series(1.0, index=seg.index)).fillna(1.0).to_numpy(dtype=float)
        arr_off = seg["offset"].to_numpy(dtype=float)
        arr_ti = seg["t_indoor"].to_numpy(dtype=float)
        arr_sup = (
            seg["t_supply"].ffill().bfill().to_numpy(dtype=float) if use_measured_supply else None
        )

        for start in range(0, len(seg) - span + 1, stride):
            sl = slice(start, start + span)
            if not np.isfinite(arr_ti[sl]).all() or not np.isfinite(arr_out[sl]).all():
                continue
            cols_out.append(arr_out[sl])
            cols_wind.append(arr_wind[sl])
            cols_sun.append(arr_sun[sl])
            cols_price.append(arr_price[sl])
            cols_offset.append(arr_off[sl])
            cols_target.append(arr_ti[sl])
            if arr_sup is not None:
                cols_supply.append(arr_sup[sl])
            ti0.append(arr_ti[start])
            tf0.append(filt[start])

    if not cols_out:
        raise ValueError(
            "Not enough contiguous history to build training windows - "
            f"need at least {span * dt_hours:.0f} h without gaps"
        )

    mask = np.zeros(span, dtype=bool)
    mask[burn:] = True

    ws = WindowSet(
        exog=Exogenous(np.array(cols_out), np.array(cols_wind), np.array(cols_sun), np.array(cols_price)),
        offset=np.array(cols_offset),
        target=np.array(cols_target),
        t_indoor0=np.array(ti0),
        t_filtered0=np.array(tf0),
        eval_mask=np.broadcast_to(mask, (len(cols_out), span)).copy(),
        supply=np.array(cols_supply) if cols_supply else None,
        dt_hours=dt_hours,
    )

    if max_windows and len(ws) > max_windows:
        # Keep an evenly spread subsample so all seasons/weather stay represented.
        idx = np.linspace(0, len(ws) - 1, max_windows).round().astype(int)
        ws = ws.subset(np.unique(idx))
    log.info("Built %d training windows (%d steps each, %d scored)", len(ws), span, span - burn)
    return ws


# Kept as a module-level alias: the slab is unobserved, so both identification
# and the residual model start it at the value implied by the current guess.
_initial_mass_temp = steady_state_mass_temp


def predict(params: ThermalParams, cfg: Config, ws: WindowSet) -> np.ndarray:
    tm0 = _initial_mass_temp(params, ws.t_indoor0, ws.exog.t_outdoor[:, 0])
    result = simulate(
        params,
        cfg.heat_pump,
        ws.exog,
        ws.offset,
        State(ws.t_indoor0, tm0, ws.t_filtered0),
        ws.dt_hours,
        supply_temp_override=ws.supply,
    )
    return result["t_indoor"]


# Directions the data constrains only weakly are held closer to their prior.
# Without this the optimiser happily trades wind sensitivity against envelope
# loss, which produces an excellent fit with a physically wrong parameter set.
REGULARISATION_WEIGHT = {
    "Ci": 1.0,
    "Cm": 1.0,
    "Him": 1.5,
    "Hie": 1.5,
    "Hme": 6.0,
    "k_wind": 4.0,
    "A_sol": 1.0,
    "f_sol_i": 6.0,
    "Q_int": 3.0,
    "Hfloor": 1.0,
}


def _normalise(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (vec - lo) / (hi - lo)


def _denormalise(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + np.clip(x, 0.0, 1.0) * (hi - lo)


def _split(ws: WindowSet, validation_fraction: float) -> tuple[WindowSet, WindowSet]:
    """Chronological split - the validation windows are the most recent ones, so
    the score answers "would this have predicted the days I have not used?"."""
    n_val = max(1, int(round(len(ws) * validation_fraction)))
    order = np.arange(len(ws))
    return ws.subset(order[: len(ws) - n_val]), ws.subset(order[len(ws) - n_val :])


def _identifiability(jac: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Report which parameter directions the data actually constrains.

    A large condition number means several parameters trade off against each
    other: prediction can still be excellent while the individual numbers are
    meaningless. Worth knowing before quoting a U-value from this fit.
    """
    try:
        _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        return {}
    singular = np.asarray(singular, dtype=float)
    if singular.size == 0 or singular[0] <= 0:
        return {}
    # The last right-singular vector spans the direction the residuals barely
    # respond to: the parameters with the largest weight there are the ones
    # trading off against each other.
    weakest = np.abs(vt[-1])
    # A parameter is well determined when it loads on at least one strong
    # singular direction, i.e. moving it alone visibly changes the fit.
    strength = (np.abs(vt) * singular[:, None]).max(axis=0)
    return {
        "condition_number": round(float(singular[0] / max(singular[-1], 1e-12)), 1),
        "singular_values": [round(float(v / singular[0]), 5) for v in singular],
        "weakest_direction": {
            names[i]: round(float(weakest[i]), 3) for i in np.argsort(weakest)[::-1][:4]
        },
        "well_determined": [names[i] for i in range(len(names)) if strength[i] > 0.1 * singular[0]],
    }


def fit_thermal(
    frame: pd.DataFrame,
    cfg: Config,
    initial: ThermalParams | None = None,
    regularisation: float | None = None,
    max_nfev: int = 400,
    restarts: int | None = None,
) -> tuple[ThermalParams, dict[str, Any]]:
    """Identify the 2R2C parameters by multi-step prediction-error minimisation.

    Two window sets are used at once: short ones (default 12 h) that pin the
    fast air dynamics, and long ones (default 48 h) that pin the slab. Fitting
    only short windows leaves the storage capacity - the thing the whole
    price-shifting idea depends on - essentially unconstrained.
    """
    tr = cfg.training
    regularisation = tr.regularisation if regularisation is None else regularisation
    restarts = tr.restarts if restarts is None else restarts

    short = make_windows(frame, cfg)
    train_short, val_short = _split(short, tr.validation_fraction)
    try:
        long = make_windows(
            frame,
            cfg,
            window_hours=tr.long_window_hours,
            stride_hours=max(tr.window_stride_hours * 2, 6.0),
            max_windows=max(60, tr.max_windows // 4),
        )
        train_long, val_long = _split(long, tr.validation_fraction)
    except ValueError:
        log.info("Not enough contiguous history for long windows; using short windows only")
        train_long = val_long = None

    prior = (initial or ThermalParams()).clipped()
    lo, hi = ThermalParams.bounds()
    x_prior = _normalise(prior.to_vector(), lo, hi)
    names = ThermalParams.names()
    reg_weights = regularisation * np.array([REGULARISATION_WEIGHT.get(n, 1.0) for n in names])

    blocks: list[tuple[WindowSet, float]] = [(train_short, 1.0)]
    if train_long is not None:
        blocks.append((train_long, 1.0))

    def residuals(x: np.ndarray) -> np.ndarray:
        params = ThermalParams.from_vector(_denormalise(x, lo, hi))
        parts = []
        for ws, weight in blocks:
            pred = predict(params, cfg, ws)
            err = (pred - ws.target)[ws.eval_mask]
            parts.append(weight * err / np.sqrt(err.size))
        parts.append(reg_weights * (x - x_prior))
        return np.concatenate(parts)

    log.info(
        "Fitting %d parameters on %d short + %d long windows (%d restarts) ...",
        len(lo), len(train_short), len(train_long) if train_long else 0, max(1, restarts),
    )

    attempts: list[dict[str, Any]] = []
    for attempt in range(max(1, restarts)):
        x0 = x_prior.copy() if attempt == 0 else np.clip(
            x_prior + np.random.default_rng(tr.seed + attempt).normal(0.0, 0.18, size=x_prior.size), 0.02, 0.98
        )
        solution = least_squares(
            residuals,
            x0,
            bounds=(np.zeros_like(x0), np.ones_like(x0)),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            verbose=0,
        )
        candidate = ThermalParams.from_vector(_denormalise(solution.x, lo, hi))
        score = _score(candidate, cfg, val_short).get("rmse_c", np.inf)
        drift = float(np.linalg.norm(solution.x - x_prior))
        log.info(
            "  restart %d: cost %.6g, validation RMSE %.4f C, distance from prior %.3f",
            attempt, solution.cost, score, drift,
        )
        attempts.append({"params": candidate, "score": score, "drift": drift, "solution": solution})

    # Restarts exist to escape local minima, not to chase noise. The likelihood
    # has a near-flat ridge, so a hundredth of a degree of validation RMSE is
    # not evidence: among fits that are effectively tied, keep the one closest
    # to the prior, which is the physically better behaved one.
    best_score = min(a["score"] for a in attempts)
    tied = [a for a in attempts if a["score"] <= best_score * 1.05 + 1e-9]
    chosen = min(tied, key=lambda a: a["drift"])
    if len(tied) > 1:
        log.info(
            "  %d restarts within 5%% of the best validation RMSE; keeping the one nearest the prior",
            len(tied),
        )
    params = chosen["params"]
    best_solution = chosen["solution"]
    metrics: dict[str, Any] = {
        "train": _score(params, cfg, train_short),
        "validation": _score(params, cfg, val_short),
        "prior": {"train": _score(prior, cfg, train_short), "validation": _score(prior, cfg, val_short)},
        "n_windows_train": len(train_short),
        "n_windows_validation": len(val_short),
        "used_measured_supply": short.supply is not None,
        "time_constants_hours": {k: round(v, 2) for k, v in params.time_constants_hours().items()},
        "heat_loss_w_per_k": round(params.heat_loss_w_per_k(), 1),
        "restarts": max(1, restarts),
    }
    if train_long is not None:
        metrics["validation_long_horizon"] = _score(params, cfg, val_long)
    if best_solution is not None:
        metrics["optimizer"] = {
            "success": bool(best_solution.success),
            "nfev": int(best_solution.nfev),
            "cost": float(best_solution.cost),
        }
        metrics["identifiability"] = _identifiability(np.asarray(best_solution.jac), names)
    return params, metrics


def _score(params: ThermalParams, cfg: Config, ws: WindowSet) -> dict[str, float]:
    """Multi-step prediction accuracy, with a persistence baseline for context."""
    if len(ws) == 0:
        return {}
    pred = predict(params, cfg, ws)
    mask = ws.eval_mask
    err = (pred - ws.target)[mask]
    baseline = (np.broadcast_to(ws.t_indoor0[:, None], ws.target.shape) - ws.target)[mask]
    return {
        "rmse_c": round(float(np.sqrt(np.mean(err**2))), 4),
        "mae_c": round(float(np.mean(np.abs(err))), 4),
        "bias_c": round(float(np.mean(err)), 4),
        "p95_abs_c": round(float(np.percentile(np.abs(err), 95)), 4),
        "baseline_rmse_c": round(float(np.sqrt(np.mean(baseline**2))), 4),
    }


def fit_heating_curve(frame: pd.DataFrame, cfg: Config) -> dict[str, Any] | None:
    """Recover slope / offset / filter time constant of the pump's heating curve.

    Requires a supply-temperature sensor. Rows where the setpoint is clipped or
    the pump is idle are excluded, since they carry no information about slope.
    """
    if "t_supply" not in frame or frame["t_supply"].notna().sum() < 200:
        return None
    dt_hours = cfg.training.resample_minutes / 60.0
    data = frame.dropna(subset=["t_supply", "t_outdoor"]).copy()
    if "offset" not in data:
        data["offset"] = 0.0
    perceived = (data["t_outdoor"] + data["offset"].fillna(0.0)).to_numpy(dtype=float)
    supply = data["t_supply"].to_numpy(dtype=float)

    pump = cfg.heat_pump
    active = (supply > pump.supply_min + 0.5) & (supply < pump.supply_max - 0.5) & (supply > data["t_indoor"].to_numpy() + 1.0)
    if active.sum() < 100:
        return None

    best: dict[str, Any] | None = None
    for tau in np.concatenate([np.arange(0.25, 3.0, 0.25), np.arange(3.0, 12.5, 0.5)]):
        filt = hp.filter_outdoor_series(perceived[None, :], float(tau), dt_hours, perceived[0])[0]
        x = filt[active]
        y = supply[active]
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rmse < best["rmse_c"]:
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
            slope = float(-coef[1])
            best = {
                "outdoor_filter_hours": round(float(tau), 2),
                "curve_slope": round(slope, 4),
                "curve_offset": round(float(coef[0]) - slope * pump.curve_ref, 3),
                "curve_ref": pump.curve_ref,
                "rmse_c": round(rmse, 4),
                "r2": round(1.0 - ss_res / ss_tot, 4),
                "n_samples": int(active.sum()),
            }
    if best is not None and (best["curve_slope"] <= 0.02 or best["r2"] < 0.2):
        log.warning("Heating-curve fit looks unreliable (slope=%.3f r2=%.2f) - keeping configured curve",
                    best["curve_slope"], best["r2"])
        best["accepted"] = False
    elif best is not None:
        best["accepted"] = True
    return best


def fit_cop(frame: pd.DataFrame, cfg: Config, params: ThermalParams) -> dict[str, Any] | None:
    """Fit the Carnot efficiency and standby power against measured power."""
    if "power" not in frame or frame["power"].notna().sum() < 200:
        return None
    dt_hours = cfg.training.resample_minutes / 60.0
    data = frame.dropna(subset=["power", "t_outdoor", "t_indoor"]).copy()
    if len(data) < 200:
        return None
    if "offset" not in data:
        data["offset"] = 0.0

    has_supply = "t_supply" in data and data["t_supply"].notna().mean() > 0.8
    supply_override = data["t_supply"].ffill().bfill().to_numpy(dtype=float)[None, :] if has_supply else None

    exog = Exogenous(
        data["t_outdoor"].to_numpy(dtype=float)[None, :],
        data.get("wind", pd.Series(0.0, index=data.index)).fillna(0.0).to_numpy(dtype=float)[None, :],
        data.get("solar_ghi", pd.Series(0.0, index=data.index)).fillna(0.0).to_numpy(dtype=float)[None, :],
        data.get("price", pd.Series(1.0, index=data.index)).fillna(1.0).to_numpy(dtype=float)[None, :],
    )
    ti0 = float(data["t_indoor"].iloc[0])
    tm0 = float(_initial_mass_temp(params, np.array([ti0]), exog.t_outdoor[:, 0])[0])
    sim = simulate(
        params,
        cfg.heat_pump,
        exog,
        data["offset"].fillna(0.0).to_numpy(dtype=float)[None, :],
        State(ti0, tm0, float(exog.t_outdoor[0, 0] + data["offset"].fillna(0.0).iloc[0])),
        dt_hours,
        supply_temp_override=supply_override,
    )

    q = sim["q_heat"][0]
    ts = sim["t_supply"][0]
    te = exog.t_outdoor[0]
    power = data["power"].to_numpy(dtype=float)

    idle = q < 0.02 * max(float(np.max(q)), 1.0)
    standby = float(np.median(power[idle])) if idle.sum() > 20 else cfg.heat_pump.standby_power_w
    standby = float(np.clip(standby, 0.0, 300.0))

    run = q > 0.15 * max(float(np.max(q)), 1.0)
    if run.sum() < 100:
        return None
    lift = np.maximum(ts[run] - te[run], 1.0)
    defrost = 1.0 - cfg.heat_pump.defrost_penalty * np.exp(-(((te[run] + 2.0) / 4.0) ** 2))
    x = q[run] * lift / ((ts[run] + hp.KELVIN) * defrost)   # = q / (COP/eta)
    z = np.maximum(power[run] - standby, 1.0)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return None
    inv_eta = float(np.sum(x * z) / denom)
    eta = float(np.clip(1.0 / max(inv_eta, 1e-6), 0.15, 0.75))

    pred = x * inv_eta + standby
    resid = pred - power[run]
    scop = float(np.sum(q[run]) / max(np.sum(power[run]), 1.0))
    return {
        "carnot_efficiency": round(eta, 4),
        "standby_power_w": round(standby, 1),
        "power_rmse_w": round(float(np.sqrt(np.mean(resid**2))), 1),
        "power_mae_w": round(float(np.mean(np.abs(resid))), 1),
        "seasonal_cop_observed": round(scop, 3),
        "n_samples": int(run.sum()),
    }
