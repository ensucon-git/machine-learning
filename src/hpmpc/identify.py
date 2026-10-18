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
from .model import build_pump
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
                stove=self.exog.stove[idx],
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
    min_heating_fraction: float | None = None,
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
    cols_offset, cols_target, cols_supply, cols_stove = [], [], [], []
    ti0, tf0 = [], []
    min_heating = tr.min_heating_fraction if min_heating_fraction is None else min_heating_fraction
    skipped_no_heat = 0

    for seg in segments(frame, tr.resample_minutes):
        if len(seg) < span:
            continue
        seg = seg.copy()
        if "offset" not in seg:
            seg["offset"] = 0.0
        seg["offset"] = seg["offset"].fillna(0.0)
        filt = _filtered_outdoor_for_segment(seg, cfg, dt_hours)
        # The offset is part of the perceived temperature, so an excitation
        # block that drags the pump below its heat-stop point counts as heating
        # even on a mild day - which is exactly the window worth keeping.
        heat_ok = hp.heating_enabled(filt, cfg.heat_pump)

        arr_out = seg["t_outdoor"].to_numpy(dtype=float)
        arr_wind = seg.get("wind", pd.Series(0.0, index=seg.index)).fillna(0.0).to_numpy(dtype=float)
        arr_sun = seg.get("solar_ghi", pd.Series(0.0, index=seg.index)).fillna(0.0).to_numpy(dtype=float)
        arr_price = seg.get("price", pd.Series(1.0, index=seg.index)).fillna(1.0).to_numpy(dtype=float)
        arr_off = seg["offset"].to_numpy(dtype=float)
        arr_stove = seg.get("stove", pd.Series(0.0, index=seg.index)).fillna(0.0).to_numpy(dtype=float)
        arr_ti = seg["t_indoor"].to_numpy(dtype=float)
        arr_sup = (
            seg["t_supply"].ffill().bfill().to_numpy(dtype=float) if use_measured_supply else None
        )

        for start in range(0, len(seg) - span + 1, stride):
            sl = slice(start, start + span)
            if not np.isfinite(arr_ti[sl]).all() or not np.isfinite(arr_out[sl]).all():
                continue
            # Only the scored part matters; the burn-in exists to settle the
            # slab estimate, not to be predicted.
            if min_heating > 0.0 and float(np.mean(heat_ok[sl][burn:])) < min_heating:
                skipped_no_heat += 1
                continue
            cols_out.append(arr_out[sl])
            cols_wind.append(arr_wind[sl])
            cols_sun.append(arr_sun[sl])
            cols_price.append(arr_price[sl])
            cols_offset.append(arr_off[sl])
            cols_target.append(arr_ti[sl])
            cols_stove.append(arr_stove[sl])
            if arr_sup is not None:
                cols_supply.append(arr_sup[sl])
            ti0.append(arr_ti[start])
            tf0.append(filt[start])

    if not cols_out:
        if skipped_no_heat:
            raise ValueError(
                f"No training windows where the pump could heat ({skipped_no_heat} skipped): this "
                f"history is all above heat_stop_temp={cfg.heat_pump.heat_stop_temp:.0f} C, so the "
                "offset did nothing and there is no gain to learn. Train on data from the heating "
                "season, or set training.min_heating_fraction: 0 to fit on it anyway."
            )
        raise ValueError(
            "Not enough contiguous history to build training windows - "
            f"need at least {span * dt_hours:.0f} h without gaps"
        )

    mask = np.zeros(span, dtype=bool)
    mask[burn:] = True

    ws = WindowSet(
        exog=Exogenous(
            np.array(cols_out), np.array(cols_wind), np.array(cols_sun), np.array(cols_price),
            stove=np.array(cols_stove),
        ),
        offset=np.array(cols_offset),
        target=np.array(cols_target),
        t_indoor0=np.array(ti0),
        t_filtered0=np.array(tf0),
        eval_mask=np.broadcast_to(mask, (len(cols_out), span)).copy(),
        supply=np.array(cols_supply) if cols_supply else None,
        dt_hours=dt_hours,
    )

    if max_windows and len(ws) > max_windows:
        # Keep an evenly spread subsample so the whole range of weather that
        # survived the heating filter stays represented.
        idx = np.linspace(0, len(ws) - 1, max_windows).round().astype(int)
        ws = ws.subset(np.unique(idx))
    if skipped_no_heat:
        log.info(
            "Skipped %d windows where the pump could not heat (below %.0f%% of the scored hours) - "
            "the offset has no effect there",
            skipped_no_heat, 100.0 * min_heating,
        )
    log.info("Built %d training windows (%d steps each, %d scored)", len(ws), span, span - burn)
    return ws


# Kept as a module-level alias: the slab is unobserved, so both identification
# and the residual model start it at the value implied by the current guess.
_initial_mass_temp = steady_state_mass_temp


def predict(params: ThermalParams, cfg: Config, ws: WindowSet) -> np.ndarray:
    tm0 = _initial_mass_temp(params, ws.t_indoor0, ws.exog.t_outdoor[:, 0])
    result = simulate(
        params,
        build_pump(cfg),
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
    # Pulled towards zero as hard as Q_int. A stove term that is not earning
    # its keep must collapse rather than drift up and absorb whatever else the
    # fit cannot explain - it is the one input here that nothing else bounds.
    "k_stove": 3.0,
}


def _normalise(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (vec - lo) / (hi - lo)


def _denormalise(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + np.clip(x, 0.0, 1.0) * (hi - lo)


# A window whose offset moves more than this carries information about the
# offset gain; one where it sits still does not.
MOVING_OFFSET_STD_K = 0.25


def _split(ws: WindowSet, validation_fraction: float) -> tuple[WindowSet, WindowSet]:
    """Chronological split, done separately for windows where the offset moves.

    The validation windows are the most recent ones, so the score answers
    "would this have predicted the days I have not used?". But a plain
    chronological cut puts the newest data in validation, and on a fresh
    install the newest data is precisely the excitation week - the only
    stretch where the offset moved for reasons the fit cannot confuse with the
    weather. On the real house it was held out of training for the first three
    fits.

    So the windows where the offset moves and the ones where it does not are
    each split chronologically on their own, and recombined. Most of the
    excitation trains the model; its last quarter still tests it.

    Measured, not assumed: on the synthetic house (31 quiet days, then a week
    of excitation, newest) both splits recover UA to the same -7 %, because
    with the heating curve known the weather alone moves the pump enough. The
    gain this buys is for houses where that is not true, and an honest
    validation score - it now has to predict windows where the offset moved.
    It was not the cause of the poor fits on the real house.
    """
    order = np.arange(len(ws))
    moving = ws.offset.std(axis=1) > MOVING_OFFSET_STD_K
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for group in (order[moving], order[~moving]):
        if len(group) < 2:
            train_idx.append(group)
            continue
        n_val = min(len(group) - 1, max(1, int(round(len(group) * validation_fraction))))
        train_idx.append(group[: len(group) - n_val])
        val_idx.append(group[len(group) - n_val:])
    train = np.sort(np.concatenate(train_idx)) if train_idx else order[:0]
    val = np.sort(np.concatenate(val_idx)) if val_idx else order[:0]
    if len(val) == 0 and len(ws) > 1:
        train, val = order[:-1], order[-1:]
    return ws.subset(train), ws.subset(val)


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


def _window_sets(frame: pd.DataFrame, cfg: Config):
    """Short and long windows, each split into training and validation."""
    tr = cfg.training
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
    return short, train_short, val_short, train_long, val_long


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

    short, train_short, val_short, train_long, val_long = _window_sets(frame, cfg)

    # Bounds and the regularisation's normalisation follow the starting
    # point's size. The temperatures cannot tell sizes apart, so fitting a
    # house four times larger inside a box four times larger is the same
    # problem - and the meter, not this fit, decides which size is true.
    start = initial or ThermalParams()
    size = start.size()
    prior = start.clipped(size)
    lo, hi = ThermalParams.bounds(size)
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
            "  restart %d: objective %.6g, validation RMSE %.4f C, distance from prior %.3f",
            attempt, solution.cost, score, drift,
        )
        attempts.append(
            {"params": candidate, "score": score, "drift": drift, "cost": float(solution.cost), "solution": solution}
        )

    # Selection is on the regularised training objective, i.e. the thing every
    # restart is actually minimising. Selecting on validation RMSE instead
    # sounds safer but is not: on a near-flat likelihood ridge a hundredth of a
    # degree is noise, and preferring it lets a restart win precisely by
    # escaping the regularisation that keeps the parameters physical. Genuine
    # ties are broken toward the prior.
    best_cost = min(a["cost"] for a in attempts)
    tied = [a for a in attempts if a["cost"] <= best_cost * 1.02 + 1e-12]
    chosen = min(tied, key=lambda a: a["drift"])
    if len(attempts) > 1:
        log.info(
            "  selected restart with objective %.6g (validation RMSE %.4f C)", chosen["cost"], chosen["score"]
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
        **_physical_summary(params),
        "restarts": max(1, restarts),
        "restart_scores": [
            {"objective": round(a["cost"], 8), "validation_rmse_c": a["score"], "drift": round(a["drift"], 4)}
            for a in attempts
        ],
    }
    metrics["stove"]["share_of_hours_burning_pct"] = round(100.0 * float(np.mean(short.exog.stove > 0.0)), 1)
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


def _physical_summary(params: ThermalParams) -> dict[str, Any]:
    return {
        "time_constants_hours": {k: round(v, 2) for k, v in params.time_constants_hours().items()},
        "heat_loss_w_per_k": round(params.heat_loss_w_per_k(), 1),
        # What the fit concluded about the weather, in units someone can argue
        # with: an effective window area rather than a bare coefficient.
        "weather_sensitivity": {
            "solar_aperture_m2": round(params.A_sol, 2),
            "solar_to_air_pct": round(100.0 * params.f_sol_i),
            "wind_pct_per_m_s": round(100.0 * params.k_wind, 1),
            "ua_at_5_m_s": round(params.heat_loss_w_per_k(wind=5.0), 1),
        },
        # Same idea for the stove: kilowatts at a fire that lifts its sensor
        # 4 K above the house, which is a number you can hold against a log
        # basket. The bare W/K gain says nothing to anyone.
        "stove": {
            "w_per_k_excess": round(params.k_stove, 1),
            "kw_at_4k_excess": round(params.k_stove * 4.0 / 1000.0, 2),
        },
    }


# The meter may only set the building's scale when it actually determines it.
# A wider uncertainty than this is the split guessing, and a guess must not
# move the heat loss the optimiser trades comfort against price with.
MAX_SCALE_UNCERTAINTY = 0.25
# Wide on purpose: this is the one direction a temperature-only fit leaves
# free, so it is where the fit can be furthest off - on the real house the
# first fits landed at a third of what the meter implied.
BUILDING_FACTOR_BOUNDS = (0.2, 8.0)
# Applying the scale must not make the temperature fit worse. Below capacity it
# cannot change at all; past it, a scale that does is one the temperatures and
# the meter disagree about.
MAX_RESCALE_RMSE_GROWTH = 1.05
# A passive house to a large draughty villa. Beyond this the meter is seeing
# something other than the house.
PLAUSIBLE_HEAT_LOSS_W_PER_K = (20.0, 2000.0)


def calibrate_building_scale(frame: pd.DataFrame, cfg: Config, params: ThermalParams) -> dict[str, Any] | None:
    """How many times larger the house's heat flows are than the fit made them.

    Indoor temperatures fix the building model's shape but not its size: scale
    every conductance, capacitance and gain by the same factor and every
    trajectory is unchanged until the pump runs out of capacity, which a
    well-sized pump almost never does. The fit's heat loss in watts is
    therefore set by its prior, not by the house. The meter is the anchor:
    with the COP taken from the performance map, measured pump electricity
    times COP is the heat actually delivered, and its ratio to the heat the
    model delivered is the factor. Returns None when there is nothing to
    calibrate against.
    """
    if cfg.power.source == "none":
        return None
    context = _simulate_period(frame, cfg, params)
    if context is None:
        return None
    has_meter = "power" in frame and frame["power"].notna().sum() >= 200
    extra: dict[str, Any] = {}
    if has_meter and cfg.power.source in {"auto", "heatpump_meter"}:
        found = _factor_from_meter(context, cfg)
        if found is None:
            return None
        factor, uncertainty = found
        method = "dedicated heat pump meter"
    elif cfg.power.source == "heatpump_meter":
        log.warning("power.source is 'heatpump_meter' but no heat pump power sensor has usable data")
        return None
    else:
        from .disaggregate import disaggregate

        daily = daily_energy_balance(context, cfg)
        if daily is None:
            return None
        factor, uncertainty = daily["factor"], daily["uncertainty"]
        method = f"whole-house meter, daily energy over {daily['days']} days"
        extra = {"daily_r2": daily["r2"], "daily_base_kwh": daily["base_kwh_per_day"]}
        # The 15-minute split is no longer what sets the scale, but its car
        # charger estimate is still the best check that the phases are right.
        split = disaggregate(
            context["data"], cfg, context["q_compressor"], context["q_backup"], context["cop_base"],
            factor_bounds=BUILDING_FACTOR_BOUNDS,
        )
        if split is not None:
            extra["car_charger_kw"] = round(split.ev_power_w / 1000.0, 2)

    lo, hi = BUILDING_FACTOR_BOUNDS
    reasons = []
    if extra.get("daily_r2") is not None and extra["daily_r2"] < MIN_DAILY_R2:
        reasons.append(
            f"the model's daily heat explains only {100 * extra['daily_r2']:.0f}% of the day-to-day "
            f"variation in metered energy (need {100 * MIN_DAILY_R2:.0f}%)"
        )
    if uncertainty is None or not np.isfinite(uncertainty) or uncertainty > MAX_SCALE_UNCERTAINTY:
        shown = "unknown" if uncertainty is None or not np.isfinite(uncertainty) else f"+/-{100 * uncertainty:.0f}%"
        reasons.append(f"the meter determines it only to {shown} (need +/-{100 * MAX_SCALE_UNCERTAINTY:.0f}%)")
    if factor <= lo * 1.02 or factor >= hi / 1.02:
        reasons.append(f"the factor {factor:.2f} is at the limit of {lo}-{hi}, so the true value is unknown")
    return {
        "factor": round(float(factor), 4),
        "uncertainty": None if uncertainty is None or not np.isfinite(uncertainty) else round(float(uncertainty), 4),
        "method": method,
        "accepted": not reasons,
        "reason": "; ".join(reasons) or None,
        "heat_loss_before_w_per_k": round(params.heat_loss_w_per_k(), 1),
        "heat_loss_after_w_per_k": round(params.scaled(factor).heat_loss_w_per_k(), 1),
        **extra,
    }


# Below this the model's daily heat does not track what the meter sees, and a
# ratio between the two means nothing however tight its standard error.
MIN_DAILY_R2 = 0.5
MIN_DAILY_COVERAGE = 0.6
MIN_DAYS = 14


def daily_energy_balance(context: dict[str, Any], cfg: Config) -> dict[str, Any] | None:
    """Metered energy per day against the model's pump energy per day.

    The scale is a question about energy, not timing, so it is answered on
    days rather than quarter hours. At 15 minutes the pump's own on/off
    cycling, hot water and defrost do not line up with a smooth model, the
    regression's R2 goes negative, and its standard error still looks tight
    because it assumes the model is right. Summed over a day all of that
    averages out: hot water and the household become an intercept, and the
    slope against the model's daily electricity is the factor - an energy
    signature, with the model supplying the COP for each day's weather.
    """
    from .disaggregate import ev_mask, house_power

    data = context["data"]
    total, balanced = house_power(data, cfg)
    if total is None:
        return None
    measured = balanced if cfg.power.target == "balanced" and balanced is not None else total
    model = context["q_compressor"] / context["cop_base"]
    target = measured.to_numpy(dtype=float) - context["q_backup"]
    usable = np.isfinite(target) & np.isfinite(model) & ~ev_mask(data, cfg)
    days = data.index.tz_convert(cfg.site.timezone).normalize()
    frame = pd.DataFrame(
        {
            "model_w": np.where(usable, model, np.nan),
            "measured_w": np.where(usable, target, np.nan),
            "t_outdoor": data["t_outdoor"].to_numpy(dtype=float),
            "usable": usable.astype(float),
        },
        index=data.index,
    )
    per_day = frame.groupby(days).mean()
    per_day = per_day[per_day["usable"] >= MIN_DAILY_COVERAGE]
    if len(per_day) < MIN_DAYS:
        log.warning("Only %d days with enough metered data to calibrate the scale (need %d)",
                    len(per_day), MIN_DAYS)
        return None
    x = per_day["model_w"].to_numpy()
    y = per_day["measured_w"].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ solution
    dof = max(len(y) - 2, 1)
    try:
        covariance = np.linalg.inv(design.T @ design) * float(residual @ residual) / dof
        standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        standard_error = float("nan")
    factor = float(solution[1])
    spread = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(residual @ residual) / spread if spread > 0 else float("nan")
    table = [
        {
            "day": str(day.date()),
            "t_outdoor": round(float(row.t_outdoor), 1),
            "measured_kwh": round(float(row.measured_w) * 24 / 1000.0, 1),
            "model_kwh": round(float(row.model_w) * 24 / 1000.0, 1),
        }
        for day, row in per_day.iterrows()
    ]
    return {
        "factor": factor,
        "uncertainty": standard_error / abs(factor) if factor else float("nan"),
        "r2": round(r2, 3),
        "days": len(per_day),
        "base_kwh_per_day": round(float(solution[0]) * 24 / 1000.0, 1),
        "table": table,
    }


def _factor_from_meter(context: dict[str, Any], cfg: Config) -> tuple[float, float] | None:
    """Least-squares ratio of metered to modelled compressor power."""
    data = context["data"]
    q = context["q_heat"]
    q_backup = context["q_backup"]
    q_compressor = context["q_compressor"]
    power = data["power"].to_numpy(dtype=float)
    finite = np.isfinite(power)
    idle = finite & (q < 0.02 * max(float(np.max(q)), 1.0))
    standby = float(np.median(power[idle])) if idle.sum() > 20 else cfg.heat_pump.standby_power_w
    standby = float(np.clip(standby, 0.0, 300.0))
    run = finite & (q_compressor > 0.15 * max(float(np.max(q_compressor)), 1.0)) & (q_backup <= 1.0)
    if run.sum() < 100:
        log.warning("Too few compressor-only samples to calibrate the building against the meter (%d)",
                    int(run.sum()))
        return None
    x = q_compressor[run] / context["cop_base"][run]
    z = power[run] - standby
    denom = float(np.sum(x * x))
    if denom <= 0:
        return None
    factor = float(np.sum(x * z)) / denom
    residual = z - factor * x
    standard_error = float(np.sqrt(np.sum(residual**2) / max(run.sum() - 1, 1) / denom))
    return factor, standard_error / max(abs(factor), 1e-9)


def rescale_thermal(
    params: ThermalParams, metrics: dict[str, Any], frame: pd.DataFrame, cfg: Config, factor: float
) -> tuple[ThermalParams, dict[str, Any]] | str:
    """Apply a building scale and re-score it, or say why it cannot be applied."""
    scaled = params.scaled(factor)
    low, high = PLAUSIBLE_HEAT_LOSS_W_PER_K
    if not low <= scaled.heat_loss_w_per_k() <= high:
        return (f"scaling by {factor:.2f} gives a heat loss of {scaled.heat_loss_w_per_k():.0f} W/K, "
                f"outside the physical bounds {low:.0f}-{high:.0f} for a dwelling")
    _, train_short, val_short, _, val_long = _window_sets(frame, cfg)
    validation = _score(scaled, cfg, val_short)
    before = metrics.get("validation", {}).get("rmse_c")
    after = validation.get("rmse_c")
    if before is not None and after is not None and after > before * MAX_RESCALE_RMSE_GROWTH + 0.005:
        return (f"scaling by {factor:.2f} worsens the temperature fit ({before:.3f} -> {after:.3f} C): "
                "the meter and the temperatures disagree about where the pump runs out of capacity")
    updated = dict(metrics)
    updated.update(_physical_summary(scaled))
    updated["stove"] = {**metrics.get("stove", {}), **updated["stove"]}
    updated["train"] = _score(scaled, cfg, train_short)
    updated["validation"] = validation
    if val_long is not None:
        updated["validation_long_horizon"] = _score(scaled, cfg, val_long)
    return scaled, updated


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


def _simulate_period(frame: pd.DataFrame, cfg: Config, params: ThermalParams) -> dict[str, Any] | None:
    """Run the identified model through the whole dataset once.

    Everything the efficiency fit needs - how much heat the compressor
    delivered, how much the backup heater made up, and the COP the map says
    applied - comes from this single pass.
    """
    dt_hours = cfg.training.resample_minutes / 60.0
    data = frame.dropna(subset=["t_outdoor", "t_indoor"]).copy()
    if len(data) < 200:
        return None
    if "offset" not in data:
        data["offset"] = 0.0

    pump = build_pump(cfg)
    has_supply = "t_supply" in data and data["t_supply"].notna().mean() > 0.8
    supply_override = data["t_supply"].ffill().bfill().to_numpy(dtype=float)[None, :] if has_supply else None

    exog = Exogenous(
        data["t_outdoor"].to_numpy(dtype=float)[None, :],
        data.get("wind", pd.Series(0.0, index=data.index)).fillna(0.0).to_numpy(dtype=float)[None, :],
        data.get("solar_ghi", pd.Series(0.0, index=data.index)).fillna(0.0).to_numpy(dtype=float)[None, :],
        data.get("price", pd.Series(1.0, index=data.index)).fillna(1.0).to_numpy(dtype=float)[None, :],
        humidity=(
            data["humidity"].to_numpy(dtype=float)[None, :]
            if "humidity" in data
            else np.full((1, len(data)), np.nan)
        ),
    )
    offsets = data["offset"].fillna(0.0).to_numpy(dtype=float)[None, :]
    ti0 = float(data["t_indoor"].iloc[0])
    tm0 = float(_initial_mass_temp(params, np.array([ti0]), exog.t_outdoor[:, 0])[0])
    sim = simulate(
        params, pump, exog, offsets,
        State(ti0, tm0, float(exog.t_outdoor[0, 0] + offsets[0, 0])),
        dt_hours, supply_temp_override=supply_override,
    )
    q_backup = sim["q_backup"][0]
    return {
        "data": data,
        "pump": pump,
        "dt_hours": dt_hours,
        "q_heat": sim["q_heat"][0],
        "q_backup": q_backup,
        "q_compressor": np.maximum(sim["q_heat"][0] - q_backup, 0.0),
        "t_supply": sim["t_supply"][0],
        "t_outdoor": exog.t_outdoor[0],
        "humidity": exog.humidity[0],
        "cop_base": np.maximum(pump.cop(sim["t_supply"][0], exog.t_outdoor[0], exog.humidity[0]), 1e-6),
    }


def fit_cop(frame: pd.DataFrame, cfg: Config, params: ThermalParams) -> dict[str, Any] | None:
    """Calibrate the pump's efficiency against measured electricity.

    The shipped performance map sets the *shape* of COP across the operating
    envelope - how it falls with lift, where defrost bites, when the compressor
    runs out of capacity. This fits the *level*: a single multiplicative scale.
    One well-determined number beats ten poorly-determined ones, and it is the
    number that decides whether load shifting is worth doing at all.

    Two ways to get there. With a meter on the pump it is a direct regression.
    Without one, the pump is separated out of a whole-house measurement - see
    :mod:`hpmpc.disaggregate` - which is less precise but is what most houses
    actually have.
    """
    context = _simulate_period(frame, cfg, params)
    if context is None:
        return None

    has_meter = "power" in frame and frame["power"].notna().sum() >= 200
    source = cfg.power.source
    if source == "none":
        return None
    if has_meter and source in {"auto", "heatpump_meter"}:
        return _fit_cop_from_meter(context, cfg)
    if source == "heatpump_meter":
        log.warning("power.source is 'heatpump_meter' but no heat pump power sensor has usable data")
        return None
    return _fit_cop_from_house(context, cfg)


def _fit_cop_from_meter(context: dict[str, Any], cfg: Config) -> dict[str, Any] | None:
    """Direct fit against a dedicated heat pump meter."""
    data = context["data"]
    q = context["q_heat"]
    q_backup = context["q_backup"]
    q_compressor = context["q_compressor"]
    cop_base = context["cop_base"]
    te = context["t_outdoor"]
    power = data["power"].to_numpy(dtype=float)
    dt_hours = context["dt_hours"]

    finite = np.isfinite(power)
    idle = finite & (q < 0.02 * max(float(np.max(q)), 1.0))
    standby = float(np.median(power[idle])) if idle.sum() > 20 else cfg.heat_pump.standby_power_w
    standby = float(np.clip(standby, 0.0, 300.0))

    run = finite & (q_compressor > 0.15 * max(float(np.max(q_compressor)), 1.0)) & (q_backup <= 1.0)
    if run.sum() < 100:
        log.warning("Too few compressor-only samples to calibrate efficiency (%d)", int(run.sum()))
        return None

    x = q_compressor[run] / cop_base[run]
    z = np.maximum(power[run] - standby, 1.0)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return None
    scale = float(np.clip(denom / float(np.sum(x * z)), 0.3, 3.0))

    predicted = q_compressor / (cop_base * scale) + q_backup + standby
    resid = predicted[run] - power[run]
    delivered = float(np.sum(q) * dt_hours / 1000.0)
    consumed = float(np.sum(power[finite]) * dt_hours / 1000.0)

    return _cop_result(
        cfg, context, scale, standby,
        power_rmse=float(np.sqrt(np.mean(resid**2))),
        power_mae=float(np.mean(np.abs(resid))),
        scop=delivered / max(consumed, 1e-6),
        n_samples=int(run.sum()),
        residual_by_ambient=_residual_by_ambient(te[run], resid),
        method="dedicated heat pump meter",
    )


def _fit_cop_from_house(context: dict[str, Any], cfg: Config) -> dict[str, Any] | None:
    """Fit against a whole-house meter, separating the pump from everything else."""
    from .disaggregate import disaggregate, quality_warnings

    result = disaggregate(
        context["data"], cfg, context["q_compressor"], context["q_backup"], context["cop_base"]
    )
    if result is None:
        return None

    dt_hours = context["dt_hours"]
    delivered = float(np.sum(context["q_heat"]) * dt_hours / 1000.0)
    consumed = float(np.sum(result.heatpump_power_w.to_numpy()) * dt_hours / 1000.0)

    payload = _cop_result(
        cfg, context, result.efficiency_scale, cfg.heat_pump.standby_power_w,
        power_rmse=result.metrics.get("residual_rmse_w") or float("nan"),
        power_mae=float("nan"),
        scop=delivered / max(consumed, 1e-6),
        n_samples=result.used_samples,
        residual_by_ambient={},
        method=f"whole-house meter, {result.target}",
    )
    payload["disaggregation"] = result.to_dict()
    warnings = quality_warnings(result)
    if warnings:
        payload["warnings"] = warnings
        for warning in warnings:
            log.warning("Power disaggregation: %s", warning)
    return payload


def _cop_result(
    cfg: Config,
    context: dict[str, Any],
    scale: float,
    standby: float,
    power_rmse: float,
    power_mae: float,
    scop: float,
    n_samples: int,
    residual_by_ambient: dict[str, float],
    method: str,
) -> dict[str, Any]:
    dt_hours = context["dt_hours"]
    result: dict[str, Any] = {
        "method": method,
        "efficiency_scale": round(scale, 4),
        "standby_power_w": round(standby, 1),
        "power_rmse_w": None if np.isnan(power_rmse) else round(power_rmse, 1),
        "power_mae_w": None if np.isnan(power_mae) else round(power_mae, 1),
        "seasonal_cop_observed": round(scop, 3),
        "n_samples": n_samples,
        "backup_heater_hours": round(float(np.sum(context["q_backup"] > 1.0) * dt_hours), 2),
        "residual_by_ambient": residual_by_ambient,
        "pump_model": context["pump"].describe(),
    }
    if context["pump"].performance is None:
        result["carnot_efficiency"] = round(float(np.clip(cfg.heat_pump.carnot_efficiency * scale, 0.15, 0.75)), 4)
        result.pop("efficiency_scale")
    return result


def _residual_by_ambient(t_ambient: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    """Mean power error per outdoor-temperature bin.

    A wrong efficiency *level* shows up as a uniform offset here; a wrong map
    *shape* shows up as a trend across the bins, which is the signal to replace
    the shipped table with the manufacturer's own.
    """
    edges = np.array([-30.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 30.0])
    out: dict[str, float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (t_ambient >= lo) & (t_ambient < hi)
        if mask.sum() >= 20:
            out[f"{lo:.0f}..{hi:.0f}C"] = round(float(np.mean(residual[mask])), 1)
    return out
