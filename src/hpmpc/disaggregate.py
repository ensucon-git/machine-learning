"""Estimating the heat pump's electricity from a whole-house meter.

Without a dedicated meter on the pump there is no direct measurement to
calibrate its efficiency against - only the whole house, which also contains an
11 kW car charger, a base load with its own daily rhythm, and every appliance
in the building. Separating them is what this module does.

The trick is that nothing here is a blind source-separation problem. Three
strong structural facts do most of the work:

**1. The car charger announces itself.** A binary sensor says when it is
charging. Those samples are simply dropped from the efficiency fit - there is
plenty of data left - so the charger's exact power never has to be known. It is
estimated afterwards anyway, purely as a sanity check: if the fit says 3 kW for
an 11 kW charger, something else is wrong too.

**2. Three-phase loads stand out.** A 16 kW heat pump and an 11 kW charger draw
roughly balanced current across L1/L2/L3; almost every household load is
single-phase and therefore unbalanced. ``3 * min(L1, L2, L3)`` is the part of
the load that is provably balanced, and with the charger excluded that is
mostly the heat pump. Fitting against it instead of the raw total removes most
of the household noise before the regression even starts.

**3. The heat pump's power is predicted by physics, not learned.** The thermal
model already says how much heat is being delivered, and the performance map
says at what COP. Only the *level* is unknown - one scalar. So this is not
"find the heat pump in the data", it is "scale a known shape to fit", which is
a far better posed question.

What remains is a linear regression::

    P_measured = c · (Q_compressor / COP_map) + Q_backup + base(hour, weekend)

solved for ``c = 1/efficiency_scale`` and a smooth daily base-load profile, with
an asymmetric robust loss: an oven switching on is unexplained load the fit
should ignore, whereas the model claiming power that was never drawn is a real
error. The loss therefore tracks the lower envelope of the measurements.

The honest caveat, stated in the metrics rather than buried: outdoor
temperature has a daily cycle and so does base load, so the two can trade off.
Running ``hpmpc excite`` matters here for a second reason - it moves the pump's
power around on a schedule that has nothing to do with the clock, which is
exactly what breaks that degeneracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .config import Config

log = logging.getLogger(__name__)

MIN_COP = 1e-6


@dataclass
class Disaggregation:
    """Result of splitting whole-house power into its parts."""

    efficiency_scale: float
    base_load_w: pd.Series
    heatpump_power_w: pd.Series
    ev_power_w: float
    used_samples: int
    span_hours: float
    target: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "efficiency_scale": round(self.efficiency_scale, 4),
            "ev_power_w": round(self.ev_power_w, 1),
            "used_samples": self.used_samples,
            "span_hours": round(self.span_hours, 1),
            "target": self.target,
            **self.metrics,
        }


def house_power(frame: pd.DataFrame, cfg: Config) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (total, provably-balanced) house power in watts.

    ``3 * min(phases)`` is the part of the load that must be balanced across all
    three phases. Single-phase loads spread over the phases do inflate it a
    little, which the base-load term absorbs.
    """
    phases = [c for c in ("house_l1", "house_l2", "house_l3") if c in frame]
    if len(phases) == 3:
        values = frame[phases].astype(float)
        if values.notna().all(axis=1).mean() < 0.5:
            return _total_only(frame)
        total = values.sum(axis=1)
        balanced = 3.0 * values.min(axis=1)
        return total, balanced
    return _total_only(frame)


def _total_only(frame: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if "house_power" in frame and frame["house_power"].notna().mean() > 0.5:
        return frame["house_power"].astype(float), None
    return None, None


def ev_mask(frame: pd.DataFrame, cfg: Config) -> np.ndarray:
    """True where the car charger is (or was just) drawing power.

    Widened by a guard band: the binary sensor flips at the start and end of a
    session, but the current ramps, and a half-loaded sample is worse than no
    sample.
    """
    n = len(frame)
    if "ev_charging" not in frame:
        return np.zeros(n, dtype=bool)
    charging = frame["ev_charging"].fillna(0.0).to_numpy(dtype=float) > 0.5
    step_minutes = _step_minutes(frame, cfg)
    guard = max(0, int(round(cfg.power.ev_guard_minutes / max(step_minutes, 1e-9))))
    if guard == 0:
        return charging
    widened = charging.copy()
    for shift in range(1, guard + 1):
        widened[shift:] |= charging[:-shift]
        widened[:-shift] |= charging[shift:]
    return widened


def _step_minutes(frame: pd.DataFrame, cfg: Config) -> float:
    if len(frame) < 2:
        return float(cfg.training.resample_minutes)
    return float(pd.Series(frame.index).diff().median().total_seconds() / 60.0)


def base_load_design(index: pd.DatetimeIndex, timezone: str, harmonics: int) -> tuple[np.ndarray, list[str]]:
    """Design matrix for the base load: clock features only.

    Deliberately no outdoor temperature, no wind, no solar. Any regressor that
    correlates with what drives the heat pump would let the base-load term
    absorb heat-pump signal, which is the one failure that would quietly
    corrupt the efficiency estimate.
    """
    local = pd.DatetimeIndex(index)
    local = local.tz_localize("UTC") if local.tz is None else local
    local = local.tz_convert(timezone)
    hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0

    columns = [np.ones(len(local))]
    names = ["base_constant"]
    for k in range(1, max(1, harmonics) + 1):
        columns.append(np.sin(2 * np.pi * k * hour / 24.0))
        columns.append(np.cos(2 * np.pi * k * hour / 24.0))
        names.extend([f"base_sin{k}", f"base_cos{k}"])
    columns.append((local.dayofweek.to_numpy() >= 5).astype(float))
    names.append("base_weekend")
    return np.column_stack(columns), names


def _irls(
    design: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    huber_scale: float,
    asymmetry: float,
    iterations: int = 12,
) -> tuple[np.ndarray, float]:
    """Bounded least squares with a robust (Huber) weight, and the standard
    error of the first coefficient.

    The Huber weight is what keeps an oven or a sauna from being read as heat
    pump. ``asymmetry`` additionally down-weights *negative* residuals - the
    model claiming less power than was measured - which pulls the fit toward
    the lower envelope of the data. That sounds attractive but biases the
    efficiency scale upward, because the base-load term can already absorb the
    average appliance load on its own; it defaults to 1.0 (symmetric) and is
    there for houses with unusually spiky loads.

    The returned standard error on the heat-pump coefficient is what says
    whether the answer means anything - far more useful than an R-squared,
    which on a whole-house measurement is dominated by appliance noise no
    matter how well the heat pump itself is resolved.
    """
    weights = np.ones(len(target))
    solution = np.zeros(design.shape[1])
    residual = target.copy()
    for _ in range(iterations):
        root = np.sqrt(weights)[:, None]
        result = lsq_linear(design * root, target * np.sqrt(weights), bounds=(lower, upper))
        solution = result.x
        residual = design @ solution - target
        scale = max(huber_scale, 1e-6)
        huber = np.minimum(1.0, scale / np.maximum(np.abs(residual), 1e-9))
        weights = np.where(residual < 0.0, huber * asymmetry, huber)

    dof = max(len(target) - design.shape[1], 1)
    sigma2 = float(np.sum(weights * residual**2) / dof)
    try:
        information = (design * weights[:, None]).T @ design
        covariance = np.linalg.inv(information) * sigma2
        standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        standard_error = float("nan")
    return solution, standard_error


def disaggregate(
    frame: pd.DataFrame,
    cfg: Config,
    q_compressor: np.ndarray,
    q_backup: np.ndarray,
    cop_base: np.ndarray,
) -> Disaggregation | None:
    """Split whole-house power and recover the pump's efficiency scale.

    ``q_compressor``, ``q_backup`` and ``cop_base`` come from simulating the
    identified thermal model through the same period, so the heat pump's power
    *shape* is known and only its level is fitted.
    """
    settings = cfg.power
    total, balanced = house_power(frame, cfg)
    if total is None:
        return None

    if settings.target == "balanced" and balanced is not None:
        measured = balanced
        target_name = "balanced (3 x min phase)"
    else:
        measured = total
        target_name = "total house power"

    charging = ev_mask(frame, cfg)
    usable = (
        np.isfinite(measured.to_numpy(dtype=float))
        & np.isfinite(q_compressor)
        & np.isfinite(cop_base)
        & (cop_base > MIN_COP)
    )
    fit_mask = usable & ~charging
    if int(fit_mask.sum()) < settings.min_samples:
        log.warning(
            "Only %d usable samples for power disaggregation (need %d); skipping",
            int(fit_mask.sum()), settings.min_samples,
        )
        return None

    basis, names = base_load_design(frame.index, cfg.site.timezone, settings.base_harmonics)
    x_hp = q_compressor / np.maximum(cop_base, MIN_COP)
    design_all = np.column_stack([x_hp, basis])
    # Backup heat is resistive, so its power is known exactly: move it across.
    y_all = measured.to_numpy(dtype=float) - q_backup

    # c = 1 / efficiency_scale, bounded to the same range the config allows.
    lower = np.full(design_all.shape[1], -np.inf)
    upper = np.full(design_all.shape[1], np.inf)
    lower[0], upper[0] = 1.0 / 3.0, 1.0 / 0.3
    lower[1] = 0.0                      # the base load's constant term

    split = int(fit_mask.sum() * (1.0 - settings.validation_fraction))
    indices = np.flatnonzero(fit_mask)
    train, validate = indices[:split], indices[split:]

    solution, standard_error = _irls(
        design_all[train], y_all[train], lower, upper, settings.huber_scale_w, settings.asymmetry
    )
    efficiency_scale = float(np.clip(1.0 / max(solution[0], 1e-6), 0.3, 3.0))
    relative_uncertainty = (
        float("nan") if np.isnan(standard_error) else float(standard_error / max(solution[0], 1e-9))
    )

    base_series = pd.Series(basis @ solution[1:], index=frame.index, name="base_load_w")
    heatpump_series = pd.Series(
        np.maximum(x_hp * solution[0] + q_backup, 0.0), index=frame.index, name="heatpump_power_w"
    )

    metrics = _metrics(
        design_all, y_all, solution, train, validate, measured, heatpump_series, base_series,
        charging, names, frame, cfg,
    )
    metrics["efficiency_scale_uncertainty"] = (
        None if np.isnan(relative_uncertainty) else round(relative_uncertainty, 4)
    )
    ev_power = _estimate_ev_power(total, heatpump_series, base_series, charging, usable)
    metrics.update(_ev_diagnostics(ev_power, cfg))

    log.info(
        "Power disaggregation on %s: efficiency scale %.3f +/- %.1f%%, balanced base load %.0f W, "
        "car charger %.1f kW (nominal %.1f kW)",
        target_name, efficiency_scale, 100 * relative_uncertainty,
        metrics["base_load_mean_w"], ev_power / 1000.0, cfg.power.ev_nominal_kw,
    )
    return Disaggregation(
        efficiency_scale=efficiency_scale,
        base_load_w=base_series,
        heatpump_power_w=heatpump_series,
        ev_power_w=ev_power,
        used_samples=int(fit_mask.sum()),
        span_hours=float(fit_mask.sum()) * _step_minutes(frame, cfg) / 60.0,
        target=target_name,
        metrics=metrics,
    )


def _metrics(
    design: np.ndarray,
    y: np.ndarray,
    solution: np.ndarray,
    train: np.ndarray,
    validate: np.ndarray,
    measured: pd.Series,
    heatpump: pd.Series,
    base: pd.Series,
    charging: np.ndarray,
    names: list[str],
    frame: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    prediction = design @ solution
    step_hours = _step_minutes(frame, cfg) / 60.0

    def r2(idx: np.ndarray) -> float:
        if idx.size < 10:
            return float("nan")
        residual = prediction[idx] - y[idx]
        variance = float(np.var(y[idx]))
        return round(float(1.0 - np.mean(residual**2) / variance), 4) if variance > 0 else float("nan")

    # How much does the heat-pump column overlap with the clock features? A high
    # number means the fit cannot tell them apart, and the efficiency scale is
    # only as trustworthy as that separation.
    hp_column = design[train, 0]
    clock = design[train, 1:]
    overlap = 0.0
    if hp_column.std() > 0:
        for j in range(clock.shape[1]):
            if clock[:, j].std() > 0:
                overlap = max(overlap, abs(float(np.corrcoef(hp_column, clock[:, j])[0, 1])))

    heat_kwh = float(np.sum(heatpump.to_numpy()) * step_hours / 1000.0)
    base_kwh = float(np.sum(np.maximum(base.to_numpy(), 0.0)) * step_hours / 1000.0)
    measured_kwh = float(np.sum(measured.to_numpy()) * step_hours / 1000.0)

    return {
        "train_r2": r2(train),
        "validation_r2": r2(validate),
        "residual_rmse_w": round(float(np.sqrt(np.mean((prediction[validate] - y[validate]) ** 2))), 1)
        if validate.size > 10
        else None,
        # With target: balanced this is the balanced part of everything that is
        # not the heat pump, not the whole household base load - most household
        # load is single-phase and never enters the fit.
        "base_load_mean_w": round(float(np.mean(np.maximum(base.to_numpy(), 0.0))), 1),
        "base_load_min_w": round(float(np.min(base.to_numpy())), 1),
        "heatpump_share_of_measured": round(heat_kwh / measured_kwh, 3) if measured_kwh > 0 else None,
        "heatpump_kwh": round(heat_kwh, 1),
        "base_load_kwh": round(base_kwh, 1),
        "charging_fraction": round(float(np.mean(charging)), 3),
        "clock_confounding": round(overlap, 3),
        "coefficients": {name: round(float(v), 2) for name, v in zip(names, solution[1:])},
    }


def _estimate_ev_power(
    total: pd.Series,
    heatpump: pd.Series,
    base: pd.Series,
    charging: np.ndarray,
    usable: np.ndarray,
) -> float:
    """What the charger appears to draw, as a check on the whole model.

    Not used for anything downstream. If an 11 kW charger comes out at 3 kW, the
    phase mapping or the entity is wrong, and so is everything else built on it.
    """
    mask = charging & usable
    if mask.sum() < 20:
        return float("nan")
    unexplained = total.to_numpy()[mask] - heatpump.to_numpy()[mask] - np.maximum(base.to_numpy()[mask], 0.0)
    return float(np.median(unexplained))


def _ev_diagnostics(ev_power: float, cfg: Config) -> dict[str, Any]:
    nominal = cfg.power.ev_nominal_kw * 1000.0
    out: dict[str, Any] = {"ev_power_estimated_w": None if np.isnan(ev_power) else round(ev_power, 1)}
    if np.isnan(ev_power) or nominal <= 0:
        return out
    ratio = ev_power / nominal
    out["ev_power_ratio"] = round(ratio, 3)
    if not 0.5 <= ratio <= 1.5:
        out["warning"] = (
            f"The car charger looks like {ev_power / 1000:.1f} kW but is configured as "
            f"{cfg.power.ev_nominal_kw:.1f} kW. Check the phase entities and the charging sensor "
            "before trusting the efficiency estimate."
        )
    return out


def quality_warnings(result: Disaggregation) -> list[str]:
    """Plain-language reasons not to trust a disaggregation."""
    warnings: list[str] = []
    metrics = result.metrics
    if result.span_hours < 14 * 24:
        warnings.append(
            f"Only {result.span_hours / 24:.1f} days of usable data went into the efficiency fit. "
            "The reported uncertainty measures precision, not accuracy - with this little data the "
            "answer can be several percent off while still looking precise. Aim for a month."
        )
    uncertainty = metrics.get("efficiency_scale_uncertainty")
    if uncertainty is not None and uncertainty > 0.15:
        warnings.append(
            f"The efficiency scale is only determined to +/-{100 * uncertainty:.0f}%. Collect more data, "
            "or run 'hpmpc excite' to move the pump's power around independently of the clock."
        )
    if (metrics.get("clock_confounding") or 0) > 0.7:
        warnings.append(
            "The heat pump's power is strongly correlated with the time of day, so it is hard to "
            "separate from base load. Run 'hpmpc excite' for a week: moving the offset on a schedule "
            "unrelated to the clock is what breaks that tie."
        )
    if (metrics.get("heatpump_share_of_measured") or 0) < 0.15:
        warnings.append(
            "The heat pump appears to be a small share of the measured power. Consider "
            "power.target: balanced, or check that the phase entities are right."
        )
    if metrics.get("base_load_min_w", 0) < -200:
        warnings.append(
            "The fitted base load goes negative at some hours, which is unphysical. Try fewer "
            "harmonics (power.base_harmonics)."
        )
    if metrics.get("warning"):
        warnings.append(metrics["warning"])
    return warnings
