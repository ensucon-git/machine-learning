"""Optional learned residual on top of the physical model.

The 2R2C model cannot know about the shower you take every morning, the wood
stove, or the fact that low winter sun hits the south windows harder than a
flat-plate irradiance number suggests. A gradient-boosted tree learns that
leftover pattern.

Deliberate restriction: features are *exogenous only* (clock, sun, wind,
outdoor temperature). The correction is therefore independent of the control
decision, which means (a) it cannot create a feedback loop that the optimiser
could exploit, and (b) it can be evaluated once per solve instead of once per
rollout step. Its magnitude is hard-clipped so a bad fit can never dominate the
physics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import Config
from .identify import _initial_mass_temp
from .model.thermal import Exogenous, State, ThermalParams, simulate

log = logging.getLogger(__name__)

FEATURES = ["hour_sin", "hour_cos", "is_weekend", "solar_ghi", "wind", "t_outdoor", "doy_sin", "doy_cos"]


def build_features(index: pd.DatetimeIndex, exog: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Exogenous feature matrix. Clock features use local time, since occupancy
    follows the wall clock and not UTC."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(timezone)
    hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0
    doy = local.dayofyear.to_numpy(dtype=float)
    out = pd.DataFrame(index=idx)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_weekend"] = (local.dayofweek.to_numpy() >= 5).astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for name in ("solar_ghi", "wind", "t_outdoor"):
        out[name] = (
            exog[name].to_numpy(dtype=float) if name in exog else np.zeros(len(idx), dtype=float)
        )
    return out[FEATURES]


@dataclass
class ResidualModel:
    estimator: HistGradientBoostingRegressor
    timezone: str
    max_correction: float
    metrics: dict[str, Any]

    def predict(self, index: pd.DatetimeIndex, exog: pd.DataFrame) -> np.ndarray:
        features = build_features(index, exog, self.timezone)
        raw = self.estimator.predict(features.to_numpy(dtype=float))
        return np.clip(raw, -self.max_correction, self.max_correction)

    def save(self, path: str | Path) -> None:
        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "timezone": self.timezone,
                "max_correction": self.max_correction,
                "metrics": self.metrics,
            },
            p,
        )
        p.with_suffix(".json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ResidualModel":
        import joblib

        payload = joblib.load(Path(path))
        return cls(
            estimator=payload["estimator"],
            timezone=payload["timezone"],
            max_correction=float(payload["max_correction"]),
            metrics=payload.get("metrics", {}),
        )


def one_step_residuals(frame: pd.DataFrame, cfg: Config, params: ThermalParams) -> pd.Series:
    """Measured minus modelled dTi/dt [K/h], one value per step.

    Each step is simulated from the *measured* indoor temperature, so the error
    is a genuine one-step residual and not accumulated drift.
    """
    dt_hours = cfg.training.resample_minutes / 60.0
    data = frame.dropna(subset=["t_indoor", "t_outdoor"]).copy()
    if "offset" not in data:
        data["offset"] = 0.0
    n = len(data)
    if n < 50:
        return pd.Series(dtype=float)

    has_supply = "t_supply" in data and data["t_supply"].notna().mean() > 0.8
    supply = data["t_supply"].ffill().bfill().to_numpy(dtype=float)[None, :] if has_supply else None

    exog = Exogenous(
        data["t_outdoor"].to_numpy(dtype=float)[None, :],
        data.get("wind", pd.Series(0.0, index=data.index)).fillna(0.0).to_numpy(dtype=float)[None, :],
        data.get("solar_ghi", pd.Series(0.0, index=data.index)).fillna(0.0).to_numpy(dtype=float)[None, :],
        data.get("price", pd.Series(1.0, index=data.index)).fillna(1.0).to_numpy(dtype=float)[None, :],
    )
    measured = data["t_indoor"].to_numpy(dtype=float)
    offsets = data["offset"].fillna(0.0).to_numpy(dtype=float)[None, :]

    # Free-running simulation gives the slab trajectory (never measured); the
    # residual is then evaluated one step at a time against the real indoor temp.
    tm0 = float(_initial_mass_temp(params, measured[:1], exog.t_outdoor[:, 0])[0])
    free = simulate(
        params,
        cfg.heat_pump,
        exog,
        offsets,
        State(measured[0], tm0, float(exog.t_outdoor[0, 0] + offsets[0, 0])),
        dt_hours,
        supply_temp_override=supply,
    )
    mass = free["t_mass"][0]
    wind = exog.wind[0]
    solar = exog.solar_ghi[0]
    te = exog.t_outdoor[0]

    hie_eff = params.Hie * (1.0 + params.k_wind * np.maximum(wind, 0.0))
    modelled_rate = (
        params.Him * (mass - measured)
        + hie_eff * (te - measured)
        + params.f_sol_i * params.A_sol * solar
        + params.Q_int
    ) / params.Ci
    measured_rate = np.gradient(measured, dt_hours)

    residual = measured_rate - modelled_rate
    return pd.Series(residual, index=data.index, name="residual_k_per_h")


def fit_residual(frame: pd.DataFrame, cfg: Config, params: ThermalParams) -> ResidualModel | None:
    """Train the residual corrector. Returns ``None`` when it does not help."""
    residual = one_step_residuals(frame, cfg, params)
    if residual.empty or len(residual) < 400:
        log.info("Too little data for a residual model")
        return None

    features = build_features(residual.index, frame.reindex(residual.index), cfg.site.timezone)
    y = residual.to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    x = features.to_numpy(dtype=float)[finite]
    y = y[finite]
    if len(y) < 400:
        return None

    split = int(len(y) * (1.0 - cfg.training.validation_fraction))
    estimator = HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=250,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_samples_leaf=40,
        random_state=cfg.training.seed,
    )
    estimator.fit(x[:split], y[:split])

    cap = cfg.training.residual_max_correction
    pred_val = np.clip(estimator.predict(x[split:]), -cap, cap)
    rmse_model = float(np.sqrt(np.mean((y[split:] - pred_val) ** 2)))
    rmse_zero = float(np.sqrt(np.mean(y[split:] ** 2)))
    improvement = 1.0 - rmse_model / max(rmse_zero, 1e-9)

    metrics = {
        "residual_rmse_k_per_h": round(rmse_model, 5),
        "uncorrected_rmse_k_per_h": round(rmse_zero, 5),
        "improvement": round(improvement, 4),
        "n_train": int(split),
        "n_validation": int(len(y) - split),
        "max_correction_k_per_h": cap,
    }
    log.info("Residual model: RMSE %.4f -> %.4f K/h (%.1f%% better)", rmse_zero, rmse_model, 100 * improvement)

    if improvement < 0.05:
        log.info("Residual model does not improve on the physics; discarding it")
        metrics["accepted"] = False
        return None
    metrics["accepted"] = True
    return ResidualModel(estimator=estimator, timezone=cfg.site.timezone, max_correction=cap, metrics=metrics)
