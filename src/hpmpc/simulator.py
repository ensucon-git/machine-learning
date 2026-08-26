"""A synthetic house, so the whole pipeline can be exercised without touching
Home Assistant.

Used by the test suite and by ``hpmpc demo``. The generated dataset has exactly
the same columns as :mod:`hpmpc.dataset` produces from real recorder history,
which means identification, residual fitting, MPC and reporting can all be
validated end to end against a known ground truth.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Config
from .model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp
from .solar import irradiance_from_cloud_cover


def synthetic_weather(index: pd.DatetimeIndex, rng: np.random.Generator, mean_outdoor: float = -2.0) -> pd.DataFrame:
    """Weather with a daily cycle, multi-day fronts, gusty wind and cloud spells."""
    hours = (index - index[0]).total_seconds() / 3600.0
    daily = 3.5 * np.sin(2 * np.pi * (hours - 9.0) / 24.0)
    # Random walk standing in for synoptic weather fronts.
    front = np.cumsum(rng.normal(0.0, 0.45, size=len(index)))
    front = pd.Series(front).rolling(96, min_periods=1, center=True).mean().to_numpy(copy=True)
    front = front - front.mean()
    t_outdoor = mean_outdoor + daily + front + rng.normal(0.0, 0.15, len(index))

    wind = np.abs(3.0 + 2.5 * np.sin(2 * np.pi * hours / 61.0) + rng.normal(0.0, 1.2, len(index)))
    cloud_walk = pd.Series(np.cumsum(rng.normal(0.0, 6.0, len(index)))).rolling(40, min_periods=1, center=True).mean()
    cloud = np.clip(50.0 + cloud_walk.to_numpy() - cloud_walk.mean(), 0.0, 100.0)
    return pd.DataFrame({"t_outdoor": t_outdoor, "wind": wind, "cloud": cloud}, index=index)


def synthetic_prices(index: pd.DatetimeIndex, rng: np.random.Generator, timezone: str = "Europe/Stockholm") -> pd.Series:
    """A Nordic-looking spot profile: night trough, morning and evening peaks."""
    local = index.tz_convert(timezone)
    hour = local.hour.to_numpy(dtype=float)
    shape = (
        0.55
        + 0.85 * np.exp(-(((hour - 8.0) / 1.8) ** 2))
        + 1.05 * np.exp(-(((hour - 18.5) / 2.2) ** 2))
        - 0.30 * np.exp(-(((hour - 3.0) / 2.5) ** 2))
    )
    day_index = (local.normalize() - local.normalize()[0]).days
    day_level = np.exp(rng.normal(0.0, 0.35, size=int(day_index.max()) + 1))
    level = day_level[day_index]
    hourly = np.clip(shape * level, 0.02, None)
    # Spot prices are constant within the hour; keep that step shape.
    series = pd.Series(hourly, index=index)
    return series.groupby(local.floor("h")).transform("first")


def excitation_offsets(
    index: pd.DatetimeIndex, rng: np.random.Generator, hold_hours: float = 6.0,
    low: float = -4.0, high: float = 3.0
) -> np.ndarray:
    """Piecewise-constant pseudo-random offset - the identification experiment.

    Without deliberate excitation the offset barely moves, and the gain from
    offset to indoor temperature is only weakly identifiable.
    """
    step_hours = (index[1] - index[0]).total_seconds() / 3600.0
    hold = max(1, int(round(hold_hours / step_hours)))
    blocks = int(np.ceil(len(index) / hold))
    values = rng.uniform(low, high, size=blocks)
    return np.repeat(values, hold)[: len(index)]


def simulate_house(
    cfg: Config,
    params: ThermalParams,
    weather: pd.DataFrame,
    price: pd.Series,
    offsets: np.ndarray,
    noise: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the true model and return a dataset in the on-disk schema."""
    rng = np.random.default_rng(seed)
    index = weather.index
    dt = (index[1] - index[0]).total_seconds() / 3600.0
    solar = irradiance_from_cloud_cover(index, weather["cloud"], cfg.site.latitude, cfg.site.longitude).to_numpy()

    exog = Exogenous(
        weather["t_outdoor"].to_numpy(dtype=float)[None, :],
        weather["wind"].to_numpy(dtype=float)[None, :],
        solar[None, :],
        price.to_numpy(dtype=float)[None, :],
    )
    t0 = float(weather["t_outdoor"].iloc[0])
    result = simulate(
        params,
        cfg.heat_pump,
        exog,
        np.asarray(offsets, dtype=float)[None, :],
        State(cfg.control.setpoint, float(steady_state_mass_temp(params, cfg.control.setpoint, t0)), t0),
        dt,
    )

    frame = pd.DataFrame(index=index)
    frame["t_indoor"] = result["t_indoor"][0]
    frame["t_outdoor"] = weather["t_outdoor"].to_numpy()
    frame["wind"] = weather["wind"].to_numpy()
    frame["cloud"] = weather["cloud"].to_numpy()
    frame["solar_ghi"] = solar
    frame["t_supply"] = result["t_supply"][0]
    frame["t_return"] = result["t_supply"][0] - cfg.heat_pump.loop_delta_t
    frame["power"] = result["p_electric"][0]
    frame["price"] = price.to_numpy()
    frame["offset"] = np.asarray(offsets, dtype=float)
    frame["t_mass_true"] = result["t_mass"][0]

    if noise:
        frame["t_indoor"] += rng.normal(0.0, 0.04, len(frame))
        frame["t_outdoor"] += rng.normal(0.0, 0.15, len(frame))
        frame["t_supply"] += rng.normal(0.0, 0.25, len(frame))
        frame["power"] = np.maximum(frame["power"] + rng.normal(0.0, 40.0, len(frame)), 0.0)
    frame.index.name = "time"
    return frame


def true_params() -> ThermalParams:
    """Ground truth used by the demo: deliberately different from the defaults
    so a successful identification is actually visible."""
    # Chosen so the house sits near the setpoint at offset 0 under the default
    # heating curve - otherwise the controller spends the whole demo saturated
    # against its offset limit and there is no room left to shift load.
    return ThermalParams(
        Ci=2100.0,
        Cm=26000.0,
        Him=980.0,
        Hie=165.0,
        Hme=30.0,
        k_wind=0.045,
        A_sol=5.5,
        f_sol_i=0.45,
        Q_int=420.0,
        Hfloor=1350.0,
    )


def make_demo_dataset(
    cfg: Config,
    days: int = 30,
    seed: int = 0,
    params: ThermalParams | None = None,
    excite: bool = True,
    start: str = "2026-01-05 00:00",
) -> tuple[pd.DataFrame, ThermalParams]:
    rng = np.random.default_rng(seed)
    params = params or true_params()
    periods = int(days * 24 * 60 / cfg.training.resample_minutes)
    index = pd.date_range(start, periods=periods, freq=f"{cfg.training.resample_minutes}min", tz="UTC")
    weather = synthetic_weather(index, rng)
    price = synthetic_prices(index, rng, cfg.site.timezone)
    offsets = excitation_offsets(index, rng) if excite else np.zeros(len(index))
    frame = simulate_house(cfg, params, weather, price, offsets, seed=seed)
    return frame, params


def perturbed(params: ThermalParams, factor: float = 1.6) -> ThermalParams:
    """A deliberately wrong starting guess, for testing identification."""
    return replace(
        params,
        Ci=params.Ci * factor,
        Cm=params.Cm / factor,
        Him=params.Him / factor,
        Hie=params.Hie * factor,
        A_sol=params.A_sol / factor,
        Q_int=params.Q_int * factor,
        Hfloor=params.Hfloor / factor,
    )
