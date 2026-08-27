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
from .model import build_pump
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
        build_pump(cfg),
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


def add_house_electricity(
    frame: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
    ev_kw: float = 11.0,
    charge_probability: float = 0.45,
    phase_split: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> pd.DataFrame:
    """Add a plausible whole-house electricity measurement.

    Everything the disaggregation has to contend with, and nothing it is told:
    a base load with a daily shape, single-phase appliances switching on and
    off, an 11 kW car charger on random evenings, and an uneven split of the
    household load across the three phases. The heat pump and the charger are
    balanced; nothing else is.
    """
    out = frame.copy()
    local = out.index.tz_convert(cfg.site.timezone)
    hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0
    weekend = (local.dayofweek.to_numpy() >= 5).astype(float)
    n = len(out)

    # Base load: night trough, morning and evening peaks, a little more at weekends.
    base = (
        260.0
        + 420.0 * np.exp(-(((hour - 7.5) / 1.6) ** 2))
        + 620.0 * np.exp(-(((hour - 19.0) / 2.4) ** 2))
        + 130.0 * weekend
    )
    base = base * rng.normal(1.0, 0.06, n).clip(0.6, 1.5)

    # Single-phase appliances: short, sharp, unpredictable.
    appliance = np.zeros(n)
    appliance_phase = np.zeros(n, dtype=int)
    step_hours = (out.index[1] - out.index[0]).total_seconds() / 3600.0
    events = int(n * step_hours / 24.0 * 6)          # roughly six a day
    for _ in range(max(events, 0)):
        start = int(rng.integers(0, n))
        length = int(rng.integers(1, max(2, int(1.0 / step_hours))))
        power = float(rng.uniform(800.0, 2600.0))
        phase = int(rng.integers(0, 3))
        end = min(start + length, n)
        appliance[start:end] += power
        appliance_phase[start:end] = phase

    # Car charging: whole evenings, on random days, balanced across the phases.
    charging = np.zeros(n)
    day_index = (local.normalize() - local.normalize()[0]).days
    for day in range(int(day_index.max()) + 1):
        if rng.random() > charge_probability:
            continue
        start_hour = float(rng.uniform(16.0, 21.0))
        duration = float(rng.uniform(1.5, 5.0))
        mask = (day_index == day) & (hour >= start_hour) & (hour < start_hour + duration)
        charging[mask] = 1.0

    heatpump = out["power"].to_numpy(dtype=float)
    ev_watts = charging * ev_kw * 1000.0

    phases = []
    for i, share in enumerate(phase_split):
        single_phase = base * share + np.where(appliance_phase == i, appliance, 0.0)
        phases.append(heatpump / 3.0 + ev_watts / 3.0 + single_phase)

    out["house_l1"], out["house_l2"], out["house_l3"] = phases
    out["house_power"] = out["house_l1"] + out["house_l2"] + out["house_l3"]
    out["ev_charging"] = charging
    out["base_load_true"] = base
    out["heatpump_power_true"] = heatpump
    return out


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
    whole_house_power: bool = False,
) -> tuple[pd.DataFrame, ThermalParams]:
    rng = np.random.default_rng(seed)
    params = params or true_params()
    periods = int(days * 24 * 60 / cfg.training.resample_minutes)
    index = pd.date_range(start, periods=periods, freq=f"{cfg.training.resample_minutes}min", tz="UTC")
    weather = synthetic_weather(index, rng)
    price = synthetic_prices(index, rng, cfg.site.timezone)
    offsets = excitation_offsets(index, rng) if excite else np.zeros(len(index))
    frame = simulate_house(cfg, params, weather, price, offsets, seed=seed)
    if whole_house_power:
        frame = add_house_electricity(frame, cfg, rng, ev_kw=cfg.power.ev_nominal_kw)
        frame = frame.drop(columns=["power"])   # no dedicated meter in this scenario
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
