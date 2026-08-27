"""Grey-box thermal model of the house (2R2C + floor loop).

Two capacities:

* ``Ti`` - indoor air and light furnishing. Fast, this is what the thermostat
  and the user feel.
* ``Tm`` - the concrete slab and heavy structure. Slow (typically 20-80 h) and
  the reason a floor-heated house is a usable thermal battery: it lets us buy
  heat when electricity is cheap and coast when it is expensive.

Continuous-time equations (W, Wh/K, hours)::

    Ci dTi/dt = Him (Tm - Ti) + Hie_eff (Te - Ti) + f_sol Qs + Qint
    Cm dTm/dt = Him (Ti - Tm) + Hme (Te - Tm) + Qfloor + (1 - f_sol) Qs
    Qfloor    = clip(Hfloor (Tw - Tm), 0, Qmax)   (0 when the pump is blocked)
    Hie_eff   = Hie (1 + k_wind * v_wind)

Everything is vectorised over a leading batch dimension so the optimiser can
roll out hundreds of candidate offset schedules in one call.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from ..config import HeatPumpConfig
from . import heatpump as hp
from .heatpump import PumpModel

MAX_SUBSTEP_HOURS = 5.0 / 60.0

# (lower, upper) identification bounds for each parameter, in the units above.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "Ci": (300.0, 30000.0),
    "Cm": (2000.0, 400000.0),
    "Him": (30.0, 6000.0),
    "Hie": (20.0, 1200.0),
    "Hme": (0.0, 150.0),
    "k_wind": (0.0, 0.25),
    "A_sol": (0.0, 40.0),
    "f_sol_i": (0.05, 0.95),
    "Q_int": (0.0, 2500.0),
    "Hfloor": (50.0, 6000.0),
}


@dataclass
class ThermalParams:
    """Identified building parameters. Defaults are a plausible 140 m2 Swedish
    house with a concrete slab - only a starting point for identification."""

    Ci: float = 1500.0
    Cm: float = 20000.0
    Him: float = 900.0
    Hie: float = 170.0
    Hme: float = 35.0
    k_wind: float = 0.030
    A_sol: float = 4.0
    f_sol_i: float = 0.55
    Q_int: float = 350.0
    Hfloor: float = 1400.0

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.names()], dtype=float)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "ThermalParams":
        return cls(**{n: float(v) for n, v in zip(cls.names(), np.asarray(vec, dtype=float))})

    @classmethod
    def bounds(cls) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([PARAM_BOUNDS[n][0] for n in cls.names()], dtype=float)
        hi = np.array([PARAM_BOUNDS[n][1] for n in cls.names()], dtype=float)
        return lo, hi

    def clipped(self) -> "ThermalParams":
        lo, hi = self.bounds()
        return ThermalParams.from_vector(np.clip(self.to_vector(), lo, hi))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThermalParams":
        return cls(**{n: float(data[n]) for n in cls.names() if n in data})

    def time_constants_hours(self) -> dict[str, float]:
        """Dominant time constants, handy for sanity-checking a fit."""
        fast = self.Ci / max(self.Him + self.Hie, 1e-6)
        slow = self.Cm / max(self.Him + self.Hme + self.Hfloor, 1e-6)
        envelope = (self.Ci + self.Cm) / max(self.Hie + self.Hme, 1e-6)
        return {"air": fast, "slab": slow, "envelope": envelope}

    def heat_loss_w_per_k(self) -> float:
        """Steady-state envelope loss coefficient [W/K] at zero wind."""
        return self.Hie + self.Hme


@dataclass
class Exogenous:
    """Known-in-advance inputs over the horizon, one value per step."""

    t_outdoor: np.ndarray
    wind: np.ndarray
    solar_ghi: np.ndarray
    price: np.ndarray
    humidity: np.ndarray = np.nan
    """Relative humidity in percent. Only used for the defrost derate; NaN means
    "assume the reference humidity", i.e. no extra derate."""
    indoor_bias: np.ndarray = 0.0
    """Learned correction to dTi/dt [K/h] from the residual model. It depends
    only on exogenous signals (time of day, sun, wind, outdoor temperature), so
    it can be precomputed once per solve and added as a known input."""

    def __post_init__(self) -> None:
        self.t_outdoor = np.atleast_2d(np.asarray(self.t_outdoor, dtype=float))
        n = self.t_outdoor.shape[-1]
        self.wind = _as_series(self.wind, n)
        self.solar_ghi = _as_series(self.solar_ghi, n)
        self.price = _as_series(self.price, n)
        self.humidity = _as_series(self.humidity, n, fill_nan=False)
        self.indoor_bias = _as_series(self.indoor_bias, n)

    def __len__(self) -> int:
        """Number of time steps in the horizon."""
        return int(self.t_outdoor.shape[-1])

    @property
    def batch(self) -> int:
        return int(self.t_outdoor.shape[0])


def _as_series(value: Any, n: int, fill_nan: bool = True) -> np.ndarray:
    """Coerce a scalar / (K,) / (B, K) input to a 2-D ``(B, K)`` array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full((1, n), float(arr))
    arr = np.atleast_2d(arr)
    if arr.shape[-1] != n:
        raise ValueError(f"exogenous series has length {arr.shape[-1]}, expected {n}")
    return np.nan_to_num(arr, nan=0.0) if fill_nan else arr


@dataclass
class State:
    """Model state: indoor air, slab, and the pump's own filtered outdoor temp.

    Fields may be scalars (a single rollout) or arrays of length ``B`` (one
    initial condition per batch element, used during identification)."""

    t_indoor: float | np.ndarray
    t_mass: float | np.ndarray
    t_filtered_outdoor: float | np.ndarray

    def as_tuple(self) -> tuple[Any, Any, Any]:
        return (self.t_indoor, self.t_mass, self.t_filtered_outdoor)


def substeps_for(dt_hours: float) -> int:
    return max(1, int(math.ceil(dt_hours / MAX_SUBSTEP_HOURS - 1e-9)))


def simulate(
    params: ThermalParams,
    pump: PumpModel | HeatPumpConfig,
    exog: Exogenous,
    offset: np.ndarray,
    state: State,
    dt_hours: float,
    supply_temp_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Roll the house forward over the horizon.

    Parameters
    ----------
    offset
        Commanded outdoor-temperature offset, shape ``(K,)`` or ``(B, K)``.
        Negative means "tell the pump it is colder than it is", which makes it
        raise the supply temperature.
    supply_temp_override
        If given (shape ``(K,)`` or ``(B, K)``), the heating curve is bypassed
        and this measured supply temperature is used instead. Used during
        identification when a supply sensor is available, which decouples the
        building fit from any error in the assumed heating curve.

    Returns
    -------
    dict of arrays with shape ``(B, K)``: ``t_indoor``, ``t_mass``,
    ``t_filtered_outdoor``, ``t_supply``, ``q_heat``, ``p_electric``.
    """
    pump = PumpModel.coerce(pump)
    u = np.atleast_2d(np.asarray(offset, dtype=float))
    steps = u.shape[-1]
    if steps != len(exog):
        raise ValueError(f"offset has {steps} steps but exogenous data has {len(exog)}")
    batch = max(u.shape[0], exog.batch, *(np.size(v) for v in state.as_tuple()))

    sub = substeps_for(dt_hours)
    h = dt_hours / sub

    te = np.repeat(exog.t_outdoor, sub, axis=1)                      # (1|B, K*S)
    wind = np.repeat(np.maximum(exog.wind, 0.0), sub, axis=1)
    qs = np.repeat(exog.solar_ghi, sub, axis=1) * params.A_sol
    bias = np.repeat(exog.indoor_bias, sub, axis=1)
    u_sub = np.repeat(u, sub, axis=1)                                # (1|B, K*S)

    # The pump's filtered outdoor temperature depends only on known inputs, so
    # the whole trajectory (and hence the supply temperature) is precomputed.
    humidity = np.repeat(exog.humidity, sub, axis=1)
    # Clamp what the pump is allowed to be shown before filtering: the actuator
    # cannot present an arbitrary temperature, and the optimiser must plan
    # against the same limit the controller will enforce.
    perceived = np.clip(te + u_sub, pump.cfg.perceived_min_c, pump.cfg.perceived_max_c)
    t_filt = hp.filter_outdoor_series(perceived, pump.cfg.outdoor_filter_hours, h, state.t_filtered_outdoor)

    if supply_temp_override is None:
        t_supply = pump.supply_setpoint(t_filt)
        enabled = pump.heating_enabled(t_filt).astype(float)
    else:
        override = np.atleast_2d(np.asarray(supply_temp_override, dtype=float))
        if override.shape[-1] != steps:
            raise ValueError("supply_temp_override must have one value per step")
        t_supply = np.repeat(override, sub, axis=1)
        enabled = np.ones_like(t_supply)
    t_water = pump.mean_water_temp(t_supply)

    # COP and capacity depend only on inputs that are already known, so resolve
    # the whole grid once instead of looking it up inside the integration loop.
    operating = pump.operating_point(t_supply, te, humidity)

    hie_eff = params.Hie * (1.0 + params.k_wind * wind)   # (1|B, K*S)

    ti = np.broadcast_to(np.asarray(state.t_indoor, dtype=float), (batch,)).astype(float).copy()
    tm = np.broadcast_to(np.asarray(state.t_mass, dtype=float), (batch,)).astype(float).copy()

    out_ti = np.empty((batch, steps))
    out_tm = np.empty((batch, steps))
    out_q = np.zeros((batch, steps))
    out_p = np.zeros((batch, steps))
    out_backup = np.zeros((batch, steps))

    inv_ci = 1.0 / params.Ci
    inv_cm = 1.0 / params.Cm

    for k in range(steps):
        for s in range(sub):
            j = k * sub + s
            te_j = te[:, j]
            demand = np.maximum(params.Hfloor * (t_water[:, j] - tm), 0.0) * enabled[:, j]
            delivered = operating.deliver_at(demand, j)
            q = delivered["q_heat"]

            d_ti = (
                params.Him * (tm - ti)
                + hie_eff[:, j] * (te_j - ti)
                + params.f_sol_i * qs[:, j]
                + params.Q_int
            ) * inv_ci + bias[:, j]
            d_tm = (
                params.Him * (ti - tm)
                + params.Hme * (te_j - tm)
                + q
                + (1.0 - params.f_sol_i) * qs[:, j]
            ) * inv_cm

            ti = ti + h * d_ti
            tm = tm + h * d_tm

            out_q[:, k] += q
            out_p[:, k] += delivered["p_electric"]
            out_backup[:, k] += delivered["q_backup"]

        out_ti[:, k] = ti
        out_tm[:, k] = tm

    out_q /= sub
    out_p /= sub
    out_backup /= sub

    return {
        "t_indoor": out_ti,
        "t_mass": out_tm,
        "t_filtered_outdoor": np.broadcast_to(t_filt[:, sub - 1 :: sub], (batch, steps)),
        "t_supply": np.broadcast_to(t_supply[:, sub - 1 :: sub], (batch, steps)),
        "q_heat": out_q,
        "q_backup": out_backup,
        "p_electric": out_p,
    }


def steady_state_mass_temp(
    params: ThermalParams, t_indoor: np.ndarray | float, t_outdoor: np.ndarray | float
) -> np.ndarray:
    """Slab temperature that holds ``t_indoor`` in steady state at ``t_outdoor``.

    The slab is never measured, so this is how the model is initialised and how
    a sensible terminal target for the optimiser is derived.
    """
    ti = np.asarray(t_indoor, dtype=float)
    te = np.asarray(t_outdoor, dtype=float)
    required = params.Hie * (ti - te) - params.Q_int
    return ti + required / max(params.Him, 1e-6)


def save_params(path: str | Path, params: ThermalParams, metadata: dict[str, Any] | None = None) -> None:
    payload = {"params": params.to_dict(), "metadata": metadata or {}}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_params(path: str | Path) -> tuple[ThermalParams, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ThermalParams.from_dict(payload["params"]), payload.get("metadata", {})
