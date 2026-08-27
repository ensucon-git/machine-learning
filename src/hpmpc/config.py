"""Configuration loading and validation.

The whole system is driven by a single YAML file. Values of the form
``${ENV_VAR}`` are expanded from the environment so secrets (the Home Assistant
long-lived token) never have to be written to disk.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _build(cls: type, data: dict[str, Any] | None):
    """Instantiate a (possibly nested) dataclass from a plain dict.

    Unknown keys raise, so a typo in the YAML fails loudly instead of silently
    leaving a default in place while the house gets cold.
    """
    data = dict(data or {})
    kwargs: dict[str, Any] = {}
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them before checking for nested dataclasses.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown configuration key '{key}' in section '{cls.__name__}'")
        ftype = hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[key] = value
    return cls(**kwargs)


@dataclass
class HomeAssistantConfig:
    base_url: str = "http://homeassistant.local:8123"
    token: str = ""
    verify_ssl: bool = True
    timeout: float = 30.0


@dataclass
class EntityConfig:
    """Entity ids in Home Assistant.

    Only ``indoor_temp``, ``outdoor_temp`` and ``price`` are strictly required.
    Everything else improves the model when present and is degraded gracefully
    when missing.
    """

    indoor_temp: str = ""
    outdoor_temp: str = ""
    price: str = ""
    wind_speed: str = ""
    cloud_cover: str = ""
    solar_radiation: str = ""
    supply_temp: str = ""
    return_temp: str = ""
    heatpump_power: str = ""
    heatpump_energy: str = ""
    house_power_l1: str = ""
    house_power_l2: str = ""
    house_power_l3: str = ""
    house_power_total: str = ""
    ev_charging: str = ""
    outdoor_humidity: str = ""
    weather: str = ""
    offset_output: str = ""
    status_entity: str = ""
    extra: list[str] = field(default_factory=list)

    def required(self) -> dict[str, str]:
        return {
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": self.outdoor_temp,
        }

    def all_sensor_ids(self) -> list[str]:
        names = [
            self.indoor_temp,
            self.outdoor_temp,
            self.price,
            self.wind_speed,
            self.cloud_cover,
            self.solar_radiation,
            self.supply_temp,
            self.return_temp,
            self.heatpump_power,
            self.heatpump_energy,
            self.house_power_l1,
            self.house_power_l2,
            self.house_power_l3,
            self.house_power_total,
            self.ev_charging,
            self.outdoor_humidity,
            self.offset_output,
        ]
        return [n for n in [*names, *self.extra] if n]


@dataclass
class SiteConfig:
    """Where the house is.

    Used for the local clear-sky solar model and for the SMHI point forecast.
    SMHI's grid is roughly 2.5 km, so anything inside the same town gives the
    same forecast - ``hpmpc geocode`` can pin it exactly if you want.
    """

    address: str = ""
    latitude: float = 59.33
    longitude: float = 18.06
    elevation_m: float = 20.0
    timezone: str = "Europe/Stockholm"


@dataclass
class ForecastConfig:
    """Where the weather and price forecasts come from."""

    weather_source: str = "home_assistant"   # "smhi" | "home_assistant"
    price_source: str = "home_assistant"     # "elprisetjustnu" | "home_assistant"
    price_area: str = "SE3"
    weather_cache_minutes: float = 30.0
    price_cache_minutes: float = 60.0
    timeout: float = 30.0


@dataclass
class HeatPumpConfig:
    """Heating curve, capacity and COP of the pump.

    ``curve_slope`` / ``curve_offset`` describe the supply-temperature setpoint
    the pump derives from the outdoor temperature it *believes* it sees::

        T_supply = curve_offset + curve_slope * (curve_ref - T_outdoor_filtered)

    Most Nordic pumps expose exactly these two numbers.
    """

    model: str = ""
    """Bundled performance map (e.g. ``daikin_erlq016caw1``) or a path to your
    own YAML. Empty falls back to the generic Carnot model."""
    efficiency_scale: float = 1.0
    capacity_scale: float = 1.0
    curve_slope: float = 0.35
    curve_offset: float = 23.0
    curve_ref: float = 20.0
    supply_min: float = 20.0
    supply_max: float = 40.0
    loop_delta_t: float = 5.0
    outdoor_filter_hours: float = 3.0
    heat_stop_temp: float = 17.0
    perceived_min_c: float = -30.0
    perceived_max_c: float = 30.0
    """Absolute bounds on the temperature the pump is allowed to be shown.

    This is a property of the machine, not of the control strategy, so the model
    honours it too - the optimiser never plans something the controller would
    then have to clip. On a Daikin it also keeps the faked value away from the
    operating-range and defrost thresholds."""
    max_heat_output_w: float = 8000.0
    standby_power_w: float = 30.0
    carnot_efficiency: float = 0.45
    cop_min: float = 1.5
    cop_max: float = 6.0
    defrost_penalty: float = 0.12


@dataclass
class NTCConfig:
    """Digital-resistor output.

    The pump reads its outdoor sensor as a resistance. We can therefore command
    a *fake* outdoor temperature by commanding a resistance. Two models are
    supported: a Beta/NTC model, or an interpolated lookup table (recommended -
    take it from the pump's service manual).
    """

    model: str = "beta"  # "beta" | "table"
    r25: float = 22000.0
    beta: float = 3700.0
    table_temp_c: list[float] = field(default_factory=list)
    table_ohm: list[float] = field(default_factory=list)
    resistance_min: float = 0.0
    resistance_max: float = 500000.0
    negative_coefficient: bool = True


@dataclass
class PowerConfig:
    """How to work out what the heat pump is drawing.

    With a dedicated meter on the pump this section is irrelevant. Without one,
    the pump's consumption is separated out of a whole-house measurement; see
    :mod:`hpmpc.disaggregate`.
    """

    source: str = "auto"        # "auto" | "heatpump_meter" | "house" | "none"
    target: str = "balanced"    # "balanced" (3 x min phase) | "total"
    ev_guard_minutes: float = 15.0
    ev_nominal_kw: float = 11.0
    base_harmonics: int = 4
    asymmetry: float = 1.0
    """Weight on negative residuals, i.e. the model claiming more power than was
    measured. Below 1 makes the fit track the lower envelope of the data, which
    sounds right for a load made of positive spikes but biases the efficiency
    upward - the base-load term already absorbs the average appliance load.
    Lower it only if your house has unusually spiky consumption."""
    huber_scale_w: float = 400.0
    validation_fraction: float = 0.25
    min_samples: int = 400


@dataclass
class ControlConfig:
    output_mode: str = "offset"  # "offset" | "fake_temperature" | "resistance"
    offset_min: float = -8.0
    offset_max: float = 5.0
    max_change_per_cycle: float = 1.5
    cycle_minutes: int = 15
    horizon_hours: float = 36.0
    step_minutes: int = 15
    block_hours: float = 3.0
    setpoint: float = 21.0
    comfort_min: float = 20.3
    comfort_max: float = 22.0
    hard_min: float = 19.0
    hard_max: float = 23.5
    weight_comfort: float = 40.0
    weight_hard: float = 400.0
    weight_offset_change: float = 0.05
    weight_terminal: float = 2.0
    weight_backup_heater: float = 0.0
    """Extra SEK per kWh of resistive backup heat, on top of what it already
    costs in electricity. Non-zero expresses "I would rather be a little cooler
    than run the immersion heater at all"."""
    max_electric_power_kw: float = 0.0
    """Soft cap on total electrical input. 0 disables it. Set it below your main
    fuse: a 16 kW compressor plus a 9 kW backup heater can trip a 25 A service,
    and a preheat plan is exactly when they would run together."""
    weight_power_limit: float = 25.0
    price_scale: float = 1.0
    price_addition: float = 0.0
    price_vat_pct: float = 0.0
    """Applied last: ``(spot * price_scale + price_addition) * (1 + vat/100)``.
    Leave at 0 when the price entity already includes VAT."""
    dry_run: bool = False
    max_data_age_minutes: float = 45.0
    observer_gain: float = 1.0
    fallback_offset: float = 0.0
    warm_start_hours: float = 24.0


@dataclass
class OptimizerConfig:
    population: int = 256
    elites: int = 26
    iterations: int = 12
    sigma_floor: float = 0.15
    seed: int = 0
    polish: bool = True


@dataclass
class TrainingConfig:
    history_days: int = 45
    resample_minutes: int = 15
    window_hours: float = 12.0
    window_stride_hours: float = 3.0
    burn_in_hours: float = 24.0
    long_window_hours: float = 48.0
    regularisation: float = 0.05
    restarts: int = 3
    max_windows: int = 900
    validation_fraction: float = 0.25
    use_residual_model: bool = True
    residual_max_correction: float = 0.4
    seed: int = 0


@dataclass
class PathsConfig:
    data_dir: str = "data"
    model_dir: str = "models"
    state_file: str = "data/controller_state.json"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8129
    api_key: str = ""


@dataclass
class Config:
    home_assistant: HomeAssistantConfig = field(default_factory=HomeAssistantConfig)
    entities: EntityConfig = field(default_factory=EntityConfig)
    site: SiteConfig = field(default_factory=SiteConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    heat_pump: HeatPumpConfig = field(default_factory=HeatPumpConfig)
    ntc: NTCConfig = field(default_factory=NTCConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    runtime_overrides: dict[str, str] = field(default_factory=dict)
    """Map ``section.field`` to a Home Assistant helper entity, and the
    controller reads it each cycle. See :mod:`hpmpc.settings`."""

    def validate(self) -> None:
        missing = [k for k, v in self.entities.required().items() if not v]
        if missing:
            raise ValueError(f"entities.{', entities.'.join(missing)} must be configured")
        c = self.control
        if c.output_mode not in {"offset", "fake_temperature", "resistance"}:
            raise ValueError(f"control.output_mode '{c.output_mode}' is not valid")
        if c.offset_min >= c.offset_max:
            raise ValueError("control.offset_min must be below control.offset_max")
        if not (c.hard_min <= c.comfort_min <= c.comfort_max <= c.hard_max):
            raise ValueError("comfort band must sit inside the hard band")
        if c.step_minutes <= 0 or c.horizon_hours <= 0:
            raise ValueError("control.step_minutes and control.horizon_hours must be positive")
        if self.optimizer.elites >= self.optimizer.population:
            raise ValueError("optimizer.elites must be smaller than optimizer.population")
        p = self.power
        if p.source not in {"auto", "heatpump_meter", "house", "none"}:
            raise ValueError(f"power.source '{p.source}' is not valid")
        if p.target not in {"balanced", "total"}:
            raise ValueError(f"power.target '{p.target}' is not valid")
        if not 0.0 < p.asymmetry <= 1.0:
            raise ValueError("power.asymmetry must be in (0, 1]")
        f = self.forecast
        if f.weather_source not in {"smhi", "home_assistant"}:
            raise ValueError(f"forecast.weather_source '{f.weather_source}' is not valid")
        if f.price_source not in {"elprisetjustnu", "home_assistant"}:
            raise ValueError(f"forecast.price_source '{f.price_source}' is not valid")
        if f.price_source == "elprisetjustnu" and f.price_area.upper() not in {"SE1", "SE2", "SE3", "SE4"}:
            raise ValueError(f"forecast.price_area '{f.price_area}' is not a Swedish bidding area")
        if f.weather_source == "smhi" and not (55.0 <= self.site.latitude <= 71.0 and 4.0 <= self.site.longitude <= 32.0):
            raise ValueError(
                f"site.latitude/longitude ({self.site.latitude}, {self.site.longitude}) is outside SMHI's "
                "coverage - set the coordinates (try 'hpmpc geocode') or use forecast.weather_source: home_assistant"
            )

    @property
    def model_path(self) -> Path:
        return Path(self.paths.model_dir) / "thermal_model.json"

    @property
    def residual_path(self) -> Path:
        return Path(self.paths.model_dir) / "residual_model.joblib"

    @property
    def cache_dir(self) -> str:
        return self.paths.data_dir

    @property
    def dataset_path(self) -> Path:
        return Path(self.paths.data_dir) / "history.csv.gz"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg: Config = _build(Config, _expand(raw))
    cfg.validate()
    return cfg
