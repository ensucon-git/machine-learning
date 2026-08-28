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

# Keys that used to exist. Saying what replaced them beats "unknown key".
OUTPUT_DOMAINS = {"number", "input_number", "sensor"}
"""Where the controller may write its decision.

``number``/``input_number`` are helpers that have to exist first, and that keep
their value across a Home Assistant restart - which is what the actuator should
be driven from. ``sensor`` entities are created by hpmpc itself through the
states API, so nothing has to be defined first, at the cost of vanishing on a
restart until the next cycle.
"""

RETIRED_KEYS: dict[str, str] = {
    "output_mode": (
        "control.output_mode is gone - the controller now writes every output entity you "
        "configure, at the same time. Set entities.offset_output (kelvin), "
        "entities.fake_temperature_output (degrees) and/or entities.resistance_output (ohm)."
    ),
}


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
            if key in RETIRED_KEYS:
                raise ValueError(RETIRED_KEYS[key])
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
    pump_outdoor_temp: str = ""
    """What the heat pump itself reports as the outdoor temperature - the Daikin
    integration exposes this. It must be the reading for the sensor you are
    emulating; if you emulate the indoor unit's external sensor but this entity
    reports the outdoor unit's own R1T, the comparison is meaningless."""
    weather: str = ""
    offset_output: str = ""
    """The controller's decision, in kelvin. Always written when configured, and
    what the history is read back from - no conversion, so nothing can drift."""
    fake_temperature_output: str = ""
    """The temperature to show the pump: real outdoor + offset, in degrees. This
    is the one to feed a digital resistor whose curve you calibrate yourself
    against the pump's display."""
    resistance_output: str = ""
    """Ohm, converted here through the NTC table. Only useful if you would
    rather hpmpc owned the sensor curve than Home Assistant."""
    wiper_output: str = ""
    """Digital-potentiometer step, converted here through the pot: section.
    Only if you want hpmpc to command the wiper directly rather than let the
    ESP32 do the conversion."""
    pot_wiper: str = ""
    """What the ESP32 reports it is actually driving the potentiometer to,
    0..255 for an MCP41100. Without entities.pump_outdoor_temp this is the only
    readback in the whole actuator chain, and it is what tells you the write
    landed and was not clamped on the way."""
    status_entity: str = ""
    extra: list[str] = field(default_factory=list)

    def required(self) -> dict[str, str]:
        return {"indoor_temp": self.indoor_temp}

    def outputs(self) -> dict[str, str]:
        return {
            "offset": self.offset_output,
            "fake_temperature": self.fake_temperature_output,
            "resistance": self.resistance_output,
            "wiper": self.wiper_output,
        }

    def self_published(self) -> set[str]:
        """Entities hpmpc puts into Home Assistant itself, through the states API.

        Nothing has to define these first, so "it does not exist yet" is the
        normal state of an install that has not completed a control cycle - not
        something to report as missing.
        """
        published = {v for v in self.outputs().values() if v.startswith("sensor.")}
        if self.status_entity:
            published.add(self.status_entity)
        return published

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
            self.pump_outdoor_temp,
            self.offset_output,
            self.fake_temperature_output,
            self.resistance_output,
            self.wiper_output,
            self.pot_wiper,
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


@dataclass
class PotConfig:
    """The digital potentiometer standing in for the pump's outdoor sensor.

    It is described here rather than only in the ESP32 firmware because its
    limits are the controller's limits. A single MCP41100 spans 0-100 kohm,
    which on a 20 kohm Daikin curve stops at about -7 C: command anything
    colder and the hardware silently clamps, the pump sees -7, and the fit
    learns a house that responds to nothing. ``hpmpc ntc-table`` prints the
    reachable band, and ``hpmpc check`` compares it with perceived_min_c.

    Devices in series add range, not resolution: two MCP41100 reach -20 C with
    the same 392 ohm step.
    """

    model: str = "mcp41100"
    resistance_ohm: float = 100000.0
    """End-to-end resistance of one device."""
    steps: int = 256
    """Tap positions on one device (8-bit = 256)."""
    devices: int = 1
    """How many are wired in series."""
    wiper_ohm: float = 100.0
    """Wiper resistance of one device, present at every tap."""
    series_ohm: float = 0.0
    """A fixed resistor in series with the pot, if you fit one. It shifts the
    whole reachable band colder without adding a second device - useful when the
    warm end is not needed, but note that holiday mode lives at the warm end."""
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
    offset_min: float = -8.0
    offset_max: float = 5.0
    max_change_per_cycle: float = 1.5
    cycle_minutes: int = 15
    horizon_hours: float = 36.0
    step_minutes: int = 15
    block_hours: float = 3.0
    setpoint: float = 21.0
    """The one number. Everything else about comfort is expressed relative to
    it, so changing it moves the whole band and there is no way to end up with
    a setpoint of 16 and a comfort band that still insists on 20.3."""
    comfort_below: float = 0.7
    comfort_above: float = 1.0
    hard_below: float = 2.0
    hard_above: float = 2.5
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

    @property
    def comfort_min(self) -> float:
        return self.setpoint - self.comfort_below

    @property
    def comfort_max(self) -> float:
        return self.setpoint + self.comfort_above

    @property
    def hard_min(self) -> float:
        return self.setpoint - self.hard_below

    @property
    def hard_max(self) -> float:
        return self.setpoint + self.hard_above
    price_scale: float = 1.0
    price_addition: float = 0.0
    price_vat_pct: float = 0.0
    """Applied last: ``(spot * price_scale + price_addition) * (1 + vat/100)``.
    Leave at 0 when the price entity already includes VAT."""
    dry_run: bool = False
    max_data_age_minutes: float = 45.0
    actuator_error_warn_c: float = 1.5
    """Warn when what the pump believes differs from what was commanded by more
    than this, averaged over many cycles. This is the only closed-loop check on
    the resistor and the NTC table; without it the actuator is pure open loop."""
    actuator_error_smoothing: float = 0.03
    observer_gain: float = 1.0
    fallback_offset: float = 0.0
    warm_start_hours: float = 24.0


@dataclass
class ModesConfig:
    """Named comfort profiles - home, away, holiday.

    Selecting a mode replaces the setpoint and, optionally, the band widths.
    Because the band is relative to the setpoint, a holiday profile is one
    number and cannot leave the configuration self-contradictory.
    """

    entity: str = ""
    """An ``input_select`` whose state names the active profile."""
    holiday_entity: str = ""
    """An ``input_boolean`` shortcut: on means the holiday profile, whatever
    the selector says. Simpler to automate and simpler to reach in a hurry."""
    holiday_profile: str = "holiday"
    default: str = "normal"
    return_entity: str = ""
    """An ``input_datetime`` for when you are back. The comfort band returns to
    normal at that moment, and the optimiser - which can see it coming - works
    out for itself when to start reheating the slab. With a ten-hour slab time
    constant that is the difference between walking into a warm house and
    waiting a day for one."""
    return_ramp_hours: float = 2.0
    profiles: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "normal": {"setpoint": 21.0},
            # Setback bands are deliberately lopsided. While nobody is home,
            # too cold is the only thing that matters; being warmer than the
            # setback target costs money and the price term already discourages
            # it. A tight upper bound here would also forbid reheating the slab
            # ahead of a return, which is the whole point of the return time.
            "away": {
                "setpoint": 18.0, "comfort_below": 1.5, "comfort_above": 3.0,
                "hard_below": 3.0, "hard_above": 6.0, "offset_max": 15.0,
            },
            "holiday": {
                "setpoint": 16.0, "comfort_below": 1.5, "comfort_above": 5.0,
                "hard_below": 3.0, "hard_above": 8.0, "offset_max": 25.0,
            },
        }
    )

    PROFILE_KEYS = (
        "setpoint",
        "comfort_below",
        "comfort_above",
        "hard_below",
        "hard_above",
        # A deep setback needs far more authority than day-to-day trimming. The
        # heating curve is designed to hold the normal setpoint, so coasting
        # down to 16 C in winter means telling the pump it is well above its
        # heating cut-off - twenty kelvin of offset, not four. The absolute
        # limit on what the pump may be shown (heat_pump.perceived_max_c) still
        # applies, and is the setting that actually keeps this safe.
        "offset_min",
        "offset_max",
    )

    def profile(self, name: str) -> dict[str, float] | None:
        return self.profiles.get(str(name).strip().lower())

    def names(self) -> list[str]:
        return sorted(self.profiles)


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
    retrain_days: int = 30
    """Retrain automatically when the model is older than this, if the
    controller is running as a service. 0 disables it. The house changes with
    the seasons - leaves, snow cover, how you actually live in it - and a model
    fitted in November is not the same house in March."""
    retrain_hour: int = 3
    """Local hour to do it at. Training pins a core for a minute or two."""
    archive: bool = True
    """Keep our own copy of the recorder history under ``data/history/``.

    Home Assistant's recorder is a rolling window purged on someone else's
    schedule. With the archive on, every control cycle copies the new rows out
    of it, so recorder retention only has to outlast the gap between two
    cycles instead of the six weeks identification wants."""
    archive_keep_days: int = 400
    """How long the archive itself keeps history. A year of 15-minute rows for
    twenty signals is a few megabytes, and more history means a better fit."""
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
    pot: PotConfig = field(default_factory=PotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
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
        if c.offset_min >= c.offset_max:
            raise ValueError("control.offset_min must be below control.offset_max")
        if not 0.0 <= c.comfort_below <= c.hard_below:
            raise ValueError("control.comfort_below must be between 0 and control.hard_below")
        if not 0.0 <= c.comfort_above <= c.hard_above:
            raise ValueError("control.comfort_above must be between 0 and control.hard_above")
        for name, profile in (self.modes.profiles or {}).items():
            unknown = set(profile) - set(ModesConfig.PROFILE_KEYS)
            if unknown:
                raise ValueError(f"modes.profiles['{name}'] has unknown keys: {', '.join(sorted(unknown))}")
            if "setpoint" not in profile:
                raise ValueError(f"modes.profiles['{name}'] must set a setpoint")
        if self.modes.default not in (self.modes.profiles or {}):
            raise ValueError(f"modes.default '{self.modes.default}' is not one of the defined profiles")
        if self.modes.holiday_profile not in (self.modes.profiles or {}):
            raise ValueError(f"modes.holiday_profile '{self.modes.holiday_profile}' is not defined")
        if c.step_minutes <= 0 or c.horizon_hours <= 0:
            raise ValueError("control.step_minutes and control.horizon_hours must be positive")
        if self.optimizer.elites >= self.optimizer.population:
            raise ValueError("optimizer.elites must be smaller than optimizer.population")
        for kind, entity_id in self.entities.outputs().items():
            if entity_id and entity_id.split(".", 1)[0] not in OUTPUT_DOMAINS:
                raise ValueError(
                    f"entities.{kind}_output '{entity_id}' must be a "
                    f"{', '.join(sorted(OUTPUT_DOMAINS))} entity"
                )
        t = self.training
        if t.archive and 0 < t.archive_keep_days < t.history_days:
            raise ValueError(
                f"training.archive_keep_days ({t.archive_keep_days}) is shorter than "
                f"training.history_days ({t.history_days}) - the archive would throw away "
                "history that training asks for"
            )
        if self.entities.outdoor_temp.startswith("weather."):
            raise ValueError(
                f"entities.outdoor_temp '{self.entities.outdoor_temp}' is a weather entity. Its "
                "state is a condition ('partlycloudy'), not a temperature, so it cannot be read "
                "as one. Put it in entities.weather instead and set forecast.weather_source: "
                "home_assistant - that uses its forecast, and the first step of that forecast "
                "also stands in for the missing sensor."
            )
        if not self.entities.outdoor_temp:
            # The forecast's first step stands in for the missing sensor, so the
            # only real requirement is that a forecast can be built at all.
            if self.forecast.weather_source != "smhi" and not self.entities.weather:
                raise ValueError(
                    "Without entities.outdoor_temp the current outdoor temperature has to come "
                    "from somewhere: set forecast.weather_source to 'smhi', configure "
                    "entities.weather, or configure the outdoor sensor."
                )
        pot = self.pot
        if pot.steps < 2 or pot.devices < 1 or pot.resistance_ohm <= 0:
            raise ValueError("pot.steps must be >= 2, pot.devices >= 1 and pot.resistance_ohm > 0")
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

    @property
    def archive_dir(self) -> Path:
        return Path(self.paths.data_dir) / "history"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg: Config = _build(Config, _expand(raw))
    cfg.validate()
    return cfg
