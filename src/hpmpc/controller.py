"""The control loop.

One cycle:

1. read the current sensor values from Home Assistant;
2. update the state estimate (the slab temperature is never measured, so it is
   tracked with an observer);
3. build the weather/price forecast;
4. solve the MPC;
5. rate-limit, clamp, sanity-check, and write the result to the digital
   resistor.

Every failure mode falls back to the configured neutral offset rather than
leaving a stale extreme value in place: a heating system that silently keeps
"tell the pump it is -8 degrees colder than reality" after the controller dies
is far worse than one that does nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .comfort import apply_mode, build_schedule, read_return_time, resolve_mode
from .config import Config, load_config
from .archive import refresh as refresh_archive
from .dataset import add_derived, column_map, pivot_history
from .forecast import build_forecast
from .ha import HomeAssistant, HomeAssistantError
from .model import build_pump
from .model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp
from .mpc import MpcResult, MpcSolver
from .ntc import (
    reachable_temperatures,
    resistance_to_temperature,
    resistance_to_wiper,
    temperature_to_resistance,
    wiper_resolution,
    wiper_span,
    wiper_to_resistance,
)
from .residual import ResidualModel
from .settings import apply as apply_settings
from .settings import read_from_home_assistant

log = logging.getLogger(__name__)

SANE_INDOOR = (0.0, 40.0)
SANE_OUTDOOR = (-50.0, 50.0)

# Fields a comfort mode owns outright. While a non-default mode is active these
# are taken from the profile, so the setpoint helper on the dashboard cannot
# quietly cancel holiday mode.
MODE_OWNED = {
    "control.setpoint",
    "control.comfort_below",
    "control.comfort_above",
    "control.hard_below",
    "control.hard_above",
    "control.offset_min",
    "control.offset_max",
}


@dataclass
class ControllerState:
    """Everything that must survive a restart."""

    t_indoor: float = 21.0
    t_mass: float = 24.0
    t_filtered_outdoor: float = 0.0
    last_offset: float = 0.0
    updated_at: str = ""
    actuator_error_c: float = 0.0
    actuator_samples: int = 0
    warm_started: bool = False
    consecutive_failures: int = 0
    last_summary: dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> State:
        return State(self.t_indoor, self.t_mass, self.t_filtered_outdoor)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ControllerState | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read controller state (%s); starting cold", exc)
            return None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class Controller:
    def __init__(
        self,
        cfg: Config,
        params: ThermalParams,
        ha: HomeAssistant,
        residual: ResidualModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.params = params
        self.ha = ha
        self.residual = residual
        self.base_cfg = cfg
        self.pump = build_pump(cfg)
        self.solver = MpcSolver(cfg, params)
        self._config_mtime: float | None = None
        self.mode = cfg.modes.default
        self.state = ControllerState.load(cfg.paths.state_file) or ControllerState(
            t_indoor=cfg.control.setpoint,
            t_mass=float(steady_state_mass_temp(params, cfg.control.setpoint, 0.0)),
        )

    def refresh_settings(self) -> list[str]:
        """Re-derive the active configuration from the file, the mode and the helpers.

        Always starts from the configuration as loaded, never from last cycle's
        result - otherwise switching out of holiday mode would never restore the
        normal setpoint. Precedence: the file, then the active mode's profile,
        then the mapped helper entities, except that a non-default mode keeps
        the comfort fields it owns.

        The solver is updated in place rather than rebuilt, so the warm start
        from the previous cycle survives. Horizon and block size define the
        solver's shape and are deliberately not changeable this way.
        """
        base = self.base_cfg
        mode, notes = resolve_mode(base, self.ha)
        candidate = apply_mode(base, mode)

        values = read_from_home_assistant(base, self.ha)
        if mode != base.modes.default:
            ignored = sorted(set(values) & MODE_OWNED)
            if ignored:
                notes.append(f"'{mode}' mode overrides {', '.join(ignored)}")
            values = {k: v for k, v in values.items() if k not in MODE_OWNED}

        updated, applied = apply_settings(candidate, values)
        notes.extend(applied)

        changed = mode != self.mode or updated.control != self.cfg.control or updated.heat_pump != self.cfg.heat_pump
        self.mode = mode
        self.cfg = updated
        if changed:
            self.pump = build_pump(updated)
            self.solver.cfg = updated
            self.solver.pump = self.pump
            log.info(
                "Active settings: mode '%s', setpoint %.1f C, comfort %.1f-%.1f C%s",
                mode, updated.control.setpoint, updated.control.comfort_min, updated.control.comfort_max,
                f" ({'; '.join(notes)})" if notes else "",
            )
        return notes

    def reload_config(self, path: str | Path) -> bool:
        """Re-read config.yaml if it changed on disk. Returns True if it did.

        Editing the file and waiting for the next cycle is the obvious mental
        model, so make it true rather than silently requiring a restart.
        """
        file_path = Path(path)
        try:
            stamp = file_path.stat().st_mtime
        except OSError:
            return False
        if self._config_mtime is not None and stamp <= self._config_mtime:
            return False
        first_load = self._config_mtime is None
        self._config_mtime = stamp
        if first_load:
            return False
        try:
            from .train import load_model

            fresh = load_config(file_path)
            working, params, residual, _ = load_model(fresh)
        except Exception as exc:
            log.error("Reloading %s failed (%s); staying on the previous configuration", file_path, exc)
            return False
        self.cfg = working
        self.base_cfg = working
        self.params = params
        self.residual = residual
        self.pump = build_pump(working)
        self.solver.cfg = working
        self.solver.pump = self.pump
        self.solver.params = params
        log.info("Reloaded %s", file_path)
        return True

    # ------------------------------------------------------------ readings

    def read_sensors(self) -> dict[str, Any]:
        e = self.cfg.entities
        ids = [
            e.indoor_temp, e.outdoor_temp, e.wind_speed, e.outdoor_humidity,
            e.supply_temp, e.price, e.offset_output, e.pump_outdoor_temp,
            e.pot_wiper,
        ]
        states = self.ha.get_states([i for i in ids if i])
        now = datetime.now(timezone.utc)

        def value(entity_id: str) -> tuple[float | None, float | None]:
            st = states.get(entity_id)
            if st is None:
                return None, None
            age = st.age(now)
            return st.numeric, (age.total_seconds() / 60.0 if age else None)

        indoor, indoor_age = value(e.indoor_temp)
        outdoor, outdoor_age = value(e.outdoor_temp)
        return {
            "t_indoor": indoor,
            "t_indoor_age_min": indoor_age,
            "t_outdoor": outdoor,
            "t_outdoor_age_min": outdoor_age,
            "wind": value(e.wind_speed)[0],
            "humidity": value(e.outdoor_humidity)[0],
            "pump_outdoor": value(e.pump_outdoor_temp)[0],
            "t_supply": value(e.supply_temp)[0],
            "price": value(e.price)[0],
            "output_raw": value(e.offset_output)[0],
            "pot_wiper": value(e.pot_wiper)[0],
        }

    def resolve_outdoor(
        self, readings: dict[str, Any], forecast: pd.DataFrame | None
    ) -> tuple[float | None, str]:
        """The current outdoor temperature, from a sensor or from the forecast.

        A sensor at the house is better - it measures the air the building
        actually loses heat to - so it wins when configured. Without one the
        forecast's first step stands in, which keeps the controller from needing
        Home Assistant as a middleman at all.
        """
        measured = readings.get("t_outdoor")
        if measured is not None:
            return float(measured), f"sensor {self.cfg.entities.outdoor_temp}"
        if forecast is not None and len(forecast):
            return float(forecast["t_outdoor"].iloc[0]), "forecast (no outdoor sensor configured)"
        return None, "unavailable"

    def check_readings(self, readings: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        max_age = self.cfg.control.max_data_age_minutes
        for name, bounds in (("t_indoor", SANE_INDOOR), ("t_outdoor", SANE_OUTDOOR)):
            v = readings.get(name)
            if v is None:
                problems.append(f"{name} is unavailable")
                continue
            if not bounds[0] <= v <= bounds[1]:
                problems.append(f"{name}={v} is outside the plausible range {bounds}")
            age = readings.get(f"{name}_age_min")
            if age is not None and age > max_age:
                problems.append(f"{name} is stale ({age:.0f} min > {max_age:.0f} min)")
        return problems

    # --------------------------------------------------------- state model

    def warm_start(self) -> bool:
        """Initialise the unmeasured slab temperature from recent history."""
        hours = self.cfg.control.warm_start_hours
        if hours <= 0:
            return False
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        try:
            long_frame = self.ha.history(list(column_map(self.cfg).keys()), start, end)
        except HomeAssistantError as exc:
            log.warning("Warm start failed (%s); using steady-state slab estimate", exc)
            return False
        frame = pivot_history(long_frame, column_map(self.cfg), self.cfg.control.step_minutes)
        if frame.empty or "t_indoor" not in frame or frame["t_indoor"].notna().sum() < 8:
            log.warning("Not enough history for a warm start")
            return False
        frame = add_derived(frame, self.cfg).dropna(subset=["t_indoor", "t_outdoor"])
        if len(frame) < 8:
            return False

        dt = self.cfg.control.step_minutes / 60.0
        ti = frame["t_indoor"].to_numpy(dtype=float)
        exog = Exogenous(
            frame["t_outdoor"].to_numpy(dtype=float)[None, :],
            frame["wind"].to_numpy(dtype=float)[None, :],
            frame["solar_ghi"].to_numpy(dtype=float)[None, :],
            frame["price"].to_numpy(dtype=float)[None, :],
            humidity=(
                frame["humidity"].to_numpy(dtype=float)[None, :]
                if "humidity" in frame
                else np.full((1, len(frame)), np.nan)
            ),
        )
        offsets = frame["offset"].to_numpy(dtype=float)[None, :]
        supply = (
            frame["t_supply"].ffill().bfill().to_numpy(dtype=float)[None, :]
            if "t_supply" in frame and frame["t_supply"].notna().mean() > 0.8
            else None
        )
        tm0 = float(steady_state_mass_temp(self.params, ti[0], float(exog.t_outdoor[0, 0])))
        result = simulate(
            self.params,
            self.pump,
            exog,
            offsets,
            State(ti[0], tm0, float(exog.t_outdoor[0, 0] + offsets[0, 0])),
            dt,
            supply_temp_override=supply,
        )
        self.state.t_mass = float(result["t_mass"][0, -1])
        self.state.t_filtered_outdoor = float(result["t_filtered_outdoor"][0, -1])
        self.state.t_indoor = float(ti[-1])
        self.state.warm_started = True
        log.info("Warm start over %.0f h -> slab %.2f C, pump sees %.2f C",
                 hours, self.state.t_mass, self.state.t_filtered_outdoor)
        return True

    def update_estimate(self, readings: dict[str, Any], elapsed_hours: float) -> None:
        """Propagate the model one cycle and correct it with the measurement.

        A plain Luenberger observer: the indoor temperature is measured and is
        simply adopted, and its prediction error is used to nudge the slab
        temperature, which is not measured.
        """
        indoor = readings.get("t_indoor")
        outdoor = readings.get("t_outdoor")
        if indoor is None or outdoor is None or elapsed_hours <= 0:
            if indoor is not None:
                self.state.t_indoor = float(indoor)
            return
        steps = max(1, int(round(elapsed_hours / (self.cfg.control.step_minutes / 60.0))))
        dt = elapsed_hours / steps
        exog = Exogenous(
            np.full((1, steps), float(outdoor)),
            np.full((1, steps), float(readings.get("wind") or 0.0)),
            np.zeros((1, steps)),
            np.ones((1, steps)),
            humidity=np.full((1, steps), float(readings.get("humidity") or np.nan)),
        )
        supply = (
            np.full((1, steps), float(readings["t_supply"]))
            if readings.get("t_supply") is not None
            else None
        )
        result = simulate(
            self.params,
            self.pump,
            exog,
            np.full((1, steps), float(self.state.last_offset)),
            self.state.to_state(),
            dt,
            supply_temp_override=supply,
        )
        predicted_indoor = float(result["t_indoor"][0, -1])
        innovation = float(indoor) - predicted_indoor
        gain = self.cfg.control.observer_gain
        self.state.t_mass = float(result["t_mass"][0, -1]) + gain * innovation
        self.state.t_filtered_outdoor = float(result["t_filtered_outdoor"][0, -1])
        self.state.t_indoor = float(indoor)
        # Keep the slab estimate physically plausible even after a long outage.
        self.state.t_mass = float(np.clip(self.state.t_mass, float(indoor) - 5.0, float(indoor) + 25.0))
        log.debug("Observer: predicted %.3f, measured %.3f, slab -> %.2f",
                  predicted_indoor, indoor, self.state.t_mass)

    # -------------------------------------------------------------- output

    def outputs(self, offset: float, t_outdoor: float) -> list[dict[str, Any]]:
        """Every representation of the decision, for every entity configured.

        The controller decides one thing - an offset in kelvin - and publishes it
        in as many forms as you have somewhere to put. There is no "mode" to pick
        between them: they are the same number, and writing them all means the
        one you consume and the one you read on a dashboard can never disagree.

        ``fake_temperature`` is the useful one when the resistor curve is
        calibrated in Home Assistant against what the pump's display actually
        shows, which for most pumps is the only place that number exists.
        """
        entities = self.cfg.entities
        fake = float(t_outdoor) + float(offset)      # already inside perceived_min/max
        candidates = [
            ("offset", entities.offset_output, round(float(offset), 2), "K"),
            ("fake_temperature", entities.fake_temperature_output, round(fake, 2), "degC"),
            (
                "resistance",
                entities.resistance_output,
                round(float(temperature_to_resistance(fake, self.cfg.ntc)), 1),
                "ohm",
            ),
            (
                "wiper",
                entities.wiper_output,
                int(resistance_to_wiper(temperature_to_resistance(fake, self.cfg.ntc), self.cfg.pot)),
                "step",
            ),
        ]
        return [
            {"kind": kind, "entity_id": entity_id, "value": value, "unit": unit}
            for kind, entity_id, value, unit in candidates
            if entity_id
        ]

    def _limit(self, target: float, t_outdoor: float | None = None) -> tuple[float, list[str]]:
        c = self.cfg.control
        pump = self.cfg.heat_pump
        notes: list[str] = []
        clamped = float(np.clip(target, c.offset_min, c.offset_max))
        if abs(clamped - target) > 1e-6:
            notes.append(f"clamped to [{c.offset_min}, {c.offset_max}]")
        if t_outdoor is not None:
            # Keep the temperature actually presented to the pump inside the
            # range the machine is happy to see, whatever the offset says.
            bounded = float(np.clip(t_outdoor + clamped, pump.perceived_min_c, pump.perceived_max_c)) - t_outdoor
            if abs(bounded - clamped) > 1e-6:
                notes.append(
                    f"limited so the pump sees between {pump.perceived_min_c} and {pump.perceived_max_c} degC"
                )
                clamped = bounded
        delta = clamped - self.state.last_offset
        if abs(delta) > c.max_change_per_cycle:
            clamped = self.state.last_offset + np.sign(delta) * c.max_change_per_cycle
            notes.append(f"rate limited to {c.max_change_per_cycle} K/cycle")
        return float(clamped), notes

    def check_actuator(self, readings: dict[str, Any]) -> dict[str, Any] | None:
        """Compare what actually reached the sensor input against what we asked for.

        Everything about the offset is otherwise open loop: we command a
        resistance and trust that it means what the NTC table says. There are
        two places the loop can be closed, and they check different lengths of
        the chain:

        ``pump_outdoor_temp`` is the whole chain - resistor, wiring, connector,
        NTC table and the pump's own linearisation - so it is preferred when the
        pump exposes it. The pump filters its outdoor reading, so a single cycle
        is dominated by lag rather than bias; only the long-run mean says
        anything about calibration, and the comparison is smoothed over many
        hours before it is allowed to complain.

        ``pot_wiper`` is what the ESP32 reports it is driving the potentiometer
        to. It stops short of the pump, so it cannot catch a wrong NTC table -
        but it does catch the failure that table cannot: a value the hardware
        could not reach and silently clamped. There is no lag to average out
        there, so that comparison is immediate.

        Deliberately reports rather than corrects. Closing a feedback loop on an
        actuator estimate would let one wrong entity walk the offset away
        quietly, which is exactly the failure this check exists to catch.
        """
        outdoor = readings.get("t_outdoor")
        if outdoor is None:
            return None
        pump = self.cfg.heat_pump
        commanded = float(
            np.clip(outdoor + self.state.last_offset, pump.perceived_min_c, pump.perceived_max_c)
        )

        if readings.get("pump_outdoor") is not None:
            return self._actuator_from_pump(float(readings["pump_outdoor"]), commanded)
        if readings.get("pot_wiper") is not None:
            return self._actuator_from_wiper(float(readings["pot_wiper"]), commanded)
        return None

    def _actuator_from_pump(self, reported: float, commanded: float) -> dict[str, Any]:
        error = reported - commanded
        alpha = self.cfg.control.actuator_error_smoothing
        if self.state.actuator_samples == 0:
            self.state.actuator_error_c = error
        else:
            self.state.actuator_error_c = (1 - alpha) * self.state.actuator_error_c + alpha * error
        self.state.actuator_samples += 1

        settled = self.state.actuator_samples >= int(2.0 / max(alpha, 1e-6))
        result: dict[str, Any] = {
            "source": "pump",
            "pump_believes_c": round(reported, 2),
            "commanded_c": round(commanded, 2),
            "error_now_c": round(error, 2),
            "error_smoothed_c": round(self.state.actuator_error_c, 2),
            "samples": self.state.actuator_samples,
            "settled": settled,
        }
        threshold = self.cfg.control.actuator_error_warn_c
        if settled and abs(self.state.actuator_error_c) > threshold:
            result["warning"] = (
                f"The pump believes it is {self.state.actuator_error_c:+.1f} C away from what was "
                f"commanded, averaged over {self.state.actuator_samples} cycles. The NTC table does "
                "not match the sensor the pump is actually reading. Run 'hpmpc calibrate-ntc' with "
                "pairs taken from the pump's own display."
            )
        return result

    def _actuator_from_wiper(self, wiper: float, commanded: float) -> dict[str, Any]:
        pot, ntc = self.cfg.pot, self.cfg.ntc
        driven_c = float(resistance_to_temperature(wiper_to_resistance(wiper, pot), ntc))
        wanted_step = float(resistance_to_wiper(temperature_to_resistance(commanded, ntc), pot))
        error = driven_c - commanded
        # One wiper step is the smallest error that can possibly exist; anything
        # inside a couple of them is quantisation, not a fault.
        quantisation = wiper_resolution(pot, ntc, commanded)
        result: dict[str, Any] = {
            "source": "pot_wiper",
            "wiper": int(wiper),
            "wiper_commanded": int(wanted_step),
            "driven_c": round(driven_c, 2),
            "commanded_c": round(commanded, 2),
            "error_now_c": round(error, 2),
            "step_resolution_c": round(float(quantisation), 3),
        }
        if int(wiper) in (0, wiper_span(pot)) and abs(error) > max(2.0 * quantisation, 0.5):
            coldest, warmest = reachable_temperatures(pot, ntc)
            result["warning"] = (
                f"The potentiometer is at its end stop ({int(wiper)}) and the pump is being shown "
                f"{driven_c:.1f} C instead of the commanded {commanded:.1f} C. This hardware can only "
                f"reach {coldest:.1f} to {warmest:.1f} C - raise heat_pump.perceived_min_c to "
                f"{coldest:.0f}, or wire another potentiometer in series."
            )
        elif abs(error) > max(3.0 * quantisation, 1.0):
            result["warning"] = (
                f"The ESP32 reports wiper {int(wiper)} but {int(wanted_step)} was commanded "
                f"({error:+.1f} C). Either the write is not arriving or the pot: section does not "
                "describe the hardware."
            )
        return result

    def _saturation_notes(self, result: MpcResult, comfort: Any) -> list[str]:
        """Warn when the offset limits make the requested temperature unreachable.

        The heating curve is designed around the normal setpoint, so a deep
        setback needs far more offset authority than day-to-day trimming - about
        twenty kelvin to coast to 16 C in a Swedish winter, not four. Without
        that headroom the controller sits pinned at its limit and the setpoint
        silently does nothing, which is exactly the kind of failure that looks
        like the model being wrong.
        """
        c = self.cfg.control
        notes: list[str] = []
        indoor = result.trajectory["t_indoor"]
        pinned_high = float(np.mean(result.offset_schedule >= c.offset_max - 0.05))
        pinned_low = float(np.mean(result.offset_schedule <= c.offset_min + 0.05))

        if pinned_high > 0.6 and float(np.min(indoor)) > float(np.max(comfort.comfort_max)) + 0.2:
            notes.append(
                f"cannot cool the house to {c.setpoint:.1f} C: the offset is pinned at "
                f"+{c.offset_max:.1f} K and the plan still stays above {float(np.min(indoor)):.1f} C. "
                "Raise offset_max for this mode, or lower the heating curve."
            )
        if pinned_low > 0.6 and float(np.max(indoor)) < float(np.min(comfort.comfort_min)) - 0.2:
            notes.append(
                f"cannot heat the house to {c.setpoint:.1f} C: the offset is pinned at "
                f"{c.offset_min:.1f} K and the plan still stays below {float(np.max(indoor)):.1f} C. "
                "Raise the heating curve, or check for a capacity limit in 'hpmpc pump-table'."
            )
        return notes

    def _unreachable(self, t_outdoor: float | None) -> bool:
        """Is it colder outside than the actuator can show the pump?

        Below ``heat_pump.perceived_min_c`` there is no resistance to command:
        every value, offset zero included, would be clamped to something warmer
        than the truth, and the pump would quietly underheat. A single
        MCP41100 runs out at about -7 C on a Daikin 20 kohm curve, so this is
        an ordinary Swedish winter night, not an exotic corner.
        """
        if not self.cfg.control.release_when_unreachable or t_outdoor is None:
            return False
        return float(t_outdoor) < self.cfg.heat_pump.perceived_min_c - 0.05

    def _release(self, now: datetime, t_outdoor: float | None,
                 report: dict[str, Any]) -> dict[str, Any]:
        """Stop commanding, and let the emulator fall back to the real sensor.

        Nothing is written, deliberately: the ESP32 watchdog and the Home
        Assistant dead-man switch both release the relay when commands stop, so
        not writing IS the instruction. The pump then behaves exactly as it did
        before any of this existed, which is the correct behaviour for weather
        the hardware cannot lie about. Control resumes by itself as soon as it
        warms back up.
        """
        floor = self.cfg.heat_pump.perceived_min_c
        message = (
            f"{t_outdoor:.1f} C outside is below what the pump can be shown ({floor:.1f} C), "
            "so no offset can be delivered - handing the pump back to its real sensor. "
            "Wire another potentiometer in series (pot.devices) to control through this weather."
        )
        report["mode"] = "released"
        report["offset"] = 0.0
        report["notes"].append(message)
        report["outputs"] = []
        log.warning("%s", message)
        self._publish_status(report)
        self.state.last_offset = 0.0
        self.state.updated_at = now.isoformat()
        self.state.save(self.cfg.paths.state_file)
        return report

    def _safety_override(self, readings: dict[str, Any]) -> tuple[float | None, str]:
        c = self.cfg.control
        indoor = readings.get("t_indoor")
        if indoor is None:
            return None, ""
        if indoor < c.hard_min:
            return c.offset_min, f"indoor {indoor:.1f} C below hard minimum {c.hard_min} C - maximum heat"
        if indoor > c.hard_max:
            return c.offset_max, f"indoor {indoor:.1f} C above hard maximum {c.hard_max} C - minimum heat"
        return None, ""

    # ---------------------------------------------------------------- step

    def archive_cycle(self, report: dict[str, Any], now: datetime | None = None) -> None:
        """Copy whatever the recorder has gained since last cycle into our own
        archive. Never allowed to disturb control: the worst case is that this
        cycle's rows are picked up by the next one instead."""
        if not self.cfg.training.archive:
            return
        try:
            report["archive"] = refresh_archive(self.cfg, self.ha, now=now)
        except (HomeAssistantError, ValueError, OSError) as exc:
            log.warning("Could not update the history archive: %s", exc)
            report["archive"] = {"error": str(exc)}

    def step(self, now: datetime | None = None, apply: bool | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        apply = (not self.cfg.control.dry_run) if apply is None else apply
        report: dict[str, Any] = {"timestamp": now.isoformat(), "applied": False, "notes": []}
        self.archive_cycle(report, now)

        setting_notes = self.refresh_settings()
        if setting_notes:
            report["settings"] = setting_notes

        readings = self.read_sensors()

        # The forecast comes first: without an outdoor sensor it is also where
        # the current outdoor temperature comes from, and the offset means
        # nothing until we know what it is being added to.
        forecast: pd.DataFrame | None = None
        sources: dict[str, Any] = {}
        try:
            forecast, sources = build_forecast(self.cfg, self.ha, now)
        except (HomeAssistantError, ValueError) as exc:
            log.error("Could not build the forecast: %s", exc)
            report["forecast_error"] = str(exc)
        report["forecast_sources"] = sources

        outdoor, outdoor_source = self.resolve_outdoor(readings, forecast)
        readings["t_outdoor"] = outdoor
        readings["t_outdoor_source"] = outdoor_source
        if outdoor is not None and not self.cfg.entities.outdoor_temp:
            readings["t_outdoor_age_min"] = 0.0      # a forecast for now is, by definition, now

        report["readings"] = {k: v for k, v in readings.items() if v is not None}

        # Colder outside than the actuator can present? Then even offset 0 is a
        # lie, and the honest move is to get out of the way entirely.
        if self._unreachable(outdoor):
            return self._release(now, outdoor, report)

        problems = self.check_readings(readings)
        if forecast is None:
            problems.append("no weather or price forecast available")

        if problems:
            self.state.consecutive_failures += 1
            report["problems"] = problems
            fallback, notes = self._limit(self.cfg.control.fallback_offset, readings.get("t_outdoor"))
            report["notes"].extend(notes)
            report["offset"] = fallback
            report["mode"] = "fallback"
            log.error("Sensor problems (%s) - falling back toward %.2f K",
                      "; ".join(problems), self.cfg.control.fallback_offset)
            self._write(fallback, readings.get("t_outdoor") or 0.0, report, apply)
            self._persist(now, report)
            return report

        self.state.consecutive_failures = 0
        elapsed_raw = self._hours_since_last_cycle(now)
        # A cold start, or a long outage, means the slab estimate is worthless.
        # Rebuild it from history rather than integrating across the gap.
        if not self.state.warm_started or elapsed_raw > 6.0:
            if elapsed_raw > 6.0:
                report["notes"].append(f"{elapsed_raw:.1f} h since the last cycle - re-warming the state estimate")
            self.state.warm_started = False
            self.warm_start()
            elapsed_raw = 0.0

        self.update_estimate(readings, float(np.clip(elapsed_raw, 0.0, 6.0)))

        actuator = self.check_actuator(readings)
        if actuator:
            report["actuator"] = actuator
            if actuator.get("warning"):
                report["notes"].append(actuator["warning"])
                log.warning("%s", actuator["warning"])

        bias = self._residual_bias(forecast)
        exog = Exogenous(
            forecast["t_outdoor"].to_numpy(dtype=float),
            forecast["wind"].to_numpy(dtype=float),
            forecast["solar_ghi"].to_numpy(dtype=float),
            forecast["price"].to_numpy(dtype=float),
            humidity=forecast["humidity"].to_numpy(dtype=float),
            indoor_bias=bias,
        )

        comfort = build_schedule(
            self.cfg, forecast.index, self.mode, read_return_time(self.base_cfg, self.ha)
        )
        report["comfort"] = comfort.summary()
        result: MpcResult = self.solver.solve(
            exog, self.state.to_state(), self.state.last_offset, comfort=comfort
        )
        report["mpc"] = result.summary()
        report["plan"] = _plan_table(forecast, result)
        report["mode"] = "mpc"

        report["notes"].extend(self._saturation_notes(result, comfort))

        target = result.offset_now
        override, reason = self._safety_override(readings)
        if override is not None:
            target = override
            report["mode"] = "safety_override"
            report["notes"].append(reason)
            log.warning("Safety override: %s", reason)

        offset, notes = self._limit(target, float(readings["t_outdoor"]))
        report["notes"].extend(notes)
        report["offset"] = offset
        self._write(offset, float(readings["t_outdoor"]), report, apply)
        self._persist(now, report)
        return report

    def excite_step(
        self,
        now: datetime | None = None,
        apply: bool | None = None,
        hold_hours: float = 6.0,
        low: float = -4.0,
        high: float = 3.0,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Identification experiment: hold a pseudo-random offset for a few hours.

        Normal operation barely moves the offset, which leaves the gain from
        offset to indoor temperature poorly determined. Running this for a week
        in the heating season produces data the fit can actually learn from. The
        comfort guard still applies, so the house never leaves the hard band.
        """
        now = now or datetime.now(timezone.utc)
        apply = (not self.cfg.control.dry_run) if apply is None else apply
        block = int(now.timestamp() // (hold_hours * 3600.0))
        rng = np.random.default_rng([seed, block])
        target = float(np.clip(rng.uniform(low, high), self.cfg.control.offset_min, self.cfg.control.offset_max))

        report: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "mode": "excitation",
            "applied": False,
            "notes": [f"excitation block {block}, hold {hold_hours} h"],
        }
        # Excitation week is exactly the data the fit needs most; archive it.
        self.archive_cycle(report, now)
        readings = self.read_sensors()
        report["readings"] = {k: v for k, v in readings.items() if v is not None}
        problems = self.check_readings(readings)
        if problems:
            report["problems"] = problems
            target = self.cfg.control.fallback_offset
            report["notes"].append("sensor problem - excitation paused")
        else:
            override, reason = self._safety_override(readings)
            if override is not None:
                target = override
                report["notes"].append(reason)
                report["mode"] = "excitation_safety_override"

        offset, notes = self._limit(target, readings.get("t_outdoor"))
        report["notes"].extend(notes)
        report["offset"] = offset
        self._write(offset, float(readings.get("t_outdoor") or 0.0), report, apply)
        self._persist(now, report)
        return report

    def _hours_since_last_cycle(self, now: datetime) -> float:
        if not self.state.updated_at:
            return 0.0
        try:
            previous = pd.Timestamp(self.state.updated_at)
        except (ValueError, TypeError):
            return 0.0
        if previous.tz is None:
            previous = previous.tz_localize("UTC")
        return max(0.0, (now - previous.to_pydatetime()).total_seconds() / 3600.0)

    def _residual_bias(self, forecast: pd.DataFrame) -> np.ndarray:
        if self.residual is None:
            return np.zeros(len(forecast))
        try:
            return self.residual.predict(forecast.index, forecast)
        except Exception as exc:  # pragma: no cover - never let it break control
            log.warning("Residual model failed (%s); continuing with physics only", exc)
            return np.zeros(len(forecast))

    def _write(self, offset: float, t_outdoor: float, report: dict[str, Any], apply: bool) -> None:
        outputs = self.outputs(offset, t_outdoor)
        report["outputs"] = outputs
        if not apply:
            report["notes"].append("dry run - nothing written to Home Assistant")
            return
        if not outputs:
            report["notes"].append("no output entity configured - nothing written")
            return

        written = 0
        for output in outputs:
            try:
                self.ha.set_number(output["entity_id"], output["value"])
                output["written"] = True
                written += 1
            except HomeAssistantError as exc:
                output["written"] = False
                output["error"] = str(exc)
                report["notes"].append(f"failed to write {output['entity_id']}: {exc}")
                log.error("Could not write %s: %s", output["entity_id"], exc)
        report["applied"] = written > 0
        if written:
            log.info(
                "Offset %+.2f K -> %s",
                offset,
                ", ".join(f"{o['entity_id']} = {o['value']} {o['unit']}" for o in outputs if o.get("written")),
            )
        self._publish_status(report)

    def _publish_status(self, report: dict[str, Any]) -> None:
        entity_id = self.cfg.entities.status_entity
        if not entity_id:
            return
        mpc = report.get("mpc", {})
        attributes = {
            "backup_heater_kwh_horizon": mpc.get("backup_heater_kwh"),
            "peak_electric_kw": mpc.get("peak_electric_kw"),
            "settings": report.get("settings"),
            "friendly_name": "Heat pump MPC",
            "unit_of_measurement": "K",
            "icon": "mdi:heat-pump",
            "control_mode": report.get("mode"),
            "applied": report.get("applied"),
            "notes": report.get("notes"),
            "predicted_indoor_min": mpc.get("predicted_indoor_min"),
            "predicted_indoor_mean": mpc.get("predicted_indoor_mean"),
            "horizon_cost_sek": mpc.get("horizon_cost_sek"),
            "predicted_saving_sek": mpc.get("predicted_saving_sek"),
            "predicted_saving_pct": mpc.get("predicted_saving_pct"),
            "offset_blocks": mpc.get("offset_blocks"),
            "mode": report.get("comfort", {}).get("mode"),
            "setpoint": report.get("comfort", {}).get("setpoint_now"),
            "comfort_band": report.get("comfort", {}).get("comfort_band_now"),
            "actuator_error_c": report.get("actuator", {}).get("error_smoothed_c"),
            "slab_temperature": round(self.state.t_mass, 2),
            "updated": report.get("timestamp"),
        }
        try:
            self.ha._request(  # noqa: SLF001 - deliberate use of the raw states endpoint
                "POST",
                f"/api/states/{entity_id}",
                json={"state": round(float(report.get("offset", 0.0)), 2), "attributes": attributes},
            )
        except HomeAssistantError as exc:
            log.debug("Could not publish status entity: %s", exc)

    def _persist(self, now: datetime, report: dict[str, Any]) -> None:
        self.state.last_offset = float(report.get("offset", self.state.last_offset))
        self.state.updated_at = now.isoformat()
        self.state.last_summary = report.get("mpc", {})
        self.state.save(self.cfg.paths.state_file)


def _plan_table(forecast: pd.DataFrame, result: MpcResult) -> list[dict[str, Any]]:
    """A compact human-readable plan, one row per hour."""
    rows: list[dict[str, Any]] = []
    traj = result.trajectory
    for i, ts in enumerate(forecast.index):
        if ts.minute != 0:
            continue
        rows.append(
            {
                "time": ts.isoformat(),
                "price": round(float(forecast["price"].iloc[i]), 3),
                "t_outdoor": round(float(forecast["t_outdoor"].iloc[i]), 1),
                "offset": round(float(result.offset_schedule[i]), 2),
                "t_supply": round(float(traj["t_supply"][i]), 1),
                "t_indoor": round(float(traj["t_indoor"][i]), 2),
                "kw": round(float(traj["p_electric"][i]) / 1000.0, 3),
            }
        )
    return rows
