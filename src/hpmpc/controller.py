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

from .config import Config
from .dataset import add_derived, column_map, pivot_history
from .forecast import build_forecast
from .ha import HomeAssistant, HomeAssistantError
from .model import build_pump
from .model.thermal import Exogenous, State, ThermalParams, simulate, steady_state_mass_temp
from .mpc import MpcResult, MpcSolver
from .ntc import temperature_to_resistance
from .residual import ResidualModel

log = logging.getLogger(__name__)

SANE_INDOOR = (0.0, 40.0)
SANE_OUTDOOR = (-50.0, 50.0)


@dataclass
class ControllerState:
    """Everything that must survive a restart."""

    t_indoor: float = 21.0
    t_mass: float = 24.0
    t_filtered_outdoor: float = 0.0
    last_offset: float = 0.0
    updated_at: str = ""
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
        self.pump = build_pump(cfg)
        self.solver = MpcSolver(cfg, params)
        self.state = ControllerState.load(cfg.paths.state_file) or ControllerState(
            t_indoor=cfg.control.setpoint,
            t_mass=float(steady_state_mass_temp(params, cfg.control.setpoint, 0.0)),
        )

    # ------------------------------------------------------------ readings

    def read_sensors(self) -> dict[str, Any]:
        e = self.cfg.entities
        ids = [
            e.indoor_temp, e.outdoor_temp, e.wind_speed, e.outdoor_humidity,
            e.supply_temp, e.price, e.offset_output,
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
            "t_supply": value(e.supply_temp)[0],
            "price": value(e.price)[0],
            "output_raw": value(e.offset_output)[0],
        }

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

    def output_value(self, offset: float, t_outdoor: float) -> tuple[float, str]:
        """Translate an offset in kelvin into the number written to the entity."""
        mode = self.cfg.control.output_mode
        if mode == "offset":
            return round(float(offset), 2), "K"
        fake = float(t_outdoor) + float(offset)
        if mode == "fake_temperature":
            return round(fake, 2), "degC"
        return round(float(temperature_to_resistance(fake, self.cfg.ntc)), 1), "ohm"

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

    def step(self, now: datetime | None = None, apply: bool | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        apply = (not self.cfg.control.dry_run) if apply is None else apply
        report: dict[str, Any] = {"timestamp": now.isoformat(), "applied": False, "notes": []}

        readings = self.read_sensors()
        report["readings"] = {k: v for k, v in readings.items() if v is not None}
        problems = self.check_readings(readings)

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

        forecast, sources = build_forecast(self.cfg, self.ha, now)
        report["forecast_sources"] = sources
        bias = self._residual_bias(forecast)
        exog = Exogenous(
            forecast["t_outdoor"].to_numpy(dtype=float),
            forecast["wind"].to_numpy(dtype=float),
            forecast["solar_ghi"].to_numpy(dtype=float),
            forecast["price"].to_numpy(dtype=float),
            humidity=forecast["humidity"].to_numpy(dtype=float),
            indoor_bias=bias,
        )

        result: MpcResult = self.solver.solve(exog, self.state.to_state(), self.state.last_offset)
        report["mpc"] = result.summary()
        report["plan"] = _plan_table(forecast, result)
        report["mode"] = "mpc"

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
        value, unit = self.output_value(offset, t_outdoor)
        report["output"] = {
            "entity_id": self.cfg.entities.offset_output,
            "value": value,
            "unit": unit,
            "mode": self.cfg.control.output_mode,
        }
        if not apply:
            report["notes"].append("dry run - nothing written to Home Assistant")
            return
        if not self.cfg.entities.offset_output:
            report["notes"].append("no output entity configured - nothing written")
            return
        try:
            self.ha.set_number(self.cfg.entities.offset_output, value)
            report["applied"] = True
            log.info("Offset %+.2f K -> %s = %s %s", offset, self.cfg.entities.offset_output, value, unit)
        except HomeAssistantError as exc:
            report["notes"].append(f"failed to write output: {exc}")
            log.error("Could not write %s: %s", self.cfg.entities.offset_output, exc)
        self._publish_status(report)

    def _publish_status(self, report: dict[str, Any]) -> None:
        entity_id = self.cfg.entities.status_entity
        if not entity_id:
            return
        mpc = report.get("mpc", {})
        attributes = {
            "backup_heater_kwh_horizon": mpc.get("backup_heater_kwh"),
            "friendly_name": "Heat pump MPC",
            "unit_of_measurement": "K",
            "icon": "mdi:heat-pump",
            "mode": report.get("mode"),
            "applied": report.get("applied"),
            "notes": report.get("notes"),
            "predicted_indoor_min": mpc.get("predicted_indoor_min"),
            "predicted_indoor_mean": mpc.get("predicted_indoor_mean"),
            "horizon_cost_sek": mpc.get("horizon_cost_sek"),
            "predicted_saving_sek": mpc.get("predicted_saving_sek"),
            "predicted_saving_pct": mpc.get("predicted_saving_pct"),
            "offset_blocks": mpc.get("offset_blocks"),
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
