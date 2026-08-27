"""Comfort targets: one setpoint, named modes, and a schedule over the horizon.

The setpoint is the single number that matters, and the comfort band is
expressed relative to it. That is not just tidiness: with absolute bounds it is
possible to set a holiday setpoint of 16 and leave a comfort band that still
insists on 20.3, so the house never actually cools down and the setting appears
to do nothing. Relative bounds make that impossible to express.

A mode is therefore just a named setpoint (plus optional band widths). Holiday
is one number.

The schedule adds time variation, which exists for one reason worth the code:
coming home. A concrete slab has a time constant around ten hours, so a house
left at 16 degrees does not warm up on arrival - it warms up the next day. Tell
the controller when you are back and the comfort band returns to normal at that
moment; the optimiser, which plans 36 hours ahead, sees the constraint coming
and works out for itself when to start reheating and which hours to buy it in.
No preheat heuristic, no fixed lead time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ControlConfig

log = logging.getLogger(__name__)


@dataclass
class ComfortSchedule:
    """Comfort targets for each step of the horizon."""

    setpoint: np.ndarray
    comfort_min: np.ndarray
    comfort_max: np.ndarray
    hard_min: np.ndarray
    hard_max: np.ndarray
    mode: str = "normal"
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.setpoint.size)

    @classmethod
    def flat(cls, control: ControlConfig, steps: int, mode: str = "normal") -> "ComfortSchedule":
        return cls(
            setpoint=np.full(steps, control.setpoint),
            comfort_min=np.full(steps, control.comfort_min),
            comfort_max=np.full(steps, control.comfort_max),
            hard_min=np.full(steps, control.hard_min),
            hard_max=np.full(steps, control.hard_max),
            mode=mode,
        )

    def summary(self) -> dict[str, Any]:
        varying = bool(np.ptp(self.setpoint) > 1e-9)
        out: dict[str, Any] = {
            "mode": self.mode,
            "setpoint_now": round(float(self.setpoint[0]), 2),
            "comfort_band_now": [round(float(self.comfort_min[0]), 2), round(float(self.comfort_max[0]), 2)],
            "varies_over_horizon": varying,
        }
        if varying:
            out["setpoint_end"] = round(float(self.setpoint[-1]), 2)
        if self.notes:
            out["notes"] = self.notes
        return out


def resolve_mode(cfg: Config, ha: Any) -> tuple[str, list[str]]:
    """Work out which comfort profile is active right now.

    The holiday switch wins over the selector: it is the one people reach for
    on the way out of the door, and it should not be possible to leave it on
    while a selector quietly says otherwise.
    """
    modes = cfg.modes
    notes: list[str] = []

    if modes.holiday_entity:
        state = ha.get_state(modes.holiday_entity)
        if state is None:
            notes.append(f"holiday switch {modes.holiday_entity} is unavailable")
        elif str(state.state).strip().lower() in {"on", "true", "home"}:
            return modes.holiday_profile, [f"holiday mode via {modes.holiday_entity}"]

    if modes.entity:
        state = ha.get_state(modes.entity)
        if state is None:
            notes.append(f"mode selector {modes.entity} is unavailable")
        else:
            name = str(state.state).strip().lower()
            if modes.profile(name) is not None:
                return name, notes
            notes.append(f"mode '{state.state}' is not a defined profile; using '{modes.default}'")

    return modes.default, notes


def apply_mode(cfg: Config, name: str) -> Config:
    """Return a config with the named profile's comfort settings applied."""
    profile = cfg.modes.profile(name)
    if not profile:
        return cfg
    return replace(cfg, control=replace(cfg.control, **{k: float(v) for k, v in profile.items()}))


def set_mode(cfg: Config, ha: Any, name: str) -> list[str]:
    """Switch the active comfort mode in Home Assistant.

    Writes the holiday switch or the selector, whichever is configured, and
    clears the holiday switch when moving to any other mode so the two cannot
    disagree.
    """
    modes = cfg.modes
    if modes.profile(name) is None:
        raise ValueError(f"'{name}' is not a defined profile. Available: {', '.join(modes.names())}")
    name = name.strip().lower()
    actions: list[str] = []

    if modes.holiday_entity:
        wanted = name == modes.holiday_profile
        ha.call_service(
            "input_boolean", "turn_on" if wanted else "turn_off", {"entity_id": modes.holiday_entity}
        )
        actions.append(f"{modes.holiday_entity} -> {'on' if wanted else 'off'}")

    if modes.entity and not (modes.holiday_entity and name == modes.holiday_profile):
        domain = modes.entity.split(".", 1)[0]
        if domain != "input_select":
            raise ValueError(f"modes.entity must be an input_select, got '{modes.entity}'")
        ha.call_service("input_select", "select_option", {"entity_id": modes.entity, "option": name})
        actions.append(f"{modes.entity} -> {name}")

    if not actions:
        raise ValueError(
            "No mode entity is configured. Set modes.entity or modes.holiday_entity in config.yaml."
        )
    return actions


def read_return_time(cfg: Config, ha: Any) -> datetime | None:
    """Read the "back home at" helper, if one is configured and set."""
    entity_id = cfg.modes.return_entity
    if not entity_id:
        return None
    state = ha.get_state(entity_id)
    if state is None:
        return None
    raw = state.attributes.get("timestamp")
    try:
        stamp = (
            pd.Timestamp(float(raw), unit="s", tz="UTC")
            if raw is not None
            else pd.Timestamp(state.state)
        )
    except (TypeError, ValueError):
        log.warning("Could not read a time from %s (%r)", entity_id, state.state)
        return None
    if stamp.tz is None:
        stamp = stamp.tz_localize(cfg.site.timezone)
    return stamp.tz_convert("UTC").to_pydatetime()


def _as_utc(moment: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


def build_schedule(
    cfg: Config,
    index: pd.DatetimeIndex,
    mode: str,
    return_time: datetime | None = None,
) -> ComfortSchedule:
    """Comfort targets across the horizon for the active mode.

    Flat, unless a return time falls inside the horizon - then the band moves
    back to the default profile as that moment arrives, and the optimiser does
    the rest.
    """
    active = apply_mode(cfg, mode).control
    steps = len(index)
    schedule = ComfortSchedule.flat(active, steps, mode=mode)

    if return_time is None or mode == cfg.modes.default:
        return schedule

    normal = apply_mode(cfg, cfg.modes.default).control
    target = _as_utc(return_time)
    stamps = pd.DatetimeIndex(index)
    stamps = stamps.tz_localize("UTC") if stamps.tz is None else stamps.tz_convert("UTC")
    hours_until = np.asarray((target - stamps).total_seconds(), dtype=float) / 3600.0
    ramp = max(cfg.modes.return_ramp_hours, 1e-6)
    weight = np.clip(1.0 - hours_until / ramp, 0.0, 1.0)

    if float(weight.max()) <= 0.0:
        schedule.notes.append(
            f"back at {target.tz_convert(cfg.site.timezone):%Y-%m-%d %H:%M}, beyond this horizon"
        )
        return schedule

    def blend(low: float, high: float) -> np.ndarray:
        return low + weight * (high - low)

    schedule.setpoint = blend(active.setpoint, normal.setpoint)
    schedule.comfort_min = blend(active.comfort_min, normal.comfort_min)
    schedule.comfort_max = blend(active.comfort_max, normal.comfort_max)
    schedule.hard_min = blend(active.hard_min, normal.hard_min)
    schedule.hard_max = blend(active.hard_max, normal.hard_max)
    schedule.notes.append(
        f"returning to '{cfg.modes.default}' at "
        f"{target.tz_convert(cfg.site.timezone):%Y-%m-%d %H:%M}"
    )
    return schedule
