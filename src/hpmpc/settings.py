"""Settings you can change without editing code or restarting anything.

Two routes, for two different situations:

* **Home Assistant entities.** Map a config field to an ``input_number`` and
  the controller picks up the new value on the next cycle. This is for the
  numbers that genuinely change - the grid transfer fee when your contract
  changes, the comfort band when the seasons do.
* **``hpmpc set``.** Edits ``config.yaml`` in place, preserving the comments,
  and refuses to write a file that would not load.

Only a curated list of fields is exposed, each with bounds. A helper entity
that reports something unexpected must not be able to talk the controller into
a 40-degree setpoint, and structural settings (horizon length, block size,
optimiser shape) are deliberately not runtime-changeable because they define
the solver that is already built.
"""

from __future__ import annotations

import logging
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

# field path -> (minimum, maximum). Bounds are sanity rails, not preferences.
OVERRIDABLE: dict[str, tuple[float, float]] = {
    "control.setpoint": (15.0, 26.0),
    "control.comfort_below": (0.0, 6.0),
    "control.comfort_above": (0.0, 6.0),
    "control.hard_below": (0.2, 10.0),
    "control.hard_above": (0.2, 10.0),
    "control.offset_min": (-15.0, 0.0),
    "control.offset_max": (0.0, 30.0),
    "control.max_change_per_cycle": (0.05, 10.0),
    "control.fallback_offset": (-10.0, 10.0),
    "control.price_scale": (0.0, 1000.0),
    "control.price_addition": (0.0, 10.0),
    "control.price_vat_pct": (0.0, 100.0),
    "control.weight_comfort": (0.0, 10000.0),
    "control.weight_offset_change": (0.0, 1000.0),
    "control.weight_backup_heater": (0.0, 100.0),
    "control.max_electric_power_kw": (0.0, 100.0),
    "control.dry_run": (0.0, 1.0),
    "heat_pump.curve_slope": (0.0, 2.0),
    "heat_pump.curve_offset": (0.0, 60.0),
    "heat_pump.curve_ref": (0.0, 40.0),
    "heat_pump.supply_max": (20.0, 65.0),
    "heat_pump.heat_stop_temp": (5.0, 30.0),
    "heat_pump.efficiency_scale": (0.3, 3.0),
}

BOOLEAN_FIELDS = {"control.dry_run"}


class SettingError(ValueError):
    pass


def _split(path: str) -> tuple[str, str]:
    if path.count(".") != 1:
        raise SettingError(f"'{path}' is not a section.field path")
    return path.split(".", 1)


def get_value(cfg: Config, path: str) -> Any:
    section, field_name = _split(path)
    if not hasattr(cfg, section):
        raise SettingError(f"Unknown configuration section '{section}'")
    target = getattr(cfg, section)
    if not hasattr(target, field_name):
        raise SettingError(f"'{section}' has no field '{field_name}'")
    return getattr(target, field_name)


def coerce(path: str, value: Any) -> Any:
    """Bounds-check a value and coerce it to the field's type."""
    if path not in OVERRIDABLE:
        raise SettingError(
            f"'{path}' is not changeable at runtime. Changeable fields: {', '.join(sorted(OVERRIDABLE))}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SettingError(f"'{path}' expects a number, got {value!r}") from exc
    low, high = OVERRIDABLE[path]
    if not low <= number <= high:
        raise SettingError(f"{path}={number} is outside the allowed range [{low}, {high}]")
    return bool(round(number)) if path in BOOLEAN_FIELDS else number


def apply(cfg: Config, values: dict[str, Any]) -> tuple[Config, list[str]]:
    """Return a config with ``values`` applied, plus notes about what changed.

    Anything that fails a bound check, or that would make the configuration
    inconsistent as a whole, is dropped with a note - never applied halfway.
    """
    notes: list[str] = []
    updates: dict[str, dict[str, Any]] = {}
    for path, raw in values.items():
        if raw is None:
            continue
        try:
            value = coerce(path, raw)
        except SettingError as exc:
            notes.append(str(exc))
            continue
        if get_value(cfg, path) == value:
            continue
        section, field_name = _split(path)
        updates.setdefault(section, {})[field_name] = value
        notes.append(f"{path} = {value}")

    if not updates:
        return cfg, notes

    candidate = cfg
    for section, changes in updates.items():
        candidate = replace(candidate, **{section: replace(getattr(candidate, section), **changes)})
    try:
        candidate.validate()
    except ValueError as exc:
        # A single bad override must not be able to take the controller down.
        log.error("Runtime overrides rejected (%s); keeping the previous settings", exc)
        return cfg, [f"rejected: {exc}"]
    return candidate, notes


def read_from_home_assistant(cfg: Config, ha: Any) -> dict[str, Any]:
    """Read the mapped helper entities and return their current values."""
    values: dict[str, Any] = {}
    for path, entity_id in (cfg.runtime_overrides or {}).items():
        if not entity_id:
            continue
        state = ha.get_state(entity_id)
        if state is None:
            log.warning("Runtime override entity %s for %s is unavailable", entity_id, path)
            continue
        if path in BOOLEAN_FIELDS and state.state in {"on", "off"}:
            values[path] = state.state == "on"
            continue
        number = state.numeric
        if number is None:
            log.warning("Runtime override entity %s reported %r, which is not a number", entity_id, state.state)
            continue
        values[path] = number
    return values


def validate_mapping(cfg: Config) -> list[str]:
    """Check the override mapping itself, so typos surface in ``hpmpc check``."""
    problems: list[str] = []
    for path, entity_id in (cfg.runtime_overrides or {}).items():
        if path not in OVERRIDABLE:
            problems.append(f"runtime_overrides: '{path}' is not a changeable field")
        if not entity_id or "." not in str(entity_id):
            problems.append(f"runtime_overrides['{path}']: '{entity_id}' is not an entity id")
    return problems


# ------------------------------------------------------- editing the file


_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?P<rest>.*)$")


def set_in_file(path: str | Path, dotted: str, value: Any) -> tuple[Any, Any]:
    """Set ``section.field`` in a YAML config file, preserving the comments.

    Rewrites the single line rather than reserialising the document, because
    the example config is mostly comments explaining why each number is what it
    is - and losing those would make the file much harder to maintain later.
    """
    from .config import load_config

    file_path = Path(path)
    section, field_name = _split(dotted)
    coerced = coerce(dotted, value)
    original = load_config(file_path)
    previous = get_value(original, dotted)

    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_indent: str | None = None
    in_section = False
    written = False

    for i, line in enumerate(lines):
        match = _LINE.match(line.rstrip("\n"))
        if not match:
            continue
        indent, key = match.group("indent"), match.group("key")
        if not in_section:
            if indent == "" and key == section:
                in_section = True
            continue
        if indent == "":
            break  # next top-level section, and the field was not found
        if section_indent is None:
            section_indent = indent
        if indent != section_indent or key != field_name:
            continue
        rest = match.group("rest")
        comment = ""
        if "#" in rest:
            comment = "  " + rest[rest.index("#") :].strip()
        rendered = "true" if coerced is True else "false" if coerced is False else _render(coerced)
        lines[i] = f"{indent}{key}: {rendered}{comment}\n"
        written = True
        break

    if not written:
        raise SettingError(
            f"Could not find '{field_name}' under '{section}:' in {file_path}. Add the line by hand, "
            "or check that the section exists."
        )

    backup = file_path.with_suffix(file_path.suffix + ".bak")
    backup.write_text("".join(lines), encoding="utf-8")
    try:
        load_config(backup)          # never leave an unloadable config behind
    except Exception as exc:
        backup.unlink(missing_ok=True)
        raise SettingError(f"Refusing to write: the result would not load ({exc})") from exc
    backup.replace(file_path)
    return previous, coerced


def _render(value: Any) -> str:
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value) if not isinstance(value, float) else f"{value:g}"


def describe() -> list[tuple[str, float, float]]:
    return [(path, low, high) for path, (low, high) in sorted(OVERRIDABLE.items())]


def config_fields(cfg: Config) -> list[str]:
    """Every section.field path, for error messages and shell completion."""
    out: list[str] = []
    for section in fields(cfg):
        target = getattr(cfg, section.name)
        if hasattr(target, "__dataclass_fields__"):
            out.extend(f"{section.name}.{f.name}" for f in fields(target))
    return sorted(out)
