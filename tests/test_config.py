from __future__ import annotations

import pytest

from hpmpc.config import Config, load_config


def write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


BASE = """
entities:
  indoor_temp: sensor.a
  outdoor_temp: sensor.b
"""


def test_loads_nested_sections(tmp_path):
    path = write(tmp_path, BASE + """
control:
  setpoint: 22.5
heat_pump:
  curve_slope: 0.42
""")
    cfg = load_config(path)
    assert cfg.control.setpoint == 22.5
    assert cfg.heat_pump.curve_slope == 0.42
    assert cfg.control.cycle_minutes == 15  # default preserved


def test_environment_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret-value")
    path = write(tmp_path, BASE + """
home_assistant:
  token: ${MY_TOKEN}
  base_url: ${MISSING_URL:-http://fallback:8123}
""")
    cfg = load_config(path)
    assert cfg.home_assistant.token == "secret-value"
    assert cfg.home_assistant.base_url == "http://fallback:8123"


def test_unknown_key_is_rejected(tmp_path):
    path = write(tmp_path, BASE + "control:\n  setpont: 21\n")
    with pytest.raises(ValueError, match="Unknown configuration key 'setpont'"):
        load_config(path)


def test_a_retired_key_says_what_replaced_it(tmp_path):
    path = write(tmp_path, BASE + "control:\n  output_mode: resistance\n")
    with pytest.raises(ValueError, match="writes every output entity"):
        load_config(path)


def test_the_outdoor_temperature_must_come_from_somewhere(tmp_path):
    path = write(tmp_path, "entities:\n  indoor_temp: sensor.a\n")
    with pytest.raises(ValueError, match="outdoor temperature has to come from"):
        load_config(path)


def test_missing_required_entity(tmp_path):
    path = write(tmp_path, "control:\n  setpoint: 21\n")
    with pytest.raises(ValueError, match="indoor_temp"):
        load_config(path)


@pytest.mark.parametrize(
    "section,changes,message",
    [
        ("control", {"offset_min": 5.0, "offset_max": -5.0}, "offset_min"),
        ("control", {"comfort_below": 5.0}, "comfort_below"),
        ("control", {"comfort_above": 9.0}, "comfort_above"),
        ("optimizer", {"elites": 999}, "elites"),
    ],
)
def test_validation_rejects_inconsistent_settings(section, changes, message):
    cfg = Config()
    cfg.entities.indoor_temp = "sensor.a"
    cfg.entities.outdoor_temp = "sensor.b"
    for key, value in changes.items():
        setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError, match=message):
        cfg.validate()
