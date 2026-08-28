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


def test_no_outdoor_sensor_is_fine_when_smhi_supplies_the_forecast(tmp_path):
    """The forecast's first step stands in for the sensor, so SMHI is enough."""
    path = write(tmp_path, "entities:\n  indoor_temp: sensor.a\n"
                           "forecast:\n  weather_source: smhi\n")
    assert load_config(path).entities.outdoor_temp == ""


def test_a_home_assistant_weather_entity_also_counts(tmp_path):
    path = write(tmp_path, "entities:\n  indoor_temp: sensor.a\n  weather: weather.home\n"
                           "forecast:\n  weather_source: home_assistant\n")
    assert load_config(path).entities.weather == "weather.home"


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


def test_the_status_entity_is_one_hpmpc_publishes_itself(cfg):
    """It goes in through the states API, so nothing defines it first and a
    fresh install simply has not created it yet."""
    cfg.entities.status_entity = "sensor.hpmpc_status"
    assert "sensor.hpmpc_status" in cfg.entities.self_published()


def test_helpers_are_not_treated_as_self_published(cfg):
    cfg.entities.offset_output = "input_number.varmepump_offset"
    cfg.entities.status_entity = ""
    assert cfg.entities.self_published() == set()


def test_a_sensor_output_counts_as_self_published(cfg):
    cfg.entities.fake_temperature_output = "sensor.hpmpc_fake_outdoor"
    assert "sensor.hpmpc_fake_outdoor" in cfg.entities.self_published()
