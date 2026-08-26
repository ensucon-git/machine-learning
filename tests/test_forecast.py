from __future__ import annotations

import pandas as pd
import pytest

from hpmpc.forecast import build_forecast, horizon_index, parse_price_attributes, parse_weather_forecast


def test_horizon_index_is_aligned_and_the_right_length(cfg):
    now = pd.Timestamp("2026-01-15 12:07", tz="UTC").to_pydatetime()
    index = horizon_index(cfg, now)
    assert len(index) == int(cfg.control.horizon_hours * 60 / cfg.control.step_minutes)
    assert index[0].minute % cfg.control.step_minutes == 0
    assert index[0] <= pd.Timestamp(now)


def test_parses_nordpool_style_attributes():
    points = parse_price_attributes(
        {
            "raw_today": [{"start": "2026-01-15T00:00:00+01:00", "end": "x", "value": 0.42}],
            "raw_tomorrow": [{"start": "2026-01-16T00:00:00+01:00", "value": 0.31}],
        },
        None,
    )
    assert [round(v, 2) for _, v in points] == [0.42, 0.31]


def test_parses_entsoe_style_attributes():
    points = parse_price_attributes(
        {"prices_today": [{"time": "2026-01-15T00:00:00+00:00", "price": 1.25}]}, None
    )
    assert points[0][1] == pytest.approx(1.25)


def test_parses_a_bare_hourly_list():
    points = parse_price_attributes({"today": [0.1, 0.2, 0.3]}, None)
    assert len(points) == 3
    assert (points[1][0] - points[0][0]) == pd.Timedelta(hours=1)


def test_falls_back_to_the_current_price_when_no_forecast_exists():
    points = parse_price_attributes({}, 1.75)
    assert len(points) == 1 and points[0][1] == pytest.approx(1.75)


def test_weather_condition_becomes_cloud_cover_when_numeric_is_missing():
    frame = parse_weather_forecast(
        [
            {"datetime": "2026-01-15T10:00:00+00:00", "temperature": 2.0, "condition": "sunny"},
            {"datetime": "2026-01-15T11:00:00+00:00", "temperature": 3.0, "condition": "cloudy"},
        ]
    )
    assert frame["cloud"].tolist() == [0.0, 90.0]


def test_build_forecast_produces_a_complete_horizon(cfg, fake_ha):
    frame, sources = build_forecast(cfg, fake_ha, fake_ha.now)
    expected = int(cfg.control.horizon_hours * 60 / cfg.control.step_minutes)
    assert len(frame) == expected
    assert not frame[["t_outdoor", "wind", "solar_ghi", "price"]].isna().any().any()
    assert "weather forecast" in sources["t_outdoor"]
    assert frame["price"].nunique() > 1


def test_forecast_is_anchored_to_the_measured_outdoor_temperature(cfg, fake_ha):
    fake_ha.set("sensor.outdoor", -12.0)
    frame, sources = build_forecast(cfg, fake_ha, fake_ha.now)
    assert frame["t_outdoor"].iloc[0] == pytest.approx(-12.0, abs=0.05)
    assert "t_outdoor_bias_correction_c" in sources
    # The correction decays, so late horizon stays close to the raw forecast.
    assert frame["t_outdoor"].iloc[-1] > frame["t_outdoor"].iloc[0]


def test_missing_weather_entity_persists_current_values(cfg, fake_ha):
    cfg.entities.weather = ""
    frame, sources = build_forecast(cfg, fake_ha, fake_ha.now)
    assert frame["t_outdoor"].nunique() == 1
    assert "persisted" in sources["t_outdoor"]


def test_price_scale_and_addition_are_applied(cfg, fake_ha):
    cfg.control.price_scale = 0.01     # ore/kWh -> SEK/kWh
    cfg.control.price_addition = 0.5   # grid fee + tax
    frame, _ = build_forecast(cfg, fake_ha, fake_ha.now)
    assert frame["price"].min() >= 0.5
    assert frame["price"].max() < 1.0


def test_no_outdoor_data_at_all_is_an_error(cfg, fake_ha):
    cfg.entities.weather = ""
    fake_ha.drop("sensor.outdoor")
    with pytest.raises(ValueError, match="No outdoor temperature"):
        build_forecast(cfg, fake_ha, fake_ha.now)
