from __future__ import annotations

import numpy as np
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


def test_a_bare_full_day_list_is_read_at_its_own_resolution():
    quarter = parse_price_attributes({"today": [0.1] * 96}, None)
    assert (quarter[1][0] - quarter[0][0]) == pd.Timedelta(minutes=15)
    hourly = parse_price_attributes({"today": [0.1] * 24}, None)
    assert (hourly[1][0] - hourly[0][0]) == pd.Timedelta(hours=1)


def test_quarter_hourly_prices_reach_the_horizon(cfg, fake_ha, monkeypatch):
    """Nord Pool settles in 15-minute periods, which matches the control step."""
    import hpmpc.forecast as module

    start = pd.Timestamp("2026-01-15", tz="UTC")
    points = [(start + pd.Timedelta(minutes=15 * i), 0.2 + 0.01 * (i % 96)) for i in range(192)]
    monkeypatch.setattr(module, "fetch_prices", lambda *a, **k: (points, {"source": "quarter-hourly"}))
    cfg.forecast.price_source = "elprisetjustnu"
    frame, sources = build_forecast(cfg, fake_ha, pd.Timestamp("2026-01-15 06:00", tz="UTC").to_pydatetime())
    assert sources["price_resolution_minutes"] == 15
    # Every control step gets its own price, not a step-held hourly one.
    assert frame["spot_price"].nunique() >= len(frame) - 1


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
    assert "weather entity" in sources["weather"]["source"]
    assert frame["price"].nunique() > 1
    assert "spot_price" in frame


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


def test_price_scale_addition_and_vat_are_applied_in_order(cfg, fake_ha):
    cfg.control.price_scale = 0.01     # ore/kWh -> SEK/kWh
    cfg.control.price_addition = 0.5   # grid transfer + energy tax
    cfg.control.price_vat_pct = 25.0
    frame, _ = build_forecast(cfg, fake_ha, fake_ha.now)
    expected = (frame["spot_price"] * 0.01 + 0.5) * 1.25
    assert frame["price"].to_numpy() == pytest.approx(expected.to_numpy())
    assert frame["price"].min() >= 0.625


def test_no_outdoor_data_at_all_is_an_error(cfg, fake_ha):
    cfg.entities.weather = ""
    fake_ha.drop("sensor.outdoor")
    with pytest.raises(ValueError, match="No outdoor temperature"):
        build_forecast(cfg, fake_ha, fake_ha.now)


# ------------------------------------------------- provider-backed forecast


def _smhi_frame(hours: int = 48) -> pd.DataFrame:
    index = pd.date_range("2026-01-15", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {"t_outdoor": np.linspace(-8.0, 2.0, hours), "wind": 4.0, "cloud": 30.0, "humidity": 82.0},
        index=index,
    )


def test_smhi_and_spot_prices_are_used_when_configured(cfg, fake_ha, monkeypatch):
    import hpmpc.forecast as module

    now = pd.Timestamp("2026-01-15 06:00", tz="UTC").to_pydatetime()
    points = [
        (pd.Timestamp("2026-01-15", tz="UTC") + pd.Timedelta(hours=h), 0.2 + 0.05 * h) for h in range(48)
    ]
    monkeypatch.setattr(module, "fetch_forecast", lambda *a, **k: (_smhi_frame(), {"source": "SMHI test"}))
    monkeypatch.setattr(module, "fetch_prices", lambda *a, **k: (points, {"source": "spot test", "first": "x", "last": "y"}))

    cfg.forecast.weather_source = "smhi"
    cfg.forecast.price_source = "elprisetjustnu"
    cfg.site.latitude, cfg.site.longitude = 58.5877, 16.1924
    frame, sources = build_forecast(cfg, fake_ha, now)

    assert sources["weather"]["source"] == "SMHI test"
    assert sources["price"]["source"] == "spot test"
    assert frame["humidity"].notna().all()
    assert frame["price"].nunique() > 1


def test_smhi_failure_falls_back_to_the_weather_entity(cfg, fake_ha, monkeypatch):
    import hpmpc.forecast as module
    from hpmpc.providers._http import ProviderError

    def boom(*args, **kwargs):
        raise ProviderError("smhi down")

    monkeypatch.setattr(module, "fetch_forecast", boom)
    cfg.forecast.weather_source = "smhi"
    cfg.site.latitude, cfg.site.longitude = 58.5877, 16.1924
    frame, sources = build_forecast(cfg, fake_ha, fake_ha.now)

    assert "smhi down" in sources["weather_error"]
    assert "weather entity" in sources["weather"]["source"]
    assert frame["t_outdoor"].notna().all()


def test_price_failure_falls_back_to_the_price_entity(cfg, fake_ha, monkeypatch):
    import hpmpc.forecast as module
    from hpmpc.providers import PriceUnavailable

    def boom(*args, **kwargs):
        raise PriceUnavailable("no prices")

    monkeypatch.setattr(module, "fetch_prices", boom)
    cfg.forecast.price_source = "elprisetjustnu"
    frame, sources = build_forecast(cfg, fake_ha, fake_ha.now)

    assert "no prices" in sources["price_error"]
    assert "Home Assistant entity" in sources["price"]["source"]
    assert frame["price"].nunique() > 1


def test_humidity_falls_back_to_the_sensor_then_to_unknown(cfg, fake_ha):
    cfg.entities.outdoor_humidity = "sensor.humidity"
    fake_ha.set("sensor.humidity", 91.0)
    frame, _ = build_forecast(cfg, fake_ha, fake_ha.now)
    assert frame["humidity"].eq(91.0).all()

    cfg.entities.outdoor_humidity = ""
    frame, _ = build_forecast(cfg, fake_ha, fake_ha.now)
    assert frame["humidity"].isna().all()   # unknown, not guessed


def test_quarter_hourly_prices_are_recognised_as_such():
    """Nord Pool settles in quarter-hours: 96 prices a day, not 24."""
    from hpmpc.forecast import price_resolution

    start = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    points = [(start + pd.Timedelta(minutes=15 * i), 1.0) for i in range(96)]
    assert price_resolution(points) == pd.Timedelta(minutes=15)


def test_an_hourly_feed_still_parses():
    from hpmpc.forecast import price_resolution

    start = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    points = [(start + pd.Timedelta(hours=i), 1.0) for i in range(24)]
    assert price_resolution(points) == pd.Timedelta(hours=1)


def test_each_quarter_keeps_its_own_price_on_the_horizon(cfg, fake_ha):
    """The planning grid is 15 minutes, so a quarter-hourly price must survive
    onto it rather than being averaged into an hour."""
    from hpmpc.forecast import build_forecast

    now = fake_ha.now.replace(minute=0, second=0, microsecond=0)
    raw = []
    for i in range(-8, 200):
        stamp = now + pd.Timedelta(minutes=15 * i)
        raw.append({"start": stamp.isoformat(), "value": 1.0 if i % 2 == 0 else 5.0})
    fake_ha.set("sensor.price", 1.0, attributes={"raw_today": raw})
    cfg.forecast.price_source = "home_assistant"
    cfg.control.step_minutes = 15

    frame, sources = build_forecast(cfg, fake_ha, now)
    assert sources.get("price_resolution_minutes") == 15
    spot = frame["spot_price"].to_numpy()[:8]
    assert len(set(spot.round(3))) == 2, "the alternating quarters were flattened"
