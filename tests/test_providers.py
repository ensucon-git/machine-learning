"""SMHI and spot-price providers, exercised against mocked HTTP responses."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from hpmpc.providers._http import ProviderError, get_json, read_cache, write_cache
from hpmpc.providers.elpris import (
    PriceUnavailable,
    fetch_prices,
    parse_day,
    price_url,
    tomorrow_is_published,
)
from hpmpc.providers.geocode import geocode
from hpmpc.providers.smhi import fetch_forecast, forecast_url, parse_forecast


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Keep the retry backoff out of the test run; the retry logic itself is
    still exercised, just without the wall-clock cost."""
    monkeypatch.setattr("hpmpc.providers._http.time.sleep", lambda _: None)


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------- SMHI


def smhi_payload(hours: int = 12, start: str = "2026-01-15T00:00:00Z") -> dict:
    base = pd.Timestamp(start)
    return {
        "approvedTime": start,
        "referenceTime": start,
        "timeSeries": [
            {
                "validTime": (base + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "parameters": [
                    {"name": "t", "values": [-5.0 + 0.5 * h]},
                    {"name": "ws", "values": [3.0]},
                    {"name": "tcc_mean", "values": [4]},
                    {"name": "r", "values": [85]},
                ],
            }
            for h in range(hours)
        ],
    }


def test_smhi_url_puts_longitude_first_with_six_decimals():
    url = forecast_url(58.5877, 16.1924)
    assert "/lon/16.192400/lat/58.587700/" in url


def test_smhi_parse_converts_octas_to_percent():
    frame = parse_forecast(smhi_payload(hours=2))
    assert list(frame.columns) == ["t_outdoor", "wind", "cloud", "humidity"]
    assert frame["cloud"].iloc[0] == pytest.approx(50.0)   # 4 octas of 8
    assert frame["humidity"].iloc[0] == pytest.approx(85.0)
    assert frame.index.tz is not None


def test_smhi_parse_rejects_an_empty_forecast():
    with pytest.raises(ProviderError, match="no time series"):
        parse_forecast({"timeSeries": []})


def test_smhi_parse_survives_missing_parameters():
    payload = {"timeSeries": [{"validTime": "2026-01-15T00:00:00Z", "parameters": [{"name": "t", "values": [2.0]}]}]}
    frame = parse_forecast(payload)
    assert frame["t_outdoor"].iloc[0] == 2.0
    assert pd.isna(frame["cloud"].iloc[0])


def test_smhi_fetch_uses_the_cache_on_the_second_call(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=smhi_payload())

    with client_for(handler) as client:
        first, meta = fetch_forecast(58.6, 16.2, cache_dir=str(tmp_path), client=client)
        assert meta["cache"] == "miss"
        second, meta2 = fetch_forecast(58.6, 16.2, cache_dir=str(tmp_path), client=client)
    assert meta2["cache"] == "fresh"
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_smhi_falls_back_to_a_stale_cache_when_the_service_is_down(tmp_path):
    write_cache(str(tmp_path), "smhi_58.6000_16.2000", smhi_payload())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with client_for(handler) as client:
        frame, meta = fetch_forecast(
            58.6, 16.2, cache_dir=str(tmp_path), cache_minutes=0.0, client=client
        )
    assert meta["cache"] == "stale"
    assert "error" in meta
    assert len(frame) == 12


def test_smhi_failure_without_a_cache_raises(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with client_for(handler) as client:
        with pytest.raises(ProviderError):
            fetch_forecast(58.6, 16.2, cache_dir=str(tmp_path), client=client)


# ------------------------------------------------------------------ price


def price_payload(day: str, base: float = 0.5) -> list[dict]:
    return [
        {
            "SEK_per_kWh": round(base + 0.1 * h, 5),
            "EUR_per_kWh": 0.04,
            "EXR": 11.5,
            "time_start": f"{day}T{h:02d}:00:00+01:00",
            "time_end": f"{day}T{h + 1:02d}:00:00+01:00",
        }
        for h in range(23)
    ]


def test_price_url_matches_the_documented_layout():
    assert price_url(datetime(2026, 1, 5).date(), "se3").endswith("/api/v1/prices/2026/01-05_SE3.json")


def test_price_parse_returns_utc_and_sorts():
    points = parse_day(price_payload("2026-01-15"))
    assert len(points) == 23
    assert points[0][0].tz is not None
    assert points == sorted(points)


def test_price_parse_ignores_malformed_entries():
    assert parse_day([{"SEK_per_kWh": 1.0}, "junk", {"time_start": "2026-01-15T00:00:00Z"}]) == []


@pytest.mark.parametrize("utc_hour,expected", [(6, False), (11, False), (12, True), (18, True)])
def test_publication_time_follows_swedish_local_time(utc_hour, expected):
    now = datetime(2026, 1, 15, utc_hour, tzinfo=timezone.utc)
    assert tomorrow_is_published(now) is expected


def test_fetch_prices_collects_yesterday_today_and_tomorrow(tmp_path):
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url).split("/")[-1])
        day = str(request.url).split("/")[-1].split("_")[0]
        return httpx.Response(200, json=price_payload(f"2026-{day}"))

    now = datetime(2026, 1, 15, 18, tzinfo=timezone.utc)
    with client_for(handler) as client:
        points, meta = fetch_prices("SE3", now=now, cache_dir=str(tmp_path), client=client)
    assert len(requested) == 3
    assert meta["area"] == "SE3"
    assert meta["tomorrow_published"] is True
    assert len(points) == 69
    assert "VAT" in meta["excludes"]


def test_tomorrow_missing_before_publication_is_not_an_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "01-16" in str(request.url):
            return httpx.Response(404)
        day = str(request.url).split("/")[-1].split("_")[0]
        return httpx.Response(200, json=price_payload(f"2026-{day}"))

    now = datetime(2026, 1, 15, 18, tzinfo=timezone.utc)
    with client_for(handler) as client:
        points, meta = fetch_prices("SE3", now=now, cache_dir=str(tmp_path), client=client)
    assert meta["tomorrow_published"] is False
    assert "warning" in meta            # past 13:00 and still missing: worth saying
    assert len(points) == 46


def test_no_prices_at_all_raises(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with client_for(handler) as client:
        with pytest.raises(PriceUnavailable):
            fetch_prices("SE3", now=datetime(2026, 1, 15, 8, tzinfo=timezone.utc),
                         cache_dir=str(tmp_path), client=client)


def test_unknown_bidding_area_is_rejected():
    with pytest.raises(ValueError, match="bidding area"):
        fetch_prices("DK1")


def test_prices_are_deduplicated_across_overlapping_days(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=price_payload("2026-01-15"))

    now = datetime(2026, 1, 15, 18, tzinfo=timezone.utc)
    with client_for(handler) as client:
        points, _ = fetch_prices("SE3", now=now, cache_dir=str(tmp_path), client=client)
    stamps = [t for t, _ in points]
    assert len(stamps) == len(set(stamps))


# --------------------------------------------------------------- plumbing


def test_get_json_retries_server_errors_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500) if attempts["n"] < 3 else httpx.Response(200, json={"ok": True})

    with client_for(handler) as client:
        assert get_json("https://example.invalid/x", client=client) == {"ok": True}
    assert attempts["n"] == 3


def test_get_json_does_not_retry_client_errors():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(403, text="denied")

    with client_for(handler) as client:
        with pytest.raises(ProviderError, match="403"):
            get_json("https://example.invalid/x", client=client)
    assert attempts["n"] == 1


def test_cache_expires(tmp_path):
    write_cache(str(tmp_path), "k", {"a": 1})
    assert read_cache(str(tmp_path), "k", 3600) == {"a": 1}
    assert read_cache(str(tmp_path), "k", 0.0) is None


def test_corrupt_cache_is_ignored(tmp_path):
    write_cache(str(tmp_path), "k", {"a": 1})
    path = tmp_path / "cache" / "k.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_cache(str(tmp_path), "k", 3600) is None


# --------------------------------------------------------------- geocode


def test_geocode_returns_coordinates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Falkv" in str(request.url)
        return httpx.Response(
            200,
            json=[{"display_name": "Falkvägen, Norrköping, Sverige", "lat": "58.58", "lon": "16.19", "type": "residential"}],
        )

    with client_for(handler) as client:
        results = geocode("Falkvägen, Norrköping", client=client)
    assert results[0]["latitude"] == pytest.approx(58.58)
    assert results[0]["longitude"] == pytest.approx(16.19)


def test_geocode_rejects_an_unexpected_response():
    with client_for(lambda r: httpx.Response(200, json={"error": "x"})) as client:
        with pytest.raises(ProviderError):
            geocode("nowhere", client=client)
