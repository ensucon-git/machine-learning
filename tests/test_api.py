from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hpmpc.api import create_app
from hpmpc.model.thermal import ThermalParams, save_params


@pytest.fixture
def app_client(tmp_path, monkeypatch, cfg, fake_ha):
    """Build the app against the fake Home Assistant and a trivial model."""
    import hpmpc.api as api_module

    save_params(cfg.model_path, ThermalParams(), {"trained_at": "2026-01-01T00:00:00+00:00"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
entities:
  indoor_temp: sensor.indoor
  outdoor_temp: sensor.outdoor
  price: sensor.price
  weather: weather.home
  offset_output: number.offset
control:
  horizon_hours: 12
  block_hours: 3
  warm_start_hours: 0
optimizer:
  population: 32
  elites: 6
  iterations: 3
  polish: false
paths:
  data_dir: {cfg.paths.data_dir}
  model_dir: {cfg.paths.model_dir}
  state_file: {cfg.paths.state_file}
server:
  api_key: test-key
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "HomeAssistant", lambda *a, **k: fake_ha)
    app = create_app(config_path, run_scheduler=False)
    with TestClient(app) as client:
        yield client, fake_ha


def test_health_is_open(app_client):
    client, _ = app_client
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["scheduler"] is False


def test_protected_routes_require_the_key(app_client):
    client, _ = app_client
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"X-API-Key": "test-key"}).status_code == 200


def test_plan_does_not_write_to_home_assistant(app_client):
    client, fake_ha = app_client
    payload = client.get("/plan", headers={"X-API-Key": "test-key"}).json()
    assert payload["applied"] is False
    assert fake_ha.written == []
    assert "mpc" in payload


def test_step_writes_and_updates_status(app_client):
    client, fake_ha = app_client
    payload = client.post("/step", headers={"X-API-Key": "test-key"}).json()
    assert payload["applied"] is True
    assert fake_ha.written[-1][0] == "number.offset"
    status = client.get("/status", headers={"X-API-Key": "test-key"}).json()
    assert status["report"]["offset"] == pytest.approx(payload["offset"])


def test_model_endpoint_exposes_the_parameters(app_client):
    client, _ = app_client
    payload = client.get("/model", headers={"X-API-Key": "test-key"}).json()
    assert payload["parameters"]["Ci"] > 0
    assert payload["heat_loss_w_per_k"] > 0


def test_metrics_are_prometheus_shaped(app_client):
    client, _ = app_client
    client.post("/step", headers={"X-API-Key": "test-key"})
    text = client.get("/metrics", headers={"X-API-Key": "test-key"}).text
    assert "hpmpc_offset_kelvin " in text
    for line in text.strip().splitlines():
        name, value = line.split()
        assert name.startswith("hpmpc_")
        float(value)
