"""Training orchestration, backtest and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hpmpc.cli import build_parser, main
from hpmpc.config import Config
from hpmpc.evaluate import backtest, format_backtest
from hpmpc.model.thermal import load_params, save_params
from hpmpc.simulator import excitation_offsets, make_demo_dataset, synthetic_prices, synthetic_weather
from hpmpc.train import apply_pump_overrides, load_model, summarise, train, write_report


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("trained")
    cfg = Config()
    cfg.entities.indoor_temp = "sensor.indoor"
    cfg.entities.outdoor_temp = "sensor.outdoor"
    cfg.entities.price = "sensor.price"
    cfg.paths.data_dir = str(workdir)
    cfg.paths.model_dir = str(workdir)
    cfg.paths.state_file = str(workdir / "state.json")
    cfg.training.max_windows = 100
    cfg.training.restarts = 1
    cfg.training.long_window_hours = 36.0
    cfg.control.horizon_hours = 24.0
    cfg.control.cycle_minutes = 60   # keeps the backtest tests quick
    cfg.optimizer.population = 64
    cfg.optimizer.elites = 10
    cfg.optimizer.iterations = 5
    cfg.validate()
    frame, truth = make_demo_dataset(cfg, days=12, seed=11)
    report = train(cfg, frame)
    return cfg, frame, truth, report


# ---------------------------------------------------------------- simulator


def test_synthetic_weather_is_varied_and_finite():
    import pandas as pd

    index = pd.date_range("2026-01-05", periods=480, freq="15min", tz="UTC")
    weather = synthetic_weather(index, np.random.default_rng(0))
    assert np.isfinite(weather.to_numpy()).all()
    assert weather["t_outdoor"].std() > 0.5
    assert weather["cloud"].between(0, 100).all()
    assert (weather["wind"] >= 0).all()


def test_synthetic_prices_are_hourly_steps_and_positive():
    import pandas as pd

    index = pd.date_range("2026-01-05", periods=192, freq="15min", tz="UTC")
    price = synthetic_prices(index, np.random.default_rng(0))
    assert (price > 0).all()
    assert price.iloc[0] == price.iloc[3]      # constant within the hour
    assert price.nunique() > 5


def test_excitation_holds_then_changes():
    import pandas as pd

    index = pd.date_range("2026-01-05", periods=96, freq="15min", tz="UTC")
    offsets = excitation_offsets(index, np.random.default_rng(0), hold_hours=6.0)
    assert len(offsets) == 96
    assert offsets[0] == offsets[23]           # 6 h hold at 15 min steps
    assert offsets[0] != offsets[24]


# ----------------------------------------------------------------- training


def test_training_produces_a_usable_model_file(trained):
    cfg, _, _, report = trained
    assert Path(cfg.model_path).exists()
    params, metadata = load_params(cfg.model_path)
    assert params.heat_loss_w_per_k() > 0
    assert metadata["trained_at"]
    assert set(report) >= {"thermal", "heating_curve", "cop", "parameters", "pump_overrides", "data"}


def test_learned_pump_settings_are_reapplied_on_load(trained):
    cfg, _, _, report = trained
    working, params, residual, metadata = load_model(cfg)
    overrides = report["pump_overrides"]
    assert overrides  # the curve and COP fits both succeed on synthetic data
    for key, value in overrides.items():
        assert getattr(working.heat_pump, key) == pytest.approx(value)
    # The original config object is left untouched.
    assert cfg.heat_pump.curve_slope == Config().heat_pump.curve_slope


def test_apply_pump_overrides_ignores_unknown_keys():
    cfg = Config()
    updated = apply_pump_overrides(cfg, {"curve_slope": 0.5, "not_a_setting": 3})
    assert updated.heat_pump.curve_slope == 0.5


def test_training_summary_mentions_the_key_numbers(trained):
    text = summarise(trained[3])
    for expected in ("Data:", "Curve:", "Building:", "validation RMSE"):
        assert expected in text


def test_training_report_is_json_serialisable(trained, tmp_path):
    path = tmp_path / "report.json"
    write_report(path, trained[3])
    assert json.loads(path.read_text())["thermal"]["validation"]["rmse_c"] >= 0


def test_retraining_warm_starts_from_the_previous_model(trained):
    cfg, frame, _, _ = trained
    before, _ = load_params(cfg.model_path)
    report = train(cfg, frame)
    after, _ = load_params(cfg.model_path)
    assert report["thermal"]["validation"]["rmse_c"] <= 0.5
    assert after.heat_loss_w_per_k() == pytest.approx(before.heat_loss_w_per_k(), rel=0.3)


def test_model_params_roundtrip_through_disk(tmp_path, trained):
    params, _ = load_params(trained[0].model_path)
    path = tmp_path / "copy.json"
    save_params(path, params, {"note": "copy"})
    restored, metadata = load_params(path)
    assert restored.to_dict() == params.to_dict()
    assert metadata["note"] == "copy"


# ----------------------------------------------------------------- backtest


def test_backtest_beats_a_constant_offset(trained):
    cfg, frame, truth, _ = trained
    working, params, residual, _ = load_model(cfg)
    result = backtest(working, params, frame, days=2.0, residual=residual)
    assert result["period"]["cycles"] > 5
    # Same average indoor temperature, and the plan must not be worse.
    assert abs(result["mpc"]["indoor_mean"] - result["baseline_matched"]["indoor_mean"]) < 0.25
    assert result["saving_sek"] >= -0.05
    assert result["mpc"]["kelvin_hours_outside_comfort"] <= result["baseline_matched"]["kelvin_hours_outside_comfort"] + 0.5


def test_backtest_output_is_formatted_and_carries_caveats(trained):
    cfg, frame, _, _ = trained
    working, params, _, _ = load_model(cfg)
    result = backtest(working, params, frame, days=1.0)
    text = format_backtest(result)
    assert "Saving:" in text and "Caveats:" in text
    assert len(result["caveats"]) >= 3


def test_backtest_needs_enough_data(trained):
    cfg, frame, _, _ = trained
    working, params, _, _ = load_model(cfg)
    with pytest.raises(ValueError, match="Not enough data"):
        backtest(working, params, frame.iloc[:8], days=1.0)


# ---------------------------------------------------------------------- CLI


def test_parser_exposes_every_command():
    parser = build_parser()
    for command in ("init-config", "check", "collect", "train", "plan", "run", "excite", "backtest", "serve", "demo"):
        assert parser.parse_args([command]).command == command


def test_init_config_writes_a_loadable_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "token")
    target = tmp_path / "config.yaml"
    assert main(["--config", str(target), "init-config"]) == 0
    from hpmpc.config import load_config

    loaded = load_config(target)
    assert loaded.control.output_mode == "resistance"
    assert loaded.heat_pump.model == "daikin_erlq016caw1"
    assert loaded.forecast.price_area == "SE3"
    # A second call must not silently clobber an edited config.
    assert main(["--config", str(target), "init-config"]) == 1


def test_train_without_a_dataset_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "token")
    config = tmp_path / "config.yaml"
    main(["--config", str(config), "init-config"])
    text = config.read_text().replace("data_dir: data", f"data_dir: {tmp_path}")
    config.write_text(text)
    assert main(["--config", str(config), "train"]) == 1


# ------------------------------------------------------------ CLI helpers


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "token")
    path = tmp_path / "config.yaml"
    path.write_text(
        "entities: {indoor_temp: sensor.a, outdoor_temp: sensor.b}\n"
        "site: {latitude: 58.5877, longitude: 16.1924}\n"
        "heat_pump: {model: daikin_erlq016caw1}\n"
        f"paths: {{data_dir: {tmp_path}, model_dir: {tmp_path}}}\n",
        encoding="utf-8",
    )
    return path


def test_curve_converts_two_endpoints(config_file, capsys):
    assert main(["--config", str(config_file), "curve", "--point=-15:40", "--point=15:25"]) == 0
    out = capsys.readouterr().out
    assert "curve_slope: 0.500" in out
    assert "curve_offset: 22.50" in out
    # The printed table must reproduce the endpoints it was given.
    assert "      -15     40.0" in out
    assert "       15     25.0" in out


def test_curve_rejects_a_single_point(config_file, capsys):
    assert main(["--config", str(config_file), "curve", "--point=-15:40"]) == 1
    assert "exactly two points" in capsys.readouterr().out


def test_curve_rejects_unparseable_input(config_file, capsys):
    assert main(["--config", str(config_file), "curve", "--point=nonsense"]) == 1
    assert "Could not parse" in capsys.readouterr().out


def test_calibrate_ntc_recovers_a_known_thermistor(config_file, capsys):
    # Points generated from a 20 kohm / B=3950 part.
    assert main([
        "--config", str(config_file), "calibrate-ntc",
        "--point=0:67300", "--point=25:20000",
    ]) == 0
    out = capsys.readouterr().out
    assert "R25 = 20000" in out
    assert "B = 39" in out
    assert "ntc:" in out


def test_calibrate_ntc_needs_two_points(config_file, capsys):
    assert main(["--config", str(config_file), "calibrate-ntc", "--point=0:67300"]) == 1
    assert "at least two" in capsys.readouterr().out


def test_pump_table_prints_cop_and_capacity(config_file, capsys):
    assert main(["--config", str(config_file), "pump-table"]) == 0
    out = capsys.readouterr().out
    assert "ERLQ016CAW1" in out
    assert "COP" in out and "Compressor capacity (kW)" in out
    assert "backup heater: enabled" in out


def test_pump_table_says_so_when_no_map_is_configured(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HA_TOKEN", "token")
    path = tmp_path / "c.yaml"
    path.write_text("entities: {indoor_temp: sensor.a, outdoor_temp: sensor.b}\n", encoding="utf-8")
    assert main(["--config", str(path), "pump-table"]) == 1
    assert "cannot see the electric" in capsys.readouterr().out


def test_ntc_table_flags_coarse_resolution(config_file, capsys):
    assert main(["--config", str(config_file), "ntc-table", "--step-ohm", "20000"]) == 0
    assert "coarse" in capsys.readouterr().out
