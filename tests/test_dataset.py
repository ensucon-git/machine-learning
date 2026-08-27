from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpmpc.dataset import (
    add_derived,
    applied_offset,
    build_dataset,
    column_map,
    describe,
    load_dataset,
    pivot_history,
    save_dataset,
    segments,
)
from hpmpc.ntc import temperature_to_resistance


def long_history(entity_values: dict[str, float], hours: int = 6) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    for entity_id, value in entity_values.items():
        for i in range(hours * 12):  # every 5 minutes
            rows.append({"entity_id": entity_id, "time": start + pd.Timedelta(minutes=5 * i), "value": value})
    return pd.DataFrame(rows)


def test_pivot_resamples_and_names_columns(cfg):
    frame = pivot_history(
        long_history({"sensor.indoor": 21.0, "sensor.outdoor": -4.0}), column_map(cfg), 15
    )
    assert set(frame.columns) == {"t_indoor", "t_outdoor"}
    assert (frame.index[1] - frame.index[0]) == pd.Timedelta(minutes=15)
    assert frame["t_indoor"].iloc[0] == pytest.approx(21.0)


def test_stepwise_signals_are_forward_filled_not_interpolated(cfg):
    start = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    rows = [
        {"entity_id": "sensor.price", "time": start, "value": 1.0},
        {"entity_id": "sensor.price", "time": start + pd.Timedelta(hours=2), "value": 5.0},
        {"entity_id": "sensor.indoor", "time": start, "value": 21.0},
        {"entity_id": "sensor.indoor", "time": start + pd.Timedelta(hours=2), "value": 21.0},
    ]
    frame = pivot_history(pd.DataFrame(rows), column_map(cfg), 15)
    # Halfway through, a stepwise price must still be the old value.
    assert frame["price"].iloc[4] == pytest.approx(1.0)


def test_add_derived_builds_irradiance_when_no_pyranometer(cfg):
    index = pd.date_range("2026-06-21 00:00", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame({"t_indoor": 21.0, "t_outdoor": 15.0, "cloud": 0.0}, index=index)
    out = add_derived(frame, cfg)
    assert out["solar_ghi"].max() > 100.0
    assert out["solar_ghi"].min() == pytest.approx(0.0)


def test_add_derived_prefers_a_measured_pyranometer(cfg):
    index = pd.date_range("2026-06-21 00:00", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"t_indoor": 21.0, "t_outdoor": 15.0, "cloud": 0.0, "solar_radiation": 123.0}, index=index
    )
    assert add_derived(frame, cfg)["solar_ghi"].eq(123.0).all()


@pytest.mark.parametrize("column", ["output_offset", "output_fake_temp", "output_resistance"])
def test_applied_offset_is_recovered_from_whichever_output_was_logged(cfg, column):
    index = pd.date_range("2026-01-15", periods=3, freq="15min", tz="UTC")
    outdoor = np.array([-5.0, -6.0, -7.0])
    offset = np.array([-2.0, 1.0, 0.0])
    logged = {
        "output_offset": offset,
        "output_fake_temp": outdoor + offset,
        "output_resistance": temperature_to_resistance(outdoor + offset, cfg.ntc),
    }[column]
    frame = pd.DataFrame({"t_outdoor": outdoor, column: logged}, index=index)
    assert applied_offset(frame, cfg).to_numpy() == pytest.approx(offset, abs=1e-4)


def test_the_kelvin_entity_wins_when_several_were_logged(cfg):
    """It needs no conversion, so it cannot disagree with what was decided."""
    index = pd.date_range("2026-01-15", periods=2, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "t_outdoor": [-5.0, -5.0],
            "output_offset": [-2.0, -2.0],
            "output_fake_temp": [99.0, 99.0],      # deliberately inconsistent
        },
        index=index,
    )
    assert applied_offset(frame, cfg).to_numpy() == pytest.approx([-2.0, -2.0])


def test_applied_offset_defaults_to_zero_without_an_output_entity(cfg):
    index = pd.date_range("2026-01-15", periods=3, freq="15min", tz="UTC")
    frame = pd.DataFrame({"t_outdoor": [-5.0, -5.0, -5.0]}, index=index)
    assert applied_offset(frame, cfg).eq(0.0).all()


def test_segments_split_on_recorder_gaps():
    index = list(pd.date_range("2026-01-15 00:00", periods=20, freq="15min", tz="UTC"))
    index += list(pd.date_range("2026-01-16 00:00", periods=20, freq="15min", tz="UTC"))
    frame = pd.DataFrame({"t_indoor": 21.0}, index=pd.DatetimeIndex(index))
    pieces = segments(frame, 15)
    assert len(pieces) == 2
    assert all(len(p) == 20 for p in pieces)


def test_build_dataset_requires_the_core_signals(cfg, fake_ha):
    fake_ha.history_frame = long_history({"sensor.outdoor": -4.0})
    with pytest.raises(ValueError, match="missing required signals|No usable history"):
        build_dataset(cfg, fake_ha)


def test_dataset_roundtrips_through_disk(cfg, tmp_path):
    index = pd.date_range("2026-01-15", periods=8, freq="15min", tz="UTC", name="time")
    frame = pd.DataFrame({"t_indoor": 21.0, "t_outdoor": -3.0}, index=index)
    path = tmp_path / "history.csv.gz"
    save_dataset(frame, path)
    restored = load_dataset(path)
    assert restored.index.tz is not None
    pd.testing.assert_frame_equal(frame, restored, check_freq=False)


def test_describe_flags_missing_excitation():
    index = pd.date_range("2026-01-15", periods=100, freq="15min", tz="UTC")
    frame = pd.DataFrame({"t_indoor": 21.0, "t_outdoor": -3.0, "offset": 0.0}, index=index)
    info = describe(frame)
    assert info["rows"] == 100
    assert info["offset_excitation"]["std"] == 0.0
