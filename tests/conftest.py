from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hpmpc.config import Config
from hpmpc.ha import EntityState


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A small, fast configuration wired to fake entities."""
    config = Config()
    config.entities.indoor_temp = "sensor.indoor"
    config.entities.outdoor_temp = "sensor.outdoor"
    config.entities.price = "sensor.price"
    config.entities.wind_speed = "sensor.wind"
    config.entities.weather = "weather.home"
    config.entities.offset_output = "number.offset"
    config.entities.fake_temperature_output = "number.fake_temp"
    config.control.horizon_hours = 12.0
    config.control.block_hours = 3.0
    config.control.warm_start_hours = 0.0
    config.optimizer.population = 48
    config.optimizer.elites = 8
    config.optimizer.iterations = 4
    config.optimizer.polish = False
    config.paths.data_dir = str(tmp_path / "data")
    config.paths.model_dir = str(tmp_path / "models")
    config.paths.state_file = str(tmp_path / "state.json")
    config.validate()
    return config


class FakeHomeAssistant:
    """In-memory stand-in for :class:`hpmpc.ha.HomeAssistant`."""

    def __init__(self, states: dict[str, Any] | None = None, now: datetime | None = None) -> None:
        # Real wall-clock time by default: the controller measures sensor
        # staleness against the actual clock, so a fixed fake date would look
        # like months-old data and trip the fallback path in every test.
        self.now = now or datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self.written: list[tuple[str, float]] = []
        self.posted: list[tuple[str, dict[str, Any]]] = []
        self.history_frame = pd.DataFrame(columns=["entity_id", "time", "value"])
        self.forecast_hours = 48
        self.fail_write = False
        self._states: dict[str, EntityState] = {}
        defaults = {
            "sensor.indoor": 21.0,
            "sensor.outdoor": -5.0,
            "sensor.wind": 3.0,
            "number.offset": 0.0,
        }
        defaults.update(states or {})
        for entity_id, value in defaults.items():
            self.set(entity_id, value)
        self.set(
            "sensor.price",
            1.0,
            attributes={
                "raw_today": [
                    {
                        "start": (self.now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=h)).isoformat(),
                        "value": 0.4
                        + 1.6 * (1 if 6 <= (self.now + timedelta(hours=h)).hour < 9 else 0)
                        + 2.0 * (1 if 17 <= (self.now + timedelta(hours=h)).hour < 21 else 0),
                    }
                    for h in range(-12, 36)
                ]
            },
        )

    # -- helpers ---------------------------------------------------------
    def set(self, entity_id: str, value: Any, attributes: dict[str, Any] | None = None,
            age_minutes: float = 1.0) -> None:
        self._states[entity_id] = EntityState(
            entity_id=entity_id,
            state=str(value),
            attributes=attributes or {},
            last_updated=self.now - timedelta(minutes=age_minutes),
        )

    def drop(self, entity_id: str) -> None:
        self._states.pop(entity_id, None)

    # -- HomeAssistant interface ----------------------------------------
    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeHomeAssistant":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_state(self, entity_id: str) -> EntityState | None:
        return self._states.get(entity_id)

    def get_states(self, entity_ids) -> dict[str, EntityState]:
        return {e: self._states[e] for e in entity_ids if e in self._states}

    def history(self, entity_ids, start, end=None, chunk_days: int = 5) -> pd.DataFrame:
        return self.history_frame

    def set_number(self, entity_id: str, value: float) -> None:
        if self.fail_write:
            from hpmpc.ha import HomeAssistantError

            raise HomeAssistantError("write refused")
        self.written.append((entity_id, float(value)))

    def call_service(self, domain, service, data, return_response=False):
        return {}

    def weather_forecast(self, entity_id: str, forecast_type: str = "hourly") -> list[dict[str, Any]]:
        base = self.now.replace(minute=0, second=0, microsecond=0)
        return [
            {
                "datetime": (base + timedelta(hours=h)).isoformat(),
                "temperature": -5.0 + 3.0 * np.sin(2 * np.pi * h / 24.0),
                "wind_speed": 3.0,
                "cloud_coverage": 40.0,
            }
            for h in range(self.forecast_hours)
        ]

    def _request(self, method: str, url: str, **kwargs: Any):
        self.posted.append((url, kwargs.get("json", {})))
        return {}

    def publish_state(self, entity_id: str, state: Any,
                      attributes: dict[str, Any] | None = None) -> None:
        if self.fail_write:
            from hpmpc.ha import HomeAssistantError

            raise HomeAssistantError("publish refused")
        self._request("POST", f"/api/states/{entity_id}",
                      json={"state": state, "attributes": attributes or {}})
        # A published entity really does appear in the state machine, so the
        # fake has to start answering for it too.
        self.set(entity_id, state, attributes=attributes or {})


class RecorderHomeAssistant(FakeHomeAssistant):
    """A fake whose history really is a rolling window, like the real recorder.

    ``keep_days`` is Home Assistant's ``purge_keep_days``: readings older than
    that exist in ``history_frame`` but can no longer be queried.
    """

    def __init__(self, *args: Any, keep_days: float = 10.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.keep_days = keep_days
        self.requests: list[tuple[datetime, datetime]] = []

    def history(self, entity_ids, start, end=None, chunk_days: int = 5) -> pd.DataFrame:
        end = end or self.now
        self.requests.append((start, end))
        frame = self.history_frame
        if frame.empty:
            return frame
        purged = self.now - timedelta(days=self.keep_days)
        window = frame[
            (frame["time"] >= max(start, purged))
            & (frame["time"] <= end)
            & frame["entity_id"].isin([e for e in entity_ids if e])
        ]
        return window.reset_index(drop=True)


@pytest.fixture
def fake_ha() -> FakeHomeAssistant:
    return FakeHomeAssistant()


@pytest.fixture
def recorder_ha() -> RecorderHomeAssistant:
    return RecorderHomeAssistant()
