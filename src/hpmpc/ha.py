"""Minimal Home Assistant REST client.

Everything runs against the local instance over the LAN: history for training,
current states for feedback, ``weather.get_forecasts`` for the forecast, and
``number.set_value`` (or ``input_number``) to move the digital resistor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import httpx
import pandas as pd

from .config import HomeAssistantConfig

log = logging.getLogger(__name__)

UNAVAILABLE = {"unknown", "unavailable", "none", "", "null"}


class HomeAssistantError(RuntimeError):
    pass


@dataclass
class EntityState:
    entity_id: str
    state: str
    attributes: dict[str, Any]
    last_updated: datetime | None

    @property
    def numeric(self) -> float | None:
        return to_float(self.state)

    def age(self, now: datetime | None = None) -> timedelta | None:
        if self.last_updated is None:
            return None
        now = now or datetime.now(timezone.utc)
        return now - self.last_updated


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in UNAVAILABLE:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


class HomeAssistant:
    """Thin synchronous wrapper. The control loop runs every 10-15 minutes, so
    there is nothing to gain from async here."""

    def __init__(self, cfg: HomeAssistantConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        if not cfg.token:
            raise HomeAssistantError("No Home Assistant token configured (home_assistant.token)")
        self._client = client or httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"},
            timeout=cfg.timeout,
            verify=cfg.verify_ssl,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HomeAssistant":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- basics

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise HomeAssistantError(f"{method} {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise HomeAssistantError(f"{method} {url} -> HTTP {response.status_code}: {response.text[:300]}")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def ping(self) -> bool:
        payload = self._request("GET", "/api/")
        return isinstance(payload, dict) and "message" in payload

    def get_state(self, entity_id: str) -> EntityState | None:
        try:
            payload = self._request("GET", f"/api/states/{entity_id}")
        except HomeAssistantError as exc:
            log.warning("Could not read %s: %s", entity_id, exc)
            return None
        if not isinstance(payload, dict):
            return None
        return EntityState(
            entity_id=payload.get("entity_id", entity_id),
            state=payload.get("state", ""),
            attributes=payload.get("attributes", {}) or {},
            last_updated=_parse_ts(payload.get("last_updated") or payload.get("last_changed")),
        )

    def get_states(self, entity_ids: Iterable[str]) -> dict[str, EntityState]:
        out: dict[str, EntityState] = {}
        for entity_id in entity_ids:
            if not entity_id:
                continue
            state = self.get_state(entity_id)
            if state is not None:
                out[entity_id] = state
        return out

    def call_service(
        self, domain: str, service: str, data: dict[str, Any], return_response: bool = False
    ) -> Any:
        url = f"/api/services/{domain}/{service}"
        params = {"return_response": ""} if return_response else None
        return self._request("POST", url, json=data, params=params)

    # --------------------------------------------------------------- history

    def history(
        self,
        entity_ids: Sequence[str],
        start: datetime,
        end: datetime | None = None,
        chunk_days: int = 5,
    ) -> pd.DataFrame:
        """Fetch recorder history as a long dataframe.

        Columns: ``entity_id``, ``time`` (UTC) and ``value`` (float, non-numeric
        states dropped). Requests are chunked because a month of 1-minute data
        across a dozen entities is a large single response for a Raspberry Pi.
        """
        entity_ids = [e for e in entity_ids if e]
        if not entity_ids:
            return pd.DataFrame(columns=["entity_id", "time", "value"])
        end = end or datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            payload = self._request(
                "GET",
                f"/api/history/period/{cursor.astimezone(timezone.utc).isoformat()}",
                params={
                    "filter_entity_id": ",".join(entity_ids),
                    "end_time": chunk_end.astimezone(timezone.utc).isoformat(),
                    "minimal_response": "",
                    "no_attributes": "",
                    "significant_changes_only": "0",
                },
            )
            rows.extend(_flatten_history(payload))
            cursor = chunk_end
        if not rows:
            return pd.DataFrame(columns=["entity_id", "time", "value"])
        frame = pd.DataFrame(rows)
        frame["time"] = pd.to_datetime(frame["time"], utc=True, format="ISO8601", errors="coerce")
        frame = frame.dropna(subset=["time", "value"]).sort_values("time")
        return frame.reset_index(drop=True)

    # ---------------------------------------------------------------- output

    def set_number(self, entity_id: str, value: float) -> None:
        """Write a value to a ``number`` or ``input_number`` entity."""
        domain = entity_id.split(".", 1)[0]
        if domain not in {"number", "input_number"}:
            raise HomeAssistantError(
                f"'{entity_id}' is not a number/input_number entity; the controller writes numbers only"
            )
        self.call_service(domain, "set_value", {"entity_id": entity_id, "value": float(value)})

    def weather_forecast(self, entity_id: str, forecast_type: str = "hourly") -> list[dict[str, Any]]:
        """Return the hourly forecast for a weather entity.

        Uses the modern ``weather.get_forecasts`` service and falls back to the
        legacy ``forecast`` attribute on older Home Assistant versions.
        """
        if not entity_id:
            return []
        try:
            payload = self.call_service(
                "weather",
                "get_forecasts",
                {"entity_id": entity_id, "type": forecast_type},
                return_response=True,
            )
            response = (payload or {}).get("service_response", {})
            entry = response.get(entity_id) or (next(iter(response.values()), {}) if response else {})
            forecast = entry.get("forecast") or []
            if forecast:
                return list(forecast)
        except HomeAssistantError as exc:
            log.warning("weather.get_forecasts failed for %s: %s", entity_id, exc)
        state = self.get_state(entity_id)
        if state is not None:
            return list(state.attributes.get("forecast") or [])
        return []


def _flatten_history(payload: Any) -> list[dict[str, Any]]:
    """Turn Home Assistant's list-of-lists history payload into flat rows."""
    rows: list[dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows
    for series in payload:
        if not isinstance(series, list) or not series:
            continue
        entity_id = ""
        for point in series:
            if not isinstance(point, dict):
                continue
            # With minimal_response only the first point carries entity_id.
            entity_id = point.get("entity_id") or entity_id
            value = to_float(point.get("state"))
            if value is None or not entity_id:
                continue
            rows.append(
                {
                    "entity_id": entity_id,
                    "time": point.get("last_updated") or point.get("last_changed"),
                    "value": value,
                }
            )
    return rows
