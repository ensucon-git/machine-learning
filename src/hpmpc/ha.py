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

from urllib.parse import urlsplit

import httpx
import pandas as pd

from .config import HomeAssistantConfig

log = logging.getLogger(__name__)

UNAVAILABLE = {"unknown", "unavailable", "none", "", "null"}

# Binary entities report words, not numbers, and different integrations pick
# different words. Mapping them here means a charging sensor can be recorded
# and resampled exactly like any other signal.
BOOLEAN_STATES = {
    "on": 1.0, "true": 1.0, "yes": 1.0, "open": 1.0, "home": 1.0,
    "charging": 1.0, "heating": 1.0, "active": 1.0, "running": 1.0,
    "off": 0.0, "false": 0.0, "no": 0.0, "closed": 0.0, "not_home": 0.0,
    "not charging": 0.0, "not_charging": 0.0, "idle": 0.0, "disconnected": 0.0,
    "standby": 0.0, "stopped": 0.0,
}


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
    lowered = text.lower()
    if lowered in UNAVAILABLE:
        return None
    if lowered in BOOLEAN_STATES:
        return BOOLEAN_STATES[lowered]
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
        self.last_error: str | None = None
        if not cfg.token:
            raise HomeAssistantError(
                "No Home Assistant token configured. Set HA_TOKEN to a long-lived access token "
                "from your Home Assistant profile page (Security -> Long-lived access tokens). "
                "HPMPC_API_KEY is a different thing: it protects hpmpc's own HTTP API and is "
                "never sent to Home Assistant."
            )
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
        """Is Home Assistant answering? Never raises - see :meth:`diagnose`."""
        try:
            payload = self._request("GET", "/api/")
        except HomeAssistantError as exc:
            self.last_error = str(exc)
            return False
        self.last_error = None
        return isinstance(payload, dict) and "message" in payload

    def diagnose(self) -> str:
        """Turn a failed connection into something a person can act on.

        The underlying errors are accurate and useless: "[Errno -3] Temporary
        failure in name resolution" is a correct description of a DNS lookup
        that did not happen, and says nothing about the .local hostname in a
        Docker container that caused it.
        """
        error = getattr(self, "last_error", None) or ""
        lowered = error.lower()
        host = urlsplit(self.cfg.base_url).hostname or self.cfg.base_url
        lines = [f"Could not reach Home Assistant at {self.cfg.base_url}", f"  {error}"]

        if "name resolution" in lowered or "nodename nor servname" in lowered \
                or "getaddrinfo" in lowered or "name or service not known" in lowered:
            lines.append(f"\nThat is DNS: the name '{host}' did not resolve. It is not the token,")
            lines.append("and not the API key - neither has been used yet at this point.")
            if host.endswith(".local"):
                lines.append(
                    f"\n'{host}' is an mDNS name. Your laptop resolves those through Bonjour or\n"
                    "Avahi; a Docker container normally has neither, so the lookup fails inside\n"
                    "the container even though the same URL works from your browser.\n"
                    "\nUse the IP address instead - find it in Home Assistant under\n"
                    "Settings -> System -> Network:\n"
                    "\n  home_assistant:\n"
                    "    base_url: http://192.168.1.42:8123\n"
                    "\nA hostname from your router's DNS works too; only .local is the problem."
                )
            else:
                lines.append(
                    f"\nCheck the spelling of '{host}', and that whatever resolves it is reachable\n"
                    "from inside the container. An IP address sidesteps the question entirely."
                )
        elif "connection refused" in lowered:
            lines.append(
                f"\nThe name resolved but nothing answered on that port. Check the port (8123 by\n"
                f"default), and that Home Assistant is running on {host}."
            )
        elif "timed out" in lowered or "timeout" in lowered:
            lines.append(
                "\nThe connection hung rather than being refused, which usually means a firewall\n"
                "is dropping it, or the container is on a network that cannot reach that address."
            )
        elif "401" in error or "403" in error:
            lines.append(
                "\nHome Assistant answered but rejected the credentials. That is HA_TOKEN - a\n"
                "long-lived access token from your HA profile page, not HPMPC_API_KEY, which\n"
                "only protects hpmpc's own HTTP API and is never sent to Home Assistant."
            )
        elif "certificate" in lowered or "ssl" in lowered:
            lines.append(
                "\nTLS could not be verified. For a self-signed certificate on your own LAN, set\n"
                "home_assistant.verify_ssl: false - or use http:// over the LAN."
            )
        return "\n".join(lines)

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

    def publish_state(self, entity_id: str, state: Any,
                      attributes: dict[str, Any] | None = None) -> None:
        """Create or update an entity directly, without a helper to write into.

        Home Assistant's states API lets an outside program put an entity into
        the state machine. Nothing has to exist first - the entity appears on
        the first call - which is what makes this the easy way to get a reading
        out of hpmpc and onto a dashboard.

        The catch, and it matters for anything in the control path: an entity
        published this way lives only in memory. A Home Assistant restart
        forgets it until the next control cycle writes it again, and it has no
        unique_id, so it cannot be renamed or assigned to an area in the UI. A
        helper you define in YAML restores its value across a restart; this does
        not.
        """
        self._request(
            "POST",
            f"/api/states/{entity_id}",
            json={"state": state, "attributes": attributes or {}},
        )

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
