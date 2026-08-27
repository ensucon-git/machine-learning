"""Shared HTTP plumbing: one client, sane retries, and an on-disk cache.

The control loop runs every 15 minutes but SMHI updates hourly and spot prices
change once a day, so caching is not an optimisation - it is basic courtesy to
a free public service, and it keeps the controller working through a brief
network outage.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "hpmpc/0.1 (self-hosted heat pump controller)"


class ProviderError(RuntimeError):
    pass


def get_json(
    url: str,
    timeout: float = 30.0,
    retries: int = 3,
    client: httpx.Client | None = None,
    headers: dict[str, str] | None = None,
    allow_404: bool = False,
) -> Any:
    """GET a JSON document, retrying transient failures with backoff."""
    own_client = client is None
    client = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
    try:
        last: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                response = client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code == 404 and allow_404:
                    return None
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ProviderError(f"{url} returned malformed JSON") from exc
                if response.status_code < 500:
                    raise ProviderError(f"{url} -> HTTP {response.status_code}: {response.text[:200]}")
                last = ProviderError(f"{url} -> HTTP {response.status_code}")
            if attempt < retries - 1:
                delay = 2.0 * (2**attempt)
                log.warning("%s failed (%s); retrying in %.0f s", url, last, delay)
                time.sleep(delay)
        raise ProviderError(f"{url} failed after {retries} attempts: {last}")
    finally:
        if own_client:
            client.close()


def cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / "cache" / f"{key}.json"


def read_cache(cache_dir: str | Path, key: str, max_age_seconds: float) -> Any | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age_seconds:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_cache(cache_dir: str | Path, key: str, payload: Any) -> None:
    path = cache_path(cache_dir, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except (OSError, TypeError) as exc:  # pragma: no cover - cache is best effort
        log.debug("Could not write cache %s: %s", path, exc)


def read_stale_cache(cache_dir: str | Path, key: str) -> Any | None:
    """Last resort: an out-of-date forecast beats no forecast at all."""
    return read_cache(cache_dir, key, max_age_seconds=float("inf"))
