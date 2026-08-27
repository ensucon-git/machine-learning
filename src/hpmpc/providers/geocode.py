"""One-shot address lookup, used by ``hpmpc geocode``.

Only ever run interactively to fill in ``site.latitude`` / ``site.longitude``
once. The control loop never geocodes: it would be a pointless dependency, and
SMHI's grid is coarse enough that the exact street does not matter.

Uses OpenStreetMap's Nominatim, whose usage policy requires a real User-Agent
and at most one request per second - both respected here.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._http import ProviderError, get_json

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode(query: str, limit: int = 5, timeout: float = 30.0, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Return candidate locations for a free-text address."""
    params = httpx.QueryParams({"q": query, "format": "json", "limit": str(int(limit)), "addressdetails": "1"})
    payload = get_json(f"{NOMINATIM_URL}?{params}", timeout=timeout, retries=2, client=client)
    if not isinstance(payload, list):
        raise ProviderError("Nominatim returned an unexpected response")
    return [
        {
            "display_name": item.get("display_name", ""),
            "latitude": round(float(item["lat"]), 6),
            "longitude": round(float(item["lon"]), 6),
            "type": item.get("type", ""),
        }
        for item in payload
        if "lat" in item and "lon" in item
    ]
