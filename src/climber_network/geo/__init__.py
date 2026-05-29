"""climber_network.geo — Pure geocoding logic (no graph, no DB).

Heuristic city extraction from IFSC event names, a GeoNames-backed
city → coordinate index, and timezone / UTC-offset helpers.
"""

from __future__ import annotations

from climber_network.geo.geocode import (
    GeoNamesIndex,
    GeoPoint,
    extract_city,
    tz_for,
    utc_offset_hours,
)

__all__ = [
    "GeoNamesIndex",
    "GeoPoint",
    "extract_city",
    "tz_for",
    "utc_offset_hours",
]
