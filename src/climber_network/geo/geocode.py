"""climber_network.geo.geocode — City extraction & GeoNames lookup.

Pure logic, no graph and no database. Three concerns:

1. :func:`extract_city` — heuristically pull a city name out of an
   IFSC-style event name (e.g. ``"IFSC World Cup Innsbruck 2023"`` →
   ``"Innsbruck"``).
2. :class:`GeoNamesIndex` — an in-memory ``(city, country)`` → coordinate
   index built either from the GeoNames ``cities1000`` dump or from explicit
   records (handy for tests).
3. :func:`tz_for` / :func:`utc_offset_hours` — IANA timezone resolution from
   coordinates and DST-aware UTC offsets.

GeoNames data file
------------------
The real index is built from the GeoNames ``cities1000`` export, a
tab-separated file with all cities of population >= 1000.

* Expected local path (gitignored): ``data/geonames/cities1000.txt``
* Download:  https://download.geonames.org/export/dump/cities1000.zip
  (unzip ``cities1000.zip`` → ``cities1000.txt``)

The TSV has no header. The columns this module reads are::

    0   geonameid     integer id
    1   name          UTF-8 name
    2   asciiname     ASCII name
    4   latitude      WGS84 degrees
    5   longitude     WGS84 degrees
    8   country code  ISO 3166-1 alpha-2
    17  timezone      IANA timezone id

Note GeoNames country codes are alpha-2; this module normalises both the
file's alpha-2 codes and any caller-supplied alpha-3 codes to a common
upper-case key, so lookups work with either form.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

# ---------------------------------------------------------------------------
# City extraction
# ---------------------------------------------------------------------------

# Series / discipline words and other noise stripped from event names. All
# matching is case-insensitive (we lower-case tokens before comparison).
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "ifsc",
        "uiaa",
        "climbing",
        "world",
        "cup",
        "championship",
        "championships",
        "boulder",
        "bouldering",
        "lead",
        "speed",
        "combined",
        "paraclimbing",
        "para",
        "youth",
        "open",
        "european",
        "asian",
        "panamerican",
        "pan-american",
        "oceania",
        "african",
        "continental",
        "and",
        "amp",  # leftover from "&amp;"
    }
)


def _is_year(token: str) -> bool:
    """True if *token* looks like a 4-digit calendar year (19xx / 20xx)."""
    return len(token) == 4 and token.isdigit() and token[:2] in {"19", "20"}


def extract_city(event_name: str, country_iso3: str | None) -> str | None:
    """Heuristically extract a city name from an IFSC-style *event_name*.

    Strips series words (IFSC / World Cup / World Championship / Boulder /
    Lead / Speed / Combined / ...) and any 4-digit year, returning the
    remaining run of tokens as the city.

    >>> extract_city("IFSC World Cup Innsbruck 2023", "AUT")
    'Innsbruck'
    >>> extract_city("IFSC Climbing World Championships Bern 2023", "CHE")
    'Bern'
    >>> extract_city("IFSC - Climbing World Cup (B) - Salt Lake City (USA) 2024", "USA")
    'Salt Lake City'

    The *country_iso3* argument is accepted for signature symmetry with the
    rest of the geo pipeline (and possible future disambiguation); the current
    heuristic does not use it. Returns ``None`` if nothing remains after
    stripping noise.
    """
    if not event_name:
        return None

    # Normalise separators: drop parenthesised groups (often country codes or
    # discipline tags) and treat punctuation as token separators.
    cleaned: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in event_name:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if depth > 0:
            continue
        if ch.isalnum() or ch in {"-", "'", "."}:
            buf.append(ch)
        else:
            if buf:
                cleaned.append("".join(buf))
                buf = []
    if buf:
        cleaned.append("".join(buf))

    kept: list[str] = []
    for tok in cleaned:
        stripped = tok.strip("-.'")
        if not stripped:
            continue
        if _is_year(stripped):
            continue
        if stripped.lower() in _STOP_WORDS:
            continue
        # Single stray separator-only tokens (e.g. "-") are skipped above.
        kept.append(stripped)

    if not kept:
        return None
    return " ".join(kept)


# ---------------------------------------------------------------------------
# GeoNames index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoPoint:
    """A resolved city location."""

    lat: float
    lon: float
    geonameid: int
    name: str
    timezone: str


# GeoNames cities1000 column indices (0-based, tab-separated, no header).
_COL_GEONAMEID = 0
_COL_NAME = 1
_COL_ASCIINAME = 2
_COL_LAT = 4
_COL_LON = 5
_COL_COUNTRY = 8
_COL_TIMEZONE = 17
_MIN_COLUMNS = _COL_TIMEZONE + 1


def _norm_city(name: str) -> str:
    """Normalise a city name for indexing: ASCII-fold, lower-case, trim.

    Accents are stripped (``"Zürich"`` → ``"zurich"``) so that lookups are
    robust to the spelling variant the caller happens to have.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return ascii_only.casefold().strip()


def _norm_country(iso: str | None) -> str | None:
    """Normalise a country code to an upper-case key, or ``None``.

    Accepts both ISO 3166-1 alpha-2 (GeoNames file form) and alpha-3
    (caller form); both are simply upper-cased. ``alpha-2`` keys and
    ``alpha-3`` keys live in the same dict, so a lookup only matches when the
    caller's code form matches what was indexed.
    """
    if not iso:
        return None
    return iso.strip().upper()


class GeoNamesIndex:
    """In-memory ``(city, country)`` → :class:`GeoPoint` index.

    Build one with :meth:`from_tsv` (the real GeoNames dump) or
    :meth:`from_records` (explicit rows, used in tests). Lookups are
    accent-insensitive and case-insensitive on the city name.

    When multiple entries share the same ``(city, country)`` key, the first
    one inserted wins (GeoNames lists the more prominent place first within a
    country, which is the desired behaviour for ambiguous names).
    """

    def __init__(self) -> None:
        self._by_city_country: dict[tuple[str, str], GeoPoint] = {}
        self._by_city: dict[str, GeoPoint] = {}

    # -- construction -------------------------------------------------------

    def _add(self, point: GeoPoint, country: str | None) -> None:
        city_key = _norm_city(point.name)
        if city_key and city_key not in self._by_city:
            self._by_city[city_key] = point
        country_key = _norm_country(country)
        if city_key and country_key is not None:
            key = (city_key, country_key)
            if key not in self._by_city_country:
                self._by_city_country[key] = point

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, object]],
    ) -> GeoNamesIndex:
        """Build an index from a list of explicit record dicts.

        Each record must provide the keys ``geonameid``, ``name``, ``lat``,
        ``lon``, ``country`` and ``timezone``. ``country`` may be alpha-2 or
        alpha-3 — whatever form callers will later pass to :meth:`lookup`.

        >>> idx = GeoNamesIndex.from_records(
        ...     [
        ...         {
        ...             "geonameid": 2775220,
        ...             "name": "Innsbruck",
        ...             "lat": 47.26266,
        ...             "lon": 11.39454,
        ...             "country": "AUT",
        ...             "timezone": "Europe/Vienna",
        ...         }
        ...     ]
        ... )
        >>> idx.lookup("Innsbruck", "AUT").geonameid
        2775220
        """
        index = cls()
        for rec in records:
            point = GeoPoint(
                lat=float(cast(float, rec["lat"])),
                lon=float(cast(float, rec["lon"])),
                geonameid=int(cast(int, rec["geonameid"])),
                name=str(rec["name"]),
                timezone=str(rec["timezone"]),
            )
            country = rec.get("country")
            index._add(point, None if country is None else str(country))
        return index

    @classmethod
    def from_tsv(cls, path: str | Path) -> GeoNamesIndex:
        """Build an index from a GeoNames ``cities1000`` TSV file.

        See the module docstring for the file's provenance and column layout.
        Rows that are short, blank, or have unparseable coordinates are
        skipped rather than raising. Country codes in this file are alpha-2.
        """
        index = cls()
        with Path(path).open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < _MIN_COLUMNS:
                    continue
                try:
                    lat = float(cols[_COL_LAT])
                    lon = float(cols[_COL_LON])
                    geonameid = int(cols[_COL_GEONAMEID])
                except ValueError:
                    continue
                name = cols[_COL_NAME] or cols[_COL_ASCIINAME]
                timezone = cols[_COL_TIMEZONE]
                country = cols[_COL_COUNTRY]
                point = GeoPoint(
                    lat=lat,
                    lon=lon,
                    geonameid=geonameid,
                    name=name,
                    timezone=timezone,
                )
                index._add(point, country)
        return index

    # -- query --------------------------------------------------------------

    def lookup(self, city: str, country_iso3: str | None) -> GeoPoint | None:
        """Return the :class:`GeoPoint` for *city* (optionally within a country).

        Matching is accent- and case-insensitive on the city name. If
        *country_iso3* is given, the ``(city, country)`` index is tried first;
        on a miss (or when no country is supplied) it falls back to a
        city-only match. Returns ``None`` if nothing matches.
        """
        city_key = _norm_city(city)
        if not city_key:
            return None
        country_key = _norm_country(country_iso3)
        if country_key is not None:
            hit = self._by_city_country.get((city_key, country_key))
            if hit is not None:
                return hit
        return self._by_city.get(city_key)


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

# A single TimezoneFinder instance is reused: it loads a sizeable lookup table
# on construction, so we build it lazily and cache it at module level.
_TZ_FINDER: TimezoneFinder | None = None


def _finder() -> TimezoneFinder:
    global _TZ_FINDER
    if _TZ_FINDER is None:
        _TZ_FINDER = TimezoneFinder()
    return _TZ_FINDER


def tz_for(lat: float, lon: float) -> str | None:
    """Return the IANA timezone id for a coordinate, or ``None`` if unknown.

    >>> tz_for(47.26266, 11.39454)
    'Europe/Vienna'
    """
    return _finder().timezone_at(lat=lat, lng=lon)


def utc_offset_hours(iana_tz: str, on_date: date) -> float:
    """Return the UTC offset in hours for *iana_tz* on *on_date* (DST-aware).

    Uses stdlib :mod:`zoneinfo`, so the offset reflects daylight-saving rules
    in effect on the given date. The offset is evaluated at local noon to
    avoid ambiguous/imaginary times around DST transitions at midnight.

    >>> utc_offset_hours("Europe/Vienna", date(2023, 7, 1))  # CEST = UTC+2
    2.0
    >>> utc_offset_hours("Europe/Vienna", date(2023, 1, 1))  # CET  = UTC+1
    1.0
    """
    tzinfo = ZoneInfo(iana_tz)
    local_noon = datetime(on_date.year, on_date.month, on_date.day, 12, tzinfo=tzinfo)
    offset = local_noon.utcoffset()
    if offset is None:
        return 0.0
    return offset.total_seconds() / 3600.0
