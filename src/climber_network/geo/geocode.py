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

import re
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
        "worldcup",  # older events spell "Worldcup" as one word
        "cup",
        "series",
        "championship",
        "championships",
        "masters",
        "invitational",
        "boulder",
        "bouldering",
        "lead",
        "speed",
        "combined",
        "paraclimbing",
        "para",
        "youth",
        "open",
        "continental",
        # Region qualifiers that prefix a host city.
        "europe",
        "european",
        "asia",
        "asian",
        "oceania",
        "oceanian",
        "americas",
        "american",
        "panamerican",
        "pan-american",
        "africa",
        "african",
        # Group qualifiers, e.g. "Lead Group A Paris" → "Paris".
        "group",
        "a",
        "b",
        "and",
        "amp",  # leftover from "&amp;"
    }
)

# Map IOC 3-letter codes (used by IFSC in parens, e.g. "Chamonix (FRA)") to
# ISO 3166-1 alpha-2 codes. The GeoNames cities1000 file is keyed on alpha-2,
# so this is the form we constrain lookups by. Note IOC codes differ from ISO
# alpha-3 (e.g. SUI≠CHE, GER≠DEU, SLO≠SVN), which is exactly why this map is
# needed. Covers every distinct parens code present in the event data plus a
# margin of likely future hosts.
IOC_TO_ALPHA2: dict[str, str] = {
    "FRA": "FR",
    "ITA": "IT",
    "CHN": "CN",
    "AUT": "AT",
    "JPN": "JP",
    "SLO": "SI",
    "SUI": "CH",
    "KOR": "KR",
    "RUS": "RU",
    "GER": "DE",
    "BEL": "BE",
    "USA": "US",
    "GBR": "GB",
    "ESP": "ES",
    "NOR": "NO",
    "CAN": "CA",
    "IND": "IN",
    "INA": "ID",
    "NED": "NL",
    "CZE": "CZ",
    "POL": "PL",
    "CHI": "CL",  # IOC CHI = Chile (ISO CL); ISO alpha-3 CHL.
    "BRA": "BR",
    "SRB": "RS",
    "SWE": "SE",
    "FIN": "FI",
    "SVK": "SK",
    "IRI": "IR",
    "TPE": "TW",
    "HKG": "HK",
    "AUS": "AU",
    "RSA": "ZA",
}

#: Inverse of the relevant slice of the IOC→ISO map: alpha-2 → ISO 3166-1
#: alpha-3, used to label the ``Country`` node (whose ids are alpha-3).
_ALPHA2_TO_ALPHA3: dict[str, str] = {
    "FR": "FRA",
    "IT": "ITA",
    "CN": "CHN",
    "AT": "AUT",
    "JP": "JPN",
    "SI": "SVN",
    "CH": "CHE",
    "KR": "KOR",
    "RU": "RUS",
    "DE": "DEU",
    "BE": "BEL",
    "US": "USA",
    "GB": "GBR",
    "ES": "ESP",
    "NO": "NOR",
    "CA": "CAN",
    "IN": "IND",
    "ID": "IDN",
    "NL": "NLD",
    "CZ": "CZE",
    "PL": "POL",
    "CL": "CHL",
    "BR": "BRA",
    "RS": "SRB",
    "SE": "SWE",
    "FI": "FIN",
    "SK": "SVK",
    "IR": "IRN",
    "TW": "TWN",
    "HK": "HKG",
    "AU": "AUS",
    "ZA": "ZAF",
}

#: Matches a parenthesised IOC country code, e.g. "... Chamonix (FRA) 2022".
_PARENS_CODE_RE = re.compile(r"\(([A-Z]{3})\)")


def parse_ioc_alpha2(event_name: str) -> str | None:
    """Return the ISO alpha-2 host country from a parenthesised IOC code, if any.

    IFSC event names embed the host country as a 3-letter IOC code in
    parentheses (``"... Chamonix (FRA) 2022"``). We extract it and map IOC→ISO
    alpha-2 so the GeoNames lookup can be constrained.

    >>> parse_ioc_alpha2("IFSC - Climbing World Cup (L,S) - Chamonix (FRA) 2022")
    'FR'
    >>> parse_ioc_alpha2("IFSC World Cup Innsbruck 2025") is None
    True

    Only true country codes count: discipline tags like ``(B,L,S)`` never match
    the ``([A-Z]{3})`` shape, and an unknown 3-letter code returns ``None``.
    """
    for code in _PARENS_CODE_RE.findall(event_name):
        alpha2 = IOC_TO_ALPHA2.get(code)
        if alpha2 is not None:
            return alpha2
    return None


def alpha2_to_alpha3(alpha2: str | None) -> str | None:
    """Map an ISO alpha-2 code to alpha-3 (for the ``Country`` node id)."""
    if not alpha2:
        return None
    return _ALPHA2_TO_ALPHA3.get(alpha2.strip().upper())


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


#: Translation table folding curly/typographic apostrophes to ASCII "'".
_APOSTROPHE_FOLD = {ord(c): "'" for c in "‘’ʼ`´"}


def _norm_city(name: str) -> str:
    """Normalise a city name for indexing: ASCII-fold, lower-case, trim.

    Accents are stripped (``"Zürich"`` → ``"zurich"``) so that lookups are
    robust to the spelling variant the caller happens to have.
    """
    # Fold the various Unicode apostrophe/quote glyphs to a plain ASCII "'" so
    # that "Tai'an" (straight quote, event data) and "Tai’an" (curly quote,
    # GeoNames file) hash to the same key.
    name = name.translate(_APOSTROPHE_FOLD)
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return ascii_only.casefold().strip()


#: Public alias for the city-name normaliser, so callers (e.g. the sync layer's
#: country backfill) key their own maps consistently with the index.
norm_city = _norm_city


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
        #: Count of *distinct* countries a city name appears in, used to tell a
        #: genuinely unique city (safe to resolve without a country) from an
        #: ambiguous one (must not guess when the country is unknown).
        self._city_countries: dict[str, set[str]] = {}

    # -- construction -------------------------------------------------------

    def _add(self, point: GeoPoint, country: str | None) -> None:
        city_key = _norm_city(point.name)
        if city_key and city_key not in self._by_city:
            self._by_city[city_key] = point
        country_key = _norm_country(country)
        if city_key and country_key is not None:
            self._city_countries.setdefault(city_key, set()).add(country_key)
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

    def lookup(self, city: str, country: str | None) -> GeoPoint | None:
        """Return the :class:`GeoPoint` for *city*, optionally within a country.

        Matching is accent- and case-insensitive on the city name. *country* may
        be ISO alpha-2 (the GeoNames file form, as produced by
        :func:`parse_ioc_alpha2`) or alpha-3 — it just needs to match whatever
        form the index was built with.

        Resolution order:

        1. If a *country* is supplied, the ``(city, country)`` index is tried
           first — this is what disambiguates names like *Madrid* (ES vs CO) or
           *Bali* (ID vs CM).
        2. On a country miss (or when no country is supplied) it falls back to a
           city-only match **only when the city name is genuinely unique** across
           all indexed countries. Ambiguous names with no usable country
           constraint return ``None`` rather than guessing the wrong place.

        Returns ``None`` if nothing matches.
        """
        city_key = _norm_city(city)
        if not city_key:
            return None
        country_key = _norm_country(country)
        if country_key is not None:
            hit = self._by_city_country.get((city_key, country_key))
            if hit is not None:
                return hit
        # Unconstrained fallback: safe only when the name is unambiguous.
        if len(self._city_countries.get(city_key, set())) <= 1:
            return self._by_city.get(city_key)
        return None


# ---------------------------------------------------------------------------
# Curated override map
# ---------------------------------------------------------------------------
#
# A small, committed, deterministic table that pins the cleaned-city names the
# pure GeoNames lookup gets wrong. Three flavours, all keyed by ``_norm_city``
# of the extracted city:
#
#   * redirect — the extracted name differs from GeoNames' canonical spelling
#     (Chamonix → "Chamonix-Mont-Blanc"; Villars → "Villars-sur-Ollon" in CH,
#     not the FR "Villars"; Brixen → "Bressanone"; "Comunidad de Madrid" →
#     "Madrid"; "Hachioji Tokyo" → "Hachioji"). Resolved through the index with
#     an explicit alpha-2 constraint.
#   * pin — the name is fine but we force the country (e.g. Navi Mumbai → IN) to
#     skip ambiguity.
#   * absent — the city is not in cities1000 at all (Wujiang, Keqiao). We supply
#     explicit coordinates + IANA timezone; ``geonameid`` is a 0 sentinel since
#     GeoNames has no id for it.


@dataclass(frozen=True)
class _Override:
    """A curated resolution. Either a redirect/pin (``canonical``+``alpha2``)
    or an absent-city literal (``point``)."""

    canonical: str | None = None
    alpha2: str | None = None
    point: GeoPoint | None = None


_CITY_OVERRIDES: dict[str, _Override] = {
    # Canonical-name mismatches (redirect into the index under the real name).
    _norm_city("Chamonix"): _Override(canonical="Chamonix-Mont-Blanc", alpha2="FR"),
    _norm_city("Villars"): _Override(canonical="Villars-sur-Ollon", alpha2="CH"),
    _norm_city("Brixen"): _Override(canonical="Bressanone", alpha2="IT"),
    # Region / administrative names that wrap a host city.
    _norm_city("Comunidad de Madrid"): _Override(canonical="Madrid", alpha2="ES"),
    # Composite names ("Hachioji, Tokyo" → the city of Hachioji).
    _norm_city("Hachioji Tokyo"): _Override(canonical="Hachioji", alpha2="JP"),
    # Country pins to skip ambiguity (the GeoNames name is already correct).
    # These IFSC hosts share their name with other places and never appear with
    # a parens code in the data, so we pin the right country explicitly.
    _norm_city("Navi Mumbai"): _Override(canonical="Navi Mumbai", alpha2="IN"),
    _norm_city("Prague"): _Override(canonical="Prague", alpha2="CZ"),
    _norm_city("Santiago"): _Override(canonical="Santiago", alpha2="CL"),
    # Cities absent from cities1000 — explicit coordinates + timezone.
    _norm_city("Wujiang"): _Override(
        alpha2="CN",
        point=GeoPoint(
            lat=31.1592,
            lon=120.6371,
            geonameid=0,
            name="Wujiang",
            timezone="Asia/Shanghai",
        ),
    ),
    _norm_city("Keqiao"): _Override(
        alpha2="CN",
        point=GeoPoint(
            lat=30.0813,
            lon=120.4889,
            geonameid=0,
            name="Keqiao",
            timezone="Asia/Shanghai",
        ),
    ),
    # The IFSC "Bali" World Cup is on the island of Bali (Denpasar area); there
    # is no cities1000 entry named "Bali" in Indonesia, so pin coordinates.
    _norm_city("Bali"): _Override(
        alpha2="ID",
        point=GeoPoint(
            lat=-8.65,
            lon=115.2167,
            geonameid=0,
            name="Bali",
            timezone="Asia/Makassar",
        ),
    ),
}


# ---------------------------------------------------------------------------
# End-to-end resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving an event name to a place.

    ``point`` is the resolved city (or ``None`` if unresolved). ``alpha2`` is
    the best-known host country (alpha-2) — set even on a city miss when the
    parens IOC code or backfill gave us one, so the caller can still anchor a
    country-level fallback.
    """

    point: GeoPoint | None
    alpha2: str | None


def override_alpha2(event_name: str) -> str | None:
    """Return the curated override's pinned host country for *event_name*, if any.

    Pure (no index): lets the sync layer settle the host country up front so the
    ``Country`` node is labelled even when the name had no parens code and no
    backfill match (e.g. "Comunidad de Madrid" → ES).
    """
    city = extract_city(event_name, None)
    if not city:
        return None
    override = _CITY_OVERRIDES.get(_norm_city(city))
    return override.alpha2 if override is not None else None


def resolve_event(
    event_name: str,
    geonames: GeoNamesIndex,
    *,
    alpha2: str | None = None,
) -> Resolution:
    """Resolve an IFSC event name to a city + host country (offline).

    Pipeline: extract the city → determine the host country (the explicit
    *alpha2* hint, else the parsed parens IOC code) → consult the curated
    override table → constrained GeoNames lookup. The *alpha2* hint lets the
    sync layer pass a backfilled country for events whose own name lacks a
    parens code (e.g. a bare "Innsbruck" inheriting AT from "Innsbruck (AUT)").

    Returns a :class:`Resolution`; ``point`` is ``None`` when no city matched.
    """
    host = alpha2 or parse_ioc_alpha2(event_name)
    city = extract_city(event_name, host)
    if not city:
        return Resolution(point=None, alpha2=host)

    override = _CITY_OVERRIDES.get(_norm_city(city))
    if override is not None:
        if override.point is not None:
            return Resolution(point=override.point, alpha2=host or override.alpha2)
        # redirect / pin: look up the canonical name under the pinned country.
        assert override.canonical is not None
        point = geonames.lookup(override.canonical, override.alpha2 or host)
        return Resolution(point=point, alpha2=host or override.alpha2)

    point = geonames.lookup(city, host)
    return Resolution(point=point, alpha2=host)


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
