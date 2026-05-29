"""sync.geo — P2b L2 Geography graph build: Event names → places in Neo4j.

Reads the ``Event`` and ``Athlete`` nodes already present in Neo4j (from the P1
L1 mirror) and idempotently MERGEs the L2 geography graph on top of them:

Nodes
    Venue, City, Country, TimeZone

Edges
    (Event)-[:HELD_AT]->(Venue)
    (Venue)-[:IN_CITY]->(City)
    (City)-[:IN_COUNTRY]->(Country)
    (Venue)-[:IN_TIMEZONE]->(TimeZone)
    (Athlete)-[:REPRESENTS]->(Country)
    (Athlete)-[:BASED_IN {source:'nationality_proxy'}]->(Country)

Resolution
    For each Event the city name is heuristically extracted from the event name
    (:func:`climber_network.geo.geocode.extract_city`) and looked up in a
    :class:`~climber_network.geo.geocode.GeoNamesIndex`. A successful lookup
    yields a high-confidence Venue at the city's coordinates, linked to its City,
    Country (ISO 3166-1 alpha-3) and TimeZone.

Country-centroid fallback
    Events whose city cannot be resolved fall back to a **country-level** Venue
    keyed at ``ven:country-{iso3}`` with a low ``geocode_confidence``. No City /
    TimeZone is attached (we only know the country). When a centroid coordinate
    is available (via the optional ``--centroids`` map, see below) it is stamped
    on the fallback Venue's ``location``; otherwise the Venue carries no point.

    Centroid source: the optional centroids file is a JSON object mapping
    ISO 3166-1 alpha-3 codes to ``[longitude, latitude]`` pairs (e.g. derived
    from the public-domain Natural Earth admin-0 centroids). It is gitignored
    (lives under ``data/``); when absent, fallback Venues simply omit a point.

Resolution cache
    Geocode resolutions are memoised to a local JSON file (gitignored path under
    ``data/``, overridable via ``--cache``) so repeat runs are deterministic and
    do not re-run the (heuristic) extraction. The cache stores the resolved
    GeoPoint fields keyed by ``"{city}|{iso3}"``; a sentinel records misses.

GeoNames data file
    The real index is built from the GeoNames ``cities1000`` TSV
    (``data/geonames/cities1000.txt``, gitignored — download + provenance are in
    :mod:`climber_network.geo.geocode`). The path is overridable via
    ``--geonames``.

Idempotency
    Every write is a MERGE keyed on a deterministic id, so re-running the build
    is a logical no-op. Node ids come from :mod:`climber_network.vocab` builders
    and every label / relationship type passes through ``assert_label`` /
    ``assert_rel`` via the GraphClient merge helpers — the single
    injection-safety gate. Labels / rel-types are NEVER interpolated here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import typer

from climber_network import vocab
from climber_network.geo.geocode import GeoNamesIndex, GeoPoint, extract_city

app = typer.Typer(add_completion=False, help="L2 geography build: Event names → places.")


# ---------------------------------------------------------------------------
# Structural types for the graph client — lets tests inject a fake recorder.
# ---------------------------------------------------------------------------


class GraphClientLike(Protocol):
    """Subset of GraphClient used by this build (structural typing)."""

    def merge_node(self, label: str, node_id: str, props: dict[str, Any]) -> None: ...

    def merge_rel(
        self,
        src_id: str,
        rel_type: str,
        tgt_id: str,
        props: dict[str, Any] | None = None,
    ) -> None: ...

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


#: Read query for the Event nodes already mirrored by P1.
EVENT_QUERY = "MATCH (e:Event) RETURN e.id AS id, e.name AS name, e.country AS country"

#: Read query for the Athlete nodes already mirrored by P1.
ATHLETE_QUERY = "MATCH (a:Athlete) RETURN a.id AS id, a.nationality AS nationality"

#: Marker stamped on the BASED_IN edge: athlete location is proxied by nationality.
NATIONALITY_PROXY = "nationality_proxy"

#: Sentinel stored in the resolution cache for an event that could not be resolved.
_CACHE_MISS = "__miss__"

#: CLI option defaults, hoisted to module-level singletons so the
#: ``typer.Option(...)`` calls are not performed in argument defaults
#: (satisfies ruff's flake8-bugbear B008).
_GEONAMES_OPT = typer.Option(
    Path("data/geonames/cities1000.txt"),
    "--geonames",
    help="Path to the GeoNames cities1000 TSV (gitignored under data/).",
)
_CACHE_OPT = typer.Option(
    Path("data/geocode_cache.json"),
    "--cache",
    help="Path to the geocode resolution cache JSON (gitignored under data/).",
)
_CENTROIDS_OPT = typer.Option(
    None,
    "--centroids",
    help="Optional ISO3 → [lon, lat] centroid JSON for fallback Venue points.",
)


# ---------------------------------------------------------------------------
# Build report — counts for logging.
# ---------------------------------------------------------------------------


@dataclass
class GeoReport:
    """Tallies emitted during a geography build, for logging."""

    src_events: int = 0
    src_athletes: int = 0

    resolved_events: int = 0
    fallback_events: int = 0
    skipped_events: int = 0  # no country at all → nothing to anchor.

    node_venues: int = 0
    node_cities: int = 0
    node_countries: int = 0
    node_timezones: int = 0

    edge_held_at: int = 0
    edge_in_city: int = 0
    edge_in_country: int = 0
    edge_in_timezone: int = 0
    edge_represents: int = 0
    edge_based_in: int = 0

    cache_hits: int = 0

    def log(self, console: Any) -> None:
        """Print a human-readable summary of counts."""
        console.print("[bold]L2 geography — build report[/bold]")
        console.print(
            f"  events:    src={self.src_events:>6}  resolved={self.resolved_events} "
            f"fallback={self.fallback_events} skipped={self.skipped_events} "
            f"(cache_hits={self.cache_hits})"
        )
        console.print(f"  athletes:  src={self.src_athletes:>6}")
        console.print(
            f"  nodes: Venue={self.node_venues} City={self.node_cities} "
            f"Country={self.node_countries} TimeZone={self.node_timezones}"
        )
        console.print(
            f"  edges: HELD_AT={self.edge_held_at} IN_CITY={self.edge_in_city} "
            f"IN_COUNTRY={self.edge_in_country} IN_TIMEZONE={self.edge_in_timezone} "
            f"REPRESENTS={self.edge_represents} BASED_IN={self.edge_based_in}"
        )


# ---------------------------------------------------------------------------
# Resolution cache — deterministic, file-backed memoisation of geocoding.
# ---------------------------------------------------------------------------


class ResolutionCache:
    """File-backed cache of geocode resolutions, keyed by ``"{city}|{iso3}"``.

    A hit stores the resolved :class:`GeoPoint` fields; a miss stores the
    ``_CACHE_MISS`` sentinel so unresolvable events are not re-extracted on every
    run. The cache is loaded on construction (missing / unreadable file → empty)
    and only written back when :meth:`flush` is called.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._data: dict[str, Any] = {}
        self._dirty = False
        if path is not None and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = {}
            if isinstance(loaded, dict):
                self._data = loaded

    @staticmethod
    def key(city: str, iso3: str | None) -> str:
        return f"{city}|{iso3 or ''}"

    def get(self, key: str) -> tuple[bool, GeoPoint | None] | None:
        """Return ``(hit_present, point)`` for *key*, or ``None`` if uncached.

        ``(True, GeoPoint)`` is a resolved hit; ``(True, None)`` is a cached miss.
        """
        if key not in self._data:
            return None
        entry = self._data[key]
        if entry == _CACHE_MISS:
            return (True, None)
        point = GeoPoint(
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            geonameid=int(entry["geonameid"]),
            name=str(entry["name"]),
            timezone=str(entry["timezone"]),
        )
        return (True, point)

    def put(self, key: str, point: GeoPoint | None) -> None:
        if point is None:
            self._data[key] = _CACHE_MISS
        else:
            self._data[key] = {
                "lat": point.lat,
                "lon": point.lon,
                "geonameid": point.geonameid,
                "name": point.name,
                "timezone": point.timezone,
            }
        self._dirty = True

    def flush(self) -> None:
        """Persist the cache to disk if it changed and a path was configured."""
        if self._path is None or not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        self._dirty = False


# ---------------------------------------------------------------------------
# Confidence levels stamped on Venue.geocode_confidence.
# ---------------------------------------------------------------------------

#: City was resolved against GeoNames at city granularity.
CONFIDENCE_CITY = 0.9
#: Only the country is known — Venue is a country-level centroid placeholder.
CONFIDENCE_COUNTRY = 0.2


# ---------------------------------------------------------------------------
# Core build logic — pure with respect to the injected client + inputs.
# ---------------------------------------------------------------------------


def _venue_point_props(point: GeoPoint) -> dict[str, Any]:
    """Venue props for a city-resolved location (with a WGS84 Point)."""
    return {
        "name": point.name,
        "location": _point(point.lon, point.lat),
        "geocode_confidence": CONFIDENCE_CITY,
    }


class _Point:
    """A neo4j ``point({longitude, latitude})`` value object.

    The real :class:`~climber_network.graph.client.GraphClient` passes node
    props straight to the driver as a parameter map, where a ``neo4j.spatial``
    point would be the canonical type. To keep this module driver-agnostic (and
    testable without a live driver) we carry longitude / latitude in a tiny,
    comparable value object whose repr matches the Cypher constructor. Tests can
    assert on ``.longitude`` / ``.latitude`` directly.
    """

    __slots__ = ("longitude", "latitude")

    def __init__(self, longitude: float, latitude: float) -> None:
        self.longitude = longitude
        self.latitude = latitude

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Point)
            and other.longitude == self.longitude
            and other.latitude == self.latitude
        )

    def __hash__(self) -> int:
        return hash((self.longitude, self.latitude))

    def __repr__(self) -> str:
        return f"point({{longitude: {self.longitude}, latitude: {self.latitude}}})"


def _point(longitude: float, latitude: float) -> _Point:
    """Build a neo4j-style ``point({longitude, latitude})`` value."""
    return _Point(longitude, latitude)


def build_geo(
    client: GraphClientLike,
    geonames: GeoNamesIndex,
    *,
    cache: ResolutionCache | None = None,
    centroids: dict[str, tuple[float, float]] | None = None,
) -> GeoReport:
    """Build the L2 geography graph from Event/Athlete nodes in *client*. Idempotent.

    Args:
        client:    The graph client (reads Event/Athlete, MERGEs geo nodes/edges).
        geonames:  The city → coordinate index.
        cache:     Optional resolution cache (memoises extraction/lookup).
        centroids: Optional ISO3 → ``(lon, lat)`` map for fallback Venue points.

    Returns:
        A :class:`GeoReport` of resolution outcomes and node/edge counts.
    """
    report = GeoReport()
    cache = cache if cache is not None else ResolutionCache(None)
    centroids = centroids or {}

    # Track which Country / TimeZone ids we've already MERGEd so the report counts
    # are logical (dedup-aware), matching the L1 mirror's convention.
    seen_countries: set[str] = set()
    seen_timezones: set[str] = set()

    def _ensure_country(iso3: str) -> str:
        ctry_id = vocab.ctry(iso3)
        if iso3 not in seen_countries:
            client.merge_node("Country", ctry_id, {"iso3": iso3})
            seen_countries.add(iso3)
            report.node_countries += 1
        return ctry_id

    def _ensure_timezone(iana: str) -> str:
        tz_id = vocab.tz(iana)
        if iana not in seen_timezones:
            client.merge_node("TimeZone", tz_id, {"iana": iana})
            seen_timezones.add(iana)
            report.node_timezones += 1
        return tz_id

    events = client.run_read(EVENT_QUERY)
    report.src_events = len(events)

    for ev in events:
        name = ev.get("name")
        country = ev.get("country")
        iso3 = str(country).strip().upper() if country else None

        point = _resolve(name, iso3, geonames, cache, report)

        if point is not None:
            _emit_resolved(client, report, ev, point, iso3, _ensure_country, _ensure_timezone)
            report.resolved_events += 1
        elif iso3:
            _emit_country_fallback(client, report, ev, iso3, centroids, _ensure_country)
            report.fallback_events += 1
        else:
            # No city and no country → nothing to anchor the event to.
            report.skipped_events += 1

    _emit_athletes(client, report, seen_countries, _ensure_country)

    cache.flush()
    return report


def _resolve(
    name: object,
    iso3: str | None,
    geonames: GeoNamesIndex,
    cache: ResolutionCache,
    report: GeoReport,
) -> GeoPoint | None:
    """Resolve an event name to a GeoPoint, consulting/populating the cache."""
    event_name = str(name) if name is not None else ""
    city = extract_city(event_name, iso3)
    if not city:
        return None

    cache_key = ResolutionCache.key(city, iso3)
    cached = cache.get(cache_key)
    if cached is not None:
        report.cache_hits += 1
        return cached[1]

    point = geonames.lookup(city, iso3)
    cache.put(cache_key, point)
    return point


def _emit_resolved(
    client: GraphClientLike,
    report: GeoReport,
    ev: dict[str, Any],
    point: GeoPoint,
    iso3: str | None,
    ensure_country: Any,
    ensure_timezone: Any,
) -> None:
    """MERGE Venue/City/Country/TimeZone + edges for a city-resolved event."""
    evt_id = vocab.evt(ev["id"])
    ven_id = vocab.ven(vocab.slug(point.name))

    client.merge_node("Venue", ven_id, _venue_point_props(point))
    report.node_venues += 1
    client.merge_rel(evt_id, "HELD_AT", ven_id)
    report.edge_held_at += 1

    # City.
    city_id = vocab.city(point.geonameid)
    client.merge_node(
        "City",
        city_id,
        {
            "name": point.name,
            "geonameid": point.geonameid,
            "location": _point(point.lon, point.lat),
        },
    )
    report.node_cities += 1
    client.merge_rel(ven_id, "IN_CITY", city_id)
    report.edge_in_city += 1

    # Country (from the event's ISO3) + IN_COUNTRY from the city.
    if iso3:
        ctry_id = ensure_country(iso3)
        client.merge_rel(city_id, "IN_COUNTRY", ctry_id)
        report.edge_in_country += 1

    # TimeZone.
    if point.timezone:
        tz_id = ensure_timezone(point.timezone)
        client.merge_rel(ven_id, "IN_TIMEZONE", tz_id)
        report.edge_in_timezone += 1


def _emit_country_fallback(
    client: GraphClientLike,
    report: GeoReport,
    ev: dict[str, Any],
    iso3: str,
    centroids: dict[str, tuple[float, float]],
    ensure_country: Any,
) -> None:
    """MERGE a low-confidence country-centroid Venue + HELD_AT / IN_COUNTRY.

    The fallback Venue is keyed at the country level (``ven:country-{iso3}``) so
    every unresolved event in the same country shares one placeholder Venue.
    """
    evt_id = vocab.evt(ev["id"])
    ven_id = vocab.ven(f"country-{iso3.lower()}")

    props: dict[str, Any] = {
        "name": f"{iso3} (country centroid)",
        "geocode_confidence": CONFIDENCE_COUNTRY,
    }
    centroid = centroids.get(iso3)
    if centroid is not None:
        lon, lat = centroid
        props["location"] = _point(lon, lat)

    client.merge_node("Venue", ven_id, props)
    report.node_venues += 1
    client.merge_rel(evt_id, "HELD_AT", ven_id)
    report.edge_held_at += 1

    ctry_id = ensure_country(iso3)
    # The centroid Venue sits in the country directly (no City to bridge).
    client.merge_rel(ven_id, "IN_COUNTRY", ctry_id)
    report.edge_in_country += 1


def _emit_athletes(
    client: GraphClientLike,
    report: GeoReport,
    seen_countries: set[str],
    ensure_country: Any,
) -> None:
    """MERGE REPRESENTS + BASED_IN (nationality proxy) from Athlete nodes."""
    athletes = client.run_read(ATHLETE_QUERY)
    report.src_athletes = len(athletes)

    # Avoid emitting the same edge twice for athletes sharing a nationality.
    represents_seen: set[tuple[str, str]] = set()

    for a in athletes:
        nationality = a.get("nationality")
        if not nationality:
            continue
        iso3 = str(nationality).strip().upper()
        ath_id = vocab.ath(a["id"])
        ctry_id = ensure_country(iso3)

        key = (ath_id, ctry_id)
        if key in represents_seen:
            continue
        represents_seen.add(key)

        client.merge_rel(ath_id, "REPRESENTS", ctry_id)
        report.edge_represents += 1
        client.merge_rel(ath_id, "BASED_IN", ctry_id, {"source": NATIONALITY_PROXY})
        report.edge_based_in += 1


# ---------------------------------------------------------------------------
# Centroid loading.
# ---------------------------------------------------------------------------


def load_centroids(path: Path | None) -> dict[str, tuple[float, float]]:
    """Load an ISO3 → ``(lon, lat)`` centroid map from a JSON file.

    The file is a JSON object mapping alpha-3 codes to ``[lon, lat]`` arrays.
    Returns an empty map when *path* is ``None`` or the file is absent.
    """
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for iso3, pair in raw.items():
        lon, lat = float(pair[0]), float(pair[1])
        out[str(iso3).strip().upper()] = (lon, lat)
    return out


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


@app.command()
def run(
    geonames: Path = _GEONAMES_OPT,
    cache: Path = _CACHE_OPT,
    centroids: Path | None = _CENTROIDS_OPT,
) -> None:
    """Build the L2 geography graph against the configured Neo4j instance."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()

    if not geonames.exists():
        console.print(
            f"[red]GeoNames file not found: {geonames}[/red]\n"
            "Download cities1000.zip from https://download.geonames.org/export/dump/ "
            "and unzip it to that path (see climber_network.geo.geocode)."
        )
        raise typer.Exit(code=1)

    index = GeoNamesIndex.from_tsv(geonames)
    resolution_cache = ResolutionCache(cache)
    centroid_map = load_centroids(centroids)

    client = get_client()
    report = build_geo(
        client,
        index,
        cache=resolution_cache,
        centroids=centroid_map,
    )
    report.log(console)
    console.print("[green]L2 geography build complete.[/green]")


if __name__ == "__main__":
    app()
