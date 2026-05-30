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
    keyed at ``ven:country-{iso3}`` with a low ``geocode_confidence``. No City is
    attached (we only know the country), but a representative **TimeZone** — the
    country's capital-city IANA zone, via
    :func:`~climber_network.geo.geocode.country_capital_tz` — is linked when
    known, so the L3 travel build can still derive a ``tz_delta_h`` for legs
    touching this Venue instead of dropping the timezone term (issue #42). When a
    centroid coordinate is available (via the optional ``--centroids`` map, see
    below) it is stamped on the fallback Venue's ``location``; otherwise the
    Venue carries no point.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer
from neo4j.spatial import WGS84Point

from climber_network import vocab
from climber_network.geo.geocode import (
    GeoNamesIndex,
    GeoPoint,
    alpha2_to_alpha3,
    country_capital_tz,
    extract_city,
    norm_city,
    override_alpha2,
    parse_ioc_alpha2,
    resolve_event,
)

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

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None: ...

    def merge_rels(self, rel_type: str, rows: list[dict[str, Any]]) -> None: ...

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
    fallback_with_tz: int = 0  # fallback events whose centroid Venue carries a capital tz.
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
            f"fallback={self.fallback_events} (fallback_with_tz={self.fallback_with_tz}) "
            f"skipped={self.skipped_events} (cache_hits={self.cache_hits})"
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
# In-memory accumulator — collects rows before the batched flush.
# ---------------------------------------------------------------------------


@dataclass
class _GeoAccumulator:
    """Collects deduped node and edge rows for a single batched flush.

    Nodes are keyed by ``(label, node_id)``; edges by ``(rel_type, src_id, tgt_id)``.
    Using dicts as the backing store means MERGE semantics are preserved: the
    last props written for a given key win, matching what MERGE + SET n += …
    would do in Neo4j.
    """

    # label → {node_id: props}
    nodes: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {
            "Country": {},
            "TimeZone": {},
            "Venue": {},
            "City": {},
        }
    )
    # rel_type → {(src_id, tgt_id): props | None}
    rels: dict[str, dict[tuple[str, str], dict[str, Any] | None]] = field(
        default_factory=lambda: {
            "HELD_AT": {},
            "IN_CITY": {},
            "IN_COUNTRY": {},
            "IN_TIMEZONE": {},
            "REPRESENTS": {},
            "BASED_IN": {},
        }
    )

    def add_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        self.nodes[label][node_id] = props

    def add_rel(
        self,
        src_id: str,
        rel_type: str,
        tgt_id: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        self.rels[rel_type][(src_id, tgt_id)] = props

    def flush(self, client: GraphClientLike) -> None:
        """Write all accumulated rows to *client* — nodes first, then edges."""
        # Nodes: Country → TimeZone → Venue → City (all before any edges so
        # merge_rels can MATCH them via the :Entity id index).
        for label in ("Country", "TimeZone", "Venue", "City"):
            node_map = self.nodes[label]
            if node_map:
                node_rows: list[dict[str, Any]] = [
                    {"id": nid, "props": props} for nid, props in node_map.items()
                ]
                client.merge_nodes(label, node_rows)

        # Edges: order doesn't affect correctness (endpoints already written
        # above), but keeping it consistent with the node-type ordering helps
        # readability of debug output.
        for rel_type in (
            "HELD_AT",
            "IN_CITY",
            "IN_COUNTRY",
            "IN_TIMEZONE",
            "REPRESENTS",
            "BASED_IN",
        ):
            rel_map = self.rels[rel_type]
            if rel_map:
                rel_rows: list[dict[str, Any]] = [
                    {"src_id": src, "tgt_id": tgt, "props": props}
                    for (src, tgt), props in rel_map.items()
                ]
                client.merge_rels(rel_type, rel_rows)


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


def _point(longitude: float, latitude: float) -> WGS84Point:
    """Build a neo4j WGS-84 ``point`` value for a Venue ``location`` prop.

    The :class:`GraphClient` passes node props straight to the driver as a
    parameter map. ``neo4j.spatial.WGS84Point`` is the *canonical*, Bolt-
    serializable point type — importing the value class needs no live driver,
    so this stays testable offline, and tests assert on ``.longitude`` /
    ``.latitude`` (native accessors on a 2-D WGS84Point).
    """
    return WGS84Point((longitude, latitude))


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

    acc = _GeoAccumulator()

    def _ensure_country(iso3: str) -> str:
        ctry_id = vocab.ctry(iso3)
        acc.add_node("Country", ctry_id, {"iso3": iso3})
        return ctry_id

    def _ensure_timezone(iana: str) -> str:
        tz_id = vocab.tz(iana)
        acc.add_node("TimeZone", tz_id, {"iana": iana})
        return tz_id

    events = client.run_read(EVENT_QUERY)
    report.src_events = len(events)

    # The upstream events table has NO country for any row, so the host country
    # must come from the event name. Most names carry a parenthesised IOC code
    # ("... Chamonix (FRA) 2022"); the rest (newer series names) do not. Build a
    # city → alpha-2 map from the coded events and use it to backfill the bare
    # ones (e.g. "World Climbing Series Innsbruck" inherits AT).
    backfill = _build_country_backfill(events)

    for ev in events:
        event_name = str(ev.get("name") or "")
        # Prefer the event's own parens code; else inherit from a same-named
        # event that had one; else fall back to a curated override's pin.
        alpha2 = (
            parse_ioc_alpha2(event_name)
            or backfill.get(_backfill_key(event_name))
            or override_alpha2(event_name)
        )

        point = _resolve(event_name, alpha2, geonames, cache, report)
        iso3 = alpha2_to_alpha3(alpha2)

        if point is not None:
            _emit_resolved(acc, ev, point, iso3, _ensure_country, _ensure_timezone)
            report.resolved_events += 1
        elif iso3:
            _emit_country_fallback(
                acc, ev, iso3, centroids, _ensure_country, _ensure_timezone, report
            )
            report.fallback_events += 1
        else:
            # No city and no country → nothing to anchor the event to.
            report.skipped_events += 1

    athletes = client.run_read(ATHLETE_QUERY)
    _emit_athletes_from_rows(acc, report, athletes, _ensure_country)

    # --- Flush all accumulated rows in a single pass (nodes then edges) ------
    # Set report counts from deduped accumulator sizes — matches how
    # sync/pg_to_neo4j.py sets report counts from batch lengths.
    report.node_countries = len(acc.nodes["Country"])
    report.node_timezones = len(acc.nodes["TimeZone"])
    report.node_venues = len(acc.nodes["Venue"])
    report.node_cities = len(acc.nodes["City"])

    report.edge_held_at = len(acc.rels["HELD_AT"])
    report.edge_in_city = len(acc.rels["IN_CITY"])
    report.edge_in_country = len(acc.rels["IN_COUNTRY"])
    report.edge_in_timezone = len(acc.rels["IN_TIMEZONE"])
    report.edge_represents = len(acc.rels["REPRESENTS"])
    report.edge_based_in = len(acc.rels["BASED_IN"])

    acc.flush(client)

    cache.flush()
    return report


def _backfill_key(event_name: str) -> str:
    """Normalised city key used by the country-backfill map (accent/case-fold)."""
    return norm_city(extract_city(event_name, None) or "")


def _build_country_backfill(events: list[dict[str, Any]]) -> dict[str, str]:
    """Map ``norm_city(name) → alpha-2`` from events that carry a parens IOC code.

    Lets events whose name lacks an explicit country (e.g. "World Climbing
    Series Innsbruck 2026") inherit the host country from a same-named coded
    event ("... Innsbruck (AUT) ..."). First definite code per city wins.
    """
    out: dict[str, str] = {}
    for ev in events:
        event_name = str(ev.get("name") or "")
        alpha2 = parse_ioc_alpha2(event_name)
        if alpha2 is None:
            continue
        key = _backfill_key(event_name)
        if key and key not in out:
            out[key] = alpha2
    return out


def _resolve(
    event_name: str,
    alpha2: str | None,
    geonames: GeoNamesIndex,
    cache: ResolutionCache,
    report: GeoReport,
) -> GeoPoint | None:
    """Resolve an event name to a :class:`GeoPoint` (or ``None``), cached.

    *alpha2* is the resolved host-country hint. The full end-to-end resolution
    (extraction → override → constrained lookup) lives in
    :func:`climber_network.geo.geocode.resolve_event`; this wrapper only adds the
    deterministic file-backed cache, keyed by ``(event_name, alpha2)`` so the
    resolution is fully determined by the cache key.
    """
    cache_key = ResolutionCache.key(event_name, alpha2)
    cached = cache.get(cache_key)
    if cached is not None:
        report.cache_hits += 1
        return cached[1]

    resolution = resolve_event(event_name, geonames, alpha2=alpha2)
    cache.put(cache_key, resolution.point)
    return resolution.point


def _emit_resolved(
    acc: _GeoAccumulator,
    ev: dict[str, Any],
    point: GeoPoint,
    iso3: str | None,
    ensure_country: Any,
    ensure_timezone: Any,
) -> None:
    """Accumulate Venue/City/Country/TimeZone + edges for a city-resolved event."""
    # ``EVENT_QUERY`` returns the Event node's ``id`` property, which is ALREADY
    # the full vocab id (e.g. "evt:4"). Do NOT re-wrap with vocab.evt() — that
    # would double-prefix to "evt:evt:4", silently matching no node so the
    # HELD_AT MERGE writes nothing.
    evt_id = ev["id"]
    ven_id = vocab.ven(vocab.slug(point.name))

    acc.add_node("Venue", ven_id, _venue_point_props(point))
    acc.add_rel(evt_id, "HELD_AT", ven_id)

    # City.
    city_id = vocab.city(point.geonameid)
    acc.add_node(
        "City",
        city_id,
        {
            "name": point.name,
            "geonameid": point.geonameid,
            "location": _point(point.lon, point.lat),
        },
    )
    acc.add_rel(ven_id, "IN_CITY", city_id)

    # Country (from the event's ISO3) + IN_COUNTRY from the city.
    if iso3:
        ctry_id = ensure_country(iso3)
        acc.add_rel(city_id, "IN_COUNTRY", ctry_id)

    # TimeZone.
    if point.timezone:
        tz_id = ensure_timezone(point.timezone)
        acc.add_rel(ven_id, "IN_TIMEZONE", tz_id)


def _emit_country_fallback(
    acc: _GeoAccumulator,
    ev: dict[str, Any],
    iso3: str,
    centroids: dict[str, tuple[float, float]],
    ensure_country: Any,
    ensure_timezone: Any,
    report: GeoReport,
) -> None:
    """Accumulate a low-confidence country-centroid Venue + HELD_AT / IN_COUNTRY.

    The fallback Venue is keyed at the country level (``ven:country-{iso3}``) so
    every unresolved event in the same country shares one placeholder Venue.

    A representative IANA timezone (the country's capital, via
    :func:`~climber_network.geo.geocode.country_capital_tz`) is attached when
    known, so the L3 travel build can still compute a ``tz_delta_h`` for legs
    that originate from or arrive at this centroid Venue instead of dropping the
    timezone term entirely (issue #42).
    """
    # ev["id"] is already the full node id (see _emit_resolved) — do not re-wrap.
    evt_id = ev["id"]
    ven_id = vocab.ven(f"country-{iso3.lower()}")

    props: dict[str, Any] = {
        "name": f"{iso3} (country centroid)",
        "geocode_confidence": CONFIDENCE_COUNTRY,
    }
    centroid = centroids.get(iso3)
    if centroid is not None:
        lon, lat = centroid
        props["location"] = _point(lon, lat)

    acc.add_node("Venue", ven_id, props)
    acc.add_rel(evt_id, "HELD_AT", ven_id)

    ctry_id = ensure_country(iso3)
    # The centroid Venue sits in the country directly (no City to bridge).
    acc.add_rel(ven_id, "IN_COUNTRY", ctry_id)

    # Capital-city timezone so the centroid still carries a tz (issue #42).
    iana = country_capital_tz(iso3)
    if iana:
        tz_id = ensure_timezone(iana)
        acc.add_rel(ven_id, "IN_TIMEZONE", tz_id)
        report.fallback_with_tz += 1


def _emit_athletes_from_rows(
    acc: _GeoAccumulator,
    report: GeoReport,
    athletes: list[dict[str, Any]],
    ensure_country: Any,
) -> None:
    """Accumulate REPRESENTS + BASED_IN (nationality proxy) rows into *acc*.

    Deduplication is implicit — ``acc.add_rel`` keyed on ``(ath_id, ctry_id)``
    so duplicate athlete–country pairs (from a re-run or a shared nationality)
    collapse to a single edge row, matching MERGE semantics.
    """
    report.src_athletes = len(athletes)

    for a in athletes:
        nationality = a.get("nationality")
        if not nationality:
            continue
        iso3 = str(nationality).strip().upper()
        # ATHLETE_QUERY returns the Athlete node's full id ("ath:5"); use it
        # directly. Re-wrapping with vocab.ath() would double-prefix and the
        # REPRESENTS / BASED_IN MERGEs would silently match no node.
        ath_id = a["id"]
        ctry_id = ensure_country(iso3)

        acc.add_rel(ath_id, "REPRESENTS", ctry_id)
        acc.add_rel(ath_id, "BASED_IN", ctry_id, {"source": NATIONALITY_PROXY})


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
