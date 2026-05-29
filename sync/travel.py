"""sync.travel — P3c L3 travel / circadian build: derive TravelLeg + RestednessState.

The "L3" differentiator. Reads the competition + geography graph already built by
P1 (``sync.pg_to_neo4j``) and P2 (``sync.geo``) and, **per athlete**, derives one
travel leg and one restedness state for every event the athlete competed in:

Nodes
    TravelLeg, RestednessState

Edges
    (Athlete)-[:TRAVELED]->(TravelLeg)-[:TO_EVENT]->(Event)
    (Athlete)-[:HAD_STATE]->(RestednessState)-[:AT_EVENT]->(Event)

Travel-origin model (PRD §9)
    Athletes usually return to a *home base* between competitions, so the
    jet-lag-relevant trip is ``home_base → venue`` — **not** ``prev_venue →
    venue``. Event-to-event is only realistic for a tight "swing" of back-to-back
    events. Therefore, per athlete (events ordered by ``start_date``):

    * if the gap to the previous event ``<= swing_gap_days`` **and** that event is
      in a *different* timezone → ``origin = prev_event`` (a swing leg);
    * otherwise → ``origin = home_base`` (the athlete's ``BASED_IN`` country's
      representative coordinates + timezone; a nationality proxy in P3).

    The first event of an athlete always originates from the home base.

Home-base resolution (documented approximation, dependency-free)
    The home country's representative coordinates + timezone are resolved **from
    the graph**: among the Venues located in that country (via
    ``City-[:IN_COUNTRY]->Country`` or a country-centroid Venue
    ``-[:IN_COUNTRY]->Country``), we take the **most common IANA timezone** and
    the **centroid** (mean longitude / latitude) of those venues' points. When
    the country cannot be resolved to any coordinate/timezone, the leg is still
    emitted with a low ``confidence`` and the timezone term is dropped (the
    restedness falls back to travel_fatigue only, with ``tz_delta_h = 0``).

    ``confidence`` is carried on every TravelLeg: lower for a ``home_base`` origin
    (nationality proxy), higher for an observed ``prev_event`` swing leg, and
    lowest when the home base could not be resolved at all.

Arrival timing (PRD §9)
    True arrival dates are unknown, so arrival is a **parameter**:
    ``arrive_date = event.start_date - arrive_days_before`` and
    ``days_since_arrival = arrive_days_before``. ``depart_date == arrive_date``
    (same-day long-haul assumption). UTC offsets are evaluated on ``arrive_date``
    (DST-aware) via :func:`climber_network.geo.geocode.utc_offset_hours`.

Idempotency
    Every write is a MERGE keyed on a deterministic id, so re-running the build
    is a logical no-op. Node ids come from :mod:`climber_network.vocab` builders
    and every label / relationship type passes through ``assert_label`` /
    ``assert_rel`` via the GraphClient merge helpers — the single
    injection-safety gate. Labels / rel-types are NEVER interpolated here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

import typer

from climber_network import vocab
from climber_network.config import TRAVEL_PARAMS, TravelParams
from climber_network.geo.geocode import utc_offset_hours
from climber_network.travel.formulas import compute_restedness, haversine_km

app = typer.Typer(
    add_completion=False, help="L3 travel/circadian build: TravelLeg + RestednessState."
)


# ---------------------------------------------------------------------------
# Structural type for the graph client — lets tests inject a fake recorder.
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


# ---------------------------------------------------------------------------
# Read queries against the P1/P2 graph.
# ---------------------------------------------------------------------------

#: One row per (athlete, event) the athlete competed in, with the destination
#: venue's coordinates + timezone. De-duplicated to one row per athlete-event in
#: Cypher (an athlete can have several Performances per event across rounds).
#:
#: ``loc`` is the venue ``location`` point (may be NULL for an unresolved venue);
#: we read its ``longitude`` / ``latitude`` components directly so the result is
#: a plain JSON-serialisable dict and tests need no neo4j point type.
ATHLETE_EVENT_QUERY = """
MATCH (a:Athlete)-[:COMPETED_IN]->(:Performance)-[:OF_ROUND]->(:Round)
      -[:OF_EVENT]->(e:Event)
OPTIONAL MATCH (e)-[:HELD_AT]->(v:Venue)
OPTIONAL MATCH (v)-[:IN_TIMEZONE]->(tz:TimeZone)
WITH a, e, v, tz,
     CASE WHEN v.location IS NULL THEN NULL ELSE v.location.longitude END AS lon,
     CASE WHEN v.location IS NULL THEN NULL ELSE v.location.latitude END AS lat
RETURN DISTINCT
       a.id        AS athlete_id,
       e.id        AS event_id,
       e.start_date AS start_date,
       lon         AS venue_lon,
       lat         AS venue_lat,
       tz.iana     AS venue_tz,
       e.discipline AS discipline
"""

#: The athlete's home-base country (nationality proxy via the BASED_IN edge
#: emitted by P2). One row per athlete (athletes without BASED_IN are absent).
ATHLETE_HOME_QUERY = """
MATCH (a:Athlete)-[:BASED_IN]->(c:Country)
RETURN a.id AS athlete_id, c.iso3 AS iso3
"""

#: Representative coordinates + timezone for each country, derived from the
#: Venues located in that country. We collect every venue point + timezone so
#: the centroid (mean lon/lat) and the most-common timezone can be computed in
#: Python (keeps the Cypher simple + testable). Venues reach a Country either via
#: a City (``Venue-[:IN_CITY]->City-[:IN_COUNTRY]->Country``) or directly
#: (country-centroid fallback Venue ``-[:IN_COUNTRY]->Country``). A
#: variable-length ``[:IN_CITY|IN_COUNTRY*1..2]`` covers both shapes; the
#: ``DISTINCT`` collapses the two ways a single Venue can reach the country.
COUNTRY_GEO_QUERY = """
MATCH (v:Venue)-[:IN_CITY|IN_COUNTRY*1..2]->(c:Country)
OPTIONAL MATCH (v)-[:IN_TIMEZONE]->(tz:TimeZone)
WITH DISTINCT c.iso3 AS iso3, v AS v, tz.iana AS iana,
     CASE WHEN v.location IS NULL THEN NULL ELSE v.location.longitude END AS lon,
     CASE WHEN v.location IS NULL THEN NULL ELSE v.location.latitude END AS lat
WHERE lon IS NOT NULL OR iana IS NOT NULL
RETURN iso3 AS iso3, lon AS lon, lat AS lat, iana AS iana
"""


# ---------------------------------------------------------------------------
# Origin model + confidence levels.
# ---------------------------------------------------------------------------

OriginKind = Literal["home_base", "prev_event"]

#: An observed swing leg (prev_event → venue): both endpoints are real venues.
CONFIDENCE_SWING = 0.8
#: A home-base origin resolved from the graph (nationality-proxy country).
CONFIDENCE_HOME_BASE = 0.4
#: Home base could not be resolved (no coords/tz) — tz term dropped.
CONFIDENCE_UNRESOLVED = 0.2


# ---------------------------------------------------------------------------
# Value objects.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Place:
    """A resolved origin / destination: coordinates (optional) + timezone (optional)."""

    lon: float | None
    lat: float | None
    tz: str | None

    @property
    def has_coords(self) -> bool:
        return self.lon is not None and self.lat is not None


@dataclass(frozen=True)
class AthleteEvent:
    """One de-duplicated (athlete, event) row: destination venue + date."""

    athlete_id: int | str
    event_id: int | str
    start_date: date | None
    venue: Place
    discipline: str | None = None


# ---------------------------------------------------------------------------
# Build report — counts for logging.
# ---------------------------------------------------------------------------


@dataclass
class TravelReport:
    """Tallies emitted during a travel build, for logging."""

    src_athletes: int = 0
    src_athlete_events: int = 0

    legs: int = 0
    states: int = 0

    origin_home_base: int = 0
    origin_prev_event: int = 0

    unresolved_origin: int = 0  # leg emitted, but origin coords/tz missing.
    skipped_no_date: int = 0  # event has no start_date → cannot place arrival.
    skipped_no_venue: int = 0  # destination venue has no coords → cannot place leg.

    edge_traveled: int = 0
    edge_to_event: int = 0
    edge_had_state: int = 0
    edge_at_event: int = 0

    def log(self, console: Any) -> None:
        """Print a human-readable summary of counts."""
        console.print("[bold]L3 travel/circadian — build report[/bold]")
        console.print(
            f"  athletes:       src={self.src_athletes:>6}  "
            f"athlete-events={self.src_athlete_events}"
        )
        console.print(
            f"  origins: home_base={self.origin_home_base} prev_event={self.origin_prev_event} "
            f"(unresolved_origin={self.unresolved_origin})"
        )
        console.print(f"  skipped: no_date={self.skipped_no_date} no_venue={self.skipped_no_venue}")
        console.print(f"  nodes: TravelLeg={self.legs} RestednessState={self.states}")
        console.print(
            f"  edges: TRAVELED={self.edge_traveled} TO_EVENT={self.edge_to_event} "
            f"HAD_STATE={self.edge_had_state} AT_EVENT={self.edge_at_event}"
        )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _parse_date(value: Any) -> date | None:
    """Coerce a graph ``start_date`` value (ISO string / date / neo4j Date) to ``date``."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value)
    # neo4j temporal types and ISO strings both render as 'YYYY-MM-DD...'.
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_home_bases(
    country_rows: list[dict[str, Any]],
) -> dict[str, Place]:
    """Resolve each country's representative :class:`Place` from venue rows.

    For every country, take the centroid (mean longitude / latitude) of all its
    venues that carry coordinates and the **most common** IANA timezone among
    them. A country with no coordinates and no timezone yields a :class:`Place`
    with all-``None`` fields (treated as unresolved downstream).

    Args:
        country_rows: Rows from :data:`COUNTRY_GEO_QUERY` — each has ``iso3`` and
            optional ``lon`` / ``lat`` / ``iana``.

    Returns:
        Mapping of upper-cased ISO3 → :class:`Place`.
    """
    lons: dict[str, list[float]] = {}
    lats: dict[str, list[float]] = {}
    tzs: dict[str, Counter[str]] = {}

    for row in country_rows:
        iso3_raw = row.get("iso3")
        if not iso3_raw:
            continue
        iso3 = str(iso3_raw).strip().upper()
        lon = _to_float(row.get("lon"))
        lat = _to_float(row.get("lat"))
        if lon is not None and lat is not None:
            lons.setdefault(iso3, []).append(lon)
            lats.setdefault(iso3, []).append(lat)
        iana = row.get("iana")
        if iana:
            tzs.setdefault(iso3, Counter())[str(iana)] += 1

    out: dict[str, Place] = {}
    seen = set(lons) | set(tzs)
    for iso3 in seen:
        lon_vals = lons.get(iso3, [])
        lat_vals = lats.get(iso3, [])
        centroid_lon = sum(lon_vals) / len(lon_vals) if lon_vals else None
        centroid_lat = sum(lat_vals) / len(lat_vals) if lat_vals else None
        tz_counter = tzs.get(iso3)
        # most_common(1) is deterministic on insertion order for ties (Counter).
        common_tz = tz_counter.most_common(1)[0][0] if tz_counter else None
        out[iso3] = Place(lon=centroid_lon, lat=centroid_lat, tz=common_tz)
    return out


def _group_athlete_events(rows: list[dict[str, Any]]) -> dict[int | str, list[AthleteEvent]]:
    """Group + de-dup read rows into ``athlete_id -> [AthleteEvent]`` ordered by date.

    The Cypher already returns DISTINCT (athlete, event) rows, but we de-dup again
    defensively by ``(athlete_id, event_id)`` and sort each athlete's events by
    ``start_date`` (events without a date sort last and are skipped downstream).
    """
    by_athlete: dict[int | str, dict[int | str, AthleteEvent]] = {}
    for row in rows:
        athlete_id = row.get("athlete_id")
        event_id = row.get("event_id")
        if athlete_id is None or event_id is None:
            continue
        ae = AthleteEvent(
            athlete_id=athlete_id,
            event_id=event_id,
            start_date=_parse_date(row.get("start_date")),
            venue=Place(
                lon=_to_float(row.get("venue_lon")),
                lat=_to_float(row.get("venue_lat")),
                tz=str(row["venue_tz"]) if row.get("venue_tz") else None,
            ),
            discipline=str(row["discipline"]) if row.get("discipline") else None,
        )
        by_athlete.setdefault(athlete_id, {})[event_id] = ae

    grouped: dict[int | str, list[AthleteEvent]] = {}
    for athlete_id, events in by_athlete.items():
        grouped[athlete_id] = sorted(
            events.values(),
            # date.max keeps undated events last but stable.
            key=lambda ae: (ae.start_date or date.max, str(ae.event_id)),
        )
    return grouped


def _select_origin(
    current: AthleteEvent,
    prev: AthleteEvent | None,
    home: Place | None,
    params: TravelParams,
) -> tuple[OriginKind, Place | None, float]:
    """Choose the leg origin for *current* per the PRD §9 swing / home-base rule.

    Returns ``(origin_kind, origin_place, confidence)``. When the gap to *prev* is
    within ``swing_gap_days`` **and** the previous event sits in a different
    timezone (and both events are dated), the previous event is a swing origin;
    otherwise the home base is used. ``origin_place`` may be ``None`` when the
    home base could not be resolved.
    """
    if (
        prev is not None
        and prev.start_date is not None
        and current.start_date is not None
        and prev.venue.tz is not None
        and current.venue.tz is not None
    ):
        gap_days = (current.start_date - prev.start_date).days
        if 0 <= gap_days <= params.swing_gap_days and prev.venue.tz != current.venue.tz:
            return "prev_event", prev.venue, CONFIDENCE_SWING

    if home is not None and (home.has_coords or home.tz is not None):
        return "home_base", home, CONFIDENCE_HOME_BASE
    return "home_base", home, CONFIDENCE_UNRESOLVED


def _emit_leg_and_state(
    client: GraphClientLike,
    report: TravelReport,
    *,
    athlete_id: int | str,
    current: AthleteEvent,
    origin_kind: OriginKind,
    origin: Place | None,
    confidence: float,
    params: TravelParams,
) -> None:
    """Compute + MERGE one TravelLeg and one RestednessState (with their 4 edges)."""
    ath_id = vocab.ath(athlete_id)
    evt_id = vocab.evt(current.event_id)
    leg_id = vocab.leg(ath_id, evt_id)
    rest_id = vocab.rest(ath_id, evt_id)

    assert current.start_date is not None  # guarded by the caller.
    arrive_date = date.fromordinal(current.start_date.toordinal() - params.arrive_days_before)
    days_since_arrival = float(params.arrive_days_before)

    venue = current.venue

    # Distance: only when both endpoints carry coordinates; else 0 (no flight term).
    distance_km = 0.0
    if origin is not None and origin.has_coords and venue.has_coords:
        assert origin.lon is not None and origin.lat is not None
        assert venue.lon is not None and venue.lat is not None
        distance_km = haversine_km(origin.lat, origin.lon, venue.lat, venue.lon)

    # Timezone offsets on the arrival date (DST-aware). Drop the tz term when
    # either side's timezone is unknown → tz_delta_h = 0, direction = none.
    if origin is not None and origin.tz is not None and venue.tz is not None:
        origin_offset = utc_offset_hours(origin.tz, arrive_date)
        venue_offset = utc_offset_hours(venue.tz, arrive_date)
    else:
        origin_offset = 0.0
        venue_offset = 0.0
        report.unresolved_origin += 1

    rep = compute_restedness(
        distance_km=distance_km,
        origin_offset_h=origin_offset,
        venue_offset_h=venue_offset,
        days_since_arrival=days_since_arrival,
        params=params,
    )

    client.merge_node(
        "TravelLeg",
        leg_id,
        {
            "origin": origin_kind,
            "origin_tz": origin.tz if origin is not None else None,
            "distance_km": rep["distance_km"],
            "est_flight_h": rep["est_flight_h"],
            "tz_delta_h": rep["tz_delta_h"],
            "direction": rep["direction"],
            "depart_date": arrive_date.isoformat(),
            "arrive_date": arrive_date.isoformat(),
            "confidence": confidence,
        },
    )
    report.legs += 1
    client.merge_rel(ath_id, "TRAVELED", leg_id)
    report.edge_traveled += 1
    client.merge_rel(leg_id, "TO_EVENT", evt_id)
    report.edge_to_event += 1

    client.merge_node(
        "RestednessState",
        rest_id,
        {
            # Denormalized join keys + breakdown dims so the ELO-validation
            # correlation (sync/validate_elo REST_QUERY) reads them directly,
            # rather than degrading to n=0. See PRD §9 validation hook.
            "athlete_id": athlete_id,
            "event_id": current.event_id,
            "discipline": current.discipline,
            "travel_direction": rep["direction"],
            "days_since_arrival": rep["days_since_arrival"],
            "recovery_days_needed": rep["recovery_days_needed"],
            "jetlag_residual": rep["jetlag_residual"],
            "travel_fatigue": rep["travel_fatigue"],
            "rested_index": rep["rested_index"],
            "model_version": rep["model_version"],
        },
    )
    report.states += 1
    client.merge_rel(ath_id, "HAD_STATE", rest_id)
    report.edge_had_state += 1
    client.merge_rel(rest_id, "AT_EVENT", evt_id)
    report.edge_at_event += 1

    if origin_kind == "home_base":
        report.origin_home_base += 1
    else:
        report.origin_prev_event += 1


# ---------------------------------------------------------------------------
# Core build logic — pure with respect to the injected client + inputs.
# ---------------------------------------------------------------------------


def build_travel(
    client: GraphClientLike,
    *,
    params: TravelParams = TRAVEL_PARAMS,
) -> TravelReport:
    """Derive TravelLeg + RestednessState for every athlete-event in *client*. Idempotent.

    Reads the athlete-event traversal, the BASED_IN home-base countries, and the
    per-country representative geography, then walks each athlete's events in date
    order applying the swing / home-base origin rule (PRD §9) and the L3 travel
    formulas.

    Args:
        client: The graph client (reads P1/P2 nodes, MERGEs L3 nodes/edges).
        params: Travel model constants.

    Returns:
        A :class:`TravelReport` of origin outcomes and node/edge counts.
    """
    report = TravelReport()

    homes_iso3 = {
        str(row["athlete_id"]): str(row["iso3"]).strip().upper()
        for row in client.run_read(ATHLETE_HOME_QUERY)
        if row.get("athlete_id") is not None and row.get("iso3")
    }
    home_places = resolve_home_bases(client.run_read(COUNTRY_GEO_QUERY))

    grouped = _group_athlete_events(client.run_read(ATHLETE_EVENT_QUERY))
    report.src_athletes = len(grouped)

    for athlete_id, events in grouped.items():
        home_iso3 = homes_iso3.get(str(athlete_id))
        home = home_places.get(home_iso3) if home_iso3 else None

        prev: AthleteEvent | None = None
        for current in events:
            report.src_athlete_events += 1

            if current.start_date is None:
                report.skipped_no_date += 1
                continue
            if not current.venue.has_coords:
                # Cannot place a leg without a destination coordinate; the
                # restedness would be meaningless. Documented skip.
                report.skipped_no_venue += 1
                # Still advance ``prev`` so a later swing test uses a real prior
                # only when that prior had a venue; an unplaced event makes a poor
                # swing origin, so we leave prev unchanged.
                continue

            origin_kind, origin, confidence = _select_origin(current, prev, home, params)
            _emit_leg_and_state(
                client,
                report,
                athlete_id=athlete_id,
                current=current,
                origin_kind=origin_kind,
                origin=origin,
                confidence=confidence,
                params=params,
            )
            prev = current

    return report


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


@app.command()
def run() -> None:
    """Build the L3 travel/circadian graph against the configured Neo4j instance."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    client = get_client()
    report = build_travel(client)
    report.log(console)
    console.print("[green]L3 travel/circadian build complete.[/green]")


if __name__ == "__main__":
    app()
