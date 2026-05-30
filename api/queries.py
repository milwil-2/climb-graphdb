"""Read-only query endpoints (U1–U5) for the Climber Network API.

This module holds the Cypher + shaping logic behind the user-facing read
endpoints wired into :mod:`api.index`:

* **U4** athlete profile (``athlete_profile``) — props + ratings + recent events.
* **U4** athlete neighborhood (``athlete_neighborhood``) — bounded ``{nodes,
  edges}`` subgraph (events, venues via ``HELD_AT``, rivals via ``FACED``).
* **U1** head-to-head (``head_to_head``) — the ``FACED`` aggregate between two
  athletes + a short co-competition summary.
* **U2** venue clusters (``venue_clusters``) — venues ranked by repeated
  co-competition (distinct athletes / repeat visits).
* **U3** jetlagged underperformers (``jetlagged_underperformers``) — join
  low-``rested_index`` ``RestednessState`` to the representative
  ``Performance.elo_residual`` (residual > 0 ⇒ worse than expected) via the
  existing ``HAD_STATE`` + ``AT_EVENT`` edges (no string-concat joins).
* **U5** athlete timeline (``athlete_timeline``) — merged chronological events
  + ``RestednessState`` via the ``AT_EVENT`` edge (+ optional ``TrainingSignal``
  / ``InjuryEvent``).

Design constraints (same as :mod:`api.rag`)
-------------------------------------------
* Self-contained: depends only on :mod:`api.db` (the read accessor) and
  ``climber_network.vocab`` — NOT on the sibling ``climbing_elo`` /
  ``knowledge_graph`` projects.
* Every label / relationship-type interpolated into a Cypher string is gated
  through ``assert_label`` / ``assert_rel`` (the single injection-safety gate);
  all caller-supplied values are passed as **bound parameters**, never
  interpolated.
* The raw climbing-elo athlete id is turned into the ``ath:{id}`` node id via
  ``vocab.ath`` before any query runs.
"""

from __future__ import annotations

from typing import Any

from climber_network.vocab import assert_label, assert_rel
from climber_network.vocab import ath as ath_id

from . import db

# ---------------------------------------------------------------------------
# Fan-out / list caps — kept small so payloads stay legible and bounded.
# ---------------------------------------------------------------------------

_RECENT_EVENTS = 10
_MAX_EVENTS = 50
_MAX_RIVALS = 50
_MAX_CLUSTERS = 25
_MAX_UNDERPERFORMERS = 50
_DEFAULT_HOPS = 2
_MAX_HOPS = 3

# ---------------------------------------------------------------------------
# Gated labels / relationship types (static literals only).
# ---------------------------------------------------------------------------

_ATHLETE = assert_label("Athlete")
_EVENT = assert_label("Event")
_ROUND = assert_label("Round")
_PERFORMANCE = assert_label("Performance")
_RATING = assert_label("Rating")
_VENUE = assert_label("Venue")
_REST = assert_label("RestednessState")
_SIGNAL = assert_label("TrainingSignal")
_INJURY = assert_label("InjuryEvent")
_SEASON = assert_label("SeasonSummary")

_COMPETED_IN = assert_rel("COMPETED_IN")
_OF_ROUND = assert_rel("OF_ROUND")
_OF_EVENT = assert_rel("OF_EVENT")
_HAS_RATING = assert_rel("HAS_RATING")
_HELD_AT = assert_rel("HELD_AT")
_FACED = assert_rel("FACED")
_HAD_STATE = assert_rel("HAD_STATE")
_AT_EVENT = assert_rel("AT_EVENT")
_HAS_SIGNAL = assert_rel("HAS_SIGNAL")
_HAD_INJURY = assert_rel("HAD_INJURY")
_HAD_SEASON = assert_rel("HAD_SEASON")

# ---------------------------------------------------------------------------
# U4 — athlete profile
# ---------------------------------------------------------------------------

#: Athlete props + ratings + the N most-recent events (by Event.start_date).
PROFILE_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$id}}) "
    f"OPTIONAL MATCH (a)-[:{_HAS_RATING}]->(rt:{_RATING}) "
    "WITH a, collect(DISTINCT {discipline:rt.discipline, mu:rt.mu, "
    "sigma:rt.sigma, n_events:rt.n_events, provisional:rt.provisional}) AS ratings "
    f"OPTIONAL MATCH (a)-[:{_COMPETED_IN}]->(:{_PERFORMANCE})"
    f"-[:{_OF_ROUND}]->(:{_ROUND})-[:{_OF_EVENT}]->(e:{_EVENT}) "
    f"OPTIONAL MATCH (e)-[:{_HELD_AT}]->(v:{_VENUE}) "
    "WITH a, ratings, e, v "
    "ORDER BY e.start_date DESC "
    "WITH a, ratings, collect(DISTINCT {id:e.id, name:e.name, "
    "start_date:toString(e.start_date), discipline:e.discipline, "
    "venue:v.name})[..$recent] AS events "
    "RETURN a.id AS id, a.name AS name, a.nationality AS nationality, "
    "a.gender AS gender, a.year_of_birth AS year_of_birth, "
    "[r IN ratings WHERE r.discipline IS NOT NULL] AS ratings, "
    "[ev IN events WHERE ev.id IS NOT NULL] AS recent_events"
)


def athlete_profile(athlete_id: str) -> dict[str, Any] | None:
    """Return the athlete profile dict for the raw *athlete_id*, or ``None``.

    Shape::

        {"id", "name", "nationality", "gender", "year_of_birth",
         "ratings": [...], "recent_events": [...]}

    ``None`` is returned when the athlete node does not exist (the API layer
    maps this to a 404).
    """
    rows = db.run_read(PROFILE_CYPHER, id=ath_id(athlete_id), recent=_RECENT_EVENTS)
    if not rows:
        return None
    row = rows[0]
    return {
        "id": str(row["id"]),
        "name": row.get("name"),
        "nationality": row.get("nationality"),
        "gender": row.get("gender"),
        "year_of_birth": row.get("year_of_birth"),
        "ratings": list(row.get("ratings") or []),
        "recent_events": list(row.get("recent_events") or []),
    }


# ---------------------------------------------------------------------------
# U4 — athlete neighborhood (bounded {nodes, edges} subgraph)
# ---------------------------------------------------------------------------

#: Bounded subgraph around an athlete: events (+ their venues via HELD_AT) and
#: rivals (via FACED). Fan-out is capped via the bound ``$max_*`` params.
NEIGHBORHOOD_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$id}}) "
    f"OPTIONAL MATCH (a)-[:{_COMPETED_IN}]->(:{_PERFORMANCE})"
    f"-[:{_OF_ROUND}]->(:{_ROUND})-[:{_OF_EVENT}]->(e:{_EVENT}) "
    f"OPTIONAL MATCH (e)-[:{_HELD_AT}]->(v:{_VENUE}) "
    "WITH a, collect(DISTINCT {id:e.id, name:e.name, "
    "start_date:toString(e.start_date), discipline:e.discipline, "
    "venue:v.name})[..$max_events] AS events "
    f"OPTIONAL MATCH (a)-[f:{_FACED}]->(r:{_ATHLETE}) "
    "WITH a, events, collect(DISTINCT {id:r.id, name:r.name, "
    "count:f.count})[..$max_rivals] AS rivals "
    "RETURN a.id AS id, a.name AS name, events, rivals"
)


def athlete_neighborhood(athlete_id: str, hops: int = _DEFAULT_HOPS) -> dict[str, Any] | None:
    """Return a bounded ``{athlete, nodes, edges, hops}`` subgraph, or ``None``.

    *hops* is clamped to ``[1, _MAX_HOPS]``; it is informational here (the query
    is a fixed 2-hop expansion: athlete → event → venue, athlete → rival), and
    is echoed back so the caller knows the bound applied. ``None`` when the
    athlete is absent.
    """
    clamped_hops = max(1, min(int(hops), _MAX_HOPS))
    rows = db.run_read(
        NEIGHBORHOOD_CYPHER,
        id=ath_id(athlete_id),
        max_events=_MAX_EVENTS,
        max_rivals=_MAX_RIVALS,
    )
    if not rows:
        return None
    row = rows[0]
    events = [e for e in (row.get("events") or []) if e and e.get("id")]
    rivals = [r for r in (row.get("rivals") or []) if r and r.get("id")]

    a_id = str(row["id"])
    nodes: list[dict[str, Any]] = [
        {"id": a_id, "label": row.get("name") or a_id, "type": "athlete"}
    ]
    edges: list[dict[str, Any]] = []
    seen: set[str] = {a_id}

    for ev in events:
        ev_id = str(ev["id"])
        if ev_id not in seen:
            nodes.append({"id": ev_id, "label": ev.get("name") or ev_id, "type": "event"})
            seen.add(ev_id)
        edges.append({"source": a_id, "target": ev_id, "type": "COMPETED_IN"})
        venue = ev.get("venue")
        if venue:
            ven_id = f"ven:{venue}"
            if ven_id not in seen:
                nodes.append({"id": ven_id, "label": venue, "type": "venue"})
                seen.add(ven_id)
            edges.append({"source": ev_id, "target": ven_id, "type": "HELD_AT"})

    for rv in rivals:
        rv_id = str(rv["id"])
        if rv_id not in seen:
            nodes.append({"id": rv_id, "label": rv.get("name") or rv_id, "type": "rival"})
            seen.add(rv_id)
        edges.append({"source": a_id, "target": rv_id, "type": "FACED", "count": rv.get("count")})

    return {
        "athlete": {"id": a_id, "name": row.get("name")},
        "hops": clamped_hops,
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# U1 — head-to-head
# ---------------------------------------------------------------------------

#: The directed FACED aggregate from athlete a → athlete b, plus the symmetric
#: edge so we can surface both directions of the (deduplicated) rivalry.
HEAD_TO_HEAD_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$a}}), (b:{_ATHLETE} {{id:$b}}) "
    f"OPTIONAL MATCH (a)-[f:{_FACED}]->(b) "
    "RETURN a.id AS a_id, a.name AS a_name, b.id AS b_id, b.name AS b_name, "
    "f.count AS count, f.round_ids AS round_ids, "
    "toString(f.first_date) AS first_date, toString(f.last_date) AS last_date"
)


def head_to_head(a_raw: str, b_raw: str) -> dict[str, Any] | None:
    """Return the FACED aggregate between two athletes, or ``None`` if either absent.

    The ``faced`` block is ``None`` when the pair has no recorded head-to-head
    (they both exist but never met in a final/semi). A short ``summary`` string
    describes the rivalry for display.
    """
    rows = db.run_read(HEAD_TO_HEAD_CYPHER, a=ath_id(a_raw), b=ath_id(b_raw))
    if not rows:
        return None
    row = rows[0]
    count = row.get("count")
    faced: dict[str, Any] | None
    if count:
        faced = {
            "count": count,
            "round_ids": list(row.get("round_ids") or []),
            "first_date": row.get("first_date"),
            "last_date": row.get("last_date"),
        }
    else:
        faced = None

    a_name = row.get("a_name") or row.get("a_id")
    b_name = row.get("b_name") or row.get("b_id")
    if faced:
        summary = (
            f"{a_name} and {b_name} have met {count} time(s) in finals/semis"
            f" between {faced['first_date']} and {faced['last_date']}."
        )
    else:
        summary = f"{a_name} and {b_name} have no recorded head-to-head meetings."

    return {
        "a": {"id": str(row["a_id"]), "name": row.get("a_name")},
        "b": {"id": str(row["b_id"]), "name": row.get("b_name")},
        "faced": faced,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# U2 — venue clusters
# ---------------------------------------------------------------------------

#: Venues ranked by repeated co-competition: distinct athletes seen there and
#: the number of events held at the venue (repeat visits).
VENUE_CLUSTERS_CYPHER = (
    f"MATCH (v:{_VENUE})<-[:{_HELD_AT}]-(e:{_EVENT}) "
    f"OPTIONAL MATCH (a:{_ATHLETE})-[:{_COMPETED_IN}]->(:{_PERFORMANCE})"
    f"-[:{_OF_ROUND}]->(:{_ROUND})-[:{_OF_EVENT}]->(e) "
    "WITH v, count(DISTINCT e) AS events, count(DISTINCT a) AS athletes "
    "RETURN v.name AS venue, events AS event_count, athletes AS athlete_count "
    "ORDER BY athlete_count DESC, event_count DESC, venue ASC "
    "LIMIT $limit"
)


def venue_clusters() -> list[dict[str, Any]]:
    """Return venues ranked by repeated co-competition (distinct athletes/events)."""
    rows = db.run_read(VENUE_CLUSTERS_CYPHER, limit=_MAX_CLUSTERS)
    return [
        {
            "venue": r.get("venue"),
            "event_count": r.get("event_count") or 0,
            "athlete_count": r.get("athlete_count") or 0,
        }
        for r in rows
        if r.get("venue")
    ]


# ---------------------------------------------------------------------------
# U3 — jetlagged underperformers
# ---------------------------------------------------------------------------

#: Join low-rested RestednessState to the representative Performance's
#: elo_residual via the HAD_STATE + AT_EVENT edges (clean edge join — no string
#: concatenation). residual > 0 ⇒ the athlete finished WORSE than expected.
#: Sorted by lowest rested / worst residual.
JETLAGGED_CYPHER = (
    f"MATCH (a:{_ATHLETE})-[:{_HAD_STATE}]->(rs:{_REST})-[:{_AT_EVENT}]->(e:{_EVENT}) "
    "WHERE rs.rested_index IS NOT NULL AND rs.rested_index < $rested_max "
    f"MATCH (a)-[:{_COMPETED_IN}]->(p:{_PERFORMANCE})"
    f"-[:{_OF_ROUND}]->(:{_ROUND})-[:{_OF_EVENT}]->(e) "
    "WHERE p.elo_residual IS NOT NULL AND p.elo_residual > 0 "
    "WITH a, e, rs, max(p.elo_residual) AS elo_residual "
    "RETURN a.id AS athlete_id, a.name AS athlete_name, "
    "e.id AS event_id, e.name AS event_name, "
    "toString(e.start_date) AS start_date, "
    "rs.rested_index AS rested_index, "
    "rs.travel_direction AS travel_direction, elo_residual "
    "ORDER BY rs.rested_index ASC, elo_residual DESC "
    "LIMIT $limit"
)

#: Below this rested_index a state is considered "jetlagged" (the index runs
#: 0..1 where 1.0 is fully rested). Tunable; kept conservative.
_RESTED_MAX = 0.85


def jetlagged_underperformers() -> list[dict[str, Any]]:
    """Return jetlagged athletes who underperformed, worst-rested first.

    Gracefully returns ``[]`` when no ``RestednessState`` / residual data exists
    yet (the travel + ELO-validation syncs have not run).
    """
    rows = db.run_read(JETLAGGED_CYPHER, rested_max=_RESTED_MAX, limit=_MAX_UNDERPERFORMERS)
    return [
        {
            "athlete_id": str(r["athlete_id"]),
            "athlete_name": r.get("athlete_name"),
            "event_id": str(r["event_id"]),
            "event_name": r.get("event_name"),
            "start_date": r.get("start_date"),
            "rested_index": r.get("rested_index"),
            "travel_direction": r.get("travel_direction"),
            "elo_residual": r.get("elo_residual"),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# U6b — season drivers (which athlete-seasons under-performed, and were they
# less rested that season?). Reads the SeasonSummary nodes built by sync.season.
# ---------------------------------------------------------------------------

#: Athlete-season summaries ranked by most under-performing first, using the
#: per-event-normalized ``mean_over_under`` (> 0 ⇒ finished worse than expected
#: per event) so the ranking isn't volume-biased toward athletes who entered
#: more events. The cumulative ``over_under`` sum is still surfaced as a "total
#: damage" measure, alongside the season's mean rested_index so the jet-lag link
#: is visible at season granularity.
SEASON_DRIVERS_CYPHER = (
    f"MATCH (a:{_ATHLETE})-[:{_HAD_SEASON}]->(s:{_SEASON}) "
    "WHERE s.mean_over_under IS NOT NULL "
    "RETURN a.id AS athlete_id, a.name AS athlete_name, "
    "s.season AS season, s.discipline AS discipline, "
    "s.over_under AS over_under, s.mean_over_under AS mean_over_under, "
    "s.mean_rested_index AS mean_rested_index, "
    "s.season_skill AS season_skill, s.season_consistency AS season_consistency, "
    "s.n_events AS n_events, s.n_upsets AS n_upsets "
    "ORDER BY s.mean_over_under DESC "
    "LIMIT $limit"
)


def season_drivers() -> list[dict[str, Any]]:
    """Return athlete-seasons ranked by most under-performing, with season restedness.

    Gracefully returns ``[]`` when no ``SeasonSummary`` nodes exist yet
    (the season aggregation sync has not run).
    """
    rows = db.run_read(SEASON_DRIVERS_CYPHER, limit=_MAX_UNDERPERFORMERS)
    return [
        {
            "athlete_id": str(r["athlete_id"]),
            "athlete_name": r.get("athlete_name"),
            "season": r.get("season"),
            "discipline": r.get("discipline"),
            "over_under": r.get("over_under"),
            "mean_over_under": r.get("mean_over_under"),
            "mean_rested_index": r.get("mean_rested_index"),
            "season_skill": r.get("season_skill"),
            "season_consistency": r.get("season_consistency"),
            "n_events": r.get("n_events"),
            "n_upsets": r.get("n_upsets"),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# U5 — athlete timeline
# ---------------------------------------------------------------------------

#: Events the athlete competed in (chronological), with the matching
#: RestednessState folded in via the AT_EVENT edge (clean edge join — no string
#: concatenation). TrainingSignal / InjuryEvent are pulled separately and may
#: be empty (L4/P5 not built yet).
TIMELINE_EVENTS_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$id}}) "
    f"MATCH (a)-[:{_COMPETED_IN}]->(:{_PERFORMANCE})"
    f"-[:{_OF_ROUND}]->(:{_ROUND})-[:{_OF_EVENT}]->(e:{_EVENT}) "
    f"OPTIONAL MATCH (e)-[:{_HELD_AT}]->(v:{_VENUE}) "
    f"OPTIONAL MATCH (a)-[:{_HAD_STATE}]->(rs:{_REST})-[:{_AT_EVENT}]->(e) "
    "WITH DISTINCT e, v, rs "
    "RETURN e.id AS event_id, e.name AS event_name, "
    "toString(e.start_date) AS start_date, e.discipline AS discipline, "
    "v.name AS venue, rs.rested_index AS rested_index, "
    "rs.travel_direction AS travel_direction "
    "ORDER BY e.start_date ASC"
)

#: Optional standalone TrainingSignal nodes (none exist until L4/P5).
TIMELINE_SIGNALS_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$id}})-[:{_HAS_SIGNAL}]->(s:{_SIGNAL}) "
    "RETURN s.id AS id, toString(s.date) AS date, s.kind AS kind "
    "ORDER BY s.date ASC"
)

#: Optional standalone InjuryEvent nodes (none exist until L4/P5).
TIMELINE_INJURIES_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$id}})-[:{_HAD_INJURY}]->(i:{_INJURY}) "
    "RETURN i.id AS id, toString(i.date) AS date, i.kind AS kind "
    "ORDER BY i.date ASC"
)


def athlete_timeline(athlete_id: str) -> dict[str, Any] | None:
    """Return the merged chronological timeline for *athlete_id*, or ``None``.

    Shape::

        {"athlete_id", "events": [...], "training_signals": [...],
         "injuries": [...]}

    Each event row folds in the matching ``RestednessState`` (``rested_index`` /
    ``travel_direction`` may be ``None`` when the travel sync has not run).
    ``training_signals`` / ``injuries`` are empty lists until L4/P5 is built —
    their absence never errors. ``None`` only when the athlete node is absent.
    """
    node_id = ath_id(athlete_id)
    event_rows = db.run_read(TIMELINE_EVENTS_CYPHER, id=node_id)
    # An athlete with no competed events still exists — distinguish "no athlete"
    # from "no events" via a cheap existence check only when there are no rows.
    if not event_rows and not _athlete_exists(node_id):
        return None

    events = [
        {
            "event_id": str(r["event_id"]),
            "event_name": r.get("event_name"),
            "start_date": r.get("start_date"),
            "discipline": r.get("discipline"),
            "venue": r.get("venue"),
            "rested_index": r.get("rested_index"),
            "travel_direction": r.get("travel_direction"),
        }
        for r in event_rows
        if r.get("event_id")
    ]
    signals = db.run_read(TIMELINE_SIGNALS_CYPHER, id=node_id)
    injuries = db.run_read(TIMELINE_INJURIES_CYPHER, id=node_id)

    return {
        "athlete_id": node_id,
        "events": events,
        "training_signals": list(signals or []),
        "injuries": list(injuries or []),
    }


#: Existence probe used only to tell "athlete absent" (→ 404) apart from
#: "athlete present but has no events" (→ empty timeline).
ATHLETE_EXISTS_CYPHER = f"MATCH (a:{_ATHLETE} {{id:$id}}) RETURN a.id AS id LIMIT 1"


def _athlete_exists(node_id: str) -> bool:
    """Return whether an Athlete node with *node_id* exists."""
    return bool(db.run_read(ATHLETE_EXISTS_CYPHER, id=node_id))
