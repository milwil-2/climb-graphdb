"""Tests for the U1–U5 read endpoints (``api.queries`` + ``api.index`` routes).

Pattern mirrors ``tests/test_api.py`` / ``tests/test_rag.py``: a module-scoped
``client`` fixture swaps ``api.db._driver`` for a seeded ``FakeNeo4jDriver``
(from ``tests/conftest.py``) whose ``read_results`` map is keyed by the *exact*
Cypher strings ``api.queries`` issues, then wraps ``api.index.app`` in a
``TestClient`` and restores the original driver on teardown. NO live Neo4j
connection is ever made.

Each endpoint is exercised for its happy path plus its empty/404 path — in
particular U3 (no ``RestednessState`` ⇒ empty) and U5 (no signals ⇒ events /
restedness only, never an error).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import db, queries
from api.index import app
from tests.conftest import FakeNeo4jDriver

# ---------------------------------------------------------------------------
# Seeded rows keyed by the EXACT cypher api.queries issues.
# ---------------------------------------------------------------------------

_ATHLETE_ID = "ath:1"
_ATHLETE_NAME = "Ada"

_PROFILE_ROWS = [
    {
        "id": _ATHLETE_ID,
        "name": _ATHLETE_NAME,
        "nationality": "USA",
        "gender": "F",
        "year_of_birth": 1999,
        "ratings": [
            {"discipline": "L", "mu": 1600.0, "sigma": 200.0, "n_events": 5, "provisional": False}
        ],
        "recent_events": [
            {
                "id": "evt:2",
                "name": "World Cup Bern",
                "start_date": "2024-07-01",
                "discipline": "B",
                "venue": "Bern Arena",
            },
            {
                "id": "evt:1",
                "name": "World Cup Innsbruck",
                "start_date": "2024-06-01",
                "discipline": "L",
                "venue": "Kletterzentrum Innsbruck",
            },
        ],
    }
]

_NEIGHBORHOOD_ROWS = [
    {
        "id": _ATHLETE_ID,
        "name": _ATHLETE_NAME,
        "events": [
            {
                "id": "evt:1",
                "name": "World Cup Innsbruck",
                "start_date": "2024-06-01",
                "discipline": "L",
                "venue": "Kletterzentrum Innsbruck",
            }
        ],
        "rivals": [
            {"id": "ath:2", "name": "Bea", "count": 2},
            {"id": "ath:3", "name": "Cleo", "count": 1},
        ],
    }
]

_HEAD_TO_HEAD_ROWS = [
    {
        "a_id": _ATHLETE_ID,
        "a_name": _ATHLETE_NAME,
        "b_id": "ath:2",
        "b_name": "Bea",
        "count": 2,
        "round_ids": [3, 4],
        "first_date": "2024-06-01",
        "last_date": "2024-07-01",
    }
]

_VENUE_CLUSTER_ROWS = [
    {"venue": "Kletterzentrum Innsbruck", "event_count": 3, "athlete_count": 12},
    {"venue": "Bern Arena", "event_count": 1, "athlete_count": 4},
]

_JETLAGGED_ROWS = [
    {
        "athlete_id": _ATHLETE_ID,
        "athlete_name": _ATHLETE_NAME,
        "event_id": "evt:2",
        "event_name": "World Cup Bern",
        "start_date": "2024-07-01",
        "rested_index": 0.42,
        "travel_direction": "east",
        "elo_residual": 3.5,
    }
]

_TIMELINE_EVENT_ROWS = [
    {
        "event_id": "evt:1",
        "event_name": "World Cup Innsbruck",
        "start_date": "2024-06-01",
        "discipline": "L",
        "venue": "Kletterzentrum Innsbruck",
        "rested_index": 0.9,
        "travel_direction": "none",
    },
    {
        "event_id": "evt:2",
        "event_name": "World Cup Bern",
        "start_date": "2024-07-01",
        "discipline": "B",
        "venue": "Bern Arena",
        "rested_index": 0.42,
        "travel_direction": "east",
    },
]

_EXISTS_ROWS = [{"id": _ATHLETE_ID}]

_SEASON_DRIVERS_ROWS = [
    {
        "athlete_id": _ATHLETE_ID,
        "athlete_name": _ATHLETE_NAME,
        "season": 2024,
        "discipline": "L",
        "over_under": 4.0,
        "mean_rested_index": 0.55,
        "season_skill": 0.3,
        "season_consistency": 1.2,
        "n_events": 5,
        "n_upsets": 1,
    }
]

_READ_RESULTS: dict[str, list[dict[str, Any]]] = {
    queries.PROFILE_CYPHER: _PROFILE_ROWS,
    queries.NEIGHBORHOOD_CYPHER: _NEIGHBORHOOD_ROWS,
    queries.HEAD_TO_HEAD_CYPHER: _HEAD_TO_HEAD_ROWS,
    queries.VENUE_CLUSTERS_CYPHER: _VENUE_CLUSTER_ROWS,
    queries.JETLAGGED_CYPHER: _JETLAGGED_ROWS,
    queries.SEASON_DRIVERS_CYPHER: _SEASON_DRIVERS_ROWS,
    queries.TIMELINE_EVENTS_CYPHER: _TIMELINE_EVENT_ROWS,
    queries.ATHLETE_EXISTS_CYPHER: _EXISTS_ROWS,
    # TrainingSignal / InjuryEvent: seeded empty ⇒ the L4/P5-era nodes don't
    # exist yet (the fake falls back to a non-iterable count-result otherwise).
    queries.TIMELINE_SIGNALS_CYPHER: [],
    queries.TIMELINE_INJURIES_CYPHER: [],
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Yield a TestClient backed by a FakeNeo4jDriver seeded with read rows."""
    original = db._driver
    db._driver = FakeNeo4jDriver(nodes=50, relationships=120, read_results=_READ_RESULTS)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        db._driver = original


#: Every read cypher api.queries can issue, seeded to NO rows. The fake driver
#: falls back to a (non-iterable) count-result for any *unseeded* cypher, so an
#: "empty" driver must map each read query explicitly to ``[]``.
_ALL_QUERY_CYPHERS = (
    queries.PROFILE_CYPHER,
    queries.NEIGHBORHOOD_CYPHER,
    queries.HEAD_TO_HEAD_CYPHER,
    queries.VENUE_CLUSTERS_CYPHER,
    queries.JETLAGGED_CYPHER,
    queries.SEASON_DRIVERS_CYPHER,
    queries.TIMELINE_EVENTS_CYPHER,
    queries.TIMELINE_SIGNALS_CYPHER,
    queries.TIMELINE_INJURIES_CYPHER,
    queries.ATHLETE_EXISTS_CYPHER,
)


def _empty_driver() -> FakeNeo4jDriver:
    """A driver that returns no rows for any query (all endpoints ⇒ empty/404)."""
    return FakeNeo4jDriver(read_results={c: [] for c in _ALL_QUERY_CYPHERS})


# ---------------------------------------------------------------------------
# U4 — athlete profile
# ---------------------------------------------------------------------------


def test_athlete_profile_happy_path(client: TestClient) -> None:
    resp = client.get("/athlete/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == _ATHLETE_ID
    assert body["name"] == _ATHLETE_NAME
    assert body["nationality"] == "USA"
    assert len(body["ratings"]) == 1
    assert body["ratings"][0]["discipline"] == "L"
    # Recent events surfaced (most-recent-first as seeded).
    assert [e["id"] for e in body["recent_events"]] == ["evt:2", "evt:1"]


def test_athlete_profile_404_when_absent(client: TestClient) -> None:
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/athlete/999")
        assert resp.status_code == 404
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# U4 — neighborhood
# ---------------------------------------------------------------------------


def test_neighborhood_happy_path(client: TestClient) -> None:
    resp = client.get("/athlete/1/neighborhood?hops=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["athlete"] == {"id": _ATHLETE_ID, "name": _ATHLETE_NAME}
    assert body["hops"] == 2
    # athlete + event + venue + two rivals = 5 nodes.
    types = sorted(n["type"] for n in body["nodes"])
    assert types == ["athlete", "event", "rival", "rival", "venue"]
    edge_types = {e["type"] for e in body["edges"]}
    assert edge_types == {"COMPETED_IN", "HELD_AT", "FACED"}


def test_neighborhood_hops_clamped(client: TestClient) -> None:
    resp = client.get("/athlete/1/neighborhood?hops=99")
    assert resp.status_code == 200
    assert resp.json()["hops"] == queries._MAX_HOPS


def test_neighborhood_404_when_absent(client: TestClient) -> None:
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/athlete/999/neighborhood")
        assert resp.status_code == 404
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# U1 — head-to-head
# ---------------------------------------------------------------------------


def test_head_to_head_happy_path(client: TestClient) -> None:
    resp = client.get("/head-to-head?a=1&b=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["a"]["id"] == _ATHLETE_ID
    assert body["b"]["id"] == "ath:2"
    assert body["faced"]["count"] == 2
    assert body["faced"]["round_ids"] == [3, 4]
    assert body["faced"]["first_date"] == "2024-06-01"
    assert "Ada" in body["summary"] and "Bea" in body["summary"]


def test_head_to_head_no_meetings(client: TestClient) -> None:
    """Both athletes exist but never met ⇒ faced is None, summary explains it."""
    rows = [
        {
            "a_id": _ATHLETE_ID,
            "a_name": _ATHLETE_NAME,
            "b_id": "ath:3",
            "b_name": "Cleo",
            "count": None,
            "round_ids": None,
            "first_date": None,
            "last_date": None,
        }
    ]
    original = db._driver
    db._driver = FakeNeo4jDriver(read_results={queries.HEAD_TO_HEAD_CYPHER: rows})
    try:
        resp = client.get("/head-to-head?a=1&b=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["faced"] is None
        assert "no recorded head-to-head" in body["summary"]
    finally:
        db._driver = original


def test_head_to_head_404_when_absent(client: TestClient) -> None:
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/head-to-head?a=1&b=999")
        assert resp.status_code == 404
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# U2 — venue clusters
# ---------------------------------------------------------------------------


def test_venue_clusters_happy_path(client: TestClient) -> None:
    resp = client.get("/venues/clusters")
    assert resp.status_code == 200
    clusters = resp.json()["clusters"]
    assert len(clusters) == 2
    assert clusters[0]["venue"] == "Kletterzentrum Innsbruck"
    assert clusters[0]["athlete_count"] == 12
    assert clusters[0]["event_count"] == 3


def test_venue_clusters_empty(client: TestClient) -> None:
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/venues/clusters")
        assert resp.status_code == 200
        assert resp.json() == {"clusters": []}
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# U3 — jetlagged underperformers
# ---------------------------------------------------------------------------


def test_jetlagged_happy_path(client: TestClient) -> None:
    resp = client.get("/insights/jetlagged-underperformers")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["athlete_id"] == _ATHLETE_ID
    assert row["event_id"] == "evt:2"
    assert row["rested_index"] == pytest.approx(0.42)
    assert row["elo_residual"] == pytest.approx(3.5)


def test_jetlagged_no_data_returns_empty(client: TestClient) -> None:
    """U3 with no RestednessState / residual data ⇒ empty list, never an error."""
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/insights/jetlagged-underperformers")
        assert resp.status_code == 200
        assert resp.json() == {"rows": []}
    finally:
        db._driver = original


def test_season_drivers_happy_path(client: TestClient) -> None:
    resp = client.get("/insights/season-drivers")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["athlete_id"] == _ATHLETE_ID
    assert row["season"] == 2024
    assert row["discipline"] == "L"
    assert row["over_under"] == pytest.approx(4.0)
    assert row["mean_rested_index"] == pytest.approx(0.55)


def test_season_drivers_no_data_returns_empty(client: TestClient) -> None:
    """U6b with no SeasonSummary nodes ⇒ empty list, never an error."""
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/insights/season-drivers")
        assert resp.status_code == 200
        assert resp.json() == {"rows": []}
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# U5 — timeline
# ---------------------------------------------------------------------------


def test_timeline_happy_path(client: TestClient) -> None:
    resp = client.get("/athlete/1/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["athlete_id"] == _ATHLETE_ID
    # Events chronological, with RestednessState folded in.
    assert [e["event_id"] for e in body["events"]] == ["evt:1", "evt:2"]
    assert body["events"][1]["rested_index"] == pytest.approx(0.42)
    # No L4/P5 nodes seeded ⇒ empty optional lists (NOT an error).
    assert body["training_signals"] == []
    assert body["injuries"] == []


def test_timeline_no_events_but_athlete_exists(client: TestClient) -> None:
    """Athlete present with no events ⇒ empty events, still 200 (existence probe)."""
    original = db._driver
    db._driver = FakeNeo4jDriver(
        read_results={
            queries.TIMELINE_EVENTS_CYPHER: [],
            queries.ATHLETE_EXISTS_CYPHER: _EXISTS_ROWS,
            queries.TIMELINE_SIGNALS_CYPHER: [],
            queries.TIMELINE_INJURIES_CYPHER: [],
        }
    )
    try:
        resp = client.get("/athlete/1/timeline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["events"] == []
        assert body["training_signals"] == []
        assert body["injuries"] == []
    finally:
        db._driver = original


def test_timeline_404_when_absent(client: TestClient) -> None:
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/athlete/999/timeline")
        assert resp.status_code == 404
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# Injection-safety: every label / rel interpolated in the module is in-vocab.
# (assert_label / assert_rel raise at import if not — this asserts the cyphers
# carry no raw user input by checking they bind via params only.)
# ---------------------------------------------------------------------------


def test_cyphers_use_bound_params_not_interpolation() -> None:
    """The athlete id never appears interpolated; it is always a $-bound param."""
    for cypher in (
        queries.PROFILE_CYPHER,
        queries.NEIGHBORHOOD_CYPHER,
        queries.TIMELINE_EVENTS_CYPHER,
        queries.ATHLETE_EXISTS_CYPHER,
    ):
        assert "$id" in cypher
    assert "$a" in queries.HEAD_TO_HEAD_CYPHER
    assert "$b" in queries.HEAD_TO_HEAD_CYPHER
