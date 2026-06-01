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
        "mean_over_under": 0.8,
        "mean_rested_index": 0.55,
        "season_skill": 0.3,
        "season_consistency": 1.2,
        "n_events": 5,
        "n_upsets": 1,
    }
]

_MC_SUMMARY_ROWS = [
    {
        "total": 23256,
        "mean_result_percentile": 0.5012,
        "mean_surprisal": 2.13,
        "mean_p_win": 0.041,
        "mean_rank_std": 3.27,
    }
]

#: A deliberately *sparse* histogram (bins 0, 5, 9 only) so the test can confirm
#: mc_summary fills the missing bins with 0 and always returns all ten.
_MC_CALIBRATION_ROWS = [
    {"bin": 0, "count": 1200},
    {"bin": 5, "count": 2400},
    {"bin": 9, "count": 800},
]

_MC_PERF_ROWS = [
    {
        "athlete_id": _ATHLETE_ID,
        "athlete_name": _ATHLETE_NAME,
        "event_id": "evt:2",
        "event_name": "World Cup Bern",
        "start_date": "2024-07-01",
        "discipline": "B",
        "round_type": "final",
        "field_size": 8,
        "actual_rank": 6,
        "expected_rank_mc": 2.1,
        "elo_residual": 3.5,
        "elo_residual_mc": 3.9,
        "result_percentile": 0.94,
        "surprisal": 4.2,
        "p_win": 0.31,
        "p_podium": 0.72,
        "rank_std": 1.4,
        "pmf_entropy": 1.9,
    }
]

#: mc_performances builds its cypher as base + the allowlisted ORDER BY clause +
#: LIMIT, so the fake driver must be keyed by that exact composed string. We
#: compose it from the module's own pieces so the test can't drift from the impl.
_MC_PERF_UPSETS_CYPHER = (
    queries.MC_PERFORMANCES_BASE + f"ORDER BY {queries._MC_SORTS['upsets']} LIMIT $limit"
)

_READ_RESULTS: dict[str, list[dict[str, Any]]] = {
    queries.PROFILE_CYPHER: _PROFILE_ROWS,
    queries.NEIGHBORHOOD_CYPHER: _NEIGHBORHOOD_ROWS,
    queries.HEAD_TO_HEAD_CYPHER: _HEAD_TO_HEAD_ROWS,
    queries.VENUE_CLUSTERS_CYPHER: _VENUE_CLUSTER_ROWS,
    queries.JETLAGGED_CYPHER: _JETLAGGED_ROWS,
    queries.SEASON_DRIVERS_CYPHER: _SEASON_DRIVERS_ROWS,
    queries.MC_SUMMARY_CYPHER: _MC_SUMMARY_ROWS,
    queries.MC_CALIBRATION_CYPHER: _MC_CALIBRATION_ROWS,
    _MC_PERF_UPSETS_CYPHER: _MC_PERF_ROWS,
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
    queries.MC_SUMMARY_CYPHER,
    queries.MC_CALIBRATION_CYPHER,
    _MC_PERF_UPSETS_CYPHER,
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
    assert row["mean_over_under"] == pytest.approx(0.8)
    assert row["mean_rested_index"] == pytest.approx(0.55)


def test_season_drivers_orders_by_normalized_over_under() -> None:
    """Ranking uses the volume-fair mean_over_under, not the cumulative sum."""
    assert "ORDER BY s.mean_over_under DESC" in queries.SEASON_DRIVERS_CYPHER
    assert "ORDER BY s.over_under DESC" not in queries.SEASON_DRIVERS_CYPHER


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
# MC — Monte-Carlo placement distribution dashboard
# ---------------------------------------------------------------------------


def test_mc_summary_happy_path(client: TestClient) -> None:
    resp = client.get("/insights/mc-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 23256
    assert body["mean_result_percentile"] == pytest.approx(0.5012)
    assert body["mean_surprisal"] == pytest.approx(2.13)
    # Always exactly 10 contiguous bins, with the unseeded ones filled to 0.
    calib = body["calibration"]
    assert len(calib) == 10
    assert [c["bin_lo"] for c in calib] == pytest.approx([i / 10 for i in range(10)])
    by_bin = {round(c["bin_lo"] * 10): c["count"] for c in calib}
    assert by_bin[0] == 1200 and by_bin[5] == 2400 and by_bin[9] == 800
    assert by_bin[1] == 0 and by_bin[8] == 0  # missing bins filled


def test_mc_summary_no_data_returns_zeros(client: TestClient) -> None:
    """No MC props yet ⇒ total 0, all-zero histogram, None means — never an error."""
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/insights/mc-summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["mean_result_percentile"] is None
        assert len(body["calibration"]) == 10
        assert all(c["count"] == 0 for c in body["calibration"])
    finally:
        db._driver = original


def test_mc_performances_happy_path(client: TestClient) -> None:
    resp = client.get("/insights/mc-performances?sort=upsets&limit=50")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["athlete_id"] == _ATHLETE_ID
    assert row["round_type"] == "final"
    assert row["actual_rank"] == 6
    assert row["result_percentile"] == pytest.approx(0.94)
    assert row["surprisal"] == pytest.approx(4.2)
    # Both outcome variables surface side by side for comparison.
    assert row["elo_residual"] == pytest.approx(3.5)
    assert row["elo_residual_mc"] == pytest.approx(3.9)


def test_mc_performances_unknown_sort_falls_back_to_default(client: TestClient) -> None:
    """An out-of-allowlist sort resolves to the default (upsets) clause, not an error."""
    resp = client.get("/insights/mc-performances?sort=DROP%20TABLE")
    assert resp.status_code == 200
    # Served by the seeded default-sort cypher ⇒ rows come back (no injection).
    assert len(resp.json()["rows"]) == 1


def test_mc_performances_no_data_returns_empty(client: TestClient) -> None:
    original = db._driver
    db._driver = _empty_driver()
    try:
        resp = client.get("/insights/mc-performances")
        assert resp.status_code == 200
        assert resp.json() == {"rows": []}
    finally:
        db._driver = original


def test_mc_sort_clauses_are_static_literals() -> None:
    """The sort allowlist holds only fixed ``p.<field> ASC|DESC`` clauses.

    The user-supplied ``sort`` selects a key; the *value* (never the raw input)
    is interpolated, so there is no injection surface. Guard that every clause is
    a plain property ordering.
    """
    for clause in queries._MC_SORTS.values():
        assert clause.startswith("p.")
        assert clause.endswith(" ASC") or clause.endswith(" DESC")


def test_mc_performances_limit_clamped_to_max() -> None:
    """An over-large limit is clamped to _MAX_MC before hitting the database."""
    captured: dict[str, Any] = {}

    original = db._driver
    db._driver = FakeNeo4jDriver(read_results={_MC_PERF_UPSETS_CYPHER: _MC_PERF_ROWS})

    # Wrap db.run_read to capture the bound limit param.
    real_run_read = db.run_read

    def _spy(cypher: str, **params: Any) -> list[dict[str, Any]]:
        captured.update(params)
        return real_run_read(cypher, **params)

    db.run_read = _spy  # type: ignore[assignment]
    try:
        queries.mc_performances(sort="upsets", limit=10_000)
        assert captured["limit"] == queries._MAX_MC
    finally:
        db.run_read = real_run_read  # type: ignore[assignment]
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
