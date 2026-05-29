"""Tests for the GraphRAG ``/ask`` endpoint and ``api.rag`` helpers.

Pattern mirrors ``tests/test_api.py``: a module-scoped ``client`` fixture swaps
``api.db._driver`` for a seeded ``FakeNeo4jDriver`` (from ``tests/conftest.py``)
whose ``read_results`` map is keyed by the *exact* Cypher strings ``api.rag``
issues, then wraps ``api.index.app`` in a ``TestClient`` and restores the
original driver on teardown. NO live Neo4j or Groq connection is ever made in
the default suite (``GROQ_API_KEY`` is unset, so the graph-only fallback runs).

The single ``@pytest.mark.network`` test exercises the real Groq path; the
parsing/shaping of a Groq response is additionally asserted fully offline
against the checked-in ``tests/fixtures/groq_ask_response.json`` fixture.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import db, rag
from api.index import app
from tests.conftest import FakeNeo4jDriver

_FIXTURES = Path(__file__).parent / "fixtures"

# Seeded athlete used across the default (offline) suite.
_ATHLETE_ID = "ath:1"
_ATHLETE_NAME = "Ada"

# Read-results map keyed by the EXACT cypher api.rag issues, with the params
# the production code binds. The fake session matches on cypher text, returning
# these rows so resolve/expand run end-to-end without a live database.
_RESOLVE_ROWS = [{"id": _ATHLETE_ID, "name": _ATHLETE_NAME}]
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
            {"id": "ath:2", "name": "Bea", "meetings": 2},
            {"id": "ath:3", "name": "Cleo", "meetings": 1},
        ],
    }
]

_READ_RESULTS: dict[str, list[dict[str, Any]]] = {
    rag.RESOLVE_CYPHER: _RESOLVE_ROWS,
    rag.NEIGHBORHOOD_CYPHER: _NEIGHBORHOOD_ROWS,
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Yield a TestClient backed by a FakeNeo4jDriver seeded with read rows."""
    original = db._driver
    db._driver = FakeNeo4jDriver(nodes=10, relationships=20, read_results=_READ_RESULTS)
    # Ensure the default suite takes the graph-only fallback path.
    had_key = "GROQ_API_KEY" in os.environ
    saved_key = os.environ.pop("GROQ_API_KEY", None)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        db._driver = original
        if had_key and saved_key is not None:
            os.environ["GROQ_API_KEY"] = saved_key


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


def test_resolve_finds_seeded_athlete(client: TestClient) -> None:
    entity = rag.resolve_athlete("ada")
    assert entity == {"id": _ATHLETE_ID, "name": _ATHLETE_NAME}


def test_resolve_unknown_returns_none(client: TestClient) -> None:
    # The fake only seeds the exact resolve cypher with Ada's row, but the
    # resolver returns whatever the query yields; an empty driver yields none.
    empty = FakeNeo4jDriver(read_results={rag.RESOLVE_CYPHER: []})
    original = db._driver
    db._driver = empty
    try:
        assert rag.resolve_athlete("Nobody") is None
    finally:
        db._driver = original


def test_resolve_blank_returns_none(client: TestClient) -> None:
    assert rag.resolve_athlete("   ") is None


# ---------------------------------------------------------------------------
# /ask endpoint — graph-only fallback (no GROQ_API_KEY)
# ---------------------------------------------------------------------------


def test_ask_returns_200_with_fallback(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "How has Ada done?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"answer", "entities", "subgraph"}
    # No key configured → deterministic graph-only fallback.
    assert body["used_llm"] is False
    assert _ATHLETE_NAME in body["answer"]


def test_ask_resolves_entity(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Tell me about Ada"})
    body = resp.json()
    assert body["entities"] == [{"id": _ATHLETE_ID, "name": _ATHLETE_NAME}]


def test_ask_subgraph_shape(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Ada rivals"})
    sg = resp.json()["subgraph"]
    assert set(sg) == {"context", "nodes", "edges"}
    # Athlete + event + venue + two rivals = 5 nodes.
    types = sorted(n["type"] for n in sg["nodes"])
    assert types == ["athlete", "event", "rival", "rival", "venue"]
    # Athlete node is present and central.
    athlete_nodes = [n for n in sg["nodes"] if n["type"] == "athlete"]
    assert athlete_nodes == [{"id": _ATHLETE_ID, "label": _ATHLETE_NAME, "type": "athlete"}]
    # Edge types cover the three relationship kinds.
    edge_types = {e["type"] for e in sg["edges"]}
    assert edge_types == {"COMPETED_IN", "HELD_AT", "FACED"}
    # Context text mentions the athlete and a rival.
    assert _ATHLETE_NAME in sg["context"]
    assert "Bea" in sg["context"]


def test_ask_unknown_entity_404(client: TestClient) -> None:
    """Unresolvable question → 404 (the fake yields no resolve rows for it)."""
    empty = FakeNeo4jDriver(read_results={rag.RESOLVE_CYPHER: []})
    original = db._driver
    db._driver = empty
    try:
        resp = client.post("/ask", json={"question": "Who is Nobody McNobody?"})
        assert resp.status_code == 404
    finally:
        db._driver = original


# ---------------------------------------------------------------------------
# Subgraph / context helpers (pure)
# ---------------------------------------------------------------------------


def test_subgraph_to_graph_no_duplicate_nodes() -> None:
    sub = {
        "athlete": {"id": "ath:1", "name": "Ada"},
        "events": [
            {"id": "evt:1", "name": "WC A", "venue": "V"},
            {"id": "evt:2", "name": "WC B", "venue": "V"},  # same venue → one node
        ],
        "rivals": [{"id": "ath:2", "name": "Bea", "meetings": 3}],
    }
    graph = rag.subgraph_to_graph(sub)
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))  # no duplicate nodes (shared venue collapsed)
    assert "ven:V" in ids


def test_fallback_summary_mentions_top_rival() -> None:
    sub = {
        "athlete": {"id": "ath:1", "name": "Ada"},
        "events": [{"id": "evt:1", "name": "WC A"}],
        "rivals": [
            {"id": "ath:2", "name": "Bea", "meetings": 2},
            {"id": "ath:3", "name": "Cleo", "meetings": 5},
        ],
    }
    summary = rag._fallback_summary(sub)
    assert "Cleo" in summary  # most-frequent opponent
    assert "5 meeting" in summary


# ---------------------------------------------------------------------------
# Groq parsing — offline against the checked-in fixture
# ---------------------------------------------------------------------------


def test_parse_groq_answer_from_fixture() -> None:
    payload = json.loads((_FIXTURES / "groq_ask_response.json").read_text())
    answer = rag.parse_groq_answer(payload)
    assert isinstance(answer, str)
    assert answer.startswith("Ada has competed")


def test_parse_groq_answer_empty_choices_raises() -> None:
    with pytest.raises(ValueError):
        rag.parse_groq_answer({"choices": []})


def test_groq_fixture_shape() -> None:
    """The checked-in Groq fixture has the keys parse_groq_answer relies on."""
    payload = json.loads((_FIXTURES / "groq_ask_response.json").read_text())
    assert payload["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Live Groq path — deselected by default (requires GROQ_API_KEY + network)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_ask_live_groq_path() -> None:
    """End-to-end ``ask`` against a live Groq, using a seeded fake graph.

    Requires a real ``GROQ_API_KEY`` and network access. Asserts that the LLM
    path is taken and returns a non-empty answer grounded in the subgraph.
    """
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    original = db._driver
    db._driver = FakeNeo4jDriver(read_results=_READ_RESULTS)
    try:
        result = rag.ask("Who has Ada faced?")
    finally:
        db._driver = original
    assert result["used_llm"] is True
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()
    assert result["entities"] == [{"id": _ATHLETE_ID, "name": _ATHLETE_NAME}]
