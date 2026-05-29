"""API/integration tests for the FastAPI app (api.index).

Pattern: a module-scoped ``client`` fixture swaps the cached Neo4j driver in
``api.db`` for a seeded ``FakeNeo4jDriver`` (from ``tests/conftest.py``), wraps
``api.index.app`` in a FastAPI ``TestClient``, and restores the original driver
on teardown. NO live Neo4j connection is ever made.

The companion ``test_graph_stats_fixture_shape`` parses a checked-in JSON
fixture so the response *shape* is also validated fully offline (and serves as
the canonical expected payload for the ``-m network`` live test).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import db
from api.index import app
from tests.conftest import FakeNeo4jDriver

_FIXTURES = Path(__file__).parent / "fixtures"
_SEEDED_NODES = 1280
_SEEDED_RELS = 5432


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Yield a TestClient backed by a seeded FakeNeo4jDriver.

    The original ``api.db._driver`` is restored on teardown so module ordering
    can never leak a fake driver into another test.
    """
    original = db._driver
    db._driver = FakeNeo4jDriver(nodes=_SEEDED_NODES, relationships=_SEEDED_RELS)
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        db._driver = original


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_graph_stats_returns_seeded_counts(client: TestClient) -> None:
    resp = client.get("/graph/stats")
    assert resp.status_code == 200
    assert resp.json() == {"nodes": _SEEDED_NODES, "relationships": _SEEDED_RELS}


def test_graph_stats_fixture_shape() -> None:
    """The checked-in fixture has the exact keys/types the route returns."""
    payload = json.loads((_FIXTURES / "graph_stats_response.json").read_text())
    assert set(payload) == {"nodes", "relationships"}
    assert isinstance(payload["nodes"], int)
    assert isinstance(payload["relationships"], int)
