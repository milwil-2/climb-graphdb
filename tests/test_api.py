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


# ---------------------------------------------------------------------------
# Developer-only MC dashboard gate (hidden in Vercel production).
# ---------------------------------------------------------------------------

_MC_ROUTES = ("/mc", "/insights/mc-summary", "/insights/mc-performances")


@pytest.mark.parametrize("route", _MC_ROUTES)
def test_mc_routes_404_in_vercel_production(
    client: TestClient, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Vercel production the MC dashboard surfaces are hidden (404, not 403)."""
    monkeypatch.setenv("VERCEL_ENV", "production")
    resp = client.get(route)
    assert resp.status_code == 404


def test_mc_dashboard_served_off_production(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locally / on preview (VERCEL_ENV not 'production') the dashboard is served."""
    monkeypatch.delenv("VERCEL_ENV", raising=False)
    resp = client.get("/mc")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.parametrize("vercel_env", ["production", "preview", "development"])
def test_mc_dashboard_not_a_public_static_asset(
    client: TestClient, vercel_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard page lives outside the public ``/static`` mount in every env.

    It must never be fetchable as a raw static file (which would bypass the gate).
    """
    monkeypatch.setenv("VERCEL_ENV", vercel_env)
    assert client.get("/static/mc.html").status_code == 404
