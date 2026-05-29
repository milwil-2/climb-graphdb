"""Live-service (network) tests — deselected by default.

These hit real external services (a live Neo4j/Aura instance via ``api.db``)
and are marked ``@pytest.mark.network`` so the default ``pytest`` run skips
them (``addopts = -m 'not network'``). Run them explicitly with::

    uv run pytest -m network

The expected response *shape* lives in ``tests/fixtures/graph_stats_response.json``
and is asserted offline by ``tests/test_api.py::test_graph_stats_fixture_shape``,
so the parsing/contract is covered even when this test is deselected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import db

_FIXTURE = Path(__file__).parent / "fixtures" / "graph_stats_response.json"


@pytest.mark.network
def test_live_graph_stats_matches_contract() -> None:
    """``graph_stats()`` against a live Neo4j returns the documented contract.

    Requires real NEO4J_* env vars and a reachable instance. The live counts
    are not asserted (they change); only the response shape is — matching the
    checked-in fixture parsed offline elsewhere.
    """
    expected_keys = set(json.loads(_FIXTURE.read_text()))

    stats = db.graph_stats()

    assert set(stats) == expected_keys
    assert isinstance(stats["nodes"], int)
    assert isinstance(stats["relationships"], int)
    assert stats["nodes"] >= 0
    assert stats["relationships"] >= 0
