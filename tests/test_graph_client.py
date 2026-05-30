"""Tests for GraphClient batched-write shortfall detection (issue #43).

``GraphClient.merge_rels`` / ``merge_rel`` MATCH both endpoints by ``:Entity``
id before MERGEing the relationship. A row whose endpoint id matches no node
yields no MATCH and is **silently skipped**. These tests cover the guard that
surfaces such dropped edges by comparing the written count (``RETURN count(r)``)
against the number of rows submitted.

The Neo4j driver is faked at the IO boundary only: the fake reports a
configurable number of "written" relationships so the real shortfall/warning
logic in ``GraphClient`` runs without a live database.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from climber_network.graph.client import GraphClient, _rel_shortfall_warning

# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


def test_rel_shortfall_warning_none_when_all_written() -> None:
    assert _rel_shortfall_warning("FACED", expected=10, written=10) is None


def test_rel_shortfall_warning_none_when_overcount() -> None:
    # Defensive: written should never exceed expected, but never warn if it does.
    assert _rel_shortfall_warning("FACED", expected=10, written=11) is None


def test_rel_shortfall_warning_reports_drop() -> None:
    msg = _rel_shortfall_warning("HELD_AT", expected=10, written=7)
    assert msg is not None
    assert "HELD_AT" in msg
    assert "3" in msg  # dropped
    assert "10" in msg  # expected
    assert "7" in msg  # written


# ---------------------------------------------------------------------------
# Fake Neo4j driver (IO boundary only)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, written: int) -> None:
        self._written = written

    def single(self) -> dict[str, int]:
        return {"written": self._written}


class _FakeTx:
    def __init__(self, drop: int) -> None:
        self._drop = drop

    def run(self, cypher: str, **params: Any) -> _FakeResult:
        # merge_rels passes `rows`; merge_rel passes src_id/tgt_id (one edge).
        rows = params.get("rows")
        n = len(rows) if rows is not None else 1
        return _FakeResult(max(n - self._drop, 0))


class _FakeSession:
    def __init__(self, drop: int) -> None:
        self._drop = drop

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute_write(self, work: Any) -> Any:
        return work(_FakeTx(self._drop))


class _FakeDriver:
    """Reports `drop` fewer written rels than submitted, per chunk."""

    def __init__(self, drop: int) -> None:
        self._drop = drop

    def session(self) -> _FakeSession:
        return _FakeSession(self._drop)


def _client_with_drop(drop: int) -> GraphClient:
    client = GraphClient()
    client._driver = _FakeDriver(drop)  # type: ignore[assignment]
    return client


# ---------------------------------------------------------------------------
# merge_rels / merge_rel wiring
# ---------------------------------------------------------------------------


def test_merge_rels_warns_on_dropped_edges(caplog: pytest.LogCaptureFixture) -> None:
    client = _client_with_drop(drop=1)
    rows = [
        {"src_id": "ath:1", "tgt_id": "evt:1"},
        {"src_id": "ath:2", "tgt_id": "evt:2"},
        {"src_id": "ath:3", "tgt_id": "evt:bad"},  # matches no node
    ]
    with caplog.at_level(logging.WARNING, logger="climber_network.graph.client"):
        client.merge_rels("HELD_AT", rows)
    assert any("HELD_AT" in r.message and "dropped" in r.message.lower() for r in caplog.records)


def test_merge_rels_silent_when_all_matched(caplog: pytest.LogCaptureFixture) -> None:
    client = _client_with_drop(drop=0)
    rows = [
        {"src_id": "ath:1", "tgt_id": "evt:1"},
        {"src_id": "ath:2", "tgt_id": "evt:2"},
    ]
    with caplog.at_level(logging.WARNING, logger="climber_network.graph.client"):
        client.merge_rels("HELD_AT", rows)
    assert not any("HELD_AT" in r.message for r in caplog.records)


def test_merge_rel_warns_when_single_edge_dropped(caplog: pytest.LogCaptureFixture) -> None:
    client = _client_with_drop(drop=1)  # the lone edge matches nothing → written 0
    with caplog.at_level(logging.WARNING, logger="climber_network.graph.client"):
        client.merge_rel("ath:1", "HELD_AT", "evt:bad")
    assert any("HELD_AT" in r.message for r in caplog.records)
