"""Tests for the shared RestednessState correlation helper (climber_network.elo.rested).

``correlate_against_rested`` is the single implementation both outcome-variable
syncs (:mod:`sync.validate_elo` → ``elo_residual``, :mod:`sync.montecarlo` →
``result_percentile``) call. These tests exercise it directly with a tiny fake
item type + the shared ``FakeGraphClient`` (canned ``run_read`` keyed on the
exact ``REST_QUERY``): the join/grouping into overall / by_discipline /
by_travel_direction Pearson blocks, the truthy-key filtering, the caller-supplied
success-signal passthrough, row dropping on missing keys, and the graceful
``n = 0`` path.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from climber_network.elo.rested import REST_QUERY, correlate_against_rested
from tests.conftest import FakeGraphClient


@dataclass(frozen=True)
class _Item:
    """Minimal stand-in for a rep with the accessors the helper needs."""

    athlete_id: int
    event_id: int
    value: float
    discipline: str | None


def _report(client: FakeGraphClient, items: list[_Item], *, signal: str = "sig") -> dict:
    return correlate_against_rested(
        client,
        items,
        athlete_id=lambda it: it.athlete_id,
        event_id=lambda it: it.event_id,
        outcome=lambda it: it.value,
        discipline=lambda it: it.discipline,
        success_signal=signal,
    )


_REST_ROWS = [
    {
        "athlete_id": 1,
        "event_id": 1,
        "rested_index": 0.9,
        "discipline": "L",
        "travel_direction": "E",
    },
    {
        "athlete_id": 2,
        "event_id": 1,
        "rested_index": 0.7,
        "discipline": "L",
        "travel_direction": "E",
    },
    {
        "athlete_id": 3,
        "event_id": 1,
        "rested_index": 0.3,
        "discipline": "B",
        "travel_direction": "W",
    },
    {
        "athlete_id": 4,
        "event_id": 1,
        "rested_index": 0.1,
        "discipline": "B",
        "travel_direction": "W",
    },
]


def test_blocks_join_group_and_correlate() -> None:
    client = FakeGraphClient(read_results={REST_QUERY: _REST_ROWS})
    # Lower rested ↔ higher value → a clean negative overall correlation.
    items = [
        _Item(1, 1, 1.0, "L"),
        _Item(2, 1, 2.0, "L"),
        _Item(3, 1, 3.0, "B"),
        _Item(4, 1, 4.0, "B"),
    ]

    out = _report(client, items, signal="my-signal")

    assert out["overall"]["n"] == 4
    assert out["overall"]["pearson_r"] is not None
    assert out["overall"]["pearson_r"] < 0
    # Grouped by the item discipline + the row travel direction.
    assert set(out["by_discipline"]) == {"L", "B"}
    assert out["by_discipline"]["L"]["n"] == 2
    assert out["by_discipline"]["B"]["n"] == 2
    assert set(out["by_travel_direction"]) == {"E", "W"}
    assert out["by_travel_direction"]["E"]["n"] == 2
    # The success-signal string is passed straight through.
    assert out["success_signal"] == "my-signal"


def test_unmatched_items_dropped() -> None:
    client = FakeGraphClient(read_results={REST_QUERY: _REST_ROWS})
    # athlete 9 has no RestednessState row → dropped; (4 → event 2) also unmatched.
    items = [_Item(1, 1, 1.0, "L"), _Item(9, 1, 2.0, "L"), _Item(4, 2, 3.0, "B")]
    out = _report(client, items)
    assert out["overall"]["n"] == 1


def test_empty_or_none_discipline_skipped_in_breakdown() -> None:
    client = FakeGraphClient(
        read_results={
            REST_QUERY: [
                {
                    "athlete_id": 1,
                    "event_id": 1,
                    "rested_index": 0.5,
                    "discipline": "L",
                    "travel_direction": None,
                },
                {
                    "athlete_id": 2,
                    "event_id": 1,
                    "rested_index": 0.6,
                    "discipline": "L",
                    "travel_direction": "",
                },
            ]
        }
    )
    items = [_Item(1, 1, 1.0, None), _Item(2, 1, 2.0, "")]
    out = _report(client, items)
    # Both items join into overall, but neither contributes a breakdown key
    # (falsy discipline and falsy / None travel direction are skipped).
    assert out["overall"]["n"] == 2
    assert out["by_discipline"] == {}
    assert out["by_travel_direction"] == {}


def test_rows_missing_keys_are_dropped() -> None:
    client = FakeGraphClient(
        read_results={
            REST_QUERY: [
                # Missing rested_index → dropped from the rested map entirely.
                {"athlete_id": 1, "event_id": 1, "rested_index": None, "discipline": "L"},
                {"athlete_id": None, "event_id": 1, "rested_index": 0.5, "discipline": "L"},
                {
                    "athlete_id": 2,
                    "event_id": 1,
                    "rested_index": 0.5,
                    "discipline": "L",
                    "travel_direction": "E",
                },
            ]
        }
    )
    items = [_Item(1, 1, 1.0, "L"), _Item(2, 1, 2.0, "L")]
    out = _report(client, items)
    # Only athlete 2 survives the map; athlete 1's row had a null rested_index.
    assert out["overall"]["n"] == 1


def test_graceful_when_no_restedness() -> None:
    client = FakeGraphClient()  # run_read returns [] for REST_QUERY
    out = _report(client, [_Item(1, 1, 1.0, "L")])
    assert out["overall"] == {"pearson_r": None, "n": 0}
    assert out["by_discipline"] == {}
    assert out["by_travel_direction"] == {}


def test_overall_pearson_matches_manual() -> None:
    client = FakeGraphClient(read_results={REST_QUERY: _REST_ROWS})
    items = [_Item(1, 1, 1.0, "L"), _Item(2, 1, 1.0, "L")]
    # Two points, identical y → zero variance → pearson undefined (None).
    out = _report(client, items)
    assert out["overall"] == {"pearson_r": None, "n": 2}


def test_perfect_positive_correlation() -> None:
    client = FakeGraphClient(read_results={REST_QUERY: _REST_ROWS})
    # value increases with rested_index (0.9, 0.7, 0.3, 0.1) → positive r.
    items = [
        _Item(1, 1, 9.0, "L"),
        _Item(2, 1, 7.0, "L"),
        _Item(3, 1, 3.0, "B"),
        _Item(4, 1, 1.0, "B"),
    ]
    out = _report(client, items)
    assert out["overall"]["pearson_r"] == pytest.approx(1.0)
