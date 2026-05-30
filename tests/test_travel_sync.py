"""Tests for the P3c L3 travel/circadian build (sync.travel).

These tests use the shared ``FakeGraphClient`` (from ``tests/conftest.py``)
seeded with canned ``run_read`` results for the three read queries
(:data:`ATHLETE_EVENT_QUERY`, :data:`ATHLETE_HOME_QUERY`,
:data:`COUNTRY_GEO_QUERY`), so NO live Neo4j connection is ever made and no
GeoNames file is needed — venue coordinates / timezones are supplied directly.

They assert TravelLeg + RestednessState node/edge creation with the correct
namespaced ids from ``vocab`` builders, the swing-vs-home-base origin selection
per the gap/timezone rule (PRD §9), eastward-vs-westward recovery reflected in
``rested_index``, home-base resolution from the graph (centroid + most-common
timezone), the unresolved-home-base fallback (low confidence, tz term dropped),
and idempotency (run twice → identical logical MERGE sets).
"""

from __future__ import annotations

import pytest

from climber_network import vocab
from climber_network.config import TRAVEL_PARAMS
from sync.travel import (
    ATHLETE_EVENT_QUERY,
    ATHLETE_HOME_QUERY,
    CONFIDENCE_HOME_BASE,
    CONFIDENCE_HOME_BASE_TZ_ONLY,
    CONFIDENCE_SWING,
    CONFIDENCE_UNRESOLVED,
    COUNTRY_GEO_QUERY,
    Place,
    build_travel,
    resolve_home_bases,
)
from tests.conftest import FakeGraphClient

# ---------------------------------------------------------------------------
# Canned graph reads.
#
# Athlete 1 (based in USA) competes in four events:
#   evt 1  Innsbruck (Europe/Vienna)  2024-06-01  — first → home_base (USA→AUT, east)
#   evt 2  Bern      (Europe/Zurich)  2024-06-05  — 4-day gap, DIFFERENT tz → swing
#   evt 3  Tokyo     (Asia/Tokyo)     2024-09-01  — 88-day gap → home_base (USA→JPN, east)
#   evt 4  Seattle   (America/LA)     2024-09-20  — 19-day gap (>swing) → home_base
#                                                    (USA→USA, ~same tz → low burden)
# Athlete 2 (based in JPN) competes in two events with a tight same-tz swing:
#   evt 5  Tokyo     (Asia/Tokyo)     2024-05-01  — first → home_base (JPN→JPN)
#   evt 6  Osaka     (Asia/Tokyo)     2024-05-04  — 3-day gap but SAME tz → home_base
# ---------------------------------------------------------------------------

# Country geography: USA venue (Seattle) gives the USA home base its coords + tz;
# AUT (Innsbruck), CHE (Bern), JPN (Tokyo) likewise. These are the rows the real
# COUNTRY_GEO_QUERY would return (one per country-venue-timezone).
_COUNTRY_ROWS: list[dict[str, object]] = [
    {"iso3": "USA", "lon": -122.33, "lat": 47.61, "iana": "America/Los_Angeles"},
    {"iso3": "AUT", "lon": 11.39, "lat": 47.26, "iana": "Europe/Vienna"},
    {"iso3": "CHE", "lon": 7.45, "lat": 46.95, "iana": "Europe/Zurich"},
    {"iso3": "JPN", "lon": 139.69, "lat": 35.69, "iana": "Asia/Tokyo"},
]

_HOME_ROWS: list[dict[str, object]] = [
    {"athlete_id": 1, "iso3": "USA"},
    {"athlete_id": 2, "iso3": "JPN"},
]

_EVENT_ROWS: list[dict[str, object]] = [
    {
        "athlete_id": 1,
        "event_id": 1,
        "start_date": "2024-06-01",
        "venue_lon": 11.39,
        "venue_lat": 47.26,
        "venue_tz": "Europe/Vienna",
    },
    {
        "athlete_id": 1,
        "event_id": 2,
        "start_date": "2024-06-05",
        "venue_lon": 7.45,
        "venue_lat": 46.95,
        "venue_tz": "Europe/Zurich",
    },
    {
        "athlete_id": 1,
        "event_id": 3,
        "start_date": "2024-09-01",
        "venue_lon": 139.69,
        "venue_lat": 35.69,
        "venue_tz": "Asia/Tokyo",
    },
    {
        "athlete_id": 1,
        "event_id": 4,
        "start_date": "2024-09-20",
        "venue_lon": -122.33,
        "venue_lat": 47.61,
        "venue_tz": "America/Los_Angeles",
    },
    {
        "athlete_id": 2,
        "event_id": 5,
        "start_date": "2024-05-01",
        "venue_lon": 139.69,
        "venue_lat": 35.69,
        "venue_tz": "Asia/Tokyo",
    },
    {
        "athlete_id": 2,
        "event_id": 6,
        "start_date": "2024-05-04",
        "venue_lon": 135.50,
        "venue_lat": 34.69,
        "venue_tz": "Asia/Tokyo",
    },
]


def _client(**overrides: object) -> FakeGraphClient:
    reads: dict[str, list[dict[str, object]]] = {
        ATHLETE_EVENT_QUERY: _EVENT_ROWS,
        ATHLETE_HOME_QUERY: _HOME_ROWS,
        COUNTRY_GEO_QUERY: _COUNTRY_ROWS,
    }
    reads.update(overrides)  # type: ignore[arg-type]
    return FakeGraphClient(read_results=reads)


# ---------------------------------------------------------------------------
# Node / edge / id creation.
# ---------------------------------------------------------------------------


def test_legs_and_states_created_with_correct_ids() -> None:
    client = _client()
    report = build_travel(client)

    ath1 = vocab.ath(1)
    # One TravelLeg + one RestednessState per placed athlete-event (6 total).
    assert report.legs == 6
    assert report.states == 6

    for evt_id in (1, 2, 3, 4):
        leg_id = vocab.leg(ath1, vocab.evt(evt_id))
        rest_id = vocab.rest(ath1, vocab.evt(evt_id))
        assert client.node_labels[leg_id] == "TravelLeg"
        assert client.node_labels[rest_id] == "RestednessState"

    # All four edge types, for event 1.
    leg1 = vocab.leg(ath1, vocab.evt(1))
    rest1 = vocab.rest(ath1, vocab.evt(1))
    assert (ath1, "TRAVELED", leg1) in client.rels
    assert (leg1, "TO_EVENT", vocab.evt(1)) in client.rels
    assert (ath1, "HAD_STATE", rest1) in client.rels
    assert (rest1, "AT_EVENT", vocab.evt(1)) in client.rels


def test_restedness_props_present_and_clamped() -> None:
    client = _client()
    build_travel(client)

    rest1 = client.nodes[vocab.rest(vocab.ath(1), vocab.evt(1))]
    assert rest1["days_since_arrival"] == float(TRAVEL_PARAMS.arrive_days_before)
    assert rest1["model_version"] == TRAVEL_PARAMS.model_version
    assert 0.0 <= rest1["rested_index"] <= 1.0
    assert 0.0 <= rest1["jetlag_residual"] <= 1.0
    assert 0.0 <= rest1["travel_fatigue"] <= 1.0


def test_restedness_carries_validate_elo_join_keys() -> None:
    """Cross-module contract: RestednessState must carry the scalar props that
    ``sync.validate_elo``'s correlation query (REST_QUERY) reads — keyed on the
    RAW climbing-elo ids — or the rested_index↔elo_residual correlation
    silently degrades to n=0 against a live graph (PRD §9 validation hook)."""
    client = _client()
    build_travel(client)

    rest1 = client.nodes[vocab.rest(vocab.ath(1), vocab.evt(1))]
    for key in ("athlete_id", "event_id", "rested_index", "discipline", "travel_direction"):
        assert key in rest1, f"RestednessState missing {key!r} (validate_elo join key)"
    # Join keys are the raw climbing-elo ids, not the namespaced ath:/evt: ids.
    assert rest1["athlete_id"] == 1
    assert rest1["event_id"] == 1
    assert rest1["travel_direction"] in {"E", "W", "none"}


# ---------------------------------------------------------------------------
# Origin selection — swing vs home_base.
# ---------------------------------------------------------------------------


def test_swing_leg_when_tight_gap_and_different_tz() -> None:
    client = _client()
    build_travel(client)

    # evt 2 (Bern) is 4 days after evt 1 (Innsbruck) — within swing_gap_days — and
    # in a DIFFERENT timezone (Europe/Zurich vs Europe/Vienna) → swing origin.
    leg2 = client.nodes[vocab.leg(vocab.ath(1), vocab.evt(2))]
    assert leg2["origin"] == "prev_event"
    assert leg2["confidence"] == CONFIDENCE_SWING
    assert leg2["origin_tz"] == "Europe/Vienna"


def test_home_base_leg_for_first_event() -> None:
    client = _client()
    build_travel(client)

    # evt 1 is the athlete's first event → home base (USA), not a swing.
    leg1 = client.nodes[vocab.leg(vocab.ath(1), vocab.evt(1))]
    assert leg1["origin"] == "home_base"
    assert leg1["confidence"] == CONFIDENCE_HOME_BASE
    # USA home base resolved from the graph (its single venue's tz).
    assert leg1["origin_tz"] == "America/Los_Angeles"


def test_home_base_leg_when_gap_exceeds_swing_window() -> None:
    client = _client()
    build_travel(client)

    # evt 3 (Tokyo) is 88 days after evt 2 — well beyond swing_gap_days → home base.
    leg3 = client.nodes[vocab.leg(vocab.ath(1), vocab.evt(3))]
    assert leg3["origin"] == "home_base"
    assert leg3["origin_tz"] == "America/Los_Angeles"


def test_home_base_when_tight_gap_but_same_tz() -> None:
    client = _client()
    build_travel(client)

    # Athlete 2: evt 6 (Osaka) is 3 days after evt 5 (Tokyo) — tight gap — but the
    # SAME timezone (Asia/Tokyo) → NOT a swing; falls back to home base.
    leg6 = client.nodes[vocab.leg(vocab.ath(2), vocab.evt(6))]
    assert leg6["origin"] == "home_base"


def test_report_origin_tallies() -> None:
    client = _client()
    report = build_travel(client)

    # Athlete 1: home(1) + swing(2) + home(3) + home(4); Athlete 2: home(5)+home(6).
    assert report.origin_prev_event == 1
    assert report.origin_home_base == 5


# ---------------------------------------------------------------------------
# Eastward vs westward recovery reflected in rested_index.
# ---------------------------------------------------------------------------


def test_eastward_harder_than_westward() -> None:
    """Eastward (phase-advance) travel costs more recovery → lower rested_index.

    Both athletes leave the same fixed UTC+3 home base (Asia/Qatar, no DST): one
    flies +3 tz east (recovery 1.0*3 = 3 days → residual 1-2/3 ≈ 0.333) and one
    flies -5 tz west (recovery 0.5*5 = 2.5 days → residual 1-2/2.5 = 0.2). Both
    stay under ``recovery_cap_days`` so the asymmetric recovery rate (1.0 day/tz
    east vs 0.5 day/tz west) leaves the eastbound athlete measurably less rested.
    """
    event_rows = [
        {
            "athlete_id": 10,
            "event_id": 100,
            "start_date": "2024-06-01",
            "venue_lon": 90.41,
            "venue_lat": 23.81,
            "venue_tz": "Asia/Dhaka",  # UTC+6 (no DST) → +3 east of UTC+3 home
        },
        {
            "athlete_id": 20,
            "event_id": 200,
            "start_date": "2024-06-01",
            "venue_lon": -36.0,
            "venue_lat": -54.0,
            "venue_tz": "Atlantic/South_Georgia",  # UTC-2 (no DST) → 3 west
        },
    ]
    home_rows = [
        {"athlete_id": 10, "iso3": "QAT"},
        {"athlete_id": 20, "iso3": "QAT"},
    ]
    # Qatar (Asia/Qatar) is a fixed UTC+3 (no DST): Dhaka is +3 east, South
    # Georgia is -5... so pick offsets that bracket evenly below the cap.
    country_rows = [{"iso3": "QAT", "lon": 51.53, "lat": 25.29, "iana": "Asia/Qatar"}]

    client = FakeGraphClient(
        read_results={
            ATHLETE_EVENT_QUERY: event_rows,
            ATHLETE_HOME_QUERY: home_rows,
            COUNTRY_GEO_QUERY: country_rows,
        }
    )
    build_travel(client)

    east = client.nodes[vocab.leg(vocab.ath(10), vocab.evt(100))]
    west = client.nodes[vocab.leg(vocab.ath(20), vocab.evt(200))]
    east_state = client.nodes[vocab.rest(vocab.ath(10), vocab.evt(100))]
    west_state = client.nodes[vocab.rest(vocab.ath(20), vocab.evt(200))]

    assert east["direction"] == "E"
    assert west["direction"] == "W"
    assert east["tz_delta_h"] > 0
    assert west["tz_delta_h"] < 0
    # Eastward recovery (1 day/tz) > westward (0.5 day/tz) → higher jetlag
    # residual → lower restedness for the eastbound athlete.
    assert east_state["jetlag_residual"] > west_state["jetlag_residual"]
    assert east_state["rested_index"] < west_state["rested_index"]


# ---------------------------------------------------------------------------
# Home-base resolution from the graph.
# ---------------------------------------------------------------------------


def test_resolve_home_bases_centroid_and_modal_tz() -> None:
    rows = [
        {"iso3": "USA", "lon": -122.0, "lat": 47.0, "iana": "America/Los_Angeles"},
        {"iso3": "USA", "lon": -74.0, "lat": 41.0, "iana": "America/New_York"},
        {"iso3": "USA", "lon": -118.0, "lat": 34.0, "iana": "America/Los_Angeles"},
    ]
    homes = resolve_home_bases(rows)
    usa = homes["USA"]
    # Centroid = mean of the three venue points.
    assert usa.lon == pytest.approx((-122.0 - 74.0 - 118.0) / 3)
    assert usa.lat == pytest.approx((47.0 + 41.0 + 34.0) / 3)
    # Most common timezone wins (LA appears twice).
    assert usa.tz == "America/Los_Angeles"


def test_resolve_home_bases_tz_only_country() -> None:
    # A country with a timezone but no coordinates still resolves a tz-only Place.
    rows = [{"iso3": "FRA", "lon": None, "lat": None, "iana": "Europe/Paris"}]
    homes = resolve_home_bases(rows)
    assert homes["FRA"] == Place(lon=None, lat=None, tz="Europe/Paris")


def test_unresolved_home_base_low_confidence_and_zero_tz() -> None:
    # Athlete 3 is based in a country with NO geography rows AND no capital-tz
    # entry ("ZZZ" is not a real country) → genuinely unresolved.
    event_rows = [
        {
            "athlete_id": 3,
            "event_id": 7,
            "start_date": "2024-06-01",
            "venue_lon": 11.39,
            "venue_lat": 47.26,
            "venue_tz": "Europe/Vienna",
        }
    ]
    home_rows = [{"athlete_id": 3, "iso3": "ZZZ"}]
    client = FakeGraphClient(
        read_results={
            ATHLETE_EVENT_QUERY: event_rows,
            ATHLETE_HOME_QUERY: home_rows,
            COUNTRY_GEO_QUERY: _COUNTRY_ROWS,  # has no ZZZ row.
        }
    )
    report = build_travel(client)

    leg = client.nodes[vocab.leg(vocab.ath(3), vocab.evt(7))]
    assert leg["origin"] == "home_base"
    assert leg["confidence"] == CONFIDENCE_UNRESOLVED
    assert leg["origin_tz"] is None
    # tz term dropped → no timezone burden, distance term also zero (no coords).
    assert leg["tz_delta_h"] == 0.0
    assert leg["direction"] == "none"
    assert leg["distance_km"] == 0.0
    assert report.unresolved_origin == 1


def test_home_base_capital_tz_fallback_resolves_direction() -> None:
    """Issue #42: a home country that hosts no resolved venue still gets a
    timezone from the capital-city fallback, so the leg crosses a timezone
    (direction E/W) instead of collapsing to ``none``/unresolved.

    Athlete 30 is based in IRN (Asia/Tehran, UTC+3:30) — absent from the country
    geography rows — and competes in Vienna (UTC+2 in summer). The westward shift
    must be modelled even though IRN has no venue coordinate in the graph.
    """
    event_rows = [
        {
            "athlete_id": 30,
            "event_id": 300,
            "start_date": "2024-06-01",
            "venue_lon": 16.37,
            "venue_lat": 48.21,
            "venue_tz": "Europe/Vienna",
        }
    ]
    home_rows = [{"athlete_id": 30, "iso3": "IRN"}]
    client = FakeGraphClient(
        read_results={
            ATHLETE_EVENT_QUERY: event_rows,
            ATHLETE_HOME_QUERY: home_rows,
            COUNTRY_GEO_QUERY: _COUNTRY_ROWS,  # has no IRN row.
        }
    )
    report = build_travel(client)

    leg = client.nodes[vocab.leg(vocab.ath(30), vocab.evt(300))]
    assert leg["origin"] == "home_base"
    # tz known but no coordinate → the dedicated tz-only confidence tier.
    assert leg["confidence"] == CONFIDENCE_HOME_BASE_TZ_ONLY
    assert leg["origin_tz"] == "Asia/Tehran"
    # Tehran (UTC+3:30) → Vienna (UTC+2) is a westward shift, now modelled.
    assert leg["direction"] == "W"
    assert leg["tz_delta_h"] < 0
    # No origin coordinate → distance/flight term stays zero (honest).
    assert leg["distance_km"] == 0.0
    # The leg is no longer counted as an unresolved origin.
    assert report.unresolved_origin == 0


def test_direction_tallies_and_tz_crossing_share() -> None:
    client = _client()
    report = build_travel(client)

    # Every emitted leg is classified into exactly one direction bucket.
    assert report.direction_e + report.direction_w + report.direction_none == report.legs
    # The fixture has athletes crossing timezones (USA→Europe/Asia) and a
    # same-tz Japan swing, so both crossing and none buckets are populated.
    assert report.direction_e + report.direction_w >= 1
    assert report.direction_none >= 1
    assert 0.0 < report._tz_crossing_share() < 1.0


# ---------------------------------------------------------------------------
# Skips for undated / unplaced events.
# ---------------------------------------------------------------------------


def test_event_without_venue_coords_is_skipped() -> None:
    event_rows = [
        {
            "athlete_id": 5,
            "event_id": 9,
            "start_date": "2024-06-01",
            "venue_lon": None,
            "venue_lat": None,
            "venue_tz": None,
        }
    ]
    client = FakeGraphClient(
        read_results={
            ATHLETE_EVENT_QUERY: event_rows,
            ATHLETE_HOME_QUERY: [{"athlete_id": 5, "iso3": "USA"}],
            COUNTRY_GEO_QUERY: _COUNTRY_ROWS,
        }
    )
    report = build_travel(client)
    assert report.skipped_no_venue == 1
    assert report.legs == 0
    assert vocab.leg(vocab.ath(5), vocab.evt(9)) not in client.nodes


def test_event_without_date_is_skipped() -> None:
    event_rows = [
        {
            "athlete_id": 6,
            "event_id": 11,
            "start_date": None,
            "venue_lon": 11.39,
            "venue_lat": 47.26,
            "venue_tz": "Europe/Vienna",
        }
    ]
    client = FakeGraphClient(
        read_results={
            ATHLETE_EVENT_QUERY: event_rows,
            ATHLETE_HOME_QUERY: [{"athlete_id": 6, "iso3": "USA"}],
            COUNTRY_GEO_QUERY: _COUNTRY_ROWS,
        }
    )
    report = build_travel(client)
    assert report.skipped_no_date == 1
    assert report.legs == 0


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------


def test_idempotent_rerun() -> None:
    first = _client()
    build_travel(first)
    second = _client()
    build_travel(second)

    # Same logical node/edge sets and same number of MERGE calls on a re-run:
    # MERGE is keyed, so the graph state is identical (0 net changes).
    assert first.nodes.keys() == second.nodes.keys()
    assert first.rels.keys() == second.rels.keys()
    assert len(first.node_calls) == len(second.node_calls)
    assert len(first.rel_calls) == len(second.rel_calls)

    # No node id is MERGEd more than once within a single run (dedup-aware).
    node_ids = [nid for (_label, nid) in first.node_calls]
    assert len(node_ids) == len(set(node_ids))


# ---------------------------------------------------------------------------
# Out-of-vocab guard (closed-vocab safety, via the fake client's gating).
# ---------------------------------------------------------------------------


def test_only_valid_labels_and_rels_used() -> None:
    client = _client()
    build_travel(client)
    # Every label / rel recorded was accepted by assert_label / assert_rel, so a
    # build that completes without raising proves the closed-vocab guard held.
    assert set(client.node_labels.values()) == {"TravelLeg", "RestednessState"}
    used_rels = {rel for (_s, rel, _t) in client.rels}
    assert used_rels == {"TRAVELED", "TO_EVENT", "HAD_STATE", "AT_EVENT"}
