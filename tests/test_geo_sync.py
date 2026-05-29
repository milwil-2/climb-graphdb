"""Tests for the P2b L2 geography build (sync.geo).

These tests use:
* the shared ``FakeGraphClient`` (from ``tests/conftest.py``) seeded with canned
  ``run_read`` results for the Event and Athlete read queries, so NO live Neo4j
  connection is ever made;
* an in-memory ``GeoNamesIndex.from_records(...)`` (no real GeoNames file).

They assert Venue/City/Country/TimeZone node + edge creation with the correct
namespaced ids from ``vocab`` builders, the country-centroid fallback for an
unresolvable event (low geocode_confidence), REPRESENTS / BASED_IN edges from
athlete nationality, and idempotency (run twice → identical logical MERGE sets).
"""

from __future__ import annotations

from pathlib import Path

from climber_network import vocab
from climber_network.geo.geocode import GeoNamesIndex
from sync.geo import (
    ATHLETE_QUERY,
    CONFIDENCE_CITY,
    CONFIDENCE_COUNTRY,
    EVENT_QUERY,
    NATIONALITY_PROXY,
    ResolutionCache,
    build_geo,
)
from tests.conftest import FakeGraphClient

# ---------------------------------------------------------------------------
# Fixtures — in-memory GeoNames + canned graph reads.
# ---------------------------------------------------------------------------

# Country codes here are ISO alpha-2 — the real GeoNames cities1000 form, and
# the form the resolver constrains by (IOC parens code → alpha-2).
_GEO_RECORDS: list[dict[str, object]] = [
    {
        "geonameid": 2775220,
        "name": "Innsbruck",
        "lat": 47.26266,
        "lon": 11.39454,
        "country": "AT",
        "timezone": "Europe/Vienna",
    },
    {
        "geonameid": 2661552,
        "name": "Bern",
        "lat": 46.94809,
        "lon": 7.44744,
        "country": "CH",
        "timezone": "Europe/Zurich",
    },
]

# The upstream events table has NO country (all NULL); the host country is
# derived from the event name's parenthesised IOC code (or backfilled from a
# same-named coded event). The ``country`` field below is therefore irrelevant
# to the build and kept only to mirror the real (NULL) schema.
#
# Event 1 → Innsbruck (AUT) → resolvable, country AUT.
# Event 2 → Bern (SUI) → resolvable, IOC SUI → ISO CHE.
# Event 3 → "Atlantis" (FRA): city not in index but the parens code gives a
#           country → country-centroid fallback (ISO FRA).
# Event 4 → no city and no country code → skipped entirely.
# NOTE: EVENT_QUERY / ATHLETE_QUERY read the existing L1 nodes, so run_read
# returns each node's ``id`` property — which is ALREADY the full vocab id
# ("evt:1", "ath:1"), NOT a raw integer. Fixtures mirror that exactly so the
# sync's id handling is exercised the way it runs in production (a regression
# guard against re-wrapping the id and double-prefixing it).
_EVENT_ROWS: list[dict[str, object]] = [
    {
        "id": vocab.evt(1),
        "name": "IFSC - Climbing World Cup (B,L) - Innsbruck (AUT) 2024",
        "country": None,
    },
    {
        "id": vocab.evt(2),
        "name": "IFSC Climbing World Championships - Bern (SUI) 2023",
        "country": None,
    },
    {"id": vocab.evt(3), "name": "IFSC World Cup (L) - Atlantis (FRA) 2025", "country": None},
    {"id": vocab.evt(4), "name": "IFSC World Cup", "country": None},
]

_ATHLETE_ROWS: list[dict[str, object]] = [
    {"id": vocab.ath(1), "nationality": "USA"},
    {"id": vocab.ath(2), "nationality": "AUT"},
    {"id": vocab.ath(3), "nationality": "AUT"},  # shares nationality with athlete 2.
    {"id": vocab.ath(4), "nationality": None},  # no nationality → no edges.
]


def _index() -> GeoNamesIndex:
    return GeoNamesIndex.from_records(_GEO_RECORDS)


def _client() -> FakeGraphClient:
    return FakeGraphClient(read_results={EVENT_QUERY: _EVENT_ROWS, ATHLETE_QUERY: _ATHLETE_ROWS})


def _build(client: FakeGraphClient) -> None:
    build_geo(client, _index())


# ---------------------------------------------------------------------------
# Resolved Venue / City / Country / TimeZone nodes + ids.
# ---------------------------------------------------------------------------


def test_resolved_venue_city_country_timezone_nodes() -> None:
    client = _client()
    _build(client)

    # Venue, keyed by city slug, carries a point + high confidence.
    ven_id = vocab.ven(vocab.slug("Innsbruck"))
    assert client.node_labels[ven_id] == "Venue"
    venue = client.nodes[ven_id]
    assert venue["geocode_confidence"] == CONFIDENCE_CITY
    assert venue["location"].longitude == 11.39454
    assert venue["location"].latitude == 47.26266

    # City keyed by geonameid.
    city_id = vocab.city(2775220)
    assert client.node_labels[city_id] == "City"
    assert client.nodes[city_id]["geonameid"] == 2775220

    # Country keyed by ISO3.
    ctry_id = vocab.ctry("AUT")
    assert client.node_labels[ctry_id] == "Country"
    assert client.nodes[ctry_id]["iso3"] == "AUT"

    # TimeZone keyed by IANA id (with '/' → '_').
    tz_id = vocab.tz("Europe/Vienna")
    assert tz_id == "tz:Europe_Vienna"
    assert client.node_labels[tz_id] == "TimeZone"
    assert client.nodes[tz_id]["iana"] == "Europe/Vienna"


def test_resolved_edges() -> None:
    client = _client()
    _build(client)

    ven_id = vocab.ven(vocab.slug("Innsbruck"))
    city_id = vocab.city(2775220)
    ctry_id = vocab.ctry("AUT")
    tz_id = vocab.tz("Europe/Vienna")

    assert (vocab.evt(1), "HELD_AT", ven_id) in client.rels
    assert (ven_id, "IN_CITY", city_id) in client.rels
    assert (city_id, "IN_COUNTRY", ctry_id) in client.rels
    assert (ven_id, "IN_TIMEZONE", tz_id) in client.rels


# ---------------------------------------------------------------------------
# Country-centroid fallback for an unresolvable event.
# ---------------------------------------------------------------------------


def test_country_centroid_fallback() -> None:
    client = _client()
    _build(client)

    # Event 3's city ("Atlantis") is not in the index, but its parens code
    # (FRA) gives a country → country-level Venue keyed at ISO3 ``fra``.
    ven_id = vocab.ven("country-fra")
    assert client.node_labels[ven_id] == "Venue"
    venue = client.nodes[ven_id]
    assert venue["geocode_confidence"] == CONFIDENCE_COUNTRY
    # No centroid map supplied → no point on the placeholder Venue.
    assert "location" not in venue

    assert (vocab.evt(3), "HELD_AT", ven_id) in client.rels
    assert (ven_id, "IN_COUNTRY", vocab.ctry("FRA")) in client.rels


def test_country_centroid_fallback_with_point() -> None:
    client = _client()
    build_geo(
        client,
        _index(),
        centroids={"FRA": (12.5, -7.25)},
    )

    venue = client.nodes[vocab.ven("country-fra")]
    assert venue["location"].longitude == 12.5
    assert venue["location"].latitude == -7.25


def test_event_without_city_or_country_is_skipped() -> None:
    client = _client()
    report = build_geo(client, _index())

    # Event 4 has no country and no extractable city → no Venue at all.
    assert report.skipped_events == 1
    # No HELD_AT edge originates from event 4.
    assert all(src != vocab.evt(4) for (src, _rel, _tgt) in client.rels)


# ---------------------------------------------------------------------------
# REPRESENTS / BASED_IN from athlete nationality.
# ---------------------------------------------------------------------------


def test_represents_and_based_in_from_nationality() -> None:
    client = _client()
    _build(client)

    usa = vocab.ctry("USA")
    assert (vocab.ath(1), "REPRESENTS", usa) in client.rels
    based = client.rels[(vocab.ath(1), "BASED_IN", usa)]
    assert based is not None
    assert based["source"] == NATIONALITY_PROXY

    # Athlete 4 (no nationality) produces no edges.
    assert all(src != vocab.ath(4) for (src, _rel, _tgt) in client.rels)


def test_country_node_shared_across_sources() -> None:
    client = _client()
    report = build_geo(client, _index())

    # AUT appears as both an event country (Innsbruck) and an athlete
    # nationality (athletes 2 & 3) but is MERGEd as ONE logical Country node.
    aut_calls = [nid for (label, nid) in client.node_calls if nid == vocab.ctry("AUT")]
    assert len(aut_calls) == 1
    # Event countries (AUT, CHE, FRA) + athlete nationality USA.
    assert report.node_countries == len({"AUT", "CHE", "FRA", "USA"})


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------


def test_idempotent_rerun() -> None:
    first = _client()
    _build(first)
    second = _client()
    _build(second)

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
# Resolution cache.
# ---------------------------------------------------------------------------


def test_cache_round_trips_via_disk(tmp_path: Path) -> None:
    cache_path = tmp_path / "geocode_cache.json"

    # First build populates and flushes the cache.
    first = _client()
    cache1 = ResolutionCache(cache_path)
    build_geo(first, _index(), cache=cache1)
    assert cache_path.exists()

    # Second build reuses the cache: every extracted city is a cache hit, and the
    # resolved graph is identical to a fresh (cache-less) run.
    second = _client()
    cache2 = ResolutionCache(cache_path)
    report2 = build_geo(second, _index(), cache=cache2)
    assert report2.cache_hits >= 1

    no_cache = _client()
    _build(no_cache)
    assert second.nodes.keys() == no_cache.nodes.keys()
    assert second.rels.keys() == no_cache.rels.keys()
