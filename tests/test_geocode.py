"""Unit tests for climber_network.geo.geocode.

All tests are pure-Python — no database and no real GeoNames file. The
GeoNamesIndex is built from a small in-memory fixture via ``from_records``.
"""

from __future__ import annotations

from datetime import date

import pytest

from climber_network.geo.geocode import (
    GeoNamesIndex,
    GeoPoint,
    extract_city,
    tz_for,
    utc_offset_hours,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_RECORDS: list[dict[str, object]] = [
    {
        "geonameid": 2775220,
        "name": "Innsbruck",
        "lat": 47.26266,
        "lon": 11.39454,
        "country": "AUT",
        "timezone": "Europe/Vienna",
    },
    {
        "geonameid": 2661552,
        "name": "Bern",
        "lat": 46.94809,
        "lon": 7.44744,
        "country": "CHE",
        "timezone": "Europe/Zurich",
    },
    {
        "geonameid": 5780993,
        "name": "Salt Lake City",
        "lat": 40.76078,
        "lon": -111.89105,
        "country": "USA",
        "timezone": "America/Denver",
    },
    {
        "geonameid": 2657896,
        "name": "Zürich",
        "lat": 47.36667,
        "lon": 8.55,
        "country": "CHE",
        "timezone": "Europe/Zurich",
    },
]


@pytest.fixture
def index() -> GeoNamesIndex:
    return GeoNamesIndex.from_records(_RECORDS)


# ---------------------------------------------------------------------------
# extract_city
# ---------------------------------------------------------------------------


class TestExtractCity:
    def test_basic_world_cup(self) -> None:
        assert extract_city("IFSC World Cup Innsbruck 2023", "AUT") == "Innsbruck"

    def test_world_championships(self) -> None:
        assert extract_city("IFSC Climbing World Championships Bern 2023", "CHE") == "Bern"

    def test_discipline_words_stripped(self) -> None:
        assert extract_city("IFSC Boulder World Cup Innsbruck 2024", "AUT") == "Innsbruck"

    def test_multi_word_city(self) -> None:
        result = extract_city("IFSC Climbing World Cup Salt Lake City 2024", "USA")
        assert result == "Salt Lake City"

    def test_parenthetical_country_dropped(self) -> None:
        result = extract_city("IFSC - Climbing World Cup (B) - Salt Lake City (USA) 2024", "USA")
        assert result == "Salt Lake City"

    def test_combined_and_speed_stripped(self) -> None:
        assert extract_city("IFSC Speed World Cup Seoul 2023", "KOR") == "Seoul"

    def test_year_removed(self) -> None:
        assert "2023" not in (extract_city("World Cup Chamonix 2023", "FRA") or "")

    def test_empty_returns_none(self) -> None:
        assert extract_city("", "AUT") is None

    def test_all_noise_returns_none(self) -> None:
        assert extract_city("IFSC World Cup 2023", None) is None

    def test_country_arg_optional(self) -> None:
        assert extract_city("IFSC World Cup Villars 2023", None) == "Villars"


# ---------------------------------------------------------------------------
# GeoNamesIndex
# ---------------------------------------------------------------------------


class TestGeoNamesIndex:
    def test_lookup_hit(self, index: GeoNamesIndex) -> None:
        point = index.lookup("Innsbruck", "AUT")
        assert point is not None
        assert isinstance(point, GeoPoint)
        assert point.geonameid == 2775220
        assert point.timezone == "Europe/Vienna"
        assert point.lat == pytest.approx(47.26266)
        assert point.lon == pytest.approx(11.39454)

    def test_lookup_case_insensitive(self, index: GeoNamesIndex) -> None:
        assert index.lookup("innsbruck", "AUT") is not None
        assert index.lookup("INNSBRUCK", "AUT") is not None

    def test_lookup_accent_insensitive(self, index: GeoNamesIndex) -> None:
        # "Zurich" (no umlaut) must match the indexed "Zürich".
        point = index.lookup("Zurich", "CHE")
        assert point is not None
        assert point.geonameid == 2657896

    def test_lookup_country_miss_falls_back_to_city(self, index: GeoNamesIndex) -> None:
        # Wrong country still resolves via the city-only fallback.
        point = index.lookup("Bern", "DEU")
        assert point is not None
        assert point.geonameid == 2661552

    def test_lookup_without_country(self, index: GeoNamesIndex) -> None:
        point = index.lookup("Salt Lake City", None)
        assert point is not None
        assert point.geonameid == 5780993

    def test_lookup_miss(self, index: GeoNamesIndex) -> None:
        assert index.lookup("Atlantis", "XXX") is None

    def test_lookup_empty_city(self, index: GeoNamesIndex) -> None:
        assert index.lookup("", "AUT") is None

    def test_from_records_roundtrip(self) -> None:
        idx = GeoNamesIndex.from_records(
            [
                {
                    "geonameid": 1,
                    "name": "Testville",
                    "lat": 1.5,
                    "lon": -2.5,
                    "country": "FRA",
                    "timezone": "Europe/Paris",
                }
            ]
        )
        point = idx.lookup("Testville", "FRA")
        assert point == GeoPoint(
            lat=1.5,
            lon=-2.5,
            geonameid=1,
            name="Testville",
            timezone="Europe/Paris",
        )


# ---------------------------------------------------------------------------
# tz_for
# ---------------------------------------------------------------------------


class TestTzFor:
    def test_innsbruck(self) -> None:
        assert tz_for(47.26266, 11.39454) == "Europe/Vienna"

    def test_salt_lake_city(self) -> None:
        assert tz_for(40.76078, -111.89105) == "America/Denver"

    def test_open_ocean_returns_none_or_etc(self) -> None:
        # Middle of the South Atlantic: either None or an Etc/GMT zone.
        result = tz_for(-40.0, -20.0)
        assert result is None or result.startswith("Etc/")


# ---------------------------------------------------------------------------
# utc_offset_hours (DST awareness)
# ---------------------------------------------------------------------------


class TestUtcOffsetHours:
    def test_vienna_summer(self) -> None:
        # CEST = UTC+2
        assert utc_offset_hours("Europe/Vienna", date(2023, 7, 1)) == 2.0

    def test_vienna_winter(self) -> None:
        # CET = UTC+1
        assert utc_offset_hours("Europe/Vienna", date(2023, 1, 1)) == 1.0

    def test_dst_difference(self) -> None:
        summer = utc_offset_hours("Europe/Vienna", date(2023, 7, 1))
        winter = utc_offset_hours("Europe/Vienna", date(2023, 1, 1))
        assert summer - winter == 1.0

    def test_utc_zero(self) -> None:
        assert utc_offset_hours("UTC", date(2023, 6, 15)) == 0.0

    def test_half_hour_offset(self) -> None:
        # India is UTC+5:30 year-round (no DST).
        assert utc_offset_hours("Asia/Kolkata", date(2023, 1, 1)) == 5.5
