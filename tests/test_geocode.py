"""Unit tests for climber_network.geo.geocode.

All tests are pure-Python — no database and no real GeoNames file. The
GeoNamesIndex is built from a small in-memory fixture via ``from_records``.
"""

from __future__ import annotations

from datetime import date
from zoneinfo import available_timezones

import pytest

from climber_network.geo.geocode import (
    _CITY_OVERRIDES,
    COUNTRY_CAPITAL_TZ,
    GeoNamesIndex,
    GeoPoint,
    _Override,
    _validate_overrides,
    alpha2_to_alpha3,
    country_capital_tz,
    extract_city,
    override_alpha2,
    parse_ioc_alpha2,
    resolve_event,
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

    def test_worldcup_one_word_stripped(self) -> None:
        # Older events spell "Worldcup" as a single token.
        assert extract_city("IFSC Climbing Worldcup (L) - Kranj (SLO) 2017", None) == "Kranj"

    def test_world_climbing_series_stripped(self) -> None:
        assert extract_city("World Climbing Series Innsbruck 2026", None) == "Innsbruck"

    def test_group_qualifier_stripped(self) -> None:
        # "Lead Group A Paris" → "Paris" (group / A / lead are all noise).
        assert extract_city("IFSC World Championship Lead Group A - Paris 2012", None) == "Paris"

    def test_continental_region_stripped(self) -> None:
        result = extract_city(
            "IFSC Europe - Continental Championships (B,S,L,C) - Moscow (RUS) 2020", None
        )
        assert result == "Moscow"


# ---------------------------------------------------------------------------
# IOC parens-code parsing + IOC→ISO mapping
# ---------------------------------------------------------------------------


class TestParseIocAlpha2:
    def test_basic(self) -> None:
        assert parse_ioc_alpha2("IFSC - World Cup (L,S) - Chamonix (FRA) 2022") == "FR"

    def test_ioc_differs_from_iso(self) -> None:
        # SUI (IOC) → CH (ISO alpha-2), NOT "SU"; GER → DE; SLO → SI.
        assert parse_ioc_alpha2("... Villars (SUI) 2021") == "CH"
        assert parse_ioc_alpha2("... Munich (GER) 2018") == "DE"
        assert parse_ioc_alpha2("... Kranj (SLO) 2017") == "SI"

    def test_discipline_tag_is_not_a_country(self) -> None:
        # "(B,L,S)" is a discipline tag, not a 3-letter code → no match.
        assert parse_ioc_alpha2("IFSC World Championships (B,L,S) - Moscow 2021") is None

    def test_no_code(self) -> None:
        assert parse_ioc_alpha2("IFSC World Cup Innsbruck 2025") is None

    def test_alpha2_to_alpha3(self) -> None:
        # IOC SUI → alpha-2 CH → ISO alpha-3 CHE (used for the Country node).
        assert alpha2_to_alpha3("CH") == "CHE"
        assert alpha2_to_alpha3("SI") == "SVN"
        assert alpha2_to_alpha3(None) is None


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
# Country-constrained disambiguation + end-to-end resolve_event
# ---------------------------------------------------------------------------

# An index seeded with ambiguous names (Madrid in CO/ES/US; Arco in IT/US) plus
# the canonical spellings used by the override redirects.
_AMBIGUOUS_RECORDS: list[dict[str, object]] = [
    {
        "geonameid": 3674962,
        "name": "Madrid",
        "lat": 4.0,
        "lon": -74.0,
        "country": "CO",
        "timezone": "America/Bogota",
    },
    {
        "geonameid": 3117735,
        "name": "Madrid",
        "lat": 40.4165,
        "lon": -3.70256,
        "country": "ES",
        "timezone": "Europe/Madrid",
    },
    {
        "geonameid": 3170831,
        "name": "Arco",
        "lat": 45.918,
        "lon": 10.884,
        "country": "IT",
        "timezone": "Europe/Rome",
    },
    {
        "geonameid": 5552450,
        "name": "Arco",
        "lat": 32.0,
        "lon": -109.0,
        "country": "US",
        "timezone": "America/Phoenix",
    },
    {
        "geonameid": 3027301,
        "name": "Chamonix-Mont-Blanc",
        "lat": 45.92375,
        "lon": 6.86933,
        "country": "FR",
        "timezone": "Europe/Paris",
    },
    {
        "geonameid": 2658126,
        "name": "Villars-sur-Ollon",
        "lat": 46.3,
        "lon": 7.06,
        "country": "CH",
        "timezone": "Europe/Zurich",
    },
    {
        "geonameid": 1835848,
        "name": "Seoul",
        "lat": 37.566,
        "lon": 126.978,
        "country": "KR",
        "timezone": "Asia/Seoul",
    },
]


@pytest.fixture
def ambiguous_index() -> GeoNamesIndex:
    return GeoNamesIndex.from_records(_AMBIGUOUS_RECORDS)


class TestCountryConstrainedLookup:
    def test_madrid_resolves_to_spain_not_colombia(self, ambiguous_index: GeoNamesIndex) -> None:
        # The headline bug: an unconstrained Madrid lookup grabbed Colombia.
        point = ambiguous_index.lookup("Madrid", "ES")
        assert point is not None
        assert point.geonameid == 3117735
        assert point.timezone == "Europe/Madrid"

    def test_arco_resolves_to_italy(self, ambiguous_index: GeoNamesIndex) -> None:
        point = ambiguous_index.lookup("Arco", "IT")
        assert point is not None
        assert point.geonameid == 3170831

    def test_ambiguous_without_country_returns_none(self, ambiguous_index: GeoNamesIndex) -> None:
        # No country + multiple countries for the name → refuse to guess.
        assert ambiguous_index.lookup("Madrid", None) is None
        assert ambiguous_index.lookup("Arco", None) is None

    def test_unique_city_still_falls_back(self, ambiguous_index: GeoNamesIndex) -> None:
        # Seoul appears in exactly one country → safe to resolve uncountried.
        point = ambiguous_index.lookup("Seoul", None)
        assert point is not None and point.geonameid == 1835848


class TestResolveEvent:
    def test_madrid_region_name_resolves_to_spain(self, ambiguous_index: GeoNamesIndex) -> None:
        # "Comunidad de Madrid" with no parens code → override pins ES.
        res = resolve_event("World Climbing Series Comunidad de Madrid 2026", ambiguous_index)
        assert res.point is not None
        assert res.point.geonameid == 3117735
        assert alpha2_to_alpha3(res.alpha2) == "ESP"

    def test_chamonix_canonical_redirect(self, ambiguous_index: GeoNamesIndex) -> None:
        res = resolve_event("IFSC World Cup Chamonix 2025", ambiguous_index)
        assert res.point is not None
        assert res.point.name == "Chamonix-Mont-Blanc"
        assert alpha2_to_alpha3(res.alpha2) == "FRA"

    def test_villars_redirects_to_swiss_canonical(self, ambiguous_index: GeoNamesIndex) -> None:
        res = resolve_event("IFSC - World Cup (L,S) - Villars (SUI) 2021", ambiguous_index)
        assert res.point is not None
        assert res.point.name == "Villars-sur-Ollon"
        assert alpha2_to_alpha3(res.alpha2) == "CHE"

    def test_arco_parens_code_constrains(self, ambiguous_index: GeoNamesIndex) -> None:
        res = resolve_event("IFSC Climbing Worldcup (L,S) - Arco (ITA) 2018", ambiguous_index)
        assert res.point is not None and res.point.geonameid == 3170831

    def test_absent_city_override(self, ambiguous_index: GeoNamesIndex) -> None:
        # Wujiang is not in cities1000 → curated absent-city override.
        res = resolve_event("IFSC World Cup Wujiang 2025", ambiguous_index)
        assert res.point is not None
        assert res.point.name == "Wujiang"
        assert res.point.timezone == "Asia/Shanghai"
        assert alpha2_to_alpha3(res.alpha2) == "CHN"

    def test_alpha2_hint_backfills_country(self, ambiguous_index: GeoNamesIndex) -> None:
        # A bare "Madrid" with an explicit alpha-2 hint (backfill) resolves to ES.
        res = resolve_event("IFSC World Cup Madrid 2025", ambiguous_index, alpha2="ES")
        assert res.point is not None and res.point.geonameid == 3117735

    def test_override_alpha2_pin(self) -> None:
        # The pure country settler used by the sync backfill chain.
        assert override_alpha2("World Climbing Series Comunidad de Madrid 2026") == "ES"
        assert override_alpha2("IFSC World Cup Innsbruck 2025") is None


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


# ---------------------------------------------------------------------------
# Country → capital-city timezone fallback (issue #42)
# ---------------------------------------------------------------------------


class TestCountryCapitalTz:
    def test_every_value_is_a_valid_iana_zone(self) -> None:
        # Guards the curated table against typos: every value must be a real
        # IANA zone (stdlib zoneinfo is the authority) so utc_offset_hours and
        # the neo4j TimeZone node never receive a bogus id.
        valid = available_timezones()
        bad = {iso: tz for iso, tz in COUNTRY_CAPITAL_TZ.items() if tz not in valid}
        assert bad == {}, f"invalid IANA zones in COUNTRY_CAPITAL_TZ: {bad}"

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("JPN", "Asia/Tokyo"),
            ("USA", "America/New_York"),  # Washington DC
            ("AUT", "Europe/Vienna"),
            ("FRA", "Europe/Paris"),
            ("AUS", "Australia/Sydney"),  # Canberra
            ("RUS", "Europe/Moscow"),
            ("KOR", "Asia/Seoul"),
            ("SVN", "Europe/Ljubljana"),
        ],
    )
    def test_known_capitals(self, code: str, expected: str) -> None:
        assert country_capital_tz(code) == expected

    def test_ioc_aliases_match_iso3(self) -> None:
        # IOC codes that diverge from ISO3 must resolve to the same zone, since
        # athlete nationality may arrive in either form.
        for ioc, iso3 in [
            ("SUI", "CHE"),
            ("GER", "DEU"),
            ("SLO", "SVN"),
            ("NED", "NLD"),
            ("IRI", "IRN"),
            ("TPE", "TWN"),
            ("RSA", "ZAF"),
        ]:
            assert country_capital_tz(ioc) == country_capital_tz(iso3)
            assert country_capital_tz(ioc) is not None

    def test_case_insensitive_and_whitespace(self) -> None:
        assert country_capital_tz("  jpn ") == "Asia/Tokyo"

    def test_unknown_and_empty_return_none(self) -> None:
        assert country_capital_tz("ZZZ") is None
        assert country_capital_tz("") is None
        assert country_capital_tz(None) is None


# ---------------------------------------------------------------------------
# Absent-city override-table integrity (issue #47)
# ---------------------------------------------------------------------------


class TestCityOverridesIntegrity:
    def test_absent_city_geonameids_are_unique_and_nonzero(self) -> None:
        # City nodes are keyed by ``vocab.city(geonameid)``, so two absent-city
        # overrides sharing a ``geonameid`` (the old ``0`` sentinel) MERGE into a
        # single City node — that's the Bali↔Wujiang↔Keqiao collision of #47.
        ids = [ov.point.geonameid for ov in _CITY_OVERRIDES.values() if ov.point is not None]
        assert all(gid != 0 for gid in ids), "absent-city override still uses geonameid=0"
        assert len(ids) == len(set(ids)), f"absent-city overrides share a geonameid: {ids}"

    def test_validate_overrides_accepts_the_real_table(self) -> None:
        # The shipped table must satisfy its own guard.
        _validate_overrides(_CITY_OVERRIDES)

    def test_validate_overrides_rejects_zero_sentinel(self) -> None:
        bad = {
            "x": _Override(
                alpha2="CN",
                point=GeoPoint(lat=0.0, lon=0.0, geonameid=0, name="X", timezone="UTC"),
            ),
        }
        with pytest.raises(ValueError, match="geonameid"):
            _validate_overrides(bad)

    def test_validate_overrides_rejects_duplicate_geonameid(self) -> None:
        dup = GeoPoint(lat=1.0, lon=2.0, geonameid=42, name="A", timezone="UTC")
        dup2 = GeoPoint(lat=3.0, lon=4.0, geonameid=42, name="B", timezone="UTC")
        bad = {
            "a": _Override(alpha2="CN", point=dup),
            "b": _Override(alpha2="ID", point=dup2),
        }
        with pytest.raises(ValueError, match="42"):
            _validate_overrides(bad)
