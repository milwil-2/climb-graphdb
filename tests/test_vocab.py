"""Unit tests for climber_network.vocab.

All tests are pure-Python — no database required.
"""

from __future__ import annotations

import hashlib

import pytest

from climber_network.vocab import (
    VALID_NODE_LABELS,
    VALID_REL_TYPES,
    assert_label,
    assert_rel,
    ath,
    camp,
    city,
    ctry,
    disc,
    doc,
    evt,
    inj,
    leg,
    perf,
    rat,
    rest,
    rnd,
    run,
    seas,
    sig,
    slug,
    source_id,
    tz,
    ven,
)

# ---------------------------------------------------------------------------
# Vocabulary completeness
# ---------------------------------------------------------------------------


def test_node_labels_count() -> None:
    """VALID_NODE_LABELS must contain exactly 19 labels."""
    assert len(VALID_NODE_LABELS) == 19


def test_rel_types_count() -> None:
    """VALID_REL_TYPES must contain exactly 24 relationship types."""
    assert len(VALID_REL_TYPES) == 24


def test_season_summary_vocab() -> None:
    """SeasonSummary label, HAD_SEASON rel, and the seas() id builder (#48 Phase 4)."""
    assert "SeasonSummary" in VALID_NODE_LABELS
    assert "HAD_SEASON" in VALID_REL_TYPES
    assert seas("ath:5", 2024, "L") == "seas:ath:5:2024:L"


def test_expected_node_labels_present() -> None:
    expected = {
        "Athlete",
        "Event",
        "Round",
        "Performance",
        "Discipline",
        "Rating",
        "Venue",
        "City",
        "Country",
        "TimeZone",
        "TravelLeg",
        "RestednessState",
        "TrainingSignal",
        "InjuryEvent",
        "TrainingCamp",
        "Source",
        "Document",
        "ExtractionRun",
        "SeasonSummary",
    }
    assert expected == VALID_NODE_LABELS


def test_expected_rel_types_present() -> None:
    expected = {
        "COMPETED_IN",
        "OF_ROUND",
        "OF_EVENT",
        "IN_DISCIPLINE",
        "HAS_RATING",
        "FACED",
        "HELD_AT",
        "IN_CITY",
        "IN_COUNTRY",
        "IN_TIMEZONE",
        "REPRESENTS",
        "BASED_IN",
        "TRAVELED",
        "TO_EVENT",
        "HAD_STATE",
        "AT_EVENT",
        "HAS_SIGNAL",
        "HAD_INJURY",
        "ATTENDED",
        "SIGNAL_AT",
        "EVIDENCED_BY",
        "FROM_SOURCE",
        "EXTRACTED_BY",
        "HAD_SEASON",
    }
    assert expected == VALID_REL_TYPES


# ---------------------------------------------------------------------------
# assert_label
# ---------------------------------------------------------------------------


class TestAssertLabel:
    def test_valid_label_returns_value(self) -> None:
        assert assert_label("Athlete") == "Athlete"
        assert assert_label("ExtractionRun") == "ExtractionRun"
        assert assert_label("TimeZone") == "TimeZone"

    def test_all_valid_labels_pass(self) -> None:
        for label in VALID_NODE_LABELS:
            assert assert_label(label) == label

    def test_invalid_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown node label"):
            assert_label("EVIL")

    def test_injection_attempt_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_label("Athlete; DROP DATABASE neo4j")

    def test_sql_style_injection_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_label("Athlete' OR '1'='1")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_label("")

    def test_lowercase_valid_label_raises(self) -> None:
        # Labels are case-sensitive: "athlete" is not in the set
        with pytest.raises(ValueError):
            assert_label("athlete")

    def test_unknown_label_error_message_lists_valid(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            assert_label("Widget")
        assert "Widget" in str(exc_info.value)
        assert "Athlete" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_rel
# ---------------------------------------------------------------------------


class TestAssertRel:
    def test_valid_rel_returns_value(self) -> None:
        assert assert_rel("COMPETED_IN") == "COMPETED_IN"
        assert assert_rel("EXTRACTED_BY") == "EXTRACTED_BY"

    def test_all_valid_rels_pass(self) -> None:
        for rel in VALID_REL_TYPES:
            assert assert_rel(rel) == rel

    def test_invalid_rel_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown relationship type"):
            assert_rel("EVIL")

    def test_injection_attempt_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_rel("COMPETED_IN] RETURN 1 //")

    def test_lowercase_valid_rel_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_rel("competed_in")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_rel("")

    def test_unknown_rel_error_message_lists_valid(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            assert_rel("KNOWS")
        assert "KNOWS" in str(exc_info.value)
        assert "COMPETED_IN" in str(exc_info.value)


# ---------------------------------------------------------------------------
# slug()
# ---------------------------------------------------------------------------


class TestSlug:
    def test_simple_lowercase(self) -> None:
        assert slug("hello world") == "hello-world"

    def test_punctuation_collapsed(self) -> None:
        assert slug("Lead / Boulder") == "lead-boulder"

    def test_multiple_spaces(self) -> None:
        assert slug("  gaps   here  ") == "gaps-here"

    def test_already_slug(self) -> None:
        assert slug("innsbruck") == "innsbruck"

    def test_numbers_preserved(self) -> None:
        assert slug("World Cup 2024") == "world-cup-2024"

    def test_unicode_normalized(self) -> None:
        # Non-ASCII letters get treated as non-alphanumeric and become hyphens
        result = slug("Briançon")
        assert "brian" in result
        assert "-" in result


# ---------------------------------------------------------------------------
# ID builders
# ---------------------------------------------------------------------------


class TestIdBuilders:
    # Simple prefix builders

    def test_ath(self) -> None:
        assert ath(42) == "ath:42"
        assert ath("99") == "ath:99"

    def test_evt(self) -> None:
        assert evt(1) == "evt:1"

    def test_rnd(self) -> None:
        assert rnd(7) == "rnd:7"

    def test_disc(self) -> None:
        assert disc("lead") == "disc:lead"

    def test_ven(self) -> None:
        assert ven("innsbruck") == "ven:innsbruck"

    def test_city(self) -> None:
        assert city(2775220) == "city:2775220"
        assert city("2775220") == "city:2775220"

    def test_ctry(self) -> None:
        assert ctry("AUT") == "ctry:AUT"

    def test_source_id(self) -> None:
        assert source_id("ifsc.results.edu") == "src:ifsc.results.edu"

    def test_run(self) -> None:
        assert run("2024-01-01T00:00:00") == "run:2024-01-01T00:00:00"

    # Compound builders

    def test_perf_format(self) -> None:
        result = perf("rnd:5", "ath:42")
        assert result == "perf:rnd:5:ath:42"

    def test_rat_format(self) -> None:
        result = rat("ath:42", "lead")
        assert result == "rat:ath:42:lead"

    def test_leg_format(self) -> None:
        result = leg("ath:1", "evt:100")
        assert result == "leg:ath:1:evt:100"

    def test_rest_format(self) -> None:
        result = rest("ath:1", "evt:100")
        assert result == "rest:ath:1:evt:100"

    def test_sig_format(self) -> None:
        result = sig("ath:1", "instagram", "abc123")
        assert result == "sig:ath:1:instagram:abc123"

    def test_inj_format(self) -> None:
        result = inj("ath:1", "deadbeef")
        assert result == "inj:ath:1:deadbeef"

    def test_camp_format(self) -> None:
        result = camp("ath:1", "2024-06-01")
        assert result == "camp:ath:1:2024-06-01"

    # TimeZone: "/" → "_"

    def test_tz_slash_replaced(self) -> None:
        assert tz("America/New_York") == "tz:America_New_York"

    def test_tz_europe(self) -> None:
        assert tz("Europe/Vienna") == "tz:Europe_Vienna"

    def test_tz_no_slash(self) -> None:
        assert tz("UTC") == "tz:UTC"

    def test_tz_multi_slash(self) -> None:
        # e.g. "America/Indiana/Indianapolis"
        assert tz("America/Indiana/Indianapolis") == "tz:America_Indiana_Indianapolis"

    # Document: SHA-1 of URL

    def test_doc_sha1(self) -> None:
        url = "https://example.com/page"
        expected_sha = hashlib.sha1(url.encode()).hexdigest()
        assert doc(url) == f"doc:{expected_sha}"

    def test_doc_deterministic(self) -> None:
        url = "https://example.com/page"
        assert doc(url) == doc(url)

    def test_doc_different_urls(self) -> None:
        assert doc("https://a.com") != doc("https://b.com")
