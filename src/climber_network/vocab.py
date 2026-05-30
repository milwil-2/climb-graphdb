"""climber_network.vocab — Closed vocabularies, ID builders, and injection-safety helpers.

All Cypher label and relationship-type interpolation MUST go through
``assert_label`` / ``assert_rel`` before use in query strings. This is the
single enforcement point: if a value is not in the frozensets below, a
ValueError is raised and the query is never executed.
"""

from __future__ import annotations

import hashlib
import re

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

VALID_NODE_LABELS: frozenset[str] = frozenset(
    {
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
)

VALID_REL_TYPES: frozenset[str] = frozenset(
    {
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
)

# ---------------------------------------------------------------------------
# Injection-safety validators
# ---------------------------------------------------------------------------


def assert_label(label: str) -> str:
    """Return *label* if it is in VALID_NODE_LABELS, else raise ValueError.

    Always call this before interpolating a label into a Cypher string::

        cypher = f"MERGE (n:{assert_label(label)} {{id:$id}})"
    """
    if label not in VALID_NODE_LABELS:
        raise ValueError(
            f"Unknown node label {label!r}. Must be one of: {sorted(VALID_NODE_LABELS)}"
        )
    return label


def assert_rel(rel: str) -> str:
    """Return *rel* if it is in VALID_REL_TYPES, else raise ValueError.

    Always call this before interpolating a relationship type into Cypher::

        cypher = f"MERGE (a)-[:{assert_rel(rel_type)}]->(b)"
    """
    if rel not in VALID_REL_TYPES:
        raise ValueError(
            f"Unknown relationship type {rel!r}. Must be one of: {sorted(VALID_REL_TYPES)}"
        )
    return rel


# ---------------------------------------------------------------------------
# Text helper
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Return a URL/ID-safe slug: lowercase, non-alphanumeric runs → hyphen.

    >>> slug("Briançon / Les Orres")
    'brian-on-les-orres'
    """
    lowered = text.lower()
    return _SLUG_RE.sub("-", lowered).strip("-")


# ---------------------------------------------------------------------------
# ID builders — all return namespaced strings
# ---------------------------------------------------------------------------


def ath(db_id: int | str) -> str:
    """Athlete node id: ``ath:<db_id>``."""
    return f"ath:{db_id}"


def evt(db_id: int | str) -> str:
    """Event node id: ``evt:<db_id>``."""
    return f"evt:{db_id}"


def rnd(db_id: int | str) -> str:
    """Round node id: ``rnd:<db_id>``."""
    return f"rnd:{db_id}"


def perf(round_id: str, athlete_id: str) -> str:
    """Performance node id: ``perf:<round_id>:<athlete_id>``."""
    return f"perf:{round_id}:{athlete_id}"


def disc(code: str) -> str:
    """Discipline node id: ``disc:<code>``."""
    return f"disc:{code}"


def rat(athlete_id: str, disc_code: str) -> str:
    """Rating node id: ``rat:<athlete_id>:<disc_code>``."""
    return f"rat:{athlete_id}:{disc_code}"


def ven(slug_str: str) -> str:
    """Venue node id: ``ven:<slug>``."""
    return f"ven:{slug_str}"


def city(geonameid: int | str) -> str:
    """City node id: ``city:<geonameid>``."""
    return f"city:{geonameid}"


def ctry(iso3: str) -> str:
    """Country node id: ``ctry:<iso3>`` (ISO 3166-1 alpha-3)."""
    return f"ctry:{iso3}"


def tz(iana: str) -> str:
    """TimeZone node id: ``tz:<iana>`` with '/' replaced by '_'.

    >>> tz("America/New_York")
    'tz:America_New_York'
    """
    return f"tz:{iana.replace('/', '_')}"


def leg(ath_id: str, to_evt_id: str) -> str:
    """TravelLeg node id: ``leg:<ath_id>:<to_evt_id>``."""
    return f"leg:{ath_id}:{to_evt_id}"


def rest(ath_id: str, evt_id: str) -> str:
    """RestednessState node id: ``rest:<ath_id>:<evt_id>``."""
    return f"rest:{ath_id}:{evt_id}"


def sig(ath_id: str, src: str, hash_val: str) -> str:
    """TrainingSignal node id: ``sig:<ath_id>:<src>:<hash>``."""
    return f"sig:{ath_id}:{src}:{hash_val}"


def inj(ath_id: str, hash_val: str) -> str:
    """InjuryEvent node id: ``inj:<ath_id>:<hash>``."""
    return f"inj:{ath_id}:{hash_val}"


def camp(ath_id: str, start_date: str) -> str:
    """TrainingCamp node id: ``camp:<ath_id>:<start_date>``."""
    return f"camp:{ath_id}:{start_date}"


def source_id(domain: str) -> str:
    """Source node id: ``src:<domain>``."""
    return f"src:{domain}"


def doc(url: str) -> str:
    """Document node id: ``doc:<sha1(url)>``.

    SHA-1 is used here purely as a deterministic content fingerprint for a
    stable node id, never for security or integrity guarantees — hence
    ``usedforsecurity=False``.
    """
    sha = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
    return f"doc:{sha}"


def run(ts: str) -> str:
    """ExtractionRun node id: ``run:<ts>``."""
    return f"run:{ts}"


def seas(athlete_id: str, season: int | str, disc_code: str) -> str:
    """SeasonSummary node id: ``seas:<athlete_id>:<season>:<disc_code>``.

    *athlete_id* is the full athlete node id (e.g. ``ath:5``); one summary per
    (athlete, season, discipline).
    """
    return f"seas:{athlete_id}:{season}:{disc_code}"
