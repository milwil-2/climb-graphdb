"""climber_network.config — Environment loading and application constants.

All environment access is centralised here. Callers import the getter
functions or the module-level ``TRAVEL_PARAMS`` instance; nothing else
should call ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root (two levels above this file:
# src/climber_network/config.py → src/climber_network/ → src/ → repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Environment getters
# ---------------------------------------------------------------------------


def NEO4J_URI() -> str:
    """Neo4j bolt/neo4j+s URI (default: bolt://localhost:7687)."""
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def NEO4J_USER() -> str:
    """Neo4j username.

    Note: on Aura the username is the *instance id* (e.g. ``your-instance-id``),
    not the literal string ``neo4j``. Always read from the environment.
    """
    return os.environ.get("NEO4J_USER", "neo4j")


def NEO4J_PASSWORD() -> str:
    """Neo4j password."""
    return os.environ.get("NEO4J_PASSWORD", "")


def DATABASE_URL() -> str:
    """PostgreSQL connection string (psycopg3 style)."""
    return os.environ.get("DATABASE_URL", "")


def GROQ_API_KEY() -> str:
    """Groq API key for LLM inference."""
    return os.environ.get("GROQ_API_KEY", "")


def INGEST_API_KEY() -> str:
    """Shared-secret key required by the ingest endpoints."""
    return os.environ.get("INGEST_API_KEY", "")


def NEWS_API_KEY() -> str:
    """API key for the news data source."""
    return os.environ.get("NEWS_API_KEY", "")


def CORS_ALLOW_ORIGINS() -> str:
    """Comma-separated list of allowed CORS origins.

    Split on comma in the FastAPI layer::

        origins = [o.strip() for o in config.CORS_ALLOW_ORIGINS().split(",") if o.strip()]
    """
    return os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3000")


# ---------------------------------------------------------------------------
# Travel / fatigue model constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TravelParams:
    """Immutable constants for the travel-fatigue scoring model.

    Attributes:
        cruise_kmh:         Effective aircraft cruise speed used for great-circle
                            distance → flight-time conversion (km/h).
        flight_overhead_h:  Fixed overhead added to every flight (check-in,
                            boarding, taxi, deplaning) in hours.
        w1:                 Weight applied to the departure-leg fatigue score.
        w2:                 Weight applied to the arrival-leg fatigue score
                            (w1 + w2 should equal 1.0).
        fatigue_full_h:     Flight duration (hours) that represents "fully
                            fatigued" — used to normalise raw flight time.
        fatigue_decay_days: Exponential half-life in days for fatigue recovery
                            after landing.
        recovery_cap_days:  Maximum number of recovery days credited (clamps the
                            exponential decay).
        arrive_days_before: Expected number of days the athlete arrives before
                            the event — used when exact arrival date is unknown.
        swing_gap_days:     Minimum days between consecutive events to be
                            classified as a "swing" rather than a consecutive
                            competition.
        model_version:      Opaque version string stamped on TravelLeg nodes so
                            the graph can distinguish re-computed legs.
    """

    cruise_kmh: float = 800.0
    flight_overhead_h: float = 1.5
    w1: float = 0.7
    w2: float = 0.3
    fatigue_full_h: float = 12.0
    fatigue_decay_days: float = 4.0
    recovery_cap_days: float = 5.0
    arrive_days_before: int = 2
    swing_gap_days: int = 10
    model_version: str = "l3-v1"


#: Module-level singleton — import and use directly.
TRAVEL_PARAMS: TravelParams = TravelParams()
