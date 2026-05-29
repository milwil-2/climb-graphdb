"""Unit tests for climber_network.config.

All tests exercise pure-Python logic (dataclass defaults, env-var getters).
No database or network access required.
"""

from __future__ import annotations

import os

import pytest

from climber_network.config import (
    CORS_ALLOW_ORIGINS,
    DATABASE_URL,
    GROQ_API_KEY,
    INGEST_API_KEY,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    NEWS_API_KEY,
    TRAVEL_PARAMS,
    TravelParams,
)

# ---------------------------------------------------------------------------
# TravelParams defaults
# ---------------------------------------------------------------------------


class TestTravelParamsDefaults:
    def test_cruise_kmh(self) -> None:
        assert TRAVEL_PARAMS.cruise_kmh == 800.0

    def test_flight_overhead_h(self) -> None:
        assert TRAVEL_PARAMS.flight_overhead_h == 1.5

    def test_weights_sum_to_one(self) -> None:
        assert abs(TRAVEL_PARAMS.w1 + TRAVEL_PARAMS.w2 - 1.0) < 1e-9

    def test_w1(self) -> None:
        assert TRAVEL_PARAMS.w1 == 0.7

    def test_w2(self) -> None:
        assert TRAVEL_PARAMS.w2 == 0.3

    def test_fatigue_full_h(self) -> None:
        assert TRAVEL_PARAMS.fatigue_full_h == 12.0

    def test_fatigue_decay_days(self) -> None:
        assert TRAVEL_PARAMS.fatigue_decay_days == 4.0

    def test_recovery_cap_days(self) -> None:
        assert TRAVEL_PARAMS.recovery_cap_days == 5.0

    def test_arrive_days_before(self) -> None:
        assert TRAVEL_PARAMS.arrive_days_before == 2

    def test_swing_gap_days(self) -> None:
        assert TRAVEL_PARAMS.swing_gap_days == 10

    def test_model_version(self) -> None:
        assert TRAVEL_PARAMS.model_version == "l3-v1"

    def test_is_frozen(self) -> None:
        """TravelParams must be a frozen dataclass (immutable)."""
        with pytest.raises((AttributeError, TypeError)):
            TRAVEL_PARAMS.cruise_kmh = 999.0  # type: ignore[misc]

    def test_singleton_identity(self) -> None:
        """The module-level TRAVEL_PARAMS is the same object on re-import."""
        from climber_network.config import TRAVEL_PARAMS as tp2

        assert TRAVEL_PARAMS is tp2

    def test_can_construct_custom(self) -> None:
        custom = TravelParams(cruise_kmh=600.0, model_version="test-v0")
        assert custom.cruise_kmh == 600.0
        assert custom.model_version == "test-v0"
        # Other fields retain defaults
        assert custom.w1 == 0.7

    def test_cruise_kmh_positive(self) -> None:
        assert TRAVEL_PARAMS.cruise_kmh > 0

    def test_all_numeric_defaults_positive(self) -> None:
        tp = TravelParams()
        for field_name in (
            "cruise_kmh",
            "flight_overhead_h",
            "w1",
            "w2",
            "fatigue_full_h",
            "fatigue_decay_days",
            "recovery_cap_days",
        ):
            val = getattr(tp, field_name)
            assert val > 0, f"{field_name} should be positive, got {val}"

    def test_int_fields_are_int(self) -> None:
        assert isinstance(TRAVEL_PARAMS.arrive_days_before, int)
        assert isinstance(TRAVEL_PARAMS.swing_gap_days, int)


# ---------------------------------------------------------------------------
# Environment getters — test fallback defaults when env vars are absent
# ---------------------------------------------------------------------------


class TestEnvGetters:
    def _clear_env(self, *keys: str) -> dict[str, str | None]:
        """Remove env vars and return their original values for cleanup."""
        original = {k: os.environ.pop(k, None) for k in keys}
        return original

    def _restore_env(self, saved: dict[str, str | None]) -> None:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_neo4j_uri_default(self) -> None:
        saved = self._clear_env("NEO4J_URI")
        try:
            assert NEO4J_URI() == "bolt://localhost:7687"
        finally:
            self._restore_env(saved)

    def test_neo4j_uri_from_env(self) -> None:
        saved = self._clear_env("NEO4J_URI")
        os.environ["NEO4J_URI"] = "neo4j+s://myaura.databases.neo4j.io"
        try:
            assert NEO4J_URI() == "neo4j+s://myaura.databases.neo4j.io"
        finally:
            self._restore_env(saved)

    def test_neo4j_user_default(self) -> None:
        saved = self._clear_env("NEO4J_USER")
        try:
            assert NEO4J_USER() == "neo4j"
        finally:
            self._restore_env(saved)

    def test_neo4j_password_default_empty(self) -> None:
        saved = self._clear_env("NEO4J_PASSWORD")
        try:
            assert NEO4J_PASSWORD() == ""
        finally:
            self._restore_env(saved)

    def test_database_url_default_empty(self) -> None:
        saved = self._clear_env("DATABASE_URL")
        try:
            assert DATABASE_URL() == ""
        finally:
            self._restore_env(saved)

    def test_groq_api_key_default_empty(self) -> None:
        saved = self._clear_env("GROQ_API_KEY")
        try:
            assert GROQ_API_KEY() == ""
        finally:
            self._restore_env(saved)

    def test_ingest_api_key_default_empty(self) -> None:
        saved = self._clear_env("INGEST_API_KEY")
        try:
            assert INGEST_API_KEY() == ""
        finally:
            self._restore_env(saved)

    def test_news_api_key_default_empty(self) -> None:
        saved = self._clear_env("NEWS_API_KEY")
        try:
            assert NEWS_API_KEY() == ""
        finally:
            self._restore_env(saved)

    def test_cors_allow_origins_default(self) -> None:
        saved = self._clear_env("CORS_ALLOW_ORIGINS")
        try:
            assert CORS_ALLOW_ORIGINS() == "http://localhost:3000"
        finally:
            self._restore_env(saved)

    def test_cors_allow_origins_from_env(self) -> None:
        saved = self._clear_env("CORS_ALLOW_ORIGINS")
        os.environ["CORS_ALLOW_ORIGINS"] = "https://app.example.com,https://dev.example.com"
        try:
            raw = CORS_ALLOW_ORIGINS()
            origins = [o.strip() for o in raw.split(",") if o.strip()]
            assert origins == ["https://app.example.com", "https://dev.example.com"]
        finally:
            self._restore_env(saved)

    def test_cors_not_wildcard_by_default(self) -> None:
        saved = self._clear_env("CORS_ALLOW_ORIGINS")
        try:
            assert CORS_ALLOW_ORIGINS() != "*"
        finally:
            self._restore_env(saved)


# ---------------------------------------------------------------------------
# Network-test creds guard (conftest.has_live_neo4j_creds)
# ---------------------------------------------------------------------------


class TestHasLiveNeo4jCreds:
    """The guard backing the ``live_neo4j`` skip fixture (see conftest)."""

    def _set_dummies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reuse conftest's single source of truth so the test can't drift if the
        # dummy fallbacks ever change.
        from tests.conftest import _DUMMY_NEO4J

        for key, val in _DUMMY_NEO4J.items():
            monkeypatch.setenv(key, val)

    def test_dummy_defaults_are_not_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.conftest import has_live_neo4j_creds

        self._set_dummies(monkeypatch)
        assert has_live_neo4j_creds() is False

    def test_real_uri_is_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.conftest import has_live_neo4j_creds

        self._set_dummies(monkeypatch)
        monkeypatch.setenv("NEO4J_URI", "neo4j+s://real.databases.neo4j.io")
        assert has_live_neo4j_creds() is True

    def test_real_password_is_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.conftest import has_live_neo4j_creds

        self._set_dummies(monkeypatch)
        monkeypatch.setenv("NEO4J_PASSWORD", "s3cret-from-aura")
        assert has_live_neo4j_creds() is True
