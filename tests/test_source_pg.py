"""Tests for the read-only source engine builder (`climber_network.source.pg`).

Regression guard: the project depends on psycopg v3 (`psycopg[binary]`), not
psycopg2. A bare ``postgresql://`` URL must be routed to the ``+psycopg`` driver
so live connections don't fail with ``ModuleNotFoundError: psycopg2``.
"""

from __future__ import annotations

from climber_network.source.pg import make_engine


def test_make_engine_postgres_uses_psycopg3() -> None:
    eng = make_engine("postgresql://user:pw@host:5432/db")
    try:
        assert eng.url.drivername == "postgresql+psycopg"
        assert eng.dialect.driver == "psycopg"
    finally:
        eng.dispose()


def test_make_engine_normalizes_bare_postgres_scheme() -> None:
    eng = make_engine("postgres://user:pw@host:5432/db")
    try:
        assert eng.url.drivername == "postgresql+psycopg"
    finally:
        eng.dispose()


def test_make_engine_sqlite_passthrough() -> None:
    eng = make_engine("sqlite:///:memory:")
    try:
        assert eng.url.drivername == "sqlite"
    finally:
        eng.dispose()
