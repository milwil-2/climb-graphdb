"""climber_network.source.pg — Read-only SQLAlchemy access to climbing-elo Postgres.

This module mirrors the *columns* of the upstream climbing-elo schema as plain
declarative READ models. It deliberately does **not** import the
``climbing_elo`` package (hard isolation rule, see CLAUDE.md): the upstream data
is consumed READ-ONLY over a database connection only.

Design notes
------------
* The engine is built from ``config.DATABASE_URL()`` with ``sslmode=require``
  (Supabase requires TLS) and ``pool_pre_ping=True`` (Supabase poolers drop
  idle connections, so verify each checkout).
* A SQLite URL is accepted verbatim for tests — the ``sslmode`` connect-arg is
  Postgres-only and is therefore only attached for Postgres URLs.
* No model carries SQLAlchemy ``relationship()`` declarations and no method ever
  issues INSERT/UPDATE/DELETE or DDL. Tests create the schema with
  ``Base.metadata.create_all`` against an in-memory SQLite database; production
  never does (the upstream tables already exist).
* Enum-typed upstream columns (gender, tier, round_type, discipline, kind) are
  read back as their raw string ``values`` (e.g. ``"final"``, ``"L"``) — we map
  to graph vocabulary in the sync layer rather than re-deriving the enums here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date

from sqlalchemy import Boolean, Date, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from climber_network import config


class Base(DeclarativeBase):
    """Declarative base for the read-only mirror models."""


# ---------------------------------------------------------------------------
# READ models — columns mirror the climbing-elo schema (PRD §6.1).
# Enum columns are stored as their string values; we read them as plain str.
# ---------------------------------------------------------------------------


class Athlete(Base):
    __tablename__ = "athletes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    year_of_birth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nationality: Mapped[str | None] = mapped_column(String, nullable=True)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    photo_url: Mapped[str | None] = mapped_column(String, nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wingspan_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retired_at: Mapped[date | None] = mapped_column(Date, nullable=True)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    tier: Mapped[str] = mapped_column(String, nullable=False)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    discipline: Mapped[str] = mapped_column(String, nullable=False)


class Round(Base):
    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    round_type: Mapped[str] = mapped_column(String, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    athlete_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    athlete_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_score: Mapped[str | None] = mapped_column(String, nullable=True)
    score_normalized: Mapped[float | None] = mapped_column(Float, nullable=True)
    dnf: Mapped[bool] = mapped_column(Boolean, default=False)
    dns: Mapped[bool] = mapped_column(Boolean, default=False)


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(Integer, nullable=False)
    discipline: Mapped[str] = mapped_column(String, nullable=False)
    mu: Mapped[float] = mapped_column(Float, nullable=False)
    sigma: Mapped[float] = mapped_column(Float, nullable=False)
    n_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_event_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    provisional: Mapped[bool] = mapped_column(Boolean, default=True)


class RatingHistory(Base):
    __tablename__ = "rating_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    round_id: Mapped[int] = mapped_column(Integer, nullable=False)
    mu_before: Mapped[float] = mapped_column(Float, nullable=False)
    mu_after: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_before: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_after: Mapped[float] = mapped_column(Float, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False, default="pair")


# ---------------------------------------------------------------------------
# Engine / session helpers — read-only.
# ---------------------------------------------------------------------------


def make_engine(url: str | None = None) -> Engine:
    """Return a SQLAlchemy engine for the upstream climbing-elo database.

    Args:
        url: Optional explicit connection URL. When omitted, the URL is read
            from ``config.DATABASE_URL()``. A SQLite URL (e.g.
            ``sqlite:///:memory:``) is accepted verbatim for tests.

    The engine is configured with ``pool_pre_ping=True`` (poolers drop idle
    connections) and, for Postgres URLs only, ``sslmode=require`` (Supabase
    mandates TLS). No DDL or writes are ever issued through this engine.
    """
    raw_url = url if url is not None else config.DATABASE_URL()
    parsed = make_url(raw_url)
    connect_args: dict[str, object] = {}
    if parsed.get_backend_name() in ("postgresql", "postgres"):
        # Use psycopg v3 (the installed `psycopg[binary]`), NOT SQLAlchemy's
        # default psycopg2 — which we don't depend on. A bare `postgresql://`
        # URL would otherwise try to import psycopg2 and fail at connect time.
        parsed = parsed.set(drivername="postgresql+psycopg")
        # Supabase requires TLS; sslmode is a libpq/psycopg connect-arg.
        if "sslmode" not in parsed.query:
            connect_args["sslmode"] = "require"
    return create_engine(parsed, pool_pre_ping=True, connect_args=connect_args)


@contextmanager
def read_session(engine: Engine) -> Iterator[Session]:
    """Yield a read-only Session, rolling back on exit so no writes can persist.

    The session is always rolled back (never committed) — this is a defensive
    guarantee that nothing written accidentally reaches the upstream store.
    """
    session = Session(engine)
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def iter_rows(session: Session, model: type[Base]) -> Iterator[Base]:
    """Yield every row of *model* in primary-key order (stable, read-only)."""
    yield from session.query(model).order_by(model.id).all()  # type: ignore[attr-defined]
