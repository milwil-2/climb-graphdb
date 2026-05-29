"""climber_network.source — Read-only access to the upstream climbing-elo store.

This package reads the climbing-elo PostgreSQL (Supabase) database via a
read-only SQLAlchemy connection. It never imports the ``climbing_elo`` package
itself (isolation constraint) and never emits writes or DDL.
"""

from __future__ import annotations

from climber_network.source import pg

__all__ = ["pg"]
