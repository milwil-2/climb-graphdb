"""FastAPI application for the Climber Network knowledge graph.

This module is the Vercel Python entry point (``api/index.py``), so it
exposes a module-level ``app``.  Routes are root-relative (no ``/api``
prefix) to match the Vercel rewrite convention.

CORS: origins are read from ``CORS_ALLOW_ORIGINS`` (comma-separated list).
The wildcard ``"*"`` is intentionally NOT used — callers must enumerate
allowed origins explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db

# Load .env so the app works when run locally with `uvicorn api.index:app`
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Climber Network API")

# CORS — split the comma-separated env var; default to localhost only
_raw_origins = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3000")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, object]:
    """Lightweight liveness probe — does NOT hit the database."""
    return {"status": "ok"}


@app.get("/graph/stats")
def graph_stats() -> dict[str, int]:
    """Return total node and relationship counts from the live database."""
    return db.graph_stats()
