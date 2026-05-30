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
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, queries, rag

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


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    question: str


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    """GraphRAG endpoint: resolve an athlete, expand a subgraph, answer.

    Returns ``{answer, entities, subgraph}``. With no ``GROQ_API_KEY`` the
    deterministic graph-only fallback answer is returned (HTTP 200). If no
    athlete entity can be resolved from the question, responds with 404.
    """
    try:
        return rag.ask(req.question)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# U1–U5 read endpoints (api.queries)
# ---------------------------------------------------------------------------


@app.get("/athlete/{athlete_id}")
def athlete_profile(athlete_id: str) -> dict[str, Any]:
    """U4 — athlete props + ratings + recent events. 404 if the athlete is absent."""
    profile = queries.athlete_profile(athlete_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown athlete: {athlete_id}")
    return profile


@app.get("/athlete/{athlete_id}/neighborhood")
def athlete_neighborhood(athlete_id: str, hops: int = 2) -> dict[str, Any]:
    """U4 — bounded ``{nodes, edges}`` subgraph around the athlete. 404 if absent."""
    neighborhood = queries.athlete_neighborhood(athlete_id, hops=hops)
    if neighborhood is None:
        raise HTTPException(status_code=404, detail=f"Unknown athlete: {athlete_id}")
    return neighborhood


@app.get("/athlete/{athlete_id}/timeline")
def athlete_timeline(athlete_id: str) -> dict[str, Any]:
    """U5 — merged chronological events + RestednessStates (+ optional signals)."""
    timeline = queries.athlete_timeline(athlete_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"Unknown athlete: {athlete_id}")
    return timeline


@app.get("/head-to-head")
def head_to_head(a: str, b: str) -> dict[str, Any]:
    """U1 — the FACED aggregate between two athletes. 404 if either is absent."""
    result = queries.head_to_head(a, b)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown athlete(s): {a}, {b}")
    return result


@app.get("/venues/clusters")
def venue_clusters() -> dict[str, Any]:
    """U2 — venues ranked by repeated co-competition (distinct athletes / events)."""
    return {"clusters": queries.venue_clusters()}


@app.get("/insights/jetlagged-underperformers")
def jetlagged_underperformers() -> dict[str, Any]:
    """U3 — low-rested athletes who underperformed (residual > 0). Empty if no data."""
    return {"rows": queries.jetlagged_underperformers()}


@app.get("/insights/season-drivers")
def season_drivers() -> dict[str, Any]:
    """U6b — athlete-seasons ranked by under-performance, with season restedness.

    Ranked by the per-event-normalized ``mean_over_under`` (not the volume-biased
    cumulative ``over_under`` sum). Empty until the season aggregation sync
    (sync.season) has run.
    """
    return {"rows": queries.season_drivers()}


# ---------------------------------------------------------------------------
# Static neighborhood viz
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def index() -> FileResponse:
    """Serve the self-contained neighborhood-viz single page."""
    return FileResponse(_STATIC_DIR / "index.html")


# Mount the static directory too (so the page could reference assets if added).
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
