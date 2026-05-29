"""GraphRAG ``/ask`` implementation for the Climber Network API.

The flow is:

1. **Entity resolution** — resolve an athlete by (sub)name against the
   ``athlete_name`` text via a case-insensitive ``CONTAINS`` read query.
2. **Neighborhood expansion** — pull a bounded N-hop subgraph around the
   resolved athlete: the events they competed in, the venues those events were
   held at, and the rivals they ``FACED``.
3. **Context assembly** — turn that subgraph into a compact textual context.
4. **Answer** — if ``GROQ_API_KEY`` is set, ask Groq for a natural-language
   answer grounded in the context; otherwise return a deterministic
   **graph-only fallback** (the context plus a templated summary).

The function returns ``{"answer", "entities", "subgraph"}`` so the static viz
can render both the prose answer and the underlying graph.

Design constraints
------------------
* Self-contained: depends only on ``api.db`` (read accessor) and
  ``climber_network.config`` / ``climber_network.vocab`` — NOT on the
  not-yet-built ``api/queries.py``.
* Groq is imported **lazily inside** :func:`_groq_answer` so the module imports
  cleanly with no key and no ``groq`` package available at import time.
* All label / relationship interpolation in Cypher goes through
  ``assert_label`` / ``assert_rel`` (the injection-safety gate); athlete input
  is always passed as a bound parameter, never interpolated.
"""

from __future__ import annotations

from typing import Any

from climber_network import config
from climber_network.vocab import assert_label, assert_rel

from . import db

# How many hops / how many rivals & events to surface. Kept small so the
# context fits comfortably in an LLM prompt and the viz stays legible.
_MAX_EVENTS = 25
_MAX_RIVALS = 25

# ---------------------------------------------------------------------------
# Cypher (labels / rels gated through the vocab; athlete name is bound)
# ---------------------------------------------------------------------------

_ATHLETE = assert_label("Athlete")
_EVENT = assert_label("Event")
_VENUE = assert_label("Venue")
_COMPETED_IN = assert_rel("COMPETED_IN")
_HELD_AT = assert_rel("HELD_AT")
_FACED = assert_rel("FACED")

#: Case-insensitive substring match against the athlete name. Bound param only.
RESOLVE_CYPHER = (
    f"MATCH (a:{_ATHLETE}) "
    "WHERE toLower(a.name) CONTAINS toLower($name) "
    "RETURN a.id AS id, a.name AS name "
    "ORDER BY size(a.name) ASC "
    "LIMIT 5"
)

#: Bounded neighborhood expansion around a resolved athlete id.
NEIGHBORHOOD_CYPHER = (
    f"MATCH (a:{_ATHLETE} {{id:$id}}) "
    f"OPTIONAL MATCH (a)-[:{_COMPETED_IN}]->(e:{_EVENT}) "
    f"OPTIONAL MATCH (e)-[:{_HELD_AT}]->(v:{_VENUE}) "
    "WITH a, collect(DISTINCT {id:e.id, name:e.name, "
    "start_date:toString(e.start_date), discipline:e.discipline, "
    "venue:v.name})[..$max_events] AS events "
    f"OPTIONAL MATCH (a)-[f:{_FACED}]->(r:{_ATHLETE}) "
    "WITH a, events, collect(DISTINCT {id:r.id, name:r.name, "
    "meetings:f.meetings})[..$max_rivals] AS rivals "
    "RETURN a.id AS id, a.name AS name, events, rivals"
)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def resolve_athlete(name: str) -> dict[str, str] | None:
    """Return the best ``{"id", "name"}`` match for *name*, or ``None``.

    Matching is a case-insensitive substring against ``Athlete.name``; the
    shortest matching name wins (a proxy for the most exact match).
    """
    cleaned = name.strip()
    if not cleaned:
        return None
    rows = db.run_read(RESOLVE_CYPHER, name=cleaned)
    if not rows:
        return None
    top = rows[0]
    return {"id": str(top["id"]), "name": str(top["name"])}


def expand_neighborhood(athlete_id: str) -> dict[str, Any]:
    """Return a bounded subgraph dict around *athlete_id*.

    Shape::

        {
          "athlete": {"id", "name"},
          "events":  [{"id", "name", "start_date", "discipline", "venue"}, ...],
          "rivals":  [{"id", "name", "meetings"}, ...],
        }
    """
    rows = db.run_read(
        NEIGHBORHOOD_CYPHER,
        id=athlete_id,
        max_events=_MAX_EVENTS,
        max_rivals=_MAX_RIVALS,
    )
    if not rows:
        return {"athlete": {"id": athlete_id, "name": athlete_id}, "events": [], "rivals": []}
    row = rows[0]
    events = [e for e in (row.get("events") or []) if e and e.get("id")]
    rivals = [r for r in (row.get("rivals") or []) if r and r.get("id")]
    return {
        "athlete": {"id": str(row["id"]), "name": str(row["name"])},
        "events": events,
        "rivals": rivals,
    }


def subgraph_to_graph(subgraph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten the neighborhood into ``{"nodes": [...], "edges": [...]}``.

    Node ``type`` is one of ``athlete`` / ``event`` / ``venue`` / ``rival`` so
    the viz can colour them. Edge ``type`` mirrors the graph relationship.
    """
    athlete = subgraph["athlete"]
    nodes: list[dict[str, Any]] = [
        {"id": athlete["id"], "label": athlete["name"], "type": "athlete"}
    ]
    edges: list[dict[str, Any]] = []
    seen: set[str] = {athlete["id"]}

    for ev in subgraph["events"]:
        ev_id = str(ev["id"])
        if ev_id not in seen:
            nodes.append({"id": ev_id, "label": ev.get("name") or ev_id, "type": "event"})
            seen.add(ev_id)
        edges.append({"source": athlete["id"], "target": ev_id, "type": "COMPETED_IN"})
        venue = ev.get("venue")
        if venue:
            ven_id = f"ven:{venue}"
            if ven_id not in seen:
                nodes.append({"id": ven_id, "label": venue, "type": "venue"})
                seen.add(ven_id)
            edges.append({"source": ev_id, "target": ven_id, "type": "HELD_AT"})

    for rv in subgraph["rivals"]:
        rv_id = str(rv["id"])
        if rv_id not in seen:
            nodes.append({"id": rv_id, "label": rv.get("name") or rv_id, "type": "rival"})
            seen.add(rv_id)
        edges.append(
            {
                "source": athlete["id"],
                "target": rv_id,
                "type": "FACED",
                "meetings": rv.get("meetings"),
            }
        )

    return {"nodes": nodes, "edges": edges}


def build_context(subgraph: dict[str, Any]) -> str:
    """Render the subgraph as a compact, LLM-ready textual context."""
    athlete = subgraph["athlete"]
    lines: list[str] = [f"Athlete: {athlete['name']} (id {athlete['id']})."]

    events = subgraph["events"]
    if events:
        lines.append(f"Competed in {len(events)} event(s):")
        for ev in events:
            parts = [str(ev.get("name") or ev.get("id"))]
            if ev.get("discipline"):
                parts.append(f"discipline {ev['discipline']}")
            if ev.get("start_date"):
                parts.append(f"on {ev['start_date']}")
            if ev.get("venue"):
                parts.append(f"at {ev['venue']}")
            lines.append("  - " + ", ".join(parts))
    else:
        lines.append("No recorded events.")

    rivals = subgraph["rivals"]
    if rivals:
        lines.append(f"Faced {len(rivals)} rival(s):")
        for rv in rivals:
            meetings = rv.get("meetings")
            suffix = f" ({meetings} meeting(s))" if meetings is not None else ""
            lines.append(f"  - {rv.get('name') or rv.get('id')}{suffix}")
    else:
        lines.append("No recorded rivals.")

    return "\n".join(lines)


def _fallback_summary(subgraph: dict[str, Any]) -> str:
    """Deterministic prose summary used when no LLM key is configured."""
    athlete = subgraph["athlete"]
    n_events = len(subgraph["events"])
    n_rivals = len(subgraph["rivals"])
    summary = (
        f"{athlete['name']} appears in {n_events} event(s) and has faced "
        f"{n_rivals} rival(s) in the knowledge graph."
    )
    rivals = subgraph["rivals"]
    if rivals:
        top = max(rivals, key=lambda r: r.get("meetings") or 0)
        if top.get("meetings"):
            summary += (
                f" The most frequent opponent is {top.get('name') or top.get('id')} "
                f"with {top['meetings']} meeting(s)."
            )
    return summary


def _groq_answer(question: str, context: str) -> str:
    """Call Groq with the question + graph context. Imported lazily.

    Any failure (missing package, API error) is allowed to propagate to the
    caller, which converts it into a graceful graph-only fallback.
    """
    from groq import Groq  # lazy: keeps import-time safe without a key/package

    client = Groq(api_key=config.GROQ_API_KEY())
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an analyst for competition climbing. Answer the "
                    "question using ONLY the provided knowledge-graph context. "
                    "If the context is insufficient, say so. Be concise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
    )
    return parse_groq_answer(_completion_to_dict(completion))


def _completion_to_dict(completion: Any) -> dict[str, Any]:
    """Best-effort normalise a Groq SDK response object into a plain dict."""
    if isinstance(completion, dict):
        return completion
    to_dict = getattr(completion, "model_dump", None) or getattr(completion, "to_dict", None)
    if callable(to_dict):
        result: dict[str, Any] = to_dict()
        return result
    raise TypeError("Unexpected Groq completion shape")


def parse_groq_answer(payload: dict[str, Any]) -> str:
    """Extract the assistant message text from a Groq chat-completion payload.

    Kept as a standalone, offline-testable function so the parsing/shaping is
    covered by a checked-in fixture without a live key.
    """
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Groq response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq response has no message content")
    return content.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ask(question: str) -> dict[str, Any]:
    """Answer *question* via GraphRAG over the climbing knowledge graph.

    Returns ``{"answer", "entities", "subgraph"}``. Raises :class:`LookupError`
    when no athlete entity can be resolved from the question (the API layer maps
    this to a 404). Groq is optional: with no key (or on any Groq error) the
    deterministic graph-only fallback answer is returned.
    """
    entity = resolve_athlete(question)
    if entity is None:
        raise LookupError(f"No athlete entity could be resolved from: {question!r}")

    neighborhood = expand_neighborhood(entity["id"])
    context = build_context(neighborhood)
    graph = subgraph_to_graph(neighborhood)

    answer: str
    used_llm = False
    if config.GROQ_API_KEY():
        try:
            answer = _groq_answer(question, context)
            used_llm = True
        except Exception:
            # Graceful degradation: never fail the request because Groq is
            # unavailable — fall back to the deterministic graph summary.
            answer = _fallback_summary(neighborhood)
    else:
        answer = _fallback_summary(neighborhood)

    return {
        "answer": answer,
        "entities": [entity],
        "subgraph": {
            "context": context,
            "nodes": graph["nodes"],
            "edges": graph["edges"],
        },
        "used_llm": used_llm,
    }
