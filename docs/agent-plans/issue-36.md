# Plan — issue #36: batch all Neo4j writes with UNWIND

**Type / risk:** perf / touches `sync/` → queue
**Status:** 🔴 BLOCKED by #14 — NOT executed tonight.

## Why blocked (verified 2026-05-29)
The issue's premise is that `GraphClient` "already exposes batched UNWIND
helpers (`merge_nodes`/`merge_rels`) added during #14" and that
`sync/pg_to_neo4j.py` / `sync/geo.py` "were already converted." **None of that is
true in the current tree:**

- `src/climber_network/graph/client.py` defines only `merge_node` / `merge_rel`
  (singular). There is **no** `merge_nodes` / `merge_rels`.
  (`grep -nE "def merge_(nodes|rels)" src/climber_network/graph/client.py` → empty.)
- `pg_to_neo4j.py` / `geo.py` contain **no** references to the batched helpers.

#14 ("Bring-up") is still open, so the batching layer it was meant to introduce
never landed. "Converting the remaining writers" therefore can't happen until
the helpers + reference conversions exist.

## What must happen first (do NOT let an agent do this unsupervised)
1. Land #14's batching layer: implement `merge_nodes(label, rows)` /
   `merge_rels(rel_type, rows)` in `graph/client.py` — chunked UNWIND in
   retrying managed transactions, `:Entity` id index, **no double-prefixing of
   already-namespaced ids** (the bug that zeroed `HELD_AT`/`REPRESENTS` in #14).
2. Convert `pg_to_neo4j.py` / `geo.py` to use them (the reference pattern #36 copies).

## Then (the actual #36 work)
- Convert `sync/travel.py` (TravelLeg/RestednessState nodes + TRAVELED/TO_EVENT/
  HAD_STATE/AT_EVENT edges; current per-call sites at `travel.py:431-474`).
- Convert `sync/validate_elo.py` (Performance prop SETs; current site `validate_elo.py:310`).
- Verify with existing `tests/test_travel_sync.py` / `tests/test_validate_elo.py`
  (behavior-equivalence: same nodes/edges).

**Recommendation:** keep #36 closed-to-agents until #14's helpers merge.
