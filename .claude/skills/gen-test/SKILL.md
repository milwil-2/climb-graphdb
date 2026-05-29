---
name: gen-test
description: Generate a test file for a climb-graphdb module, matching the conventions in tests/conftest.py and existing test files. Use when the user asks to "generate tests", "add tests for", "create test file", or "scaffold tests".
user-invocable: true
allowed-tools: [Bash, Read, Write, Edit]
---

# Generate Tests (climb-graphdb)

Scaffold a pytest test file for a module in THIS repo, matching the existing
conventions. Do NOT invent fixtures or import paths — read the real ones first.

## Steps

1. **Identify the target module** from the user's request and find it under
   `src/climber_network/`, `api/`, or `sync/`.

2. **Read the conventions before writing anything:**
   - `tests/conftest.py` — the shared fixtures (authoritative source of truth):
     - `source_engine` / `source_session` — fresh per-test in-memory SQLite
       built from `climber_network.source.pg` (`make_engine("sqlite://")` →
       `pg.Base.metadata.create_all` → engine disposed on teardown).
     - `sample_athlete` — one athlete + minimal event/round/result.
     - `seeded_session` — a representative competition (4 athletes; qualification
       + semi + final rounds across two events; one DNS row; two ratings).
     - `FakeGraphClient` — records `merge_node` / `merge_rel`, gates labels/rels
       through `vocab.assert_label` / `assert_rel`; optional canned `run_read`.
     - `FakeNeo4jDriver` — `.session().run(...).single()["c"]` shape for `api.db`.
   - One or two exemplar tests that match the module's layer:
     - pure logic → `tests/test_travel_formulas.py`, `tests/test_geocode.py`,
       `tests/test_vocab.py`, `tests/test_expected.py`
     - sync / graph writer → `tests/test_l1_mirror.py`
     - FastAPI routes → `tests/test_api.py`
     - live external service → `tests/test_network.py`

3. **Pick fixtures by layer:**
   - `src/climber_network/source` or `sync/*` → `seeded_session` /
     `source_session` + `FakeGraphClient`.
   - `api/*` route → module-scoped `client` fixture that sets
     `api.db._driver = FakeNeo4jDriver(nodes=..., relationships=...)`, wraps
     `api.index.app` in `fastapi.testclient.TestClient`, and **restores the
     original `_driver` on teardown**.
   - pure `src/climber_network/{travel,geo,elo,vocab,config}` → no DB; build
     small in-memory inputs directly.

4. **Match these conventions exactly:**
   - File at `tests/test_{module_name}.py`; start with
     `from __future__ import annotations`.
   - Import production code from `climber_network.*`, `api.*`, or `sync.*`.
     NEVER import `climbing_elo` or `knowledge_graph` (hard isolation rule).
   - Reuse shared fixtures via `from tests.conftest import FakeGraphClient,
     FakeNeo4jDriver` — do NOT redefine them.
   - Typed test signatures (`-> None`); `pytest.approx` for floats.
   - Mark any live-service test `@pytest.mark.network` and check the expected
     payload into `tests/fixtures/` so the parsing is also covered offline.

5. **Assert the invariants that matter for the layer:**
   - sync → **idempotency** (re-run = same logical node/edge sets, no duplicate
     MERGE per id), correct namespaced ids from `vocab` builders, documented
     filters (DNS skipped), and count validation pass/fail.
   - any graph write → **closed-vocab guard**: out-of-vocab label/rel raises
     `ValueError`.
   - graph client cypher → **bound parameters / no string interpolation** of
     untrusted values; labels/rels only via `assert_label` / `assert_rel`.
   - formulas → **clamps and monotonicity** at boundaries (0, caps, negatives).
   - config → env-getter defaults with save/restore of `os.environ`.

6. **Run the new tests:**
   ```bash
   uv run pytest tests/test_{module_name}.py -v
   ```
   For a network test: `uv run pytest tests/test_{module_name}.py -m network -v`.
