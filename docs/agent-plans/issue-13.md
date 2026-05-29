# Plan — issue #13: P5c staged review + gated /ingest endpoints + promote

**Type / risk:** feat / touches `api/` + `sync/` → queue
**Status:** 🔴 BLOCKED — NOT executed tonight.

## Why blocked
Header: **"Blocked by #12 (need `:Staged` facts to promote)."** #12 is blocked
(see `issue-12.md`), so #13 is three levels deep in the P5 chain. There is
nothing staged to promote and no ingest pipeline to gate.

## Ready when
#12 produces `:Staged` facts. Then: `sync/ingest/promote.py` (strip `:Staged`
for approved ids, preserve provenance edges); wire `POST /ingest` +
`POST /ingest/approve` into `api/index.py`, both gated by `INGEST_API_KEY`
(bearer header) and returning 404/503 when the key is unset (cloud-disabled).
Tests via the conftest `client` fixture (key set vs unset). Imputed health/injury
stays private-by-default (excluded from public read endpoints).
