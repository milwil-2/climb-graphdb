# Plan — issue #12: P5b LLM extraction (Documents → :Staged typed facts)

**Type / risk:** feat / touches `sync/` → queue
**Status:** 🔴 BLOCKED — NOT executed tonight.

## Why blocked
Header: **"Blocked by #11 (need `Document` nodes) and research #7 (extraction
approach)."** #11 is itself blocked (see `issue-11.md`), so #12 is two levels
deep. There are no `Document` nodes to extract from and no validated extraction
prompt yet.

## Ready when
#11 has landed `Document` nodes and #7 has validated the extraction prompt +
offline fixtures. Then: `sync/ingest/extract.py` (lazy `from groq import Groq`),
extract into the **closed** `TrainingSignal.kind` vocab
(`gym_session, volume_high, volume_taper, focus_lead, focus_boulder, focus_speed,
illness, travel_personal, return_from_injury, camp_attendance`), write nodes
under a `:Staged` label with `confidence` + the three provenance edges
(`EVIDENCED_BY`/`FROM_SOURCE`/`EXTRACTED_BY`). Invariant test: no fact without
`EVIDENCED_BY`; out-of-vocab `kind` rejected; low-confidence dropped.

**Note:** `:Staged` and `ExtractionRun` provenance are not yet in
`vocab.VALID_NODE_LABELS` as a quarantine mechanism — confirm the labelling
approach during #12 design (`:Staged` is a secondary label, not a new node type).
