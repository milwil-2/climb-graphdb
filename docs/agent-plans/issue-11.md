# Plan — issue #11: P5a news ingestion (Document/Source acquisition)

**Type / risk:** feat / touches `sync/` (+ `data/`) → queue
**Status:** 🔴 BLOCKED — NOT executed tonight.

## Why blocked
The issue header states: **"Blocked by research: #6 (sources) and #7 (fetch
approach) must land first."** Both are still open. Concretely, #11 needs:
- `data/athlete_handles.yaml` — a hand-picked map of **~10 real athletes** to
  real `news_query`/handles. An unattended agent cannot responsibly invent which
  ~10 World-Cup climbers to seed or their correct handles — that's the human
  curation #6 is meant to produce.
- A **chosen news API/feed** (the output of #7) before any fetch code is sound.

Building this unsupervised would mean fabricating athlete data and committing to
an arbitrary API — exactly the failure mode to avoid overnight.

## Ready when
#6 + #7 are landed (their drafts are queued tonight — see `docs/research/`).
Then: write `athlete_handles.yaml` from #6's recommendation, implement
`sync/ingest/news.py` (typer CLI) MERGEing `Source`/`Document` via the
vocab-gated `GraphClient`, offline JSON fixture tests + one `@pytest.mark.network`
live test, `NEWS_API_KEY` from env only.
