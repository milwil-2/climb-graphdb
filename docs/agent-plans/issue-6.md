# Plan — issue #6: research sources for training country & profile signals

**Type / risk:** docs (research) / → queue (never auto-merge)
**Status:** ✅ executing tonight → `docs/research/issue-6-training-country-sources.md` (queued PR)

**Acceptance criteria (from the issue):**
- [ ] Written comparison of ≥3 candidate sources for training country/base
      (coverage, freshness, reliability, access/ToS).
- [ ] Recommended primary source + fallback for a real `BASED_IN` (not the
      nationality proxy), with confidence expectations.
- [ ] Documented athlete→handle matching strategy + the initial
      `data/athlete_handles.yaml` schema (~10 hand-picked athletes — schema +
      method, with athlete selection left for human curation).
- [ ] Legal/ToS notes (proposed text for `SECURITY.md`/`docs/`, public-data &
      responsible-use per PRD §8.4). **Do not edit `SECURITY.md` directly** —
      propose the wording in the doc for human review.
- [ ] Findings recorded so #11 (`athlete_handles.yaml`) and downstream can build on them.

**Worker constraints:**
- Real, **fetched & cited** sources only; flag anything uncertain `**VERIFY:**`.
- This is a DRAFT for human review; recommendations are non-binding until approved.
