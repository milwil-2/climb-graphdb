# Plan — issue #7: research training-state inference from public docs

**Type / risk:** docs (research) / → queue (never auto-merge)
**Status:** ✅ executing tonight → `docs/research/issue-7-extraction-approach.md` (queued PR)

**Acceptance criteria (from the issue):**
- [ ] Comparison of fetch methods for ≥2 source types (recommend the news-first
      path with a concrete API/feed).
- [ ] A working extraction prompt + a few **offline JSON fixtures** (real-ish
      documents) the parser is tested against (no live calls in CI;
      `@pytest.mark.network` for live).
- [ ] Mapping from extracted output → the closed `TrainingSignal.kind` vocab;
      out-of-vocab / low-confidence handling defined.
- [ ] Confirmed provenance + `:Staged` flow (what's stored; how review/promote works).
- [ ] Go/no-go recommendation for the P5 prototype on the ~10 seed athletes.

**Worker constraints:**
- Fetched & cited sources; `**VERIFY:**` on anything uncertain (esp. ToS, API
  pricing/limits). DRAFT for human review.
- Provide the extraction prompt + 1–2 example fixture JSONs **inline in the doc**
  (don't add code/tests to the tree in this research draft — that's #12's job).
