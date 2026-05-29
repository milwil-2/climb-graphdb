# Plan — issue #35: gitignore .claude/worktrees/

**Type / risk:** chore / low → `auto-merge-ok`
**Status:** ready ✅ (executed tonight, auto-merged)

**Acceptance criteria:**
- [x] `/.claude/worktrees/` added to `.gitignore`, anchored to repo root.
- [x] `git status` no longer shows `?? .claude/worktrees/` (unblocks `/ship` clean-tree precondition).

**Files to touch:**
- `.gitignore` — add the anchored ignore line near the `.claude/settings.local.json` convention.

**Test plan:**
- No code → no unit test. Verify by creating a dummy `.claude/worktrees/x` and confirming `git status --porcelain` is clean.

**Out of scope:** ignoring anything else under `.claude/` (settings.json stays committed).

**Notes:** This is the auto-merge warm-up that also validates the review→label→merge path end-to-end.
