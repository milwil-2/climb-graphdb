---
description: Ship the current branch end-to-end — push, open PR, run security + code review, and (if clean) label, auto-merge, and clean up. Optional arg is an existing PR number.
argument-hint: "[pr-number]"
---

You are running the **/ship** workflow: take the current feature branch from "committed" all the way to "merged and cleaned up", autonomously, **without pausing for confirmation between steps**. Only stop if a precondition fails or a review finds a blocking issue.

`$ARGUMENTS` — optional PR number to operate on. If empty, operate on the current branch's PR.

## Hard rules
- **Never bypass the review gate.** The merge only proceeds after `/security-review` and `/code-review` run in this session and come back with no High/Critical (security) or blocking (correctness) findings. If `.claude/hooks/require-pr-review.sh` blocks the merge, the label/review step didn't complete — fix it, do not work around it.
- **Never push or merge to `main` directly. Never commit on the user's behalf** — if the working tree is dirty, stop and ask them to commit.

## Steps
1. **Preconditions** (abort with a clear message if any fail):
   - Current branch ≠ `main` (`git rev-parse --abbrev-ref HEAD`).
   - Working tree clean (`git status --porcelain` empty). If dirty → STOP and tell the user to commit first.
   - Branch is ahead of main (`git rev-list --count main..HEAD` > 0).
2. **Push** the current branch: `git push -u origin "$(git rev-parse --abbrev-ref HEAD)"`.
3. **PR**: if `$ARGUMENTS` is set, use that PR number. Otherwise find the branch's open PR (`gh pr view --json number,url`); if none exists, create one with `gh pr create --base main --fill` (if the branch name contains an issue number like `fix/12-...`, put `Closes #12` in the body). Capture the PR number as N.
4. **Review** (mandatory): run `/security-review`, then `/code-review`. Read both reports. **If either has a High/Critical security finding or a blocking correctness bug → STOP**: summarize the findings, leave the PR open, do not label or merge.
5. **Label + merge** (clean reviews only) — run as **two separate** Bash calls (the hook inspects label state before the merge runs, so a combined command is blocked):
   - `gh pr edit N --add-label claude-reviewed`
   - `gh pr merge N --auto --squash`
6. **Cleanup**: check `gh pr view N --json state --jq .state`.
   - If `MERGED`: `git switch main && git pull --prune`; then `git branch -D <branch>` (squash-merge marks the branch "not fully merged", so `-D` is expected); then `git worktree prune`.
   - If not yet merged (CI still running): report that auto-merge is enabled and cleanup will run once it lands — don't block waiting.
7. **Report**: PR URL, merge state, whether the branch was cleaned up, and current `main` HEAD (`git log --oneline -1`).
