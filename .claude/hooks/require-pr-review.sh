#!/usr/bin/env bash
# PreToolUse gate (shipped in committed .claude/settings.json so EVERY Claude
# session, in any clone, is bound by it): block `gh pr merge` unless the PR
# carries the `claude-reviewed` label. That label is added only AFTER a
# comprehensive review (/security-review + /code-review) finds no blocking
# issues — see CONTRIBUTING.md. This makes skipping the review a deliberate act,
# not an accident: a fresh session that tries to merge is stopped and told what
# to do first.
set -euo pipefail

input=$(cat)
# Extract the Bash command from the PreToolUse payload (no eval of it).
cmd=$(printf '%s' "$input" | python3 -c \
  "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" \
  2>/dev/null || true)

# Only police actual `gh pr merge` invocations — match at a command position
# (start, or right after a shell separator), NOT the substring inside quoted
# args like a PR body that happens to contain the text "gh pr merge".
if ! printf '%s' "$cmd" | grep -qE '(^|[;&|(]|&&|\|\|)[[:space:]]*gh[[:space:]]+pr[[:space:]]+merge([[:space:]]|$)'; then
  exit 0
fi

# Resolve EVERY PR this command targets — handles compound lines like
# `gh pr merge 5 && gh pr merge 9` (don't let a labeled PR smuggle an
# unlabeled one through).
prs=$(printf '%s' "$cmd" | grep -oE "gh pr merge[[:space:]]+[0-9]+" | grep -oE "[0-9]+$" || true)
if [ -z "${prs:-}" ]; then
  # No explicit number (e.g. current-branch `gh pr merge --auto`) → resolve via gh.
  prs=$(gh pr view --json number --jq .number 2>/dev/null || true)
fi
if [ -z "${prs:-}" ]; then
  echo "BLOCKED: cannot determine the PR for this merge. Use 'gh pr merge <N>' after a Claude review." >&2
  exit 2
fi

# Block unless EVERY targeted PR carries the review label.
for pr in $prs; do
  if gh pr view "$pr" --json labels --jq '.labels[].name' 2>/dev/null | grep -qx "claude-reviewed"; then
    continue
  fi
  cat >&2 <<EOF
BLOCKED: PR #$pr is not review-verified, so it cannot be merged.

Run a comprehensive review in this session first:
  1) /security-review        (vulnerabilities)
  2) /code-review            (correctness / won't-break-the-codebase)

Confirm there are NO high/critical findings, then mark it reviewed:
  gh pr edit $pr --add-label claude-reviewed

…and re-run the merge. (Enforced by .claude/hooks/require-pr-review.sh — see CONTRIBUTING.md.)
EOF
  exit 2
done
exit 0
