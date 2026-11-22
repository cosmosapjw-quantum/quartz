#!/usr/bin/env bash
set -euo pipefail
STAGE="$1";shift
ROOT="$(git rev-parse --show-toplevel)"
BRANCH="research/quartz-general-c22-inprocess-gate-20261122"
BACKUP="/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta22_backups"
cd "$ROOT"
test "$(git branch --show-current)" = "$BRANCH"
test "$#" -gt 0
git add -- "$@"
git diff --cached --check -- . ':(exclude,glob)**/*.log' ':(exclude,glob)**/*.jsonl'
if git diff --cached --quiet;then echo "NO_STAGE_DELTA $STAGE";exit 3;fi
git commit -m "research(general-c22 $STAGE): active inprocess computation gating" >/dev/null
SHA="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short=12 HEAD)"
mkdir -p "$BACKUP"
OUT="$BACKUP/QUARTZ_GENERAL_C22_$(printf "%s_" "$STAGE")$SHORT.tar.gz"
git archive --format=tar HEAD | gzip -n > "$OUT"
test -s "$OUT"
HASH="$(sha256sum "$OUT"|cut -d' ' -f1)"
git push --quiet -u origin "HEAD:refs/heads/$BRANCH"
REMOTE="$(git ls-remote --heads origin "$BRANCH"|cut -f1)"
test "$SHA" = "$REMOTE"
test "$HASH" = "$(sha256sum "$OUT"|cut -d' ' -f1)"
printf 'STAGE=%s SHA=%s REMOTE=%s ARCHIVE=%s SHA256=%s\n' "$STAGE" "$SHA" "$REMOTE" "$OUT" "$HASH"
