#!/usr/bin/env bash
set -euo pipefail
STAGE="$1"; shift
BRANCH="research/quartz-general-metareasoning-identity-20261122"
BACKUP="/home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta17_backups"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(git branch --show-current)" = "$BRANCH"
test "$#" -gt 0
git add -- "$@"
git diff --cached --check -- . ':(exclude,glob)**/*.log'
if git diff --cached --quiet; then echo "NO_STAGE_DELTA $STAGE";exit 3;fi
git commit -m "research(meta17 $STAGE): restore domain-neutral QUARTZ metareasoning" >/dev/null
HEAD="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short=12 HEAD)"
mkdir -p "$BACKUP"
OUT="$BACKUP/QUARTZ_META17_$(printf "%s_" "$STAGE")$SHORT.tar.gz"
git archive --format=tar HEAD | gzip -n > "$OUT"
test -s "$OUT"
HASH="$(sha256sum "$OUT" | cut -d' ' -f1)"
git push --quiet -u origin "HEAD:refs/heads/$BRANCH"
REMOTE="$(git ls-remote --heads origin "$BRANCH" | cut -f1)"
test "$HEAD" = "$REMOTE"
test "$HASH" = "$(sha256sum "$OUT" | cut -d' ' -f1)"
printf 'STAGE=%s HEAD=%s REMOTE=%s ARCHIVE=%s SHA256=%s\n' "$STAGE" "$HEAD" "$REMOTE" "$OUT" "$HASH"
