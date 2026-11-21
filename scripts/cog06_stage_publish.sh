#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:?R01..R10 or C01..C10}"; shift
ROOT="$(git rev-parse --show-toplevel)"
BRANCH="research/quartz-cog06-nonterminal-fork-wdl-20261121"
ARCHIVES="/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups"
cd "$ROOT"
test "$(git branch --show-current)" = "$BRANCH"
test "$#" -gt 0
git add -- "$@"
if git diff --cached --quiet; then echo "NO_STAGE_DELTA $STAGE";exit 3; fi
# Preserve authentic raw test log bytes; text whitespace hygiene only applies to source/document changes.
git diff --cached --check -- . ':(exclude,glob)**/*.log'
git commit -m "research(cog06 $STAGE): nonterminal fork and matched-CPU WDL" >/dev/null
SHA="$(git rev-parse HEAD)"
mkdir -p "$ARCHIVES"
ARCHIVE="$ARCHIVES/QUARTZ_COG06_NONTERMINAL16_${STAGE}_${SHA:0:12}.tar.gz"
git archive --format=tar HEAD | gzip -n > "$ARCHIVE"
test -s "$ARCHIVE"
DIGEST="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
git push --quiet -u origin "HEAD:refs/heads/$BRANCH"
REMOTE="$(git ls-remote --heads origin "$BRANCH" | cut -f1)"
test "$SHA" = "$REMOTE"
test "$DIGEST" = "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
printf 'STAGE=%s COMMIT=%s REMOTE=%s ARCHIVE=%s HASH=%s\n' "$STAGE" "$SHA" "$REMOTE" "$ARCHIVE" "$DIGEST"
