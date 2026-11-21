#!/usr/bin/env bash
# COG05: for each completed bounded stage, preserve content and verify remote identity.
set -euo pipefail
STAGE="${1:?R01..R10 or C01..C10}"; shift
ROOT="$(git rev-parse --show-toplevel)"
BRANCH="research/quartz-cog05-trained-prior-fair-cpu-20261121"
ARCHIVES="/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups"
cd "$ROOT"
test "$(git branch --show-current)" = "$BRANCH"
test "$#" -gt 0
git add -- "$@"
if git diff --cached --quiet; then
  echo "NO_COMMIT: stage $STAGE has no new evidence"; exit 3
fi
git commit -m "research(cog05 $STAGE): trained-prior fair-cost research" >/dev/null
SHA="$(git rev-parse HEAD)"
mkdir -p "$ARCHIVES"
ARCHIVE="$ARCHIVES/QUARTZ_COG05_FAIRCOST15_${STAGE}_${SHA:0:12}.tar.gz"
git archive --format=tar HEAD | gzip -n > "$ARCHIVE"
test -s "$ARCHIVE"
DIGEST="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
# Push exact source tree; verify the provider's ref in a separate remote readback.
git push --quiet -u origin "HEAD:refs/heads/$BRANCH"
REMOTE="$(git ls-remote --heads origin "$BRANCH" | cut -f1)"
test "$SHA" = "$REMOTE"
test "$DIGEST" = "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
printf 'STAGE=%s COMMIT=%s REMOTE=%s BACKUP=%s SHA256=%s\n' "$STAGE" "$SHA" "$REMOTE" "$ARCHIVE" "$DIGEST"
