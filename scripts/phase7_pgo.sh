#!/usr/bin/env bash
# Phase 7 I (2026-04-26): Profile-Guided Optimization build.
#
# Two-step PGO workflow on the controller-fixed-iterations fast-path
# bench. Produces a release-grade PGO-optimized test binary at
#   target/x86_64-unknown-linux-gnu/release/deps/mcts_demo-<hash>
#
# CI continues to ship the non-PGO binary; the PGO binary is for
# release tagging and the Phase 7 final perf delta doc only.
#
# Prereqs (run once):
#   cargo install cargo-pgo
#   rustup component add llvm-tools-preview
#
# The `llvm-tools-preview` rustup component ships an `llvm-profdata`
# whose raw-profile-format version matches the rustc that produced
# the .profraw files. The system /usr/bin/llvm-profdata is typically
# one major version behind rustc and rejects the profiles ("raw
# profile version mismatch: Profile uses raw profile format version
# = 10; expected version = 9"); we work around by prepending the
# rustup-shipped binary path to PATH.

set -euo pipefail

cd "$(dirname "$0")/.."

# Use the matching llvm-profdata from the rustup llvm-tools-preview
# component to avoid the format-version mismatch with /usr/bin.
RUST_LLVM="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin"
if [[ ! -x "$RUST_LLVM/llvm-profdata" ]]; then
  echo "ERROR: $RUST_LLVM/llvm-profdata not found. Run:"
  echo "       rustup component add llvm-tools-preview"
  exit 1
fi
export PATH="$RUST_LLVM:$PATH"

BENCH_FILTER="mcts::tests::bench_search_controller_fixed_iterations_fast_path"
PROFILE_DIR="target/pgo-profiles"

echo "=== Phase 7 I: PGO build workflow ==="

# Step 1: clean prior profiles + run the bench under instrumentation.
rm -rf "$PROFILE_DIR"
echo "[1/3] Building instrumented test binary + running bench..."
cargo pgo test -- --locked "$BENCH_FILTER" -- --exact --ignored --nocapture

# Step 2: collect more profile by running the instrumented binary
# directly 5 more times. (cargo pgo test runs the bench once.)
INSTR_TB=$(ls -t target/x86_64-unknown-linux-gnu/release/deps/mcts_demo-*[0-9a-f] | grep -v '\.d$' | head -1)
echo "[2/3] Collecting additional profile (bench × 5 on $INSTR_TB)..."
for i in 1 2 3 4 5; do
  echo "  bench run $i/5"
  "$INSTR_TB" --exact --ignored --nocapture "$BENCH_FILTER" > /dev/null 2>&1
done

PROFRAW_BYTES=$(du -sh "$PROFILE_DIR" 2>/dev/null | cut -f1 || echo "?")
echo "  profile dir size: $PROFRAW_BYTES"

# Step 3: optimize.
echo "[3/3] Building PGO-optimized binary..."
cargo pgo optimize test -- --locked --no-run

PGO_TB=$(ls -t target/x86_64-unknown-linux-gnu/release/deps/mcts_demo-*[0-9a-f] | grep -v '\.d$' | head -1)
echo
echo "=== Done. PGO test binary: $PGO_TB ==="
ls -l "$PGO_TB"

# Quick sanity check.
echo
echo "=== Sanity: full suite on PGO binary ==="
"$PGO_TB" 2>&1 | tail -3
