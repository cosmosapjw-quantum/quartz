#!/usr/bin/env bash
# Q7 (audit_codex_20260428.md W'10): one-shot reproduction harness for the
# Rust-side ablation scaffolds in src/ablation_*.rs. Each file contains
# `#[test] #[ignore]` entries that print results; this script runs them
# explicitly so the numbers cited in commit messages and docs/QUARTZ_THEORY.md
# are reproducible without users having to memorize the per-file invocation.
#
# Usage:
#   scripts/run_rust_ablations.sh                # run every Rust ablation
#   scripts/run_rust_ablations.sh ablation_vl    # run a single named module
#
# Output goes to results/rust_ablations/<module>/output.log so artifacts are
# attached to the same results/ tree as the Python ablation harness.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ALL_MODULES=(
    ablation_vl
    ablation_pflip
    ablation_phase1b
    ablation_refresh
    ablation_refresh_v2
    ablation_h3
    calibration
)

if [[ $# -gt 0 ]]; then
    MODULES=("$@")
else
    MODULES=("${ALL_MODULES[@]}")
fi

OUT_ROOT="results/rust_ablations"
mkdir -p "$OUT_ROOT"

# Build once; subsequent test runs reuse the artifact cache.
cargo build --release --bin mcts_demo

for module in "${MODULES[@]}"; do
    module_dir="$OUT_ROOT/$module"
    mkdir -p "$module_dir"
    log_file="$module_dir/output.log"
    echo "[run_rust_ablations] $module -> $log_file"
    # `--nocapture` is required to surface the per-test println! aggregates
    # that these scaffolds emit; `--ignored` activates the gated test names.
    cargo test --release --bin mcts_demo -- "$module" --ignored --nocapture \
        2>&1 | tee "$log_file"
done

echo "[run_rust_ablations] done. Artifacts under $OUT_ROOT/"
