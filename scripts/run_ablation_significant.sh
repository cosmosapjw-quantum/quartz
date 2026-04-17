#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
RUST_BINARY="${RUST_BINARY:-$ROOT_DIR/target/release/mcts_demo}"

ITERATIONS="${ITERATIONS:-30}"
EVAL_GAMES="${EVAL_GAMES:-160}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
SEEDS="${SEEDS:-41,42,43,44,45}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-96}"

BACKEND="${BACKEND:-torch}"

SEARCH_OUTPUT="${SEARCH_OUTPUT:-results/ablation_search_vl_sig}"
CONTROLLER_OUTPUT="${CONTROLLER_OUTPUT:-results/ablation_controller_factorial_sig}"

# ROCm workarounds
export MIOPEN_DEBUG_CONV_GEMM=0
if [ "$BACKEND" = "jax" ]; then
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
fi

# torch.compile kernel cache: persist across subprocesses for faster startup
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

GAMES=()
STUDIES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ablation_significant.sh [all|gomoku15|renju|gomoku15_renju] [search_vl|controller_factorial]...

Examples:
  scripts/run_ablation_significant.sh
  scripts/run_ablation_significant.sh gomoku15
  scripts/run_ablation_significant.sh renju controller_factorial

Environment overrides:
  PYTHON_BIN
  RUST_BINARY
  BACKEND          (torch|jax, default: torch)
  ITERATIONS
  EVAL_GAMES
  EVAL_INTERVAL
  SEEDS
  TIMEOUT_HOURS
  SEARCH_OUTPUT
  CONTROLLER_OUTPUT
EOF
}

normalize_game_arg() {
  case "$1" in
    all) echo "all" ;;
    gomoku15) echo "gomoku15" ;;
    renju|gomoku15_renju) echo "gomoku15_renju" ;;
    *)
      echo "Unknown game target: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

normalize_study_arg() {
  case "$1" in
    search_vl|controller_factorial) echo "$1" ;;
    *)
      echo "Unknown study target: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    all|gomoku15|renju|gomoku15_renju)
      GAMES+=("$(normalize_game_arg "$arg")")
      ;;
    search_vl|controller_factorial)
      STUDIES+=("$(normalize_study_arg "$arg")")
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ ${#GAMES[@]} -eq 0 ]; then
  GAMES=("gomoku15" "gomoku15_renju")
fi

if [ ${#STUDIES[@]} -eq 0 ]; then
  STUDIES=("search_vl" "controller_factorial")
fi

run_study() {
  local game="$1"
  local study="$2"
  local output_root

  case "$study" in
    search_vl) output_root="$SEARCH_OUTPUT" ;;
    controller_factorial) output_root="$CONTROLLER_OUTPUT" ;;
    *)
      echo "Unsupported study: $study" >&2
      exit 1
      ;;
  esac

  echo
  echo "==> Running study=$study game=$game"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/ablation_study.py" \
    --study "$study" \
    --game "$game" \
    --iterations "$ITERATIONS" \
    --eval-games "$EVAL_GAMES" \
    --eval-interval "$EVAL_INTERVAL" \
    --seeds "$SEEDS" \
    --paired-seed-eval \
    --include-strict-reference \
    --backend "$BACKEND" \
    --resident-session \
    --timeout-hours "$TIMEOUT_HOURS" \
    --output "$output_root"

  echo
  echo "==> Reporting study=$study game=$game"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/ablation_study.py" \
    --report "$ROOT_DIR/$output_root/$game"
}

for game in "${GAMES[@]}"; do
  if [ "$game" = "all" ]; then
    for expanded_game in gomoku15 gomoku15_renju; do
      for study in "${STUDIES[@]}"; do
        run_study "$expanded_game" "$study"
      done
    done
    continue
  fi

  for study in "${STUDIES[@]}"; do
    run_study "$game" "$study"
  done
done
