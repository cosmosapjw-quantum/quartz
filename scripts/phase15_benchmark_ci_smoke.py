#!/usr/bin/env python3
"""Self-contained CI smoke for the phase15 continuation benchmark gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import controller_sweep as sweep

from quartz.models_torch import AlphaZeroNet
from quartz.phase15_ablation import phase15_systems_csv, resolve_phase15_systems_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic smoke inputs and run the phase15 benchmark gate"
    )
    parser.add_argument("--game", default="gomoku7")
    parser.add_argument("--output", default="results/phase15_ci_gate")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--systems", default="A4,B1,B2")
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Search threads for the deterministic CI comparison",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Unmeasured rounds used to remove process/model cold-start cost from the speed gate",
    )
    parser.add_argument("--search-stall-timeout-s", type=float, default=180.0)
    return parser.parse_args()


def deterministic_positions(board_size: int) -> list[dict[str, object]]:
    n = int(board_size) * int(board_size)
    empty = [0] * n
    row_open = [0] * n
    row_open[3 * board_size + 3] = 1
    row_open[3 * board_size + 4] = -1
    row_open[4 * board_size + 3] = 1
    row_open[2 * board_size + 2] = -1
    immediate_win = [0] * n
    immediate_win[2 * board_size + 1] = 1
    immediate_win[2 * board_size + 2] = 1
    immediate_win[2 * board_size + 3] = 1
    immediate_win[4 * board_size + 5] = -1
    forced_block = [0] * n
    forced_block[1 * board_size + 1] = -1
    forced_block[1 * board_size + 2] = -1
    forced_block[1 * board_size + 3] = -1
    forced_block[4 * board_size + 4] = 1
    return [
        {"id": "P0001", "board": empty, "player": 1},
        {"id": "P0002", "board": row_open, "player": 1},
        {"id": "P0003", "board": immediate_win, "player": 1},
        {"id": "P0004", "board": forced_block, "player": 1},
    ]


def write_ci_smoke_inputs(base_dir: Path, *, game: str, seed: int) -> tuple[Path, Path]:
    import torch

    base_cfg, _device = sweep.build_base_cfg(game, "cpu")
    torch.manual_seed(int(seed))
    model = AlphaZeroNet(base_cfg)
    checkpoint_path = base_dir / "fixtures" / f"{game}_seed{int(seed)}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    positions_path = base_dir / "fixtures" / f"{game}_positions.json"
    positions_payload = {"positions": deterministic_positions(int(base_cfg["board"]))}
    positions_path.write_text(json.dumps(positions_payload, indent=2), encoding="utf-8")
    return checkpoint_path, positions_path


def build_benchmark_command(
    args: argparse.Namespace, checkpoint_path: Path, positions_path: Path
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "phase15_benchmark.py"),
        "--game",
        args.game,
        "--output",
        args.output,
        "--checkpoints",
        str(checkpoint_path),
        "--positions-file",
        str(positions_path),
        "--max-positions",
        "4",
        "--systems",
        phase15_systems_csv(args.systems),
        "--budgets",
        str(args.budgets),
        "--seed",
        str(int(args.seed)),
        "--n-threads",
        str(int(args.n_threads)),
        "--repeats",
        "1",
        "--warmup-rounds",
        str(int(args.warmup_rounds)),
        "--search-stall-timeout-s",
        str(float(args.search_stall_timeout_s)),
        "--rust-binary",
        str(args.rust_binary),
    ]
    if getattr(args, "enforce_gate", True):
        command.append("--enforce-gate")
    return command


def build_ci_smoke_contract_summary(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    positions_path: Path,
    benchmark_payload: dict[str, object],
) -> dict[str, object]:
    benchmark_contract = benchmark_payload.get("contract_summary")
    return {
        "runner": {
            "game": str(args.game),
            "systems": list(resolve_phase15_systems_arg(args.systems)),
            "budgets": str(args.budgets),
            "seed": int(args.seed),
            "n_threads": int(args.n_threads),
            "warmup_rounds": int(args.warmup_rounds),
            "search_stall_timeout_s": float(args.search_stall_timeout_s),
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "positions_path": str(positions_path),
            "summary_path": str(
                Path(args.output)
                / args.game
                / "phase15_continuation_benchmark_summary.json"
            ),
        },
        "benchmark_contract_summary": benchmark_contract,
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, positions_path = write_ci_smoke_inputs(
        base_dir, game=args.game, seed=int(args.seed)
    )
    command = build_benchmark_command(args, checkpoint_path, positions_path)
    subprocess.run(command, check=True)
    summary_path = base_dir / "phase15_continuation_benchmark_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "checkpoint_path": str(checkpoint_path),
        "positions_path": str(positions_path),
        "summary_path": str(summary_path),
        "gate": payload.get("gate", {}),
        "contract_summary": build_ci_smoke_contract_summary(
            args,
            checkpoint_path=checkpoint_path,
            positions_path=positions_path,
            benchmark_payload=payload,
        ),
    }
    (base_dir / "phase15_ci_smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
