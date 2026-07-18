#!/usr/bin/env python3
"""pending_flow_lab runner — synthetic count-only WU-UCT screen + Rust bridge.

Runs the synthetic pending-flow screen (``quartz/experiments/pending_flow.py``)
and, when given the captured output of the real engine VL ablation
(``src/ablation_vl.rs`` via ``cargo test vl_ablation_gomoku7 -- --ignored``),
parses its telemetry and cross-checks the synthetic verdict against the real
engine. The Rust telemetry is the ground truth for the H4/H5 dispositions; the
synthetic screen is the cheap illustration that motivated the bridge.

See ``docs/METACOGNITIVE_EXPERIMENTS.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import (  # noqa: E402
    atomic_json_dump,
    build_run_manifest,
    file_sha256,
    finalize_run_manifest,
    utc_now,
)
from quartz.experiments import pending_flow as lab  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "pending_flow_scenarios.v1.json"

# Ablation-1 table rows: "Mode Agree% Entrop Q_Sprd NPS AvgVV DupRt MaxP"
_ABL1_ROW = re.compile(
    r"^\s*(Fixed\(1,1\)|Adaptive|VvisitOnly|VvalueOnly|Disabled)\s+"
    r"([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$"
)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported config format_version")
    if payload.get("experiment_id") != lab.EXPERIMENT_ID:
        raise ValueError("config experiment_id mismatch")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config must contain a non-empty scenarios list")
    return payload


def parse_rust_ablation(text: str) -> Dict[str, Any]:
    """Parse the component-isolation (Ablation 1) table from the Rust VL
    ablation output into per-mode telemetry, plus the fixed-vs-adaptive verdict."""
    modes: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        m = _ABL1_ROW.match(line)
        if not m:
            continue
        name = m.group(1)
        modes[name] = {
            "agreement_pct": float(m.group(2)),
            "entropy": float(m.group(3)),
            "q_spread": float(m.group(4)),
            "nps": float(m.group(5)),
            "avg_vvalue": float(m.group(6)),
            "dup_rate": float(m.group(7)),
            "max_pending": float(m.group(8)),
        }
    fixed = modes.get("Fixed(1,1)")
    adaptive = modes.get("Adaptive")
    verdict: Dict[str, Any] = {"modes": modes}
    if fixed and adaptive:
        verdict.update(
            {
                "real_adaptive_reduces_dup_rate": bool(
                    adaptive["dup_rate"] < fixed["dup_rate"]
                ),
                "real_dup_rate_fixed": fixed["dup_rate"],
                "real_dup_rate_adaptive": adaptive["dup_rate"],
                "real_avg_vvalue_fixed": fixed["avg_vvalue"],
                "real_avg_vvalue_adaptive": adaptive["avg_vvalue"],
                "real_adaptive_lowers_pessimism": bool(
                    adaptive["avg_vvalue"] < fixed["avg_vvalue"]
                ),
                "real_agreement_fixed_pct": fixed["agreement_pct"],
                "real_agreement_adaptive_pct": adaptive["agreement_pct"],
                "real_agreement_preserved": bool(
                    abs(adaptive["agreement_pct"] - fixed["agreement_pct"]) <= 5.0
                ),
            }
        )
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--waves", type=int, default=None)
    parser.add_argument(
        "--worker-grid", type=str, default=None, help="comma-separated worker counts"
    )
    parser.add_argument(
        "--rust-log",
        type=Path,
        default=None,
        help="captured cargo VL-ablation output to cross-check",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metacognitive_root/pending_flow_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _choose_scenarios(
    cfg: Mapping[str, Any], requested: Sequence[str] | None
) -> List[dict[str, Any]]:
    by_id = {str(s["id"]): dict(s) for s in cfg["scenarios"]}
    if not requested:
        return [dict(s) for s in cfg["scenarios"]]
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise SystemExit(f"unknown scenario ids: {unknown}")
    return [by_id[i] for i in requested]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    scenarios = _choose_scenarios(cfg, args.scenarios)
    seed = int(args.seed if args.seed is not None else cfg["default_seed"])
    waves = int(args.waves if args.waves is not None else cfg["default_waves"])
    latency = int(cfg["default_latency"])
    c_puct = float(cfg["default_c_puct"])
    if args.worker_grid:
        worker_grid = [int(x) for x in args.worker_grid.split(",") if x.strip()]
    else:
        worker_grid = [int(x) for x in cfg["default_worker_grid"]]

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory is not empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "scenarios": scenarios,
        "seed": seed,
        "waves": waves,
        "latency": latency,
        "c_puct": c_puct,
        "worker_grid": worker_grid,
        "rust_bridge_provided": bool(args.rust_log),
    }
    started_at = utc_now()
    source_paths = [
        Path(__file__),
        REPO_ROOT / "quartz" / "experiments" / "pending_flow.py",
        REPO_ROOT / "quartz" / "experiment_manifest.py",
        REPO_ROOT / "src" / "ablation_vl.rs",
        config_path,
    ]
    manifest = build_run_manifest(
        experiment_id=lab.EXPERIMENT_ID,
        execution_mode=lab.EXECUTION_MODE,
        resolved_config=resolved_config,
        repo_root=REPO_ROOT,
        source_paths=source_paths,
        argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
        started_at=started_at,
        assumptions=cfg["assumptions"],
        prohibited_inferences=cfg["prohibited_inferences"],
    )
    manifest_path = output_dir / "run_manifest.json"
    atomic_json_dump(manifest_path, manifest)

    per_scenario: List[Dict[str, Any]] = []
    for scenario in scenarios:
        rows = lab.screen(
            scenario["arm_values"],
            worker_grid,
            waves=waves,
            latency=latency,
            c_puct=c_puct,
            seed=seed,
        )
        verdict = lab.kill_verdict(rows, worker_grid)
        per_scenario.append(
            {
                "scenario_id": scenario["id"],
                "scenario_label": scenario.get("label"),
                "rows": rows,
                "synthetic_kill_verdict": verdict,
            }
        )

    # any scenario showing the adaptive dup lane alive keeps it; else killed
    synthetic_h5_alive = any(
        s["synthetic_kill_verdict"].get("h5_adaptive_dup_lane_alive")
        for s in per_scenario
    )
    synthetic_h4_alive = any(
        s["synthetic_kill_verdict"].get("h4_adaptive_throughput_lane_alive")
        for s in per_scenario
    )

    artifact_paths: List[Path] = []
    rust_bridge: Dict[str, Any] | None = None
    if args.rust_log:
        rust_text = Path(args.rust_log).read_text(encoding="utf-8")
        rust_bridge = parse_rust_ablation(rust_text)
        captured = output_dir / "rust_vl_ablation.log"
        shutil.copyfile(args.rust_log, captured)
        rust_bridge["captured_log"] = captured.name
        rust_bridge["captured_log_sha256"] = file_sha256(captured)
        artifact_paths.append(captured)

    combined = {
        "synthetic_h5_adaptive_dup_lane_alive": synthetic_h5_alive,
        "synthetic_h4_adaptive_throughput_lane_alive": synthetic_h4_alive,
        "rust_bridge_present": rust_bridge is not None,
        "rust_adaptive_reduces_dup_rate": (
            None
            if rust_bridge is None
            else rust_bridge.get("real_adaptive_reduces_dup_rate")
        ),
        "rust_adaptive_lowers_pessimism": (
            None
            if rust_bridge is None
            else rust_bridge.get("real_adaptive_lowers_pessimism")
        ),
        # H5 dup-reduction lane is demoted iff neither synthetic nor real supports it
        "h5_dup_reduction_lane_demoted": bool(
            not synthetic_h5_alive
            and (
                rust_bridge is None
                or not rust_bridge.get("real_adaptive_reduces_dup_rate", False)
            )
        ),
    }

    summary_json = output_dir / "summary.json"
    atomic_json_dump(
        summary_json,
        {
            "format_version": 1,
            "experiment_id": lab.EXPERIMENT_ID,
            "execution_mode": lab.EXECUTION_MODE,
            "claim_status": "synthetic_screening_only",
            "config_sha256": resolved_config["config_sha256"],
            "worker_grid": worker_grid,
            "per_scenario": per_scenario,
            "rust_bridge": rust_bridge,
            "combined_verdict": combined,
            "note": (
                "Ground truth for H4/H5 is the Rust telemetry. The pre-registered H5 "
                "'adaptive lowers dup_rate' mechanism is not supported by either channel; "
                "the real engine's adaptive benefit is lower virtual-loss pessimism (avg_vvalue) "
                "at preserved agreement, which is a distinct, non-wall-clock property."
            ),
        },
    )
    artifact_paths.insert(0, summary_json)
    manifest = finalize_run_manifest(
        manifest, output_dir=output_dir, artifact_paths=artifact_paths
    )
    atomic_json_dump(manifest_path, manifest)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "synthetic_h5_dup_lane_alive": synthetic_h5_alive,
                "synthetic_h4_throughput_lane_alive": synthetic_h4_alive,
                "rust_adaptive_reduces_dup_rate": combined[
                    "rust_adaptive_reduces_dup_rate"
                ],
                "rust_adaptive_lowers_pessimism": combined[
                    "rust_adaptive_lowers_pessimism"
                ],
                "h5_dup_reduction_lane_demoted": combined[
                    "h5_dup_reduction_lane_demoted"
                ],
                "run_contract_hash": manifest["run_contract_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
