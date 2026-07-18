#!/usr/bin/env python3
"""candidate_morphology_lab runner — priced widening + omission/ranking regret.

Runs the preregistered synthetic candidate-morphology assay (see
``docs/METACOGNITIVE_EXPERIMENTS.md`` and
``quartz/experiments/candidate_morphology.py``) and, as a Stage-3 companion,
the H1 argmax-stability discrimination gate on a synthetic ground-truth bank
(``quartz/experiments/h1_synthetic_gate.py``).

Two preregistered kill checks are reported:

* ``widening_lane_demoted``: True iff NO widen price in NO scenario gives a
  CI-separated reduction in paired omission regret vs ``no_widen``.
* H1 ``gate_pass``: False iff the argmax-stability signal is degenerate on
  synthetic ground truth (kills H1 online wiring before the engine work).

Both are synthetic mechanism screens — see the manifest ``prohibited_inferences``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import (  # noqa: E402
    atomic_json_dump,
    build_run_manifest,
    file_sha256,
    finalize_run_manifest,
    utc_now,
)
from quartz.experiments import candidate_morphology as lab  # noqa: E402
from quartz.experiments import h1_synthetic_gate as h1gate  # noqa: E402

DEFAULT_SCENARIO_BANK = REPO_ROOT / "configs" / "candidate_morphology_scenarios.v1.json"


def parse_int_csv(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def load_scenario_bank(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported scenario-bank format_version")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario bank must contain a non-empty scenarios list")
    seen: set[str] = set()
    for row in scenarios:
        scenario_id = str(row.get("id") or "")
        means = row.get("means")
        n_visible = row.get("n_visible")
        budgets = row.get("budgets")
        if not scenario_id or scenario_id in seen:
            raise ValueError(f"missing or duplicate scenario id: {scenario_id!r}")
        seen.add(scenario_id)
        if not isinstance(means, list) or len(means) < 2:
            raise ValueError(f"scenario {scenario_id} requires at least two means")
        if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in means):
            raise ValueError(f"scenario {scenario_id} has an invalid Bernoulli mean")
        if not isinstance(n_visible, int) or not 1 <= n_visible <= len(means):
            raise ValueError(f"scenario {scenario_id} has an invalid n_visible")
        if (
            not isinstance(row.get("prior_noise"), (int, float))
            or row["prior_noise"] < 0
        ):
            raise ValueError(f"scenario {scenario_id} has an invalid prior_noise")
        if not isinstance(budgets, list) or not budgets:
            raise ValueError(f"scenario {scenario_id} requires budgets")
        if len(set(budgets)) != len(budgets):
            raise ValueError(f"scenario {scenario_id} has duplicate budgets")
        if min(budgets) < n_visible:
            raise ValueError(f"scenario {scenario_id} budget below n_visible")
        if max(budgets) > lab.MAX_SUPPORTED_BUDGET:
            raise ValueError(f"scenario {scenario_id} exceeds the audited budget scope")
    return payload


def choose_scenarios(
    bank: Mapping[str, Any], requested: Sequence[str] | None
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): dict(row) for row in bank["scenarios"]}
    if not requested:
        return [dict(row) for row in bank["scenarios"]]
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown scenario ids: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("scenario ids must be unique")
    return [by_id[item] for item in requested]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # Union of keys preserves columns even when baseline rows omit paired fields.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw_row in rows:
            row = dict(raw_row)
            for key, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"), sort_keys=True)
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-bank", type=Path, default=DEFAULT_SCENARIO_BANK)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--allocators", nargs="+", choices=lab.ALLOCATORS, default=list(lab.ALLOCATORS)
    )
    parser.add_argument(
        "--widen-costs",
        type=parse_int_csv,
        default=None,
        help="override widen-cost grid",
    )
    parser.add_argument(
        "--budgets",
        type=parse_int_csv,
        default=None,
        help="optional common budget override",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--quick", action="store_true", help="use 60 trials")
    parser.add_argument(
        "--skip-h1-gate",
        action="store_true",
        help="skip the H1 synthetic gate pre-validation",
    )
    parser.add_argument(
        "--h1-boot", type=int, default=4000, help="Dirichlet draws for the H1 gate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metacognitive_root/candidate_morphology_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bank_path = args.scenario_bank.resolve()
    bank = load_scenario_bank(bank_path)
    scenarios = choose_scenarios(bank, args.scenarios)
    trials = 60 if args.quick else int(args.trials or bank["default_trials"])
    if trials < 1:
        raise SystemExit("trials must be positive")
    allocators = list(args.allocators)
    if len(set(allocators)) != len(allocators):
        raise SystemExit("allocators must be unique")
    if lab.BASELINE_ALLOCATOR not in allocators:
        raise SystemExit(
            f"{lab.BASELINE_ALLOCATOR} baseline must be included for paired deltas"
        )
    default_widen_costs = (
        args.widen_costs or bank.get("default_widen_costs") or [1, 2, 4, 8]
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory is not empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_scenarios = []
    for scenario in scenarios:
        budgets = list(args.budgets or scenario["budgets"])
        if len(set(budgets)) != len(budgets):
            raise SystemExit("budget override must contain unique values")
        if min(budgets) < scenario["n_visible"]:
            raise SystemExit(f"budget override below n_visible for {scenario['id']}")
        resolved = dict(scenario)
        resolved["budgets"] = budgets
        resolved["widen_costs"] = list(
            scenario.get("widen_costs") or default_widen_costs
        )
        resolved_scenarios.append(resolved)

    resolved_config = {
        "scenario_bank": str(bank_path),
        "scenario_bank_sha256": file_sha256(bank_path),
        "scenarios": resolved_scenarios,
        "allocators": allocators,
        "allocator_contracts": {
            name: lab.ALLOCATOR_CONTRACTS[name] for name in allocators
        },
        "default_widen_costs": default_widen_costs,
        "trials": trials,
        "seed": args.seed,
        "baseline_allocator": lab.BASELINE_ALLOCATOR,
        "budget_execution": "independent_rerun_per_budget_and_widen_cost",
        "h1_gate": (not args.skip_h1_gate),
        "h1_boot": args.h1_boot,
    }
    started_at = utc_now()
    source_paths = [
        Path(__file__),
        REPO_ROOT / "quartz" / "experiments" / "candidate_morphology.py",
        REPO_ROOT / "quartz" / "experiments" / "h1_synthetic_gate.py",
        REPO_ROOT / "quartz" / "experiment_manifest.py",
        bank_path,
    ]
    manifest = build_run_manifest(
        experiment_id=lab.EXPERIMENT_ID,
        execution_mode=lab.EXECUTION_MODE,
        resolved_config=resolved_config,
        repo_root=REPO_ROOT,
        source_paths=source_paths,
        argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
        started_at=started_at,
        assumptions=bank["assumptions"],
        prohibited_inferences=bank["prohibited_inferences"],
    )
    manifest_path = output_dir / "run_manifest.json"
    atomic_json_dump(manifest_path, manifest)

    summary_rows: list[dict[str, Any]] = []
    trial_path = output_dir / "trials.jsonl.gz"
    trial_tmp = trial_path.with_suffix(trial_path.suffix + ".tmp")
    with trial_tmp.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            trial_handle = io.TextIOWrapper(gz, encoding="utf-8")
            for scenario in resolved_scenarios:
                records, summaries = lab.run_experiment(
                    means=scenario["means"],
                    n_visible=scenario["n_visible"],
                    prior_noise=float(scenario["prior_noise"]),
                    budgets=scenario["budgets"],
                    widen_costs=scenario["widen_costs"],
                    trials=trials,
                    seed=args.seed,
                    allocators=allocators,
                )
                prefix = {
                    "scenario_id": scenario["id"],
                    "scenario_family": scenario.get("family"),
                    "scenario_label": scenario.get("label"),
                    "n_visible": scenario["n_visible"],
                    "prior_noise": scenario["prior_noise"],
                }
                for row in summaries:
                    summary_rows.append({**prefix, **row})
                for record in records:
                    payload = {**prefix, **asdict(record)}
                    trial_handle.write(
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    )
                    trial_handle.write("\n")
            trial_handle.flush()
            trial_handle.detach()
    os.replace(trial_tmp, trial_path)

    global_verdict = lab.widening_kill_verdict(summary_rows)
    per_scenario_verdict = {
        scenario["id"]: lab.widening_kill_verdict(
            [row for row in summary_rows if row["scenario_id"] == scenario["id"]]
        )
        for scenario in resolved_scenarios
    }

    h1_verdict = None
    if not args.skip_h1_gate:
        h1_verdict = h1gate.run_gate(n_boot=args.h1_boot)

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    write_csv(summary_csv, summary_rows)
    atomic_json_dump(
        summary_json,
        {
            "format_version": 1,
            "experiment_id": lab.EXPERIMENT_ID,
            "execution_mode": lab.EXECUTION_MODE,
            "claim_status": "synthetic_screening_only",
            "scenario_bank_sha256": resolved_config["scenario_bank_sha256"],
            "baseline_allocator": lab.BASELINE_ALLOCATOR,
            "summary_rows": summary_rows,
            "widening_kill_verdict": global_verdict,
            "widening_kill_verdict_per_scenario": per_scenario_verdict,
            "h1_synthetic_gate": h1_verdict,
            "interval_note": "MC95 intervals are descriptive normal Monte Carlo intervals, not guarantees.",
        },
    )
    manifest = finalize_run_manifest(
        manifest,
        output_dir=output_dir,
        artifact_paths=[summary_csv, summary_json, trial_path],
    )
    atomic_json_dump(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "summary_rows": len(summary_rows),
                "widening_lane_demoted": global_verdict["widening_lane_demoted"],
                "widen_prices_with_omission_improvement": global_verdict[
                    "widen_prices_with_omission_improvement"
                ],
                "net_total_improvement_found": global_verdict[
                    "net_total_improvement_found"
                ],
                "h1_gate_pass": (
                    None if h1_verdict is None else h1_verdict["gate_pass"]
                ),
                "run_contract_hash": manifest["run_contract_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
