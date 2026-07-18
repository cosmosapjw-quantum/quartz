#!/usr/bin/env python3
"""Run the preregistered synthetic Bernoulli root-ranking assay."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import itertools
import json
import math
import os
import random
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
from quartz.experiments import bernoulli_root as lab  # noqa: E402


DEFAULT_SCENARIO_BANK = REPO_ROOT / "configs" / "metacognitive_root_scenarios.v1.json"


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
        budgets = row.get("budgets")
        if not scenario_id or scenario_id in seen:
            raise ValueError(f"missing or duplicate scenario id: {scenario_id!r}")
        seen.add(scenario_id)
        if not isinstance(means, list) or len(means) < 2:
            raise ValueError(f"scenario {scenario_id} requires at least two means")
        if any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1
            for value in means
        ):
            raise ValueError(f"scenario {scenario_id} has an invalid Bernoulli mean")
        if not isinstance(budgets, list) or not budgets:
            raise ValueError(f"scenario {scenario_id} requires budgets")
        if len(set(budgets)) != len(budgets):
            raise ValueError(f"scenario {scenario_id} has duplicate budgets")
        if min(budgets) < len(means):
            raise ValueError(
                f"scenario {scenario_id} cannot fund raw sequential halving"
            )
        if max(budgets) > lab.MAX_SUPPORTED_BUDGET:
            raise ValueError(f"scenario {scenario_id} exceeds the audited budget scope")
    return payload


def choose_scenarios(
    bank: Mapping[str, Any], requested: Sequence[str] | None
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): dict(row) for row in bank["scenarios"]}
    if not requested:
        return [by_id[str(row["id"])] for row in bank["scenarios"]]
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown scenario ids: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("scenario ids must be unique")
    return [by_id[item] for item in requested]


def arm_permutation(
    num_arms: int, seed: int, scenario_id: str, permutation_id: int
) -> list[int]:
    permutation = list(range(num_arms))
    rng = random.Random(
        lab.stable_seed(seed, "arm_permutation", scenario_id, permutation_id)
    )
    rng.shuffle(permutation)
    return permutation


def two_sided_exact_binomial_pvalue(successes: int, trials: int) -> float | None:
    """Exact two-sided sign-test p-value for a fair Bernoulli null."""

    if trials <= 0:
        return None
    tail = sum(math.comb(trials, k) for k in range(0, successes + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def pairwise_contrasts(
    records: Sequence[lab.TrialRecord],
    algorithms: Sequence[str],
) -> list[dict[str, Any]]:
    by_key = {(row.algorithm, row.budget, row.trial): row for row in records}
    budgets = sorted({row.budget for row in records})
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        trials = sorted({row.trial for row in records if row.budget == budget})
        for left, right in itertools.combinations(algorithms, 2):
            pairs = [
                (by_key[(left, budget, trial)], by_key[(right, budget, trial)])
                for trial in trials
            ]
            regret_deltas = [a.simple_regret - b.simple_regret for a, b in pairs]
            pcs_deltas = [a.correct_selection - b.correct_selection for a, b in pairs]
            mean_delta = sum(regret_deltas) / len(regret_deltas)
            if len(regret_deltas) > 1:
                variance = sum((value - mean_delta) ** 2 for value in regret_deltas) / (
                    len(regret_deltas) - 1
                )
            else:
                variance = 0.0
            mcse = math.sqrt(variance / len(regret_deltas))
            left_only_correct = sum(
                1 for a, b in pairs if a.correct_selection and not b.correct_selection
            )
            right_only_correct = sum(
                1 for a, b in pairs if b.correct_selection and not a.correct_selection
            )
            discordant = left_only_correct + right_only_correct
            rows.append(
                {
                    "budget": budget,
                    "left_algorithm": left,
                    "right_algorithm": right,
                    "paired_trials": len(pairs),
                    "mean_regret_delta_left_minus_right": mean_delta,
                    "regret_delta_mcse": mcse,
                    "regret_delta_mc95_low": mean_delta - 1.96 * mcse,
                    "regret_delta_mc95_high": mean_delta + 1.96 * mcse,
                    "mean_pcs_delta_left_minus_right": sum(pcs_deltas)
                    / len(pcs_deltas),
                    "left_only_correct": left_only_correct,
                    "right_only_correct": right_only_correct,
                    "discordant_pairs": discordant,
                    "mcnemar_exact_pvalue": two_sided_exact_binomial_pvalue(
                        min(left_only_correct, right_only_correct), discordant
                    ),
                    "inference_status": "exploratory_unadjusted",
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0].keys())
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
        "--algorithms", nargs="+", choices=lab.ALGORITHMS, default=list(lab.ALGORITHMS)
    )
    parser.add_argument(
        "--budgets",
        type=parse_int_csv,
        default=None,
        help="optional common budget override",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--quick", action="store_true", help="use 50 trials and two permutations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metacognitive_root/bernoulli_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bank_path = args.scenario_bank.resolve()
    bank = load_scenario_bank(bank_path)
    scenarios = choose_scenarios(bank, args.scenarios)
    trials = 50 if args.quick else int(args.trials or bank["default_trials"])
    permutations = (
        2 if args.quick else int(args.permutations or bank["default_permutations"])
    )
    if trials < 1 or permutations < 1:
        raise SystemExit("trials and permutations must be positive")
    algorithms = list(args.algorithms)
    if len(set(algorithms)) != len(algorithms):
        raise SystemExit("algorithms must be unique")

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
        if min(budgets) < len(scenario["means"]):
            raise SystemExit(f"budget override cannot fund {scenario['id']}")
        resolved = dict(scenario)
        resolved["budgets"] = budgets
        resolved_scenarios.append(resolved)

    resolved_config = {
        "scenario_bank": str(bank_path),
        "scenario_bank_sha256": file_sha256(bank_path),
        "scenarios": resolved_scenarios,
        "algorithms": algorithms,
        "algorithm_contracts": {
            name: lab.ALGORITHM_CONTRACTS[name] for name in algorithms
        },
        "trials_per_permutation": trials,
        "permutations": permutations,
        "seed": args.seed,
        "budget_execution": "independent_rerun_per_budget",
        "trial_artifact": "gzip_jsonl_always_emitted",
        "multiplicity_policy": "exploratory_unadjusted_all_pairwise_contrasts",
    }
    started_at = utc_now()
    source_paths = [
        Path(__file__),
        REPO_ROOT / "quartz" / "experiments" / "bernoulli_root.py",
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
    contrast_rows: list[dict[str, Any]] = []
    trial_path = output_dir / "trials.jsonl.gz"
    trial_tmp = trial_path.with_suffix(trial_path.suffix + ".tmp")
    with trial_tmp.open("wb") as raw_trial_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_trial_handle,
            mtime=0,
        ) as compressed_trial_handle:
            trial_handle = io.TextIOWrapper(compressed_trial_handle, encoding="utf-8")
            for scenario in resolved_scenarios:
                canonical_means = [float(value) for value in scenario["means"]]
                experiment_seed = lab.stable_seed(args.seed, "scenario", scenario["id"])
                for permutation_id in range(permutations):
                    permutation = arm_permutation(
                        len(canonical_means),
                        args.seed,
                        str(scenario["id"]),
                        permutation_id,
                    )
                    presented_means = [canonical_means[index] for index in permutation]
                    records, summaries = lab.run_experiment(
                        presented_means,
                        scenario["budgets"],
                        trials,
                        experiment_seed,
                        algorithms,
                        arm_keys=permutation,
                    )
                    prefix = {
                        "scenario_id": scenario["id"],
                        "scenario_family": scenario["family"],
                        "scenario_label": scenario["label"],
                        "permutation_id": permutation_id,
                        "presented_to_canonical": permutation,
                    }
                    for row in summaries:
                        summary_rows.append({**prefix, **row})
                    for row in pairwise_contrasts(records, algorithms):
                        contrast_rows.append({**prefix, **row})
                    for record in records:
                        payload = {
                            **prefix,
                            **asdict(record),
                            "selected_canonical_arm": permutation[record.selected_arm],
                        }
                        trial_handle.write(
                            json.dumps(payload, sort_keys=True, separators=(",", ":"))
                        )
                        trial_handle.write("\n")
            trial_handle.flush()
            trial_handle.detach()
    os.replace(trial_tmp, trial_path)

    summary_csv = output_dir / "summary.csv"
    contrasts_csv = output_dir / "contrasts.csv"
    summary_json = output_dir / "summary.json"
    write_csv(summary_csv, summary_rows)
    write_csv(contrasts_csv, contrast_rows)
    atomic_json_dump(
        summary_json,
        {
            "format_version": 1,
            "experiment_id": lab.EXPERIMENT_ID,
            "execution_mode": lab.EXECUTION_MODE,
            "claim_status": "synthetic_screening_only",
            "scenario_bank_sha256": resolved_config["scenario_bank_sha256"],
            "summary_rows": summary_rows,
            "contrast_rows": contrast_rows,
            "interval_note": "MC95 intervals are descriptive normal Monte Carlo intervals, not guarantees.",
            "multiplicity_policy": resolved_config["multiplicity_policy"],
        },
    )
    manifest = finalize_run_manifest(
        manifest,
        output_dir=output_dir,
        artifact_paths=[summary_csv, contrasts_csv, summary_json, trial_path],
    )
    atomic_json_dump(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "summary_rows": len(summary_rows),
                "contrast_rows": len(contrast_rows),
                "run_contract_hash": manifest["run_contract_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
