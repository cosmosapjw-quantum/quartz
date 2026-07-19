#!/usr/bin/env python3
"""Plan or run one claim-safe Idea Foundry scientific gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.idea_foundry.axis_workflow import load_jsonl_strict  # noqa: E402
from quartz.experiment_manifest import file_sha256  # noqa: E402
from quartz.idea_foundry.studies import (  # noqa: E402
    StudyError,
    StudyOutcome,
    publish_outcome,
    run_inprocess_study,
    study_plan,
    study_spec,
)
from scripts.idea_foundry_study_all import (  # noqa: E402
    TERMINAL,
    _validate_artifact_set,
)


def _default_output(axis_id: str, profile: str) -> Path:
    return (
        REPO_ROOT / "results" / "idea_foundry_studies" / f"manual-{profile}" / axis_id
    )


def _run_command(command: Sequence[str], native_dir: Path) -> None:
    native_dir.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = native_dir.parent / f"{native_dir.name}.stdout.log"
    stderr_path = native_dir.parent / f"{native_dir.name}.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8", buffering=1) as stdout,
        stderr_path.open("w", encoding="utf-8", buffering=1) as stderr,
    ):
        process = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise StudyError(
            f"native executor failed with exit {process.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )


def _archive_incomplete_output(output_dir: Path) -> Path:
    attempt = 1
    while True:
        archived = output_dir.with_name(
            f"{output_dir.name}.incomplete-attempt-{attempt}"
        )
        if not archived.exists():
            break
        attempt += 1
    output_dir.rename(archived)
    for suffix in ("stdout.log", "stderr.log"):
        log_path = output_dir.parent / f"{output_dir.name}.{suffix}"
        if log_path.exists():
            log_path.rename(output_dir.parent / f"{archived.name}.{suffix}")
    return archived


def _run_or_reuse_native(command: Sequence[str], native_dir: Path) -> str:
    if native_dir.exists():
        validated_status = _validate_artifact_set(native_dir)
        if validated_status in TERMINAL:
            return "verified_reuse"
        _archive_incomplete_output(native_dir)
    _run_command(command, native_dir)
    if _validate_artifact_set(native_dir) not in TERMINAL:
        raise StudyError(f"native executor published invalid artifacts: {native_dir}")
    return "executed"


def _a15_outcome(native_dir: Path) -> StudyOutcome:
    summary_path = native_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, list) or not aggregates:
        raise StudyError("A15 native summary contains no service-curve aggregates")
    by_cell: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregates:
        by_cell[(int(row["batch_size"]), int(row["inflight"]))][str(row["backend"])] = (
            row
        )
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for (batch_size, inflight), pair in sorted(by_cell.items()):
        if set(pair) != {"cpu", "cuda"}:
            raise StudyError(
                f"A15 matched backend pair is incomplete: {(batch_size, inflight)}"
            )
        cpu = float(pair["cpu"]["items_per_s_median"])
        cuda = float(pair["cuda"]["items_per_s_median"])
        effect = math.log(cuda / cpu)
        group = f"batch-{batch_size}"
        grouped[group].append(effect)
        rows.append(
            {
                "schema_version": 1,
                "axis_id": "A15",
                "independent_group_id": group,
                "unit_id": f"batch-{batch_size}-inflight-{inflight}",
                "candidate": cuda,
                "reference": cpu,
                "paired_effect": effect,
                "batch_size": batch_size,
                "inflight": inflight,
                "semantic_parity_passed": bool(summary["semantic_parity"]["passed"]),
                "representative_workload_only": True,
            }
        )
    return StudyOutcome(
        rows,
        dict(grouped),
        outcome_detail="MATCHED_SERVICE_CURVE_DIAGNOSTIC_COMPLETED",
        notes=(
            "The workload is representative and power was not measured; this is not an energy-efficiency claim.",
        ),
        inputs=(
            summary_path,
            native_dir / "rows.jsonl",
            native_dir / "run_manifest.json",
        ),
    )


def _a18_outcome(native_dir: Path) -> StudyOutcome:
    native_rows = load_jsonl_strict(native_dir / "rows.jsonl")
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in native_rows:
        by_seed[int(row["seed"])][str(row["variant"])] = row
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for seed, variants in sorted(by_seed.items()):
        required = {"matched_direct_baseline", "latent_gaussian_denoise"}
        if set(variants) != required:
            raise StudyError(f"A18 paired variants are incomplete for seed {seed}")
        baseline = variants["matched_direct_baseline"]
        candidate = variants["latent_gaussian_denoise"]
        baseline_loss = float(baseline["policy_target_nll"]) + float(
            baseline["value_mse"]
        )
        candidate_loss = float(candidate["policy_target_nll"]) + float(
            candidate["value_mse"]
        )
        effect = baseline_loss - candidate_loss
        group = f"seed-{seed}"
        grouped[group].append(effect)
        rows.append(
            {
                "schema_version": 1,
                "axis_id": "A18",
                "independent_group_id": group,
                "unit_id": group,
                "candidate": candidate_loss,
                "reference": baseline_loss,
                "paired_effect": effect,
                "evaluation_kind": candidate["evaluation_kind"],
                "parameter_count_matched": candidate["parameter_count"]
                == baseline["parameter_count"],
                "direct_inference_flops_matched": candidate["direct_inference_flops"]
                == baseline["direct_inference_flops"],
            }
        )
    return StudyOutcome(
        rows,
        dict(grouped),
        outcome_detail="PAIRED_EVALUATOR_DIAGNOSTIC_COMPLETED",
        notes=(
            "The study evaluates a frozen replay contract and does not establish play strength.",
        ),
        inputs=(
            native_dir / "rows.jsonl",
            native_dir / "summary.json",
            native_dir / "run_manifest.json",
        ),
    )


def _a19_outcome(native_dir: Path) -> StudyOutcome:
    native_rows = load_jsonl_strict(native_dir / "proxy_results.jsonl")
    finalized_dir = native_dir.with_name(f"{native_dir.name}.finalized")
    selected: set[int] = set()
    extra_inputs: tuple[Path, ...] = ()
    if finalized_dir.is_dir():
        shortlist = json.loads(
            (finalized_dir / "a19_graph_seed_shortlist.v1.json").read_text(
                encoding="utf-8"
            )
        )
        selected = {int(value) for value in shortlist["shortlisted_graph_seeds"]}
        extra_inputs = (
            finalized_dir / "run_manifest.json",
            finalized_dir / "summary.json",
            finalized_dir / "a19_graph_seed_shortlist.v1.json",
        )
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in native_rows:
        by_seed[int(row["replicate_seed"])].append(row)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for replicate_seed, candidates in sorted(by_seed.items()):
        scores = [
            0.65 * float(row["metrics"]["policy_kl"])
            + 0.35 * float(row["metrics"]["value_mse"])
            for row in candidates
        ]
        median = sorted(scores)[len(scores) // 2]
        for row, score in zip(candidates, scores, strict=True):
            effect = median - score
            group = f"seed-{replicate_seed}"
            grouped[group].append(effect)
            rows.append(
                {
                    "schema_version": 1,
                    "axis_id": "A19",
                    "independent_group_id": group,
                    "unit_id": f"graph-{row['graph_seed']}",
                    "candidate": score,
                    "reference": median,
                    "paired_effect": effect,
                    "graph_seed": int(row["graph_seed"]),
                    "selected_by_preregistered_finalize": int(row["graph_seed"])
                    in selected,
                    "fixed_controller": True,
                    "resource_match": row["resources"],
                }
            )
    return StudyOutcome(
        rows,
        dict(grouped),
        outcome_detail="FIXED_REPLAY_GRAPH_SEED_PROXY_COMPLETED",
        notes=("A proxy graph shortlist is not evaluator or play-strength evidence.",),
        inputs=(
            native_dir / "proxy_results.jsonl",
            native_dir / "summary.json",
            native_dir / "run_manifest.json",
            *extra_inputs,
        ),
    )


def _run_native(
    axis_id: str,
    *,
    profile: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    native_dir = output_dir.with_name(f"{output_dir.name}.native")
    if axis_id == "A15":
        native_profile = "diagnostic" if profile == "pilot" else "full"
        command = [
            sys.executable,
            "scripts/a15_matched_service_curve.py",
            "--profile",
            native_profile,
            "--output-dir",
            str(native_dir),
        ]
        adapter = _a15_outcome
        sources = (REPO_ROOT / "scripts" / "a15_matched_service_curve.py",)
    elif axis_id == "A18":
        spec_name = (
            "a18_evaluator_ablation.smoke.v1.json"
            if profile == "pilot"
            else "a18_evaluator_ablation.study.v1.json"
        )
        command = [
            sys.executable,
            "scripts/a18_evaluator_ablation.py",
            "--spec",
            f"configs/{spec_name}",
            "--output-dir",
            str(native_dir),
            "--device",
            "cuda",
            "run",
        ]
        adapter = _a18_outcome
        sources = (REPO_ROOT / "scripts" / "a18_evaluator_ablation.py",)
    elif axis_id == "A19":
        command = [
            sys.executable,
            "scripts/a19_proxy_screen.py",
            "--profile",
            profile,
            "--seed",
            str(seed),
            "--output-dir",
            str(native_dir),
        ]
        adapter = _a19_outcome
        sources = (REPO_ROOT / "scripts" / "a19_proxy_screen.py",)
    else:
        raise StudyError(f"unsupported native axis: {axis_id}")
    _run_or_reuse_native(command, native_dir)
    if axis_id == "A19" and profile == "full":
        finalized_dir = native_dir.with_name(f"{native_dir.name}.finalized")
        controller = REPO_ROOT / "configs" / "idea_foundry.a19.controller.v1.json"
        proxy_results = native_dir / "proxy_results.jsonl"
        finalize_command = [
            sys.executable,
            "scripts/a19_prepare_ablation.py",
            "--screen-plan",
            "configs/idea_foundry.a19.screen.v1.json",
            "--replay-manifest",
            "configs/idea_foundry.a19.replays.v1.json",
            "--controller-checkpoint",
            str(controller),
            "--controller-sha256",
            file_sha256(controller),
            "--output-dir",
            str(finalized_dir),
            "--run-id",
            f"{output_dir.parent.name}-{output_dir.name}-a19-finalized",
            "--proxy-results",
            str(proxy_results),
            "--proxy-results-sha256",
            file_sha256(proxy_results),
        ]
        _run_or_reuse_native(finalize_command, finalized_dir)
    outcome = adapter(native_dir)
    return publish_outcome(
        axis_id=axis_id,
        profile=profile,
        seed=seed,
        output_dir=output_dir,
        outcome=outcome,
        extra_sources=(
            Path(__file__).resolve(),
            REPO_ROOT / "scripts" / "idea_foundry_study_all.py",
            *sources,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--json", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--axis", required=True)
    run_parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    run_parser.add_argument("--seed", type=int, default=20260719)
    run_parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            payload = study_plan()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        spec = study_spec(args.axis)
        output = args.output_dir or _default_output(spec.axis_id, args.profile)
        if spec.runner.endswith("_native"):
            summary = _run_native(
                spec.axis_id,
                profile=args.profile,
                seed=args.seed,
                output_dir=output,
            )
        else:
            summary = run_inprocess_study(
                spec.axis_id,
                profile=args.profile,
                seed=args.seed,
                output_dir=output,
                entrypoint=Path(__file__).resolve(),
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, StudyError) as exc:
        print(f"IDEA FOUNDRY STUDY BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
