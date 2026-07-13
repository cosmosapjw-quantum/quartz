#!/usr/bin/env python3
# ruff: noqa: E402
"""forked_voc_lab runner — label frozen phase15 traces by realized VOC.

Loads phase15 trace-cache bundles (``trace_budgets`` + ``trace_policies``),
labels each computation step by its realized root-decision change, and writes
a provenance-manifested screening summary. See
``docs/METACOGNITIVE_EXPERIMENTS.md`` and ``quartz/experiments/forked_voc.py``.

Kill-criterion: if the VOC proxy is degenerate (std ~ 0) on real trained
traces, the P3 discriminating-signature design must be reworked.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import atomic_json_dump, build_run_manifest, finalize_run_manifest
from quartz.experiments import forked_voc as lab

PROHIBITED = [
    "reading a frozen-trace VOC label as a runtime control signal",
    "claiming any allocator improves play from these labels alone",
    "transfer from these labels to a play-strength or CPU-efficiency claim",
    "treating a random-init checkpoint's labels as skill-discrimination evidence",
]
ASSUMPTIONS = [
    "trace bundles are frozen phase15 root-policy ladders (independent reruns per budget)",
    "VOC proxy is an offline oracle-style label, never a runtime input (THESIS.md P3 guard)",
]


def load_bundles(trace_dir: str) -> list[dict[str, Any]]:
    bundles = []
    for path in sorted(glob.glob(str(Path(trace_dir) / "*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "trace_policies" in data and "trace_budgets" in data:
            bundles.append({"__path": path, "trace_budgets": data["trace_budgets"], "trace_policies": data["trace_policies"]})
    return bundles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label frozen phase15 traces by realized value of computation")
    p.add_argument("--trace-dir", required=True, help="phase15 trace_cache dir (bundles as *.json)")
    p.add_argument("--strong-trace-dir", default=None, help="optional second (stronger) checkpoint's traces for a discrimination preview")
    p.add_argument("--output-dir", default="results/metacognitive_root/forked_voc_v1")
    p.add_argument("--label", default="weak", help="label for the primary trace group")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundles = load_bundles(args.trace_dir)
    if not bundles:
        print(json.dumps({"status": "no_bundles", "trace_dir": args.trace_dir}))
        return 2

    screen = lab.screen_bundles(bundles)
    per_bundle = [
        {"path": Path(b["__path"]).name, **{k: v for k, v in lab.label_trace_bundle(b).items() if k != "steps"}}
        for b in bundles
    ]
    summary: dict[str, Any] = {
        "experiment_id": lab.EXPERIMENT_ID,
        "execution_mode": lab.EXECUTION_MODE,
        "primary_label": args.label,
        "screen": screen,
        "per_bundle": per_bundle,
    }

    if args.strong_trace_dir:
        strong = load_bundles(args.strong_trace_dir)
        if strong:
            summary["discrimination"] = lab.discrimination(bundles, strong)
            summary["strong_screen"] = lab.screen_bundles(strong)

    resolved_config = {
        "trace_dir": str(args.trace_dir),
        "strong_trace_dir": str(args.strong_trace_dir) if args.strong_trace_dir else None,
        "n_bundles": len(bundles),
        "schema_version": lab.FORKED_VOC_SCHEMA_VERSION,
    }
    manifest = build_run_manifest(
        experiment_id=lab.EXPERIMENT_ID,
        execution_mode=lab.EXECUTION_MODE,
        resolved_config=resolved_config,
        repo_root=REPO_ROOT,
        source_paths=[REPO_ROOT / "quartz/experiments/forked_voc.py", REPO_ROOT / "scripts/forked_voc_lab.py"],
        argv=sys.argv,
        started_at=started_at,
        assumptions=ASSUMPTIONS,
        prohibited_inferences=PROHIBITED,
    )
    summary_path = out / "summary.json"
    atomic_json_dump(summary_path, summary)
    manifest = finalize_run_manifest(manifest, output_dir=out, artifact_paths=[summary_path])
    atomic_json_dump(out / "run_manifest.json", manifest)

    print(json.dumps({
        "status": "completed",
        "output_dir": str(out),
        "n_positions": screen["n_positions"],
        "voc_proxy_std": screen["voc_proxy_std"],
        "degenerate": screen["degenerate"],
        "overturn_rate": screen["overturn_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
