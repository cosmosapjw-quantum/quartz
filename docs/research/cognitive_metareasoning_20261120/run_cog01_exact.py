"""Run a frozen exact COG-01 screen; refuses to overwrite an existing result.

Usage from repo root:
  python docs/research/cognitive_metareasoning_20261120/run_cog01_exact.py \\
    --output-dir docs/research/cognitive_metareasoning_20261120/evidence/exact_20261120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

from cognitive_exact import SOURCE, SCENARIOS, U, COST, evaluate, posterior


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def exact_number(v: F) -> dict[str, str | float]:
    if not isinstance(v, F):
        raise TypeError("all science values must be exact Fractions")
    return {"exact": str(v), "decimal": float(v)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    script = Path(__file__).resolve()
    directory = script.parent
    repo = next(
        p for p in directory.parents if (p / ".git").exists()
    )
    tracked = [
        directory / "COG01_CONTRACT.md",
        directory / "cognitive_exact.py",
        directory / "run_cog01_exact.py",
        directory / "tests_research" / "test_cog01_exact.py",
    ]
    if any(not path.is_file() for path in tracked):
        raise RuntimeError("required contract/source file missing")
    status = git(repo, "status", "--porcelain")
    if status:
        raise RuntimeError("source tree must be clean before exact evidence execution")

    # Output must be new. No silently reused or overwritten rows.
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=False)

    source = {
        "p": exact_number(SOURCE.p),
        "se": exact_number(SOURCE.se),
        "fpr": exact_number(SOURCE.fpr),
        "q_plus": exact_number(posterior(SOURCE, "+")),
        "q_minus": exact_number(posterior(SOURCE, "-")),
        "u": exact_number(U),
        "cost": exact_number(COST),
    }
    rows = []
    for label, world in SCENARIOS.items():
        result = evaluate(SOURCE, world)
        rows.append({
            "scenario": label,
            "true_parameters": {
                "p": exact_number(world.p),
                "se": exact_number(world.se),
                "fpr": exact_number(world.fpr),
            },
            "metrics": {k: exact_number(v) for k, v in sorted(result.items())},
        })

    payload = {
        "schema": "quartz_cog01_exact_v1",
        "kind": "exact_finite_bayesian_reference_not_game_engine",
        "source": source,
        "rows": rows,
        "scenario_count": len(rows),
        "restrictions": [
            "binary_X", "known_u", "perfect_verification",
            "fixed_cost", "source_trained_q", "no_neural_checkpoint",
            "random_rate_is_deployment-matched_diagnostic",
        ],
    }
    evidence_file = target / "exact_results.json"
    evidence_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "quartz_cog01_run_manifest_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git(repo, "rev-parse", "HEAD"),
        "git_tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "git_branch": git(repo, "branch", "--show-current"),
        "source_clean_before_execution": True,
        "source_sha256": {str(p.relative_to(repo)): digest(p) for p in tracked},
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "randomness": "none_exact_rational_enumeration",
        "raw_output_sha256": digest(evidence_file),
        "not_claimed": [
            "learned_evaluator_calibration", "MCTS_strength",
            "human_mechanism_identification", "time_uniform_confidence",
        ],
    }
    (target / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "source_head": manifest["git_head"], "rows": len(rows),
        "result_sha256": manifest["raw_output_sha256"],
        "output": str(target),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
