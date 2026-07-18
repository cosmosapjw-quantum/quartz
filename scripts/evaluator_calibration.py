#!/usr/bin/env python3
"""Generate held-out evaluator calibration artifacts for ablation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from quartz import runtime_support
from quartz.replay import ReplayBuffer

import ablation_study


def sha256_file(path: Path, prefix_len: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    out = h.hexdigest()
    return out[:prefix_len] if prefix_len else out


def load_replay_examples(path: Path, limit: int | None = None):
    replay = ReplayBuffer(1_000_000)
    n = replay.load(str(path))
    if n <= 0:
        raise ValueError(f"no replay examples loaded from {path}")
    count = min(n, int(limit)) if limit else n
    return replay.examples_at_indices(range(count))


def calibration_metrics(actor, device, examples, batch_size: int = 128) -> dict:
    states = np.asarray([ex.state for ex in examples], dtype=np.float32)
    target_policies = [np.asarray(ex.policy, dtype=np.float32) for ex in examples]
    target_values = np.asarray([ex.value for ex in examples], dtype=np.float32)
    n = int(len(examples))
    if n <= 0:
        raise ValueError("calibration requires at least one replay example")
    batch_size = max(1, int(batch_size))
    eps = 1e-8
    policy_nll = 0.0
    brier = 0.0
    top1_hits = 0
    value_sqerr = 0.0
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        probs, values = runtime_support.run_model_batch(
            actor, device, states[start:end]
        )
        probs = np.asarray(probs, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        targets = np.asarray(target_policies[start:end], dtype=np.float64)
        policy_nll += float(-(targets * np.log(np.maximum(probs, eps))).sum())
        diff = probs - targets
        brier += float((diff * diff).sum())
        target_argmax = np.argmax(targets, axis=1)
        pred_argmax = np.argmax(probs, axis=1)
        top1_hits += int((target_argmax == pred_argmax).sum())
        value_diff = values - target_values[start:end]
        value_sqerr += float((value_diff * value_diff).sum())
    return {
        "n_positions": n,
        "policy_nll": float(policy_nll / n),
        "value_mse": float(value_sqerr / n),
        "top1_acc": float(top1_hits / n),
        "brier": float(brier / n),
    }


def build_payload(
    ablation_dir: Path, dataset: Path, device: str, limit: int | None, batch_size: int
) -> dict:
    examples = load_replay_examples(dataset, limit=limit)
    runs = ablation_study.discover_model_runs(ablation_dir)
    models = {}
    errors = {}
    for run in runs:
        model_id = str(run.get("id") or "")
        model_path = run.get("model_path")
        if not model_id or not model_path:
            continue
        try:
            cfg, device_obj = ablation_study.build_eval_cfg(
                run.get("game") or "gomoku7",
                run.get("train_cfg") or {},
                device,
                model_path=model_path,
            )
            actor = runtime_support.load_actor_source_from_checkpoint(
                model_path,
                cfg,
                device_obj,
                backend_preference="torch",
            )
            models[model_id] = calibration_metrics(
                actor, device_obj, examples, batch_size=batch_size
            )
            models[model_id]["model_path"] = str(model_path)
            models[model_id]["model_sha256"] = sha256_file(
                Path(model_path), prefix_len=16
            )
        except Exception as exc:
            errors[model_id] = str(exc)
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ablation_dir": str(ablation_dir),
        "dataset_path": str(dataset),
        "dataset_sha256": sha256_file(dataset, prefix_len=16),
        "dataset_format": "quartz_replay_npz",
        "sampled_positions": int(len(examples)),
        "device": str(device),
        "batch_size": int(batch_size),
        "models": models,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluator_calibration.json from held-out replay NPZ"
    )
    parser.add_argument(
        "--ablation-dir",
        required=True,
        help="Ablation game directory containing model runs",
    )
    parser.add_argument("--dataset", required=True, help="Held-out ReplayBuffer NPZ")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path; default <ablation-dir>/evaluator_calibration.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum positions to evaluate"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)
    output = (
        Path(args.output)
        if args.output
        else ablation_dir / "evaluator_calibration.json"
    )
    payload = build_payload(
        ablation_dir,
        Path(args.dataset),
        args.device,
        args.limit,
        args.batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {output} ({len(payload['models'])} models, {len(payload['errors'])} errors)"
    )


if __name__ == "__main__":
    main()
