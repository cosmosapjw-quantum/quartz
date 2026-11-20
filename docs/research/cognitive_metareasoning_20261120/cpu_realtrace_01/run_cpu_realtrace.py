#!/usr/bin/env python3
"""COG-01B CPU real-checkpoint and observational frozen-trace experiment.

Run only with the recorded input repository checkout and in a clean research commit.
This is retrospective independent-restart alignment, NOT true marginal MCTS VOC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from cpu_realtrace import build_records, crossfit, summarize


CHECKPOINT_IDS = (
    "C01_gen_5", "C02_latest", "C03_gen_5",
    "C04_latest", "C05_gen_5", "C06_latest",
)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True, stderr=subprocess.DEVNULL
    ).strip()


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _jsonl_dump(path: Path, rows) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent
    repo = next(parent for parent in directory.parents if (parent / ".git").exists())
    root = args.input_root.resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise RuntimeError("input root must be an existing Git checkout")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("study source working tree is not clean")
    study_head = _git(repo, "rev-parse", "HEAD")
    lock_path = directory / "INPUT_LOCK.json"
    lock = json.loads(lock_path.read_text())
    if lock["schema"] != "quartz_cog01b_cpu_inputs_v1" or len(lock["files"]) != 296:
        raise RuntimeError("invalid frozen input schema/count")
    if _git(root, "rev-parse", "HEAD") != lock["input_repository_observed_commit"]:
        raise RuntimeError("input source checkout HEAD changed")

    for entry in lock["files"]:
        p = root / entry["path"]
        if not p.is_file() or p.stat().st_size != entry["bytes"] or _sha(p) != entry["sha256"]:
            raise RuntimeError(f"FROZEN_INPUT_SHA_MISMATCH: {entry['path']}")

    import numpy as np
    import torch
    from quartz.models_torch import AlphaZeroNet
    from quartz.runtime_support import encode_board

    torch.set_num_threads(2)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("this run requires a verified CPU-only PyTorch build")
    base = lock["declared_inputs"]
    positions = json.loads((root / base["position_suite"]).read_text())["positions"]
    assays = [json.loads(line) for line in (root / base["assay"]).read_text().splitlines() if line]
    if len(positions) != 48 or len(assays) != 6912:
        raise RuntimeError("unexpected source sizes")
    traces = [json.loads((root / e["path"]).read_text()) for e in lock["files"]
              if e["path"].startswith(base["trace_dir"] + "/") and e["path"].endswith(".json")]
    if len(traces) != 288:
        raise RuntimeError("expected 288 original frozen bundles")

    # All candidate paths come from frozen hash-lock; run no code from input records.
    prior = {}
    inference_runs = []
    checkpoint_paths = base["checkpoint_paths"]
    if len(checkpoint_paths) != 6:
        raise RuntimeError("expected 6 checkpoints")
    for ck, relative in zip(CHECKPOINT_IDS, checkpoint_paths):
        checkpoint = root / relative
        raw = torch.load(checkpoint, weights_only=True, map_location="cpu")
        cfg = raw["cfg"]
        if not (cfg["board"] == 7 and cfg["ch"] == 17 and cfg["actions"] == 49):
            raise RuntimeError(f"checkpoint network shape outside registered scope: {ck}")
        model = AlphaZeroNet(cfg).eval().to("cpu")
        model.load_state_dict(raw["model_state_dict"], strict=True)
        encoded = np.stack([
            encode_board(cfg, p["board"], int(p["player"]))
            for p in positions
        ]).astype(np.float32)
        if encoded.shape != (48, 17, 7, 7):
            raise RuntimeError("unexpected Gomoku7 encoder input shape")
        x = torch.from_numpy(encoded)
        start = time.perf_counter()
        with torch.inference_mode():
            logits, values = model(x)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            values = values.cpu().numpy()
        elapsed_s = time.perf_counter() - start
        if not (np.isfinite(probabilities).all() and np.isfinite(values).all()):
            raise RuntimeError(f"NN output nonfinite for {ck}")
        for p, v, prob in zip(positions, values, probabilities):
            prior[(ck, p["id"])] = {"policy": [float(z) for z in prob], "value": float(v)}
        inference_runs.append({
            "checkpoint_id": ck, "path": relative,
            "sha256": next(e["sha256"] for e in lock["files"] if e["path"] == relative),
            "input_shape": list(encoded.shape),
            "policy_shape": list(probabilities.shape),
            "batch_cpu_wall_s": elapsed_s,
            "device": "cpu",
        })
        del model, raw

    rows = build_records(traces, assays, prior, positions)
    if not (len(rows) == 864 and len({(r["checkpoint_id"], r["position_id"]) for r in rows}) == 288):
        raise RuntimeError("unexpected joined evidence cardinality")
    predictions = crossfit(rows, folds=6)
    summary = summarize(rows, predictions, folds=6, bootstraps=2000, seed=20261120)

    # Recheck hashes after all input reads to ensure no concurrent edits.
    for entry in lock["files"]:
        if _sha(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"FROZEN_INPUT_CHANGED_DURING_RUN: {entry['path']}")

    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=False)
    _jsonl_dump(target / "records.jsonl", rows)
    _jsonl_dump(target / "predictions_oof.jsonl", [
        {
            "checkpoint_id": r["checkpoint_id"],
            "position_id": r["position_id"], "budget": r["budget"],
            "reference_delta": r["delta_reference"],
            "M0": predictions["M0"][i], "M1": predictions["M1"][i],
            "M2": predictions["M2"][i],
        }
        for i, r in enumerate(rows)
    ])
    _json_dump(target / "summary.json", summary)
    _json_dump(target / "cpu_model_checks.json", inference_runs)

    out_files = sorted(p for p in target.iterdir() if p.is_file())
    manifest = {
        "schema": "quartz_cog01b_cpu_realtrace_manifest_v1",
        "repo_head": study_head,
        "repo_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "input_git_head_observed": lock["input_repository_observed_commit"],
        "input_lock_sha256": _sha(lock_path),
        "output_sha256": {p.name: _sha(p) for p in out_files},
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "randomness": {"bootstrap_seed": 20261120, "bootstrap_units": "position_id",
                       "crossfit_folds": 6, "bootstrap_refits_model": False},
        "nonclaims": ["not_marginal_VOC", "not_minimax_truth", "not_elo",
                      "not_independent_position_sampling",
                      "not_gpu_throughput", "not_novel_neuroscience"],
    }
    _json_dump(target / "run_manifest.json", manifest)
    print(json.dumps({
        "status": "completed",
        "source_commit": study_head,
        "n_records": len(rows),
        "n_model_roots": len(prior),
        "n_positions": summary["n_positions"],
        "input_lock": manifest["input_lock_sha256"],
        "signed_counts": summary["signed_counts"],
        "model_mse": summary["model_mse"],
        "output": str(target),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
