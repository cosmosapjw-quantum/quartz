#!/usr/bin/env python3
"""service_curve_lab runner — measure the GPU evaluator service curve.

Sweeps batch size x global inflight credit on a representative evaluator body and
records throughput/latency (and power when nvidia-smi is available), then reports
the H4 scheduler-lane verdict (does inflight credit beat the best fixed batch?).
See ``quartz/experiments/service_curve.py`` and
``docs/METACOGNITIVE_EXPERIMENTS.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
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
from quartz.experiments import service_curve as lab  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "service_curve.v1.json"


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported config format_version")
    if payload.get("experiment_id") != lab.EXPERIMENT_ID:
        raise ValueError("config experiment_id mismatch")
    return payload


def _int_csv(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(x) for x in str(raw).split(",") if x.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=str, default=None, help="comma-separated override")
    parser.add_argument("--inflight-grid", type=str, default=None, help="comma-separated override")
    parser.add_argument("--n-waves", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metacognitive_root/service_curve_v1"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    m = cfg["model"]

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    device = torch.device(requested)

    batch_sizes = _int_csv(args.batch_sizes) or [int(x) for x in cfg["default_batch_sizes"]]
    inflight_grid = _int_csv(args.inflight_grid) or [int(x) for x in cfg["default_inflight_grid"]]
    n_waves = int(args.n_waves if args.n_waves is not None else cfg["default_n_waves"])
    warmup = int(args.warmup if args.warmup is not None else cfg["default_warmup"])
    min_gain = float(cfg.get("default_min_gain", 0.05))

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    resolved_config = {
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model": m,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "batch_sizes": batch_sizes,
        "inflight_grid": inflight_grid,
        "n_waves": n_waves,
        "warmup": warmup,
        "min_gain": min_gain,
    }
    started_at = utc_now()
    manifest = build_run_manifest(
        experiment_id=lab.EXPERIMENT_ID,
        execution_mode=lab.EXECUTION_MODE,
        resolved_config=resolved_config,
        repo_root=REPO_ROOT,
        source_paths=[
            Path(__file__),
            REPO_ROOT / "quartz" / "experiments" / "service_curve.py",
            config_path,
        ],
        argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
        started_at=started_at,
        assumptions=cfg["assumptions"],
        prohibited_inferences=cfg["prohibited_inferences"],
    )
    manifest_path = output_dir / "run_manifest.json"
    atomic_json_dump(manifest_path, manifest)

    model = lab.build_eval_model(
        in_ch=int(m["in_ch"]), channels=int(m["channels"]), blocks=int(m["blocks"]),
        board=int(m["board"]), actions=int(m["actions"]),
    ).to(device).eval()

    rows = lab.service_curve(
        model, device, batch_sizes=batch_sizes, inflight_grid=inflight_grid,
        in_ch=int(m["in_ch"]), board=int(m["board"]), n_waves=n_waves, warmup=warmup,
    )
    verdict = lab.scheduler_verdict(rows, min_gain=min_gain)

    curve_csv = output_dir / "service_curve.csv"
    summary_json = output_dir / "summary.json"
    write_csv(curve_csv, rows)
    atomic_json_dump(
        summary_json,
        {
            "format_version": 1,
            "experiment_id": lab.EXPERIMENT_ID,
            "execution_mode": lab.EXECUTION_MODE,
            "claim_status": "measured_gpu_service_curve",
            "config_sha256": resolved_config["config_sha256"],
            "device": str(device),
            "device_name": device_name,
            "model": m,
            "rows": rows,
            "scheduler_verdict": verdict,
            "note": (
                "Quality-free throughput/latency only (re-scoped H4). Power/energy are "
                "best-effort nvidia-smi samples, not a controlled power measurement."
            ),
        },
    )
    manifest = finalize_run_manifest(manifest, output_dir=output_dir, artifact_paths=[curve_csv, summary_json])
    atomic_json_dump(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "device_name": device_name,
                "best_overall_items_per_s": verdict["best_overall_items_per_s"],
                "best_overall_batch": verdict["best_overall_batch"],
                "best_overall_inflight": verdict["best_overall_inflight"],
                "inflight_throughput_gain": verdict["inflight_throughput_gain"],
                "h4_inflight_scheduler_lane_alive": verdict["h4_inflight_scheduler_lane_alive"],
                "run_contract_hash": manifest["run_contract_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
