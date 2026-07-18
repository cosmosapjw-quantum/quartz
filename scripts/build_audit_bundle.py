#!/usr/bin/env python3
"""Build a reproducible external-audit bundle for install + ablation review."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import os
import shutil
import zipfile
from pathlib import Path


ROOT_FILES = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "Cargo.toml",
    "Cargo.lock",
    "phase15_strategy_revision_v2.md",
]

DOC_FILES = [
    "docs/SETUP.md",
    "docs/INSTALL.md",
    "docs/QUICKSTART.md",
    "docs/TRAINING_GUIDE.md",
    "docs/ABLATION_GUIDE.md",
    "docs/RESEARCH_READINESS.md",
    "docs/QUARTZ_THEORY.md",
    "docs/GOMOCUP_BRAIN.md",
    "docs/TT_NOTES.md",
]

SCRIPT_FILES = [
    "scripts/ablation_study.py",
    "scripts/build_audit_bundle.py",
    "scripts/build_gomocup_brain.sh",
    "scripts/controller_optuna.py",
    "scripts/controller_sweep.py",
    "scripts/evaluator_calibration.py",
    "scripts/phase15_ablation_study.py",
    "scripts/phase15_benchmark.py",
    "scripts/phase15_benchmark_ci_smoke.py",
    "scripts/phase15_mine_suite.py",
    "scripts/phase15_online_ablation.py",
    "scripts/run_ablation_significant.sh",
    "scripts/smoke_e2e.py",
]

TEST_FILES = [
    "tests/fixtures/regression_positions.json",
    "tests/test_ablation_study.py",
    "tests/test_batch_protocol.py",
    "tests/test_controller_optuna.py",
    "tests/test_controller_sweep.py",
    "tests/test_evaluation_pipeline_regressions.py",
    "tests/test_phase15_ablation.py",
    "tests/test_training_pipeline_regressions.py",
]

EXTRA_FILES = [
    ".github/workflows/phase15-benchmark-gate.yml",
]

TREE_DIRS = [
    "src",
    "quartz",
    "configs",
]

IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.so",
    ".DS_Store",
    "quartz/static/play/*",
]


def should_ignore(rel_path: str) -> bool:
    normalized = rel_path.replace(os.sep, "/")
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def copy_file(repo_root: Path, stage_root: Path, relative: str) -> None:
    src = repo_root / relative
    if not src.exists():
        raise FileNotFoundError(relative)
    dst = stage_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(repo_root: Path, stage_root: Path, relative: str) -> None:
    src_root = repo_root / relative
    if not src_root.exists():
        raise FileNotFoundError(relative)
    for path in sorted(src_root.rglob("*")):
        if path.is_dir():
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        if should_ignore(rel_path):
            continue
        copy_file(repo_root, stage_root, rel_path)


def iter_stage_files(stage_root: Path) -> list[str]:
    return sorted(
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file()
    )


def write_manifest(stage_root: Path, bundle_name: str) -> None:
    files = iter_stage_files(stage_root)
    manifest_text = "\n".join(
        [
            f"bundle={bundle_name}",
            "scope=install + rust build + training-level ablation + supporting controller/phase15 audit surface",
            "includes=root metadata, rust source, python source, configs, selected scripts, selected docs, targeted tests",
            "reproducibility=source-level bundle; external dependency resolution is still required for cargo/pip install",
            "excludes=target/ venv/ results/ tmp/ artifacts/ caches/ gui static assets",
            "",
            "Top-level contents:",
            *files,
            "",
        ]
    )
    (stage_root / "AUDIT_PACKAGE_MANIFEST.txt").write_text(
        manifest_text, encoding="utf-8"
    )
    (stage_root / "FILELIST.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def build_bundle(repo_root: Path, stage_root: Path) -> None:
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    for relative in ROOT_FILES + DOC_FILES + SCRIPT_FILES + TEST_FILES + EXTRA_FILES:
        copy_file(repo_root, stage_root, relative)
    for relative in TREE_DIRS:
        copy_tree(repo_root, stage_root, relative)

    write_manifest(stage_root, stage_root.name)


def zip_stage(stage_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in iter_stage_files(stage_root):
            zf.write(stage_root / rel_path, arcname=rel_path)


def parse_args() -> argparse.Namespace:
    today = dt.datetime.now().strftime("%Y%m%d")
    default_name = f"quartz_external_audit_minimal_{today}"
    parser = argparse.ArgumentParser(description="Build the external audit zip bundle.")
    parser.add_argument("--name", default=default_name)
    parser.add_argument(
        "--stage-dir", default=None, help="Optional explicit staging directory"
    )
    parser.add_argument("--output", default=None, help="Optional explicit zip path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    stage_root = Path(args.stage_dir) if args.stage_dir else Path("/tmp") / args.name
    zip_path = (
        Path(args.output)
        if args.output
        else repo_root / "artifacts" / f"{args.name}.zip"
    )
    build_bundle(repo_root, stage_root)
    zip_stage(stage_root, zip_path)
    print(f"Stage: {stage_root}")
    print(f"Zip:   {zip_path}")
    print(f"Files: {len(iter_stage_files(stage_root))}")


if __name__ == "__main__":
    main()
