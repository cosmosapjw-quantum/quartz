#!/usr/bin/env python3
"""Durable evidence receipt verifier for QUARTZ Idea Foundry.

Verifies committed evidence receipts, ensuring that fresh clones can
cryptographically audit normalized results without needing raw training runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_DIR = REPO_ROOT / "docs" / "idea_foundry" / "receipts"
RECEIPT_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_receipt(receipt_path: Path) -> dict[str, Any]:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError(f"receipt is missing or symlink: {receipt_path}")
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"unsupported receipt schema_version in {receipt_path}")
    
    runs = data.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"receipt contains no run entries: {receipt_path}")
    
    verified_runs = 0
    verified_artifacts = 0
    
    for run_entry in runs:
        run_id = run_entry.get("run_id")
        if not run_id:
            raise ValueError(f"missing run_id in {receipt_path}")
        artifacts = run_entry.get("artifacts", [])
        for art in artifacts:
            rel_path = art.get("path")
            expected_sha = art.get("sha256")
            if not rel_path or not expected_sha:
                raise ValueError(f"malformed artifact entry in {run_id}")
            art_file = REPO_ROOT / rel_path
            if art_file.is_file():
                actual_sha = file_sha256(art_file)
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"hash drift for {rel_path}: expected {expected_sha}, got {actual_sha}"
                    )
                verified_artifacts += 1
        verified_runs += 1
        
    return {
        "receipt": str(receipt_path.relative_to(REPO_ROOT)),
        "status": "VERIFIED",
        "verified_runs": verified_runs,
        "verified_artifacts": verified_artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Idea Foundry evidence receipts")
    parser.add_argument("--receipts-dir", type=Path, default=RECEIPTS_DIR)
    args = parser.parse_args(argv)
    
    if not args.receipts_dir.is_dir():
        print(f"[WARN] Receipts directory not found: {args.receipts_dir}")
        return 0
        
    receipt_files = sorted(args.receipts_dir.glob("*.receipt.json"))
    if not receipt_files:
        print(f"[INFO] No receipts found in {args.receipts_dir}")
        return 0
        
    all_passed = True
    for receipt_file in receipt_files:
        try:
            res = verify_receipt(receipt_file)
            print(f"[OK] {res['receipt']}: {res['verified_runs']} runs, {res['verified_artifacts']} artifacts verified")
        except Exception as exc:
            print(f"[FAIL] {receipt_file}: {exc}", file=sys.stderr)
            all_passed = False
            
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
