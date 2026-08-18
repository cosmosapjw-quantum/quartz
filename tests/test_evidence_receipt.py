"""Test suite for durable evidence receipt verification."""

import json
from pathlib import Path
import pytest
from scripts.verify_evidence_receipt import verify_receipt, RECEIPTS_DIR, REPO_ROOT


def test_committed_evidence_receipts_are_valid() -> None:
    if not RECEIPTS_DIR.is_dir():
        return
    receipt_files = sorted(RECEIPTS_DIR.glob("*.receipt.json"))
    assert receipt_files, "expected at least one evidence receipt"
    for receipt_file in receipt_files:
        res = verify_receipt(receipt_file)
        assert res["status"] == "VERIFIED"
        assert res["verified_runs"] == 2
        assert res["verified_artifacts"] >= 8


def test_missing_declared_artifact_fails_closed(tmp_path: Path) -> None:
    bad_receipt = tmp_path / "bad.receipt.json"
    bad_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": "test_run",
                        "artifacts": [
                            {
                                "path": "nonexistent/file/path.json",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared artifact missing"):
        verify_receipt(bad_receipt)


def test_hash_drift_fails_closed(tmp_path: Path) -> None:
    bad_receipt = tmp_path / "bad.receipt.json"
    bad_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": "test_run",
                        "artifacts": [
                            {
                                "path": "configs/idea_foundry.studies.v1.json",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash drift"):
        verify_receipt(bad_receipt)


def test_unresolvable_git_commit_fails_closed(tmp_path: Path) -> None:
    bad_receipt = tmp_path / "bad.receipt.json"
    bad_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "git_head": "0123456789abcdef0123456789abcdef01234567",
                "runs": [
                    {
                        "run_id": "test_run",
                        "artifacts": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="git_head .* cannot be resolved"):
        verify_receipt(bad_receipt)
