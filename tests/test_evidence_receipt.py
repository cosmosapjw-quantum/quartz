"""Test suite for durable evidence receipt verification."""

from pathlib import Path
from scripts.verify_evidence_receipt import verify_receipt, RECEIPTS_DIR


def test_committed_evidence_receipts_are_valid() -> None:
    if not RECEIPTS_DIR.is_dir():
        return
    receipt_files = sorted(RECEIPTS_DIR.glob("*.receipt.json"))
    assert receipt_files, "expected at least one evidence receipt"
    for receipt_file in receipt_files:
        res = verify_receipt(receipt_file)
        assert res["status"] == "VERIFIED"
        assert res["verified_runs"] > 0
        assert res["verified_artifacts"] > 0
