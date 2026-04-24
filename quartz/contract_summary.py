from __future__ import annotations

import hashlib
import json


def stable_json_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def summarize_contract_collection(contracts: list[dict] | None, discarded: list[dict] | None = None, hash_key: str | None = None) -> dict:
    contracts = list(contracts or [])
    discarded = list(discarded or [])
    contract_hashes = sorted(stable_json_hash(row) for row in contracts)
    summary = {
        "count": len(contracts),
        "collection_hash": stable_json_hash(contract_hashes),
        "discarded_count": len(discarded),
        "legacy_partial_count": sum(1 for row in contracts if row.get("legacy_partial")),
    }
    if hash_key is not None:
        summary["hash_key"] = hash_key
    return summary


def summarize_named_contract_map(contract_map: dict | None, discarded: list[dict] | None = None, *, name_key: str) -> dict:
    rows = [
        {name_key: name, "contract": contract}
        for name, contract in sorted(dict(contract_map or {}).items())
    ]
    return summarize_contract_collection(rows, discarded, hash_key=name_key)


def summarize_plain_contracts(contracts: list[dict] | None) -> dict:
    return summarize_contract_collection(list(contracts or []), [], hash_key="stable_json_hash")
