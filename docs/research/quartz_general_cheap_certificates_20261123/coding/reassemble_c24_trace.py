"""Losslessly split/reassemble C24 raw gzip bytes with deterministic SHA-256 manifests.

Parts are raw consecutive slices, not separately compressed gzip streams.
The 45 MiB ceiling is 47,185,920 bytes. Existing parts, manifests and assembly
outputs are never overwritten. Packing never removes or modifies the raw gzip.

Examples (from repository root):
  venv/bin/python .../reassemble_c24_trace.py pack --split TEST
  venv/bin/python .../reassemble_c24_trace.py assemble --split TEST
  venv/bin/python .../reassemble_c24_trace.py assemble --split TRAIN --output /tmp/c24_train_check.jsonl.gz
"""
import argparse
import hashlib
import json
import os
import pathlib
import tempfile

PART_BYTES = 45 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024
SCHEMA = "quartz.c24.trace-byte-parts.v1"

def digest(path):
    size = 0
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(BLOCK_BYTES), b""):
            result.update(chunk)
            size += len(chunk)
    return size, result.hexdigest()

def require_absent(paths):
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"Preserved output exists; refusing overwrite: {path}")

def names(split):
    return f"C24_{split}_TRACE.jsonl.gz", f"C24_{split}_TRACE.parts.json"

def pack(evidence, split):
    original_name, manifest_name = names(split)
    source = evidence / original_name
    source_bytes, source_sha = digest(source)
    if source_bytes == 0:
        raise ValueError("Empty input is not a raw trace gzip")
    count = (source_bytes + PART_BYTES - 1) // PART_BYTES
    part_names = [f"C24_{split}_TRACE.part{index:02d}" for index in range(1, count + 1)]
    targets = [evidence / name for name in part_names] + [evidence / manifest_name]
    require_absent(targets)
    run_manifest = evidence / f"C24_{split}_TRACE.manifest.json"
    if run_manifest.exists():
        run = json.loads(run_manifest.read_text())
        if run["output_sha256"] != source_sha or run["output_bytes"] != source_bytes:
            raise ValueError("Raw trace differs from its preserved native-run manifest")
    manifest = {
        "schema": SCHEMA, "split": split, "original_name": original_name,
        "original_bytes": source_bytes, "original_sha256": source_sha,
        "max_part_bytes": PART_BYTES,
        "encoding": "ordered raw byte slices of original gzip; concatenate without decompression",
        "parts": [],
    }
    # Same-filesystem staging allows atomic, no-clobber hard-link publication.
    with tempfile.TemporaryDirectory(prefix=".c24_parts_", dir=evidence) as temp:
        staging = pathlib.Path(temp)
        copied = hashlib.sha256()
        total = 0
        with source.open("rb") as stream:
            for index, name in enumerate(part_names, 1):
                chunk = stream.read(PART_BYTES)
                if not chunk:
                    raise ValueError("Raw trace shortened while splitting")
                (staging / name).write_bytes(chunk)
                total += len(chunk)
                copied.update(chunk)
                manifest["parts"].append({
                    "index": index, "name": name, "bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                })
            if stream.read(1):
                raise ValueError("Raw trace grew while splitting")
        if total != source_bytes or copied.hexdigest() != source_sha:
            raise ValueError("Copied bytes differ from original identity")
        if digest(source) != (source_bytes, source_sha):
            raise ValueError("Raw trace changed during packing")
        (staging / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        for target in targets:
            os.link(staging / target.name, target)
    return manifest

def load_manifest(evidence, split):
    original_name, manifest_name = names(split)
    manifest = json.loads((evidence / manifest_name).read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("split") != split
            or manifest.get("original_name") != original_name
            or manifest.get("max_part_bytes") != PART_BYTES):
        raise ValueError("Manifest schema/split/original/part-size contract mismatch")
    parts = manifest["parts"]
    if not parts or not isinstance(manifest["original_bytes"], int) or manifest["original_bytes"] <= 0:
        raise ValueError("Invalid part population or original size")
    expected_count = (manifest["original_bytes"] + PART_BYTES - 1) // PART_BYTES
    if len(parts) != expected_count:
        raise ValueError("Part count does not cover whole byte size")
    for index, part in enumerate(parts, 1):
        expected_name = f"C24_{split}_TRACE.part{index:02d}"
        expected_size = min(PART_BYTES, manifest["original_bytes"] - (index - 1) * PART_BYTES)
        if (part["index"] != index or part["name"] != expected_name
                or part["bytes"] != expected_size):
            raise ValueError("Part order/name/size contract mismatch")
        if len(part["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in part["sha256"]):
            raise ValueError("Invalid part SHA-256")
    if sum(part["bytes"] for part in parts) != manifest["original_bytes"]:
        raise ValueError("Part sizes do not sum to original size")
    return manifest

def assemble(evidence, split, output=None):
    manifest = load_manifest(evidence, split)
    output = output or evidence / manifest["original_name"]
    require_absent([output])
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=".c24_assemble_", suffix=".tmp", delete=False) as dest:
            temporary = pathlib.Path(dest.name)
            whole = hashlib.sha256()
            total = 0
            for part in manifest["parts"]:
                source = evidence / part["name"]
                if source.stat().st_size != part["bytes"]:
                    raise ValueError(f"Part size mismatch: {source.name}")
                part_hash = hashlib.sha256()
                part_size = 0
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(BLOCK_BYTES), b""):
                        dest.write(chunk)
                        whole.update(chunk)
                        part_hash.update(chunk)
                        part_size += len(chunk)
                if part_size != part["bytes"] or part_hash.hexdigest() != part["sha256"]:
                    raise ValueError(f"Part byte identity mismatch: {source.name}")
                total += part_size
            if total != manifest["original_bytes"] or whole.hexdigest() != manifest["original_sha256"]:
                raise ValueError("Reassembled whole byte identity mismatch")
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "bytes": total, "sha256": whole.hexdigest(),
            "parts_verified": len(manifest["parts"]), "whole_verified": True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "assemble"])
    parser.add_argument("--split", choices=["TRAIN", "CAL", "TEST"], required=True)
    parser.add_argument("--evidence-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path,
                        help="assembly output; default is original gzip path, which must be absent")
    args = parser.parse_args()
    evidence = args.evidence_dir or pathlib.Path(__file__).resolve().parents[1] / "evidence"
    if args.action == "pack":
        if args.output is not None:
            parser.error("--output applies only to assemble")
        result = pack(evidence, args.split)
    else:
        result = assemble(evidence, args.split, args.output)
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
