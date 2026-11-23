"""C25 lossless chunk IO and provenance helpers; no numerical algorithms."""
import datetime
import gzip
import hashlib
import json
import pathlib
import subprocess

MAX_CHUNK_BYTES = 45 * 1024 * 1024

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_manifest(path):
    path=pathlib.Path(path)
    manifest=json.loads(path.read_text())
    assert manifest["status"]=="completed" and manifest["exit_code"]==0,"completed native run required"
    assert manifest["input_hashes_before"]==manifest["input_hashes_after"],"run input identity"
    chunks=manifest["chunks"]
    assert len(manifest["source_ids"])==len(set(manifest["source_ids"])),"duplicate source chunks"
    assert chunks and len({c["path"] for c in chunks})==len(chunks),"nonempty distinct chunks"
    assert [c["source_id"] for c in chunks]==manifest["source_ids"],"ordered chunk/source coverage"
    for chunk in chunks:
        file=(path.parent/chunk["path"]).resolve()
        assert file.is_relative_to(path.parent.resolve()),"chunk path escapes evidence directory"
        assert file.stat().st_size==chunk["bytes"]<=MAX_CHUNK_BYTES,"chunk byte identity/cap"
        assert sha(file)==chunk["sha256"],"chunk byte hash"
    assert sum(c["requests"] for c in chunks)==manifest["requests"],"chunk request count"
    return manifest

def rows(path):
    path=pathlib.Path(path)
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path,"rt") as stream:
            for line in stream:
                yield json.loads(line)
        return
    manifest=load_manifest(path)
    for chunk in manifest["chunks"]:
        count=0
        with gzip.open(path.parent/chunk["path"],"rt") as stream:
            for line in stream:
                row=json.loads(line)
                assert row["source_id"]==chunk["source_id"],"chunk source identity"
                count+=1
                yield row
        assert count==chunk["requests"],"chunk row count"

def manifest_files(path):
    path=pathlib.Path(path)
    manifest=load_manifest(path)
    return [path]+[path.parent/c["path"] for c in manifest["chunks"]]

def fingerprints(paths,root=None):
    root=pathlib.Path(root).resolve() if root else None
    result={}
    for path in paths:
        p=pathlib.Path(path).resolve()
        key=str(p.relative_to(root)) if root and p.is_relative_to(root) else str(p)
        result[key]=sha(p)
    return result

def scientific_paths(root,doc):
    return [root/"src/bin/quartz_c24_cheap_certificate.rs",
            root/"src/bin/quartz_c25_incumbent_price.rs",doc/"SOURCE_SPECS.json",
            doc/"CODE_CONTRACT.json",doc/"C04_RUN_CONTRACT.json"]+sorted((doc/"coding").glob("c25_*.py"))

def assert_committed(root,paths):
    relative=[str(pathlib.Path(p).resolve().relative_to(root.resolve())) for p in paths]
    for path in relative:
        subprocess.run(["git","ls-files","--error-unmatch",path],cwd=root,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(["git","diff","--exit-code","HEAD","--"]+relative,cwd=root,check=True,
                   stdout=subprocess.DEVNULL)

def semantic_signature(row):
    """Actual deterministic scientific outputs; elapsed times and rep are omitted."""
    skip={"rep","total_us","validation_us","initial_us","solve_us","bounds_us","trace","final_x"}
    value={k:v for k,v in row.items() if k not in skip}
    value["c25"]={k:v for k,v in row["c25"].items() if k not in ("continuation_us","decisions")}
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
