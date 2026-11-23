"""Stream frozen C24 tasks through six real native policies, preserving raw evidence.

Use venv/bin/python. Full cohorts require committed scientific inputs; --allow-dirty
is restricted to UNIT fixtures and records the complete dirty scope.
"""
import argparse
import datetime
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import struct
import subprocess
import sys
from c24_sources import configured_specs, sha, validate_configuration

POLICIES = ("PLAIN_STOP","PLAIN_ALWAYS","FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK","RESIDUAL")

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def to_wire(task):
    if set(task)!={"id","side","rho","scale","rhs"}:
        raise ValueError("public task must have exactly id/side/rho/scale/rhs; oracle fields forbidden")
    def scalar(value):
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
            raise ValueError("archive scientific value is not a finite binary64 number")
        return repr(float(value))
    return {"id":task["id"],"side":task["side"],"rho":task["rho"],
            "scale":scalar(task["scale"]),"rhs":[[scalar(v) for v in row] for row in task["rhs"]]}

def bits(value):
    return struct.unpack("<Q",struct.pack("<d",value))[0]

def validate_response(out, task, budget, repeats, trace):
    assert "error" not in out, out.get("error")
    assert out["id"]==task["id"] and out["additional"]==budget, "response/request identity"
    expected={(p,r) for p in POLICIES for r in range(repeats)}
    actual=[(r["policy"],r["rep"]) for r in out["results"]]
    assert len(actual)==len(expected) and set(actual)==expected, "policy/rep coverage or duplicate"
    if trace:
        assert out["input_rhs_bits"]==[[bits(v) for v in row] for row in task["rhs"]], "native RHS bits"
        assert out["input_scale_bits"]==bits(task["scale"]), "native scale bits"
        assert all("final_x" in r and "trace" in r for r in out["results"]), "trace state absent"
    else:
        assert all(not r.get("trace") and not r.get("final_x") for r in out["results"]), "timing copied trace state"

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],required=True)
    ap.add_argument("--mode",choices=["trace","timing"],required=True)
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    ap.add_argument("--allow-dirty",action="store_true")
    args=ap.parse_args()
    if args.allow_dirty and args.split!="UNIT":
        raise ValueError("--allow-dirty is limited to UNIT fixture integration")
    doc=pathlib.Path(__file__).resolve().parents[1];root=doc.parents[2]
    ev=(args.evidence_dir or doc/"evidence").resolve()
    config=json.loads((doc/"SOURCE_SPECS.json").read_text());validate_configuration(config)
    expected=[s["id"] for s in configured_specs(config,args.split)]
    binary=root/"target/release/quartz_c24_cheap_certificate"
    public=ev/f"C24_PUBLIC_{args.split}.jsonl.gz";source_manifest=ev/f"C24_SOURCES_{args.split}.json"
    output=ev/f"C24_{args.split}_{args.mode.upper()}.jsonl.gz"
    manifest=output.with_suffix("").with_suffix(".manifest.json")
    errlog=ev/f"C24_{args.split}_{args.mode.upper()}_STDERR.log"
    if any(p.exists() for p in [output,manifest,errlog]):
        raise FileExistsError("preserved run exists; never overwrite raw output, manifest or stderr")
    scientific=[root/"src/bin/quartz_c24_cheap_certificate.rs",doc/"SOURCE_SPECS.json",
        doc/"CODE_CONTRACT.json",doc/"C04_RUN_CONTRACT.json"]+sorted((doc/"coding").glob("c24_*.py"))
    if not args.allow_dirty:
        for p in scientific:
            subprocess.run(["git","ls-files","--error-unmatch",str(p.relative_to(root))],
                cwd=root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run(["git","diff","--exit-code","HEAD","--"]+[str(p.relative_to(root)) for p in scientific],
            cwd=root,check=True,stdout=subprocess.DEVNULL)
    sm=json.loads(source_manifest.read_text())
    assert sm["status"]=="completed" and sm["count"]==len(expected)
    assert sm["source_ids"]==expected
    assert sm["artifacts"][public.name]["sha256"]==sha(public), "source-manifest binding"
    assert sm["source_specs_sha256"]==sha(doc/"SOURCE_SPECS.json")
    tracked=scientific+[binary,public,source_manifest]
    def fingerprints():
        return {str(p.relative_to(root)) if p.is_relative_to(root) else str(p):sha(p) for p in tracked}
    before=fingerprints()
    def dirty():
        return subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],
            cwd=root,text=True).splitlines()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    repeats=config["repeats"] if args.mode=="timing" else 1
    warmup=config["warmup"] if args.mode=="timing" else 0
    result={"schema":"quartz.c24.run.v1","status":"running","split":args.split,"mode":args.mode,
        "argv":sys.argv,"cwd":str(root),"start_utc":utc(),"source_head":head,
        "dirty_scope_before":dirty(),"allow_dirty_unit":args.allow_dirty,"input_hashes_before":before,
        "binary_sha256":sha(binary),"public_source_sha256":sha(public),
        "scripts_sha256":{str(p.relative_to(root)):sha(p) for p in scientific if p.suffix==".py"},
        "python":sys.version,"packages":{p:importlib.metadata.version(p) for p in ["numpy","scipy"]},
        "platform":platform.platform(),"logical_cpus":os.cpu_count(),"load_start":list(os.getloadavg()),
        "environment":{k:os.environ.get(k,"") for k in ["RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS",
            "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]},
        "clock":"native Instant wall microseconds; all validation/readouts/solver work/destruction included; JSON excluded",
        "repeats":repeats,"warmup":warmup,"additional":config["additional"],
        "repeats_are_independent_parents":False,"wire_encoding":"f64 repr decimal strings; Rust FromStr; trace exact input bits",
        "oracle_fields_sent":False,"source_ids":[],"requests":0,"native_policy_executions":0,
        "warmup_policy_executions":0,"exit_code":None}
    process=None;failure=None
    try:
        with errlog.open("x") as err,output.open("xb") as raw:
            process=subprocess.Popen([str(binary)],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=err,text=True,bufsize=1)
            with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as dest,gzip.open(public,"rt") as inputs:
                for position,line in enumerate(inputs):
                    task=json.loads(line)
                    assert position<len(expected) and task["id"]==expected[position], "source order/coverage"
                    wire=to_wire(task)
                    for budget in config["additional"]:
                        request={"task":wire,"additional":budget,"repeats":repeats,
                            "warmup":warmup,"trace":args.mode=="trace"}
                        process.stdin.write(json.dumps(request,separators=(",",":"),allow_nan=False)+"\n")
                        process.stdin.flush();answer=process.stdout.readline()
                        if not answer:raise RuntimeError("native ended without response; preserved stderr")
                        out=json.loads(answer);validate_response(out,task,budget,repeats,args.mode=="trace")
                        envelope={"source_id":task["id"],"split":args.split,"mode":args.mode,"additional":budget,
                            "task_sha256":hashlib.sha256(json.dumps(task,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                            "native":out}
                        dest.write((json.dumps(envelope,separators=(",",":"),allow_nan=False)+"\n").encode())
                        result["requests"]+=1;result["native_policy_executions"]+=len(out["results"])
                        result["warmup_policy_executions"]+=len(POLICIES)*warmup
                    result["source_ids"].append(task["id"])
                    print(json.dumps({"source_completed":task["id"],"requests":result["requests"]}),flush=True)
            process.stdin.close()
            extra=process.stdout.read()
            result["exit_code"]=process.wait()
            assert result["exit_code"]==0 and not extra.strip(), "native exit or unsolicited output"
        assert result["source_ids"]==expected
        assert result["requests"]==len(expected)*len(config["additional"])
        assert fingerprints()==before,"inputs changed during run"
        result["status"]="completed"
    except BaseException as exc:
        failure=exc;result["status"]="failed";result["error"]=f"{type(exc).__name__}: {exc}"
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if process:result["exit_code"]=process.returncode
    finally:
        result["input_hashes_after"]=fingerprints()
        if result["input_hashes_after"]!=before:result["status"]="inputs-changed"
        result["dirty_scope_after"]=dirty();result["end_utc"]=utc();result["load_end"]=list(os.getloadavg())
        result["output"]=output.name
        result["output_sha256"]=sha(output) if output.exists() else None
        result["output_bytes"]=output.stat().st_size if output.exists() else 0
        result["stderr_sha256"]=sha(errlog) if errlog.exists() else None
        with manifest.open("x") as out:out.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result))
    if failure:raise failure
if __name__=="__main__":
    main()
