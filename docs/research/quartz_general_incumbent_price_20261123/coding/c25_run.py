"""Run frozen C25 sources through six native policies with lossless parent chunks.

Full cohort runs require committed scientific code. --allow-dirty applies only to
UNIT. TRAIN sends prediction zero; CAL/TEST require the committed TRAIN predictor.
"""
import argparse
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
from c25_sources import configured_specs, validate_configuration
from c25_io import MAX_CHUNK_BYTES, assert_committed, fingerprints, scientific_paths, sha, utc

POLICIES=("PLAIN_STOP","PLAIN_ALWAYS","FAST_BLOCK","RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME")

def bits(value):
    return struct.unpack("<Q",struct.pack("<d",value))[0]

def to_wire(task):
    if set(task)!={"id","side","rho","scale","rhs"}:
        raise ValueError("public task must have exactly id/side/rho/scale/rhs; oracle fields forbidden")
    def scalar(value):
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
            raise ValueError("scientific value is not a finite binary64 number")
        return repr(float(value))
    return {"id":task["id"],"side":task["side"],"rho":task["rho"],"scale":scalar(task["scale"]),
            "rhs":[[scalar(v) for v in row] for row in task["rhs"]]}

def validate_response(out,task,budget,repeats,trace):
    assert "error" not in out,out.get("error")
    assert out["id"]==task["id"] and out["additional"]==budget,"response/request identity"
    expected={(p,r) for p in POLICIES for r in range(repeats)}
    actual=[(r["policy"],r["rep"]) for r in out["results"]]
    assert len(actual)==len(expected) and set(actual)==expected,"policy/repeat coverage or duplicate"
    assert all("c25" in r for r in out["results"]),"C25 metadata missing"
    if trace:
        assert out["input_rhs_bits"]==[[bits(v) for v in row] for row in task["rhs"]],"native RHS bits"
        assert out["input_scale_bits"]==bits(task["scale"]),"native scale bits"
        assert all(r.get("trace") and r.get("final_x") for r in out["results"]),"trace state absent"
        assert all(r["c25"].get("decisions") for r in out["results"] if r["policy"].startswith("RETAIN_"))
    else:
        assert all(not r.get("trace") and not r.get("final_x") and not r["c25"].get("decisions") for r in out["results"]),"timing contains trace state"

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],required=True)
    ap.add_argument("--mode",choices=["trace","timing"],required=True)
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    ap.add_argument("--allow-dirty",action="store_true")
    ap.add_argument("--unit-predicted-us",type=float,default=1.0)
    args=ap.parse_args()
    if args.allow_dirty and args.split!="UNIT":
        raise ValueError("--allow-dirty is limited to UNIT integration")
    if not math.isfinite(args.unit_predicted_us) or args.unit_predicted_us<0:
        raise ValueError("UNIT prediction must be finite and nonnegative")
    doc=pathlib.Path(__file__).resolve().parents[1];root=doc.parents[2]
    ev=(args.evidence_dir or doc/"evidence").resolve();ev.mkdir(parents=True,exist_ok=True)
    config=json.loads((doc/"SOURCE_SPECS.json").read_text());validate_configuration(config)
    expected=[s["id"] for s in configured_specs(config,args.split)]
    binary=root/"target/release/quartz_c25_incumbent_price"
    public=ev/f"C25_PUBLIC_{args.split}.jsonl.gz";source_manifest=ev/f"C25_SOURCES_{args.split}.json"
    prefix=f"C25_{args.split}_{args.mode.upper()}"
    chunk_dir=ev/prefix;manifest=ev/(prefix+".manifest.json");errlog=ev/(prefix+"_STDERR.log")
    if any(p.exists() for p in [chunk_dir,manifest,errlog]):
        raise FileExistsError("preserved run exists; never overwrite chunks, manifest or stderr")
    scientific=scientific_paths(root,doc)
    predictor_path=None;predictor=None
    if args.split in ("CAL","TEST"):
        predictor_path=ev/config["predictor"]["artifact"]
        predictor=json.loads(predictor_path.read_text())
        assert predictor["status"]=="TRAIN_ONLY_FROZEN_POINT_ESTIMATE"
        assert predictor["source_specs_sha256"]==sha(doc/"SOURCE_SPECS.json")
        assert predictor["input_hashes_before"]==predictor["input_hashes_after"]
        assert not predictor["oracle_fields_used"] and not predictor["heldout_inputs_used"]
        assert predictor["fit_policy"]=="RETAIN_BLOCK" and predictor["fit_additional"]==32
        assert predictor["training_parent_ids"]==[s["id"] for s in config["sources"] if s["split"]=="TRAIN"]
        scientific.append(predictor_path)
    if not args.allow_dirty:
        assert_committed(root,scientific)
    source=json.loads(source_manifest.read_text())
    assert source["status"]=="completed" and source["source_ids"]==expected and source["count"]==len(expected)
    assert source["artifacts"][public.name]["sha256"]==sha(public)
    assert source["source_specs_sha256"]==sha(doc/"SOURCE_SPECS.json")
    tracked=scientific+[binary,public,source_manifest];before=fingerprints(tracked,root)
    def dirty():
        return subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],cwd=root,text=True).splitlines()
    repeats=config["repeats"] if args.mode=="timing" else 1
    warmup=config["warmup"] if args.mode=="timing" else 0
    result={"schema":"quartz.c25.run.v1","status":"running","split":args.split,"mode":args.mode,
            "argv":sys.argv,"cwd":str(root),"start_utc":utc(),
            "source_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
            "dirty_scope_before":dirty(),"allow_dirty_unit":args.allow_dirty,"input_hashes_before":before,
            "binary_sha256":sha(binary),"public_source_sha256":sha(public),
            "predictor_sha256":sha(predictor_path) if predictor_path else None,
            "prediction_source":"FROZEN_TRAIN" if predictor else "TRAIN_DISABLED_ZERO" if args.split=="TRAIN" else "UNIT_EXPLICIT",
            "python":sys.version,"packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy")},
            "platform":platform.platform(),"logical_cpus":os.cpu_count(),"load_start":list(os.getloadavg()),
            "environment":{k:os.environ.get(k,"") for k in ("RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
            "clock":"native full Instant wall microseconds, validation/initial/readouts/controller/destruction included; JSON excluded",
            "repeats":repeats,"warmup":warmup,"additional":config["additional"],
            "repeats_are_independent_parents":False,"wire_encoding":"f64 repr decimal strings; Rust FromStr; exact trace bits",
            "oracle_fields_sent":False,"source_ids":[],"requests":0,"native_policy_executions":0,
            "warmup_policy_executions":0,"exit_code":None,"chunks":[],"max_chunk_bytes":MAX_CHUNK_BYTES}
    process=None;failure=None;chunk_dir.mkdir()
    try:
        with errlog.open("x") as err:
            process=subprocess.Popen([str(binary)],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
            with gzip.open(public,"rt") as inputs:
                for position,line in enumerate(inputs):
                    task=json.loads(line)
                    assert position<len(expected) and task["id"]==expected[position],"source order/coverage"
                    wire=to_wire(task)
                    prediction=0.0 if args.split=="TRAIN" else args.unit_predicted_us if args.split=="UNIT" else predictor["by_side"][str(task["side"])]["predicted_block_us"]
                    assert math.isfinite(prediction) and prediction>=0
                    if args.split in ("CAL","TEST"):
                        assert prediction>0
                    chunk=chunk_dir/f"{position:03d}_{task['id']}.jsonl.gz"
                    descriptor={"path":str(chunk.relative_to(ev)),"source_id":task["id"],"requests":0}
                    result["chunks"].append(descriptor)
                    with chunk.open("xb") as raw:
                        with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as dest:
                            for budget in config["additional"]:
                                request={"task":wire,"additional":budget,"repeats":repeats,"warmup":warmup,
                                         "trace":args.mode=="trace","predicted_block_us":prediction}
                                process.stdin.write(json.dumps(request,separators=(",",":"),allow_nan=False)+"\n")
                                process.stdin.flush();answer=process.stdout.readline()
                                if not answer:
                                    raise RuntimeError("native ended without response; inspect preserved stderr")
                                out=json.loads(answer);validate_response(out,task,budget,repeats,args.mode=="trace")
                                envelope={"source_id":task["id"],"split":args.split,"mode":args.mode,"additional":budget,
                                          "predicted_block_us":prediction,
                                          "task_sha256":hashlib.sha256(json.dumps(task,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                                          "native":out}
                                dest.write((json.dumps(envelope,separators=(",",":"),allow_nan=False)+"\n").encode())
                                descriptor["requests"]+=1;result["requests"]+=1
                                result["native_policy_executions"]+=len(out["results"])
                                result["warmup_policy_executions"]+=len(POLICIES)*warmup
                    descriptor.update(sha256=sha(chunk),bytes=chunk.stat().st_size)
                    assert descriptor["bytes"]<=MAX_CHUNK_BYTES,"parent chunk exceeds publication cap"
                    result["source_ids"].append(task["id"])
                    print(json.dumps({"source_completed":task["id"],"requests":result["requests"],"chunk_bytes":descriptor["bytes"]}),flush=True)
            process.stdin.close();extra=process.stdout.read();result["exit_code"]=process.wait()
            assert result["exit_code"]==0 and not extra.strip(),"native exit or unsolicited output"
        assert result["source_ids"]==expected and result["requests"]==len(expected)*len(config["additional"])
        assert fingerprints(tracked,root)==before,"inputs changed during run"
        result["status"]="completed"
    except BaseException as exc:
        failure=exc;result["status"]="failed";result["error"]=f"{type(exc).__name__}: {exc}"
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        if process:
            result["exit_code"]=process.returncode
    finally:
        for descriptor in result["chunks"]:
            chunk=ev/descriptor["path"]
            if chunk.exists():
                descriptor.update(sha256=sha(chunk),bytes=chunk.stat().st_size)
        result["input_hashes_after"]=fingerprints(tracked,root)
        if result["input_hashes_after"]!=before:
            result["status"]="inputs-changed"
        result["dirty_scope_after"]=dirty();result["end_utc"]=utc();result["load_end"]=list(os.getloadavg())
        result["stderr_sha256"]=sha(errlog) if errlog.exists() else None
        with manifest.open("x") as stream:
            stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"manifest":str(manifest),"status":result["status"],"requests":result["requests"],
                      "native_policy_executions":result["native_policy_executions"],"exit_code":result["exit_code"]}))
    if failure:
        raise failure

if __name__=="__main__":
    main()
