#!/usr/bin/env python3
"""Real finite-budget Rapfi teacher, train+validation ONLY. Resume checked raw ledger."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time
from collections import Counter

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def import_teacher(p):
 spec=importlib.util.spec_from_file_location('c14_rapfi',p)
 mod=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=mod
 spec.loader.exec_module(mod)
 return mod

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--stage",choices=("trainval","heldout"),required=True)
 args=parser.parse_args()
 d=Path(__file__).resolve().parent
 home=d.parent
 repo=next(p for p in d.parents if (p/'.git').exists())
 # This caller itself must have been committed and published before reading any labels.
 if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
    raise RuntimeError('refuse label unblinding on unpublished or dirty source tree')
 head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
 remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','--heads','origin',branch],text=True).split()[0]
 if head!=remote:raise RuntimeError('teacher source commit not remote-confirmed')
 frozen=home/"COHORT_224_FROZEN.json"
 manifest=json.loads((home/"C04_FREEZE_MANIFEST.json").read_text())
 if manifest["source_file_sha256"]!=sha(frozen):raise RuntimeError("cohort tampered")
 cohort=json.loads(frozen.read_text())
 lock=json.loads((d/"C01_INPUT_LOCK.json").read_text())
 refs={x["name"]:x for x in lock["source"]}
 for name in ("Rapfi_executable","Rapfi_config","Rapfi_classical_model","Rapfi_NNUE_freestyle"):
    x=refs[name]
    if sha(x["path"])!=x["sha256"]:raise RuntimeError("teacher weight identity changed "+name)
 teacher=import_teacher(repo/"docs/research/cognitive_metareasoning_20261120/highbranch_transfer_14/coding/rapfi_teacher14.py")
 executable=Path(refs["Rapfi_executable"]["path"])
 if args.stage=="trainval":
    requested=("train","validation")
    outdir=home/"evidence"/"teacher_train_validation"
 else:
    requested=("test_natural","test_tactical")
    outdir=home/"evidence"/"teacher_heldout"
    model=home/"evidence"/"model"/"MODEL_TRAINED.json"
    if not model.exists():raise RuntimeError("frozen trained model absent")
    model_in_git=subprocess.run(['git','-C',str(repo),'cat-file','-e',
        'HEAD:'+str(model.relative_to(repo))],check=False)
    if model_in_git.returncode!=0:raise RuntimeError("model not published before TEST labels")
    m=json.loads(model.read_text())
    if m["schema"]!="quartz_cog05_loglinear_policy13_v1" or m["heldout_test_teacher_labels_used"]:
       raise RuntimeError("incorrect model before test labels")
    freeze=home/"evidence"/"model"/"MODEL_C07_FREEZE.json"
    if freeze.exists() and json.loads(freeze.read_text())["model_sha256"]!=sha(model):
       raise RuntimeError("frozen trained model bytes changed")
 outdir.mkdir(parents=True,exist_ok=True)
 raw=outdir/"PROGRESS.jsonl"
 expected={x["id"]:(split,x) for split in requested for x in cohort["splits"][split]}
 completed={}
 if raw.exists():
    for line in raw.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row["id"] not in expected or row["history"]!=expected[row["id"]][1]["history"]:
           raise RuntimeError("stale or tampered teacher ledger")
        if row["id"] in completed:raise RuntimeError("duplicate teacher ledger entry")
        completed[row["id"]]=row
 pending=[(split,src) for split in requested for src in cohort["splits"][split]
          if src["id"] not in completed]
 print("TEACHER_START",args.stage,"already",len(completed),"pending",len(pending),flush=True)
 for split,src in pending:
    r=teacher.query_one(executable,executable.parent,src["history"],0,300,hard_deadline_s=20)
    out={"id":src["id"],"split":split,"history":src["history"],
        "label":r.get("flat_action"),"status":r["status"],
        "engine_elapsed_ms":r["elapsed_wall_ms"],"engine_weight_name":r.get("loaded_expected_weight_name"),
        "engine_verified_pretrained":r.get("compatible_weight_reported"),
        "source_seed":src["source_seed"],"teacher_budget_ms":300,
        "teacher_is_finite_search":True}
    if r["status"]!="OK" or not r.get("saw_loaded_nnue") or out["label"] in src["history"]:
        print("TEACHER_ERROR",str(out),str(r)[:550],flush=True)
        raise RuntimeError("teacher output not verifiably legal / pretrained")
    with raw.open("a") as f:
        f.write(json.dumps(out,sort_keys=True)+"\n");f.flush()
    completed[src["id"]]=out
    if len(completed)%16==0:print("TEACHER_PROGRESS",len(completed),"/",len(expected),flush=True)
 if set(completed)!=set(expected):raise RuntimeError("missing teacher label")
 payload={"schema":"quartz_cog05_Rapfi250615_supervised_teacher_v1",
          "label_source_git_commit":head,
          "cohort_sha256":sha(frozen),
          "Rapfi_binary_sha256":refs["Rapfi_executable"]["sha256"],
          "Rapfi_NNUE_freestyle_sha256":refs["Rapfi_NNUE_freestyle"]["sha256"],
          "rule":0,"teacher_budget_ms":300,
          "teacher_exact_minimax":False,
          "train_validation_only":args.stage=="trainval",
          "model_published_before_test":args.stage=="heldout",
          "splits":{split:[completed[s["id"]] for s in cohort["splits"][split]] for split in requested},
          "source_progress_sha256":sha(raw)}
 target=outdir/("TEACHER_TRAIN_VALIDATION.json" if args.stage=="trainval" else "TEACHER_HELDOUT.json")
 target.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
 print("TEACHER_COMPLETE",json.dumps({"split":args.stage,"total":len(completed),
         "label_sha256":sha(target),"mean_wall_ms":sum(v["engine_elapsed_ms"] for v in completed.values())/len(completed)}),flush=True)

if __name__=="__main__":main()
