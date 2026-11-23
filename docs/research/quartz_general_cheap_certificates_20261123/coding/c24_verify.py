"""Independent exact C24 source, interval, transport, fallback and residual checks.

No source-generator, solver or runner imports. Fractions use exact stored binary64
values and endpoint comparisons never widen by a tolerance. Large-vector true
residual checks use the frozen C04 TEST subset once per stored state in trace mode.
"""
import argparse
import gzip
import hashlib
import json
import math
import pathlib
import struct
import sys
from fractions import Fraction as F

POLICIES=("PLAIN_STOP","PLAIN_ALWAYS","FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK","RESIDUAL")
GATES=("FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK")
TIMES=("total_us","validation_us","initial_us","solve_us","bounds_us")

def rows(path):
    with gzip.open(path,"rt") as stream:
        for line in stream:yield json.loads(line)

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def bits(value):
    return struct.unpack("<Q",struct.pack("<d",value))[0]

def keyed(path):
    result={}
    for row in rows(path):
        assert row["id"] not in result,("duplicate source ID",row["id"])
        result[row["id"]]=row
    return result

def verify_sources(public,oracles):
    assert public.keys()==oracles.keys(),"public/oracle identity coverage"
    count=components=actions=0
    for ident,task in public.items():
        oracle=oracles[ident];p=oracle["dyadic_exponent"];n=task["side"];rho=task["rho"];scale=F(task["scale"])
        assert task["id"]==oracle["id"]==ident
        assert set(task)=={"id","side","rho","scale","rhs"},"oracle or unknown field in public task"
        assert isinstance(n,int) and 2<=n<=128 and isinstance(rho,int) and 0<=rho<=64
        assert math.isfinite(task["scale"]) and task["scale"]>=sys.float_info.min
        assert math.frexp(task["scale"])[0]==0.5
        assert p==30,"manufactured dyadic exponent"
        assert len(task["rhs"])==len(oracle["x_num"])==len(oracle["u_exact"])==3,"complete K=3 action coverage"
        qvals=list(map(F,oracle["u_exact"]))
        assert all(0<=q<=1 for q in qvals)
        assert oracle["optimal_actions"]==[i for i,q in enumerate(qvals) if q==max(qvals)]
        assert F(oracle["exact_gap"])==sorted(qvals,reverse=True)[0]-sorted(qvals,reverse=True)[1]
        for x,b,q in zip(oracle["x_num"],task["rhs"],qvals):
            assert len(x)==len(b)==n*n,"full component coverage"
            assert all(isinstance(v,int) and not isinstance(v,bool) for v in x)
            assert all(math.isfinite(v) for v in b)
            # Independent edge-energy form includes each internal edge once,
            # plus the missing-neighbour Dirichlet boundary contribution.
            energy=sum(z*z for z in x)
            for i,z in enumerate(x):
                degree=0
                if i%n+1<n:energy+=rho*(z-x[i+1])**2
                if i+n<n*n:energy+=rho*(z-x[i+n])**2
                if i%n:degree+=1
                if i%n+1<n:degree+=1
                if i>=n:degree+=1
                if i+n<n*n:degree+=1
                energy+=rho*(4-degree)*z*z
                adjacent=[j for j in [i-1,i+1,i-n,i+n]
                    if 0<=j<n*n and abs(i//n-j//n)+abs(i%n-j%n)==1]
                numerator=(1+4*rho)*z-rho*sum(x[j] for j in adjacent)
                bn,bd=b[i].as_integer_ratio()
                assert bn*(1<<p)==numerator*bd,(ident,"stored RHS mismatch",i)
                components+=1
            assert F(energy,1<<(2*p))/scale==q,(ident,"exact edge-energy oracle mismatch")
            assert sum(F(v)**2 for v in b)<=scale
            actions+=1
        count+=1
    return {"parents":count,"actions":actions,"exact_rhs_components":components,
        "oracle":"independent exact Dirichlet edge energy"}

def residual_square(task,b,x):
    side=task["side"];rho=F(task["rho"]);xx=list(map(F,x))
    assert len(xx)==len(b)==side*side
    residual=[]
    for i,(bi,xi) in enumerate(zip(b,xx)):
        adjacent=[j for j in [i-1,i+1,i-side,i+side]
            if 0<=j<len(xx) and abs(i//side-j//side)+abs(i%side-j%side)==1]
        ax=(1+4*rho)*xi-rho*sum((xx[j] for j in adjacent),F(0))
        residual.append(F(bi)-ax)
    return sum((r*r for r in residual),F(0)),sum((F(v)**2 for v in b),F(0))

def final_plain(result):
    return result["policy"] in ("PLAIN_STOP","PLAIN_ALWAYS") or (
        result["policy"]=="FAST_ONCE" and not result["certified"])

def diagnostic_contract(result,k):
    assert len(result["residual_relative"])==k
    assert all(math.isfinite(v) and v>=0 for v in result["residual_relative"])
    if final_plain(result):
        assert result["residual_upper"] is None and result["residual_goal_met"] is None,"plain residual is unverified"
        assert all(v["lo"]==0.0 and v["hi"]==1.0 for v in result["final_bounds"])
    else:
        upper=result["residual_upper"]
        assert isinstance(upper,list) and len(upper)==k
        assert all(math.isfinite(v) and v>=0 for v in upper)
        assert isinstance(result["residual_goal_met"],bool)
        assert result["residual_goal_met"]==all(v<=1e-8 for v in upper),"certified residual goal flag"
    f=result["fast_evaluations"];o=result["outward_evaluations"];ordinary=result["ordinary_evaluations"]
    assert all(isinstance(v,int) and v>=0 and v%k==0 for v in [f,o,ordinary])
    assert result["bound_evaluations"]==f+o
    name=result["policy"]
    if name=="PLAIN_STOP":assert (f,o,ordinary)==(0,0,k)
    if name=="PLAIN_ALWAYS":assert (f,o,ordinary)==(0,0,2*k)
    if name=="FAST_ONCE":assert (f,o,ordinary)==(k,0,0 if result["certified"] else k)
    if name in ("FAST_BLOCK","RESIDUAL"):assert f>=k and o==ordinary==0
    if name=="OUTWARD_BLOCK":assert o>=k and f==ordinary==0
    if name in ("PLAIN_STOP","PLAIN_ALWAYS"):assert result["certified"] is False
    if "trace" in result and result["trace"]:
        assert len(result["trace"])*k==f+o+ordinary,"actual readout/trace count"

def verify_native(public,oracles,path,config=None,exact_residual_ids=None):
    counts={"policy_executions_checked":0,"interval_scalar_containments":0,"certified_optimal":0,
        "identical_noncertified_fallbacks":0,"byte_exact_transport_requests":0,
        "exact_residual_upper_action_checks":0,"exact_residual_vector_evaluations":0,
        "plain_unverified_diagnostics":0,"requests":0}
    seen=set();budgets=set();audit_seen=set();split_seen=set();mode_seen=set()
    for entry in rows(path):
        ident=entry["source_id"];budget=entry["additional"];key=(ident,budget)
        assert key not in seen,("duplicate parent/budget",key)
        seen.add(key);budgets.add(budget);split_seen.add(entry["split"]);mode_seen.add(entry["mode"])
        task=public[ident];q=[F(x) for x in oracles[ident]["u_exact"]];k=len(q)
        native=entry["native"]
        assert native["id"]==ident and native["additional"]==budget
        mode=entry["mode"];assert mode in ("trace","timing")
        trace=mode=="trace";repeats=1 if trace else (config["repeats"] if config else 9)
        rr=native["results"];expected={(p,rep) for p in POLICIES for rep in range(repeats)}
        actual=[(r["policy"],r["rep"]) for r in rr]
        assert len(actual)==len(expected) and set(actual)==expected,"complete distinct policy/rep coverage"
        if "task_sha256" in entry:
            canonical=json.dumps(task,sort_keys=True,separators=(",",":")).encode()
            assert entry["task_sha256"]==hashlib.sha256(canonical).hexdigest()
        if trace:
            assert native["input_rhs_bits"]==[[bits(v) for v in b] for b in task["rhs"]],(ident,"RHS transport bits")
            assert native["input_scale_bits"]==bits(task["scale"]),(ident,"scale transport bits")
            counts["byte_exact_transport_requests"]+=1
        audit=trace and ((exact_residual_ids is not None and ident in exact_residual_ids and budget==32)
            or (entry["split"]=="UNIT") or (config is None and task["side"]<=4))
        cache={}
        if audit:audit_seen.add(ident)
        by_rep={}
        for result in rr:
            counts["policy_executions_checked"]+=1
            by_rep.setdefault(result["rep"],{})[result["policy"]]=result
            assert isinstance(result["certified"],bool)
            assert isinstance(result["selected"],int) and 0<=result["selected"]<k
            assert isinstance(result["initial_selected"],int) and 0<=result["initial_selected"]<k
            assert len(result["iterations"])==k and all(isinstance(v,int) and v>=0 for v in result["iterations"])
            assert result["solver_matvecs"]==sum(result["iterations"]),"actual CG-call accounting"
            assert all(math.isfinite(result[t]) and result[t]>=0 for t in TIMES)
            diagnostic_contract(result,k)
            if final_plain(result):counts["plain_unverified_diagnostics"]+=1
            families=[result["initial_bounds"],result["final_bounds"],result["decision_bounds"]]
            for checkpoint in result.get("trace",[]):
                assert len(checkpoint["iterations"])==k
                families.extend([checkpoint["fresh_bounds"],checkpoint["bounds"]])
            for bounds in families:
                assert len(bounds)==k
                for u,interval in zip(q,bounds):
                    assert all(math.isfinite(interval[a]) for a in ("lo","hi","ell_hat"))
                    assert 0<=F(interval["lo"])<=u<=F(interval["hi"])<=1,(ident,result["policy"],str(u),interval)
                    counts["interval_scalar_containments"]+=1
            if result["certified"]:
                index=result["selected"];b=result["decision_bounds"]
                assert F(b[index]["lo"])>max(F(v["hi"]) for j,v in enumerate(b) if j!=index)
                assert q[index]==max(q) and sum(v==max(q) for v in q)==1
                counts["certified_optimal"]+=1
            if trace:
                assert len(result["final_x"])==k
                assert all(len(x)==task["side"]**2 and all(math.isfinite(v) for v in x) for x in result["final_x"])
                assert result["trace"],"requested trace missing"
            else:
                assert not result.get("trace") and not result.get("final_x"),"timing trace/state overhead"
            if audit and result["residual_upper"] is not None:
                for index,(b,x,upper) in enumerate(zip(task["rhs"],result["final_x"],result["residual_upper"])):
                    state=(index,tuple(bits(v) for v in x))
                    if state not in cache:
                        cache[state]=residual_square(task,b,x)
                        counts["exact_residual_vector_evaluations"]+=1
                    residual,norm=cache[state]
                    assert F(upper)**2*norm>=residual,(ident,result["policy"],index,"true residual upper")
                    if norm==0:assert residual==0 and upper==0
                    counts["exact_residual_upper_action_checks"]+=1
        for group in by_rep.values():
            base=group["PLAIN_ALWAYS"]
            for name in GATES:
                result=group[name]
                assert q[result["selected"]]>=q[base["selected"]],(ident,name,"paired utility noninferiority")
                if not result["certified"]:
                    assert result["selected"]==base["selected"]
                    assert result["iterations"]==base["iterations"]
                    assert result["solver_matvecs"]==base["solver_matvecs"]
                    assert [bits(b["ell_hat"]) for b in result["final_bounds"]]==[bits(b["ell_hat"]) for b in base["final_bounds"]],(ident,name,"ordinary fallback ell bits")
                    if trace:
                        assert [[bits(v) for v in x] for x in result["final_x"]]==[[bits(v) for v in x] for x in base["final_x"]],(ident,name,"full stored iterate bits")
                    counts["identical_noncertified_fallbacks"]+=1
            assert len({r["initial_selected"] for r in group.values()})==1
            assert len({tuple(bits(b["ell_hat"]) for b in r["initial_bounds"]) for r in group.values()})==1
        counts["requests"]+=1
    assert len(mode_seen)==len(split_seen)==1 and seen,"nonempty homogeneous run"
    expected_budgets=config["additional"] if config else sorted(budgets)
    assert seen=={(ident,b) for ident in public for b in expected_budgets},"complete source/budget coverage"
    if exact_residual_ids is not None and split_seen=={"TEST"} and mode_seen=={"trace"}:
        assert audit_seen==set(exact_residual_ids),"frozen exact-residual audit subset coverage"
    counts["exact_residual_parent_ids"]=sorted(audit_seen)
    counts["observed_source_budget_pairs"]=len(seen)
    counts["scope"]="All source/budget/policy/rep q intervals and fallback states; exact residual vectors only stated trace subset"
    return counts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],required=True)
    ap.add_argument("--mode",choices=["trace","timing"],required=True)
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];ev=args.evidence_dir or doc/"evidence"
    config=json.loads((doc/"SOURCE_SPECS.json").read_text())
    audit=json.loads((doc/"C04_RUN_CONTRACT.json").read_text())["exact_residual_audit"]
    output=ev/f"C24_VERIFY_{args.split}_{args.mode.upper()}.json"
    if output.exists():raise FileExistsError("verification evidence exists")
    public_path=ev/f"C24_PUBLIC_{args.split}.jsonl.gz";oracle_path=ev/f"C24_ORACLE_{args.split}.jsonl.gz"
    native_path=ev/f"C24_{args.split}_{args.mode.upper()}.jsonl.gz"
    manifest_path=native_path.with_suffix("").with_suffix(".manifest.json")
    source_manifest=ev/f"C24_SOURCES_{args.split}.json"
    manifest=json.loads(manifest_path.read_text());sm=json.loads(source_manifest.read_text())
    assert manifest["status"]=="completed" and manifest["exit_code"]==0
    assert manifest["input_hashes_before"]==manifest["input_hashes_after"]
    assert manifest["output_sha256"]==sha(native_path)
    assert manifest["public_source_sha256"]==sha(public_path)
    for p in [public_path,oracle_path]:
        assert sm["artifacts"][p.name]["sha256"]==sha(p)
    public=keyed(public_path);oracles=keyed(oracle_path)
    if args.split!="UNIT":
        specs={s["id"]:s for s in config["sources"] if s["split"]==args.split}
        assert public.keys()==specs.keys(),"frozen source population"
        assert all(oracles[i]["spec"]==s for i,s in specs.items()),"frozen source specification"
    result={"schema":"quartz.c24.verification.v1","status":"PASS","split":args.split,"mode":args.mode,
        "argv":sys.argv,"python":sys.version,"independent_of_production_imports":True,
        "source":verify_sources(public,oracles),
        "native":verify_native(public,oracles,native_path,config,set(audit["parent_ids"])),
        "input_sha256":{p.name:sha(p) for p in [public_path,oracle_path,native_path,manifest_path,source_manifest,
            doc/"SOURCE_SPECS.json",doc/"C04_RUN_CONTRACT.json"]},
        "verifier_sha256":sha(pathlib.Path(__file__)),
        "assertion_tested":"Finite exact manufactured sources, q endpoint containment, optimal strict certificates, bit-identical fixed fallback, declared true residual upper subset",
        "non_claims":"No arbitrary-input CG repair, universal runtime superiority, or independent theorem proof"}
    assert result["native"]["requests"]==manifest["requests"]
    assert result["native"]["policy_executions_checked"]==manifest["native_policy_executions"]
    with output.open("x") as out:out.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result))
if __name__=="__main__":
    main()
