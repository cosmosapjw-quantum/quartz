"""Independent exact C25 source, interval, transport, retention and residual checks.

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

POLICIES=("PLAIN_STOP","PLAIN_ALWAYS","FAST_BLOCK","RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME")
RETAIN=("RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME")
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

from c25_io import rows, load_manifest, manifest_files, semantic_signature

def tick(counts,key,amount=1):
    counts[key]=counts.get(key,0)+amount

def gain_upper(bounds,index):
    upper=max((b["hi"] for j,b in enumerate(bounds) if j!=index),default=bounds[index]["lo"])
    lower=bounds[index]["lo"]
    return 0.0 if upper<=lower else math.nextafter(upper-lower,math.inf)

def threshold(policy,k,n,predicted):
    if policy=="RETAIN_BLOCK":
        return None,"NONE"
    if policy=="RETAIN_WORK":
        return max(0.,math.nextafter((2.0**-30)*(k*n),-math.inf)),"CELL_STENCIL"
    lower_price=math.nextafter(1e-6,-math.inf)
    return max(0.,math.nextafter(lower_price*predicted,-math.inf)),"TRAIN_PREDICTED_US"

def verify_retention(result,task,utilities,budget,predicted_block_us,counts):
    name=result["policy"];q=list(map(F,utilities));k=len(q);n=task["side"]**2
    assert name in RETAIN
    metadata=result["c25"];blocks=metadata["executed_blocks"]
    assert type(blocks) is int and 0<=blocks<=(budget+7)//8,"admitted block count"
    assert math.isfinite(metadata["continuation_us"]) and metadata["continuation_us"]>=0
    expected_charge=n*(result["solver_matvecs"]+result["fast_evaluations"]+result["ordinary_evaluations"])
    assert metadata["charged_cell_stencils"]==expected_charge,"actual total typed work charge"
    assert type(metadata["price_stop_certified"]) is bool and metadata["price_stop_certified"]==(result["reason"]=="PRICE_WORK"),"charged-price flag semantics"
    assert result["fast_evaluations"]==k*(blocks+1) and result["ordinary_evaluations"]==result["outward_evaluations"]==0
    initial=result["initial_selected"];selected=result["selected"]
    assert 0<=initial<k and 0<=selected<k
    assert q[selected]>=q[initial],"retained exact utility below initial"
    final_g=metadata["final_gain_upper"]
    assert math.isfinite(final_g) and final_g>=0
    assert bits(final_g)==bits(gain_upper(result["decision_bounds"],selected)),"final directed gain upper"
    assert F(final_g)>=max(q)-q[selected],"final true regret exceeds gain upper"
    assert type(metadata["weakly_optimal"]) is bool and metadata["weakly_optimal"]==(final_g==0.)
    if metadata["weakly_optimal"]:
        assert q[selected]==max(q),"false weak optimum"
        tick(counts,"weak_optimal_retention_outputs")
    tick(counts,"retained_utility_noninferiority_checks")
    if not result.get("trace"):
        assert not metadata.get("decisions"),"timing checkpoint allocation"
        return
    trace=result["trace"];decisions=metadata["decisions"]
    assert len(trace)==len(decisions)==blocks+1,"trace/readout/decision alignment"
    assert trace[0]["fresh_bounds"]==result["initial_bounds"]
    assert trace[-1]["fresh_bounds"]==result["final_bounds"]
    assert trace[-1]["bounds"]==result["decision_bounds"]
    assert trace[-1]["iterations"]==result["iterations"]
    current=initial;previous=None;previous_g=None;previous_charge=None;previous_iterations=None
    wanted_threshold,wanted_kind=threshold(name,k,n,predicted_block_us)
    for checkpoint,(point,decision) in enumerate(zip(trace,decisions)):
        bounds=point["bounds"];fresh=point["fresh_bounds"]
        assert len(bounds)==len(fresh)==k
        assert decision["checkpoint"]==decision["executed_blocks"]==checkpoint
        assert decision["incumbent_before"]==current
        for j,(bound,new) in enumerate(zip(bounds,fresh)):
            expected_lo=max(previous[j]["lo"],new["lo"]) if previous else new["lo"]
            expected_hi=min(previous[j]["hi"],new["hi"]) if previous else new["hi"]
            assert bound["lo"]==expected_lo and bound["hi"]==expected_hi,"historical intersection mismatch"
            assert 0<=F(bound["lo"])<=q[j]<=F(bound["hi"])<=1,"checkpoint exact utility enclosure"
        challenger=max(range(k),key=lambda j:(bounds[j]["lo"],-j))
        switched=bounds[challenger]["lo"]>bounds[current]["hi"]
        expected=challenger if switched else current
        assert decision["incumbent_after"]==expected,"unsafe/missing deterministic incumbent switch"
        proof=decision["switch_proof"]
        if switched:
            assert proof is not None and proof["from"]==current and proof["to"]==expected
            assert proof["challenger_lower"]==bounds[expected]["lo"]
            assert proof["incumbent_upper"]==bounds[current]["hi"]
            assert F(proof["challenger_lower"])>F(proof["incumbent_upper"])
            assert q[expected]>q[current],"accepted switch did not improve exact utility"
            tick(counts,"strict_accepted_switches")
        else:
            assert proof is None,"spurious switch proof"
        current=expected
        gap=decision["gain_upper"]
        assert math.isfinite(gap) and bits(gap)==bits(gain_upper(bounds,current)),"directed checkpoint gain bound"
        rectangle=max(F(0),max((F(b["hi"])-F(bounds[current]["lo"]) for j,b in enumerate(bounds) if j!=current),default=F(0)))
        assert F(gap)>=rectangle>=max(q)-q[current],"gain/true regret bound"
        if previous_g is not None:
            assert gap<=previous_g,"retained gap increased"
        assert decision["cost_kind"]==wanted_kind
        actual_threshold=decision["price_threshold"]
        assert actual_threshold==wanted_threshold,"fixed directed price threshold"
        if actual_threshold is not None:
            assert math.isfinite(actual_threshold) and actual_threshold>=0
            exact_cost=F(k*n,1<<30) if name=="RETAIN_WORK" else F(1,1000000)*F(predicted_block_us)
            assert F(actual_threshold)<=exact_cost,"price threshold is not a lower enclosure"
        iterations=point["iterations"]
        assert len(iterations)==k and all(type(x) is int and x>=0 for x in iterations)
        charge=n*(sum(iterations)+k*(checkpoint+1))
        assert decision["charged_cell_stencils"]==charge,"checkpoint actual cell-stencil charge"
        if previous_charge is not None:
            assert charge-previous_charge>=k*n,"mandatory fresh-readout future floor"
            assert all(0<=v-old<=8 for v,old in zip(iterations,previous_iterations)),"block solver count"
            tick(counts,"admitted_blocks_with_fresh_readout_floor")
        strict=bounds[current]["lo"]>max((b["hi"] for j,b in enumerate(bounds) if j!=current),default=-math.inf)
        reason=decision["reason"]
        if strict:
            assert reason==("INITIAL_CERTIFICATE" if checkpoint==0 else "BLOCK_CERTIFICATE"),"strict certificate precedes priced stop"
        elif gap==0:
            assert reason=="ZERO_REGRET_OPTIMUM","zero-regret optimum precedes priced stop"
        elif reason in ("PRICE_WORK","PRICE_TIME"):
            assert reason==("PRICE_WORK" if name=="RETAIN_WORK" else "PRICE_TIME" if name=="RETAIN_TIME" else None)
            assert checkpoint*8<budget,"priced stop mislabels exhausted budget"
            assert gap<=actual_threshold and (name!="RETAIN_TIME" or predicted_block_us>0),"false positive price stop"
            if previous_iterations is not None:
                assert iterations!=previous_iterations,"priced stop mislabels no progress"
            tick(counts,"certified_work_price_stops" if reason=="PRICE_WORK" else "heuristic_time_price_stops")
        elif reason=="BUDGET":
            assert checkpoint*8>=budget,"premature budget exhaustion"
        elif reason=="NO_PROGRESS":
            assert checkpoint>0 and iterations==previous_iterations,"false no-progress exhaustion"
        else:
            assert reason=="CONTINUE" and checkpoint*8<budget,"invalid continuation reason/cap"
            if name=="RETAIN_WORK" or (name=="RETAIN_TIME" and predicted_block_us>0):
                assert gap>actual_threshold,"ignored prespecified positive-price stop"
        if checkpoint<len(decisions)-1:
            assert reason=="CONTINUE","continued after stop"
        else:
            assert reason!="CONTINUE","unterminated final decision"
        tick(counts,"retention_checkpoint_gain_checks")
        previous=bounds;previous_g=gap;previous_charge=charge;previous_iterations=iterations
    assert current==selected and decisions[-1]["reason"]==result["reason"]
    assert decisions[-1]["charged_cell_stencils"]==metadata["charged_cell_stencils"]
    assert bits(decisions[-1]["gain_upper"])==bits(final_g)

def diagnostic_contract(result,k):
    name=result["policy"];plain=name in ("PLAIN_STOP","PLAIN_ALWAYS")
    assert len(result["residual_relative"])==k
    assert all(math.isfinite(v) and v>=0 for v in result["residual_relative"])
    if plain:
        assert result["residual_upper"] is None and result["residual_goal_met"] is None,"plain residual unverified"
        assert all(v["lo"]==0. and v["hi"]==1. for v in result["final_bounds"])
        assert not result["certified"]
    else:
        upper=result["residual_upper"]
        assert isinstance(upper,list) and len(upper)==k
        assert all(math.isfinite(v) and v>=0 for v in upper)
        assert type(result["residual_goal_met"]) is bool
        assert result["residual_goal_met"]==all(v<=1e-8 for v in upper)
    fast=result["fast_evaluations"];outward=result["outward_evaluations"];ordinary=result["ordinary_evaluations"]
    assert all(type(v) is int and v>=0 and v%k==0 for v in [fast,outward,ordinary])
    assert result["bound_evaluations"]==fast+outward and outward==0
    if name=="PLAIN_STOP":
        assert (fast,ordinary)==(0,k)
    elif name=="PLAIN_ALWAYS":
        assert (fast,ordinary)==(0,2*k)
    else:
        assert fast>=k and ordinary==0
    if result.get("trace"):
        assert len(result["trace"])*k==fast+ordinary,"actual trace readout count"

def verify_native(public,oracles,path,config=None,exact_residual_ids=None):
    counts={"policy_executions_checked":0,"interval_scalar_containments":0,"certified_optimal":0,
            "identical_noncertified_fallbacks":0,"byte_exact_transport_requests":0,
            "exact_residual_upper_action_checks":0,"exact_residual_vector_evaluations":0,
            "plain_unverified_diagnostics":0,"requests":0,"repeat_consistency_groups":0}
    seen=set();budgets=set();audit_seen=set();split_seen=set();mode_seen=set()
    for entry in rows(path):
        ident=entry["source_id"];budget=entry["additional"];key=(ident,budget)
        assert key not in seen,("duplicate parent/budget",key)
        seen.add(key);budgets.add(budget);split_seen.add(entry["split"]);mode_seen.add(entry["mode"])
        task=public[ident];q=list(map(F,oracles[ident]["u_exact"]));k=len(q)
        native=entry["native"]
        assert native["id"]==ident and native["additional"]==budget
        mode=entry["mode"];assert mode in ("trace","timing")
        trace=mode=="trace";repeats=1 if trace else (config["repeats"] if config else 9)
        predicted=entry["predicted_block_us"]
        assert math.isfinite(predicted) and predicted>=0
        if entry["split"]=="TRAIN":
            assert predicted==0.,"TRAIN positive-price stopping disabled"
        if entry["split"] in ("CAL","TEST"):
            assert predicted>0.,"heldout frozen positive prediction"
        rr=native["results"];expected={(p,rep) for p in POLICIES for rep in range(repeats)}
        actual=[(r["policy"],r["rep"]) for r in rr]
        assert len(actual)==len(expected) and set(actual)==expected,"complete distinct policy/rep coverage"
        if "task_sha256" in entry:
            canonical=json.dumps(task,sort_keys=True,separators=(",",":")).encode()
            assert entry["task_sha256"]==hashlib.sha256(canonical).hexdigest()
        if trace:
            assert native["input_rhs_bits"]==[[bits(v) for v in b] for b in task["rhs"]]
            assert native["input_scale_bits"]==bits(task["scale"])
            counts["byte_exact_transport_requests"]+=1
        audit=trace and ((exact_residual_ids is not None and ident in exact_residual_ids and budget==32)
                        or entry["split"]=="UNIT" or (config is None and task["side"]<=4))
        cache={}
        if audit:
            audit_seen.add(ident)
        by_rep={};by_policy={}
        for result in rr:
            counts["policy_executions_checked"]+=1
            by_rep.setdefault(result["rep"],{})[result["policy"]]=result
            by_policy.setdefault(result["policy"],[]).append(result)
            assert type(result["certified"]) is bool
            assert type(result["selected"]) is int and 0<=result["selected"]<k
            assert type(result["initial_selected"]) is int and 0<=result["initial_selected"]<k
            assert len(result["iterations"])==k and all(type(v) is int and v>=0 for v in result["iterations"])
            assert result["solver_matvecs"]==sum(result["iterations"])
            assert all(math.isfinite(result[t]) and result[t]>=0 for t in TIMES)
            assert result["total_us"]>=max(result[t] for t in TIMES if t!="total_us"),"component exceeds full native wall time"
            diagnostic_contract(result,k)
            if result["policy"] in ("PLAIN_STOP","PLAIN_ALWAYS"):
                counts["plain_unverified_diagnostics"]+=1
            families=[result["initial_bounds"],result["final_bounds"],result["decision_bounds"]]
            for point in result.get("trace",[]):
                assert len(point["iterations"])==k
                families.extend([point["fresh_bounds"],point["bounds"]])
            for bounds in families:
                assert len(bounds)==k
                for u,bound in zip(q,bounds):
                    assert all(math.isfinite(bound[a]) for a in ("lo","hi","ell_hat"))
                    assert 0<=F(bound["lo"])<=u<=F(bound["hi"])<=1,(ident,result["policy"],str(u),bound)
                    counts["interval_scalar_containments"]+=1
            if result["certified"]:
                index=result["selected"];b=result["decision_bounds"]
                assert F(b[index]["lo"])>max(F(v["hi"]) for j,v in enumerate(b) if j!=index)
                assert q[index]==max(q) and sum(v==max(q) for v in q)==1
                counts["certified_optimal"]+=1
            if trace:
                assert len(result["final_x"])==k
                assert all(len(x)==task["side"]**2 and all(math.isfinite(v) for v in x) for x in result["final_x"])
                assert result["trace"]
            else:
                assert not result.get("trace") and not result.get("final_x"),"timing vector/trace overhead"
            metadata=result["c25"]
            ordinary=max(range(k),key=lambda j:(result["final_bounds"][j]["ell_hat"],-j))
            assert metadata["final_ordinary_selected"]==ordinary,"final ordinary index diagnostic"
            assert metadata["charged_cell_stencils"]==task["side"]**2*(result["solver_matvecs"]+result["fast_evaluations"]+result["ordinary_evaluations"])
            if result["policy"] in RETAIN:
                verify_retention(result,task,q,budget,predicted,counts)
                assert metadata["continuation_us"]<=result["total_us"]
            else:
                assert metadata["continuation_us"] is None and metadata["final_gain_upper"] is None and metadata["weakly_optimal"] is None
                assert not metadata.get("decisions") and metadata["price_stop_certified"] is False
                assert metadata["executed_blocks"]==(result["fast_evaluations"]+result["ordinary_evaluations"])//k-1
            if audit and result["residual_upper"] is not None:
                for index,(b,x,upper) in enumerate(zip(task["rhs"],result["final_x"],result["residual_upper"])):
                    state=(index,tuple(bits(v) for v in x))
                    if state not in cache:
                        cache[state]=residual_square(task,b,x)
                        counts["exact_residual_vector_evaluations"]+=1
                    residual,norm=cache[state]
                    assert F(upper)**2*norm>=residual,(ident,result["policy"],index,"true residual upper")
                    if norm==0:
                        assert residual==0 and upper==0
                    counts["exact_residual_upper_action_checks"]+=1
        for group in by_rep.values():
            base=group["PLAIN_ALWAYS"];gate=group["FAST_BLOCK"]
            assert q[gate["selected"]]>=q[base["selected"]],"C24 FAST paired utility noninferiority"
            if not gate["certified"]:
                for field in ("selected","iterations","solver_matvecs"):
                    assert gate[field]==base[field],("C24 unchanged FAST fallback",field)
                assert [bits(b["ell_hat"]) for b in gate["final_bounds"]]==[bits(b["ell_hat"]) for b in base["final_bounds"]]
                if trace:
                    assert [[bits(v) for v in x] for x in gate["final_x"]]==[[bits(v) for v in x] for x in base["final_x"]]
                counts["identical_noncertified_fallbacks"]+=1
            assert len({r["initial_selected"] for r in group.values()})==1
            assert len({tuple(bits(b["ell_hat"]) for b in r["initial_bounds"]) for r in group.values()})==1
            for name in RETAIN:
                assert q[group[name]["selected"]]>=q[group["PLAIN_STOP"]["selected"]]
        for group in by_policy.values():
            assert len({semantic_signature(r) for r in group})==1,"scientific output changed across repeats"
            counts["repeat_consistency_groups"]+=1
        counts["requests"]+=1
    assert len(mode_seen)==len(split_seen)==1 and seen,"nonempty homogeneous run"
    expected_budgets=config["additional"] if config else sorted(budgets)
    assert seen=={(ident,b) for ident in public for b in expected_budgets},"complete source/budget coverage"
    if exact_residual_ids is not None and split_seen=={"TEST"} and mode_seen=={"trace"}:
        assert audit_seen==set(exact_residual_ids),"frozen exact-residual subset coverage"
    counts["exact_residual_parent_ids"]=sorted(audit_seen)
    counts["observed_source_budget_pairs"]=len(seen)
    counts["scope"]="All exact utility intervals, final retention/gain/counters/repeats; all trace checkpoints/switches; exact residual vectors only stated trace subset"
    return counts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],required=True)
    ap.add_argument("--mode",choices=["trace","timing"],required=True)
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];ev=(args.evidence_dir or doc/"evidence").resolve()
    config=json.loads((doc/"SOURCE_SPECS.json").read_text())
    output=ev/f"C25_VERIFY_{args.split}_{args.mode.upper()}.json"
    if output.exists():
        raise FileExistsError("preserved verification evidence exists")
    public_path=ev/f"C25_PUBLIC_{args.split}.jsonl.gz";oracle_path=ev/f"C25_ORACLE_{args.split}.jsonl.gz"
    native_path=ev/f"C25_{args.split}_{args.mode.upper()}.manifest.json"
    source_manifest=ev/f"C25_SOURCES_{args.split}.json"
    manifest=load_manifest(native_path);sm=json.loads(source_manifest.read_text())
    assert manifest["public_source_sha256"]==sha(public_path)
    for p in [public_path,oracle_path]:
        assert sm["artifacts"][p.name]["sha256"]==sha(p)
    public=keyed(public_path);oracles=keyed(oracle_path)
    if args.split!="UNIT":
        specs={s["id"]:s for s in config["sources"] if s["split"]==args.split}
        assert public.keys()==specs.keys(),"frozen source population"
        assert all(oracles[i]["spec"]==s for i,s in specs.items()),"frozen source specifications"
    inputs=[public_path,oracle_path,source_manifest,doc/"SOURCE_SPECS.json",doc/"C04_RUN_CONTRACT.json"]+manifest_files(native_path)
    if args.split in ("CAL","TEST"):
        predictor=ev/config["predictor"]["artifact"];fit=json.loads(predictor.read_text())
        assert sha(predictor)==manifest["predictor_sha256"],"frozen heldout predictor binding"
        assert fit["source_specs_sha256"]==sha(doc/"SOURCE_SPECS.json")
        for env in rows(native_path):
            assert env["predicted_block_us"]==fit["by_side"][str(public[env["source_id"]]["side"])]["predicted_block_us"]
        inputs.append(predictor)
    before={str(p.relative_to(ev)) if p.is_relative_to(ev) else p.name:sha(p) for p in inputs}
    result={"schema":"quartz.c25.verification.v1","status":"PASS","split":args.split,"mode":args.mode,
            "argv":sys.argv,"python":sys.version,"independent_of_production_numerical_imports":True,
            "source":verify_sources(public,oracles),
            "native":verify_native(public,oracles,native_path,config,set(config["residual_exact_subset"])),
            "input_sha256":before,"verifier_sha256":sha(pathlib.Path(__file__)),
            "assertion_tested":"Exact finite manufactured utilities, all interval/gain containment, safe retention, charged threshold/floor, unchanged FAST fallback, fixed residual subset and repetitions",
            "non_claims":"No universal hardware-time bound, global allocation optimality, or theorem from finite tests"}
    assert before=={str(p.relative_to(ev)) if p.is_relative_to(ev) else p.name:sha(p) for p in inputs},"verification inputs changed"
    assert result["native"]["requests"]==manifest["requests"]
    assert result["native"]["policy_executions_checked"]==manifest["native_policy_executions"]
    with output.open("x") as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result))

if __name__=="__main__":
    main()
