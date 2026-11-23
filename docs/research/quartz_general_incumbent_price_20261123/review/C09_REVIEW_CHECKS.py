"""Independent C09 review checks, from preserved raw evidence.

No imports from production analyzers, verifiers, generators, or runners.
Reads source/data/contracts; writes only this review's JSON and test log.
No native benchmark/build, git mutation, source generation, or model refit.
"""
import collections
import datetime
from fractions import Fraction as Q
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path("/home/cosmosapjw/quantum/research/quartz_general_meta25_20261123")
DOC = ROOT / "docs/research/quartz_general_incumbent_price_20261123"
EV = DOC / "evidence"
REVIEW = DOC / "review"
SNAPSHOT = "d45173685f955820bd848bf4ddb2044382ca7e67"
POLICIES = ("PLAIN_STOP","PLAIN_ALWAYS","FAST_BLOCK","RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME")
RETAIN = POLICIES[3:]
locked = {}
counts = collections.Counter()
report = {
    "schema": "quartz.c25.c09-independent-review-checks.v1",
    "reviewer": "/root/c25_review",
    "reviewed_snapshot": SNAPSHOT,
    "claim_scope": "Frozen C25 SPD retention, declared work-cost class, empirical TEST estimates; not original MCTS/P2",
    "assertion_tested": "Preserved raw TEST inputs/outcomes satisfy retention and bound contracts; reported finite summaries/bootstrap and TRAIN-only point predictor reconstruct independently.",
    "command": [sys.executable, str(Path(__file__).resolve())],
    "status": "running",
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python": sys.version,
    "numpy": np.__version__,
    "platform": platform.platform(),
    "native_campaigns_started": 0,
    "production_imports": [],
}
started = time.monotonic()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def lock(path):
    path = Path(path).resolve()
    value = digest(path)
    previous = locked.setdefault(str(path), value)
    assert previous == value, ("input changed during read", str(path))
    return path

def document(path):
    return json.loads(lock(path).read_text())

def gzrows(path):
    with gzip.open(lock(path), "rt") as f:
        for line in f:
            yield json.loads(line)

def raw(split, mode):
    mpath = EV / ("C25_" + split + "_" + mode.upper() + ".manifest.json")
    m = document(mpath)
    assert m["status"] == "completed" and m["exit_code"] == 0
    assert m["input_hashes_before"] == m["input_hashes_after"]
    assert m["additional"] == [8,32,128]
    assert m["repeats"] == (9 if mode == "timing" else 1)
    assert m["warmup"] == (1 if mode == "timing" else 0)
    assert m["binary_sha256"] == freeze["frozen_inputs"]["target/release/quartz_c25_incumbent_price"]
    expected = [s["id"] for s in cfg["sources"] if s["split"] == split]
    assert m["source_ids"] == expected
    assert [c["source_id"] for c in m["chunks"]] == expected
    assert len({c["path"] for c in m["chunks"]}) == len(expected)
    requests = 0
    for c in m["chunks"]:
        p = (EV / c["path"]).resolve()
        assert p.is_relative_to(EV.resolve())
        assert p.stat().st_size == c["bytes"] <= 45 * 1024 * 1024
        assert digest(p) == c["sha256"]
        budgets = []
        for e in gzrows(p):
            assert e["source_id"] == c["source_id"] and e["split"] == split and e["mode"] == mode
            assert e["native"]["id"] == e["source_id"] and e["native"]["additional"] == e["additional"]
            assert "error" not in e["native"]
            budgets.append(e["additional"])
            requests += 1
            yield e
        assert budgets == [8,32,128] and c["requests"] == 3
    assert requests == m["requests"] == len(expected) * 3
    counts[split + "_" + mode + "_requests"] = requests

def sig(row):
    value = {k:v for k,v in row.items() if k not in
             {"rep","total_us","validation_us","initial_us","solve_us","bounds_us","trace","final_x"}}
    value["c25"] = {k:v for k,v in row["c25"].items() if k not in {"continuation_us","decisions"}}
    return json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)

def check_bounds(bounds, q, counter):
    assert len(bounds) == len(q)
    for b, u in zip(bounds, q):
        assert all(math.isfinite(b[k]) for k in ("lo","hi","ell_hat"))
        assert Q(0) <= Q(b["lo"]) <= u <= Q(b["hi"]) <= Q(1)
        counts[counter] += 1

def exactmean(values):
    return sum(values, Q(0)) / len(values)

def close(a,b,atol=1e-12):
    assert math.isclose(a,b,rel_tol=1e-14,abs_tol=atol), (a,b)

def inspect_trace(e, truth, timing):
    ident, budget = e["source_id"], e["additional"]
    q = truth[ident]
    n = specs[ident]["side"]**2
    k = len(q)
    rows = e["native"]["results"]
    assert len(rows) == 6 and {r["policy"] for r in rows} == set(POLICIES)
    for row in rows:
        name = row["policy"]
        assert sig(row) == timing[(ident,budget)][name]["signature"]
        counts["TEST_trace_timing_policy_matches"] += 1
        for part in ("initial_bounds","final_bounds","decision_bounds"):
            check_bounds(row[part],q,"TEST_trace_bound_enclosures")
        previous_bounds = None
        for t in row["trace"]:
            for part in ("fresh_bounds","bounds"):
                check_bounds(t[part],q,"TEST_trace_bound_enclosures")
            if previous_bounds is not None:
                for old,fresh,new in zip(previous_bounds,t["fresh_bounds"],t["bounds"]):
                    assert new["lo"] == max(old["lo"],fresh["lo"])
                    assert new["hi"] == min(old["hi"],fresh["hi"])
            else:
                assert t["bounds"] == t["fresh_bounds"]
            previous_bounds = t["bounds"]
        if name not in RETAIN:
            continue
        meta = row["c25"]
        assert len(row["trace"]) == len(meta["decisions"]) == meta["executed_blocks"]+1
        current = row["initial_selected"]
        old_gap, old_charge, old_iterations = None, None, None
        for t, d in zip(row["trace"],meta["decisions"]):
            bounds = t["bounds"]
            assert d["incumbent_before"] == current
            challenger = sorted(range(k), key=lambda j:(-bounds[j]["lo"],j))[0]
            new = d["incumbent_after"]
            if bounds[challenger]["lo"] > bounds[current]["hi"]:
                assert new == challenger
                proof = d["switch_proof"]
                assert proof == {"from":current,"to":new,"challenger_lower":bounds[new]["lo"],"incumbent_upper":bounds[current]["hi"]}
                assert q[new] > q[current]
                counts["TEST_strict_switches"] += 1
            else:
                assert new == current and d["switch_proof"] is None
            current = new
            rectangle = max([Q(0)] + [Q(b["hi"])-Q(bounds[current]["lo"]) for j,b in enumerate(bounds) if j != current])
            gap = d["gain_upper"]
            assert Q(gap) >= rectangle >= max(q)-q[current]
            upper = max(b["hi"] for j,b in enumerate(bounds) if j != current)
            expected_gap = 0.0 if upper <= bounds[current]["lo"] else math.nextafter(upper-bounds[current]["lo"], math.inf)
            assert gap == expected_gap
            if rectangle == 0:
                assert gap == 0.0
            if old_gap is not None:
                assert gap <= old_gap
            cp = d["checkpoint"]
            charge = n*(sum(t["iterations"])+k*(cp+1))
            assert d["executed_blocks"] == cp and d["charged_cell_stencils"] == charge
            if old_charge is not None:
                assert charge-old_charge >= k*n
                counts["TEST_mandatory_readout_floors"] += 1
            threshold = d["price_threshold"]
            if name == "RETAIN_WORK":
                assert Q(threshold) <= Q(k*n, 1 << 30)
                assert threshold == math.nextafter((k*n)*2.0**-30,-math.inf)
                assert d["cost_kind"] == "CELL_STENCIL"
            elif name == "RETAIN_TIME":
                assert Q(threshold) <= Q(1,1000000)*Q(e["predicted_block_us"])
                assert d["cost_kind"] == "TRAIN_PREDICTED_US"
            else:
                assert threshold is None and d["cost_kind"] == "NONE"
            unique = all(bounds[current]["lo"] > b["hi"] for j,b in enumerate(bounds) if j != current)
            remaining = max(0,budget-8*cp)
            no_progress = old_iterations is not None and t["iterations"] == old_iterations
            wanted = ("INITIAL_CERTIFICATE" if cp == 0 else "BLOCK_CERTIFICATE") if unique else (
                "ZERO_REGRET_OPTIMUM" if gap == 0 else "BUDGET" if remaining == 0 else
                "NO_PROGRESS" if no_progress else
                ("PRICE_WORK" if name == "RETAIN_WORK" else "PRICE_TIME") if threshold is not None and gap <= threshold
                else "CONTINUE")
            assert d["reason"] == wanted
            if wanted == "PRICE_WORK":
                assert Q(gap) <= Q(k*n,1 << 30)
                counts["TEST_price_work_stops"] += 1
            if wanted == "PRICE_TIME":
                counts["TEST_price_time_stops"] += 1
            counts["TEST_retention_checkpoints"] += 1
            old_gap,old_charge,old_iterations = gap,charge,t["iterations"]
        assert current == row["selected"]
        assert q[current] >= q[row["initial_selected"]]
        assert meta["final_gain_upper"] == old_gap
        assert row["fast_evaluations"] == k*(meta["executed_blocks"]+1)
        assert meta["charged_cell_stencils"] == old_charge

try:
    cfg = document(DOC/"SOURCE_SPECS.json")
    freeze = document(EV/"C07_PRE_TEST_FREEZE.json")
    predictor = document(EV/"C25_TRAIN_COST_PREDICTOR.json")
    summary = document(EV/"C25_TEST_SUMMARY.json")
    report["observed_head_start"] = subprocess.check_output(["git","--no-optional-locks","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    report["dirty_start"] = subprocess.check_output(["git","--no-optional-locks","-C",str(ROOT),"status","--short"],text=True).splitlines()
    assert report["observed_head_start"] == SNAPSHOT
    assert freeze["status"] == "FROZEN_BEFORE_TEST_GENERATION_AND_OUTCOMES"
    for relative,h in freeze["frozen_inputs"].items():
        assert digest(lock(ROOT/relative)) == h, ("frozen input",relative)
    report["frozen_input_count"] = len(freeze["frozen_inputs"])
    oldtext = lock(ROOT/"src/bin/quartz_c24_cheap_certificate.rs").read_text()
    newtext = lock(ROOT/"src/bin/quartz_c25_incumbent_price.rs").read_text()
    prefix = newtext.split("// C25 extends only the controller/output contract;")[0]
    prefix = prefix.replace("#[allow(dead_code)]\nfn c24_main()", "fn main()")
    assert prefix.rstrip() == oldtext.rstrip()
    report["c24_numerical_prefix_reuse"] = "EXACT_EXCEPT_MAIN_RENAME_AND_TRAILING_WHITESPACE"
    assert collections.Counter(s["split"] for s in cfg["sources"]) == {"TRAIN":36,"CAL":36,"TEST":72}
    assert len({s["id"] for s in cfg["sources"]}) == len({s["seed"] for s in cfg["sources"]}) == 144
    assert cfg["additional"] == [8,32,128] and cfg["policies"] == list(POLICIES)
    assert cfg["bootstrap_seed"] == 2500923 and cfg["bootstrap_resamples"] == 2000
    assert cfg["primary_policy"] == "RETAIN_TIME" and cfg["primary_baseline"] == "PLAIN_STOP"
    assert cfg["primary_additional"] == 32 and Q(cfg["primary_price_exact"]) == Q(1,1000000)
    assert Q(cfg["work_price_exact"]) == Q(1,1 << 30)
    specs = {s["id"]:s for s in cfg["sources"]}
    assert "c24_test_34_1" not in specs
    test_source = document(EV/"C25_SOURCES_TEST.json")
    assert predictor["created_utc"] < freeze["utc"] < test_source["start_utc"]
    report["chronology"] = {"predictor":predictor["created_utc"],"freeze":freeze["utc"],"TEST_generation":test_source["start_utc"]}
    for side in range(2,129):
        for k in range(2,9):
            assert Q((k*side*side)*2.0**-30) == Q(k*side*side,1 << 30)
            counts["exact_dyadic_work_threshold_products"] += 1

    truth = {}
    public_metadata = {}
    public = gzrows(EV/"C25_PUBLIC_TEST.jsonl.gz")
    oracle = gzrows(EV/"C25_ORACLE_TEST.jsonl.gz")
    for task, latent in zip(public,oracle,strict=True):
        ident = task["id"]
        assert latent["id"] == ident and latent["spec"] == specs[ident]
        assert set(task) == {"id","side","rho","scale","rhs"}
        side,rho,p = task["side"],task["rho"],latent["dyadic_exponent"]
        assert p == 30
        q = list(map(Q,latent["u_exact"]))
        assert len(q) == len(task["rhs"]) == len(latent["x_num"]) == 3
        for b,x,u in zip(task["rhs"],latent["x_num"],q):
            assert len(x) == len(b) == side*side
            bx_num = 0
            for r in range(side):
                for c in range(side):
                    i = r*side+c
                    near = []
                    if r: near.append(x[i-side])
                    if r+1 < side: near.append(x[i+side])
                    if c: near.append(x[i-1])
                    if c+1 < side: near.append(x[i+1])
                    a = (1+4*rho)*x[i]-rho*sum(near)
                    num,den = b[i].as_integer_ratio()
                    assert num*(1 << p) == a*den
                    bx_num += a*x[i]
                    counts["independent_TEST_Ax_equals_b_components"] += 1
            assert Q(bx_num,1 << (2*p))/Q(task["scale"]) == u
            assert Q(0) <= u <= Q(1)
            counts["independent_TEST_exact_utilities"] += 1
        truth[ident] = q
        public_metadata[ident] = {"side":side,"rho":rho,"task_sha256":hashlib.sha256(json.dumps(task,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    expected_test = [s["id"] for s in cfg["sources"] if s["split"] == "TEST"]
    assert list(truth) == expected_test
    print("Independent TEST sources and exact utilities checked",flush=True)

    timing = {}
    for e in raw("TEST","timing"):
        ident,budget = e["source_id"],e["additional"]
        q = truth[ident]
        assert e["task_sha256"] == public_metadata[ident]["task_sha256"]
        assert e["predicted_block_us"] == predictor["by_side"][str(specs[ident]["side"])]["predicted_block_us"]
        groups = {p:[] for p in POLICIES}
        for row in e["native"]["results"]:
            groups[row["policy"]].append(row)
            for part in ("initial_bounds","final_bounds","decision_bounds"):
                check_bounds(row[part],q,"TEST_timing_bound_enclosures")
        stored = {}
        baseline = groups["PLAIN_STOP"][0]
        base_scores = [b["ell_hat"] for b in baseline["initial_bounds"]]
        for name,rr in groups.items():
            assert len(rr) == 9 and {r["rep"] for r in rr} == set(range(9))
            signature = sig(rr[0])
            assert all(sig(r) == signature for r in rr)
            assert all(r["initial_selected"] == baseline["initial_selected"] for r in rr)
            assert all([b["ell_hat"] for b in r["initial_bounds"]] == base_scores for r in rr)
            first = rr[0]
            assert first["initial_selected"] == max(range(len(q)),key=lambda j:base_scores[j])
            assert all(math.isfinite(r["total_us"]) and r["total_us"] >= 0 for r in rr)
            n = specs[ident]["side"]**2
            assert first["c25"]["charged_cell_stencils"] == n*(first["solver_matvecs"]+first["fast_evaluations"]+first["ordinary_evaluations"])
            if name in RETAIN:
                assert q[first["selected"]] >= q[first["initial_selected"]]
                counts["TEST_source_budget_retention_noninferiority"] += 1
                assert first["c25"]["price_stop_certified"] == (first["reason"] == "PRICE_WORK")
            stored[name] = {"signature":signature,"utility":q[first["selected"]],"regret":max(q)-q[first["selected"]],
                            "median_us":sorted(r["total_us"] for r in rr)[4],"charged":first["c25"]["charged_cell_stencils"],
                            "reason":first["reason"],"certified":first["certified"]}
            counts["TEST_timing_policy_rows"] += len(rr)
        assert (ident,budget) not in timing
        timing[(ident,budget)] = stored
    assert set(timing) == {(i,b) for i in expected_test for b in [8,32,128]}
    print("Independent TEST timing reductions checked",flush=True)

    comparisons = []
    rng = np.random.default_rng(2500923)
    for budget in [8,32,128]:
        group = [timing[(i,budget)] for i in expected_test]
        indices = rng.integers(0,len(group),(2000,len(group)))
        published = next(s for s in summary["summaries"] if s["additional"] == budget)["policies"]
        values = {}
        for name in POLICIES:
            policy = published[name]
            us = [g[name]["median_us"] for g in group]
            regret = [g[name]["regret"] for g in group]
            du = [g[name]["utility"]-g["PLAIN_STOP"]["utility"] for g in group]
            dt = [Q(g[name]["median_us"])-Q(g["PLAIN_STOP"]["median_us"]) for g in group]
            net = [u-Q(1,1000000)*t for u,t in zip(du,dt)]
            ci = np.percentile(np.asarray(list(map(float,net)))[indices].mean(axis=1),[2.5,97.5]).tolist()
            exact_est = exactmean(net)
            primary = next(p for p in policy["priced"] if Q(p["lambda_exact"]) == Q(1,1000000))
            assert Q(primary["mean_net_difference_vs_PLAIN_STOP_exact"]) == exact_est
            assert ci == primary["paired_parent_bootstrap_net_difference_vs_PLAIN_STOP_95"]
            assert policy["wrong_decision_parents"] == sum(r>0 for r in regret)
            assert Q(policy["mean_regret_exact"]) == exactmean(regret)
            close(statistics.mean(us),policy["mean_parent_median_us"])
            assert policy["utility_decrease_vs_PLAIN_STOP_parents"] == sum(d<0 for d in du)
            assert policy["reason_counts"] == dict(collections.Counter(g[name]["reason"] for g in group))
            work = [u-Q(1,1 << 30)*(g[name]["charged"]-g["PLAIN_STOP"]["charged"]) for u,g in zip(du,group)]
            assert Q(policy["work_priced"]["mean_net_difference_vs_PLAIN_STOP_exact"]) == exactmean(work)
            values[name] = {"mean_parent_median_us":statistics.mean(us),"wrong_parents":sum(r>0 for r in regret),
                            "mean_regret_exact":str(exactmean(regret)),"mean_net_vs_STOP":float(exact_est),
                            "mean_net_vs_STOP_exact":str(exact_est),"bootstrap_95":ci,
                            "utility_decreases_vs_STOP":sum(d<0 for d in du)}
            counts["independent_TEST_policy_budget_summaries"] += 1
        comparisons.append({"additional":budget,"policies":values})
    report["TEST_independent_summaries"] = comparisons
    report["primary"] = comparisons[1]["policies"]["RETAIN_TIME"]
    print("Independent TEST exact means and all 18 primary-price bootstrap intervals matched",flush=True)

    trained = {}
    for e in raw("TRAIN","timing"):
        assert e["predicted_block_us"] == 0.0
        if e["additional"] != 32:
            continue
        rr = [r for r in e["native"]["results"] if r["policy"] == "RETAIN_BLOCK"]
        assert len(rr) == 9 and {r["rep"] for r in rr} == set(range(9))
        blocks = {r["c25"]["executed_blocks"] for r in rr}
        assert len(blocks) == 1
        b = blocks.pop()
        v = statistics.mean([r["c25"]["continuation_us"]/b for r in rr]) if b else None
        trained[e["source_id"]] = v
    expected_train = [s["id"] for s in cfg["sources"] if s["split"] == "TRAIN"]
    assert list(trained) == predictor["training_parent_ids"] == expected_train
    reconstructed = {}
    for side in (8,32,96):
        active = [(i,v) for i,v in trained.items() if specs[i]["side"] == side and v is not None]
        v = statistics.median([v for _,v in active])
        want = predictor["by_side"][str(side)]
        assert v == want["predicted_block_us"] and [i for i,_ in active] == want["active_parent_ids"]
        reconstructed[str(side)] = {"predicted_block_us":v,"active_parents":len(active)}
    report["TRAIN_predictor_independent"] = reconstructed
    print("Independent TRAIN-only predictor matched",flush=True)

    for e in raw("TEST","trace"):
        assert e["task_sha256"] == public_metadata[e["source_id"]]["task_sha256"]
        inspect_trace(e,truth,timing)
    assert counts["TEST_retention_checkpoints"] == 1529
    assert counts["TEST_mandatory_readout_floors"] == 881
    assert counts["TEST_strict_switches"] == 76
    assert counts["TEST_source_budget_retention_noninferiority"] == 648
    assert counts["TEST_timing_bound_enclosures"] == 104976
    assert counts["TEST_trace_bound_enclosures"] == 28584
    print("Independent TEST retained decisions, gaps, switches and readout floors checked",flush=True)

    unit_summary = document(EV/"C25_UNIT_PARENT_METRICS.json")
    oldfixture = [p for p in unit_summary if p["id"] == "c24_test_34_1" and p["additional"] == 32]
    assert len(oldfixture) == 1
    fixture = oldfixture[0]["policies"]
    assert Q(fixture["PLAIN_ALWAYS"]["utility_exact"]) < Q(fixture["PLAIN_STOP"]["utility_exact"])
    assert all(Q(fixture[n]["utility_exact"]) >= Q(fixture["PLAIN_STOP"]["utility_exact"]) for n in RETAIN)
    report["historical_fixture"] = "PRESERVED_UNIT_SUMMARY_CONFIRMS_C24_REGRESSION; not counted as a fresh TEST parent"

    argv = [str(ROOT/"venv/bin/python"),"-B","-m","unittest","discover","-s","tests","-p","test_c25_sources.py","-v"]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(argv,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log = REVIEW/"C09_PYTHON_TESTS.log"
    log.write_text(proc.stdout)
    report["python_tests"] = {"argv":argv,"exit_code":proc.returncode,"log":str(log),"log_sha256":digest(log)}
    assert proc.returncode == 0, proc.stdout[-4000:]

    for path,h in locked.items():
        assert digest(path) == h, ("input drift",path)
    report["input_hashes_before"] = locked
    report["input_hashes_after"] = dict(locked)
    report["counts"] = dict(counts)
    report["status"] = "PASS"
    report["exit_code"] = 0
except BaseException as error:
    report["status"] = "FAIL"
    report["exit_code"] = 1
    report["error"] = type(error).__name__+": "+str(error)
    report["traceback"] = traceback.format_exc()
    report["counts"] = dict(counts)
    report["input_hashes_observed"] = locked
    traceback.print_exc()
finally:
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["elapsed_seconds"] = time.monotonic()-started
    report["review_script_sha256"] = digest(Path(__file__))
    report["observed_head_end"] = subprocess.check_output(["git","--no-optional-locks","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    report["dirty_end"] = subprocess.check_output(["git","--no-optional-locks","-C",str(ROOT),"status","--short"],text=True).splitlines()
    out = REVIEW/"C09_REVIEW_CHECKS.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"exit_code":report["exit_code"],"elapsed_seconds":report["elapsed_seconds"],"counts":report["counts"],"output":str(out)}),flush=True)
sys.exit(report["exit_code"])
