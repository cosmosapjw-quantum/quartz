#!/usr/bin/env python3
"""C21 heldout science: matched native Beta task & evaluator plus sensor control.

All C21 source collectors/scorer code was frozen and Git-published at C07.
TRAIN-only resource tariff at C07A, TEST raw separately published at C08A.
No policy is actuated: all task followups are physically executed for labels.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import hashlib,json,random,statistics,subprocess,time
from quartz.experiments.c21_native_session import (
  native_beta_with_real_task,native_sensor_model_batch,source_only_beta,
  sensor_public_case,sha,BETA,SENSOR)
from quartz.experiments.c19_signed_value import shadow_predict as coarse_predict
from quartz.experiments.c20_predictive_value import beta_evsi,sensor_evsi

ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_WORLDS_C21_FROZEN.json"
RAW=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
MODEL=HOME/"FROZEN_C21_RESOURCE_PRICE.json"
C19=ROOT/"docs/research/quartz_general_signed_voc_20261122/SHADOW_SIGNED_MODEL_FROZEN.json"
EXPECTED_SOURCE="036b7c2e77d998558de46ff80845e4b0be382ff4a35d365b03a7174d6c98dad8"
EXPECTED_C19="5fdd172a13d51c4d17dc8e7829fb7c5518421028046ebac6e418a1158adf6735"

def published_head():
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
        raise RuntimeError("heldout score requires clean published scientific code and TEST raw")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
    if head!=remote:raise RuntimeError("GitHub HEAD mismatch before opening heldout")
    return head

def avg(xs):return sum(xs)/len(xs)
def parent_bootstrap(rows,deltas,seed):
    groups=defaultdict(list)
    for r,d in zip(rows,deltas):groups[r["group_id"]].append(float(d))
    assert len(groups)==24
    values=[avg(groups[k]) for k in sorted(groups)]
    rng=random.Random(seed)
    draws=sorted(avg([values[rng.randrange(len(values))] for _ in values]) for _ in range(500))
    return {"n_parent_groups":len(groups),"n_resamples":len(draws),
        "mean_group_delta":avg(values),"descriptive_95pct":[draws[12],draws[487]]}

def run():
    head=published_head()
    assert sha(SOURCE)==EXPECTED_SOURCE
    assert sha(C19)==EXPECTED_C19
    model=json.loads(MODEL.read_text())
    assert model["no_TEST_rows_accessed"] and model["source_world_SHA256"]==sha(SOURCE)
    assert model["prespecified_lambda_sensitivity"]==[0.001,0.004,0.01]
    assert model["source_code_previously_published_git_head"]
    c19=json.loads(C19.read_text())
    rows=[json.loads(s) for s in RAW.read_text().splitlines() if s.strip()]
    assert len(rows)==144 and all(r["split"]=="test" for r in rows)
    assert len({r["group_id"] for r in rows})==48
    beta=[r for r in rows if r["family"]=="beta_arms"]
    sensor=[r for r in rows if r["family"]=="finite_sensor"]
    assert len(beta)==len(sensor)==72
    assert len({r["group_id"] for r in beta})==24
    assert len({r["group_id"] for r in sensor})==24
    natives,bmeta=native_beta_with_real_task(beta)
    sensorm,smeta=native_sensor_model_batch(sensor)

    # Independently recompute mathematical value with old full-grid Python
    # and old exact finite-sensor Python, OUTSIDE the charged native model.
    deviations=[]
    source_data=[]
    for src,x in zip(beta,natives):
        posterior=source_only_beta(src)["posterior"]
        independent=beta_evsi(posterior,10)
        err=abs(independent["evsi"]-x["gross_exact_EVSI"])
        deviations.append(err)
        assert err<=1e-10,(err,src["group_id"])
        source_data.append({
           "group_id":src["group_id"],"rep":src["rep"],"family":"beta_arms",
           "n_actions":src["n_actions"],"gap":src["gap"],
           "observed_signed_true_utility":x["actual_native_source_signed_gain"],
           "observed_decision_revision":x["actual_native_source_revision"],
           "observed_task_us":x["actual_native_extra_task_internal_us"],
           "meta_native_function_us":x["native_predictor_function_internal_us"],
           "meta_native_server_roundtrip_us":x["native_predictor_streaming_roundtrip_us"],
           "predicted_gross_EVSI":x["gross_exact_EVSI"],
           "predictive_mass":x["exact_predictive_mass"],
           "unique_support_points":x["unique_support_points"],
           "resource_bucket":"beta_K"+str(src["n_actions"]),
           "reference_fullgrid_EVSI":independent["evsi"],
           "original_Python_followup_independent_of_Rust_native_workload":True,
           "all_followups_executed":True
        })
    for src,x in zip(sensor,sensorm):
        pub=sensor_public_case(src)
        reference=sensor_evsi(pub["prior"],pub["likelihood"],pub["utilities"],pub["extra"])
        err=abs(reference["evsi"]-x["gross_exact_EVSI"])
        deviations.append(err)
        assert err<=1e-10,(err,src["group_id"])
        source_data.append({
          "group_id":src["group_id"],"rep":src["rep"],"family":"finite_sensor",
          "n_actions":3,"gap":src["gap"],
          "observed_signed_true_utility":float(src["signed_gain"]),
          "observed_decision_revision":int(src["label"]),
          "observed_task_us":float(src["actual_extra_cost_us"]),
          "meta_native_function_us":x["native_predictor_function_internal_us"],
          "meta_native_server_roundtrip_us":smeta["amortized_batch_host_us_per_case"],
          "predicted_gross_EVSI":x["gross_exact_EVSI"],
          "predictive_mass":x["exact_predictive_mass"],
          "unique_support_points":x["unique_support_points"],
          "resource_bucket":"sensor_K3",
          "reference_fullgrid_EVSI":reference["evsi"],
          "source_clock_and_predictor_clock_different_languages":True,
          "all_followups_executed":True})
    assert max(deviations)<=1e-10
    old_rows=[{"gap":r["gap"],"n_actions":r["n_actions"],"compute_kind":r["family"]} for r in source_data]
    historical=coarse_predict(c19["trained_model"],old_rows,
         resource_price=.004,
         cost_by_kind=c19["predicted_incremental_cost_us_by_compute_kind"],
         calibration=c19["calibration_only_intercepts"])
    for r,c in zip(source_data,historical):
        r["historical_C19_coarse_signed_gross"]=c["gross_signed_value"]
    out=HOME/"evidence/C08_TEST/NATIVE_TEST_SHADOW_RECORDS.jsonl"
    out.write_text("".join(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n" for r in source_data))
    results={}
    for family in ("beta_arms","finite_sensor"):
        group=[r for r in source_data if r["family"]==family]
        assert len(group)==72
        G=[r["predicted_gross_EVSI"] for r in group]
        Z=[r["observed_signed_true_utility"] for r in group]
        base=[r["historical_C19_coarse_signed_gross"] for r in group]
        C=[r["observed_task_us"] for r in group]
        Ei=[r["meta_native_function_us"] for r in group]
        Es=[r["meta_native_server_roundtrip_us"] for r in group]
        mse_G=[(x-z)**2 for x,z in zip(G,Z)]
        mse_coarse=[(x-z)**2 for x,z in zip(base,Z)]
        by_family={
          "n_independent_test_parent_worlds":24,
          "n_real_stochastic_extra_task_samples":72,
          "n_changed_terminal_decisions":sum(r["observed_decision_revision"] for r in group),
          "n_negative_realized_utility":sum(z<0 for z in Z),
          "mean_true_signed_gain":avg(Z),
          "mean_exact_predicted_EVSI":avg(G),
          "mean_historical_coarse_predicted_gain":avg(base),
          "exact_EVSI_MSE_against_realized_utility":avg(mse_G),
          "historical_C19_coarse_MSE":avg(mse_coarse),
          "exact_minus_coarse_MSE_group_CI":parent_bootstrap(group,[a-b for a,b in zip(mse_G,mse_coarse)],210021),
          "mean_task_actual_us":avg(C),
          "mean_native_function_internal_eval_us":avg(Ei),
          "mean_native_end_to_end_boundary_us":avg(Es),
          "n_source_only_reference_matches":len(group),
          "time_boundary":("matched Rust native Beta stochastic sampler and Beta EVSI evaluator; stream per case"
                           if family=="beta_arms" else "Python finite-sensor stochastic work and separately compiled Rust sensor; amortized C20 batch host IO"),
          "prices":{}}
        for price in model["prespecified_lambda_sensitivity"]:
            # Policy choices use only source-locked model G, and TRAIN cost estimates.
            always=[z-price*c for z,c in zip(Z,C)]
            sel_un=[];sel_internal=[];sel_stream=[];sel_coarse=[]
            for r in group:
                bucket=model["price_by_declared_workload_bucket"][r["resource_bucket"]]
                c0=bucket["train_task_median_us"]
                ei=bucket["train_predictor_function_median_us"]
                es=bucket["train_predictor_boundary_median_us"]
                sel_un.append(r["predicted_gross_EVSI"]>price*c0)
                sel_internal.append(r["predicted_gross_EVSI"]>price*(c0+ei))
                sel_stream.append(r["predicted_gross_EVSI"]>price*(c0+es))
                sel_coarse.append(r["historical_C19_coarse_signed_gross"]>price*c0)
            def chosen(v,selectors,overhead=None):
                return [(x if active else 0.0)-(price*o if overhead is not None else 0.0)
                     for x,active,o in zip(v,selectors,overhead if overhead is not None else [0.]*len(v))]
            net={
             "NEVER":[0.]*len(group),
             "ALWAYS":always,
             "HISTORICAL_C19_COARSE_UNCHARGED":chosen(always,sel_coarse),
             "C21_EXACT_UNCHARGED":chosen(always,sel_un),
             "C21_NATIVE_FUNCTION_CHARGED":chosen(always,sel_internal,Ei),
             "C21_NATIVE_HOST_BOUNDARY_CHARGED":chosen(always,sel_stream,Es)}
            by_family["prices"][str(price)]={
             "mean_net_utility_gain_vs_immediate_STOP":{name:avg(vals) for name,vals in net.items()},
             "n_recommended_COMPUTE":{"UNCHARGED":sum(sel_un),
                 "NATIVE_FUNCTION":sum(sel_internal),"NATIVE_HOST_BOUNDARY":sum(sel_stream)},
             "n_native_internal_missed_positive_task_gain":
               sum((not active) and v>0 for v,active in zip(always,sel_internal)),
             "n_native_internal_selected_negative_task_gain":
               sum(active and v<0 for v,active in zip(always,sel_internal)),
             "CI_native_function_minus_ALWAYS":parent_bootstrap(group,
                [x-y for x,y in zip(net["C21_NATIVE_FUNCTION_CHARGED"],always)],211001+int(price*1e6)),
             "CI_native_boundary_minus_ALWAYS":parent_bootstrap(group,
                [x-y for x,y in zip(net["C21_NATIVE_HOST_BOUNDARY_CHARGED"],always)],211101+int(price*1e6)),
             "CI_native_function_minus_NEVER":parent_bootstrap(group,
                net["C21_NATIVE_FUNCTION_CHARGED"],211201+int(price*1e6))}
        results[family]=by_family
    receipt={"schema":"quartz_C21_C08_scoped_heldout_native_exact_EVSI_v1",
      "source_code_clean_remote_HEAD_before_scoring":head,
      "new_SOURCE_C21_SHA256":sha(SOURCE),
      "TRAIN_only_price_model_SHA256":sha(MODEL),
      "TEST_raw_source_SHA256":sha(RAW),
      "native_TEST_shadow_records_SHA256":sha(out),
      "C21_beta_Rust_executable_SHA256":sha(BETA),
      "C20_sensor_Rust_reference_SHA256":sha(SENSOR),
      "historical_C19_frozen_model_SHA256":sha(C19),
      "n_new_independent_TEST_parent_groups":48,
      "n_actual_TEST_task_followups":144,
      "max_abs_EVSI_discrepancy_vs_separate_Python_reference":max(deviations),
      "beta_native_server_first_request_cold_us":bmeta["server_first_request_cold_start_us"],
      "finite_sensor_amortized_native_batch_host_us_per_case":smeta["amortized_batch_host_us_per_case"],
      "predefined_primary_lambda":0.004,"price_sensitivity":[0.001,0.004,0.01],
      "all_source_tasks_actually_executed":True,
      "shadow_not_actuated":True,
      "real_hardware_work_avoided":"NOT_MEASURED",
      "real_neural_MCTS_signed_utility":"NOT_ESTIMABLE",
      "generalized_model_calibration":"NOT_ESTABLISHED",
      "domains":results}
    report=HOME/"evidence/C08_TEST/C08_HELDOUT_NATIVE_EVSI_RESULTS.json"
    report.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    compact={k:{
       "gross":v["mean_exact_predicted_EVSI"],"realized_Z":v["mean_true_signed_gain"],
       "mse":v["exact_EVSI_MSE_against_realized_utility"],
       "task_us":v["mean_task_actual_us"],"meta_internal_us":v["mean_native_function_internal_eval_us"],
       "meta_boundary_us":v["mean_native_end_to_end_boundary_us"],
       "primary_net":v["prices"]["0.004"]["mean_net_utility_gain_vs_immediate_STOP"],
       "CI_internal_vs_ALWAYS":v["prices"]["0.004"]["CI_native_function_minus_ALWAYS"]["descriptive_95pct"]
     } for k,v in results.items()}
    print("C21_HELDOUT",json.dumps({"n_test":144,
       "max_EVSI_reference_error":max(deviations),
       "new_SOURCE_SHA256":sha(SOURCE),"domains":compact},sort_keys=True))
if __name__=="__main__":run()
