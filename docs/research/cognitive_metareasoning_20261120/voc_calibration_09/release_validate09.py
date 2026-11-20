#!/usr/bin/env python3
"""Live RELEASE-binary validation-only CPU time comparator calibration.

Prior signed-gain policy, neural weights and lambda remain byte-frozen. Re-run
only original 192 VALIDATION-family roots in release mode to find fixed-clock
caps using actual CPU search time. No val or test exact oracle labels consulted.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time


GRID_CAPS_MS=[i/100.0 for i in range(1,1001)]  # 0.01..10ms, fixed before run


def modules():
    d=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(d) not in sys.path:
        sys.path.insert(0,str(d))
    from evaluate_live08 import model_request,run_batch,compare_rust_current_policy
    from train_gain08 import model_identity,sha
    return model_request,run_batch,compare_rust_current_policy,model_identity,sha


def stop_from_full(stage_us,cap_ms):
    if len(stage_us)!=4 or any(type(x) is not int or x<0 for x in stage_us):
        raise ValueError('full Rust stage wall microseconds required')
    if not math.isfinite(cap_ms) or cap_ms<=0:
        raise ValueError('invalid CPU time cap')
    total=0.0
    for i,v in enumerate(stage_us):
        total+=v/1000.0
        if i==3 or total>=cap_ms:
            return [8,16,32,64][i],total
    raise RuntimeError('unreachable time cap')


def choose_cap(roots,target_ms):
    if not roots or not math.isfinite(target_ms) or target_ms<=0:
        raise ValueError('positive target and full validation roots required')
    stage_costs=[[x['step_wall_us'] for x in row['steps']] for row in roots]
    candidates=[]
    for cap in GRID_CAPS_MS:
        stops=[stop_from_full(cost,cap) for cost in stage_costs]
        mean_ms=sum(x[1] for x in stops)/len(stops)
        candidates.append({
            'cap_ms':cap,
            'mean_search_ms':mean_ms,
            'mean_stopped_visits':sum(x[0] for x in stops)/len(stops),
            'stop_visits':dict(Counter(str(x[0]) for x in stops)),
            'absolute_cost_mismatch_ms':abs(mean_ms-target_ms),
        })
    picked=min(candidates,key=lambda x:(x['absolute_cost_mismatch_ms'],x['cap_ms']))
    return picked


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--binary',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('require committed clean source before release validation run')
    model_request,run_batch,compare_rust_current_policy,model_identity,sha=modules()
    old=d.parent/'cost_matched_voc_08'
    study=d.parent/'connect4_generalization_07'
    policies=json.loads((old/'evidence/frozen_surrogate_20261121/frozen_surrogates.json').read_text())
    assert policies['test_set_used_for_selection'] is False
    cohort=json.loads((study/'COHORT_1024_FROZEN.json').read_text())
    cases=[r for r in cohort['rows'] if r['split']=='val']
    assert len(cases)==192 and len({r['family'] for r in cases})==12
    paths=model_identity(repo)
    modelhashes={mode:sha(file) for mode,file in paths.items()}
    exe=args.binary.resolve()
    if not exe.is_file():raise FileNotFoundError(exe)
    requests=[]
    order=[]
    for row in cases:
        for mode in ('uniform','new_mlp','conv'):
            p=policies['mode_policies'][mode]
            for arm in ('full','adaptive'):
                requests.append(model_request(row,mode,arm,paths,modelhashes,
                                 p if arm=='adaptive' else None,None))
                order.append((row['id'],mode,arm))
    assert len(requests)==1152
    assert all('oracle' not in req and 'exact_action_scores' not in req for req in requests)
    start=time.perf_counter()
    replies=run_batch(exe,requests)
    elapsed=time.perf_counter()-start
    seen={}
    for (ident,mode,arm),result in zip(order,replies):
        if result['eval_mode']!={'uniform':'uniform_no_neural_network',
                                'new_mlp':'trained_tiny_mlp_cpu',
                                'conv':'trained_conv3x3_cpu'}[mode]:
            raise ValueError('research evaluator identity changed')
        seen[(ident,mode,arm)]=result
        if arm=='adaptive':
            compare_rust_current_policy(result,policies['mode_policies'][mode])
        if arm=='full' and (len(result['steps'])!=4 or result['fresh16']['resident_match'] is not True):
            raise ValueError('full root search did not reach64')
    for root in cases:
        for mode in ('uniform','new_mlp','conv'):
            full=seen[(root['id'],mode,'full')]
            adapted=seen[(root['id'],mode,'adaptive')]
            for a,b in zip(full['steps'],adapted['steps']):
                if (a['best_action']!=b['best_action'] or a['visit_policy']!=b['visit_policy']
                    or a['evaluator_calls']!=b['evaluator_calls']):
                    raise RuntimeError('live adaptive prefix differs from full deterministic root')
    caps={}
    for mode in ('uniform','new_mlp','conv'):
        full=[seen[(row['id'],mode,'full')] for row in cases]
        live=[seen[(row['id'],mode,'adaptive')] for row in cases]
        livecost=[sum(q['step_wall_us'] for q in row['steps'])/1000.0 for row in live]
        mean_cost=sum(livecost)/len(livecost)
        selection=choose_cap(full,mean_cost)
        caps[mode]={
            **selection,
            'adaptive_release_validation_mean_search_ms':mean_cost,
            'relative_validation_cost_mismatch':abs(selection['mean_search_ms']-mean_cost)/mean_cost,
            'live_validation_adaptive_stop_visits':dict(Counter(str(row['stopped_at_visits']) for row in live)),
            'original_voc08_fixed_lambda':policies['mode_policies'][mode]['lambda_wdl_per_ms'],
            'original_voc08_cost_forecast_ms':policies['mode_policies'][mode]['forecast_cost_ms'],
        }
    output={
        'schema':'quartz_cog02_voc09_release_validation_clock_caps_v1',
        'source_git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'release_rust_binary_sha256':sha(exe),
        'release_binary_built_from_commit':'1c2b00c1bad2585380fee97bde65e3862dc1f25d',
        'prior_policies_sha256':sha(old/'evidence/frozen_surrogate_20261121/frozen_surrogates.json'),
        'old_validation_roots':192,'old_validation_families':12,
        'source_old_validation_cohort_sha256':sha(study/'COHORT_1024_FROZEN.json'),
        'no_oracle_label_used':True,
        'no_retrained_or_rethresholded_voc_predictor':True,
        'fixed_grid_caps_ms':[0.01,10.0,0.01],
        'modes':caps,
        'timing_scope':'release Rust actual MCTS search wall; model load amortized in same process',
        'total_release_validation_wall_s':elapsed,
        'no_new_test_cohort_access':True,
    }
    for mode,file in paths.items():
        if sha(file)!=modelhashes[mode]:
            raise RuntimeError('frozen neural weights changed during release timing')
    folder=args.output_dir.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'RAW_1152_VALIDATION.json').write_text(json.dumps(replies,sort_keys=True)+'\n')
    (folder/'VALIDATION_RELEASE_CAPS.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    manifest={
        'schema':'quartz_cog02_voc09_release_validation_execution_manifest_v1',
        'source_commit':output['source_git_head'],
        'executable_sha256':sha(exe),
        'old_policy_sha256':output['prior_policies_sha256'],
        'number_real_val_requests':len(replies),
        'new_cohort_label_accessed':False,
        'validation_output_sha256':{p.name:sha(p) for p in folder.iterdir()},
    }
    (folder/'execution_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('CAL09_RELEASE_VALIDATION_CALIBRATED',json.dumps({
        'n_actual_rust_validation_requests':len(replies),'runtime_s':elapsed,
        'modes':caps,
    },sort_keys=True))


if __name__=='__main__':main()
