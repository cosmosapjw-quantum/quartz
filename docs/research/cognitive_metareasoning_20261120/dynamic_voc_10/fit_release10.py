#!/usr/bin/env python3
"""Freeze actual release CPU one-step/two-step signed-gain policies using G07 train/val ONLY.

Test G07 and both C09/C10 new families remain unavailable to model fitting.
The dynamic and one-step policies share identical stagewise Ridge and cost
models; only the two-step commit option differs.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import subprocess

from dynamic_model10 import (
    BUDGETS, fit_dynamic_from_train, simulate_bounded_policy,
)


LAMBDAS=(0.0,0.25,0.5,1.0,2.0,4.0)
VAL_COST_PRICE=1.0
CLOCK_GRID=[j/100.0 for j in range(1,1001)]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()


def checkpoint_cap(wall,cap):
    if len(wall)!=4 or not math.isfinite(cap) or cap<=0:
        raise ValueError('four fully observed MCTS time stages and positive cap required')
    if any(not math.isfinite(t) or t<0 for t in wall):
        raise ValueError('nonfinite/negative CPU clock batch time')
    elapsed=0.
    for i,stage_ms in enumerate(wall):
        elapsed+=stage_ms
        if elapsed>=cap or i==3:
            return BUDGETS[i],elapsed
    raise RuntimeError('unreachable checkpoint cap')


def select_validation_candidate(table):
    if not table:raise ValueError('validation candidate table cannot be empty')
    return max(table,key=lambda q:(
        q['val']['objective'],
        -q['val']['mean_search_ms'],
        q['lambda']))


def report_validation(model,rows,variant):
    outputs=[simulate_bounded_policy(model,row,variant=variant) for row in rows]
    n=len(rows)
    utility=sum(o['utility'] for o in outputs)/n
    ms=sum(o['search_ms'] for o in outputs)/n
    regret=sum(row['root_exact_WDL']-out['utility'] for row,out in zip(rows,outputs))/n
    return {
        'n_rows':n,'n_distinct_families':len({x['family'] for x in rows}),
        'mean_U':utility,'mean_regret':regret,
        'mean_search_ms':ms,
        'objective':utility-VAL_COST_PRICE*ms,
        'mean_visits':sum(o['visits'] for o in outputs)/n,
        'stop_visits':dict(Counter(str(o['visits']) for o in outputs)),
        'chosen_joint_commit_count':sum(len(o['skipped_stages']) for o in outputs),
        'negative_realized_stop_regret_count':sum(row['root_exact_WDL']-out['utility']>0
            for row,out in zip(rows,outputs)),
    }


def select_time_cap(rows,target_ms):
    if not rows or not math.isfinite(target_ms) or target_ms<=0:
        raise ValueError('val source and matched positive time required')
    results=[]
    for cap in CLOCK_GRID:
        cost=[checkpoint_cap(r['search_stage_wall_ms'],cap) for r in rows]
        mean=sum(v[1] for v in cost)/len(cost)
        results.append({
            'cap_ms':cap,
            'validation_mean_search_ms':mean,
            'validation_stop_distribution':dict(Counter(str(x[0]) for x in cost)),
            'difference_ms':abs(mean-target_ms),
        })
    return min(results,key=lambda x:(x['difference_ms'],x['cap_ms']))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--trajectories',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if git(repo,'status','--porcelain'):
        raise RuntimeError('freeze model only from committed clean source')
    train_source=args.trajectories.resolve()
    receipts=json.loads((train_source/'EXECUTION_MANIFEST.json').read_text())
    if receipts.get('number_real_Rust_searches')!=2496:
        raise RuntimeError('missing full valid 2496 source')
    if sha(train_source/'JOINED_TRAIN_VAL_RELEASE.json')!=receipts['files_sha256']['JOINED_TRAIN_VAL_RELEASE.json']:
        raise RuntimeError('modified training source after release search')
    rows=json.loads((train_source/'JOINED_TRAIN_VAL_RELEASE.json').read_text())
    if len(rows)!=2496:raise RuntimeError('incomplete actual-trace training')
    result={
        'schema':'quartz_cog02_dynamic10_release_train_val_frozen_v1',
        'source_commit':git(repo,'rev-parse','HEAD'),
        'actual_release_train_source_sha256':sha(train_source/'JOINED_TRAIN_VAL_RELEASE.json'),
        'fixed_validation_price_WDL_per_ms':VAL_COST_PRICE,
        'fixed_lambda_grid_WDL_per_ms':list(LAMBDAS),
        'clock_cap_grid_ms':[0.01,10.0,0.01],
        'fit_source':'G07 40 train families only; 12 validation families only for lambda and time cap',
        'used_old_or_new_test_labels_for_fit':False,
        'no_Bayesian_VOC_theory_claim':True,
        'modes':{},
    }
    for mode in ('uniform','new_mlp','conv'):
        tr=[r for r in rows if r['mode']==mode and r['split']=='train']
        val=[r for r in rows if r['mode']==mode and r['split']=='val']
        if len(tr)!=640 or len(val)!=192:
            raise RuntimeError(f'incomplete or contaminated train-val partition {mode}')
        tr_family={r['family'] for r in tr}
        val_family={r['family'] for r in val}
        assert len(tr_family)==40 and len(val_family)==12 and not (tr_family&val_family)
        fitted=fit_dynamic_from_train(tr)
        tables={}
        for variant in ('dynamic','myopic'):
            candidates=[]
            for lam in LAMBDAS:
                frozen={**fitted,'lambda_wdl_per_ms':lam}
                candidates.append({
                    'lambda':lam,'val':report_validation(frozen,val,variant),
                })
            selected=select_validation_candidate(candidates)
            tables[variant]={
                'lambda_grid_results':candidates,
                'selected_lambda':selected['lambda'],
                'selected_val':selected['val'],
            }
        dynamic=copy.deepcopy(fitted)
        dynamic['lambda_wdl_per_ms']=tables['dynamic']['selected_lambda']
        myopic=copy.deepcopy(fitted)
        myopic['lambda_wdl_per_ms']=tables['myopic']['selected_lambda']
        # The Rust endpoint uses one JSON two-option policy for both. Zero
        # joint coefficients mean only myopic one-step values affect choice.
        myopic['two_step_gain_coefficients']=[[0.0]*9 for _ in range(2)]
        target_time=tables['dynamic']['selected_val']['mean_search_ms']
        clock=select_time_cap(val,target_time)
        # Cost-match the myopic variant using validation CPU time ONLY.
        # Do not use validation WDL utility to choose this extra comparator.
        match_candidate=min(tables['myopic']['lambda_grid_results'],key=lambda x:(
            abs(x['val']['mean_search_ms']-target_time),x['lambda']))
        myopic_time_match=copy.deepcopy(myopic)
        myopic_time_match['lambda_wdl_per_ms']=match_candidate['lambda']
        for name,model in [('dynamic',dynamic),('myopic',myopic),
                           ('myopic_time_match',myopic_time_match)]:
            # The production COG10 Rust parser must read this exact schema.
            for field,num in [('one_step_gain_coefficients',3),
                              ('two_step_gain_coefficients',2),
                              ('one_step_cost_log_coefficients',3),
                              ('two_step_cost_log_coefficients',2)]:
                matrix=model[field]
                assert len(matrix)==num and all(len(v)==9 for v in matrix)
            assert model['schema']=='quartz_cog02_dynamic_two_step_v1'
            assert model['lambda_wdl_per_ms'] in LAMBDAS
        result['modes'][mode]={
            'frozen_policies':{'dynamic':dynamic,'myopic':myopic,
                               'myopic_time_match':myopic_time_match},
            'validation':tables,
            'clock_validation_time_match':clock,
            'myopic_time_match_lambda':match_candidate['lambda'],
            'myopic_time_match_validation_mean_search_ms':
                match_candidate['val']['mean_search_ms'],
            'dynamic_val_mean_search_ms':target_time,
            'conditional_training_diagnostics':fitted['train_diagnostics'],
            'n_train':640,'n_val':192,
            'test_sealed_until_after_model_freeze':True,
        }
        print('C10_FITTED',mode,json.dumps({
            'lambda_dynamic':dynamic['lambda_wdl_per_ms'],
            'lambda_myopic':myopic['lambda_wdl_per_ms'],
            'lambda_myopic_time_match':myopic_time_match['lambda_wdl_per_ms'],
            'val_dynamic':tables['dynamic']['selected_val'],
            'val_myopic':tables['myopic']['selected_val'],
            'clock_cap_ms':clock['cap_ms'],
            'clock_cost_diff_ms':clock['difference_ms'],
        },sort_keys=True),flush=True)
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    p=out/'FROZEN_RELEASE_POLICIES.json'
    p.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    manifest={
        'schema':'quartz_cog02_dynamic10_release_policy_fit_manifest_v1',
        'source_commit':result['source_commit'],
        'input_2496_sha256':result['actual_release_train_source_sha256'],
        'output_policy_sha256':sha(p),
        'n_training_roots_per_mode':640,'n_validation_roots_per_mode':192,
        'train_only_gain_and_logcost_ridge':True,
        'validation_only_lambda_and_cpu_time_cap':True,
        'new_holdout_g09_and_g10_never_accessed':True,
    }
    (out/'FIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('C10_RELEASE_POLICY_FIT_DONE',json.dumps({
        m:{'lambda_dynamic':v['frozen_policies']['dynamic']['lambda_wdl_per_ms'],
           'lambda_myopic':v['frozen_policies']['myopic']['lambda_wdl_per_ms'],
           'lambda_costmatched':v['myopic_time_match_lambda'],
           'clock_cap_ms':v['clock_validation_time_match']['cap_ms'],
           'selected_val_utility':v['validation']['dynamic']['selected_val']['mean_U'],
           'selected_val_cost':v['validation']['dynamic']['selected_val']['mean_search_ms']}
        for m,v in result['modes'].items()},sort_keys=True))


if __name__=='__main__':main()
