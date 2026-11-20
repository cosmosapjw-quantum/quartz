#!/usr/bin/env python3
"""Freeze fixed checkpoint-time comparators from VALIDATION root costs only.

A cap is checked AFTER each completed MCTS checkpoint. Stage overshoot is
unavoidable, and is measured. The cap is never fitted to test outcomes or
test latencies.
"""
from pathlib import Path
import hashlib
import json
import subprocess

TIME_CAP_GRID_MS=[x/4.0 for x in range(1,161)]  # 0.25..40ms fixed before test


def sha(p:Path)->str:
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def selected_budget_and_cost(row,cap_ms):
    costs=row['search_stage_wall_ms']
    assert len(costs)==4 and all(x>=0 for x in costs)
    consumed=0.0
    for idx,c in enumerate(costs):
        consumed+=c
        if idx==3 or consumed>=cap_ms:
            return (8,16,32,64)[idx],consumed
    raise RuntimeError('unreachable time cap')


def main():
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('frozen cap selection requires committed clean source')
    data=d/'evidence/train_val_2496_20261121'
    trained=d/'evidence/frozen_surrogate_20261121'
    rows=json.loads((data/'joined_train_val.json').read_text())
    s=json.loads((trained/'frozen_surrogates.json').read_text())
    assert len(rows)==2496
    result={
        'schema':'quartz_cog02_validation_only_time_cap_v1',
        'policy_sha256':sha(trained/'frozen_surrogates.json'),
        'training_trajectories_SHA256':sha(data/'joined_train_val.json'),
        'baseline_selection_rule':'min_abs_mean_search_ms_difference_from_validation_adaptive; tie cap smaller',
        'cap_candidates_ms':TIME_CAP_GRID_MS,
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'modes':{},'test_family_used':False,
        'time_cap_decision':'stop when cumulative COMPLETED checkpoint search time reaches cap; overshoot allowed'
    }
    for mode,policy in s['mode_policies'].items():
        val=[r for r in rows if r['mode']==mode and r['split']=='val']
        assert len(val)==192 and len({r['family'] for r in val})==12
        target=policy['selected_validation_policy']['mean_search_ms']
        options=[]
        for cap in TIME_CAP_GRID_MS:
            observations=[selected_budget_and_cost(r,cap) for r in val]
            measured=sum(x[1] for x in observations)/len(observations)
            options.append({
                'cap_ms':cap,
                'validation_mean_completed_search_ms':measured,
                'validation_mean_stopped_visits':sum(x[0] for x in observations)/len(observations),
                'validation_stop_distribution':{str(b):sum(x[0]==b for x in observations) for b in (8,16,32,64)},
                'absolute_cost_difference_ms':abs(measured-target),
            })
        best=min(options,key=lambda x:(x['absolute_cost_difference_ms'],x['cap_ms']))
        result['modes'][mode]={
            'adaptive_validation_mean_search_ms_target':target,
            'selected_cap_ms':best['cap_ms'],
            'fixed_validation_mean_search_ms':best['validation_mean_completed_search_ms'],
            'fixed_validation_stop_distribution':best['validation_stop_distribution'],
            'absolute_validation_cost_mismatch_ms':best['absolute_cost_difference_ms'],
            'validation_relative_mismatch_fraction':best['absolute_cost_difference_ms']/max(target,1e-9),
            'validation_cap_not_selected_by_WDL_or_regret':True,
        }
    out=d/'evidence/validation_time_caps_20261121'
    out.mkdir(parents=True,exist_ok=False)
    p=out/'TIME_CAPS_FROZEN.json'
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    m={'schema':'quartz_cog02_time_budget_validation_manifest_v1',
       'source_commit':result['source_commit'],
       'frozen_time_caps_sha256':sha(p),
       'all_time_caps_from_train_validation_only':True,
       'mode_caps_ms':{k:v['selected_cap_ms'] for k,v in result['modes'].items()}}
    (out/'time_cap_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print('COG02_FIXED_TIME_CAPS_FROM_VALIDATION',json.dumps(result['modes'],sort_keys=True))


if __name__=='__main__':main()
