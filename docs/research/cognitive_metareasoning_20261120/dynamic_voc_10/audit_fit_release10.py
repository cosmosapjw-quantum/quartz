#!/usr/bin/env python3
"""Independent Ridge normal-equation, positive cost and validation selection readback.

Does not import the production fitting or stop-decision functions and does not
consult any prior/new test family to optimize models.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import numpy as np


LAMBDA_GRID=[0.,.25,.5,1.,2.,4.]
CAPS=[j/100 for j in range(1,1001)]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def predicted(fitted,x,stage,long=False):
    g='two_step' if long else 'one_step'
    gain=float(sum(a*b for a,b in zip(fitted[f'{g}_gain_coefficients'][stage],x)))
    log=float(sum(a*b for a,b in zip(fitted[f'{g}_cost_log_coefficients'][stage],x)))
    cost=max(1e-6,math.exp(max(-20,min(10,log)))*fitted[f'{g}_cost_scale'][stage])
    return gain,cost


def run(fitted,row,variant):
    stage=0;decisions=[]
    while stage<3:
        x=row['features'][stage]
        gain,cost=predicted(fitted,x,stage)
        single=gain-fitted['lambda_wdl_per_ms']*cost
        two=None
        if variant=='dynamic' and stage<2:
            gg,cc=predicted(fitted,x,stage,long=True)
            two=gg-fitted['lambda_wdl_per_ms']*cc
        if two is not None and two>max(single,0):
            decisions.append(('commit_two',stage))
            stage+=2
        elif single>0:
            decisions.append(('single',stage))
            stage+=1
        else:
            decisions.append(('stop',stage))
            break
    return stage,row['utility'][stage],sum(row['search_stage_wall_ms'][:stage+1]),decisions


def evaluate(fitted,rows,variant):
    outcomes=[run(fitted,r,variant) for r in rows]
    n=len(outcomes)
    mU=sum(x[1] for x in outcomes)/n
    ms=sum(x[2] for x in outcomes)/n
    return {'objective':mU-ms,'mean_U':mU,'mean_search_ms':ms,
            'stop_visits':dict(Counter(str([8,16,32,64][x[0]]) for x in outcomes)),
            'commits':sum(sum(a=='commit_two' for a,_ in x[3]) for x in outcomes)}


def main():
    d=Path(__file__).resolve().parent
    source=d/'evidence/release_train_val_2496_20261121'
    fit=d/'evidence/frozen_release_policies_20261121'
    original=json.loads((source/'JOINED_TRAIN_VAL_RELEASE.json').read_text())
    models=json.loads((fit/'FROZEN_RELEASE_POLICIES.json').read_text())
    receipt=json.loads((fit/'FIT_MANIFEST.json').read_text())
    assert sha(fit/'FROZEN_RELEASE_POLICIES.json')==receipt['output_policy_sha256']
    assert sha(source/'JOINED_TRAIN_VAL_RELEASE.json')==receipt['input_2496_sha256']
    assert len(original)==2496
    assert models['used_old_or_new_test_labels_for_fit'] is False
    assert models['source_commit']=='f10dc235dd12b0710a9dc1f4d7e5cbfde790f173'
    verification={}
    for mode in ('uniform','new_mlp','conv'):
        train=[x for x in original if x['mode']==mode and x['split']=='train']
        val=[x for x in original if x['mode']==mode and x['split']=='val']
        assert len(train)==640 and len(val)==192
        assert len({r['family'] for r in train})==40 and len({r['family'] for r in val})==12
        assert not ({r['family'] for r in train}&{r['family'] for r in val})
        data=models['modes'][mode]
        fitted=data['frozen_policies']['dynamic']
        for name,num in [('one_step',3),('two_step',2)]:
            for stage in range(num):
                X=np.asarray([r['features'][stage] for r in train],dtype=np.float64)
                U=np.asarray([r['utility'] for r in train],dtype=float)
                Wall=np.asarray([r['search_stage_wall_ms'] for r in train],dtype=float)
                gain=U[:,stage+(2 if name=='two_step' else 1)]-U[:,stage]
                cost=(Wall[:,stage+1]+Wall[:,stage+2]) if name=='two_step' else Wall[:,stage+1]
                positive=np.maximum(.001,cost)
                alpha=np.eye(9);alpha[0,0]=0
                w=np.asarray(fitted[f'{name}_gain_coefficients'][stage])
                wc=np.asarray(fitted[f'{name}_cost_log_coefficients'][stage])
                gradient=np.linalg.norm(X.T@(X@w-gain)+alpha@w)
                cost_gradient=np.linalg.norm(X.T@(X@wc-np.log(positive))+alpha@wc)
                assert gradient<1e-6,(mode,stage,name,gradient)
                assert cost_gradient<1e-6,(mode,stage,name,cost_gradient)
                pred=np.exp(np.clip(X@wc,-20,10))*fitted[f'{name}_cost_scale'][stage]
                assert abs(pred.mean()-positive.mean())<1e-8
        selected={}
        for variant in ('myopic','dynamic'):
            tab=data['validation'][variant]['lambda_grid_results']
            assert len(tab)==len(LAMBDA_GRID)==6
            for item,lam in zip(tab,LAMBDA_GRID):
                assert item['lambda']==lam
                pred=evaluate({**fitted,'lambda_wdl_per_ms':lam},val,variant)
                for key in ('objective','mean_U','mean_search_ms'):
                    assert abs(pred[key]-item['val'][key])<1e-10,(mode,variant,key)
                assert pred['stop_visits']==item['val']['stop_visits']
                assert pred['commits']==item['val']['chosen_joint_commit_count']
            best=max(tab,key=lambda r:(r['val']['objective'],-r['val']['mean_search_ms'],r['lambda']))
            assert best['lambda']==data['validation'][variant]['selected_lambda']
            selected[variant]=best['lambda']
        assert data['frozen_policies']['myopic']['two_step_gain_coefficients']==[[0.]*9]*2
        assert data['frozen_policies']['myopic_time_match']['two_step_gain_coefficients']==[[0.]*9]*2
        target=data['dynamic_val_mean_search_ms']
        costs=[]
        for cap in CAPS:
            stopped=[]
            for r in val:
                cumulative=0
                for i,s in enumerate(r['search_stage_wall_ms']):
                    cumulative+=s
                    if i==3 or cumulative>=cap:
                        stopped.append(cumulative);break
            costs.append((abs(sum(stopped)/len(stopped)-target),cap))
        best=min(costs)
        assert best[1]==data['clock_validation_time_match']['cap_ms']
        verification[mode]={
            'ridge_gradients_small':True,'release_positive_cost_scaled_to_train_mean':True,
            'selected_val_lambda':selected,
            'validation_cap_ms':best[1],
            'dynamic_val_commits':data['validation']['dynamic']['selected_val']['chosen_joint_commit_count'],
            'test_data_was_not_read':True}
    report={
        'schema':'quartz_cog02_dynamic10_ridge_and_val_independent_v1',
        'source_commit':receipt['source_commit'],
        'input_train_val_SHA256':receipt['input_2496_sha256'],
        'model_sha256':receipt['output_policy_sha256'],
        'training_roots_per_mode':640,'validation_roots_per_mode':192,
        'all_normal_equations_and_train_positive_costs_checked':True,
        'val_best_lambda_clock_cost_recomputed':True,
        'mode_receipts':verification,
        'no_new_holdout_used_for_selection':True,
    }
    p=fit/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('refuse duplicate')
    p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('C10_RELEASE_RIDGE_COST_AND_VAL_INDEPENDENT_AUDIT_PASS',json.dumps(verification,sort_keys=True))


if __name__=='__main__':main()
