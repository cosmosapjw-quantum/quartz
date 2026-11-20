#!/usr/bin/env python3
"""Independent train-only ridge and validation-threshold reproducibility audit.

This audit never loads heldout Connect4 test rows or uses exact test labels.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def evaluate(p,rows):
    # Stagewise sequential choices using only saved *current* root inputs.
    selections=[]
    for r in rows:
        stage=3
        for i in range(3):
            x=np.asarray(r['features'][i],dtype=float)
            pred=float(np.dot(np.asarray(p['coefficients'][i],dtype=float),x))
            if pred<=p['lambda_wdl_per_ms']*p['forecast_cost_ms'][i]:
                stage=i
                break
        cpu=sum(r['search_stage_wall_ms'][:stage+1])
        selections.append((stage,float(r['utility'][stage]),cpu))
    n=len(rows)
    return {
        'mean_U':sum(v[1] for v in selections)/n,
        'mean_search_ms':sum(v[2] for v in selections)/n,
        'mean_objective':sum(v[1]-0.04*v[2] for v in selections)/n,
        'stops':dict(Counter(str([8,16,32,64][v[0]]) for v in selections)),
    }


def main():
    d=Path(__file__).resolve().parent
    prior=d/'evidence/train_val_2496_20261121'
    fit=d/'evidence/frozen_surrogate_20261121'
    data=json.loads((prior/'joined_train_val.json').read_text())
    model=json.loads((fit/'frozen_surrogates.json').read_text())
    m=json.loads((fit/'surrogate_manifest.json').read_text())
    assert sha(fit/'frozen_surrogates.json')==m['surrogates_SHA256']
    assert sha(prior/'joined_train_val.json')==m['trajectories_input_SHA256']
    assert len(data)==2496 and model['test_set_used_for_selection']==False
    assert model['fixed_cost_price_WDL_per_ms']==0.04
    reports={}
    for mode in ['uniform','new_mlp','conv']:
        train=[r for r in data if r['mode']==mode and r['split']=='train']
        val=[r for r in data if r['mode']==mode and r['split']=='val']
        assert len(train)==640 and len(val)==192
        assert not {r['family'] for r in train}&{r['family'] for r in val}
        p=model['mode_policies'][mode]
        assert p['source_train_roots']==640 and p['source_val_roots']==192
        for stage in range(3):
            X=np.asarray([r['features'][stage] for r in train],dtype=float)
            y=np.asarray([r['gains'][stage] for r in train],dtype=float)
            penalty=np.eye(9,dtype=float)
            penalty[0,0]=0.
            # Numerically independent verification of ridge's penalized
            # normal equation, no production fit helper called.
            w=np.asarray(p['coefficients'][stage])
            residual=np.linalg.norm(X.T@(X@w-y)+penalty@w)
            assert residual<2e-7,(mode,stage,residual)
            cost=np.mean([r['next_wall_ms'][stage] for r in train])
            assert abs(cost-p['forecast_cost_ms'][stage])<1e-9
        criterion=[]
        for lam in model['fixed_candidate_lambdas']:
            score=evaluate({**p,'lambda_wdl_per_ms':lam},val)
            criterion.append((score['mean_objective'],-score['mean_search_ms'],lam,score))
        _,_,best_lam,best_val=max(criterion,key=lambda item:item[:3])
        assert best_lam==p['lambda_wdl_per_ms']
        expected=p['selected_validation_policy']
        assert abs(best_val['mean_U']-expected['mean_U'])<1e-12
        assert abs(best_val['mean_search_ms']-expected['mean_search_ms'])<1e-12
        assert best_val['stops']==expected['stop_budgets']
        reports[mode]={'lambda':best_lam,
            'validation_stop_budgets':best_val['stops'],
            'validation_mean_search_ms':best_val['mean_search_ms'],
            'ridge_normal_equation_checked':True,
            'training_families':40,'validation_families':12}
    audit={
        'schema':'quartz_cog02_costed_signed_gain_independent_validation_v1',
        'train_val_signed_label_rows':2496,
        'strictly_train_only_ridge_and_cost_forecast':True,
        'strictly_validation_only_lambda_selection':True,
        'no_test_family_used':True,
        'surrogate_model_SHA256':m['surrogates_SHA256'],
        'roles':reports,
        'negative_findings':['new MLP chosen lambda stops ALL validation roots at budget8'],
    }
    p=fit/'INDEPENDENT_SURROGATE_AUDIT.json'
    if p.exists():raise FileExistsError('refusing duplicate audit')
    p.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT_SIGNED_GAIN_SURROGATES_AUDIT_PASS',json.dumps(reports,sort_keys=True))


if __name__=='__main__':main()
