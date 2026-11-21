#!/usr/bin/env python3
"""Freeze validation-only *randomized blocked* myopic-time-matched comparator.

Model selection uses ONLY G07 original 192 validation-root measured CPU time,
never heldout C09/C10/new-C11 WDL, regret, or timing. The test assignment uses
predeclared SHA256 root/family keyed pseudorandom ranks, not MCTS search noise.
Each of two 12-family test cohorts has a fixed quota of expensive one-step
policies per family. This reduces accidental cost imbalance from random draws.

No new heldout family generated/accessed by the mixture freeze command.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import subprocess


ASSIGNMENT_SEEDS=(202611210101,202611210102,202611210103)
ALLOWED_LAMBDAS=(0.,0.25,0.5,1.,2.,4.)


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def select_validation_cost_bracket(candidates,target_ms,preferred_lambda=None):
    if not math.isfinite(target_ms) or target_ms<=0 or not candidates:
        raise ValueError('positive target and nonempty original validation costs required')
    costs=[]
    for row in candidates:
        lam=float(row['lambda'])
        cost=float(row['cost_ms'])
        if not math.isfinite(lam) or lam not in ALLOWED_LAMBDAS or not math.isfinite(cost) or cost<=0:
            raise ValueError('bad validation candidate/cost')
        costs.append({'lambda':lam,'cost_ms':cost})
    if len({r['lambda'] for r in costs})!=len(costs):
        raise ValueError('duplicate myopic lambda model')
    exact=[r for r in costs if abs(r['cost_ms']-target_ms)<=1e-12]
    if exact:
        selected=min(exact,key=lambda r:(r['lambda']!=preferred_lambda,-r['lambda']))
        return {
            'lower_cpu_lambda':selected['lambda'],
            'higher_cpu_lambda':selected['lambda'],
            'lower_cpu_cost_ms':selected['cost_ms'],
            'higher_cpu_cost_ms':selected['cost_ms'],
            'higher_cpu_probability':0.0,
            'expected_mean_ms':selected['cost_ms'],
            'target_dynamic_validation_mean_ms':target_ms,
            'validation_cost_bracket':'EXACT_VALIDATION_COST',
        }
    ordered=sorted(costs,key=lambda r:r['cost_ms'])
    candidates_bracket=[]
    for lo,hi in zip(ordered,ordered[1:]):
        if lo['cost_ms']<=target_ms<=hi['cost_ms'] and hi['cost_ms']>lo['cost_ms']:
            candidates_bracket.append((hi['cost_ms']-lo['cost_ms'],lo,hi))
    if candidates_bracket:
        _,lo,hi=min(candidates_bracket,key=lambda x:(x[0],x[1]['cost_ms']))
        fraction=(target_ms-lo['cost_ms'])/(hi['cost_ms']-lo['cost_ms'])
        assert 0<fraction<1
        return {
            'lower_cpu_lambda':lo['lambda'],
            'higher_cpu_lambda':hi['lambda'],
            'lower_cpu_cost_ms':lo['cost_ms'],
            'higher_cpu_cost_ms':hi['cost_ms'],
            'higher_cpu_probability':fraction,
            'expected_mean_ms':(1-fraction)*lo['cost_ms']+fraction*hi['cost_ms'],
            'target_dynamic_validation_mean_ms':target_ms,
            'validation_cost_bracket':'VALIDATION_TIME_BRACKET',
        }
    # Conservatively report no feasible exact time-match; never fabricate
    # a bracket by extrapolating to a new hyperparameter.
    nearest=min(costs,key=lambda r:(abs(r['cost_ms']-target_ms),r['lambda']))
    return {
        'lower_cpu_lambda':nearest['lambda'],
        'higher_cpu_lambda':nearest['lambda'],
        'lower_cpu_cost_ms':nearest['cost_ms'],
        'higher_cpu_cost_ms':nearest['cost_ms'],
        'higher_cpu_probability':0.,
        'expected_mean_ms':nearest['cost_ms'],
        'target_dynamic_validation_mean_ms':target_ms,
        'validation_cost_bracket':'NO_BRACKET_NEAREST_ONLY',
    }


def deterministic_hash_uniform(seed,identity):
    if type(seed) is not int or seed<0 or not isinstance(identity,str) or not identity:
        raise ValueError('invalid fixed assignment seed or source identity')
    src=f'quartz_cog02_costmatch11_v1|{seed}|{identity}'.encode('utf-8')
    h=hashlib.sha256(src).digest()
    return int.from_bytes(h[:8],'big')/2**64


def apply_myopic_lambda(model,lam):
    if lam not in ALLOWED_LAMBDAS or model.get('schema')!='quartz_cog02_dynamic_two_step_v1':
        raise ValueError('locked candidate lambda/schema mismatch')
    rows=model.get('two_step_gain_coefficients')
    if not isinstance(rows,list) or len(rows)!=2 or any(
        not isinstance(r,list) or len(r)!=9 or any(x!=0 for x in r) for r in rows
    ):
        raise ValueError('baseline must be a genuine one-step-only predictor')
    out=copy.deepcopy(model)
    out['lambda_wdl_per_ms']=float(lam)
    return out


def balanced_group_assignments(cases,mixture_modes,seed):
    if not cases:
        raise ValueError('frozen full test families required')
    grouped=defaultdict(lambda:defaultdict(list))
    ids=set()
    for r in cases:
        rid=r['id']
        if not isinstance(rid,str) or rid in ids:
            raise ValueError('duplicate/invalid root id')
        ids.add(rid)
        grouped[r['group']][r['family']].append(r)
    if len(grouped)!=2 or any(len(fam)!=12 for fam in grouped.values()):
        raise ValueError('C11 requires two 12-family cohorts')
    if any(len(rows)!=16 for fam in grouped.values() for rows in fam.values()):
        raise ValueError('each distinct opening family needs exactly 16 descendants')
    assignments={}
    for mode,m in mixture_modes.items():
        p=float(m['higher_cpu_probability'])
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('invalid validation-frozen cost mixture probability')
        assignments[mode]={}
        for group,families in grouped.items():
            base=math.floor(16*p)
            needed=round(len(families)*16*p)-len(families)*base
            if not 0<=needed<=len(families):
                raise ValueError('balanced family quotas unavailable')
            family_rank=sorted(families,key=lambda f:deterministic_hash_uniform(
                seed,f'family|{mode}|{group}|{f}'))
            extras=set(family_rank[:needed])
            for family,roots in families.items():
                expensive=base+int(family in extras)
                ranking=sorted(roots,key=lambda row:deterministic_hash_uniform(
                    seed,f'root|{mode}|{row["id"]}'))
                chosen={root['id'] for root in ranking[:expensive]}
                for row in roots:
                    assignments[mode][row['id']]='higher' if row['id'] in chosen else 'lower'
    assert all(len(q)==len(cases) for q in assignments.values())
    return assignments


def freeze_plan():
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    prior=d.parent/'dynamic_voc_10'
    modelpath=prior/'evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json'
    manifest=prior/'evidence/frozen_release_policies_20261121/FIT_MANIFEST.json'
    input_training=prior/'evidence/release_train_val_2496_20261121/JOINED_TRAIN_VAL_RELEASE.json'
    prior_model=json.loads(modelpath.read_text())
    meta=json.loads(manifest.read_text())
    if sha(modelpath)!=meta['output_policy_sha256']:
        raise RuntimeError('source frozen policy manifest mismatch')
    if sha(input_training)!=meta['input_2496_sha256']:
        raise RuntimeError('source 2496 G07 train/val bytes changed')
    if prior_model.get('used_old_or_new_test_labels_for_fit') is not False:
        raise RuntimeError('source policy itself was not heldout-sealed')
    modes={}
    for mode in ('uniform','new_mlp','conv'):
        block=prior_model['modes'][mode]
        dynamic_target=float(block['dynamic_val_mean_search_ms'])
        candidates=[{'lambda':float(item['lambda']),
                     'cost_ms':float(item['val']['mean_search_ms'])}
                    for item in block['validation']['myopic']['lambda_grid_results']]
        chosen=select_validation_cost_bracket(
            candidates,dynamic_target,preferred_lambda=float(block['validation']['myopic']['selected_lambda']))
        for lam in (chosen['lower_cpu_lambda'],chosen['higher_cpu_lambda']):
            apply_myopic_lambda(block['frozen_policies']['myopic'],lam)
        modes[mode]={
            **chosen,
            'validation_myopic_candidates_cost_only':candidates,
            'source_old_frozen_myopic_lambda':block['frozen_policies']['myopic']['lambda_wdl_per_ms'],
            'source_old_frozen_dynamic_lambda':block['frozen_policies']['dynamic']['lambda_wdl_per_ms'],
            'experiment_scope':'G07 old validation search time only, not WDL labels or heldout',
        }
    # Exactly this source is frozen *before* new C11 cohort generation.
    return {
        'schema':'quartz_cog02_dynamic_option_costmatch11_validation_only_random_mixture_v1',
        'source_parent_git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'source_frozen_C10_policy_sha256':sha(modelpath),
        'source_original_G07_trainval_2496_sha256':sha(input_training),
        'selection_only_validation_release_CPU_times':True,
        'no_C09_C10_test_outcome_or_future_C11_data_used':True,
        'randomization':'predeclared SHA256 family+position ranking without oracle, balanced block randomization within each new 12-family test cohort',
        'quota_rule':'per cohort: 12x16 roots, round(192 p) higher-cost assignments; family-level quota floor(16p) + up to one',
        'test_cohort_groups':['new_A','new_B'],
        'assignment_seeds':list(ASSIGNMENT_SEEDS),
        'three_assignments_all_reported_not_cherry_picked':True,
        'mode_mixtures':modes,
    }


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if not args.freeze:
        parser.error('explicit --freeze required')
    d=Path(__file__).resolve().parent
    dest=d/'MIXTURE_PLAN_FROZEN.json'
    if dest.exists():
        raise FileExistsError('original calibration already frozen; refuse overwrite')
    result=freeze_plan()
    dest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('C11_VALIDATION_ONLY_MIXTURE_LOCK',json.dumps({
        'modes':{name:{
            'low_cpu_lambda':v['lower_cpu_lambda'],
            'high_cpu_lambda':v['higher_cpu_lambda'],
            'p_high':v['higher_cpu_probability'],
            'val_expected_ms':v['expected_mean_ms'],
            'dyn_val_target_ms':v['target_dynamic_validation_mean_ms']}
          for name,v in result['mode_mixtures'].items()},
        'source_commit':result['source_parent_git_head'],'sha256':sha(dest)
    },sort_keys=True))
