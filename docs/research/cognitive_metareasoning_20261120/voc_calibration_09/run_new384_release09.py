#!/usr/bin/env python3
"""Two disjoint 12-family validation-cohort generalization experiments in RELEASE.

Do not refit the existing COG02 signed-gain predictor or its cost thresholds.
Live test includes actual separate fixed16, observed-clock, online adaptive and
full64 arms. No exact external oracle data are read until ALL Rust searches end.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time


MODES=('uniform','new_mlp','conv')
ARMS=('full64','adaptive','live_fixed16','validation_clock')
BUDGETS=(8,16,32,64)
GAIN_EDGES=(-.2,-.05,0.,.05,.2)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def forecast_bin(value):
    if not math.isfinite(value):
        raise ValueError('signed-gain forecast must be finite')
    for i,limit in enumerate(GAIN_EDGES):
        if value<=limit:return i
    return len(GAIN_EDGES)


def paired_cluster_summary(left,right,seed=20261125):
    if len(left)!=len(right) or not left:
        raise ValueError('pair root cardinality mismatch')
    families=defaultdict(list)
    for l,r in zip(left,right):
        if l['family']!=r['family']:
            raise ValueError('paired policy family mismatch')
        families[l['family']].append(l['regret']-r['regret'])
    m=len(families)
    means=[sum(v)/len(v) for _,v in sorted(families.items())]
    n=len(left)
    rng=random.Random(seed)
    boot=sorted(sum(rng.choice(means) for _ in means)/m for _ in range(3000))
    return {
        'n_families':m,'n_roots':n,
        'mean_left_minus_right_regret':sum(means)/m,
        'conditional_95_interval':[boot[74],boot[2924]],
        'family_means':[float(v) for v in means],
        'root_IID_confidence_claimed':False,
        'fitted_policy_retrained_within_bootstrap':False,
    }


def imports():
    old=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    from evaluate_live08 import (
        model_request,run_batch,score_completed_live,compare_rust_current_policy
    )
    from train_gain08 import current_root_features,model_identity
    return model_request,run_batch,score_completed_live,compare_rust_current_policy,current_root_features,model_identity


def summarise_row_arm(rows,mode,arm):
    x=[row['policies'][mode][arm] for row in rows]
    n=len(x)
    if not n:raise ValueError('no independent test roots')
    return {
        'n_roots':n,'n_families':len({z['family'] for z in rows}),
        'mean_WDL':sum(t['WDL'] for t in x)/n,
        'mean_regret':sum(t['regret'] for t in x)/n,
        'optimal_WDL_count':sum(t['regret']==0 for t in x),
        'mean_search_ms':sum(t['search_ms'] for t in x)/n,
        'mean_request_ms_without_fresh':sum(t['request_ms_without_fresh'] for t in x)/n,
        'mean_estimated_game_request_ms':sum(t['estimated_game_request_ms'] for t in x)/n,
        'mean_evaluator_calls':sum(t['evaluator_calls'] for t in x)/n,
        'mean_root_visits':sum(t['visits'] for t in x)/n,
        'stops':dict(Counter(str(t['visits']) for t in x)),
        'WDL_counts':dict(Counter(str(t['WDL']) for t in x)),
        'regret_counts':dict(Counter(str(t['regret']) for t in x)),
    }


def mode_calibration(cohort,mode,raw_lookup,oracle,predictor,current_root_features):
    stages=[]
    for stage in range(3):
        obs=[]
        for root in cohort:
            full=raw_lookup[(root['id'],mode,'full64')]
            exact=oracle[root['id']]['exact_action_scores']
            a,b=full['steps'][stage:stage+2]
            x=current_root_features(full,stage)
            forecast=sum(f*g for f,g in zip(predictor['coefficients'][stage],x))
            scores=lambda q:(q>0)-(q<0)
            realized=scores(exact[b['best_action']])-scores(exact[a['best_action']])
            measured_ms=b['step_wall_us']/1000.0
            forecast_ms=predictor['forecast_cost_ms'][stage]
            flag=forecast>predictor['lambda_wdl_per_ms']*forecast_ms
            obs.append({'id':root['id'],'family':root['family'],
                        'forecast':forecast,'realized_signed_gain':realized,
                        'forecast_cost_ms':forecast_ms,'measured_release_cost_ms':measured_ms,
                        'continue_if_observed':flag,'prediction_bin':forecast_bin(forecast)})
        n=len(obs)
        bin_data={}
        for bucket in range(6):
            v=[o for o in obs if o['prediction_bin']==bucket]
            bin_data[str(bucket)]={
                'n':len(v),'mean_forecast':(sum(z['forecast'] for z in v)/len(v) if v else None),
                'mean_realized':(sum(z['realized_signed_gain'] for z in v)/len(v) if v else None),
                'observed_negative':sum(z['realized_signed_gain']<0 for z in v),
                'observed_positive':sum(z['realized_signed_gain']>0 for z in v),
            }
        stages.append({
            'from_visits':BUDGETS[stage], 'to_visits':BUDGETS[stage+1],
            'n_roots':n,'mean_predicted_signed_gain':sum(x['forecast'] for x in obs)/n,
            'mean_realized_signed_gain':sum(x['realized_signed_gain'] for x in obs)/n,
            'MSE':sum((x['forecast']-x['realized_signed_gain'])**2 for x in obs)/n,
            'MAE':sum(abs(x['forecast']-x['realized_signed_gain']) for x in obs)/n,
            'realized_signed_counts':dict(Counter(str(x['realized_signed_gain']) for x in obs)),
            'mean_original_debug_forecast_cost_ms':sum(x['forecast_cost_ms'] for x in obs)/n,
            'mean_actual_release_next_batch_ms':sum(x['measured_release_cost_ms'] for x in obs)/n,
            'debug_forecast_div_release_actual_cost_ratio':(
                sum(x['forecast_cost_ms'] for x in obs)/
                max(1e-12,sum(x['measured_release_cost_ms'] for x in obs))),
            'counterfactual_false_stop_with_positive_next_gain':sum(
                not x['continue_if_observed'] and x['realized_signed_gain']>0 for x in obs),
            'counterfactual_continue_with_nonpositive_next_gain':sum(
                x['continue_if_observed'] and x['realized_signed_gain']<=0 for x in obs),
            'gain_prediction_bins':bin_data,
        })
    return stages


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--binary',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('run new-family test only from clean committed source')
    model_request,run_batch,score_completed_live,compare_rust_current_policy,current_root_features,model_identity=imports()
    before=d.parent/'cost_matched_voc_08'
    policies_path=before/'evidence/frozen_surrogate_20261121/frozen_surrogates.json'
    caps_path=d/'evidence/release_validation_1152_20261121/VALIDATION_RELEASE_CAPS.json'
    locked=json.loads((d/'INPUT_LOCK.json').read_text())
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    policies=json.loads(policies_path.read_text())
    caps=json.loads(caps_path.read_text())
    if caps['prior_policies_sha256']!=sha(policies_path):
        raise RuntimeError('release validation cap source is not frozen current policy')
    if policies['test_set_used_for_selection'] or not caps['no_new_test_cohort_access']:
        raise RuntimeError('test leakage in original fitted policy/clock cap')
    roots=cohort['rows']
    assert len(roots)==384
    assert Counter(x['cohort'] for x in roots)=={'replica_A':192,'replica_B':192}
    assert len({x['family'] for x in roots})==24
    models=model_identity(repo)
    modelhash={name:sha(p) for name,p in models.items()}
    for item in locked['input_source_sha256']:
        p=Path(item['path'])
        if p.name in ('mcts_server.rs','cog02.rs','connect4.rs'):continue
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise RuntimeError('immutable original model/solver/cohort input changed: '+str(p))
    queries=[];keys=[]
    for root in roots:
        for mode in MODES:
            policy=policies['mode_policies'][mode]
            cap=caps['modes'][mode]['cap_ms']
            for arm in ARMS:
                if arm=='full64':
                    q=model_request(root,mode,'full',models,modelhash,None,None)
                elif arm=='adaptive':
                    q=model_request(root,mode,'adaptive',models,modelhash,policy,None)
                elif arm=='validation_clock':
                    q=model_request(root,mode,'fixed_clock',models,modelhash,None,cap)
                elif arm=='live_fixed16':
                    q=model_request(root,mode,'full',models,modelhash,None,None)
                    q['force_stop_visits']=16
                else:raise RuntimeError('unknown arm')
                assert 'exact_action_scores' not in q and 'oracle' not in q
                keys.append((root['id'],mode,arm))
                queries.append(q)
    assert len(queries)==4608
    exe=args.binary.resolve()
    if not exe.is_file():raise FileNotFoundError(exe)
    if sha(exe)!=caps['release_rust_binary_sha256']:
        raise RuntimeError('compiled rust binary differs from fixed validation executable')
    started=time.perf_counter()
    raw=run_batch(exe,queries)
    run_elapsed=time.perf_counter()-started
    assert len(raw)==4608
    # Exact external oracle labels loaded ONLY AFTER executing 4608 searches.
    label_path=d/'evidence/new384_exact_teacher_20261121/labels.json'
    labels=json.loads(label_path.read_text())
    if labels['statuses']!={'OK':384}:raise RuntimeError('new-family exact oracle incomplete')
    oracle={row['id']:row for row in labels['rows']}
    assert len(oracle)==384
    originals={row['id']:row for row in roots}
    byroot={r['id']:{'id':r['id'],'family':r['family'],'cohort':r['cohort'],
                 'history':r['history'],'side_to_move':r['side_to_move'],'policies':{}}
            for r in roots}
    rawlookup={}
    for (ident,mode,arm),reply in zip(keys,raw):
        orig=originals[ident]
        assert reply['history']==orig['history']
        assert reply['board_bottom_up']==orig['board']
        assert reply['side_to_move']==orig['side_to_move']
        assert reply['legal_columns_0based']==orig['legal_columns']
        assert reply['eval_mode']=={
            'uniform':'uniform_no_neural_network',
            'new_mlp':'trained_tiny_mlp_cpu',
            'conv':'trained_conv3x3_cpu'
        }[mode]
        if mode!='uniform' and not reply['cache_model']:
            raise RuntimeError('trained model should be in-process cached')
        scored=score_completed_live(reply,oracle[ident]['exact_action_scores'])
        byroot[ident]['policies'].setdefault(mode,{})[arm]=scored
        rawlookup[(ident,mode,arm)]=reply
        if arm=='adaptive':
            compare_rust_current_policy(reply,policies['mode_policies'][mode])
        elif arm=='live_fixed16':
            assert reply['force_stop_visits']==16
            assert reply['stopped_at_visits']==16 and len(reply['steps'])==2
            assert reply['fresh16']['resident_match']
        elif arm=='validation_clock':
            assert reply['time_budget_ms']==caps['modes'][mode]['cap_ms']
    paired=[byroot[row['id']] for row in roots]
    for row in paired:
        for mode in MODES:
            assert set(row['policies'][mode])==set(ARMS)
            base=rawlookup[(row['id'],mode,'full64')]
            assert base['stopped_at_visits']==64 and len(base['steps'])==4
            for arm in ('adaptive','validation_clock','live_fixed16'):
                actual=rawlookup[(row['id'],mode,arm)]
                for old,new in zip(base['steps'],actual['steps']):
                    if (old['best_action']!=new['best_action'] or
                        old['visit_policy']!=new['visit_policy'] or
                        old['evaluator_calls']!=new['evaluator_calls']):
                        raise RuntimeError('resident prefix changed by stopping control')
                assert row['policies'][mode][arm]['root_hash']==row['policies'][mode]['full64']['root_hash']
    result={
        'schema':'quartz_cog02_voc_cal09_new_family_release_live_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'actual_release_requests':len(raw),'new_roots':384,'new_families':24,
        'release_wall_s_for_all_requests':run_elapsed,
        'pre_frozen_policy_sha256':sha(policies_path),
        'release_validation_caps_sha256':sha(caps_path),
        'new_cohort_labels_sha256':sha(label_path),
        'model_sha256':modelhash,
        'release_binary_sha256':sha(exe),
        'modes':{},
        'science_nonclaims':['not Bayesian preposterior VOC',
                             'not stochastic seed replication',
                             'not per-root exact time matching',
                             'not iid 384 roots'],
    }
    for name in ('replica_A','replica_B','pooled'):
        subset=[row for row in paired if name=='pooled' or row['cohort']==name]
        result['modes'][name]={}
        for mode in MODES:
            arms={arm:summarise_row_arm(subset,mode,arm) for arm in ARMS}
            adaptive=arms['adaptive']
            clock=arms['validation_clock']
            fixed=arms['live_fixed16']
            mismatch=abs(adaptive['mean_search_ms']-clock['mean_search_ms'])/max(1e-12,adaptive['mean_search_ms'])
            pair_adapt=[{'family':x['family'],'regret':x['policies'][mode]['adaptive']['regret']} for x in subset]
            pair_clock=[{'family':x['family'],'regret':x['policies'][mode]['validation_clock']['regret']} for x in subset]
            pair_fixed=[{'family':x['family'],'regret':x['policies'][mode]['live_fixed16']['regret']} for x in subset]
            other={
                'mean_search_ms_cost_mismatch_fraction':mismatch,
                'clock_cost_match_status':'MATCHED_10_PERCENT' if mismatch<=0.10 else 'NOT_MATCHED_10_PERCENT',
                'adaptive_minus_clock_regret':adaptive['mean_regret']-clock['mean_regret'],
                'adaptive_minus_actual_live16_regret':adaptive['mean_regret']-fixed['mean_regret'],
                'adaptive_vs_clock_clustered':paired_cluster_summary(pair_adapt,pair_clock),
                'adaptive_vs_actual_live16_clustered':paired_cluster_summary(pair_adapt,pair_fixed),
                'registered_frozen_lambda':policies['mode_policies'][mode]['lambda_wdl_per_ms'],
                'release_val_only_fixed_cap_ms':caps['modes'][mode]['cap_ms'],
            }
            result['modes'][name][mode]={'arms':arms,'comparison':other,
                 'calibration':mode_calibration(subset,mode,rawlookup,oracle,
                         policies['mode_policies'][mode],current_root_features)}
    for mode,p in models.items():
        if sha(p)!=modelhash[mode]:
            raise RuntimeError('trained model mutated during sealed new-family evaluation')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name,body in [
        ('RAW_4608_RELEASE_MCTS.json',raw),
        ('PAIRED_384_NEW_FAMILY.json',paired),
        ('SUMMARY_CALIBRATION.json',result)]:
        (out/name).write_text(json.dumps(body,sort_keys=True,ensure_ascii=False)+'\n')
    receipt={
        'schema':'quartz_cog02_voc_cal09_release_test_manifest_v1',
        'source_commit':result['source_commit'],
        'source_tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
        'binary_sha256':sha(exe),
        'compiled_rust_source_commit':'1c2b00c1bad2585380fee97bde65e3862dc1f25d',
        'old_policy_sha256':sha(policies_path),
        'release_validation_caps_sha256':sha(caps_path),
        'new384_cohort_sha256':sha(d/'COHORT_384_FROZEN.json'),
        'new384_teacher_sha256':sha(label_path),
        'n_real_release_requests':4608,'n_cohorts':2,'n_clusters':24,
        'output_sha256':{p.name:sha(p) for p in out.iterdir()},
        'oracle_join_only_after_all_search':True,
        'no_runtime_retraining_or_hyperparam_tuning':True,
        'model_weights_resident_within_server':True,
    }
    (out/'EXECUTION_MANIFEST.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('CAL09_NEWFAMILY_RELEASE_TEST_DONE',json.dumps({
        'n':len(raw),'wall_s':run_elapsed,
        'summary':{cohort:{
            mode:{
                'adaptive_regret':result['modes'][cohort][mode]['arms']['adaptive']['mean_regret'],
                'live_fixed16_regret':result['modes'][cohort][mode]['arms']['live_fixed16']['mean_regret'],
                'clock_regret':result['modes'][cohort][mode]['arms']['validation_clock']['mean_regret'],
                'adaptive_search_ms':result['modes'][cohort][mode]['arms']['adaptive']['mean_search_ms'],
                'clock_search_ms':result['modes'][cohort][mode]['arms']['validation_clock']['mean_search_ms'],
                'time_match':result['modes'][cohort][mode]['comparison']['clock_cost_match_status'],
            } for mode in MODES}
           for cohort in ('replica_A','replica_B','pooled')}
    },sort_keys=True))

if __name__=='__main__':main()
