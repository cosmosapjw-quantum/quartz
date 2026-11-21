#!/usr/bin/env python3
"""Prospective C11 2-step vs blocked-random myopic policy at real RELEASE CPU cost.

Models, validation CPU mixture p, all 3 assignment streams, and 384 new root
IDs were committed BEFORE test exact oracle scores were computed. Only
CPU allocations (never solver scores/future search) are sent to Rust.
All 10368 actual QUARTZ MCTS root computations execute in one warmed server.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

from mix_plan11 import apply_myopic_lambda,balanced_group_assignments

MODES=('uniform','new_mlp','conv')
ARMS=('full64','dynamic','myopic_low','myopic_high',
      'mixture_1','mixture_2','mixture_3','validation_clock','live_fixed16')
BUDGETS=(8,16,32,64)
BINARY_SHA='4db989b0ede0c56077de77a820a99cd777adcf87c98d7faedc62611491eb8a03'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def imports():
    prev=Path(__file__).resolve().parent.parent/'dynamic_voc_10'
    if str(prev) not in sys.path:sys.path.insert(0,str(prev))
    from run_live10 import paired_family_ci,mean_metrics
    from dynamic_model10 import predict_option
    old=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    from evaluate_live08 import model_request,run_batch,score_completed_live
    from train_gain08 import model_identity,current_root_features
    return paired_family_ci,mean_metrics,predict_option,model_request,run_batch,score_completed_live,model_identity,current_root_features


def mean_time_match(dynamic_ms,baseline_ms):
    if not math.isfinite(dynamic_ms) or dynamic_ms<=0 or not math.isfinite(baseline_ms) or baseline_ms<=0:
        raise ValueError('CPU timing must be positive finite physical search time')
    return 'MATCHED_10_PERCENT' if abs(dynamic_ms-baseline_ms)/dynamic_ms<=0.10 else 'NOT_MATCHED_10_PERCENT'


def make_locked_mixture_policy(root,one_step_source,mixture_modes,selected_assignment,mode):
    if mode not in mixture_modes or mode not in selected_assignment:
        raise ValueError('not an admitted candidate source/evaluator')
    choice=selected_assignment[mode].get(root['id'])
    if choice not in ('lower','higher'):
        raise ValueError('unfrozen/invalid root assignment')
    field='lower_cpu_lambda' if choice=='lower' else 'higher_cpu_lambda'
    config=mixture_modes[mode]
    return apply_myopic_lambda(one_step_source,float(config[field]))


def mixture_wdl_expectation(low,high,probability):
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('invalid validation-frozen probability')
    result={'E_regret':(1-probability)*low['regret']+probability*high['regret'],
            'E_search_ms':(1-probability)*low['search_ms']+probability*high['search_ms']}
    if 'WDL' in low and 'WDL' in high:
        result['E_WDL']=(1-probability)*low['WDL']+probability*high['WDL']
    return result


def base_model_request(root,mode,arm,models,shas,old,plan,assigned,seeds):
    paired_family_ci,mean_metrics,predict_option,model_request,run_batch,score_completed_live,model_identity,current_root_features=imports()
    query=model_request(root,mode,'full',models,shas,None,None)
    mix=plan['mode_mixtures'][mode]
    block=old['modes'][mode]
    myopic=block['frozen_policies']['myopic']
    if arm=='dynamic':
        query['dynamic_policy']=block['frozen_policies']['dynamic']
    elif arm=='myopic_low':
        query['dynamic_policy']=apply_myopic_lambda(myopic,float(mix['lower_cpu_lambda']))
    elif arm=='myopic_high':
        query['dynamic_policy']=apply_myopic_lambda(myopic,float(mix['higher_cpu_lambda']))
    elif arm.startswith('mixture_'):
        i=int(arm.split('_')[1])-1
        if not 0<=i<3:raise ValueError('unregistered assignment arm')
        key=str(seeds[i])
        query['dynamic_policy']=make_locked_mixture_policy(root,myopic,plan['mode_mixtures'],assigned[key],mode)
    elif arm=='validation_clock':
        query['time_budget_ms']=block['clock_validation_time_match']['cap_ms']
    elif arm=='live_fixed16':
        query['force_stop_visits']=16
    elif arm!='full64':
        raise ValueError('unknown declared arm')
    assert all(x not in query for x in ('oracle','exact_action_scores','future_WDL'))
    return query


def metrics(subset,mode,arm):
    arr=[r['policies'][mode][arm] for r in subset]
    n=len(arr)
    if not arr:raise ValueError('empty new opening-family heldout subset')
    return {
        'n':n,'family_clusters':len({r['family'] for r in subset}),
        'mean_WDL':sum(x['WDL'] for x in arr)/n,
        'mean_regret':sum(x['regret'] for x in arr)/n,
        'mean_actual_search_ms':sum(x['search_ms'] for x in arr)/n,
        'mean_game_request_ms_plus_setup':sum(x['estimated_game_request_ms'] for x in arr)/n,
        'mean_root_setup_ms':sum(x['setup_ms'] for x in arr)/n,
        'mean_request_ms_without_fresh':sum(x['request_ms_without_fresh'] for x in arr)/n,
        'mean_evaluator_calls':sum(x['evaluator_calls'] for x in arr)/n,
        'mean_visits':sum(x['visits'] for x in arr)/n,
        'optimal_WDL_action_count':sum(x['regret']==0 for x in arr),
        'stop_visits':dict(Counter(str(x['visits']) for x in arr)),
        'regret_counts':dict(Counter(str(x['regret']) for x in arr)),
        'signed_WDL_counts':dict(Counter(str(x['WDL']) for x in arr)),
        'two_step_commits':sum(bool(x.get('dynamic_skipped_checkpoints')) for x in arr),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('run new heldout experiment only from clean committed source')
    paired_family_ci,_,predict_option,model_request,run_batch,score_completed_live,model_identity,current_root_features=imports()
    mixture_path=d/'MIXTURE_PLAN_FROZEN.json'
    assigned_path=d/'ASSIGNMENTS_FROZEN.json'
    cohort_path=d/'COHORT_384_FROZEN.json'
    plan=json.loads(mixture_path.read_text())
    assignments=json.loads(assigned_path.read_text())
    cohort=json.loads(cohort_path.read_text())
    locks=json.loads((d/'INPUT_LOCK.json').read_text())
    if (sha(mixture_path)!=locks['precommitted_mode_mix_sha256'] or
        sha(mixture_path)!=assignments['model_mixture_input_SHA256'] or
        sha(cohort_path)!=assignments['new_cohort_sha256']):
        raise RuntimeError('pretest mixture/assignment or new family provenance drift')
    rows=cohort['rows']
    if len(rows)!=384 or len(cohort['families'])!=24 or (
        Counter(r['group'] for r in rows)!={'new_A':192,'new_B':192}):
        raise RuntimeError('wrong sealed fresh cohort')
    seeds=plan['assignment_seeds']
    if seeds!=[202611210101,202611210102,202611210103]:
        raise RuntimeError('assignment seeds unexpectedly changed')
    assert balanced_group_assignments(rows,plan['mode_mixtures'],seeds[0])==assignments['seed_assignments'][str(seeds[0])]
    # Experimental coefficient identity frozen in C10, independent of new test labels.
    old_dir=d.parent/'dynamic_voc_10'
    modelpath=old_dir/'evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json'
    model_fit=json.loads((old_dir/'evidence/frozen_release_policies_20261121/FIT_MANIFEST.json').read_text())
    if sha(modelpath)!=model_fit['output_policy_sha256'] or (
        sha(modelpath)!=plan['source_frozen_C10_policy_sha256']):
        raise RuntimeError('frozen C10 root policy changed')
    old=json.loads(modelpath.read_text())
    if old['used_old_or_new_test_labels_for_fit'] is not False:
        raise RuntimeError('original root policy invalid/test leakage')
    models=model_identity(repo)
    modelhash={name:sha(p) for name,p in models.items()}
    for mode in MODES:
        if mode not in plan['mode_mixtures']:
            raise RuntimeError('missing original candidate evaluator')
    wires=[];keys=[]
    for row in rows:
        for mode in MODES:
            for arm in ARMS:
                wire=base_model_request(row,mode,arm,models,modelhash,old,plan,
                                        assignments['seed_assignments'],seeds)
                wires.append(wire)
                keys.append((row['id'],mode,arm))
    assert len(wires)==384*3*len(ARMS)==10368
    assert all('oracle' not in q and 'exact_action_scores' not in q for q in wires)
    binpath=args.binary.resolve()
    if not binpath.is_file() or sha(binpath)!=BINARY_SHA:
        raise RuntimeError('wrong compiled Rust release CPU engine or bytes')
    start=time.perf_counter()
    raw=run_batch(binpath,wires)
    duration=time.perf_counter()-start
    assert len(raw)==len(wires)==10368
    # CRITICAL: external exact WDL file opened ONLY AFTER all Rust searches finish.
    exactfile=d/'evidence/exact_new384_20261121/EXACT_NEW384_LABELS.json'
    exact=json.loads(exactfile.read_text())
    if not exact['status']['all_ok'] or exact['status']['counts']!={'OK':384}:
        raise RuntimeError('heldout exact solver data missing/partial')
    teacher={x['id']:x for x in exact['rows']}
    assert len(teacher)==384
    orig={r['id']:r for r in rows}
    paired={r['id']:{
        'id':r['id'],'group':r['group'],'family':r['family'],
        'history':r['history'],'side_to_move':r['current_player'],
        'oracle_optimal_WDL':teacher[r['id']]['root_wdl'],
        'policies':{},
    } for r in rows}
    lookup={}
    actual_assignments={str(seed):{m:Counter() for m in MODES} for seed in seeds}
    for (id,mode,arm),result in zip(keys,raw):
        original=orig[id]
        if (result['history']!=original['history'] or result['seed']!=17 or
            result['board_bottom_up']!=original['board'] or
            result['legal_columns_0based']!=original['legal_columns'] or
            result['side_to_move']!=original['current_player']):
            raise RuntimeError('actual Rust game state, side or source input corrupted')
        expected={'uniform':'uniform_no_neural_network',
                  'new_mlp':'trained_tiny_mlp_cpu',
                  'conv':'trained_conv3x3_cpu'}[mode]
        if result['eval_mode']!=expected:
            raise RuntimeError('wrong neural evaluator in a live experiment')
        if mode!='uniform' and not result['cache_model']:
            raise RuntimeError('nonresident neural model in experiment')
        if arm in ('dynamic','myopic_low','myopic_high') or arm.startswith('mixture_'):
            if arm=='dynamic':
                policy=old['modes'][mode]['frozen_policies']['dynamic']
                variant='dynamic'
            elif arm=='myopic_low':
                policy=apply_myopic_lambda(old['modes'][mode]['frozen_policies']['myopic'],
                                          float(plan['mode_mixtures'][mode]['lower_cpu_lambda']))
                variant='myopic'
            elif arm=='myopic_high':
                policy=apply_myopic_lambda(old['modes'][mode]['frozen_policies']['myopic'],
                                          float(plan['mode_mixtures'][mode]['higher_cpu_lambda']))
                variant='myopic'
            else:
                idx=int(arm.split('_')[1])-1
                seed=str(seeds[idx])
                plan_selected=assignments['seed_assignments'][seed]
                actual_assignments[seed][mode][plan_selected[mode][id]]+=1
                policy=make_locked_mixture_policy(original,
                         old['modes'][mode]['frozen_policies']['myopic'],
                         plan['mode_mixtures'],plan_selected,mode)
                variant='myopic'
            for dec in result['dynamic_decisions']:
                stage={8:0,16:1,32:2}[dec['at_visits']]
                features=current_root_features(result,stage)
                check=predict_option(policy,stage,features,variant=variant)
                if dec['action']!=check['action'] or max(abs(a-b) for a,b in zip(
                    dec['root_features'],features))>3e-9:
                    raise RuntimeError('future-leak/online Rust feature and stop mismatch')
                for field in ('predicted_single_gain','predicted_single_cost_ms','net_single'):
                    if abs(dec[field]-check[field])>3e-9:
                        raise RuntimeError('release Rust vs current-root signed gain/cost mismatch')
            if variant=='myopic' and result['dynamic_skipped_checkpoints']:
                raise RuntimeError('myopic baseline secretly used a two-step commitment')
        scored=score_completed_live(result,teacher[id]['exact_action_scores'])
        scored['dynamic_skipped_checkpoints']=result.get('dynamic_skipped_checkpoints',[])
        paired[id]['policies'].setdefault(mode,{})[arm]=scored
        lookup[(id,mode,arm)]=result
        if arm=='live_fixed16':
            if result.get('force_stop_visits')!=16 or result['stopped_at_visits']!=16:
                raise RuntimeError('independent fixed16 arm did not stop at16')
        if arm=='validation_clock' and (
            result.get('time_budget_ms')!=old['modes'][mode]['clock_validation_time_match']['cap_ms']):
            raise RuntimeError('fixed clock cap illegally changed using heldout')
    for seed in seeds:
        key=str(seed)
        for mode in MODES:
            high_count=actual_assignments[key][mode]['higher']
            low_count=actual_assignments[key][mode]['lower']
            if mode=='uniform' and (high_count!=228 or low_count!=156):
                raise RuntimeError('balanced precommitted full-cohort allocations not executed')
            if mode!='uniform' and (high_count!=0 or low_count!=384):
                raise RuntimeError('negative control allocation changed')
    paired_rows=[paired[r['id']] for r in rows]
    for row in paired_rows:
        for mode in MODES:
            base=lookup[(row['id'],mode,'full64')]
            if len(base['steps'])!=4 or base['stopped_at_visits']!=64:
                raise RuntimeError('actual MCTS full64 source missing')
            for arm in ARMS[1:]:
                current=lookup[(row['id'],mode,arm)]
                if current['root_hash']!=base['root_hash']:
                    raise RuntimeError('different game root hashes')
                for x,y in zip(base['steps'],current['steps']):
                    if (x['best_action']!=y['best_action'] or
                        x['evaluator_calls']!=y['evaluator_calls'] or
                        x['visit_policy']!=y['visit_policy']):
                        raise RuntimeError('real MCTS prefix changed with policy arm')
    result={
        'schema':'quartz_cog02_dynamic_costmatch11_prospective_heldout_release_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'compiled_rust_release_binary_sha256':sha(binpath),
        'new384_cohort_sha256':sha(cohort_path),
        'new384_exact_label_sha256':sha(exactfile),
        'frozen_original_C10_model_sha256':sha(modelpath),
        'frozen_validation_mix_sha256':sha(mixture_path),
        'frozen_preoracle_assignment_sha256':sha(assigned_path),
        'n_real_Rust_inprocess_release_search_requests':len(raw),
        'n_distinct_new_roots':len(rows),'n_distinct_families':24,
        'batch_elapsed_s':duration,
        'assignment_seeds':seeds,
        'actual_assignments_each_seed':{
            key:{mode:dict(v) for mode,v in val.items()}
            for key,val in actual_assignments.items()},
        'comparisons':{},
        'nonclaims':['three assignment streams are NOT new game positions',
                     'not independent MCTS seed noise',
                     'not Bayesian optimal nonmyopic cognition',
                     'root descendants correlated; use family-cluster bootstrap',
                     'wall CPU time not process CPU or hard deadline'],
    }
    for group in ('new_A','new_B','pooled'):
        subset=[r for r in paired_rows if group=='pooled' or r['group']==group]
        assert len(subset)==(384 if group=='pooled' else 192)
        result['comparisons'][group]={}
        for mode in MODES:
            all_arms={arm:metrics(subset,mode,arm) for arm in ARMS}
            dy=all_arms['dynamic']
            comparisons={}
            for arm in ARMS[2:]:
                other=all_arms[arm]
                pairL=[{'family':r['family'],'regret':r['policies'][mode]['dynamic']['regret']} for r in subset]
                pairR=[{'family':r['family'],'regret':r['policies'][mode][arm]['regret']} for r in subset]
                comparisons['dynamic_minus_'+arm]={
                    'regret_difference':dy['mean_regret']-other['mean_regret'],
                    'relative_actual_search_ms_mismatch':abs(dy['mean_actual_search_ms']-other['mean_actual_search_ms'])/dy['mean_actual_search_ms'],
                    'mean_CPU_time_match':mean_time_match(dy['mean_actual_search_ms'],other['mean_actual_search_ms']),
                    'family_cluster_conditional_bootstrap':paired_family_ci(pairL,pairR),
                }
            mixconfig=plan['mode_mixtures'][mode]
            p=float(mixconfig['higher_cpu_probability'])
            x=[mixture_wdl_expectation(r['policies'][mode]['myopic_low'],r['policies'][mode]['myopic_high'],p) for r in subset]
            Ereg=sum(z['E_regret'] for z in x)/len(x)
            Ec=sum(z['E_search_ms'] for z in x)/len(x)
            expected={
                'n':len(x),'prob_high':p,
                'ex_ante_randomization_expected_regret_conditional_on_frozen_sample':Ereg,
                'ex_ante_randomization_expected_search_ms_conditional_on_frozen_sample':Ec,
                'dynamic_minus_expected_randomized_regret':dy['mean_regret']-Ereg,
                'relative_search_cpu_mismatch_to_expected_randomization':abs(dy['mean_actual_search_ms']-Ec)/dy['mean_actual_search_ms'],
                'cost_match_expected_randomization':mean_time_match(dy['mean_actual_search_ms'],Ec),
                'not_a_per_root_actual_randomized_result':True,
            }
            informative=[r for r in subset if len({(teacher[r['id']]['exact_action_scores'][col]>0)-
                        (teacher[r['id']]['exact_action_scores'][col]<0)
                        for col in orig[r['id']]['legal_columns']})>1]
            group_result={
                'arms':all_arms,
                'paired':comparisons,
                'expected_randomization':expected,
                'decision_informative_root_count':len(informative),
                'exact_WDL_root_class_counts':dict(Counter(str(teacher[r['id']]['root_wdl']) for r in subset)),
                'informative_only_mean_regret_by_arm':{
                    arm:sum(r['policies'][mode][arm]['regret'] for r in informative)/len(informative)
                        if informative else None for arm in ARMS},
                'by_side':{
                    str(side):{
                        arm:{'n':sum(r['side_to_move']==side for r in subset),
                             'mean_regret':sum(r['policies'][mode][arm]['regret'] for r in subset if r['side_to_move']==side)/
                                 max(1,sum(r['side_to_move']==side for r in subset))}
                        for arm in ('dynamic','mixture_1','mixture_2','mixture_3','validation_clock')}
                    for side in (-1,1)},
            }
            result['comparisons'][group][mode]=group_result
    for name,p in models.items():
        if sha(p)!=modelhash[name]:
            raise RuntimeError('neural model weights mutated during test')
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    for name,data in [('RAW_10368_RELEASE_ROOTS.json',raw),
                       ('PAIRED_384_NEW_POSITIONS.json',paired_rows),
                       ('SUMMARY_C11_COST_MATCH.json',result)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n')
    receipt={
        'schema':'quartz_cog02_dynamic_costmatch11_live_run_manifest_v1',
        'source_commit':result['source_commit'],
        'source_tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
        'compiled_rust_source_commit':'4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2',
        'compiled_release_binary_sha256':sha(binpath),
        'new_cohort_sha256':sha(cohort_path),
        'validation_mix_sha256':sha(mixture_path),
        'assignment_sha256':sha(assigned_path),
        'exact_oracle_sha256':sha(exactfile),
        'frozen_model_sha256':sha(modelpath),
        'n_real_Rust_searches':len(raw),
        'external_exact_WDL_loaded_only_after_Rust_search':True,
        'assignment_seed_streams_all_included':True,
        'all_myopic_arms_Rust_one_step_online_not_retrospective_actions':True,
        'output_sha256':{p.name:sha(p) for p in output.iterdir()},
    }
    (output/'EXECUTION_MANIFEST.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('C11_PROSPECTIVE_RELEASE_FINISHED',json.dumps({
        'real_searches':len(raw),'elapsed_s':duration,
        'uniform':{group:{
            'dynamic_regret':result['comparisons'][group]['uniform']['arms']['dynamic']['mean_regret'],
            'mix_regret':[result['comparisons'][group]['uniform']['arms'][f'mixture_{i}']['mean_regret'] for i in (1,2,3)],
            'cost_match':[result['comparisons'][group]['uniform']['paired'][f'dynamic_minus_mixture_{i}']['mean_CPU_time_match'] for i in (1,2,3)],
            'expectation_regret':result['comparisons'][group]['uniform']['expected_randomization']['ex_ante_randomization_expected_regret_conditional_on_frozen_sample'],
            'expected_cost_match':result['comparisons'][group]['uniform']['expected_randomization']['cost_match_expected_randomization'],
        } for group in ('new_A','new_B','pooled')}
    },sort_keys=True))


if __name__=='__main__':
    main()
