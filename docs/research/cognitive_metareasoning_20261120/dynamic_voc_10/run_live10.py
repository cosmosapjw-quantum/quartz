#!/usr/bin/env python3
"""Actual heldout 384-root CPU release comparison of seven MCTS scheduling arms.

New C10 exact oracle scores are not read until all searches complete.
Original G07 train and validation alone determined gains, costs, lambda,
myopic time-match and fixed clock caps. This is an empirical signed-gain
heuristic, NOT exact Bayes knowledge gradient or a metalevel optimality claim.
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

from dynamic_model10 import BUDGETS,predict_option,simulate_bounded_policy

MODES=('uniform','new_mlp','conv')
ARMS=('full64','dynamic','myopic','myopic_time_match',
      'legacy_debug_myopic','live_fixed16','validation_clock')
GAIN_EDGES=(-.2,-.05,0.,.05,.2)


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def forecast_bin(x):
    if not math.isfinite(x):raise ValueError('nonfinite signed-gain forecast')
    for i,edge in enumerate(GAIN_EDGES):
        if x<=edge:return i
    return 5


def cost_match_status(adaptive_ms,comparator_ms):
    if not math.isfinite(adaptive_ms) or adaptive_ms<=0 or not math.isfinite(comparator_ms) or comparator_ms<=0:
        raise ValueError('positive physical release CPU search cost required')
    return 'MATCHED_10_PERCENT' if abs(adaptive_ms-comparator_ms)/adaptive_ms<=0.10 else 'NOT_MATCHED_10_PERCENT'


def paired_family_ci(left,right,seed=20261129):
    if not left or len(left)!=len(right):
        raise ValueError('paired nonempty family sample required')
    fam=defaultdict(list)
    for l,r in zip(left,right):
        if l['family']!=r['family']:
            raise ValueError('inconsistent matched opening family')
        fam[l['family']].append(l['regret']-r['regret'])
    families=sorted(fam)
    means=[sum(fam[k])/len(fam[k]) for k in families]
    n=len(families)
    rng=random.Random(seed)
    draws=sorted(sum(rng.choice(means) for _ in means)/n for _ in range(3000))
    return {
        'n_roots':len(left),'n_family_clusters':n,
        'mean_regret_difference':sum(means)/n,
        'conditional_family_bootstrap_95':[draws[74],draws[2924]],
        'family_mean_paired_regret':dict(zip([str(k) for k in families],means)),
        'not_IID_per_root':True,'model_not_retrained_under_bootstrap':True,
    }


def imports():
    prev=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(prev) not in sys.path:sys.path.insert(0,str(prev))
    from evaluate_live08 import model_request,run_batch,score_completed_live,compare_rust_current_policy
    from train_gain08 import model_identity,current_root_features
    return model_request,run_batch,score_completed_live,compare_rust_current_policy,model_identity,current_root_features


def make_wire(row,mode,arm,models,modelhash,new,old):
    model_request,_,_,_,_,_=imports()
    q=model_request(row,mode,'full',models,modelhash,None,None)
    if arm in ('dynamic','myopic','myopic_time_match'):
        q['dynamic_policy']=new['modes'][mode]['frozen_policies'][arm]
    elif arm=='legacy_debug_myopic':
        q['adaptive_policy']=old['mode_policies'][mode]
    elif arm=='live_fixed16':
        q['force_stop_visits']=16
    elif arm=='validation_clock':
        q['time_budget_ms']=new['modes'][mode]['clock_validation_time_match']['cap_ms']
    elif arm!='full64':
        raise RuntimeError('unknown frozen live arm')
    assert 'oracle' not in q and 'exact_action_scores' not in q
    assert not('dynamic_policy' in q and 'adaptive_policy' in q)
    return q


def mean_metrics(subset,mode,arm):
    records=[row['policies'][mode][arm] for row in subset]
    n=len(records)
    if not n:raise ValueError('empty test cohort')
    return {
        'n_roots':n,'n_families':len({r['family'] for r in subset}),
        'mean_WDL':sum(x['WDL'] for x in records)/n,
        'mean_regret':sum(x['regret'] for x in records)/n,
        'optimal_WDL_action_count':sum(x['regret']==0 for x in records),
        'mean_search_ms':sum(x['search_ms'] for x in records)/n,
        'mean_request_ms_without_fresh':sum(x['request_ms_without_fresh'] for x in records)/n,
        'mean_game_request_ms_including_setup':sum(x['estimated_game_request_ms'] for x in records)/n,
        'mean_model_load_ms':sum(x['model_load_ms'] for x in records)/n,
        'mean_root_setup_ms':sum(x['setup_ms'] for x in records)/n,
        'mean_evaluator_calls':sum(x['evaluator_calls'] for x in records)/n,
        'mean_root_visits':sum(x['visits'] for x in records)/n,
        'stop_distribution':dict(Counter(str(x['visits']) for x in records)),
        'regret_distribution':dict(Counter(str(x['regret']) for x in records)),
        'WDL_distribution':dict(Counter(str(x['WDL']) for x in records)),
        'two_step_committed_count':sum(bool(x.get('dynamic_skipped_checkpoints')) for x in records),
    }


def calibration(subset,mode,rawlookup,oracle,model,current_root_features):
    result=[]
    for stage in range(3):
        rows=[]
        for source in subset:
            root=rawlookup[(source['id'],mode,'full64')]
            S=oracle[source['id']]['exact_action_scores']
            xs=current_root_features(root,stage)
            p=predict_option(model,stage,xs,variant='dynamic')
            sign=lambda x:(x>0)-(x<0)
            values=[sign(S[k['best_action']]) for k in root['steps']]
            gain=values[stage+1]-values[stage]
            cost=root['steps'][stage+1]['step_wall_us']/1000.0
            row={'id':source['id'],'family':source['family'],'prediction':p['predicted_single_gain'],
                 'realized':gain,'predicted_cost_ms':p['predicted_single_cost_ms'],
                 'actual_cost_ms':cost,'bucket':forecast_bin(p['predicted_single_gain']),
                 'decide_continue':p['action'] in ('single','commit_two')}
            if stage<2:
                row['joint_pred']=p['predicted_joint_gain']
                row['joint_actual']=values[stage+2]-values[stage]
                row['joint_forecast_cost_ms']=p['predicted_joint_cost_ms']
                row['joint_actual_cost_ms']=cost+root['steps'][stage+2]['step_wall_us']/1000.0
            rows.append(row)
        n=len(rows)
        buckets={}
        for bucket in range(6):
            arr=[r for r in rows if r['bucket']==bucket]
            buckets[str(bucket)]={
                'n':len(arr),
                'mean_predicted_gain':sum(r['prediction'] for r in arr)/len(arr) if arr else None,
                'mean_observed_gain':sum(r['realized'] for r in arr)/len(arr) if arr else None,
                'observed_negative':sum(r['realized']<0 for r in arr),
                'observed_positive':sum(r['realized']>0 for r in arr),
            }
        output={
            'from_budget':BUDGETS[stage], 'to_budget':BUDGETS[stage+1],
            'n':n,
            'mean_predicted_signed_gain':sum(r['prediction'] for r in rows)/n,
            'mean_realized_signed_gain':sum(r['realized'] for r in rows)/n,
            'signed_gain_MSE':sum((r['prediction']-r['realized'])**2 for r in rows)/n,
            'signed_gain_MAE':sum(abs(r['prediction']-r['realized']) for r in rows)/n,
            'signed_gain_counts':dict(Counter(str(r['realized']) for r in rows)),
            'mean_forecast_release_cost_ms':sum(r['predicted_cost_ms'] for r in rows)/n,
            'mean_real_release_cost_ms':sum(r['actual_cost_ms'] for r in rows)/n,
            'false_stop_if_one_step_positive':sum(not r['decide_continue'] and r['realized']>0 for r in rows),
            'continue_despite_one_step_nonpositive':sum(r['decide_continue'] and r['realized']<=0 for r in rows),
            'pre_registered_six_bins':buckets,
        }
        if stage<2:
            output['joint_two_step_calibration']={
                'target_budget':BUDGETS[stage+2],
                'mean_predicted_joint_gain':sum(r['joint_pred'] for r in rows)/n,
                'mean_realized_joint_gain':sum(r['joint_actual'] for r in rows)/n,
                'joint_gain_MSE':sum((r['joint_pred']-r['joint_actual'])**2 for r in rows)/n,
                'joint_gain_MAE':sum(abs(r['joint_pred']-r['joint_actual']) for r in rows)/n,
                'mean_predicted_joint_cost_ms':sum(r['joint_forecast_cost_ms'] for r in rows)/n,
                'mean_real_joint_cost_ms':sum(r['joint_actual_cost_ms'] for r in rows)/n,
                'positive_joint_given_nonpositive_one_step':sum(
                    r['joint_actual']>0 and r['realized']<=0 for r in rows),
            }
        result.append(output)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('run only from clean committed source with frozen models')
    model_request,run_batch,score_completed_live,compare_rust_current_policy,model_identity,current_root_features=imports()
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    roots=cohort['rows']
    assert len(roots)==384 and len({x['family'] for x in roots})==24
    assert Counter(x['group'] for x in roots)=={'fresh_A':192,'fresh_B':192}
    modelsfile=d/'evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json'
    fitmanifest=json.loads((d/'evidence/frozen_release_policies_20261121/FIT_MANIFEST.json').read_text())
    new=json.loads(modelsfile.read_text())
    assert sha(modelsfile)==fitmanifest['output_policy_sha256']
    assert not new['used_old_or_new_test_labels_for_fit']
    oldfile=d.parent/'cost_matched_voc_08/evidence/frozen_surrogate_20261121/frozen_surrogates.json'
    old=json.loads(oldfile.read_text())
    assert old['test_set_used_for_selection'] is False
    models=model_identity(repo)
    modelhash={mode:sha(p) for mode,p in models.items()}
    requests=[];keys=[]
    for root in roots:
        for mode in MODES:
            for arm in ARMS:
                q=make_wire(root,mode,arm,models,modelhash,new,old)
                requests.append(q)
                keys.append((root['id'],mode,arm))
    assert len(requests)==384*3*7==8064
    exe=args.binary.resolve()
    assert exe.is_file() and sha(exe)=='4db989b0ede0c56077de77a820a99cd777adcf87c98d7faedc62611491eb8a03'
    t=time.perf_counter()
    raw=run_batch(exe,requests)
    wall=time.perf_counter()-t
    assert len(raw)==8064
    # Game-theoretic exact labels are loaded AFTER ALL actual Rust root
    # searches. Only then can we evaluate regret and signed realized gain.
    source=d/'evidence/teacher_new384_20261121/EXACT_384_LABELS.json'
    teacher=json.loads(source.read_text())
    assert teacher['statuses']=={'OK':384}
    oracle={r['id']:r for r in teacher['rows']}
    assert len(oracle)==384
    byroot={r['id']:{
        'id':r['id'],'family':r['family'],'group':r['group'],
        'history':r['history'],'current_player':r['current_player'],
        'policies':{}
    } for r in roots}
    rootindex={r['id']:r for r in roots}
    lookup={}
    for (ident,mode,arm),response in zip(keys,raw):
        orig=rootindex[ident]
        if (response['history']!=orig['history'] or
            response['board_bottom_up']!=orig['board'] or
            response['legal_columns_0based']!=orig['legal_columns'] or
            response['side_to_move']!=orig['current_player'] or
            response['seed']!=17):
            raise RuntimeError('Rust source game/board/policy request identity mismatch')
        if response['eval_mode']!={'uniform':'uniform_no_neural_network',
                                  'new_mlp':'trained_tiny_mlp_cpu',
                                  'conv':'trained_conv3x3_cpu'}[mode]:
            raise RuntimeError('Rust evaluator role changed')
        if mode!='uniform' and response['cache_model'] is not True:
            raise RuntimeError('NN weights not resident in Rust server')
        scored=score_completed_live(response,oracle[ident]['exact_action_scores'])
        scored['dynamic_skipped_checkpoints']=response.get('dynamic_skipped_checkpoints',[])
        byroot[ident]['policies'].setdefault(mode,{})[arm]=scored
        lookup[(ident,mode,arm)]=response
        if arm=='legacy_debug_myopic':
            compare_rust_current_policy(response,old['mode_policies'][mode])
        if arm in ('dynamic','myopic','myopic_time_match'):
            model=new['modes'][mode]['frozen_policies'][arm]
            for dec in response['dynamic_decisions']:
                stage={8:0,16:1,32:2}[dec['at_visits']]
                x=current_root_features(response,stage)
                py=predict_option(model,stage,x,variant='dynamic' if arm=='dynamic' else 'myopic')
                if dec['action']!=py['action'] or max(abs(a-b) for a,b in zip(dec['root_features'],x))>3e-9:
                    raise RuntimeError('Rust dynamic stop choice/feature disagreement')
                for key in ('predicted_single_gain','predicted_single_cost_ms','net_single'):
                    if abs(dec[key]-py[key])>3e-9:
                        raise RuntimeError('Rust release dynamic forecast diverged '+key)
            # Full-trace replay remains a POSTHOC check, never a runtime input.
            # Computation was never selected using oracle actions.
        if arm=='live_fixed16':
            assert response['force_stop_visits']==16
            assert response['stopped_at_visits']==16 and len(response['steps'])==2
            assert response['fresh16']['resident_match']
        if arm=='validation_clock':
            assert response['time_budget_ms']==new['modes'][mode]['clock_validation_time_match']['cap_ms']
    paired=[byroot[r['id']] for r in roots]
    for root in paired:
        for mode in MODES:
            baseline=lookup[(root['id'],mode,'full64')]
            assert len(baseline['steps'])==4 and baseline['stopped_at_visits']==64
            for arm in ARMS[1:]:
                actual=lookup[(root['id'],mode,arm)]
                assert actual['root_hash']==baseline['root_hash']
                for x,y in zip(baseline['steps'],actual['steps']):
                    if (x['best_action']!=y['best_action'] or
                        x['evaluator_calls']!=y['evaluator_calls'] or
                        x['visit_policy']!=y['visit_policy']):
                        raise RuntimeError('MCTS prefix differs under early stop controller')
                if arm in ('dynamic','myopic','myopic_time_match'):
                    model=new['modes'][mode]['frozen_policies'][arm]
                    S=oracle[root['id']]['exact_action_scores']
                    signed=[(S[s['best_action']]>0)-(S[s['best_action']]<0) for s in baseline['steps']]
                    record={
                        'features':[current_root_features(baseline,i) for i in range(3)],
                        'utility':signed,
                        'search_stage_wall_ms':[s['step_wall_us']/1000 for s in baseline['steps']],
                    }
                    replay=simulate_bounded_policy(model,record,
                                variant='dynamic' if arm=='dynamic' else 'myopic')
                    assert replay['visits']==actual['stopped_at_visits']
                    assert replay['skipped_stages']==actual['dynamic_skipped_checkpoints']
                    assert [x['action'] for x in replay['decisions']]==[x['action'] for x in actual['dynamic_decisions']]
    summary={
        'schema':'quartz_cog02_dynamic10_two_step_release_heldout_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'number_live_release_root_searches':8064,
        'new_cohorts':2,'new_unique_opening_families':24,'new_roots':384,
        'release_rust_binary_sha256':sha(exe),
        'frozen_new_model_sha256':sha(modelsfile),
        'previous_debug_surrogate_sha256':sha(oldfile),
        'exact_new_oracle_sha256':sha(source),
        'wall_seconds_all_root_searches':wall,
        'mode_results':{},
        'not_claimed':['Bayesian posterior or optimal knowledge gradient',
                       'root level search seeds are independent',
                       'root descendants IID','exact per-root CPU cost match'],
    }
    for group in ('fresh_A','fresh_B','pooled'):
        subset=[r for r in paired if group=='pooled' or r['group']==group]
        result={}
        for mode in MODES:
            arms={arm:mean_metrics(subset,mode,arm) for arm in ARMS}
            comps={}
            for other in ('myopic','myopic_time_match','legacy_debug_myopic',
                          'live_fixed16','validation_clock'):
                lhs=[{'family':r['family'],'regret':r['policies'][mode]['dynamic']['regret']} for r in subset]
                rhs=[{'family':r['family'],'regret':r['policies'][mode][other]['regret']} for r in subset]
                ref=arms[other]
                dy=arms['dynamic']
                ratio=abs(dy['mean_search_ms']-ref['mean_search_ms'])/dy['mean_search_ms']
                comps['dynamic_vs_'+other]={
                    'regret_difference_dynamic_minus_other':dy['mean_regret']-ref['mean_regret'],
                    'actual_mean_search_cost_mismatch_fraction':ratio,
                    'matched_cost_10pct':cost_match_status(dy['mean_search_ms'],ref['mean_search_ms']),
                    'conditional_family_bootstrap':paired_family_ci(lhs,rhs),
                    'per_root_cost_matched':False,
                }
            informative=[]
            for r in subset:
                S=oracle[r['id']]['exact_action_scores']
                legal=rootindex[r['id']]['legal_columns']
                if len({(S[c]>0)-(S[c]<0) for c in legal})>1:informative.append(r)
            result[mode]={
                'arms':arms,'paired_comparisons':comps,
                'informative_WDL_decision_root_count':len(informative),
                'informative_root_adaptive_regret':(
                    sum(r['policies'][mode]['dynamic']['regret'] for r in informative)/len(informative) if informative else None),
                'by_side':{str(side):{
                    arm:{
                        'n':sum(r['current_player']==side for r in subset),
                        'mean_regret':sum(r['policies'][mode][arm]['regret'] for r in subset if r['current_player']==side)/
                            max(1,sum(r['current_player']==side for r in subset))}
                    for arm in ('dynamic','myopic','live_fixed16','validation_clock')}
                    for side in (-1,1)},
                'stagewise_calibration':calibration(subset,mode,lookup,oracle,
                    new['modes'][mode]['frozen_policies']['dynamic'],current_root_features),
                'lambda_dynamic':new['modes'][mode]['frozen_policies']['dynamic']['lambda_wdl_per_ms'],
                'lambda_myopic':new['modes'][mode]['frozen_policies']['myopic']['lambda_wdl_per_ms'],
                'lambda_myopic_costmatched':new['modes'][mode]['frozen_policies']['myopic_time_match']['lambda_wdl_per_ms'],
                'validation_chosen_clock_cap_ms':new['modes'][mode]['clock_validation_time_match']['cap_ms'],
            }
        summary['mode_results'][group]=result
    for name,p in models.items():
        assert sha(p)==modelhash[name]
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name,body in [
        ('RAW_8064_RELEASE_ROOTS.json',raw),
        ('PAIRED_384_NEW_ROOTS.json',paired),
        ('SUMMARY_TWO_STEP.json',summary)
    ]:
        (out/name).write_text(json.dumps(body,sort_keys=True,ensure_ascii=False)+'\n')
    manifest={
        'schema':'quartz_cog02_dynamic10_heldout_execution_manifest_v1',
        'source_commit':summary['source_commit'],
        'source_tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
        'compiled_rust_source_commit':'4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2',
        'binary_sha256':sha(exe),'fitted_model_sha256':sha(modelsfile),
        'old_debug_model_sha256':sha(oldfile),
        'new_test_cohort_sha256':sha(d/'COHORT_384_FROZEN.json'),
        'new_exact_external_oracle_sha256':sha(source),
        'n_actual_release_root_searches':len(raw),
        'uses_new_test_labels_only_after_Rust':True,
        'source_training_families_only':True,
        'output_sha256':{p.name:sha(p) for p in out.iterdir()},
    }
    (out/'EXECUTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('C10_DYNAMIC_HELDOUT_RELEASE_DONE',json.dumps({
        'n_live_roots':len(raw),'elapsed_s':wall,
        'uniform':{group:{
            'dynamic_regret':summary['mode_results'][group]['uniform']['arms']['dynamic']['mean_regret'],
            'myopic_regret':summary['mode_results'][group]['uniform']['arms']['myopic']['mean_regret'],
            'clock_regret':summary['mode_results'][group]['uniform']['arms']['validation_clock']['mean_regret'],
            'dynamic_cost_ms':summary['mode_results'][group]['uniform']['arms']['dynamic']['mean_search_ms'],
            'dynamic_commits':summary['mode_results'][group]['uniform']['arms']['dynamic']['two_step_committed_count'],
            'clock_match':summary['mode_results'][group]['uniform']['paired_comparisons']['dynamic_vs_validation_clock']['matched_cost_10pct'],
        } for group in ('fresh_A','fresh_B','pooled')}
    },sort_keys=True))


if __name__=='__main__':main()
