#!/usr/bin/env python3
"""Live CPU cost-aware metareasoning experiment on disjoint Connect4 families.

Separate FIRST the actual Rust search (full, adaptive, checkpoint-clock) from
reading external exact oracle WDL labels. No oracle path appears in the
request. This is a supervised signed-gain surrogate, not Bayesian KG.

All evaluators stay resident in a single Rust server process with a frozen
per-model byte identity. Search times reflect actual measured debug CPU work.
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
import time

from train_gain08 import (
    BUDGETS,MODES,sha,git,current_root_features,
    classify_outcome, policy_stop_budget,model_identity
)


def score_completed_live(raw:dict,exact_scores:list[int])->dict:
    if len(exact_scores)!=7:
        raise ValueError('seven exact Connect4 per-column scores required')
    steps=raw.get('steps')
    if not isinstance(steps,list) or not 1<=len(steps)<=4:
        raise ValueError('no real MCTS checkpoint')
    if [s.get('budget') for s in steps] != list(BUDGETS[:len(steps)]):
        raise ValueError('nonresident checkpoint budget trace')
    b=steps[-1]['budget']
    if raw.get('stopped_at_visits')!=b:
        raise ValueError('reported stop budget differs from executed visits')
    root=raw.get('root_hash')
    legal=raw.get('legal_columns_0based')
    if not isinstance(root,str) or not root or not isinstance(legal,list) or not legal:
        raise ValueError('missing root/legality identity')
    for col in range(7):
        if (col in legal)!=(exact_scores[col]!=-1000):
            raise ValueError('exact external legal sentinel mismatch')
    init=raw.get('root_initial_evaluator_calls')
    if type(init)!=int or init<0:
        raise ValueError('missing root initial evaluation count')
    prior=init
    for s in steps:
        if s.get('root_hash')!=root or s.get('root_visits')!=s['budget']:
            raise ValueError('root changed during resident search')
        action=s.get('best_action')
        if type(action)!=int or action not in legal or exact_scores[action]==-1000:
            raise ValueError('illegal selected Connect4 action')
        calls=s.get('evaluator_calls')
        if type(calls)!=int or calls<prior or calls-prior!=s.get('additional_evaluator_calls'):
            raise ValueError('incorrect realized evaluator call accounting')
        prior=calls
        t=s.get('step_wall_us')
        if type(t)!=int or t<0:
            raise ValueError('bad measured search wall time')
    if b>=16 and raw.get('fresh16',{}).get('resident_match') is not True:
        raise ValueError('fresh16 did not agree with resident prefix')
    if b==8 and raw.get('fresh16',{}).get('available') is not False:
        raise ValueError('stopped before 16 yet future control claimed available')
    action=steps[-1]['best_action']
    wdl=classify_outcome(exact_scores[action])
    best=max(classify_outcome(exact_scores[col]) for col in legal)
    load=raw.get('model_load_wall_us',0)
    setup=raw.get('root_setup_wall_us',0)
    request=raw.get('request_wall_us_without_fresh',0)
    if any(type(v) is not int or v<0 for v in (load,setup,request)):
        raise ValueError('malformed actual CPU latency fields')
    search=sum(s['step_wall_us'] for s in steps)
    return {
        'visits':b,'action':action,'WDL':wdl,'regret':best-wdl,
        'optimal_root_WDL':best,
        'evaluator_calls':prior,
        'search_ms':search/1000.,
        'setup_ms':setup/1000.,
        'model_load_ms':load/1000.,
        'request_ms_without_fresh':request/1000.,
        'estimated_game_request_ms':(load+setup+search)/1000.,
        'stage_count':len(steps),
        'cache_hit':bool(raw.get('model_cache_hit')),
        'root_hash':root,
        'prefix_actions':[s['best_action'] for s in steps],
        'prefix_evaluator_calls':[s['evaluator_calls'] for s in steps],
    }


def compare_rust_current_policy(raw:dict,policy:dict)->dict:
    steps=raw.get('steps')
    decisions=raw.get('adaptive_decisions')
    if not isinstance(steps,list) or not isinstance(decisions,list) or len(decisions)>3:
        raise ValueError('actual adaptive observations missing')
    expected_cont=[]
    for i,record in enumerate(decisions):
        if record.get('stage_visits')!=BUDGETS[i]:
            raise ValueError('adaptive stage order changed')
        actual_features=current_root_features(raw,i)
        rust_features=record.get('root_features')
        if not isinstance(rust_features,list) or len(rust_features)!=9:
            raise ValueError('live Rust feature vector missing')
        err=max(abs(x-y) for x,y in zip(actual_features,rust_features))
        if err>3e-9:
            raise ValueError('Rust/Python current root feature parity failed')
        predicted=sum(a*b for a,b in zip(policy['coefficients'][i],actual_features))
        cost=policy['forecast_cost_ms'][i]
        expected=bool(predicted>policy['lambda_wdl_per_ms']*cost)
        if abs(record['predicted_signed_gain']-predicted)>3e-9:
            raise ValueError('learned next-gain forecast has future/source contamination')
        if abs(record['forecast_next_ms']-cost)>1e-9:
            raise ValueError('actual predicted next cost differs from frozen training cost')
        if record['continue'] is not expected:
            raise ValueError('actual stop/continue disagrees with frozen training coefficients')
        expected_cont.append(expected)
    stopped=raw['stopped_at_visits']
    if stopped!=64:
        assert decisions and decisions[-1]['continue'] is False
    if len(steps) != (list(BUDGETS).index(stopped)+1):
        raise ValueError('MCTS executed unrequested future steps')
    return {'n_decisions':len(decisions),
            'max_feature_parity_error':max(0,max(
                max(abs(a-b) for a,b in zip(current_root_features(raw,i),row['root_features']))
                for i,row in enumerate(decisions)) if decisions else 0),
            'continue':expected_cont,
            'observed_stop_visits':stopped}


def model_request(root:dict,mode:str,kind:str,model_paths:dict,
                  models_sha:dict,policy:dict|None,cap:float|None):
    request={
        'cmd':'research_connect4_cpu',
        'history':root['history'],'seed':17,
        'budgets':list(BUDGETS),'cache_model':True,
    }
    if mode!='uniform':
        request.update({
            'eval_mode':'tiny_mlp' if mode=='new_mlp' else 'conv3x3',
            'model_path':str(model_paths[mode].resolve()),
            'model_identity':models_sha[mode],
        })
    if kind=='adaptive':
        assert policy is not None and cap is None
        request['adaptive_policy']={
            'schema':policy['schema'],'features':policy['features'],
            'coefficients':policy['coefficients'],
            'forecast_cost_ms':policy['forecast_cost_ms'],
            'lambda_wdl_per_ms':policy['lambda_wdl_per_ms'],
        }
    elif kind=='fixed_clock':
        assert cap is not None and policy is None
        request['time_budget_ms']=cap
    elif kind!='full':
        raise ValueError('unexpected experimental controller arm')
    return request


def run_batch(exe:Path,requests:list[dict])->list[dict]:
    wire='\n'.join(json.dumps(q,separators=(',',':'),sort_keys=True)
        for q in requests)+'\n'+json.dumps({'cmd':'quit'})+'\n'
    result=subprocess.run([str(exe),'--server'],input=wire.encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=190,check=False)
    if result.returncode!=0:
        raise RuntimeError(f'Rust MCTS server failed: {result.returncode}, stderr={result.stderr[-1500:]!r}')
    replies=[json.loads(s) for s in result.stdout.decode('utf-8','replace').splitlines() if s.strip()]
    if len(replies)!=len(requests):
        raise RuntimeError(f'root result cardinality mismatch: {len(replies)} vs {len(requests)}')
    for q,r in zip(requests,replies):
        if r.get('error') or r.get('history')!=q['history'] or r.get('seed')!=17:
            raise RuntimeError('invalid request/response identity '+str(r)[:420])
    return replies


def summarize_cases(rows:list[dict],mode:str,kind:str)->dict:
    measurements=[r['modes'][mode][kind] for r in rows]
    n=len(measurements)
    if n!=192:raise ValueError('expected heldout exactly 192 roots')
    return {
        'n':n,
        'families':len({r['family'] for r in rows}),
        'mean_WDL':sum(r['WDL'] for r in measurements)/n,
        'mean_regret':sum(r['regret'] for r in measurements)/n,
        'optimal_WDL_count':sum(r['regret']==0 for r in measurements),
        'mean_actual_search_ms':sum(r['search_ms'] for r in measurements)/n,
        'mean_actual_request_ms_without_fresh':sum(r['request_ms_without_fresh'] for r in measurements)/n,
        'mean_estimated_game_request_ms':sum(r['estimated_game_request_ms'] for r in measurements)/n,
        'mean_evaluator_calls':sum(r['evaluator_calls'] for r in measurements)/n,
        'mean_visits':sum(r['visits'] for r in measurements)/n,
        'stop_visits':dict(Counter(str(r['visits']) for r in measurements)),
        'WDL_counts':dict(Counter(str(r['WDL']) for r in measurements)),
        'regret_counts':dict(Counter(str(r['regret']) for r in measurements)),
    }


def paired_group_CI(rows,mode,left,right,seed=20261121):
    groups=defaultdict(list)
    for row in rows:
        a=row['modes'][mode][left]['regret']
        b=row['modes'][mode][right]['regret']
        groups[row['family']].append(a-b)
    groupnames=sorted(groups)
    assert len(groupnames)==12 and all(len(groups[k])==16 for k in groupnames)
    family_mean=[sum(groups[k])/16 for k in groupnames]
    rng=random.Random(seed)
    boot=[sum(rng.choice(family_mean) for _ in family_mean)/12 for _ in range(3000)]
    boot.sort()
    return {'n_families':12,
            'mean_regret_difference_left_minus_right':sum(family_mean)/12,
            'conditional_95_family_bootstrap':[boot[74],boot[2924]],
            'fitted_policy_retrained_in_bootstrap':False,
            'sampling_unit':'opening family, not individual descendant root'}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--rust-binary',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if git(repo,'status','--porcelain'):
        raise RuntimeError('live 192-family test only from clean committed source')
    paths=model_identity(repo)
    shas={name:sha(p) for name,p in paths.items()}
    frozen=json.loads((d/'evidence/frozen_surrogate_20261121/frozen_surrogates.json').read_text())
    time_caps=json.loads((d/'evidence/validation_time_caps_20261121/TIME_CAPS_FROZEN.json').read_text())
    trained=json.loads((d.parent/'connect4_generalization_07/COHORT_1024_FROZEN.json').read_text())
    labels_file=d.parent/'connect4_generalization_07/evidence/teacher_1024_20261121/labels.json'
    roots=[x for x in trained['rows'] if x['split']=='test']
    assert len(roots)==192 and len({r['family'] for r in roots})==12
    if frozen['test_set_used_for_selection'] or time_caps['test_family_used']:
        raise RuntimeError('test leakage flagged in training/validation model')
    if sha(d/'evidence/frozen_surrogate_20261121/frozen_surrogates.json')!=time_caps['policy_sha256']:
        raise RuntimeError('policies changed since validation-only cap freeze')
    # Third-party source data are independent of Rust MCTS. No solver action
    # score or target enters the wire requests in ANY arm.
    requests=[]
    keylist=[]
    for row in roots:
        for mode in MODES:
            for arm in ('full','adaptive','fixed_clock'):
                p=frozen['mode_policies'][mode] if arm=='adaptive' else None
                cap=time_caps['modes'][mode]['selected_cap_ms'] if arm=='fixed_clock' else None
                requests.append(model_request(row,mode,arm,paths,shas,p,cap))
                keylist.append((row['id'],mode,arm))
    assert len(requests)==1728
    for req in requests:
        assert 'exact_action_scores' not in req and 'oracle' not in req
    exe=args.rust_binary.resolve()
    if not exe.is_file():raise FileNotFoundError(exe)
    started=time.perf_counter()
    raw=run_batch(exe,requests)
    total_runtime_s=time.perf_counter()-started
    # Critical scientific isolation: reference outcomes read AFTER all actual
    # live (or full deterministic-prefix) searches have completed.
    oracle=json.loads(labels_file.read_text())
    known={x['id']:x for x in oracle['rows'] if x['split']=='test' and x['status']=='OK'}
    assert len(known)==192
    byroot={r['id']:r for r in roots}
    assembled={r['id']:{'id':r['id'],'family':r['family'],
             'history':r['history'],'current_player':r['current_player'],'modes':{}}
             for r in roots}
    rawmap={}
    for (id,mode,arm),response in zip(keylist,raw):
        source=byroot[id]
        if response['side_to_move']!=source['current_player'] or response['ply']!=source['plies']:
            raise RuntimeError('actual Rust game state differs from holdout')
        if response['board_bottom_up']!=source['board'] or response['legal_columns_0based']!=source['legal_columns']:
            raise RuntimeError('actual Rust gravity or action legality discrepancy')
        expected={'uniform':'uniform_no_neural_network','new_mlp':'trained_tiny_mlp_cpu','conv':'trained_conv3x3_cpu'}[mode]
        if response['eval_mode']!=expected:
            raise RuntimeError('actual frozen evaluator mode mismatch')
        if mode!='uniform' and response['cache_model'] is not True:
            raise RuntimeError('model should be resident in a single Rust process')
        score=score_completed_live(response,known[id]['exact_action_scores'])
        assembled[id]['modes'].setdefault(mode,{})[arm]=score
        rawmap[(id,mode,arm)]=response
        if arm=='adaptive':
            compare_rust_current_policy(response,frozen['mode_policies'][mode])
            if not response['adaptive_decisions']:
                raise RuntimeError('adaptive route did not run learned controller')
        if arm=='fixed_clock':
            if response['time_budget_ms']!=time_caps['modes'][mode]['selected_cap_ms']:
                raise RuntimeError('fixed-time cap drifted from validation lock')
    paired=[assembled[r['id']] for r in roots]
    for item in paired:
        for mode in MODES:
            cases=item['modes'][mode]
            assert len(cases)==3
            complete=rawmap[(item['id'],mode,'full')]
            full=cases['full']
            assert full['visits']==64
            for arm in ('adaptive','fixed_clock'):
                live=rawmap[(item['id'],mode,arm)]
                idx=len(live['steps'])
                for ref,actual in zip(complete['steps'][:idx],live['steps']):
                    if (ref['best_action']!=actual['best_action']
                        or ref['evaluator_calls']!=actual['evaluator_calls']
                        or ref['visit_policy']!=actual['visit_policy']):
                        raise RuntimeError('live online policy does not preserve actual full-tree prefix')
                assert cases[arm]['root_hash']==full['root_hash']
    report={
        'schema':'quartz_cog02_cost_matched_voc08_live_test_v1',
        'source_head':git(repo,'rev-parse','HEAD'),
        'n_test_roots':192,'test_families':12,'n_rust_requests':len(requests),
        'runtime_wall_s':total_runtime_s,
        'mode_policy_lambda':{mode:frozen['mode_policies'][mode]['lambda_wdl_per_ms'] for mode in MODES},
        'validation_selected_time_caps_ms':{mode:time_caps['modes'][mode]['selected_cap_ms'] for mode in MODES},
        'model_cache_residency':{
            mode:{
                'hits':sum(bool(rawmap[(r['id'],mode,arm)]['model_cache_hit']) for r in roots for arm in ('full','adaptive','fixed_clock')),
                'misses':sum(not rawmap[(r['id'],mode,arm)]['model_cache_hit'] for r in roots for arm in ('full','adaptive','fixed_clock')),
            } for mode in MODES
        },
        'comparisons':{},'nonclaims':[
            'trained surrogate is NOT Bayesian KG',
            'same visit budget does not mean same CPU cost',
            'zero-cost model initialization not assumed; one cache miss per model',
            'costed CPU timer checked only at discrete checkpoints',
            '12 test opening families, not 192 IID samples',
        ],
    }
    for mode in MODES:
        per={arm:summarize_cases(paired,mode,arm)
             for arm in ('full','adaptive','fixed_clock')}
        # Fixed initial 8,16,32,64 from physically measured FULL trajectories:
        # legitimate retrospective prefixes, NOT four independent live stops.
        for idx,b in enumerate(BUDGETS):
            x=[]
            for row in paired:
                rawrow=rawmap[(row['id'],mode,'full')]
                full_score=row['modes'][mode]['full']
                best=rawrow['steps'][idx]
                optimal=full_score['optimal_root_WDL']
                exact=known[row['id']]['exact_action_scores']
                u=classify_outcome(exact[best['best_action']])
                x.append({'U':u,'regret':optimal-u,
                          'eval_calls':best['evaluator_calls'],
                          'search_ms':sum(s['step_wall_us'] for s in rawrow['steps'][:idx+1])/1000.0})
            per['fixed_'+str(b)]={
                'n':192,
                'mean_WDL':sum(z['U'] for z in x)/len(x),
                'mean_regret':sum(z['regret'] for z in x)/len(x),
                'mean_actual_search_ms':sum(z['search_ms'] for z in x)/len(x),
                'mean_evaluator_calls':sum(z['eval_calls'] for z in x)/len(x),
                'description':'counterfactual prefix of genuine deterministic full MCTS',
            }
        obs=per['adaptive']; fixed=per['fixed_clock']
        relative=abs(obs['mean_actual_search_ms']-fixed['mean_actual_search_ms'])/max(obs['mean_actual_search_ms'],1e-12)
        state='COST_MATCHED_WITHIN_10_PERCENT' if relative<=0.10 else 'COST_MATCHED_NOT_ESTABLISHED'
        per['adaptive_minus_fixed_clock']={
            'mean_regret_difference':obs['mean_regret']-fixed['mean_regret'],
            'mean_WDL_difference':obs['mean_WDL']-fixed['mean_WDL'],
            'relative_search_cost_mismatch_fraction':relative,
            'cost_match_status':state,
            'paired_test_family_interval':paired_group_CI(paired,mode,'adaptive','fixed_clock'),
            'primary_scientific_admission':state=='COST_MATCHED_WITHIN_10_PERCENT',
        }
        # Report signed gain actually purchased by the policy at each
        # completed stage; use posthoc full trajectory to label impact,
        # NEVER fed into the online policy.
        spent=Counter()
        for row in paired:
            full=rawmap[(row['id'],mode,'full')]
            reached=row['modes'][mode]['adaptive']['visits']
            scores=known[row['id']]['exact_action_scores']
            utility=[classify_outcome(scores[s['best_action']]) for s in full['steps']]
            for i,(from_b,to_b) in enumerate(zip(BUDGETS,BUDGETS[1:])):
                if reached>=to_b:
                    spent[str(utility[i+1]-utility[i])]+=1
        per['adaptive_paid_incremental_signed_gains']=dict(spent)
        report['comparisons'][mode]=per
    for name,p in paths.items():
        if sha(p)!=shas[name]:
            raise RuntimeError('model file changed during frozen live test')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name,payload in [('raw_1728_rust_requests.json',raw),
                        ('paired_192_livetest.json',paired),
                        ('summary.json',report)]:
        (out/name).write_text(json.dumps(payload,sort_keys=True,ensure_ascii=False)+'\n')
    manifest={
        'schema':'quartz_cog02_live_eval08_execution_manifest_v1',
        'source_head':git(repo,'rev-parse','HEAD'),'source_tree':git(repo,'rev-parse','HEAD^{tree}'),
        'rust_binary_sha256':sha(exe),
        'frozen_surrogate_sha256':sha(d/'evidence/frozen_surrogate_20261121/frozen_surrogates.json'),
        'frozen_time_caps_sha256':sha(d/'evidence/validation_time_caps_20261121/TIME_CAPS_FROZEN.json'),
        'frozen_heldout_cohort_sha256':sha(d.parent/'connect4_generalization_07/COHORT_1024_FROZEN.json'),
        'frozen_test_oracle_sha256':sha(labels_file),
        'model_sha256':shas,
        'number_queries':len(requests),'sample_unit':'12 test families, 192 correlated roots',
        'output_sha256':{p.name:sha(p) for p in out.iterdir()},
        'online_adaptive':True,'live_fixed_time_cap':True,
        'no_oracle_in_rust_runtime':True,
    }
    (out/'execution_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print('COG02_LIVE_HELDOUT_COMPLETED',json.dumps({
        'queries':len(requests),'wall_s':total_runtime_s,
        'by_mode':{mode:{
            'adaptive_regret':report['comparisons'][mode]['adaptive']['mean_regret'],
            'fixed_clock_regret':report['comparisons'][mode]['fixed_clock']['mean_regret'],
            'actual_search_time_ratio_mismatch':report['comparisons'][mode]['adaptive_minus_fixed_clock']['relative_search_cost_mismatch_fraction'],
            'cost_match_status':report['comparisons'][mode]['adaptive_minus_fixed_clock']['cost_match_status'],
            'adaptive_stops':report['comparisons'][mode]['adaptive']['stop_visits'],
        } for mode in MODES},'output':str(out)
    },sort_keys=True))


if __name__=='__main__':main()
