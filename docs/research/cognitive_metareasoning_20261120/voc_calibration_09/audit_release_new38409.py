#!/usr/bin/env python3
"""One independent exact 4608-release-search source, cost and calibration audit.

No calls to production scoring or stop-controller code. Recomputes the WDL
of actual actions from newly frozen external oracle, current-only estimates,
fixed16 independent termination, clock consistency, and 24+24 source re-solves.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import subprocess


MODES=('uniform','new_mlp','conv')
ARMS=('full64','adaptive','live_fixed16','validation_clock')


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def replay_board(history):
    b=[0]*42
    heights=[0]*7
    for i,char in enumerate(history):
        x=int(char)-1
        assert 0<=x<7 and heights[x]<6
        b[heights[x]*7+x]=1 if i%2==0 else -1
        heights[x]+=1
    return b,[i for i in range(7) if heights[i]<6]


def sign(n):
    assert type(n) is int and n!=-1000 and -21<=n<=21
    return int(n>0)-int(n<0)


def features(root,step):
    legal=root['legal_columns_0based']
    b=step['budget']
    dense=[0.0]*7
    seen=set()
    for x,p in step['visit_policy']:
        assert x in legal and x not in seen and math.isfinite(p) and p>=0
        seen.add(x);dense[x]=float(p)
    z=sum(dense); assert abs(z-1)<1e-4
    dense=[a/z for a in dense]
    ranked=sorted(dense,reverse=True)
    entropy=-sum(p*math.log(p) for p in dense if p>0)/math.log(7)
    return [1.0,b/64,entropy,ranked[0],ranked[0]-ranked[1],
            len(legal)/7,step['evaluator_calls']/(b+1),
            root['ply']/42,1.0 if root['side_to_move']==1 else 0.0]


def solver(binpath,book,h):
    p=subprocess.run([str(binpath),'-a','-b',str(book)],
                     input=(h+'\n').encode(),capture_output=True,timeout=12,check=True)
    fields=p.stdout.decode().split()
    assert len(fields)==8 and fields[0]==h
    return [int(v) for v in fields[1:]]


def main():
    d=Path(__file__).resolve().parent
    ev=d/'evidence/new384_release_4608_20261121'
    m=json.loads((ev/'EXECUTION_MANIFEST.json').read_text())
    pairs=json.loads((ev/'PAIRED_384_NEW_FAMILY.json').read_text())
    raw=json.loads((ev/'RAW_4608_RELEASE_MCTS.json').read_text())
    report=json.loads((ev/'SUMMARY_CALIBRATION.json').read_text())
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    truth=json.loads((d/'evidence/new384_exact_teacher_20261121/labels.json').read_text())
    source=d.parent/'cost_matched_voc_08'
    policy=json.loads((source/'evidence/frozen_surrogate_20261121/frozen_surrogates.json').read_text())
    cap=json.loads((d/'evidence/release_validation_1152_20261121/VALIDATION_RELEASE_CAPS.json').read_text())
    assert m['source_commit']=='7b854f1b1c05e87ca7fceb6c153d630fafb30469'
    assert m['n_real_release_requests']==len(raw)==4608
    assert len(pairs)==len(cohort['rows'])==len(truth['rows'])==384
    assert sha(d/'COHORT_384_FROZEN.json')==m['new384_cohort_sha256']
    assert sha(d/'evidence/new384_exact_teacher_20261121/labels.json')==m['new384_teacher_sha256']
    assert sha(source/'evidence/frozen_surrogate_20261121/frozen_surrogates.json')==m['old_policy_sha256']
    assert sha(d/'evidence/release_validation_1152_20261121/VALIDATION_RELEASE_CAPS.json')==m['release_validation_caps_sha256']
    for name,expected in m['output_sha256'].items():
        assert sha(ev/name)==expected,name
    binary=Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo')
    assert sha(binary)==m['binary_sha256']
    assert m['compiled_rust_source_commit']=='1c2b00c1bad2585380fee97bde65e3862dc1f25d'
    oracle={r['id']:r for r in truth['rows']}
    mode_summary=defaultdict(lambda:defaultdict(dict))
    cache=Counter()
    adaptive_pred_errors=0
    clock_decisions=0
    root_keys=set()
    for idx,(root,joined) in enumerate(zip(cohort['rows'],pairs)):
        assert root['id']==joined['id'] and root['history']==joined['history']
        assert root['family']==joined['family'] and root['cohort']==joined['cohort']
        board,legal=replay_board(root['history'])
        assert board==root['board'] and legal==root['legal_columns']
        scores=oracle[root['id']]['exact_action_scores']
        assert all((scores[c]==-1000)==(c not in legal) for c in range(7))
        optimal=max(sign(scores[c]) for c in legal)
        root_signature=(root['history'],root['family'])
        assert root_signature not in root_keys
        root_keys.add(root_signature)
        for j,mode in enumerate(MODES):
            p=policy['mode_policies'][mode]
            clockcap=cap['modes'][mode]['cap_ms']
            controls={}
            for k,arm in enumerate(ARMS):
                r=raw[(idx*3+j)*4+k]
                assert r['history']==root['history'] and r['board_bottom_up']==board
                assert r['legal_columns_0based']==legal
                assert r['side_to_move']==root['side_to_move']
                assert r['seed']==17
                assert r['eval_mode']=={
                    'uniform':'uniform_no_neural_network',
                    'new_mlp':'trained_tiny_mlp_cpu',
                    'conv':'trained_conv3x3_cpu',
                }[mode]
                if mode!='uniform':
                    cache[(mode,bool(r['model_cache_hit']))]+=1
                assert r['stopped_at_visits']==r['steps'][-1]['budget']
                assert len(r['steps']) in (1,2,3,4)
                assert r['fresh16']['available'] is (len(r['steps'])>=2)
                if len(r['steps'])>=2:assert r['fresh16']['resident_match']
                initial=r['root_initial_evaluator_calls']
                t=0
                for stage,s in enumerate(r['steps']):
                    assert s['budget']==s['root_visits']==[8,16,32,64][stage]
                    assert s['best_action'] in legal
                    assert s['root_hash']==r['root_hash']
                    assert s['evaluator_calls']>=initial
                    assert s['additional_evaluator_calls']==s['evaluator_calls']-initial
                    initial=s['evaluator_calls']
                    t+=s['step_wall_us']
                target=joined['policies'][mode][arm]
                action=r['steps'][-1]['best_action']
                assert target['action']==action and target['WDL']==sign(scores[action])
                assert target['regret']==optimal-sign(scores[action])
                assert target['visits']==r['steps'][-1]['budget']
                assert abs(target['search_ms']-t/1000)<1e-12
                assert target['evaluator_calls']==initial
                if arm=='live_fixed16':
                    assert r['force_stop_visits']==16 and len(r['steps'])==2
                if arm=='adaptive':
                    for stage,dec in enumerate(r['adaptive_decisions']):
                        x=features(r,r['steps'][stage])
                        assert len(x)==len(dec['root_features'])==9
                        err=max(abs(a-b) for a,b in zip(x,dec['root_features']))
                        adaptive_pred_errors=max(adaptive_pred_errors,err)
                        assert err<3e-9
                        g=sum(a*b for a,b in zip(p['coefficients'][stage],x))
                        c=p['forecast_cost_ms'][stage]
                        expected=g>p['lambda_wdl_per_ms']*c
                        assert abs(g-dec['predicted_signed_gain'])<3e-9
                        assert dec['continue'] is expected
                if arm=='validation_clock':
                    assert r['time_budget_ms']==clockcap
                    clock_decisions+=1
                    for dec in r['time_budget_stop_decisions'][:-1]:
                        assert not dec['stop']
                    if r['stopped_at_visits']<64:
                        assert r['time_budget_stop_decisions'][-1]['stop']
                controls[arm]=r
            base=controls['full64']
            assert len(base['steps'])==4
            for arm in ('adaptive','live_fixed16','validation_clock'):
                other=controls[arm]
                assert base['root_hash']==other['root_hash']
                for a,b in zip(base['steps'],other['steps']):
                    assert a['best_action']==b['best_action']
                    assert a['evaluator_calls']==b['evaluator_calls']
                    assert a['visit_policy']==b['visit_policy']
    for mode in ('new_mlp','conv'):
        assert cache[(mode,False)]==1
        assert cache[(mode,True)]==1535
    summary={}
    for name in ('replica_A','replica_B','pooled'):
        subs=[r for r in pairs if name=='pooled' or r['cohort']==name]
        assert len(subs)==(384 if name=='pooled' else 192)
        modes={}
        for mode in MODES:
            rr=report['modes'][name][mode]
            blocks={}
            for arm in ARMS:
                s=[r['policies'][mode][arm] for r in subs]
                n=len(s)
                regret=sum(x['regret'] for x in s)/n
                cost=sum(x['search_ms'] for x in s)/n
                assert abs(regret-rr['arms'][arm]['mean_regret'])<1e-12
                assert abs(cost-rr['arms'][arm]['mean_search_ms'])<1e-12
                blocks[arm]={'mean_regret':regret,'mean_search_ms':cost}
            comp=rr['comparison']
            d16=blocks['adaptive']['mean_regret']-blocks['live_fixed16']['mean_regret']
            dc=blocks['adaptive']['mean_regret']-blocks['validation_clock']['mean_regret']
            assert abs(d16-comp['adaptive_minus_actual_live16_regret'])<1e-12
            assert abs(dc-comp['adaptive_minus_clock_regret'])<1e-12
            mismatch=abs(blocks['adaptive']['mean_search_ms']-blocks['validation_clock']['mean_search_ms'])/blocks['adaptive']['mean_search_ms']
            assert abs(mismatch-comp['mean_search_ms_cost_mismatch_fraction'])<1e-12
            for stage,r in enumerate(rr['calibration']):
                assert r['n_roots']==len(subs)
                values=[]
                for root in subs:
                    response=raw[([x['id'] for x in cohort['rows']].index(root['id'])*3+MODES.index(mode))*4]
                    src_scores=oracle[root['id']]['exact_action_scores']
                    a,b=response['steps'][stage:stage+2]
                    gain=sign(src_scores[b['best_action']])-sign(src_scores[a['best_action']])
                    values.append(gain)
                assert abs(sum(values)/len(values)-r['mean_realized_signed_gain'])<1e-12
            modes[mode]={'arms':blocks,
                         'adaptive_minus_clock':dc,
                         'adaptive_minus_live16':d16,
                         'matched_10pct':mismatch<=0.10,
                         'cost_mismatch':mismatch}
        summary[name]=modes
    # Fresh 24 original+24 mirrored exact external re-solves.
    ext=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    checked=0
    for root in truth['rows']:
        if root['id'].endswith('_R00'):
            scores=solver(ext/'c4solver',ext/'7x6_small.book',root['history'])
            mirror=''.join(str(8-int(ch)) for ch in root['history'])
            reverse=solver(ext/'c4solver',ext/'7x6_small.book',mirror)
            assert scores==root['exact_action_scores']
            assert reverse==scores[::-1]
            checked+=1
    assert checked==24
    result={
        'schema':'quartz_cog02_voc09_independent_release_newfamily_readback_v1',
        'source_commit':m['source_commit'],
        'actual_live_release_queries':4608,
        'distinct_new_roots':384,'disjoint_new_test_families':24,
        'independent_pascal_pons_origin_requeries':checked,
        'independent_pascal_pons_mirror_requeries':checked,
        'all_actual_live_fixed16_end_at_16':True,
        'all_4608_per_root_WDL_regret_and_evaluator_cost_recomputed':True,
        'all_full_to_early_stop_prefix_actions_policies_and_evals_identical':True,
        'frozen_policy_current_features_max_abs_error':adaptive_pred_errors,
        'model_cache_hits':{str(k):v for k,v in cache.items()},
        'summary':summary,
        'limitations':['CPU wall clock not hard realtime or process CPU',
                       'release cost forecasts frozen from debug underpredict/overpredict',
                       'new families from same synthetic generator; not IID population',
                       'one effective MCTS search trajectory seed17',
                       'true Bayesian KG not implemented'],
    }
    output=ev/'INDEPENDENT_READBACK.json'
    if output.exists():raise FileExistsError('refuse duplicate')
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('COG02_CAL09_INDEPENDENT_4608_RELEASE_AUDIT_PASS',json.dumps({
        'n':4608,'fresh_solver_queries':checked*2,
        'adaptive_vs_live16':{name:result['summary'][name]['uniform']['adaptive_minus_live16']
            for name in ('replica_A','replica_B','pooled')},
        'cost_match_uniform':{name:result['summary'][name]['uniform']['matched_10pct']
            for name in ('replica_A','replica_B','pooled')}
    },sort_keys=True))


if __name__=='__main__':main()
