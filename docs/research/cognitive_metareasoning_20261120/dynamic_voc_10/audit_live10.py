#!/usr/bin/env python3
"""Independent real 8064-release-MCTS, two-step commitment, WDL & CPU cost audit.

Recomputes actual actions and utility from frozen external exact oracle.
Checks physical call/wall accounting, forward-only stop decisions and model
content identity without importing experiment runner or fitted Python model.
"""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import math
import subprocess


MODES=('uniform','new_mlp','conv')
ARMS=('full64','dynamic','myopic','myopic_time_match',
      'legacy_debug_myopic','live_fixed16','validation_clock')


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def board_and_moves(h):
    b=[0]*42
    heights=[0]*7
    for i,k in enumerate(h):
        c=int(k)-1
        assert 0<=c<7 and heights[c]<6
        b[heights[c]*7+c]=1 if i%2==0 else -1
        heights[c]+=1
    return b,[c for c in range(7) if heights[c]<6]


def external(exe,book,h):
    result=subprocess.run([str(exe),'-a','-b',str(book)],
       input=(h+'\n').encode(),capture_output=True,timeout=12,check=True)
    v=result.stdout.decode().split()
    assert len(v)==8 and v[0]==h
    return [int(t) for t in v[1:]]


def observed_features(root,step):
    legal=root['legal_columns_0based']
    b=step['budget']
    dense=[0.0]*7
    seen=set()
    for c,p in step['visit_policy']:
        assert c in legal and c not in seen and math.isfinite(p) and p>=0
        seen.add(c);dense[c]=p
    z=sum(dense);assert abs(z-1)<1e-4
    dense=[x/z for x in dense]
    rank=sorted(dense,reverse=True)
    entropy=-sum(p*math.log(p) for p in dense if p>0)/math.log(7)
    return [1.,b/64.,entropy,rank[0],rank[0]-rank[1],
            len(legal)/7.,step['evaluator_calls']/(b+1),
            root['ply']/42.,1. if root['side_to_move']==1 else 0.]


def forecast(model,stage,x):
    lam=model['lambda_wdl_per_ms']
    def dot(a):return sum(w*y for w,y in zip(a,x))
    def cost(prefix):
        a=model[f'{prefix}_cost_log_coefficients'][stage]
        scale=model[f'{prefix}_cost_scale'][stage]
        return max(1e-6,math.exp(max(-20.,min(10.,dot(a))))*scale)
    short_gain=dot(model['one_step_gain_coefficients'][stage])
    short_cost=cost('one_step')
    netshort=short_gain-lam*short_cost
    joint_gain=None;joint_cost=None;netjoint=None
    if stage<2:
        joint_gain=dot(model['two_step_gain_coefficients'][stage])
        joint_cost=cost('two_step')
        netjoint=joint_gain-lam*joint_cost
    if netjoint is not None and netjoint>0 and netjoint>max(netshort,0):
        action='commit_two'
    elif netshort>0:
        action='single'
    else:
        action='stop'
    return {'action':action,
            'predicted_single_gain':short_gain,
            'predicted_joint_gain':joint_gain,
            'predicted_single_cost_ms':short_cost,
            'predicted_joint_cost_ms':joint_cost,
            'net_single':netshort,
            'net_joint':netjoint}


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/heldout_8064_release_20261121'
    meta=json.loads((e/'EXECUTION_MANIFEST.json').read_text())
    raw=json.loads((e/'RAW_8064_RELEASE_ROOTS.json').read_text())
    paired=json.loads((e/'PAIRED_384_NEW_ROOTS.json').read_text())
    s=json.loads((e/'SUMMARY_TWO_STEP.json').read_text())
    co=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    oraclejson=json.loads((d/'evidence/teacher_new384_20261121/EXACT_384_LABELS.json').read_text())
    modelsfile=d/'evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json'
    models=json.loads(modelsfile.read_text())
    assert meta['source_commit']=='83e02589c4b39236354636f3ec9de9bd6bfc6c1e'
    assert meta['n_actual_release_root_searches']==len(raw)==8064
    assert len(paired)==len(co['rows'])==len(oraclejson['rows'])==384
    assert sha(modelsfile)==meta['fitted_model_sha256']
    assert sha(d/'COHORT_384_FROZEN.json')==meta['new_test_cohort_sha256']
    assert sha(d/'evidence/teacher_new384_20261121/EXACT_384_LABELS.json')==meta['new_exact_external_oracle_sha256']
    for name,expected in meta['output_sha256'].items():
        assert sha(e/name)==expected,name
    exe=Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo')
    assert sha(exe)==meta['binary_sha256']
    teacher={r['id']:r for r in oraclejson['rows']}
    cache=Counter()
    ndec=Counter()
    informative=Counter()
    for i,(case,joined) in enumerate(zip(co['rows'],paired)):
        assert case['id']==joined['id'] and case['family']==joined['family']
        assert case['history']==joined['history'] and case['group']==joined['group']
        board,legal=board_and_moves(case['history'])
        assert board==case['board'] and legal==case['legal_columns']
        S=teacher[case['id']]['exact_action_scores']
        assert len(S)==7
        assert all((S[c]==-1000)==(c not in legal) for c in range(7))
        values={(S[c]>0)-(S[c]<0) for c in legal}
        if len(values)>1:informative[case['group']]+=1
        u=lambda val:(val>0)-(val<0)
        best=max(u(S[c]) for c in legal)
        for mi,mode in enumerate(MODES):
            full=raw[(i*3+mi)*7+0]
            assert full['stopped_at_visits']==64 and len(full['steps'])==4
            for ai,arm in enumerate(ARMS):
                row=raw[(i*3+mi)*7+ai]
                assert row['history']==case['history']
                assert row['board_bottom_up']==board
                assert row['side_to_move']==case['current_player']
                assert row['legal_columns_0based']==legal
                assert row['seed']==17
                assert row['root_hash']==full['root_hash']
                cache[(mode,str(row['model_cache_hit']))]+=1
                assert row['stopped_at_visits']==row['steps'][-1]['budget']
                assert len(row['steps']) in (1,2,3,4)
                assert row['fresh16']['available'] is (len(row['steps'])>=2)
                if len(row['steps'])>=2:
                    assert row['fresh16']['resident_match']
                last_calls=row['root_initial_evaluator_calls']
                micros=0
                for idx,step in enumerate(row['steps']):
                    assert step['budget']==step['root_visits']==(8,16,32,64)[idx]
                    assert step['best_action'] in legal and S[step['best_action']]!=-1000
                    assert step['root_hash']==row['root_hash']
                    assert step['evaluator_calls']>=last_calls
                    assert step['evaluator_calls']-last_calls==step['additional_evaluator_calls']
                    assert step['step_wall_us']>=0
                    last_calls=step['evaluator_calls']
                    micros+=step['step_wall_us']
                    root_prefix=full['steps'][idx]
                    assert root_prefix['best_action']==step['best_action']
                    assert root_prefix['evaluator_calls']==step['evaluator_calls']
                    assert root_prefix['visit_policy']==step['visit_policy']
                a=row['steps'][-1]['best_action']
                seen=joined['policies'][mode][arm]
                assert seen['action']==a and seen['WDL']==u(S[a])
                assert seen['regret']==best-u(S[a])
                assert seen['visits']==row['stopped_at_visits']
                assert abs(seen['search_ms']-micros/1000)<1e-12
                assert seen['evaluator_calls']==last_calls
                if arm=='live_fixed16':
                    assert row['force_stop_visits']==16 and len(row['steps'])==2
                if arm=='validation_clock':
                    assert row['time_budget_ms']==models['modes'][mode]['clock_validation_time_match']['cap_ms']
                if arm in ('dynamic','myopic','myopic_time_match'):
                    policy=models['modes'][mode]['frozen_policies'][arm]
                    assert policy['schema']=='quartz_cog02_dynamic_two_step_v1'
                    for dec in row['dynamic_decisions']:
                        j={8:0,16:1,32:2}[dec['at_visits']]
                        x=observed_features(row,row['steps'][j])
                        assert max(abs(a-b) for a,b in zip(x,dec['root_features']))<3e-9
                        p=forecast(policy,j,x)
                        assert p['action']==dec['action']
                        for key in ('predicted_single_gain','predicted_single_cost_ms','net_single'):
                            assert abs(p[key]-dec[key])<3e-9
                        for key in ('predicted_joint_gain','predicted_joint_cost_ms','net_joint'):
                            assert (p[key] is None and dec[key] is None) or abs(p[key]-dec[key])<3e-9
                        ndec[(mode,arm,dec['action'])]+=1
                    if arm!='dynamic':
                        assert not row['dynamic_skipped_checkpoints']
                    if arm=='dynamic' and row['dynamic_skipped_checkpoints']:
                        assert 'commit_two' in [x['action'] for x in row['dynamic_decisions']]
    assert len(raw)==8064 and len(paired)==384
    assert cache[('new_mlp','False')]==1 and cache[('conv','False')]==1
    assert cache[('new_mlp','True')]==384*7-1 and cache[('conv','True')]==384*7-1
    summary={}
    for group in ('fresh_A','fresh_B','pooled'):
        subset=[r for r in paired if group=='pooled' or r['group']==group]
        assert len(subset)==(384 if group=='pooled' else 192)
        summary[group]={}
        for mode in MODES:
            body=s['mode_results'][group][mode]
            arms={}
            for arm in ARMS:
                rows=[r['policies'][mode][arm] for r in subset]
                n=len(rows)
                reg=sum(z['regret'] for z in rows)/n
                cost=sum(z['search_ms'] for z in rows)/n
                assert abs(reg-body['arms'][arm]['mean_regret'])<1e-12
                assert abs(cost-body['arms'][arm]['mean_search_ms'])<1e-12
                arms[arm]={'mean_regret':reg,'mean_search_ms':cost}
            for other in ('myopic','myopic_time_match','legacy_debug_myopic','live_fixed16','validation_clock'):
                z=body['paired_comparisons']['dynamic_vs_'+other]
                assert abs(z['regret_difference_dynamic_minus_other']-
                          (arms['dynamic']['mean_regret']-arms[other]['mean_regret']))<1e-12
                actual_mismatch=abs(arms['dynamic']['mean_search_ms']-arms[other]['mean_search_ms'])/arms['dynamic']['mean_search_ms']
                assert abs(z['actual_mean_search_cost_mismatch_fraction']-actual_mismatch)<1e-12
                assert z['matched_cost_10pct']==('MATCHED_10_PERCENT' if actual_mismatch<=.1 else 'NOT_MATCHED_10_PERCENT')
            summary[group][mode]={
                'arms':arms,
                'dynamic_two_step_commits':body['arms']['dynamic']['two_step_committed_count'],
                'n_decision_informative_roots':body['informative_WDL_decision_root_count'],
                'clock_matched':body['paired_comparisons']['dynamic_vs_validation_clock']['matched_cost_10pct'],
                'dynamic_minus_release_myopic_regret':arms['dynamic']['mean_regret']-arms['myopic']['mean_regret']
            }
    assert informative['fresh_A']+informative['fresh_B']==s['mode_results']['pooled']['uniform']['informative_WDL_decision_root_count']
    # Additional 24 original and 24 mirrored independent external re-solves.
    ext=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    checked=0
    for row in oraclejson['rows']:
        if row['id'].endswith('_R00'):
            h=row['history']
            actual=external(ext/'c4solver',ext/'7x6_small.book',h)
            reverse=''.join(str(8-int(x)) for x in h)
            actual_mirror=external(ext/'c4solver',ext/'7x6_small.book',reverse)
            assert actual==row['exact_action_scores'] and actual_mirror==actual[::-1]
            checked+=1
    assert checked==24
    result={
        'schema':'quartz_cog02_dynamic10_8064_release_independent_audit_v1',
        'source_commit':meta['source_commit'],
        'actual_Rust_release_requests':len(raw),
        'new_roots':384,'new_family_clusters':24,
        'independent_external_solver_origin_requeries':checked,
        'independent_external_solver_mirror_requeries':checked,
        'all_actual_actions_WDL_costs_and_prefixes_recomputed':True,
        'actual_two_step_model_forecasts_and_skips_Rust_vs_independent_Python':True,
        'model_cache_hit_counts':{str(key):num for key,num in cache.items()},
        'decision_counts':{str(key):num for key,num in ndec.items()},
        'informative_WDL_roots_by_group':dict(informative),
        'cohort_model_results':summary,
        'correct_cost_match_admission':True,
        'theory':"empirical signed realized gain and finite XOR synergy, NOT true Bayesian preposterior KG",
        'strong_nonmyopic_benefit_claim':'NOT_ESTABLISHED_WITHOUT_MATCHED_MYOPIC_COST',
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('refuse duplicate audit')
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('C10_INDEPENDENT_8064_RELEASE_EXACT_AUDIT_PASS',json.dumps({
        'requests':8064,'external_requeries':48,
        'informative':dict(informative),
        'uniform_dynamic_regret':{g:summary[g]['uniform']['arms']['dynamic']['mean_regret'] for g in ('fresh_A','fresh_B','pooled')},
        'uniform_vs_myopic_diff_pooled':summary['pooled']['uniform']['dynamic_minus_release_myopic_regret'],
    },sort_keys=True))


if __name__=='__main__':main()
