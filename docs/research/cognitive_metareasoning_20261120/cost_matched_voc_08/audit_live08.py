#!/usr/bin/env python3
"""One independent full readback of 1728 live Rust CPU MCTS root requests.

Recompute all WDL outcomes, incremental evaluator calls, cache residency,
time-cap stopping and signed-gain adaptive predictions. Never refit a model.
"""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def board_from_columns(history):
    board=[0]*42
    heights=[0]*7
    for i,s in enumerate(history):
        c=int(s)-1
        assert 0<=c<7 and heights[c]<6
        board[heights[c]*7+c]=1 if i%2==0 else -1
        heights[c]+=1
    return board,[c for c in range(7) if heights[c]<6]


def from_current(root,step):
    legal=root['legal_columns_0based']
    policy=[0.0]*7;seen=set()
    for c,p in step['visit_policy']:
        assert c in legal and c not in seen and p>=0 and math.isfinite(p)
        seen.add(c)
        policy[c]=float(p)
    assert abs(sum(policy)-1)<1e-4
    z=sum(policy)
    policy=[p/z for p in policy]
    sortedp=sorted(policy,reverse=True)
    h=-sum(p*math.log(p) for p in policy if p>0)/math.log(7)
    budget=step['budget']
    return [1.0,budget/64,h,sortedp[0],sortedp[0]-sortedp[1],len(legal)/7,
            step['evaluator_calls']/(budget+1),root['ply']/42,
            1.0 if root['side_to_move']==1 else 0.0]


def solver(exe,book,h):
    result=subprocess.run([str(exe),'-a','-b',str(book)],
        input=(h+'\n').encode(),capture_output=True,timeout=12,check=True)
    values=result.stdout.decode().split()
    assert len(values)==8 and values[0]==h
    return [int(c) for c in values[1:]]


def bootstrap_12_by16(x,y):
    assert len(x)==len(y)==192
    groups=defaultdict(list)
    for a,b in zip(x,y):
        assert a['family']==b['family']
        groups[a['family']].append(a['regret']-b['regret'])
    assert len(groups)==12 and all(len(t)==16 for t in groups.values())
    means=[sum(t)/16 for _,t in sorted(groups.items())]
    rng=random.Random(20261121)
    replicates=sorted(sum(rng.choice(means) for _ in range(12))/12 for _ in range(3000))
    return {'mean_left_minus_right':sum(means)/12,
            'conditional_family_bootstrap_95':[replicates[74],replicates[2924]],
            'n_families':12}


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/live_test_1728_20261121'
    man=json.loads((e/'execution_manifest.json').read_text())
    raw=json.loads((e/'raw_1728_rust_requests.json').read_text())
    joined=json.loads((e/'paired_192_livetest.json').read_text())
    summary=json.loads((e/'summary.json').read_text())
    trained=json.loads((d/'evidence/frozen_surrogate_20261121/frozen_surrogates.json').read_text())
    caps=json.loads((d/'evidence/validation_time_caps_20261121/TIME_CAPS_FROZEN.json').read_text())
    labels=json.loads((d.parent/'connect4_generalization_07/evidence/teacher_1024_20261121/labels.json').read_text())
    cohort=json.loads((d.parent/'connect4_generalization_07/COHORT_1024_FROZEN.json').read_text())
    if man['source_head']!='8682f805cb8c74deef967e3ed9595bc41bea474b':
        raise AssertionError('live experiment source revision mismatch')
    for p,expected in man['output_sha256'].items():
        assert sha(e/p)==expected
    assert sha(d/'evidence/frozen_surrogate_20261121/frozen_surrogates.json')==man['frozen_surrogate_sha256']
    assert sha(d/'evidence/validation_time_caps_20261121/TIME_CAPS_FROZEN.json')==man['frozen_time_caps_sha256']
    assert len(raw)==1728 and len(joined)==192 and summary['n_rust_requests']==1728
    tests=[r for r in cohort['rows'] if r['split']=='test']
    labelmap={r['id']:r for r in labels['rows'] if r['split']=='test' and r['status']=='OK'}
    assert len(labelmap)==len(tests)==192
    score=lambda v:(v>0)-(v<0)
    counts=Counter()
    bymode=defaultdict(list)
    byname={'uniform':'uniform_no_neural_network',
            'new_mlp':'trained_tiny_mlp_cpu','conv':'trained_conv3x3_cpu'}
    for i,(case,joined_case) in enumerate(zip(tests,joined)):
        assert joined_case['id']==case['id']
        assert joined_case['history']==case['history']
        board,legal=board_from_columns(case['history'])
        assert board==case['board'] and legal==case['legal_columns']
        correct=labelmap[case['id']]['exact_action_scores']
        assert len(correct)==7
        target=max(score(correct[c]) for c in legal)
        for mi,mode in enumerate(('uniform','new_mlp','conv')):
            full=raw[i*9+mi*3]
            adaptive=raw[i*9+mi*3+1]
            clock=raw[i*9+mi*3+2]
            policy=trained['mode_policies'][mode]
            for kind,row in [('full',full),('adaptive',adaptive),('fixed_clock',clock)]:
                assert row['history']==case['history'] and row['eval_mode']==byname[mode]
                assert row['board_bottom_up']==board and row['legal_columns_0based']==legal
                assert row['seed']==17 and row['root_hash']==full['root_hash']
                assert row['stopped_at_visits'] in (8,16,32,64)
                assert row['stopped_at_visits']==row['steps'][-1]['budget']
                assert row['fresh16']['resident_match'] if row['stopped_at_visits']>=16 else row['fresh16']['available'] is False
                prior=row['root_initial_evaluator_calls']
                spent=0
                for j,s in enumerate(row['steps']):
                    assert s['root_visits']==s['budget']==[8,16,32,64][j]
                    assert s['best_action'] in legal
                    assert s['root_hash']==row['root_hash']
                    assert s['evaluator_calls']>=prior
                    assert s['evaluator_calls']-prior==s['additional_evaluator_calls']
                    prior=s['evaluator_calls']
                    spent+=s['step_wall_us']
                    if kind!='full':
                        original=full['steps'][j]
                        assert original['best_action']==s['best_action']
                        assert original['evaluator_calls']==s['evaluator_calls']
                        assert original['visit_policy']==s['visit_policy']
                if kind=='fixed_clock':
                    target_us=caps['modes'][mode]['selected_cap_ms']*1000
                    if row['stopped_at_visits']<64:
                        assert spent>=target_us
                    for step in row['time_budget_stop_decisions'][:-1]:
                        assert step['stop'] is False
                    if row['time_budget_stop_decisions'] and row['stopped_at_visits']<64:
                        assert row['time_budget_stop_decisions'][-1]['stop'] is True
                if kind=='adaptive':
                    for j,decision in enumerate(row['adaptive_decisions']):
                        s=row['steps'][j]
                        assert decision['stage_visits']==s['budget']
                        x=from_current(row,s)
                        assert max(abs(a-b) for a,b in zip(x,decision['root_features']))<3e-9
                        pred=sum(a*b for a,b in zip(policy['coefficients'][j],x))
                        assert abs(pred-decision['predicted_signed_gain'])<3e-9
                        should_continue=pred>policy['lambda_wdl_per_ms']*policy['forecast_cost_ms'][j]
                        assert should_continue is decision['continue']
                        counts[(mode,'adaptive_continue' if should_continue else 'adaptive_stop')]+=1
                last=row['steps'][-1]
                u=score(correct[last['best_action']])
                actual=joined_case['modes'][mode][kind]
                assert actual['WDL']==u and actual['regret']==target-u
                assert actual['evaluator_calls']==prior
                assert abs(actual['search_ms']-spent/1000)<1e-9
                assert actual['visits']==last['budget']
                counts[(mode,kind)]+=1
            bymode[mode].append({
                'family':case['family'],
                'regret_adaptive':joined_case['modes'][mode]['adaptive']['regret'],
                'regret_clock':joined_case['modes'][mode]['fixed_clock']['regret'],
                'regret_fixed16':target-score(correct[full['steps'][1]['best_action']]),
                'time_adaptive':joined_case['modes'][mode]['adaptive']['search_ms'],
                'time_clock':joined_case['modes'][mode]['fixed_clock']['search_ms'],
                'time_fixed16':sum(x['step_wall_us'] for x in full['steps'][:2])/1000,
            })
    for mode in ('uniform','new_mlp','conv'):
        assert counts[(mode,'full')]==counts[(mode,'adaptive')]==counts[(mode,'fixed_clock')]==192
    # New independent external source proof on one root in each test family.
    ext=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    sampled=[row for row in tests if row['id'].endswith('_R00')]
    assert len(sampled)==12
    for row in sampled:
        expected=labelmap[row['id']]['exact_action_scores']
        actual=solver(ext/'c4solver',ext/'7x6_small.book',row['history'])
        mirrored=''.join(str(8-int(ch)) for ch in row['history'])
        mirrored_values=solver(ext/'c4solver',ext/'7x6_small.book',mirrored)
        assert actual==expected and mirrored_values==expected[::-1]
    bymode_summary={}
    for mode,results in bymode.items():
        n=len(results)
        dclock=[r['regret_adaptive']-r['regret_clock'] for r in results]
        groups=defaultdict(list)
        for r in results:
            groups[r['family']].append(r['regret_adaptive']-r['regret_fixed16'])
        familymeans=[sum(s)/len(s) for _,s in sorted(groups.items())]
        assert len(familymeans)==12
        rng=random.Random(20261121)
        boot=sorted(sum(rng.choice(familymeans) for _ in range(12))/12 for _ in range(3000))
        report=summary['comparisons'][mode]['adaptive_minus_fixed_clock']
        assert abs(sum(dclock)/n-report['mean_regret_difference'])<1e-12
        diff_time=abs(sum(r['time_adaptive'] for r in results)/n-
            sum(r['time_clock'] for r in results)/n)/(sum(r['time_adaptive'] for r in results)/n)
        assert abs(diff_time-report['relative_search_cost_mismatch_fraction'])<1e-12
        bymode_summary[mode]={
            'adaptive_minus_clock_regret':sum(dclock)/n,
            'relative_actual_search_ms_mismatch':diff_time,
            'adaptive_minus_fixed16_mean_regret':sum(familymeans)/12,
            'adaptive_minus_fixed16_family_bootstrap_95':[boot[74],boot[2924]],
            'validated_frozen_mode_lambda':trained['mode_policies'][mode]['lambda_wdl_per_ms'],
            'adaptive_selected_stops':dict(Counter(str(row['modes'][mode]['adaptive']['visits']) for row in joined)),
        }
    independent={
        'schema':'quartz_cog02_live_1728_independent_readback_v1',
        'source_commit':man['source_head'],
        'full_rust_server_calls':1728,
        'unique_test_roots':192,'test_family_clusters':12,
        'independent_external_solver_requeries':12,
        'independent_mirrored_solver_requeries':12,
        'model_source_byte_hashes_verified':len(man['model_sha256']),
        'all_saved_results_bytes_SHA256_verified':len(man['output_sha256']),
        'all_action_WDL_costs_and_live_decisions_recomputed':True,
        'observed_cache_liveness':summary['model_cache_residency'],
        'mode_summaries':bymode_summary,
        'status':'implementation verified; finite 12-group conditional external-WDL observation',
        'not_claimed':['true Bayesian preposterior KG',
                       'per-root time matching', 'one global representative population',
                       'unseen research seed/generalization','optimized Rust release build'],
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise RuntimeError('refuse overwrite independent live validation')
    p.write_text(json.dumps(independent,indent=2,sort_keys=True)+'\n')
    print('COG02_REAL_LIVE_1728_INDEPENDENT_AUDIT_PASS',json.dumps(
        bymode_summary,sort_keys=True))


if __name__=='__main__':
    main()
