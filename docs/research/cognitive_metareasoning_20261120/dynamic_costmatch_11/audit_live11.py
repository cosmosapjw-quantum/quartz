#!/usr/bin/env python3
"""Independent readback of 10368 actual C11 release Rust searches and exact solver labels.

No imports from C11 runner, calibration, scoring or trained inference.
This audit recomputes locked blocked-random assignments, online stagewise
gain/cost decisions, root-action WDL regret, eval/cpu call counts, prefix
identity, fixed CPU match gates and 24+24 new external oracle mirror queries.
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
ARMS=('full64','dynamic','myopic_low','myopic_high',
      'mixture_1','mixture_2','mixture_3','validation_clock','live_fixed16')
BUDGETS=(8,16,32,64)


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sign(x):
    if type(x) is not int or x==-1000 or not -21<=x<=21:
        raise ValueError('invalid exact signed WDL column score')
    return (x>0)-(x<0)


def replay(history):
    board=[0]*42
    h=[0]*7
    for i,ch in enumerate(history):
        c=int(ch)-1
        if not 0<=c<7 or h[c]>=6:raise ValueError('non-gravity source')
        board[h[c]*7+c]=1 if i%2==0 else -1
        h[c]+=1
    return board,[c for c in range(7) if h[c]<6]


def independent_hash(seed,identity):
    source=f'quartz_cog02_costmatch11_v1|{seed}|{identity}'.encode()
    return int.from_bytes(hashlib.sha256(source).digest()[:8],'big')/2**64


def reproduce_source_assignments(cohort,plan):
    cases=cohort['rows']
    group=defaultdict(lambda:defaultdict(list))
    for r in cases:
        group[r['group']][r['family']].append(r)
    assert len(group)==2 and all(len(family)==12 for family in group.values())
    assert all(len(roots)==16 for blocks in group.values() for roots in blocks.values())
    assignments={}
    for seed in plan['assignment_seeds']:
        chosen={}
        for mode,cfg in plan['mode_mixtures'].items():
            q={}
            p=cfg['higher_cpu_probability']
            base=math.floor(16*p)
            for g,families in group.items():
                ordered=sorted(families,
                    key=lambda family:independent_hash(seed,f'family|{mode}|{g}|{family}'))
                more=round(16*len(families)*p)-len(families)*base
                assert 0<=more<=len(families)
                overflow=set(ordered[:more])
                for family,roots in families.items():
                    sorted_roots=sorted(roots,
                        key=lambda r:independent_hash(seed,f'root|{mode}|{r["id"]}'))
                    hi={r['id'] for r in sorted_roots[:base+int(family in overflow)]}
                    for r in roots:
                        q[r['id']]='higher' if r['id'] in hi else 'lower'
            chosen[mode]=q
        assignments[str(seed)]=chosen
    return assignments


def current_x(root,stage):
    row=root['steps'][stage]
    b=row['budget'];assert b in (8,16,32)
    legal=root['legal_columns_0based']
    dense=[0.0]*7;seen=set()
    for a,p in row['visit_policy']:
        assert a in legal and a not in seen and p>=0 and math.isfinite(p)
        dense[a]=float(p);seen.add(a)
    z=sum(dense);assert abs(z-1)<1e-4
    dense=[x/z for x in dense]
    sortedp=sorted(dense,reverse=True)
    h=-sum(x*math.log(x) for x in dense if x>0)/math.log(7)
    return [1.,b/64,h,sortedp[0],sortedp[0]-sortedp[1],
        len(legal)/7,row['evaluator_calls']/(b+1),
        root['ply']/42,1. if root['side_to_move']==1 else 0.]


def forecast(model,stage,x):
    lam=model['lambda_wdl_per_ms']
    dot=lambda a:sum(v*w for v,w in zip(a,x))
    sg=dot(model['one_step_gain_coefficients'][stage])
    sc=max(1e-6,math.exp(max(-20.,min(10.,dot(model['one_step_cost_log_coefficients'][stage]))))*
           model['one_step_cost_scale'][stage])
    ns=sg-lam*sc
    jg=jc=nj=None
    if stage<2:
        jg=dot(model['two_step_gain_coefficients'][stage])
        jc=max(1e-6,math.exp(max(-20.,min(10.,dot(model['two_step_cost_log_coefficients'][stage]))))*
               model['two_step_cost_scale'][stage])
        nj=jg-lam*jc
    action='commit_two' if nj is not None and nj>0 and nj>max(ns,0) else ('single' if ns>0 else 'stop')
    return {'action':action,'predicted_single_gain':sg,
        'predicted_single_cost_ms':sc,'net_single':ns,
        'predicted_joint_gain':jg,'predicted_joint_cost_ms':jc,'net_joint':nj}


def external(exe,book,h):
    run=subprocess.run([str(exe),'-a','-b',str(book)],input=(h+'\n').encode(),
                       capture_output=True,timeout=12,check=True)
    a=run.stdout.decode().split()
    assert len(a)==8 and a[0]==h
    return [int(v) for v in a[1:]]


def bootstrap_paired(subset,mode,other,seed=20261129):
    fam=defaultdict(list)
    for row in subset:
        lhs=row['policies'][mode]['dynamic']['regret']
        rhs=row['policies'][mode][other]['regret']
        fam[row['family']].append(lhs-rhs)
    assert all(len(a)==16 for a in fam.values())
    means=[sum(a)/len(a) for _,a in sorted(fam.items())]
    n=len(means)
    rng=random.Random(seed)
    vals=sorted(sum(rng.choice(means) for _ in means)/n for _ in range(3000))
    return {'mean_delta':sum(means)/n,'CI95':[vals[74],vals[2924]],
            'n_distinct_families':n}


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/live_heldout_20261121'
    man=json.loads((e/'EXECUTION_MANIFEST.json').read_text())
    raw=json.loads((e/'RAW_10368_RELEASE_ROOTS.json').read_text())
    paired=json.loads((e/'PAIRED_384_NEW_POSITIONS.json').read_text())
    summary=json.loads((e/'SUMMARY_C11_COST_MATCH.json').read_text())
    co=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    origmod=d.parent/'dynamic_voc_10/evidence/frozen_release_policies_20261121/FROZEN_RELEASE_POLICIES.json'
    fitted=json.loads(origmod.read_text())
    mix=json.loads((d/'MIXTURE_PLAN_FROZEN.json').read_text())
    blocked=json.loads((d/'ASSIGNMENTS_FROZEN.json').read_text())
    lab=json.loads((d/'evidence/exact_new384_20261121/EXACT_NEW384_LABELS.json').read_text())
    src={r['id']:r for r in co['rows']}
    oracle={r['id']:r for r in lab['rows']}
    assert man['source_commit']=='348ce4e0ad311ea0c14cd512b1ad15d2ca19b95c'
    assert len(raw)==man['n_real_Rust_searches']==summary['n_real_Rust_inprocess_release_search_requests']==10368
    assert len(paired)==len(src)==len(oracle)==384
    assert lab['status']['all_ok']
    assert sha(origmod)==man['frozen_model_sha256']
    assert sha(d/'MIXTURE_PLAN_FROZEN.json')==man['validation_mix_sha256']
    assert sha(d/'ASSIGNMENTS_FROZEN.json')==man['assignment_sha256']
    assert sha(d/'COHORT_384_FROZEN.json')==man['new_cohort_sha256']
    assert sha(d/'evidence/exact_new384_20261121/EXACT_NEW384_LABELS.json')==man['exact_oracle_sha256']
    assert sha(Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo'))==man['compiled_release_binary_sha256']
    for name,expected in man['output_sha256'].items():
        assert sha(e/name)==expected,name
    assert reproduce_source_assignments(co,mix)==blocked['seed_assignments']
    cache=Counter()
    choices=Counter()
    regdiff=defaultdict(Counter)
    assignment=blocked['seed_assignments']
    seeds=mix['assignment_seeds']
    for i,(case,row) in enumerate(zip(co['rows'],paired)):
        assert case['id']==row['id'] and row['history']==case['history']
        assert case['family']==row['family'] and case['group']==row['group']
        b,legal=replay(case['history'])
        assert b==case['board'] and legal==case['legal_columns']
        S=oracle[case['id']]['exact_action_scores']
        assert all((S[col]==-1000)==(col not in legal) for col in range(7))
        ideal=max(sign(S[col]) for col in legal)
        for mi,mode in enumerate(MODES):
            full=raw[(i*3+mi)*9]
            assert len(full['steps'])==4 and full['stopped_at_visits']==64
            source_model=fitted['modes'][mode]['frozen_policies']['myopic']
            for ai,arm in enumerate(ARMS):
                search=raw[(i*3+mi)*9+ai]
                score=row['policies'][mode][arm]
                assert search['history']==case['history'] and search['root_hash']==full['root_hash']
                assert search['board_bottom_up']==b and search['legal_columns_0based']==legal
                assert search['side_to_move']==case['current_player'] and search['seed']==17
                assert search['stopped_at_visits']==search['steps'][-1]['budget']
                assert len(search['steps']) in (1,2,3,4)
                cache[(mode,bool(search['model_cache_hit']))]+=1
                pre=search['root_initial_evaluator_calls']
                total_ms=0.0
                for k,s in enumerate(search['steps']):
                    assert s['budget']==s['root_visits']==BUDGETS[k]
                    assert s['best_action'] in legal and s['root_hash']==search['root_hash']
                    assert s['evaluator_calls']>=pre
                    assert s['additional_evaluator_calls']==s['evaluator_calls']-pre
                    assert s['step_wall_us']>=0
                    pre=s['evaluator_calls']
                    total_ms+=s['step_wall_us']/1000
                    assert full['steps'][k]['best_action']==s['best_action']
                    assert full['steps'][k]['visit_policy']==s['visit_policy']
                    assert full['steps'][k]['evaluator_calls']==s['evaluator_calls']
                if len(search['steps'])<2:
                    assert search['fresh16']['available'] is False
                else:
                    assert search['fresh16']['resident_match'] is True
                action=search['steps'][-1]['best_action']
                assert score['action']==action and score['WDL']==sign(S[action])
                assert score['regret']==ideal-sign(S[action])
                assert score['evaluator_calls']==pre
                assert score['visits']==search['stopped_at_visits']
                assert abs(score['search_ms']-total_ms)<1e-10
                if arm=='live_fixed16':
                    assert search['force_stop_visits']==16 and len(search['steps'])==2
                if arm=='validation_clock':
                    assert search['time_budget_ms']==fitted['modes'][mode]['clock_validation_time_match']['cap_ms']
                if arm=='dynamic':
                    model=fitted['modes'][mode]['frozen_policies']['dynamic']
                elif arm in ('myopic_low','myopic_high') or arm.startswith('mixture_'):
                    if arm in ('myopic_low','myopic_high'):
                        source=('lower' if arm=='myopic_low' else 'higher')
                    else:
                        idx=int(arm[-1])-1
                        source=assignment[str(seeds[idx])][mode][case['id']]
                    col=('lower_cpu_lambda' if source=='lower' else 'higher_cpu_lambda')
                    lam=mix['mode_mixtures'][mode][col]
                    assert all(all(v==0.0 for v in z) for z in source_model['two_step_gain_coefficients'])
                    model={**source_model,'lambda_wdl_per_ms':lam}
                else:
                    continue
                if arm.startswith('mixture_'):
                    choices[(mode,arm,source)]+=1
                skip=[]
                for dec in search['dynamic_decisions']:
                    stage=BUDGETS.index(dec['at_visits'])
                    assert stage<3
                    x=current_x(search,stage)
                    assert max(abs(v-w) for v,w in zip(x,dec['root_features']))<3e-9
                    p=forecast(model,stage,x)
                    assert p['action']==dec['action']
                    for name in ('predicted_single_gain','predicted_single_cost_ms','net_single'):
                        assert abs(p[name]-dec[name])<3e-9
                    for name in ('predicted_joint_gain','predicted_joint_cost_ms','net_joint'):
                        if p[name] is None:assert dec[name] is None
                        else:assert abs(p[name]-dec[name])<3e-9
                    if dec['action']=='commit_two':
                        skip.append(BUDGETS[stage+1])
                assert skip==search['dynamic_skipped_checkpoints']
                if arm!='dynamic':assert not skip
            for arm in ('mixture_1','mixture_2','mixture_3'):
                if row['policies'][mode]['dynamic']['regret']!=row['policies'][mode][arm]['regret']:
                    regdiff[(mode,arm)]['different_WDL_regret_roots']+=1
    for mode in ('new_mlp','conv'):
        assert cache[(mode,False)]==1 and cache[(mode,True)]==384*len(ARMS)-1
    for seed_index in range(3):
        arm=f'mixture_{seed_index+1}'
        for mode in MODES:
            assert choices[(mode,arm,'higher')]==(228 if mode=='uniform' else 0)
            assert choices[(mode,arm,'lower')]==(156 if mode=='uniform' else 384)
    groups={}
    for group in ('new_A','new_B','pooled'):
        positions=[r for r in paired if group=='pooled' or r['group']==group]
        assert len(positions)==(384 if group=='pooled' else 192)
        groups[group]={}
        for mode in MODES:
            out=summary['comparisons'][group][mode]
            recomputed={}
            for arm in ARMS:
                records=[row['policies'][mode][arm] for row in positions]
                n=len(records)
                meanreg=sum(r['regret'] for r in records)/n
                meantime=sum(r['search_ms'] for r in records)/n
                assert abs(meanreg-out['arms'][arm]['mean_regret'])<1e-12
                assert abs(meantime-out['arms'][arm]['mean_actual_search_ms'])<1e-12
                recomputed[arm]={'regret':meanreg,'ms':meantime}
            comparison={}
            for arm in ('mixture_1','mixture_2','mixture_3','validation_clock'):
                left=recomputed['dynamic']
                right=recomputed[arm]
                mean_delta=left['regret']-right['regret']
                mismatch=abs(left['ms']-right['ms'])/left['ms']
                source=out['paired']['dynamic_minus_'+arm]
                assert abs(mean_delta-source['regret_difference'])<1e-12
                assert abs(mismatch-source['relative_actual_search_ms_mismatch'])<1e-12
                actual='MATCHED_10_PERCENT' if mismatch<=0.10 else 'NOT_MATCHED_10_PERCENT'
                assert source['mean_CPU_time_match']==actual
                infer=bootstrap_paired(positions,mode,arm)
                saved=source['family_cluster_conditional_bootstrap']
                assert abs(infer['mean_delta']-saved['mean_regret_difference'])<1e-12
                assert all(abs(a-b)<1e-12 for a,b in zip(infer['CI95'],saved['conditional_family_bootstrap_95']))
                comparison[arm]={'mean_regret_difference':mean_delta,'cost_mismatch':mismatch,
                                 'matched_10pct':actual,'CI95':infer['CI95']}
            cfg=mix['mode_mixtures'][mode]
            p=cfg['higher_cpu_probability']
            Ereg=sum((1-p)*row['policies'][mode]['myopic_low']['regret']+
                     p*row['policies'][mode]['myopic_high']['regret'] for row in positions)/len(positions)
            assert abs(Ereg-out['expected_randomization']['ex_ante_randomization_expected_regret_conditional_on_frozen_sample'])<1e-12
            groups[group][mode]={'arms':recomputed,'contrasts':comparison,
                                 'conditional_expected_random_mixture_regret':Ereg}
    ext=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    count=0
    for x in lab['rows']:
        if x['id'].endswith('_R00'):
            score=external(ext/'c4solver',ext/'7x6_small.book',x['history'])
            refl=''.join(str(8-int(c)) for c in x['history'])
            mirrored=external(ext/'c4solver',ext/'7x6_small.book',refl)
            assert score==x['exact_action_scores']
            assert mirrored==score[::-1]
            count+=1
    assert count==24
    audit={
        'schema':'quartz_cog02_costmatch11_10368_full_independent_readback_v1',
        'source_git_commit':man['source_commit'],
        'actual_release_Rust_requests':len(raw),
        'new_root_count':384,'opening_family_clusters':24,
        'additional_external_solver_original_requeries':count,
        'additional_external_solver_horizontal_mirror_requeries':count,
        'all_actual_game_states_and_WDL_regret_recomputed':True,
        'all_online_dynamic_and_myopic_features_and_decisions_independently_recomputed':True,
        'all_actual_search_cpu_times_and_evaluator_calls_recomputed':True,
        'full64_deterministic_root_prefix_action_visit_and_evals_identical':True,
        'blocked_assignment_independently_rederived_without_test_labels':True,
        'cache_hit_counts':{str(k):v for k,v in cache.items()},
        'assignment_counts':{str(k):v for k,v in choices.items()},
        'root_WDL_regret_difference_counts':{str(k):dict(v) for k,v in regdiff.items()},
        'cohorts':groups,
        'no_claim_of_independent_two_step_advantage_if_CI_touches_zero':True,
        'no_seed_increases_game_sample_size':True,
        'science_status':'finite exact oracle and implementation verified, metareasoning novelty unresolved',
    }
    destination=e/'INDEPENDENT_READBACK.json'
    if destination.exists():raise FileExistsError('refuse overwrite independent audit')
    destination.write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
    print('C11_INDEPENDENT_REAL_10368_RELEASE_AND_EXACT_AUDIT_PASS',json.dumps({
        'actual_MCTS':len(raw),'solver_original_mirror_requeries':count*2,
        'uniform_pooled_dynamic_regret':groups['pooled']['uniform']['arms']['dynamic']['regret'],
        'uniform_pooled_mix_deltas':{a:groups['pooled']['uniform']['contrasts'][a] for a in ('mixture_1','mixture_2','mixture_3')}
    },sort_keys=True))

if __name__=='__main__':
    main()
