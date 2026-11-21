#!/usr/bin/env python3
"""Independent audit of 9216 real CPU root-restart searches, live one-ply triggers
and exact external Connect4 WDL outcomes, without importing production code.

One bounded readback; mirrored solver original+reflection 24+24 additional
queries. Never re-fit prior networks or select posthoc trigger thresholds.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math,random,subprocess

MODES=('uniform','new_mlp','conv')
ARMS=('closed_sham','evidence_reopen','random_1','random_2','random_3',
      'always_full_reopen','plain_full','closed_no_probe')
SEEDS=(202611210201,202611210202,202611210203)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def sign(s):
    if type(s)!=int or s==-1000 or not -21<=s<=21:raise ValueError('invalid exact WDL score')
    return (s>0)-(s<0)

def replay(h):
    b=[0]*42;heights=[0]*7
    for i,v in enumerate(h):
        c=int(v)-1
        assert 0<=c<7 and heights[c]<6
        b[heights[c]*7+c]=1 if i%2==0 else -1
        heights[c]+=1
    return b,heights

def is_winning_next(b,heights,player,col):
    if col<0 or col>=7 or heights[col]>=6:return False
    y=heights[col]
    tmp=b.copy()
    tmp[y*7+col]=player
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        n=1
        for direction in (-1,1):
            x=col+dx*direction
            z=y+dy*direction
            while 0<=x<7 and 0<=z<6 and tmp[z*7+x]==player:
                n+=1;x+=dx*direction;z+=dy*direction
        if n>=4:return True
    return False

def vulnerability(b,heights,player,col):
    if is_winning_next(b,heights,player,col):return False
    after=b.copy();h=heights.copy()
    after[h[col]*7+col]=player;h[col]+=1
    return any(is_winning_next(after,h,-player,c) for c in range(7) if h[c]<6)

def tactical_from_source(root,initial,ranked,firstbest,firstpi):
    b,heights=replay(root['history'])
    player=root['current_player']
    if firstbest not in initial:firstbest=initial[0] # plain-full control
    omitted=[c for c in ranked if c not in initial]
    wins=[c for c in omitted if is_winning_next(b,heights,player,c)]
    safe=[c for c in omitted if c not in wins and not vulnerability(b,heights,player,c)]
    vuln=vulnerability(b,heights,player,firstbest)
    fixation=max(float(x[1]) for x in firstpi)
    if fixation<.5:chosen=[];reason='fixation_threshold_not_reached'
    elif wins:chosen=wins[:2];reason='excluded_immediate_win'
    elif vuln and safe:chosen=safe[:2];reason='leader_allows_immediate_opponent_win'
    else:chosen=[];reason='no_tactical_reopening_evidence'
    return {'suggested':chosen,'reason':reason,
            'immediate_wins':wins,'safe':safe,'vulnerable':vuln,
            'fixation_share':fixation,'fixation_gate':fixation>=.5,
            'trigger':bool(chosen)}

def hashed_random_omissions(omitted,n,seed,history):
    def key(col):
        h=(0xcbf29ce484222325 ^ seed)%(2**64)
        for b in history.encode()+bytes([col]):
            h^=b
            h=(h*0x1000000001b3)%(2**64)
        return h
    return sorted(omitted,key=key)[:n]

def solver(exe,book,h):
    p=subprocess.run([str(exe),'-a','-b',str(book)],
        input=(h+'\n').encode(),capture_output=True,check=True,timeout=12)
    x=p.stdout.decode().split()
    assert len(x)==8 and x[0]==h
    return [int(v) for v in x[1:]]

def paired_family_bootstrap(subset,mode,other):
    families=defaultdict(list)
    for row in subset:
        pair=row['modes'][mode]
        families[row['family']].append(pair['evidence_reopen']['regret']-
                                       pair[other]['regret'])
    assert all(len(x)==16 for x in families.values())
    n=len(families)
    means=[sum(x)/len(x) for _,x in sorted(families.items())]
    rng=random.Random(20261121)
    samples=sorted(sum(rng.choice(means) for _ in means)/n for _ in range(2500))
    return (sum(means)/n,[samples[62],samples[2437]])

def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/live_9216_20261121'
    receipt=json.loads((e/'EXECUTION_MANIFEST.json').read_text())
    raw=json.loads((e/'RAW_9216_RELEASE_ROOT_RESTARTS.json').read_text())
    paired=json.loads((e/'PAIRED_384_WDL_ATTENTION.json').read_text())
    report=json.loads((e/'SUMMARY_REOPEN12.json').read_text())
    cohort=json.loads((d/'COHORT_384_FROZEN.json').read_text())
    exact=json.loads((d/'evidence/new_384_exact_20261121/EXACT_NEW384_LABELS.json').read_text())
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    assert receipt['source_commit']=='f9b7caa45a420f09f1ed7665c827b8e6a6c8ea6e'
    assert receipt['real_Rust_root_restart_requests']==len(raw)==9216
    assert receipt['real_MCTS_searches']==18432
    assert len(paired)==len(cohort['rows'])==len(exact['rows'])==384
    assert exact['status']['statuses']=={'OK':384}
    assert sha(d/'COHORT_384_FROZEN.json')==receipt['new384_data_sha256']
    assert sha(d/'evidence/new_384_exact_20261121/EXACT_NEW384_LABELS.json')==receipt['new384_external_labels_sha256']
    assert sha(Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/release/mcts_demo'))==receipt['compiled_binary_sha256']
    for name,h in receipt['file_sha256'].items():
        assert sha(e/name)==h,name
    oracle={r['id']:r for r in exact['rows']}
    source={r['id']:r for r in cohort['rows']}
    cache=Counter()
    exact_changed=Counter()
    summary={}
    for i,root in enumerate(cohort['rows']):
        position=paired[i]
        assert position['id']==root['id'] and position['group']==root['group']
        board,heights=replay(root['history'])
        assert board==root['board'] and root['legal_columns']==[c for c in range(7) if heights[c]<6]
        S=oracle[root['id']]['exact_action_scores']
        legal=root['legal_columns']
        assert len(S)==7 and all((S[c]==-1000)==(c not in legal) for c in range(7))
        max_exact=max(sign(S[c]) for c in legal)
        for mi,mode in enumerate(MODES):
            first=raw[(i*3+mi)*8]
            initial=first['initial_candidates']
            prior=first['unmasked_evaluator_prior_rank']
            assert len(prior)==len(legal) and [x[0] for x in prior[:len(initial)]]==initial
            assert sorted(x[0] for x in prior)==sorted(legal)
            # independent original neural-or-uniform action ranking audit;
            # score ties prefer center and then ascending column.
            expected=sorted(prior,key=lambda x:(-float(x[1]),abs(int(x[0])-3),int(x[0])))
            assert expected==prior
            ordered=[c for c,_ in prior]
            omitted=[c for c in ordered if c not in initial]
            decisive=[c for c in legal if c not in initial and sign(S[c])>
                      max(sign(S[a]) for a in initial)]
            for ai,arm in enumerate(ARMS):
                r=raw[(i*3+mi)*8+ai]
                score=position['modes'][mode][arm]
                assert r['history']==root['history'] and r['side_to_move']==root['current_player']
                assert r['seed']==17 and r['legal_columns_0based']==legal
                assert r['board_bottom_up']==board
                assert r['first_search']['root_visits']==8
                assert r['second_search']['root_visits']==16
                cache[(mode,bool(r['model_cache_hit']))]+=1
                if arm.startswith('random_'):
                    assert r['arm']=='random_reopen'
                    assert r['reopen_seed']==SEEDS[int(arm[-1])-1]
                else:
                    assert r['arm']==arm
                assert r['first_search']['best_action'] in (legal if arm=='plain_full' else initial)
                assert r['second_search']['best_action'] in r['final_candidates']
                assert r['first_search']['root_hash']==first['first_search']['root_hash'] if arm!='plain_full' else True
                assert r['sum_both_root_search_wall_us']==r['first_search']['search_wall_us']+r['second_search']['search_wall_us']
                assert r['total_evaluator_calls_with_prior']==1+r['first_search']['root_evaluator_calls']+r['second_search']['root_evaluator_calls']
                assert r['total_decision_wall_us_without_model_load']>0
                assert r['total_decision_wall_us_without_model_load']>=r['sum_both_root_search_wall_us']
                assert r['MCTS_restart_instead_of_same_tree_reopening'] and r['no_online_exact_game_oracle']
                assert max(float(p) for c,p in r['first_search']['root_policy'])<=1.0+1e-6
                probe=tactical_from_source(root,initial,ordered,r['first_search']['best_action'],
                                           r['first_search']['root_policy'])
                if arm!='closed_no_probe':
                    z=r['tactical_probe']
                    assert z is not None and z['reason']==probe['reason'],(root['id'],mode,arm,z,probe)
                    assert z['fixation_gate']==probe['fixation_gate']
                    assert abs(z['fixation_share']-probe['fixation_share'])<1e-7
                    assert z['initial_best_reply_vulnerable']==probe['vulnerable']
                    assert z['excluded_immediate_wins']==probe['immediate_wins']
                    assert z['excluded_safe_actions']==probe['safe']
                    assert z['suggested_reopen']==probe['suggested']
                else:
                    assert r['tactical_probe'] is None
                if arm=='evidence_reopen':proposal=probe['suggested']
                elif arm.startswith('random_'):
                    proposal=hashed_random_omissions(omitted,len(probe['suggested']),
                        SEEDS[int(arm[-1])-1],root['history'])
                elif arm=='always_full_reopen':proposal=omitted
                else:proposal=[]
                assert r['actually_added_candidates']==proposal,(root['id'],mode,arm,r['actually_added_candidates'],proposal)
                expected_final=legal if arm=='plain_full' else initial+proposal
                assert r['final_candidates']==expected_final
                a=r['second_search']['best_action']
                WDL=sign(S[a])
                regret=max_exact-WDL
                assert score['action']==a and score['WDL']==WDL and score['regret']==regret
                assert score['omitted_strictly_better_exact_WDL']==decisive
                assert score['reopened_decisive_candidate']==bool(set(proposal)&set(decisive))
                assert score['chosen_strictly_better_omission']==(a in decisive)
                assert abs(score['decision_wall_ms']-r['total_decision_wall_us_without_model_load']/1000)<1e-12
                assert abs(score['MCTS_search_only_ms']-r['sum_both_root_search_wall_us']/1000)<1e-12
                if arm=='evidence_reopen':
                    before=position['modes'][mode]['closed_sham']['WDL']
                    exact_changed[(mode,'better' if WDL>before else 'worse' if WDL<before else 'same')]+=1
    for mode in ('new_mlp','conv'):
        assert cache[(mode,False)]==1 and cache[(mode,True)]==384*8-1,cache
    for group in ('reopen_A','reopen_B','pooled'):
        subset=[r for r in paired if group=='pooled' or r['group']==group]
        assert len(subset)==(384 if group=='pooled' else 192)
        summary[group]={}
        for mode in MODES:
            original=report['modes'][group][mode]
            mean={}
            for arm in ARMS:
                scores=[r['modes'][mode][arm] for r in subset]
                n=len(scores)
                regrets=sum(t['regret'] for t in scores)/n
                wall=sum(t['decision_wall_ms'] for t in scores)/n
                src=original['arms'][arm]
                assert abs(regrets-src['mean_regret'])<1e-12
                assert abs(wall-src['mean_decision_wall_ms_including_probe_and_setup'])<1e-12
                mean[arm]={'regret':regrets,'decision_wall_ms':wall}
            paired_result={}
            for arm in ('closed_sham','random_1','random_2','random_3',
                        'always_full_reopen','plain_full'):
                observed=original['compared']['evidence_minus_'+arm]
                delta=mean['evidence_reopen']['regret']-mean[arm]['regret']
                mismatch=abs(mean['evidence_reopen']['decision_wall_ms']-
                             mean[arm]['decision_wall_ms'])/mean['evidence_reopen']['decision_wall_ms']
                assert abs(observed['delta_regret']-delta)<1e-12
                assert abs(observed['CPU_mean_relative_mismatch']-mismatch)<1e-12
                expected_match='MATCHED_10_PERCENT' if mismatch<=.10 else 'NOT_MATCHED_10_PERCENT'
                assert expected_match==observed['CPU_10pct_gate']
                independent_delta,independent_ci=paired_family_bootstrap(subset,mode,arm)
                assert abs(independent_delta-observed['family_cluster_bootstrap']['mean_regret_delta'])<1e-12
                assert all(abs(a-b)<1e-12 for a,b in zip(independent_ci,observed['family_cluster_bootstrap']['CI95']))
                paired_result[arm]={'delta':delta,'CI95':independent_ci,
                                    'cost_mismatch':mismatch,'match':expected_match}
            source_eligible=sum(bool(r['modes'][mode]['evidence_reopen']['omitted_strictly_better_exact_WDL'])
                                for r in subset)
            assert source_eligible==original['posthoc_decisive_omission_eligible_count']
            summary[group][mode]={'arms':mean,'paired':paired_result,
                'posthoc_decisive_omission_count':source_eligible,
                'trigger_count':original['arms']['evidence_reopen']['triggered_count']}
    locksrc={x['identity']:x for x in lock['source_entries']}
    exe=Path(locksrc['ExternalSolver']['path'])
    book=Path(locksrc['ExternalBook']['path'])
    count=0
    for row in exact['rows']:
        if row['id'].endswith('_R00'):
            a=solver(exe,book,row['history'])
            flip=''.join(str(8-int(x)) for x in row['history'])
            b=solver(exe,book,flip)
            assert a==row['exact_action_scores'] and b==a[::-1]
            count+=1
    assert count==24
    output={
        'schema':'quartz_cog03_reopening12_independent_9216_CPU_and_exact_WDL_v1',
        'source_commit':receipt['source_commit'],
        'n_Rust_endpoint_requests':len(raw),'n_real_MCTS_root_searches':len(raw)*2,
        'new_game_roots':384,'new_mirror_disjoint_families':24,
        'independent_new_exact_solver_queries':count,
        'independent_new_horizontal_mirror_queries':count,
        'all_current_root_priors_and_first_second_root_candidate_masks_recomputed':True,
        'all_Rust_oneply_tactical_evidence_rederived_from_gravity_game_rules':True,
        'all_true_online_trigger_and_random_reopen_seeds_checked':True,
        'all_action_WDL_regret_cost_and_parent_trees_recomputed':True,
        'all_family_cluster_bootstrap_statistics_recomputed':True,
        'warm_model_cache_counts':{str(key):n for key,n in cache.items()},
        'WDL_changes_evidence_vs_sham':{str(key):n for key,n in exact_changed.items()},
        'cohort_results':summary,
        'not_claimed':['genuine human Einstellung', 'same-tree re-expansion',
                       'general engine superiority over all-legal full reopening']
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('duplicate full 9216 audit')
    p.write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
    print('COG03_INDEPENDENT_9216_TWO_TREE_RUST_AND_WDL_PASS',json.dumps({
        'actual_root_requests':len(raw),'additional_exact_original_and_mirror':count*2,
        'pooled_MLP_regret':summary['pooled']['new_mlp']['arms']['evidence_reopen']['regret'],
        'pooled_Conv_regret':summary['pooled']['conv']['arms']['evidence_reopen']['regret'],
        'Uniform_reopen_improvement':'NONE'
    },sort_keys=True))

if __name__=='__main__':
    main()
