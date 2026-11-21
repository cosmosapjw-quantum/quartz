#!/usr/bin/env python3
"""Real CPU QUARTZ MCTS root-candidate fixation/reopening, fixed new C12 cohorts.

No Pascal Pons external exact-score information is passed to Rust; posthoc WDL
labels are read ONLY after every actual server search response is complete.
The engine runs masked first8+fresh second16 in EVERY arm, so none silently
gets a free continuation of the first tree. CPU includes prior/oneply probe.
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
ARMS=('closed_sham','evidence_reopen','random_1','random_2','random_3',
      'always_full_reopen','plain_full','closed_no_probe')
SEEDS=(202611210201,202611210202,202611210203)
BINARY_SHA='af4c64eaad0ff325d390d30a661bb292d090a80709cd1ec048964c0d54a07a8e'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sign(s):
    if type(s)!=int or s==-1000 or not -21<=s<=21:
        raise ValueError('invalid Connect4 exact legal-action score')
    return (s>0)-(s<0)


def exact_regret_for_action(scores,legal,action):
    if len(scores)!=7 or not legal or action not in legal or any(
        col not in range(7) or scores[col]==-1000 for col in legal):
        raise ValueError('exact WDL and legal move identity mismatch')
    if any((c in legal)!=(scores[c]!=-1000) for c in range(7)):
        raise ValueError('illegal external sentinel was inconsistent')
    return max(sign(scores[c]) for c in legal)-sign(scores[action])


def identify_decisive_omissions(scores,legal,initial):
    if not initial or not set(initial).issubset(legal) or len(set(initial))!=len(initial):
        raise ValueError('invalid original root prior candidates')
    m=max(sign(scores[c]) for c in initial)
    return [c for c in legal if c not in initial and sign(scores[c])>m]


def decision_cost_match(new_ms,reference_ms):
    if not math.isfinite(new_ms) or not math.isfinite(reference_ms) or new_ms<=0 or reference_ms<=0:
        raise ValueError('positive measured physical CPU wall time required')
    return 'MATCHED_10_PERCENT' if abs(new_ms-reference_ms)/new_ms<=0.1 else 'NOT_MATCHED_10_PERCENT'


def paired_cluster_interval(left,right,seed=20261121):
    if len(left)!=len(right) or not left:
        raise ValueError('nonempty matched-family root comparisons required')
    family=defaultdict(list)
    for a,b in zip(left,right):
        if a['family']!=b['family']:
            raise ValueError('mismatched paired opening-family identity')
        family[a['family']].append(a['regret']-b['regret'])
    assert all(len(x)==16 for x in family.values()),'not a full family cohort'
    means=[sum(y)/len(y) for _,y in sorted(family.items())]
    n=len(means)
    rng=random.Random(seed)
    boot=sorted(sum(rng.choice(means) for _ in means)/n for _ in range(2500))
    return {
        'n_roots':len(left),'n_families':n,
        'mean_regret_delta':sum(means)/n,
        'CI95':[boot[62],boot[2437]],
        'root_IID_inference':False,
        'one_fitted_policy_no_bootstrap_refitting':True,
    }


def refs():
    prev=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(prev) not in sys.path:sys.path.insert(0,str(prev))
    from evaluate_live08 import model_request,run_batch
    from train_gain08 import model_identity
    return model_request,run_batch,model_identity


def request_from_source(row,mode,arm,weights,shas):
    model_request,_,_=refs()
    q=model_request(row,mode,'full',weights,shas,None,None)
    q['cmd']='research_connect4_reopen_cpu'
    q['arm']='random_reopen' if arm.startswith('random_') else arm
    q['reopen_seed']=SEEDS[int(arm[-1])-1] if arm.startswith('random_') else SEEDS[0]
    q.pop('budgets',None)
    assert not any(k in q for k in ('oracle','exact_action_scores','future_checkpoint','teacher_WDL'))
    return q


def summary_arm(group,mode,arm):
    t=[r['modes'][mode][arm] for r in group]
    n=len(t)
    return {
        'n':n,'n_family_clusters':len({r['family'] for r in group}),
        'mean_regret':sum(x['regret'] for x in t)/n,
        'mean_exact_WDL':sum(x['WDL'] for x in t)/n,
        'mean_decision_wall_ms_including_probe_and_setup':sum(x['decision_wall_ms'] for x in t)/n,
        'mean_search_only_ms':sum(x['MCTS_search_only_ms'] for x in t)/n,
        'mean_root_setup_ms':sum(x['setup_ms'] for x in t)/n,
        'mean_probe_ms':sum(x['probe_ms'] for x in t)/n,
        'mean_prior_eval_ms':sum(x['prior_eval_ms'] for x in t)/n,
        'mean_evaluator_calls':sum(x['total_evaluator_calls'] for x in t)/n,
        'mean_final_candidate_count':sum(len(x['final_candidates']) for x in t)/n,
        'triggered_count':sum(x['probe_trigger'] for x in t),
        'actually_opened_count':sum(bool(x['added_candidates']) for x in t),
        'reopened_decisive_candidate_count':sum(x['reopened_decisive_candidate'] for x in t),
        'chosen_decisive_low_prior_move_count':sum(x['chosen_strictly_better_omission'] for x in t),
        'stop_regret_counts':dict(Counter(str(x['regret']) for x in t)),
        'final_decision_WDL_counts':dict(Counter(str(x['WDL']) for x in t)),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('new heldout evaluation requires a clean committed source')
    model_request,run_batch,model_identity=refs()
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    cohortfile=d/'COHORT_384_FROZEN.json'
    cohort=json.loads(cohortfile.read_text())
    if len(cohort['rows'])!=384 or len(cohort['families'])!=24:
        raise RuntimeError('new C12 source cardinality changed')
    if lock['freeze_new_cohort_seeds']!=cohort['generation_seeds']:
        raise RuntimeError('new root family RNG drifted')
    if set(SEEDS)!=set(lock['random_reopen_seed_selection']):
        raise RuntimeError('random opening seeds changed')
    first_sha=sha(cohortfile)
    weights=model_identity(repo)
    weight_sha={n:sha(p) for n,p in weights.items()}
    records=[]
    keys=[]
    for row in cohort['rows']:
        for mode in MODES:
            for arm in ARMS:
                records.append(request_from_source(row,mode,arm,weights,weight_sha))
                keys.append((row['id'],mode,arm))
    assert len(records)==384*3*8==9216
    assert all('exact_action_scores' not in r and 'oracle' not in r for r in records)
    exe=args.binary.resolve()
    if not exe.is_file() or sha(exe)!=BINARY_SHA:
        raise RuntimeError('COG03 Rust release binary SHA mismatch')
    start=time.perf_counter()
    raw=run_batch(exe,records)
    elapsed=time.perf_counter()-start
    assert len(raw)==9216
    # Only now load external exact target labels, not before the 9216 searches.
    exactfile=d/'evidence/new_384_exact_20261121/EXACT_NEW384_LABELS.json'
    label=json.loads(exactfile.read_text())
    if label['status']['statuses']!={'OK':384} or not label['status']['all_OK']:
        raise RuntimeError('external solver source incomplete')
    oracle={x['id']:x for x in label['rows']}
    assert len(oracle)==384
    source={x['id']:x for x in cohort['rows']}
    pairs={x['id']:{
        'id':x['id'],'group':x['group'],'family':x['family'],
        'history':x['history'],'side_to_move':x['current_player'],
        'modes':{},
    } for x in cohort['rows']}
    rawmap={}
    for (ident,mode,arm),r in zip(keys,raw):
        old=source[ident]
        if r['history']!=old['history'] or r['seed']!=17:
            raise RuntimeError('real MCTS source history or deterministic seed changed')
        if r['board_bottom_up']!=old['board'] or r['side_to_move']!=old['current_player']:
            raise RuntimeError('original Connect4 game state changed')
        if r['legal_columns_0based']!=old['legal_columns']:
            raise RuntimeError('actual legal Connect4 moves are inconsistent')
        if r['eval_mode']!={'uniform':'uniform_no_neural_network',
                           'new_mlp':'trained_tiny_mlp_cpu',
                           'conv':'trained_conv3x3_cpu'}[mode]:
            raise RuntimeError('wrong source neural evaluator')
        if mode!='uniform' and r['cache_model'] is not True:
            raise RuntimeError('frozen evaluator not resident in Rust server')
        if r['first_search']['root_visits']!=8 or r['second_search']['root_visits']!=16:
            raise RuntimeError('not two actually completed independent root searches')
        initial=r['initial_candidates']
        final=r['final_candidates']
        legal=r['legal_columns_0based']
        if not 1<=len(initial)<=3 or not set(initial).issubset(legal):
            raise RuntimeError('invalid original three-candidate attention gate')
        if len(set(initial))!=len(initial) or len(set(final))!=len(final) or not set(final).issubset(legal):
            raise RuntimeError('invalid final root gate state')
        if r['first_search']['best_action'] not in (legal if arm=='plain_full' else initial):
            raise RuntimeError('first root violated real candidate mask')
        if r['second_search']['best_action'] not in final:
            raise RuntimeError('second root violated real reopened mask')
        if not r['MCTS_restart_instead_of_same_tree_reopening']:
            raise RuntimeError('falsely claiming same tree retained')
        if arm.startswith('random_') and r['reopen_seed']!=SEEDS[int(arm[-1])-1]:
            raise RuntimeError('random-reopening seed changed')
        if len(r['unmasked_evaluator_prior_rank'])!=len(legal):
            raise RuntimeError('full prior list incomplete')
        sorted_prior=[col for col,_ in r['unmasked_evaluator_prior_rank']]
        if sorted_prior[:len(initial)]!=initial or sorted(sorted_prior)!=sorted(legal):
            raise RuntimeError('non-oracle root prior attention order mismatch')
        p=r['tactical_probe']
        if arm!='closed_no_probe' and (p is None or not isinstance(p['trigger'],bool)):
            raise RuntimeError('sham/treatment probe not evaluated')
        if arm=='closed_no_probe' and p is not None:
            raise RuntimeError('no-probe control unexpectedly used tactical lookahead')
        if arm in ('closed_sham','closed_no_probe') and final!=initial:
            raise RuntimeError('closed baseline actually reopened root')
        if arm=='evidence_reopen':
            if r['actually_added_candidates']!=p['suggested_reopen']:
                raise RuntimeError('online candidate reopening did not follow predeclared evidence')
            if len(r['actually_added_candidates'])>2:
                raise RuntimeError('more than two excluded moves opened')
            if not p['trigger'] and r['actually_added_candidates']:
                raise RuntimeError('reopened without a trigger')
        if arm.startswith('random_') and len(r['actually_added_candidates'])!=len(p['suggested_reopen']):
            raise RuntimeError('random control altered number of reopened candidates')
        if arm=='always_full_reopen' and set(final)!=set(legal):
            raise RuntimeError('indiscriminate opening failed')
        if arm=='plain_full' and set(final)!=set(legal):
            raise RuntimeError('unrestricted root control was masked')
        if r['total_evaluator_calls_with_prior']!=(
            1+r['first_search']['root_evaluator_calls']+r['second_search']['root_evaluator_calls']):
            raise RuntimeError('model call source dropped')
        S=oracle[ident]['exact_action_scores']
        action=r['second_search']['best_action']
        regret=exact_regret_for_action(S,legal,action)
        WDL=sign(S[action])
        omitted=identify_decisive_omissions(S,legal,initial)
        added=r['actually_added_candidates']
        score={
            'regret':regret,'WDL':WDL,'action':action,
            'final_action_prior_rank':sorted_prior.index(action)+1,
            'initial_best':r['first_search']['best_action'],
            'initial_best_WDL':sign(S[r['first_search']['best_action']]),
            'initial_candidates':initial,
            'final_candidates':final,
            'added_candidates':added,
            'probe_trigger':bool(p['trigger']) if p is not None else False,
            'trigger_reason':p['reason'] if p is not None else 'probe_not_run',
            'fixation_share':float(p['fixation_share']) if p is not None else None,
            'omitted_strictly_better_exact_WDL':omitted,
            'reopened_decisive_candidate':bool(set(added)&set(omitted)),
            'chosen_strictly_better_omission':action in omitted,
            'decision_wall_ms':r['total_decision_wall_us_without_model_load']/1000,
            'MCTS_search_only_ms':r['sum_both_root_search_wall_us']/1000,
            'setup_ms':(r['first_search']['setup_wall_us']+r['second_search']['setup_wall_us'])/1000,
            'probe_ms':r['probe_wall_us']/1000,
            'prior_eval_ms':r['prior_inference_wall_us']/1000,
            'model_load_ms':r['model_load_wall_us']/1000,
            'total_evaluator_calls':r['total_evaluator_calls_with_prior'],
            'full_game_best_WDL':oracle[ident]['root_wdl']
        }
        pairs[ident]['modes'].setdefault(mode,{})[arm]=score
        rawmap[(ident,mode,arm)]=r
    paired=[pairs[r['id']] for r in cohort['rows']]
    for root in paired:
        for mode in MODES:
            c=rawmap[(root['id'],mode,'closed_sham')]
            t=rawmap[(root['id'],mode,'evidence_reopen')]
            for field in ('initial_candidates','unmasked_evaluator_prior_rank','initially_omitted'):
                if c[field]!=t[field]:
                    raise RuntimeError('test/control root prior or candidate mismatch')
            for field in ('best_action','root_policy','root_evaluator_calls'):
                if c['first_search'][field]!=t['first_search'][field]:
                    raise RuntimeError('online probe experiment changed first root 8-visit search')
            if c['tactical_probe']['suggested_reopen']!=t['tactical_probe']['suggested_reopen']:
                raise RuntimeError('sham control did not see the same tactical evidence')
            for arm in ('random_1','random_2','random_3'):
                r=rawmap[(root['id'],mode,arm)]
                if r['tactical_probe']['suggested_reopen']!=t['tactical_probe']['suggested_reopen']:
                    raise RuntimeError('random reopening source evidence changed')
            if set(t['final_candidates'])==set(c['final_candidates']):
                # Source response cost may differ by OS jitter, but best action
                # must remain exactly equal in deterministic resident CPU.
                assert t['second_search']['best_action']==c['second_search']['best_action']
    for name,p in weights.items():
        if sha(p)!=weight_sha[name]:
            raise RuntimeError('trained evaluator bytes changed during CPU experiment')
    summary={
        'schema':'quartz_cog03_einstellung12_heldout_actual_reopening_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'source_cohort_sha256':first_sha,
        'external_oracle_sha256':sha(exactfile),
        'compiled_Rust_release_binary_sha256':sha(exe),
        'trained_NN_model_sha256':weight_sha,
        'actual_inprocess_Rust_2root_search_requests':len(raw),
        'n_physical_Rust_MCTS_root_searches':2*len(raw),
        'root_restarts_are_real':True,
        'n_distinct_opening_families':24,
        'n_distinct_new_positions':384,
        'two_groups':2,
        'elapsed_actual_live_server_s':elapsed,
        'modes':{},
        'nonclaims':['artificial masked root not a human Einstellung experiment',
                     'not resident same-tree root edge insertion',
                     'one-ply safety not perfect-game optimality',
                     'no human search or skill inference',
                     'OS wall-clock and conditional 24 opening families only']
    }
    for group in ('reopen_A','reopen_B','pooled'):
        subset=[x for x in paired if x['group']==group or group=='pooled']
        assert len(subset)==(384 if group=='pooled' else 192)
        summary['modes'][group]={}
        for mode in MODES:
            arms={arm:summary_arm(subset,mode,arm) for arm in ARMS}
            closed=arms['closed_sham']
            evidence=arms['evidence_reopen']
            comparisons={}
            for arm in ('evidence_reopen','random_1','random_2','random_3','always_full_reopen','plain_full','closed_no_probe'):
                if arm=='closed_sham':continue
                other=arms[arm]
                pairL=[{'family':r['family'],'regret':r['modes'][mode]['evidence_reopen']['regret']} for r in subset]
                pairR=[{'family':r['family'],'regret':r['modes'][mode][arm]['regret']} for r in subset]
                comparisons['evidence_minus_'+arm]={
                    'delta_regret':evidence['mean_regret']-other['mean_regret'],
                    'CPU_mean_relative_mismatch':abs(evidence['mean_decision_wall_ms_including_probe_and_setup']-
                        other['mean_decision_wall_ms_including_probe_and_setup'])/
                        evidence['mean_decision_wall_ms_including_probe_and_setup'],
                    'CPU_10pct_gate':decision_cost_match(
                        evidence['mean_decision_wall_ms_including_probe_and_setup'],
                        other['mean_decision_wall_ms_including_probe_and_setup']),
                    'family_cluster_bootstrap':paired_cluster_interval(pairL,pairR)
                }
            evidence_pairs=[r['modes'][mode]['evidence_reopen'] for r in subset]
            closed_pairs=[r['modes'][mode]['closed_sham'] for r in subset]
            pairE=[{'family':r['family'],'regret':r['modes'][mode]['evidence_reopen']['regret']} for r in subset]
            pairC=[{'family':r['family'],'regret':r['modes'][mode]['closed_sham']['regret']} for r in subset]
            comparisons['evidence_minus_closed_sham']={
                'delta_regret':evidence['mean_regret']-closed['mean_regret'],
                'CPU_mean_relative_mismatch':abs(evidence['mean_decision_wall_ms_including_probe_and_setup']-
                    closed['mean_decision_wall_ms_including_probe_and_setup'])/
                    evidence['mean_decision_wall_ms_including_probe_and_setup'],
                'CPU_10pct_gate':decision_cost_match(
                    evidence['mean_decision_wall_ms_including_probe_and_setup'],
                    closed['mean_decision_wall_ms_including_probe_and_setup']),
                'family_cluster_bootstrap':paired_cluster_interval(pairE,pairC)
            }
            eligible=[x for x in evidence_pairs if x['omitted_strictly_better_exact_WDL']]
            informative=[r for r in subset if len({sign(oracle[r['id']]['exact_action_scores'][a])
                for a in source[r['id']]['legal_columns']})>1]
            winner_changes=Counter()
            for before,after in zip(closed_pairs,evidence_pairs):
                if after['WDL']>before['WDL']:winner_changes['better']+=1
                elif after['WDL']<before['WDL']:winner_changes['worse']+=1
                else:winner_changes['same']+=1
            summary['modes'][group][mode]={
                'arms':arms,'compared':comparisons,
                'root_WDL_class_counts':dict(Counter(str(oracle[r['id']]['root_wdl']) for r in subset)),
                'WDL_informative_root_count':len(informative),
                'posthoc_decisive_omission_eligible_count':len(eligible),
                'eligible_and_probe_trigger_count':sum(x['probe_trigger'] for x in eligible),
                'eligible_and_decisive_reopen_count':sum(x['reopened_decisive_candidate'] for x in eligible),
                'chosen_action_WDL_change_vs_closed':dict(winner_changes),
                'trigger_cause_counts':dict(Counter(x['trigger_reason'] for x in evidence_pairs)),
                'informative_only_mean_regret':{
                    arm:sum(r['modes'][mode][arm]['regret'] for r in informative)/len(informative)
                    if informative else None for arm in ARMS
                },
            }
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    for name,content in [
        ('RAW_9216_RELEASE_ROOT_RESTARTS.json',raw),
        ('PAIRED_384_WDL_ATTENTION.json',paired),
        ('SUMMARY_REOPEN12.json',summary)
    ]:
        (output/name).write_text(json.dumps(content,sort_keys=True,ensure_ascii=False)+'\n')
    manifest={
        'schema':'quartz_cog03_reopening12_live_execution_manifest_v1',
        'source_commit':summary['source_commit'],
        'source_tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
        'compiled_Rust_source_commit':'afdc426d8062f227c8affc3d571fd2bc42d1bb36',
        'compiled_binary_sha256':BINARY_SHA,
        'trained_neural_sha256':weight_sha,
        'new384_data_sha256':first_sha,
        'new384_external_labels_sha256':sha(exactfile),
        'real_Rust_root_restart_requests':len(raw),
        'real_MCTS_searches':2*len(raw),
        'oracle_file_first_read_after_all_Rust_response_completion':True,
        'file_sha256':{p.name:sha(p) for p in output.iterdir()}
    }
    (output/'EXECUTION_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print('COG03_REAL_CPU_REOPEN12_DONE',json.dumps({
        'n_Rust_requests':len(raw),'actual_MCTS_searches':2*len(raw),
        'server_elapsed_s':elapsed,
        'modes':{mode:{group:{
            'evidence_regret':summary['modes'][group][mode]['arms']['evidence_reopen']['mean_regret'],
            'sham_regret':summary['modes'][group][mode]['arms']['closed_sham']['mean_regret'],
            'triggered':summary['modes'][group][mode]['arms']['evidence_reopen']['triggered_count'],
            'strictly_decisive_omissions':summary['modes'][group][mode]['posthoc_decisive_omission_eligible_count'],
            'newly_reopened_decisive':summary['modes'][group][mode]['eligible_and_decisive_reopen_count'],
            'physical_cost_match':summary['modes'][group][mode]['compared']['evidence_minus_closed_sham']['CPU_10pct_gate'],
        } for group in ('reopen_A','reopen_B','pooled')} for mode in MODES}
    },sort_keys=True))


if __name__=='__main__':
    main()
