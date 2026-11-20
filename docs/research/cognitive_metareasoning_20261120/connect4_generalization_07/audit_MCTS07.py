#!/usr/bin/env python3
"""Independent raw MCTS game-theoretic readback, no production scorer imported.

Requeries a solver on twelve NEW heldout family exemplars and mirrors.
One readback pass only. Confirms 816 actual Rust evaluations, full cohort,
no teacher or oracle injected into runtime, checksum and cost identities.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import subprocess


def sha(p:Path):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def root_board(history):
    board=[0]*42
    heights=[0]*7
    for i,digit in enumerate(history):
        col=int(digit)-1
        assert 0<=col<7 and heights[col]<6
        board[heights[col]*7+col]=1 if i%2==0 else -1
        heights[col]+=1
    return board,[col for col in range(7) if heights[col]<6]


def score_sign(score):
    assert -21<=score<=21 and score!=-1000
    return (score>0)-(score<0)


def solver(exe,book,h):
    p=subprocess.run([str(exe),'-a','-b',str(book)],
        input=(h+'\n').encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=12,check=True)
    s=p.stdout.decode().split()
    assert len(s)==8 and s[0]==h
    return [int(x) for x in s[1:]]


def main():
    d=Path(__file__).resolve().parent
    out=d/'evidence/real_MCTS_816_20261121'
    manifest=json.loads((out/'execution_manifest.json').read_text())
    frozen=json.loads((d/'COHORT_1024_FROZEN.json').read_text())
    truth=json.loads((d/'evidence/teacher_1024_20261121/labels.json').read_text())
    old=json.loads((d.parent/'standard_external_04/evidence/20261120_standard_cpu/c4_oracle_results.json').read_text())
    oldpositions=json.loads((d.parent/'standard_external_04/POSITIONS_FIXED.json').read_text())['connect4']
    raw=json.loads((out/'raw_rust_results.json').read_text())
    joined=json.loads((out/'joined_WDL.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert manifest['source_head']=='c29a0b35226f305fff9da648737a8235b5771b53'
    for name,expected in manifest['output_sha256'].items():
        assert sha(out/name)==expected,name
    assert sha(d/'COHORT_1024_FROZEN.json')==manifest['cohort_sha256']
    assert sha(d/'evidence/teacher_1024_20261121/labels.json')==manifest['oracle_label_sha256']
    assert sha(d.parent/'standard_external_04/evidence/20261120_standard_cpu/c4_oracle_results.json')==manifest['old12_oracle_sha256']
    rust=Path('/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target/debug/mcts_demo')
    assert sha(rust)==manifest['binary_sha256']
    assert len(raw)==816 and len(joined)==204
    test=[r for r in frozen['rows'] if r['split']=='test']
    assert len(test)==192
    assert len({r['family'] for r in test})==12
    assert Counter(r['current_player'] for r in test)==Counter({1:96,-1:96})
    c={r['id']:r['exact_action_scores'] for r in truth['rows'] if r['split']=='test'}
    assert len(c)==192
    prior={r['position_id']:r['scores'] for r in old}
    assert len(prior)==12
    assert [r['id'] for r in joined[:12]]==[r['id'] for r in oldpositions]
    assert [r['id'] for r in joined[12:]]==[r['id'] for r in test]
    modes=summary['mode_names']
    assert modes==['uniform','old_mlp','new_mlp','conv']
    for root_i,root in enumerate(joined):
        oracle=prior[root['id']] if root_i<12 else c[root['id']]
        assert root['exact_action_scores']==oracle
        board,legal=root_board(root['history'])
        assert (root_i<12)==(root['split']=='historical12')
        root_hashes=set()
        for j,mode in enumerate(modes):
            reply=raw[j*204+root_i]
            assert reply['history']==root['history']
            assert reply['board_bottom_up']==board
            assert reply['legal_columns_0based']==legal
            assert reply['fresh16']['resident_match']
            root_hashes.add(reply['root_hash'])
            r=root['models'][mode]
            assert reply['eval_mode']==r['mode']
            expectedmode={'uniform':'uniform_no_neural_network',
                          'old_mlp':'trained_tiny_mlp_cpu',
                          'new_mlp':'trained_tiny_mlp_cpu',
                          'conv':'trained_conv3x3_cpu'}[mode]
            assert expectedmode==reply['eval_mode']
            expected_reward=max(score_sign(oracle[col]) for col in legal)
            prev=reply['root_initial_evaluator_calls']
            previous_us=reply['model_load_wall_us']+reply['root_setup_wall_us']
            for i,budget in enumerate((8,16,32,64)):
                step=reply['steps'][i]
                assert step['root_visits']==step['budget']==budget
                assert step['root_hash']==reply['root_hash']
                action=step['best_action']
                assert action in legal
                assert oracle[action]!=-1000
                assert step['evaluator_calls']>=prev
                assert step['additional_evaluator_calls']==step['evaluator_calls']-prev
                previous_us+=step['step_wall_us']
                prev=step['evaluator_calls']
                assert r['utility'][i]==score_sign(oracle[action])
                assert r['regret'][i]==expected_reward-score_sign(oracle[action])
                assert r['cumulative_evaluations'][i]==prev
                assert r['estimated_end_to_end_us_at_budget'][i]==previous_us
                assert r['actions'][i]==action
        assert len(root_hashes)==1
    # Solver oracle integrity: one root/family and its mirror, at fixed
    # R00 selection. No source labels are sent to the Rust server.
    ex=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
    count=0
    for row in test:
        if row['id'].endswith('_R00'):
            assert solver(ex/'c4solver',ex/'7x6_small.book',row['history'])==c[row['id']]
            reflected=''.join(str(8-int(z)) for z in row['history'])
            assert solver(ex/'c4solver',ex/'7x6_small.book',reflected)==list(reversed(c[row['id']]))
            count+=1
    assert count==12
    # Recompute all modes, two cohorts and four budgets using only *raw*
    # chosen actions and exact stored WDL oracle; do not call benchmark07.py.
    recomputed={}
    for cohort in ('historical12','new_test192'):
        subs=[x for x in joined if x['split']==cohort]
        assert len(subs)==(12 if cohort=='historical12' else 192)
        recomputed[cohort]={}
        for mode in modes:
            recomputed[cohort][mode]={}
            for i,b in enumerate((8,16,32,64)):
                n=len(subs)
                regret=sum(x['models'][mode]['regret'][i] for x in subs)/n
                calls=sum(x['models'][mode]['cumulative_evaluations'][i] for x in subs)/n
                wall=sum(x['models'][mode]['estimated_end_to_end_us_at_budget'][i] for x in subs)/n
                result=summary['cohorts'][cohort]['models'][mode]['by_budget'][str(b)]
                assert result['mean_regret']==regret
                assert result['mean_evaluator_calls']==calls
                assert result['mean_estimated_end_to_end_us']==wall
                recomputed[cohort][mode][str(b)]={
                    'mean_regret':regret,'mean_evaluator_calls':calls,
                    'mean_estimated_end_to_end_us':wall,
                    'optimal_actions':sum(x['models'][mode]['regret'][i]==0 for x in subs)}
    audit={
      'schema':'quartz_connect4_generalization07_independent_MCTS_audit_v1',
      'source_commit':manifest['source_head'],
      'exact_oracle_confirmed_at_12_new_families':count,
      'solver_mirror_confirmations':count,
      'n_root_states':204,'n_model_root_searches':816,
      'n_new_test_families':12,'n_new_test_both_colors':192,
      'binary_sha256_checked':True,'all_prior_uniform_12_reproduced':True,
      'model_sha256_checked':len(manifest['model_sha256']),
      'raw_output_sha256_checked':len(manifest['output_sha256']),
      'all_new_192_outcome_recomputed':True,
      'recomputed':recomputed,
      'interpretation_limits':['user provided non-IID roots grouped 12 families',
                               'matched root visits but NOT physical CPU cost',
                               'model file loads repeated per root',
                               'not a Bayesian computation-selection controller',
                               'bootstrap conditional on fixed trained models'],
    }
    destination=out/'INDEPENDENT_READBACK.json'
    if destination.exists():raise RuntimeError('refuse overwrite MCTS readback')
    destination.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT_MCTS_AUDIT_OK',json.dumps({
        'queries':816,'heldout_roots':192,'heldout_families':12,
        'solver_mirror_requeries':count,'source_head':manifest['source_head'],
    },sort_keys=True))

if __name__=='__main__':main()
