#!/usr/bin/env python3
"""Independent raw 2496 resident trajectories and train/val labels readback.

Read the actual QUARTZ JSON and exact frozen teacher label set anew; no
production training helpers, no test labels, no model fitting.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import math


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    d=Path(__file__).resolve().parent
    e=d/'evidence/train_val_2496_20261121'
    m=json.loads((e/'execution_manifest.json').read_text())
    raw=json.loads((e/'full_rust_trajectories.json').read_text())
    pairs=json.loads((e/'joined_train_val.json').read_text())
    frozen=json.loads((d.parent/'connect4_generalization_07/COHORT_1024_FROZEN.json').read_text())
    exact=json.loads((d.parent/'connect4_generalization_07/evidence/teacher_1024_20261121/labels.json').read_text())
    assert len(raw)==len(pairs)==m['requests']==2496
    assert m['source_head']=='62ab5cfccdc467e6b68b24cd202401a78a2581fc'
    for file,digest in m['output_sha256'].items():
        assert sha(e/file)==digest
    o={x['id']:x for x in frozen['rows'] if x['split']!='test'}
    label={x['id']:x for x in exact['rows'] if x['split']!='test' and x['status']=='OK'}
    assert len(o)==len(label)==832
    gain=defaultdict(Counter);groups=defaultdict(set);model_cache=defaultdict(Counter)
    costs=defaultdict(list)
    for v,(root,joined) in enumerate(zip(raw,pairs)):
        case=o[joined['id']]
        assert joined['history']==root['history']==case['history']
        assert joined['split']==case['split'] and joined['family']==case['family']
        assert root['board_bottom_up']==case['board']
        assert root['legal_columns_0based']==case['legal_columns']
        assert root['root_hash']==joined['root_hash']
        assert root['seed']==17
        assert root['fresh16']['resident_match'] is True
        assert root['stopped_at_visits']==64
        assert len(root['steps'])==4
        mode=joined['mode']
        assert root['eval_mode']=={'uniform':'uniform_no_neural_network','new_mlp':'trained_tiny_mlp_cpu','conv':'trained_conv3x3_cpu'}[mode]
        scores=label[joined['id']]['exact_action_scores']
        selected=[]
        prev=root['root_initial_evaluator_calls']
        for i,step in enumerate(root['steps']):
            assert step['budget']==step['root_visits']==[8,16,32,64][i]
            assert step['root_hash']==root['root_hash']
            col=step['best_action']
            assert col in case['legal_columns'] and scores[col]!=-1000
            selected.append((scores[col]>0)-(scores[col]<0))
            assert step['evaluator_calls']>=prev
            assert step['additional_evaluator_calls']==step['evaluator_calls']-prev
            assert step['evaluator_calls']==joined['cumulative_evaluator_calls'][i]
            if i>0:
                ms=step['step_wall_us']/1000
                assert abs(ms-joined['next_wall_ms'][i-1])<1e-9
                costs[(mode,i-1)].append(ms)
            prev=step['evaluator_calls']
        assert selected==joined['utility']
        assert [b-a for a,b in zip(selected,selected[1:])]==joined['gains']
        for i,g in enumerate(joined['gains']):
            gain[(mode,joined['split'],i)][str(g)]+=1
        groups[(mode,joined['split'])].add(case['family'])
        model_cache[mode][str(root['model_cache_hit'])]+=1
    assert all(len(groups[(mode,split)])==(40 if split=='train' else 12)
               for mode in ('uniform','new_mlp','conv') for split in ('train','val'))
    assert model_cache['new_mlp']['True']==831 and model_cache['conv']['True']==831
    assert model_cache['new_mlp']['False']==1 and model_cache['conv']['False']==1
    audit={
        'schema':'quartz_cog02_2496_training_independent_readback_v1',
        'source_commit':m['source_head'],
        'n_full_real_MCTS_resident_trajectories':2496,
        'training_roots_by_mode':640,
        'validation_roots_by_mode':192,
        'model_cache_hits':{mode:dict(v) for mode,v in model_cache.items()},
        'gain_counts':{f'{mode}/{split}/{i}':dict(dist)
                        for (mode,split,i),dist in sorted(gain.items())},
        'stage_next_wall_ms_mean':{f'{mode}/{i}':sum(v)/len(v)
                                   for (mode,i),v in sorted(costs.items())},
        'raw_shas_verified':len(m['output_sha256']),
        'no_test_root_labels_in_training_or_validation_source':True,
        'all_signed_gains_recomputed_from_frozen_external_exact_WDL':True,
        'not_Bayesian_KG':True,
    }
    p=e/'INDEPENDENT_READBACK.json'
    if p.exists():raise RuntimeError('refusing to overwrite independent source audit')
    p.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('COG02_TRAIN_FULL_INDEPENDENT_AUDIT_PASS',json.dumps({
        'trajectories':2496,'cache_hits':audit['model_cache_hits'],
        'model_cost_ms':audit['stage_next_wall_ms_mean'],
        'all_signed_labels_recomputed':True},sort_keys=True))


if __name__=='__main__':main()
