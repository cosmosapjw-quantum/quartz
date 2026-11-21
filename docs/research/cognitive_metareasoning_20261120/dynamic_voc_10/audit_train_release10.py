#!/usr/bin/env python3
"""Independent 2496 exact signed-gain root-sequence/CPU cost readback.

Never imports the production collector or training code.
"""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import math


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def independent_features(root,step):
    legal=root['legal_columns_0based']
    dense=[0.0]*7
    seen=set()
    for c,p in step['visit_policy']:
        assert c in legal and c not in seen and p>=0 and math.isfinite(p)
        seen.add(c);dense[c]=float(p)
    z=sum(dense)
    assert abs(z-1)<1e-4
    dense=[x/z for x in dense]
    values=sorted(dense,reverse=True)
    entropy=-sum(p*math.log(p) for p in dense if p>0)/math.log(7)
    b=step['budget']
    return [1.,b/64.,entropy,values[0],values[0]-values[1],
            len(legal)/7,step['evaluator_calls']/(b+1),
            root['ply']/42,1.0 if root['side_to_move']==1 else 0.0]


def main():
    d=Path(__file__).resolve().parent
    ev=d/'evidence/release_train_val_2496_20261121'
    receipt=json.loads((ev/'EXECUTION_MANIFEST.json').read_text())
    raw=json.loads((ev/'RAW_2496_RELEASE_FULL64.json').read_text())
    joined=json.loads((ev/'JOINED_TRAIN_VAL_RELEASE.json').read_text())
    g=d.parent/'connect4_generalization_07'
    cases=json.loads((g/'COHORT_1024_FROZEN.json').read_text())['rows']
    training=[x for x in cases if x['split'] in ('train','val')]
    exact=json.loads((g/'evidence/teacher_1024_20261121/labels.json').read_text())['rows']
    known={x['id']:x for x in exact if x['split'] in ('train','val')}
    assert len(training)==len(known)==832
    assert len(raw)==len(joined)==receipt['number_real_Rust_searches']==2496
    assert receipt['compiled_rust_commit']=='4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2'
    assert receipt['source_commit']=='0e36f3bf0ab5f43142ce2a33028d9845a76b67ca'
    assert sha(g/'COHORT_1024_FROZEN.json')==receipt['source_training_cohort_sha256']
    assert sha(g/'evidence/teacher_1024_20261121/labels.json')==receipt['teacher_label_sha256']
    for name,digest in receipt['files_sha256'].items():
        assert sha(ev/name)==digest
    for j,mode in enumerate(('uniform','new_mlp','conv')):
        for i,case in enumerate(training):
            k=j*len(training)+i
            r=raw[k];a=joined[k]
            assert r['history']==a['history']==case['history']
            assert r['board_bottom_up']==case['board']
            assert r['legal_columns_0based']==case['legal_columns']
            assert r['root_hash']==a['root_hash']
            assert a['family']==case['family'] and a['split']==case['split'] and a['mode']==mode
            assert r['side_to_move']==case['current_player']
            assert r['stopped_at_visits']==64 and r['fresh16']['resident_match']
            assert len(r['steps'])==4
            score=known[case['id']]['exact_action_scores']
            selected=[];ms=[]
            for stage,b in enumerate((8,16,32,64)):
                s=r['steps'][stage]
                assert s['root_visits']==s['budget']==b
                action=s['best_action']
                assert action in case['legal_columns'] and score[action]!=-1000
                u=(score[action]>0)-(score[action]<0)
                selected.append(u)
                ms.append(s['step_wall_us']/1000)
                if stage<3:
                    computed=independent_features(r,s)
                    assert len(computed)==len(a['features'][stage])==9
                    assert max(abs(x-y) for x,y in zip(computed,a['features'][stage]))<3e-9
            assert selected==a['utility']
            assert [selected[n+1]-selected[n] for n in range(3)]==a['gain_one']
            assert [selected[n+2]-selected[n] for n in range(2)]==a['gain_two']
            assert a['search_stage_wall_ms']==ms
            assert a['cost_one_ms']==ms[1:]
            assert all(abs(x-y)<1e-12 for x,y in zip(a['cost_two_ms'],(ms[1]+ms[2],ms[2]+ms[3])))
    result={
        'schema':'quartz_cog02_dynamic10_release_training_independent_v1',
        'source_commit':receipt['source_commit'],
        'n_actual_release_Rust_roots':2496,
        'n_train_by_mode':640,'n_val_by_mode':192,
        'n_train_families':40,'n_val_families':12,
        'all_joined_exact_WDL_root_actions_verified':True,
        'all_one_and_joint_two_step_gains_recomputed':True,
        'all_observed_release_stage_costs_recomputed':True,
        'all_current_root_nine_features_crosschecked':True,
        'new_C09_and_C10_test_roots_used_for_fit':False,
        'binary_sha256':receipt['compiled_release_binary_sha256'],
        'source_files_sha256':receipt['files_sha256'],
    }
    p=ev/'INDEPENDENT_READBACK.json'
    if p.exists():raise FileExistsError('refuse overwrite')
    p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('C10_RELEASE_TRAIN_2496_AUDIT_PASS',json.dumps({
        'n':2496,'features':9,'count_train':640,'count_val':192
    },sort_keys=True))


if __name__=='__main__':main()
