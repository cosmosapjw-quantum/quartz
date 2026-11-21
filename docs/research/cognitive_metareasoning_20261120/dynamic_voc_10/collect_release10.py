#!/usr/bin/env python3
"""Actual release MCTS full64 for G07 train/validation roots, exact oracle joined after.

640 train +192 validation opening-family roots, 3 evaluators=2496 real
in-process Rust root searches. No G07 test, C09 or C10 test used here.
Exact solver/teacher labels cannot enter runtime search requests.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

from dynamic_model10 import BUDGETS


def old_modules():
    old=Path(__file__).resolve().parent.parent/'cost_matched_voc_08'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    from train_gain08 import current_root_features,model_identity,rust_batch_requests,batch_run
    return current_root_features,model_identity,rust_batch_requests,batch_run


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()


def sign(score):
    if type(score)!=int or score==-1000 or not -21<=score<=21:
        raise ValueError('Connect4 signed external oracle invalid/sentinel score')
    return (score>0)-(score<0)


def labels_from_full(raw,action_scores):
    if len(action_scores)!=7:
        raise ValueError('seven exact external WDL action scores required')
    steps=raw.get('steps',[])
    if len(steps)!=4 or [s.get('budget') for s in steps]!=list(BUDGETS):
        raise ValueError('need full four-stage MCTS trajectory before joining labels')
    legal=raw.get('legal_columns_0based',[])
    if not legal or any((col in legal)!=(action_scores[col]!=-1000) for col in range(7)):
        raise ValueError('exact illegal column sentinel mismatches Rust legal actions')
    prev=raw.get('root_initial_evaluator_calls')
    if type(prev)!=int or prev<0:
        raise ValueError('missing initial evaluator count')
    signed=[]
    wall=[]
    current_root_features,*_=old_modules()
    for i,step in enumerate(steps):
        action=step.get('best_action')
        calls=step.get('evaluator_calls')
        micros=step.get('step_wall_us')
        if type(action)!=int or action not in legal:
            raise ValueError('illegal root action selected')
        if type(calls)!=int or calls<prev or type(micros)!=int or micros<0:
            raise ValueError('invalid real CPU search accounting')
        prev=calls
        signed.append(sign(action_scores[action]))
        wall.append(micros/1000.0)
    features=[current_root_features(raw,i) for i in range(3)]
    return {
        'features':features,
        'utility':signed,
        'gain_one':[signed[i+1]-signed[i] for i in range(3)],
        'gain_two':[signed[i+2]-signed[i] for i in range(2)],
        'search_stage_wall_ms':wall,
        'cost_one_ms':[wall[i+1] for i in range(3)],
        'cost_two_ms':[wall[i+1]+wall[i+2] for i in range(2)],
        'actions':[s['best_action'] for s in steps],
        'evaluator_calls':[s['evaluator_calls'] for s in steps],
    }


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--binary',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if git(repo,'status','--porcelain'):
        raise RuntimeError('collection requires committed clean source')
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    for it in lock['origin_files']:
        p=Path(it['path'])
        if p.name in ('mcts_server.rs','cog02.rs'):
            continue
        if not p.is_file() or p.stat().st_size!=it['size'] or sha(p)!=it['sha256']:
            raise RuntimeError('sealed data/model/file bytes changed: '+str(p))
    study=d.parent/'connect4_generalization_07'
    g=json.loads((study/'COHORT_1024_FROZEN.json').read_text())
    cases=[row for row in g['rows'] if row['split'] in ('train','val')]
    assert len(cases)==832 and Counter(r['split'] for r in cases)=={'train':640,'val':192}
    assert len({r['family'] for r in cases if r['split']=='train'})==40
    assert len({r['family'] for r in cases if r['split']=='val'})==12
    current_root_features,model_identity,rust_batch_requests,batch_run=old_modules()
    models=model_identity(repo)
    model_shas={name:sha(p) for name,p in models.items()}
    requests=rust_batch_requests(cases,repo)
    assert len(requests)==2496
    assert all('oracle' not in x and 'exact_action_scores' not in x for x in requests)
    exe=args.binary.resolve()
    if not exe.is_file():raise FileNotFoundError(exe)
    timer=time.perf_counter()
    raw=batch_run(exe,requests,timeout_s=120)
    wall=time.perf_counter()-timer
    assert len(raw)==2496
    # No exact label file was read until the server had completed every root.
    teacher=json.loads((study/'evidence/teacher_1024_20261121/labels.json').read_text())
    known={r['id']:r for r in teacher['rows'] if r['split'] in ('train','val') and r['status']=='OK'}
    assert len(known)==832
    joined=[]
    for j,mode in enumerate(('uniform','new_mlp','conv')):
        for i,case in enumerate(cases):
            root=raw[j*len(cases)+i]
            if (root.get('history')!=case['history'] or
                root.get('board_bottom_up')!=case['board'] or
                root.get('legal_columns_0based')!=case['legal_columns'] or
                root.get('seed')!=17 or root.get('stopped_at_visits')!=64 or
                root.get('fresh16',{}).get('resident_match') is not True):
                raise RuntimeError('Rust root/side/position/visited validation failed')
            if root.get('eval_mode')!={'uniform':'uniform_no_neural_network',
                                      'new_mlp':'trained_tiny_mlp_cpu',
                                      'conv':'trained_conv3x3_cpu'}[mode]:
                raise RuntimeError('wrong evaluator on exact fixed root')
            if mode!='uniform' and i>0 and root['model_cache_hit'] is not True:
                raise RuntimeError('model weights were not resident after first request')
            record=labels_from_full(root,known[case['id']]['exact_action_scores'])
            joined.append({
                'id':case['id'],'history':case['history'],'split':case['split'],
                'family':case['family'],'mode':mode,'root_hash':root['root_hash'],
                'root_exact_WDL':known[case['id']]['root_wdl'],
                **record
            })
    for name,p in models.items():
        if sha(p)!=model_shas[name]:
            raise RuntimeError('neural evaluator model mutated during release collection')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name,data in [('RAW_2496_RELEASE_FULL64.json',raw),
                      ('JOINED_TRAIN_VAL_RELEASE.json',joined)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n')
    receipt={
        'schema':'quartz_cog02_dynamic10_release_train_collection_manifest_v1',
        'source_commit':git(repo,'rev-parse','HEAD'),
        'source_tree':git(repo,'rev-parse','HEAD^{tree}'),
        'compiled_release_binary_sha256':sha(exe),
        'compiled_rust_commit':'4c7d490d3c23e05dbe0bbf74be55ad4b45c7f4f2',
        'teacher_label_sha256':sha(study/'evidence/teacher_1024_20261121/labels.json'),
        'source_training_cohort_sha256':sha(study/'COHORT_1024_FROZEN.json'),
        'model_sha256':model_shas,
        'number_real_Rust_searches':len(raw),'wall_elapsed_seconds':wall,
        'source_only_G07_train_val':True,'oracle_loaded_after_all_full64_traces':True,
        'files_sha256':{p.name:sha(p) for p in out.iterdir()}
    }
    (out/'EXECUTION_MANIFEST.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('C10_RELEASE_TRAIN_VAL_COMPLETED',json.dumps({
        'n':len(joined),'elapsed_s':wall,
        'mode_split':dict(Counter(str((x['mode'],x['split'])) for x in joined)),
        'stage0_negative_gain_counts':{
           mode:dict(Counter(str(x['gain_one'][0]) for x in joined
                     if x['mode']==mode and x['split']=='train'))
           for mode in ('uniform','new_mlp','conv')
        }
    },sort_keys=True))


if __name__=='__main__':main()
