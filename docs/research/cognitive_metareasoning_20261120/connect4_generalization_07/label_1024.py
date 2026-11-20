#!/usr/bin/env python3
"""Exact WDL teacher labels for pre-frozen 64-family 1024-root cohort.

Bounded to 5 s per root and 180 s globally. Timeout/unattempted statuses are
retained. Original 12 and previous 192 roots remain outside this cohort.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

from generate_cohort import canonical_history, canonical_state, board_from_history


def sha(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_cohort(cohort:dict)->dict:
    if cohort.get('schema')!='quartz_connect4_generalization07_frozen_v1':
        raise ValueError('frozen cohort schema mismatch')
    groups=cohort['families']
    rows=cohort['rows']
    if len(groups)!=64 or len(rows)!=1024:
        raise ValueError('frozen family/root count mismatch')
    prefix_map={}
    for g in groups:
        key=canonical_history(g['five_ply_prefix'])
        if key in prefix_map:
            raise ValueError('mirror-equivalent family duplicate')
        prefix_map[key]=g['family']
        if g['canonical_prefix']!=key:
            raise ValueError('canonical group key wrong')
    state_keys=set()
    counts=Counter()
    colors=Counter()
    depth=Counter()
    family_counts=Counter()
    for row in rows:
        g=groups[row['family']]
        if g['family']!=row['family'] or row['split']!=g['split']:
            raise ValueError('family split leaked')
        if not row['history'].startswith(g['five_ply_prefix']):
            raise ValueError('incorrect source family')
        if row['mirror_history']!=''.join(str(8-int(x)) for x in row['history']):
            raise ValueError('left-right history mismatch')
        board,heights=board_from_history(row['history'])
        if row['board']!=board:
            raise ValueError('gravity board/replay identity mismatch')
        key=canonical_state(board)
        if key in state_keys:
            raise ValueError('mirror-equivalent board collision')
        state_keys.add(key)
        if row['legal_columns']!=[c for c in range(7) if heights[c]<6]:
            raise ValueError('gravity legal moves mismatch')
        if row['current_player']!=(1 if len(row['history'])%2==0 else -1):
            raise ValueError('side-to-move parity mismatch')
        if row['plies']!=len(row['history']):
            raise ValueError('position depth wrong')
        counts[row['split']]+=1
        colors['black' if row['current_player']==1 else 'white']+=1
        depth[row['plies']]+=1
        family_counts[row['family']]+=1
    if dict(counts)!={'train':640,'val':192,'test':192}:
        raise ValueError('fixed split cardinalities changed')
    if any(n!=16 for n in family_counts.values()):
        raise ValueError('incomplete family')
    if colors!={'black':512,'white':512}:
        raise ValueError('color balance broken')
    return {'families':len(groups),'rows':len(rows),'splits':dict(counts),
            'colors':dict(colors),'depths':dict(depth)}


def exact_wdl_target(scores:list[int],legal:list[int])->dict:
    # The established COG06 target computes the full *WDL-optimal set*,
    # intentionally not a fastest-win single-move target.
    previous=Path(__file__).resolve().parent.parent/'connect4_recognizer_06'
    if str(previous) not in sys.path:
        sys.path.insert(0,str(previous))
    from teacher_cpu import parse_exact_label
    return parse_exact_label('',scores,legal)


def assert_complete_teacher(rows:list[dict],expected:int)->None:
    if len(rows)!=expected:
        raise RuntimeError('missing source rows or teacher results')
    missing=Counter(row['status'] for row in rows if row.get('status')!='OK')
    if missing:
        raise RuntimeError(f'missing exact labels: {dict(missing)}')


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip():
        raise RuntimeError('teacher experiment requires clean committed source')
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    cohort=json.loads((d/'COHORT_1024_FROZEN.json').read_text())
    before=verify_cohort(cohort)
    assert lock['frozen_cohort_rows']==1024
    for item in lock['source_sha256']:
        p=Path(item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise RuntimeError('LOCKED_TEACHER_INPUT_MISMATCH: '+str(p))
    previous=repo/'docs/research/cognitive_metareasoning_20261120/connect4_recognizer_06'
    sys.path.insert(0,str(previous))
    from teacher_cpu import query_one, encode_relative
    exe=next(Path(it['path']) for it in lock['source_sha256'] if it['path'].endswith('/c4solver'))
    book=next(Path(it['path']) for it in lock['source_sha256'] if it['path'].endswith('/7x6_small.book'))
    started=time.perf_counter()
    rows=[]
    for index,row in enumerate(cohort['rows']):
        if time.perf_counter()-started >= 180:
            answer={'status':'NOT_ATTEMPTED_GLOBAL_CAP'}
        else:
            answer=query_one(exe,book,row['history'],timeout_s=5.0)
        if answer['status']=='OK':
            try:
                label=exact_wdl_target(answer['scores'],row['legal_columns'])
                if len(label['optimal_actions'])==0:
                    raise ValueError('empty WDL-optimal legal action set')
                answer.update(label)
                answer['features']=encode_relative(row['board'],row['current_player'])
            except ValueError as exc:
                answer={'status':'INVALID_LABEL','error':str(exc)}
        rows.append({'id':row['id'],'family':row['family'],'split':row['split'],
                     'history':row['history'],'plies':row['plies'],
                     'current_player':row['current_player'],
                     'legal_columns':row['legal_columns'],**answer})
        if (index+1)%128==0:
            print('LABEL_PROGRESS',index+1,'of',1024,'elapsed_s',round(time.perf_counter()-started,2),flush=True)
    # ALL 13 freeze source files are unchanged, including prior model and teacher.
    for item in lock['source_sha256']:
        if sha(Path(item['path']))!=item['sha256']:
            raise RuntimeError('SOURCE_MODIFIED_DURING_LABEL_GENERATION: '+item['path'])
    count=dict(Counter(row['status'] for row in rows))
    by_split={s:dict(Counter(r['status'] for r in rows if r['split']==s)) for s in ('train','val','test')}
    result={
        'schema':'quartz_connect4_generalization07_oracle_labels_v1',
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'cohort_sha256':sha(d/'COHORT_1024_FROZEN.json'),
        'lock_sha256':sha(d/'INPUT_LOCK.json'),
        'oracle_binary_sha256':sha(exe),'oracle_book_sha256':sha(book),
        'rows':rows,'status_counts':count,'split_status':by_split,
        'full_validity':count=={'OK':1024},
        'elapsed_s':time.perf_counter()-started,
        'target_definition':'WDL-optimal full action set, no hidden oracle runtime calls',
    }
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    labels=target/'labels.json'
    labels.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    receipt={
        'schema':'quartz_c4_generalization07_teacher_manifest_v1',
        'study_source_commit':result['source_commit'],
        'cohort_sha256':result['cohort_sha256'],
        'locked_input_sha256':result['lock_sha256'],
        'labels_sha256':sha(labels),
        'total':len(rows),
        'statuses':count,
        'split_status':by_split,
        'source_sha256':sha(Path(__file__)),
        'per_position_timeout_s':5.0,
        'global_cap_s':180.0,
        'no_imputed_teacher_values':True,
    }
    (target/'teacher_manifest.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('TEACHER_COMPLETE',json.dumps({'count':count,'by_split':by_split,'wall_s':result['elapsed_s'],
                'labels_sha256':receipt['labels_sha256']},sort_keys=True),flush=True)


if __name__=='__main__':main()
