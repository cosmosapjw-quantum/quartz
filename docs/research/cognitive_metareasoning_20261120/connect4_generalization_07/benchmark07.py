#!/usr/bin/env python3
"""Actual Rust CPU MCTS WDL-utility comparison of four evaluator modes.

Runs at frozen cumulative root visits 8,16,32,64 on 192 disjoint opening
family test roots + old 12 no-training roots. Exact external solver scores are
joined *after* all Rust searches complete. No Bayesian VOC controller claim.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time

from generate_cohort import board_from_history,canonical_state

BUDGETS=[8,16,32,64]
MODES=['uniform','old_mlp','new_mlp','conv']
EVAL_NAMES={
    'uniform':'uniform_no_neural_network',
    'old_mlp':'trained_tiny_mlp_cpu',
    'new_mlp':'trained_tiny_mlp_cpu',
    'conv':'trained_conv3x3_cpu',
}


def sha(p:Path)->str:
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(p:Path,*args:str)->str:
    return subprocess.check_output(['git','-C',str(p),*args],text=True,stderr=subprocess.DEVNULL).strip()


def wdl(score:int)->int:
    if type(score)!=int or score == -1000 or not -21<=score<=21:
        raise ValueError('illegal/sentinel Connect4 exact score')
    return int(score>0)-int(score<0)


def score_response(raw:dict,scores:list[int],board:list[int],legal:list[int])->dict:
    if raw.get('schema')!='quartz_connect4_resident_mcts_v1':
        raise ValueError('unknown Rust MCTS protocol schema')
    if len(scores)!=7 or len(board)!=42 or not legal:
        raise ValueError('bad exact per-column source/board')
    if raw.get('board_bottom_up')!=board or raw.get('legal_columns_0based')!=legal:
        raise ValueError('Rust gravity board/legality identity mismatch')
    if raw.get('fresh16',{}).get('resident_match') is not True:
        raise ValueError('fresh16 changed search prefix or evaluator output')
    root=raw.get('root_hash')
    if not isinstance(root,str) or not root:
        raise ValueError('missing root identity')
    for col in range(7):
        if (col in legal)!=(scores[col]!=-1000):
            raise ValueError('solver sentinel does not match legal columns')
    visits=raw.get('steps')
    if not isinstance(visits,list) or [s.get('budget') for s in visits]!=BUDGETS:
        raise ValueError('bad visit budget schedule')
    if any(s.get('root_hash')!=root or s.get('root_visits')!=b for b,s in zip(BUDGETS,visits)):
        raise ValueError('root/visit drift in continuation')
    load=raw.get('model_load_wall_us')
    setup=raw.get('root_setup_wall_us')
    total_with_fresh=raw.get('total_request_wall_us_including_fresh16')
    if any(type(x) is not int or x<0 for x in (load,setup,total_with_fresh)):
        raise ValueError('cost metadata not recorded')
    actions=[];u=[];cost=[];times=[];prev=raw.get('root_initial_evaluator_calls')
    if type(prev)!=int or prev<0:
        raise ValueError('bad initial evaluator count')
    for s in visits:
        a=s.get('best_action')
        if type(a)!=int or a not in legal:
            raise ValueError('illegal column selected by Rust MCTS')
        current=s.get('evaluator_calls')
        increment=s.get('additional_evaluator_calls')
        micros=s.get('step_wall_us')
        if (type(current)!=int or current<prev or
            type(increment)!=int or increment != current-prev):
            raise ValueError('evaluator counter inconsistent')
        if type(micros)!=int or micros<0:
            raise ValueError('search wall-time not recorded')
        actions.append(a)
        u.append(wdl(scores[a]))
        cost.append(current)
        times.append(micros)
        prev=current
    star=max(wdl(scores[col]) for col in legal)
    cum=[]
    for i in range(len(BUDGETS)):
        cum.append(load+setup+sum(times[:i+1]))
    return {
        'root_hash':root,'actions':actions,'utility':u,
        'regret':[star-v for v in u],
        'delta':[b-a for a,b in zip(u,u[1:])],
        'best_root_WDL':star,
        'cumulative_evaluations':cost,
        'stage_wall_us':times,
        'estimated_end_to_end_us_at_budget':cum,
        'model_load_wall_us':load,
        'root_setup_wall_us':setup,
        'total_request_wall_us_including_fresh16':total_with_fresh,
        'fresh16_reproduced':True,
        'mode':raw.get('eval_mode'),
    }


def summarize_mode(rows:list[dict])->dict:
    if not rows:raise ValueError('cannot summarize empty mode')
    vals=[x['score'] for x in rows]
    out={
        'n':len(rows),'families':len({x['family'] for x in rows}),
        'by_budget':{},'by_transition':{},
    }
    for i,b in enumerate(BUDGETS):
        u=[s['utility'][i] for s in vals]
        r=[s['regret'][i] for s in vals]
        cost=[s['cumulative_evaluations'][i] for s in vals]
        wall=[s['estimated_end_to_end_us_at_budget'][i] for s in vals]
        out['by_budget'][str(b)]={
            'n':len(vals),
            'mean_WDL':sum(u)/len(vals),
            'mean_regret':sum(r)/len(vals),
            'optimal_action_count':sum(z==0 for z in r),
            'WDL_counts':dict(Counter(str(z) for z in u)),
            'mean_evaluator_calls':sum(cost)/len(vals),
            'mean_stage_cumulative_wall_us':sum(
                sum(s['stage_wall_us'][:i+1]) for s in vals)/len(vals),
            'mean_model_load_wall_us':sum(s['model_load_wall_us'] for s in vals)/len(vals),
            'mean_root_setup_wall_us':sum(s['root_setup_wall_us'] for s in vals)/len(vals),
            'mean_estimated_end_to_end_us':sum(wall)/len(vals),
        }
    for i,(a,b) in enumerate(zip(BUDGETS,BUDGETS[1:])):
        gain=[s['delta'][i] for s in vals]
        out['by_transition'][f'{a}->{b}']={
            'signed_counts':dict(Counter(str(g) for g in gain)),
            'mean_gain':sum(gain)/len(gain),
            'mean_additional_evaluator_calls':sum(s['cumulative_evaluations'][i+1]-
                s['cumulative_evaluations'][i] for s in vals)/len(vals),
        }
    return out


def paired_family_bootstrap(pairs:list[dict],left:str,right:str,index:int,seed=20261121)->dict:
    groups=defaultdict(list)
    for row in pairs:
        groups[row['family']].append(row['models'][left]['regret'][index]-
                                     row['models'][right]['regret'][index])
    names=sorted(groups)
    # Previous 12 roots were selected differently; no family interval for them.
    if len(names)!=12 or any(len(groups[x])!=16 for x in names):
        return {'status':'not_eligible_for_12_family_bootstrap'}
    means=[sum(groups[k])/len(groups[k]) for k in names]
    rng=random.Random(seed)
    samples=[sum(rng.choice(means) for _ in means)/len(means) for _ in range(3000)]
    samples.sort()
    return {'n_families':12,'n_per_family':16,'point_mean':sum(means)/len(means),
            'conditional_95_interval':[samples[74],samples[2924]],
            'bootstrap_seed':seed,
            'bootstrap_refits_models':False}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--rust-binary',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    args=cli.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    if git(repo,'status','--porcelain'):
        raise RuntimeError('run only from committed clean code and frozen weights')
    locked=json.loads((d/'INPUT_LOCK.json').read_text())
    for it in locked['source_sha256']:
        p=Path(it['path'])
        if p.name not in ('COHORT_1024_FROZEN.json','COHORT_FROZEN.json',
                          'tiny_mlp_weights.json','POSITIONS_FIXED.json',
                          'c4_oracle_results.json','c4solver','7x6_small.book',
                          'generate_cohort.py','train_cpu.py','teacher_cpu.py'):
            continue
        if not p.is_file() or sha(p)!=it['sha256']:
            raise RuntimeError('frozen source/input mismatch: '+str(p))
    cohort=json.loads((d/'COHORT_1024_FROZEN.json').read_text())
    all_new=[r for r in cohort['rows'] if r['split']=='test']
    assert len(all_new)==192 and len({r['family'] for r in all_new})==12
    original=json.loads((d.parent/'standard_external_04/POSITIONS_FIXED.json').read_text())['connect4']
    assert len(original)==12
    cases=[]
    for r in original:
        cases.append({'id':r['id'],'family':r['id'],'split':'historical12',
                      'history':r['sequence'],'board':r['board'],
                      'legal_columns':[col for col in range(7) if r['board'][35+col]==0]})
    for r in all_new:
        cases.append({'id':r['id'],'family':r['family'],'split':'new_test192',
                      'history':r['history'],'board':r['board'],
                      'legal_columns':r['legal_columns']})
    assert len(cases)==204
    groups={s:sum(c['split']==s for c in cases) for s in ('historical12','new_test192')}
    assert groups=={'historical12':12,'new_test192':192}
    for r in cases:
        board,height=board_from_history(r['history'])
        if board!=r['board'] or r['legal_columns']!=[c for c in range(7) if height[c]<6]:
            raise RuntimeError('pre-oracle game position identity mismatch')
    old_model=d.parent/'connect4_recognizer_06/evidence/nn_train_20261120/tiny_mlp_weights.json'
    folder=d/'evidence/mlp_conv_train_20261121'
    model_paths={'old_mlp':old_model,'new_mlp':folder/'mlp_model.json','conv':folder/'conv_model.json'}
    train_receipt=json.loads((folder/'execution_manifest.json').read_text())
    assert train_receipt['source_commit']=='588495c705093b8618f9cc41aa4303677df23f8e'
    for name,rel in [('new_mlp','mlp_model.json'),('conv','conv_model.json')]:
        assert sha(model_paths[name])==train_receipt['raw_output_sha256'][rel]
    assert sha(old_model)==next(it['sha256'] for it in locked['source_sha256'] if it['path']==str(old_model.resolve()))
    queries=[]
    for mode in MODES:
        for root in cases:
            q={'cmd':'research_connect4_cpu',
               'history':root['history'],'seed':17,'budgets':BUDGETS}
            if mode in ('old_mlp','new_mlp'):
                q.update({'eval_mode':'tiny_mlp','model_path':str(model_paths[mode].resolve())})
            elif mode=='conv':
                q.update({'eval_mode':'conv3x3','model_path':str(model_paths['conv'].resolve())})
            queries.append(q)
    assert len(queries)==816
    exe=args.rust_binary.resolve()
    if not exe.is_file():raise FileNotFoundError(exe)
    run_payload='\n'.join(json.dumps(q,sort_keys=True,separators=(',',':')) for q in queries)
    run_payload+='\n'+json.dumps({'cmd':'quit'})+'\n'
    start=time.perf_counter()
    proc=subprocess.run([str(exe),'--server'],input=run_payload.encode(),
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        timeout=180,check=False)
    runtime_s=time.perf_counter()-start
    if proc.returncode!=0:
        raise RuntimeError(f'Rust CPU search failed: {proc.returncode},stderr={proc.stderr[-750:]!r}')
    results=[json.loads(line) for line in proc.stdout.decode('utf-8','replace').splitlines() if line.strip()]
    if len(results)!=len(queries):
        raise RuntimeError(f'Rust response mismatch {len(results)} != {len(queries)}')
    for q,result in zip(queries,results):
        if result.get('error'):
            # Preserve the exact request/root failure. No best-effort zero WDL.
            raise RuntimeError(f'Rust query failed history={q["history"]}: {result["error"]}')
        if result.get('history')!=q['history'] or result.get('schema')!='quartz_connect4_resident_mcts_v1':
            raise RuntimeError('Rust root/source response identity mismatch')
    # Labels are read *after* all search results were created.
    teacher=json.loads((d/'evidence/teacher_1024_20261121/labels.json').read_text())
    exact_new={r['id']:r['exact_action_scores'] for r in teacher['rows'] if r['split']=='test' and r['status']=='OK'}
    older=json.loads((d.parent/'standard_external_04/evidence/20261120_standard_cpu/c4_oracle_results.json').read_text())
    exact_old={r['position_id']:r['scores'] for r in older}
    assert len(exact_new)==192 and len(exact_old)==12
    joined=[]
    for i,root in enumerate(cases):
        scores=exact_old[root['id']] if root['split']=='historical12' else exact_new[root['id']]
        models={}
        for j,mode in enumerate(MODES):
            response=results[j*len(cases)+i]
            if response['eval_mode'] != {
                'uniform':'uniform_no_neural_network',
                'old_mlp':'trained_tiny_mlp_cpu',
                'new_mlp':'trained_tiny_mlp_cpu',
                'conv':'trained_conv3x3_cpu'}[mode]:
                raise RuntimeError('Rust evaluator/source mode mismatch')
            models[mode]=score_response(response,scores,root['board'],root['legal_columns'])
        same_root={entry['root_hash'] for entry in models.values()}
        if len(same_root)!=1:
            raise RuntimeError('same Connect4 position yielded different initial MCTS state hashes')
        joined.append({'id':root['id'],'family':root['family'],'split':root['split'],
                       'history':root['history'],'current_player':1 if len(root['history'])%2==0 else -1,
                       'exact_action_scores':scores,'models':models})
    # Compare uniform to original 2026-11-20 science archive. Added telemetry
    # fields must not affect actions, policy or evaluator-call counts.
    previous=json.loads((d.parent/'connect4_gamestate_05/evidence/20261120_connect4_exact_12/raw_mcts_responses.json').read_text())
    assert len(previous)==12
    for idx,(old,joined_root) in enumerate(zip(previous,joined[:12])):
        assert old['history']==joined_root['history']
        now=results[idx]
        for a,b in zip(old['steps'],now['steps']):
            if a['best_action']!=b['best_action'] or a['evaluator_calls']!=b['evaluator_calls']:
                raise RuntimeError('uniform baseline changed relative to previous archived science')
    summaries={}
    for cohort_label in ('historical12','new_test192'):
        subset=[r for r in joined if r['split']==cohort_label]
        summaries[cohort_label]={
            'n':len(subset),'models':{
                mode:summarize_mode([{'id':r['id'],'family':r['family'],'score':r['models'][mode]}
                                     for r in subset])
                for mode in MODES
            },
        }
    # Clustered paired differences on twelve fixed heldout families,
    # no IID per-position standard errors, and no retraining refit.
    valtest=[r for r in joined if r['split']=='new_test192']
    summary={
        'schema':'quartz_connect4_generalization07_real_MCTS_v1',
        'run_source_head':git(repo,'rev-parse','HEAD'),
        'n_hist':12,'n_new_test':192,'n_queries':len(queries),
        'mode_names':MODES,'visit_budget_schedule':BUDGETS,
        'total_server_wall_s':runtime_s,
        'cohorts':summaries,
        'new_test_paired_regret_by_family':{
            str(b):{
                'conv_minus_new_mlp':paired_family_bootstrap(valtest,'conv','new_mlp',i),
                'conv_minus_uniform':paired_family_bootstrap(valtest,'conv','uniform',i),
                'conv_minus_old_mlp':paired_family_bootstrap(valtest,'conv','old_mlp',i),
                'new_mlp_minus_old_mlp':paired_family_bootstrap(valtest,'new_mlp','old_mlp',i),
            } for i,b in enumerate(BUDGETS)
        },
        'interpretation':{
            'matched':'MCTS cumulative root visits only',
            'not_matched':['evaluator_calls','model_inference_wall_time','end_to_end_time'],
            'all_old_12_black_to_move':True,
            'new_test_wdls_from':'fixed independent exact PascalPons teacher labels',
            'model_load_cost_repeated_per_root':True,
            'total_request_cost_includes_fresh16_control':True,
            'nonnormal_noniid_families':True,
        },
    }
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name,value in [('raw_rust_results.json',results),
                        ('joined_WDL.json',joined),
                        ('summary.json',summary)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    for key,file in model_paths.items():
        if sha(file) != (train_receipt['raw_output_sha256']['mlp_model.json'] if key=='new_mlp'
                         else train_receipt['raw_output_sha256']['conv_model.json'] if key=='conv'
                         else next(it['sha256'] for it in locked['source_sha256'] if it['path']==str(file.resolve()))):
            raise RuntimeError('frozen model weights changed mid experiment')
    manifest={
        'schema':'quartz_connect4_generalization07_MCTS_manifest_v1',
        'source_head':git(repo,'rev-parse','HEAD'),
        'source_tree':git(repo,'rev-parse','HEAD^{tree}'),
        'binary_sha256':sha(exe),
        'model_sha256':{name:sha(p) for name,p in model_paths.items()},
        'cohort_sha256':sha(d/'COHORT_1024_FROZEN.json'),
        'oracle_label_sha256':sha(d/'evidence/teacher_1024_20261121/labels.json'),
        'old12_oracle_sha256':sha(d.parent/'standard_external_04/evidence/20261120_standard_cpu/c4_oracle_results.json'),
        'output_sha256':{p.name:sha(p) for p in out.iterdir()},
        'CPU_model_load_per_root':True,
        'trials':len(queries),'model_modes':MODES,
        'no_GPU':True,
        'no_Bayesian_VOC_controller':True,
    }
    (out/'execution_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print('COG07_MCTS_DONE',json.dumps({
        'runtime_s':runtime_s,'n_queries':len(queries),
        'new_test_regret':{mode:summaries['new_test192']['models'][mode]['by_budget']['16']['mean_regret'] for mode in MODES},
        'historical_regret':{mode:summaries['historical12']['models'][mode]['by_budget']['16']['mean_regret'] for mode in MODES},
        'output':str(out),
    },sort_keys=True),flush=True)


if __name__=='__main__':main()
