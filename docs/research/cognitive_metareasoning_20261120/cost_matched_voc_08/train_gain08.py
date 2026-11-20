#!/usr/bin/env python3
"""Current-root-only signed-gain surrogate for QUARTZ's CPU MCTS (COG02-VOC08).

The exact Pascal Pons Connect4 solver values are offline target labels ONLY.
Runtime inference uses nine current root features and frozen linear weights.
This is not a Bayesian KG estimator. WDL improvements CAN be negative.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np

BUDGETS=(8,16,32,64)
STAGE_VISITS=(8,16,32)
LAMBDAS=(0.0,0.01,0.03,0.05,0.10)
TRAIN_VAL_SETS={'train':640,'val':192}
MODES=('uniform','new_mlp','conv')
SCHEMA='quartz_cog02_signed_gain_surrogate_v1'
FEATURES='root_visits_v1'
RIDGE_ALPHA=1.0
VALIDATION_COST_PRICE=0.04  # exact WDL utility per ms; fixed before results


def sha(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(repo:Path,*args:str)->str:
    return subprocess.check_output(['git','-C',str(repo),*args],
        text=True,stderr=subprocess.DEVNULL).strip()


def default_features_schema()->list[str]:
    return ['bias','budget_div64','visit_entropy_normalized7',
            'max_visit_share','top1_minus_top2','fraction_legal',
            'completed_eval_calls_div_budget_plus1','plies_div42','black_to_move']


def current_root_features(root:dict,stage:int)->list[float]:
    if stage not in (0,1,2):
        raise ValueError('COG02 current features only at 8/16/32')
    steps=root.get('steps')
    if not isinstance(steps,list) or len(steps)<=stage:
        raise ValueError('missing current search checkpoint')
    step=steps[stage]
    b=BUDGETS[stage]
    if step.get('root_visits')!=b or step.get('budget')!=b:
        raise ValueError('current root visit identity mismatch')
    legal=root.get('legal_columns_0based')
    if not isinstance(legal,list) or not legal or any(type(a)!=int or not 0<=a<7 for a in legal):
        raise ValueError('invalid gravity legal columns')
    rows=step.get('visit_policy')
    if not isinstance(rows,list) or not rows:
        raise ValueError('missing current root visit distribution')
    dense=[0.0]*7
    seen=set()
    for pair in rows:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:
            raise ValueError('malformed sparse root policy')
        a,v=pair
        if type(a)!=int or a not in legal or a in seen or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
            raise ValueError('invalid current-root visit probability')
        seen.add(a)
        dense[a]=float(v)
    z=sum(dense)
    if not math.isfinite(z) or abs(z-1)>1e-4:
        raise ValueError('current-root distribution must sum one')
    dense=[v/z for v in dense]
    vals=sorted(dense,reverse=True)
    entropy=-sum(p*math.log(p) for p in dense if p>0)/math.log(7)
    eval_calls=step.get('evaluator_calls')
    if type(eval_calls)!=int or not 0<=eval_calls<=4*(b+1):
        raise ValueError('invalid observed evaluator-call count')
    ply=root.get('ply')
    if ply is None:
        ply=len(root.get('history',''))
    if type(ply)!=int or not 0<=ply<42:
        raise ValueError('invalid root ply')
    side=root.get('side_to_move')
    if side not in (-1,1):
        raise ValueError('missing current-player sign')
    features=[1.0,b/64.0,entropy,vals[0],vals[0]-vals[1],
              len(legal)/7.0,eval_calls/(b+1),
              ply/42.0,1.0 if side==1 else 0.0]
    if len(features)!=9 or any(not math.isfinite(x) for x in features):
        raise ValueError('invalid/nonfinite observation features')
    return features


def classify_outcome(score:int)->int:
    if type(score)!=int or score==-1000 or not -21<=score<=21:
        raise ValueError('invalid/sentinel exact game-theoretic score')
    return int(score>0)-int(score<0)


def continuation_gain_labels(root:dict,exact_scores:list[int])->dict:
    if len(exact_scores)!=7:raise ValueError('exact Connect4 action vector requires seven columns')
    steps=root.get('steps',[])
    if len(steps)!=4 or [s['budget'] for s in steps]!=list(BUDGETS):
        raise ValueError('full resident trajectory required for training')
    actions=[s['best_action'] for s in steps]
    legal=root['legal_columns_0based']
    if any(type(a)!=int or a not in legal or exact_scores[a]==-1000 for a in actions):
        raise ValueError('MCTS action not legal in exact external oracle')
    score=[classify_outcome(exact_scores[a]) for a in actions]
    gains=[b-a for a,b in zip(score,score[1:])]
    cost=[s['step_wall_us']/1000.0 for s in steps[1:]]
    if any(not math.isfinite(c) or c<0 for c in cost):
        raise ValueError('negative/nonfinite observed CPU cost')
    return {'utility':score,'gains':gains,'next_wall_ms':cost,
            'features':[current_root_features(root,i) for i in range(3)]}


def fit_stagewise_ridge(rows:list[dict],alpha:float=RIDGE_ALPHA)->dict:
    if not rows or not math.isfinite(alpha) or alpha<=0:
        raise ValueError('positive ridge alpha and nonempty train samples required')
    coefficients=[]
    forecast=[]
    train_metrics=[]
    for stage in range(3):
        X=np.asarray([r['features'][stage] for r in rows],dtype=np.float64)
        y=np.asarray([r['gains'][stage] for r in rows],dtype=np.float64)
        cost=np.asarray([r['next_wall_ms'][stage] for r in rows],dtype=np.float64)
        if X.shape!=(len(rows),9) or not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(cost).all() or np.any(cost<0):
            raise ValueError('invalid supervised signed gain/cost matrix')
        ridge=np.eye(9,dtype=np.float64)*alpha
        ridge[0,0]=0.0  # the intercept is unregularized
        coef=np.linalg.solve(X.T@X+ridge,X.T@y)
        pred=X@coef
        coefficients.append(coef.tolist())
        mean_ms=float(np.mean(cost))
        forecast.append(max(mean_ms,1e-6))
        train_metrics.append({
            'stage_visits':BUDGETS[stage],
            'label_signed_counts':dict(Counter(str(x) for x in y.astype(int))),
            'mean_label_gain':float(y.mean()),
            'train_MSE':float(np.mean((pred-y)**2)),
            'cost_mean_ms':mean_ms,
        })
    return {'schema':SCHEMA,'features':FEATURES,
            'feature_order':default_features_schema(),
            'coefficients':coefficients,'forecast_cost_ms':forecast,
            'lambda_wdl_per_ms':0.0,'training_metrics':train_metrics,
            'alpha':alpha}


def policy_stop_budget(policy:dict,root:dict)->int:
    if policy.get('schema')!=SCHEMA or policy.get('features')!=FEATURES:
        raise ValueError('unknown COG02 frozen model schema')
    coefs=policy['coefficients']
    costs=policy['forecast_cost_ms']
    price=policy['lambda_wdl_per_ms']
    if len(coefs)!=3 or len(costs)!=3 or not 0<=price<=1 or any(not math.isfinite(c) or c<=0 for c in costs):
        raise ValueError('invalid stage costs/coefficient identity')
    for stage,budget in enumerate(STAGE_VISITS):
        x=current_root_features(root,stage)
        c=coefs[stage]
        if len(c)!=9 or any(not math.isfinite(v) for v in c):
            raise ValueError('invalid frozen signed-gain coefficients')
        forecast=sum(a*b for a,b in zip(c,x))
        if not math.isfinite(forecast):
            raise ValueError('nonfinite next-batch forecast')
        if forecast <= price*costs[stage]:
            return budget
    return 64


def observed_search_ms(root:dict,budget:int)->float:
    stages=root['steps']
    index=list(BUDGETS).index(budget)
    return sum(s['step_wall_us'] for s in stages[:index+1])/1000.0


def compact_frozen_sources(dir:Path,lock:dict)->dict:
    for item in lock['source_files']:
        p=Path(item['path'])
        # The Rust runtime files legitimately change during new COG02 work,
        # so exact frozen prior SOURCE identity is tracked in input lock
        # but only immutable DATA/MODEL/ORACLE files are strict at execution.
        if p.name in ('mcts_server.rs','connect4_nn.rs','connect4.rs','connect4_conv.rs'):
            continue
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise RuntimeError('frozen data/model/oracle file changed: '+str(p))
    return {'locked_data_model_count':len(lock['source_files'])-4}


def selected_cases(repo:Path,splits:list[str])->list[dict]:
    d=repo/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07'
    cohort=json.loads((d/'COHORT_1024_FROZEN.json').read_text())
    rows=[r for r in cohort['rows'] if r['split'] in splits]
    counts=Counter(r['split'] for r in rows)
    assert all(counts[s]==TRAIN_VAL_SETS[s] for s in splits)
    return rows


def model_identity(repo:Path)->dict[str,Path]:
    d=repo/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07'
    return {'new_mlp':d/'evidence/mlp_conv_train_20261121/mlp_model.json',
            'conv':d/'evidence/mlp_conv_train_20261121/conv_model.json'}


def rust_batch_requests(cases:list[dict],repo:Path)->list[dict]:
    models=model_identity(repo)
    req=[]
    for mode in MODES:
        for row in cases:
            q={'cmd':'research_connect4_cpu','history':row['history'],
               'budgets':list(BUDGETS),'seed':17,'cache_model':True}
            if mode!='uniform':
                q.update({'eval_mode':'tiny_mlp' if mode=='new_mlp' else 'conv3x3',
                          'model_path':str(models[mode].resolve()),
                          'model_identity':sha(models[mode])})
            req.append(q)
    return req


def batch_run(binary:Path,requests:list[dict],timeout_s:int)->list[dict]:
    lines='\n'.join(json.dumps(q,separators=(',',':'),sort_keys=True) for q in requests)
    lines+='\n'+json.dumps({'cmd':'quit'})+'\n'
    proc=subprocess.run([str(binary),'--server'],input=lines.encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout_s,check=False)
    if proc.returncode!=0:
        raise RuntimeError(f'MCTS server error code={proc.returncode}, stderr={proc.stderr[-1200:]!r}')
    rows=[json.loads(x) for x in proc.stdout.decode('utf-8','replace').splitlines() if x.strip()]
    if len(rows)!=len(requests):
        raise RuntimeError(f'server response count mismatch: {len(rows)} vs {len(requests)}')
    for q,r in zip(requests,rows):
        if 'error' in r or r.get('history')!=q['history'] or r.get('seed')!=17:
            raise RuntimeError('Rust root failed or source/response identity mismatch: '+str(r)[:350])
    return rows


def collect_trajectories(args,repo,d,lock):
    if git(repo,'status','--porcelain'):
        raise RuntimeError('collect only on clean committed source')
    compact_frozen_sources(d,lock)
    cases=selected_cases(repo,['train','val'])
    assert len(cases)==832
    models=model_identity(repo)
    binpath=args.binary.resolve()
    if not binpath.is_file():raise FileNotFoundError(binpath)
    req=rust_batch_requests(cases,repo)
    assert len(req)==2496
    started=time.perf_counter()
    raw=batch_run(binpath,req,timeout_s=160)
    elapsed=time.perf_counter()-started
    # Train/validation exact labels are read AFTER the entire MCTS batch.
    oracle=json.loads((d.parent/'connect4_generalization_07/evidence/teacher_1024_20261121/labels.json').read_text())
    teacher={r['id']:r for r in oracle['rows'] if r['split'] in TRAIN_VAL_SETS and r['status']=='OK'}
    assert len(teacher)==832
    joined=[]
    for j,mode in enumerate(MODES):
        for i,row in enumerate(cases):
            r=raw[j*len(cases)+i]
            if r.get('eval_mode')!=('uniform_no_neural_network' if mode=='uniform'
                else 'trained_tiny_mlp_cpu' if mode=='new_mlp' else 'trained_conv3x3_cpu'):
                raise RuntimeError('research evaluator role drift')
            if r['ply']!=row['plies'] or r['side_to_move']!=row['current_player']:
                raise RuntimeError('root board/parity invalid')
            if r['board_bottom_up']!=row['board']:
                raise RuntimeError('Rust gravity board differs from frozen teacher')
            if r['legal_columns_0based']!=row['legal_columns']:
                raise RuntimeError('legal-column mismatch')
            if r['fresh16']['resident_match'] is not True or len(r['steps'])!=4:
                raise RuntimeError('full-tree 16 revisit identity failed')
            if mode!='uniform' and i>0 and r['model_cache_hit'] is not True:
                raise RuntimeError('frozen model not cache-resident after first root')
            labels=continuation_gain_labels(r,teacher[row['id']]['exact_action_scores'])
            joined.append({
                'id':row['id'],'family':row['family'],'split':row['split'],
                'mode':mode,'history':row['history'],
                'root_hash':r['root_hash'],
                'features':labels['features'],
                'gains':labels['gains'],
                'utility':labels['utility'],
                'next_wall_ms':labels['next_wall_ms'],
                'search_stage_wall_ms':[s['step_wall_us']/1000.0 for s in r['steps']],
                'cumulative_evaluator_calls':[s['evaluator_calls'] for s in r['steps']],
                'initial_evaluator_calls':r['root_initial_evaluator_calls'],
                'model_load_wall_us':r['model_load_wall_us'],
                'root_setup_wall_us':r['root_setup_wall_us'],
                'model_cache_hit':r['model_cache_hit'],
            })
    for model,p in models.items():
        if sha(p)!=next(x['sha256'] for x in lock['source_files'] if x['path']==str(p)):
            raise RuntimeError('model weights changed during cost data acquisition')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    files=[('full_rust_trajectories.json',raw),('joined_train_val.json',joined)]
    for name,payload in files:
        (out/name).write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True)+'\n')
    meta={'schema':'quartz_cog02_training_trajectories_manifest_v1',
          'source_head':git(repo,'rev-parse','HEAD'),
          'source_tree':git(repo,'rev-parse','HEAD^{tree}'),
          'rust_binary_sha256':sha(binpath),'model_shas':{k:sha(v) for k,v in models.items()},
          'cohort_sha256':sha(d.parent/'connect4_generalization_07/COHORT_1024_FROZEN.json'),
          'oracle_teacher_sha256':sha(d.parent/'connect4_generalization_07/evidence/teacher_1024_20261121/labels.json'),
          'requests':len(req),'train_val_roots':len(cases),
          'mode_count':len(MODES),'elapsed_s':elapsed,
          'output_sha256':{p.name:sha(p) for p in out.iterdir()},
          'model_cache_enabled':True,
          'fresh16_control_cost_not_in_primary_stage_times':True,
          'oracle_only_joined_after_batch_search':True}
    (out/'execution_manifest.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    print('COLLECTED_COG02_TRAJECTORIES',json.dumps({
        'runs':len(raw),'rows':len(joined),'elapsed_s':elapsed,
        'train':sum(x['split']=='train' for x in joined),
        'val':sum(x['split']=='val' for x in joined),
        'output':str(out)},sort_keys=True))


def replay_choice(model:dict,row:dict)->tuple[int,float,float]:
    # Counterfactual valid only because the actual full tree is the same
    # deterministic resident prefix. It is not an online test result.
    pseudo={'history':row['history'],
            'ply':len(row['history']),
            'side_to_move':1 if len(row['history'])%2==0 else -1,
            'legal_columns_0based':[i for i in range(7)],
            'steps':[{'budget':BUDGETS[i],'root_visits':BUDGETS[i],
                       'visit_policy':None,'evaluator_calls':row['cumulative_evaluator_calls'][i]}
                      for i in range(4)]}
    # Use saved actual *current* features only, not future action labels.
    coef=model['coefficients'];price=model['lambda_wdl_per_ms'];cost=model['forecast_cost_ms']
    chosen=3
    for stage in range(3):
        features=row['features'][stage]
        predicted=sum(x*y for x,y in zip(coef[stage],features))
        if predicted <= price*cost[stage]:
            chosen=stage;break
    realized_search_ms=sum(row['search_stage_wall_ms'][:chosen+1])
    return chosen,row['utility'][chosen],realized_search_ms


def evaluate_policy_rows(model:dict,rows:list[dict])->dict:
    if not rows:raise ValueError('cannot evaluate no held-out training/validation rows')
    results=[replay_choice(model,r) for r in rows]
    n=len(results)
    return {'n':n,'mean_U':sum(x[1] for x in results)/n,
            'mean_search_ms':sum(x[2] for x in results)/n,
            'mean_visits':sum(BUDGETS[x[0]] for x in results)/n,
            'stop_budgets':dict(Counter(str(BUDGETS[x[0]]) for x in results)),
            'mean_cost_adjusted_objective':sum(x[1]-VALIDATION_COST_PRICE*x[2] for x in results)/n}


def fit_from_trajectories(args,repo,d,lock):
    if git(repo,'status','--porcelain'):
        raise RuntimeError('freeze source before fitting')
    prior=args.trajectories.resolve()
    m=json.loads((prior/'execution_manifest.json').read_text())
    if m['requests']!=2496 or m['train_val_roots']!=832:
        raise RuntimeError('source training corpus cardinality mismatch')
    if sha(prior/'joined_train_val.json')!=m['output_sha256']['joined_train_val.json']:
        raise RuntimeError('training source bytes changed')
    data=json.loads((prior/'joined_train_val.json').read_text())
    if len(data)!=2496:raise RuntimeError('train/validation paired roots lost')
    result={}
    for mode in MODES:
        train=[x for x in data if x['mode']==mode and x['split']=='train']
        val=[x for x in data if x['mode']==mode and x['split']=='val']
        if len(train)!=640 or len(val)!=192:
            raise RuntimeError('source group split role mismatch')
        assert not ({r['family'] for r in train} & {r['family'] for r in val})
        fitted=fit_stagewise_ridge(train)
        choices=[]
        for lam in LAMBDAS:
            candidate={**fitted,'lambda_wdl_per_ms':lam}
            observed=evaluate_policy_rows(candidate,val)
            choices.append({'lambda':lam,'validation':observed})
        # Precommitted lexicographic criterion: utility minus time price;
        # ties choose lower mean time and larger lambda.
        selected=max(choices,key=lambda x:(
            x['validation']['mean_cost_adjusted_objective'],
            -x['validation']['mean_search_ms'],x['lambda']))
        candidate={**fitted,'lambda_wdl_per_ms':selected['lambda']}
        result[mode]={
            'schema':SCHEMA,'features':FEATURES,
            'coefficients':candidate['coefficients'],
            'forecast_cost_ms':candidate['forecast_cost_ms'],
            'lambda_wdl_per_ms':selected['lambda'],
            'source_train_roots':640,'source_val_roots':192,
            'validation_threshold_table':choices,
            'selected_validation_policy':selected['validation'],
            'train_gain_distribution':fitted['training_metrics'],
        }
    output={'schema':'quartz_cog02_costed_signed_gain_models_v1',
            'source_commit':git(repo,'rev-parse','HEAD'),
            'fixed_cost_price_WDL_per_ms':VALIDATION_COST_PRICE,
            'fixed_candidate_lambdas':list(LAMBDAS),
            'ridge_alpha':RIDGE_ALPHA,
            'family_split':'train40/validation12/test12, never test labels before fitted coefficients sealed',
            'mode_policies':result,
            'primary_estimand':'supervised signed realized next-batch WDL gain, not Bayesian KG',
            'test_set_used_for_selection':False}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dest=out/'frozen_surrogates.json'
    dest.write_text(json.dumps(output,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    receipt={'schema':'quartz_cog02_train_only_surrogate_manifest_v1',
             'source_commit':output['source_commit'],
             'trajectories_input_SHA256':sha(prior/'joined_train_val.json'),
             'surrogates_SHA256':sha(dest),
             'model_ids':list(MODES),
             'lambda_picked_on_validation_only':True,
             'oracle_test_data_accessed':False}
    (out/'surrogate_manifest.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('G02_FITTED_VALIDATED_SURROGATES',json.dumps({
        mode:{'lambda':r['lambda_wdl_per_ms'],
              'val_objective':r['selected_validation_policy']['mean_cost_adjusted_objective'],
              'stop_budget_counts':r['selected_validation_policy']['stop_budgets']}
        for mode,r in result.items()},sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--trajectories',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    d=Path(__file__).resolve().parent
    repo=next(p for p in d.parents if (p/'.git').exists())
    lock=json.loads((d/'INPUT_LOCK.json').read_text())
    if lock['schema']!='quartz_cog02_cost_matched_voc08_input_lock_v1':
        raise RuntimeError('bad source input contract')
    if args.collect == args.fit:
        parser.error('select exactly one of --collect or --fit')
    if args.collect:
        if args.binary is None:parser.error('--collect requires --binary')
        collect_trajectories(args,repo,d,lock)
    else:
        if args.trajectories is None:parser.error('--fit requires --trajectories')
        fit_from_trajectories(args,repo,d,lock)
