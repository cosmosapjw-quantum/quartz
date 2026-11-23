#!/usr/bin/env python3
"""Frozen C29 actual TicTacToe rollout-channel and native retrospective campaign.

The CLI only gives game history, cue seed, C26 public predictor and TRAIN-fitted
channel to the C29 binary. Complete minimax utility stays in this offline verifier.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,importlib.util,json,gzip,math,os,statistics,subprocess,time
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[4]
DOC=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/research/quartz_general_game_metareasoning_20261123/coding/data'
C28=ROOT/'docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py'
DEFAULT_BINARY=Path('/home/cosmosapjw/quantum/research/quartz_c29_target_20261123/release/quartz_c29_noisy_query')
SOURCE_SHA='9c9f52a0344a4b6c7c25cde6d519479784a0695fcb8f4681486403fb21fe15c1'
MODEL_SHA='0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126'
MASK64=(1<<64)-1
NOISE_GAMMA=0xbf58476d1ce4e5b9

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    import sys;sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod
C28MOD=import_file('quartz_c29_pinned_c28_campaign',C28)
ORACLE=import_file('quartz_c29_independent_frac',DOC/'theory/noisy_exact.py')

def inputs():
    if sha(DATA/'public_sources.jsonl')!=SOURCE_SHA or sha(DATA/'model.json')!=MODEL_SHA:raise RuntimeError('C26 inputs changed')
    sources=[json.loads(line) for line in (DATA/'public_sources.jsonl').read_text().splitlines()]
    assert len(sources)==300 and len({r['id'] for r in sources})==300
    return sources,json.loads((DATA/'model.json').read_text())

def low_cost_rollout(source,move,round_index=0):
    board,rootp=C28MOD.replay(source['history'])
    child=C28MOD.play(board,move,rootp)
    assert not C28MOD.terminal(child)
    seed=source['cue_seed'] ^ (((move+1)*C28MOD.GAMMA)&MASK64) ^ (((round_index+101)*NOISE_GAMMA)&MASK64)
    rng=C28MOD.SplitMix64(seed)
    state=child; turn=-rootp;nodes=1
    while not C28MOD.terminal(state):
        legal=[i for i,v in enumerate(state) if v==0]
        state=C28MOD.play(state,legal[rng.index(len(legal))],turn)
        turn=-turn;nodes+=1
    observed=C28MOD.winner(state)*rootp+1
    return observed,nodes

def store(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def calibrate(output):
    sources,model=inputs()
    # Frozen index split before seeing any new response; the source is retrospective.
    counts=[[0]*3 for _ in range(3)]
    train=0
    start=time.perf_counter()
    for root in sources[:180]:
        for c in C28MOD.game_candidates(root['history'],root['cue_seed'],model):
            if c['terminal_known']:continue
            y,nodes=low_cost_rollout(root,c['move'],0)
            u=next(i for i,x in enumerate((F(0),F(1,2),F(1))) if x==c['utility'])
            counts[u][y]+=1;train+=1
    # 1-count row pseudocount, fixed in C04 to prevent unsupported observation.
    channels=[[float(F(v+1,sum(row)+3)) for v in row] for row in counts]
    assert train>0 and all(abs(sum(row)-1)<1e-12 for row in channels)
    result={'schema':'quartz-c29-observation-channel-v1','status':'TRAIN_FITTED_RETROSPECTIVE','training_roots':180,'cal_roots':60,'diagnostic_roots':60,
        'training_events':train,'training_counts_true_u_by_rollout_y':counts,'laplace_pseudocount':1,
        'channel':channels,'source_sha256':SOURCE_SHA,'model_sha256':MODEL_SHA,
        'calibration_assumption':'global C(Y_rollout|U_exact), independent draws conditional on same U; not physically guaranteed conditional on U alone',
        'seed_policy':'cue_seed XOR (move+1)*GAMMA XOR (round+101)*NOISE_GAMMA; SplitMix64',
        'same_source_already_observed':True,'new_TEST':False,'duration_seconds':time.perf_counter()-start}
    store(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('channel','training_counts_true_u_by_rollout_y')},indent=2))
    print('fit channel',channels)

def game_request(source,model,channel,repeats=0):
    return {'mode':'game','id':source['id'],'history':source['history'],'cue_seed':source['cue_seed'],
        'model':model,'channel':channel,'rho':0.75,'horizon':2,'repeats':repeats}

def run_native(reqs,binary,output):
    if not binary.exists():raise RuntimeError('native binary missing')
    raw='\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in reqs)+'\n'
    start=time.perf_counter()
    p=subprocess.run([str(binary)],input=raw,text=True,capture_output=True,timeout=180)
    if p.returncode or p.stderr or len(p.stdout.splitlines())!=len(reqs):
        raise RuntimeError(f'native error {p.returncode}; stderr {p.stderr[-1000:]}; lines {len(p.stdout.splitlines())}/{len(reqs)}')
    rows=[json.loads(line) for line in p.stdout.splitlines()]
    for req,res in zip(reqs,rows):
        assert res.get('id')==req['id'] and res.get('mode')=='game' and res.get('schema')=='c29-noisy-v1' and 'error' not in res,res
        assert res['offline_oracle_not_runtime_input'] is True
    output.mkdir(parents=True,exist_ok=False)
    for name,lines in [('requests',raw),('responses',p.stdout)]:
        with gzip.open(output/(name+'.jsonl.gz'),'wt',encoding='utf-8') as f:f.write(lines)
    return rows,{'wall_seconds':time.perf_counter()-start,'binary_sha256':sha(binary),
        'raw_requests_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_responses_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),
        'request_count':len(reqs),'native_exit_code':p.returncode,'native_stderr_bytes':len(p.stderr)}

def verify_game(source,result,model,channel):
    board,rootp=C28MOD.replay(source['history'])
    legal=[i for i,v in enumerate(board) if v==0]
    assert result['baseline_move'] in legal and result['noisy_move'] in legal
    candidates=C28MOD.game_candidates(source['history'],source['cue_seed'],model)
    moves={c['move']:c for c in candidates}
    assert result['cue_nodes']==sum(c['cue_nodes'] for c in candidates)
    for key,choice in [('baseline',result['baseline_move']),('noisy',result['noisy_move'])]:
        expected=float(moves[choice]['utility'])
        assert result['offline_realized_'+key]==expected
    assert result['query_count']==len(result['observations'])<=2
    assert result['baseline_charged_nodes']==result['cue_nodes']+result['initial_nodes']
    assert result['noisy_charged_nodes']==result['baseline_charged_nodes']+result['additional_rollout_nodes']
    total=0
    for index,trace in enumerate(result['observations']):
        assert trace['move_id'] in moves
        assert not moves[trace['move_id']]['terminal_known']
        expected_y,n=low_cost_rollout(source,trace['move_id'],index)
        assert trace['y']==expected_y and trace['nodes']==n
        total+=n
    assert total==result['additional_rollout_nodes']
    assert abs(result['actual_query_cost']-total/1024)<1e-12
    if result['query_count']==0:assert result['baseline_move']==result['noisy_move']
    # Reconstruct posterior and optimal deployment independently of native Rust.
    ps=[];rs=[];costs=[];moves_sorted=[]
    for candidate in candidates:
        known=candidate['terminal_known'] or candidate['move']==result['initial_known_move']
        prob=tuple(F(int(x==candidate['utility'])) for x in (F(0),F(1,2),F(1))) if known else candidate['p']
        ps.append(prob);rs.append(F(0) if known else F(3,4))
        costs.append(F(0) if known else F(candidate['cue_nodes'],4*1024))
        moves_sorted.append(candidate['move'])
    matrix=tuple(tuple(F(str(v))/sum(F(str(x)) for x in row) for v in row) for row in channel)
    specification=ORACLE.model(ps,rs,[matrix]*len(ps),costs,b=0,m=0)
    state=ORACLE.Factor(*ORACLE.components(specification))
    initial_means=state.posterior_means()
    max_initial=max(initial_means)
    expected_baseline=moves_sorted[next(i for i,v in enumerate(initial_means) if v>=max_initial-F(1,10**12))]
    assert result['baseline_move']==expected_baseline
    assert abs(result['baseline_expected']-float(max_initial))<1e-10
    for trace in result['observations']:
        i=moves_sorted.index(trace['move_id'])
        next_state,prob_y=state.update(specification,i,trace['y'])
        assert prob_y>0
        state=next_state
        assert abs(float(state.q[1])-trace['q_plus'])<1e-10
        assert abs(float(state.posterior_means()[i])-trace['expected_after'])<1e-10
    means=state.posterior_means()
    maximum=max(means)
    oracle_move=moves_sorted[next(i for i,v in enumerate(means) if v>=maximum-F(1,10**12))]
    assert result['noisy_move']==oracle_move,(result['id'],result['noisy_move'],oracle_move)
    assert abs(float(maximum)-result['noisy_expected'])<1e-10
    return True

def quantiles(values):
    return {'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}

def run_campaign(mode,output,channel_file,binary):
    sources,model=inputs()
    channel_manifest=json.loads(Path(channel_file).read_text())
    assert channel_manifest['status']=='TRAIN_FITTED_RETROSPECTIVE'
    channel=channel_manifest['channel']
    if mode=='validate':selection=sources[180:240];repeats=0
    elif mode=='benchmark':selection=sources;repeats=7
    else:raise ValueError(mode)
    reqs=[game_request(s,model,channel,repeats=repeats) for s in selection]
    rows,meta=run_native(reqs,binary,Path(output))
    observed=0;count_queries=0;changes=0;new_positive=0
    for src,row in zip(selection,rows):
        verify_game(src,row,model,channel)
        observed+=len(row['observations']);count_queries+=row['query_count']
        changes+=int(row['baseline_move']!=row['noisy_move'])
        new_positive+=int(row['offline_realized_noisy']>row['offline_realized_baseline'])
    dx=[a['offline_realized_noisy']-a['offline_realized_baseline'] for a in rows]
    prices=[a['actual_query_cost'] for a in rows]
    net=[d-c for d,c in zip(dx,prices)]
    cost=[a['additional_rollout_nodes'] for a in rows]
    results={'scope':mode,'metadata':meta,'index_range':[180,240] if mode=='validate' else [0,300],
        'source_sha256':SOURCE_SHA,'channel_sha256':sha(channel_file),'channel_status':channel_manifest['status'],
        'retrospective':True,'fresh_test':False,'query_count_total':count_queries,'query_decision_count':observed,'action_change_roots':changes,
        'realized_utility_improvement_count':new_positive,'utility_delta':quantiles(dx),'actual_query_cost':quantiles(prices),
        'net_delta_utility_minus_query_price':quantiles(net),'additional_rollout_nodes':quantiles(cost),
        'benchmark_boundary':'one Rust invocation per timed replay: cues+initial exact solve+planner+paid rollout+final selection; excludes input JSON/serialization/offline minimax',
        'notes':'Calibration channel learned globally from first 180 old roots, selection is conditioned on observed cues and may be miscalibrated; no matched-total-time strength claim'}
    if mode=='benchmark':
        btime=[statistics.median(a['timing_baseline_ns']) for a in rows]
        ntime=[statistics.median(a['timing_noisy_ns']) for a in rows]
        results['time_baseline_ns']=quantiles(btime);results['time_noisy_ns']=quantiles(ntime)
        results['paired_mean_noisy_minus_baseline_ns']=statistics.mean(a-b for a,b in zip(ntime,btime))
        results['ratio_means']=statistics.mean(ntime)/statistics.mean(btime)
        results['timing_pairs']=7*len(rows)
        results['group_CAL_60']={'utility_net_mean':statistics.mean(net[180:240]),'changed_actions':sum(a['baseline_move']!=a['noisy_move'] for a in rows[180:240])}
    store(Path(output)/'summary.json',results)
    print(json.dumps({k:v for k,v in results.items() if k not in ('notes','metadata')},indent=2))
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['fit','validate','benchmark'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--channel',type=Path)
    ap.add_argument('--binary',type=Path,default=DEFAULT_BINARY)
    args=ap.parse_args()
    if args.mode=='fit':calibrate(args.output)
    else:
        if args.channel is None:ap.error('--channel required')
        run_campaign(args.mode,args.output,args.channel,args.binary)
