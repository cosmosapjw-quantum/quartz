"""Train on exact signed gains; execute heldout arms prospectively with genuine live resident sessions."""
import json,math,random,statistics as st,time,sys,hashlib,gzip
from collections import defaultdict
from pathlib import Path
from scheduler import features,fit,score,signed_order
from exact import labels
from population import D,key,snapshot
from engine import Engine,MODELS,BIN,sha
SEED=20261124081
def dump(name,x):
    p=D/name
    assert not p.exists(),p
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sources(split):
    return [json.loads(l) for l in (D/'sources.jsonl').read_text().splitlines() if json.loads(l)['split']==split]
def legal(s):return [i for i,x in enumerate(s['board']) if x==0]
def warm(e,s):
    sid,old=e.open(s)
    try:return old,e.step(s,sid,old)
    finally:e.close(sid)
def train():
    ss=sources('TRAIN');eng=Engine('gomoku7');rr=[]
    try:
        warm(eng,ss[0])
        for s in ss:
            sid,p=eng.open(s)
            try:q=eng.step(s,sid,p)
            finally:eng.close(sid)
            # exact labels are TRAIN-only and never passed into engine.
            u=labels(s)
            rr.append(dict(id=s['id'],ply=s['ply'],prefix=p,final=q,labels=u,x=features(p,legal(s)),gain=u[str(q['best_move'])]-u[str(p['best_move'])]))
    finally:eng.stop()
    assert len(rr)==24 and all(r['prefix']['actual_nn_rows']>0 for r in rr)
    m=fit([r['x'] for r in rr],[r['gain'] for r in rr])
    costs=sorted(r['final']['wall_ns'] for r in rr)
    m.update(median_chunk_ns=int(st.median(costs)),q75_chunk_ns=costs[math.ceil(.75*len(costs))-1],K=12)
    m.update(optional_budget_ns=12*m['q75_chunk_ns'],source_sha256=sha(D/'sources.jsonl'),model_sha256=sha(MODELS['gomoku7']),binary_sha256=sha(BIN),campaign_sha256=sha(Path(__file__)),scheduler_sha256=sha(D/'scheduler.py'),engine_sha256=sha(D/'engine.py'),exact_sha256=sha(D/'exact.py'),train_ids=[r['id'] for r in rr])
    dump('C07_TRAIN.json',rr);dump('C07_MODEL.json',m)
    print('TRAIN',json.dumps({'n':len(rr),'gain_counts':{str(z):sum(r['gain']==z for r in rr) for z in [-1,-.5,0,.5,1]},'median_chunk_ms':m['median_chunk_ns']/1e6,'budget_ms':m['optional_budget_ns']/1e6,'model':m},indent=2),flush=True)
def arm(eng,ss,name,m):
    t=time.perf_counter_ns();base=[];sids=[];results=[]
    try:
        for s in ss:
            sid,row=eng.open(s);sids.append(sid);base.append(row)
        prefix_ns=time.perf_counter_ns()-t
        x=[features(row,legal(s)) for row,s in zip(base,ss)]
        pred=[score(m,z) for z in x]
        ids=[s['id'] for s in ss]
        if name=='RANDOM':
            order=list(range(len(ss)));random.Random(SEED).shuffle(order)
        elif name=='ENTROPY':order=signed_order([z[0] for z in x],ids)
        elif name=='SIGNED':order=signed_order(pred,ids)
        elif name=='PREFIX_ONLY':order=[]
        else:raise ValueError(name)
        start_optional=time.perf_counter_ns();deadline=t+prefix_ns+m['optional_budget_ns'];selected={}
        for i in order:
            if len(selected)>=m['K'] or deadline-time.perf_counter_ns()<m['median_chunk_ns']:break
            selected[i]=eng.step(ss[i],sids[i],base[i])
        finish_optional=time.perf_counter_ns()
        for i,s in enumerate(ss):
            results.append(dict(id=s['id'],prefix=base[i],final=selected.get(i,base[i]),selected=i in selected,x=x[i],predicted_gain=pred[i]))
    finally:
        for sid in sids:eng.close(sid)
    total=time.perf_counter_ns()-t
    return dict(arm=name,total_ns=total,prefix_ns=prefix_ns,optional_ns=finish_optional-start_optional,optional_deadline_overshoot_ns=max(0,finish_optional-deadline),selected_count=len(selected),order=[ids[i] for i in order],rows=results)
def heldout():
    m=json.loads((D/'C07_MODEL.json').read_text())
    for k,path in [('source_sha256',D/'sources.jsonl'),('model_sha256',MODELS['gomoku7']),('binary_sha256',BIN),('campaign_sha256',Path(__file__)),('scheduler_sha256',D/'scheduler.py'),('engine_sha256',D/'engine.py'),('exact_sha256',D/'exact.py')]:
        assert m[k]==sha(path),k
    ss=sources('HELDOUT')
    assert len(ss)==24 and not set(m['train_ids'])&{s['id'] for s in ss}
    eng=Engine('gomoku7');arms=[]
    try:
        # Warm on TRAIN, no heldout exact labels are evaluated before raw live results.
        warm(eng,sources('TRAIN')[0])
        for name in ['PREFIX_ONLY','RANDOM','ENTROPY','SIGNED']:
            result=arm(eng,ss,name,m);arms.append(result)
            print('LIVE',name,result['selected_count'],result['total_ns']/1e6,flush=True)
    finally:eng.stop()
    dump('C08_LIVE.json',arms)
def score_heldout():
    ss=sources('HELDOUT');m=json.loads((D/'C07_MODEL.json').read_text())
    raw=json.loads((D/'C08_LIVE.json').read_text())
    # Separate offline exact evaluation only after immutable raw policy results are produced.
    u={s['id']:labels(s) for s in ss}
    metrics={}
    for a in raw:
        rr=[]
        for r in a['rows']:
            lab=u[r['id']];p=lab[str(r['prefix']['best_move'])];f=lab[str(r['final']['best_move'])]
            rr.append(dict(id=r['id'],initial=p,utility=f,gain=f-p,regret=max(lab.values())-f,selected=r['selected'],changed=r['prefix']['best_move']!=r['final']['best_move'],predicted_gain=r['predicted_gain']))
        metrics[a['arm']]=dict(total_ms=a['total_ns']/1e6,prefix_ms=a['prefix_ns']/1e6,optional_ms=a['optional_ns']/1e6,overshoot_ms=a['optional_deadline_overshoot_ns']/1e6,selected=a['selected_count'],initial=st.mean(r['initial'] for r in rr),utility=st.mean(r['utility'] for r in rr),regret=st.mean(r['regret'] for r in rr),gain=st.mean(r['gain'] for r in rr),improved=sum(r['gain']>0 for r in rr),degraded=sum(r['gain']<0 for r in rr),changed=sum(r['changed'] for r in rr),nn_rows=sum(r['prefix']['actual_nn_rows']+(r['final']['actual_nn_rows'] if r['selected'] else 0) for r in a['rows']),native_evals=sum(r['final']['evaluator_calls'] for r in a['rows']),rows=rr)
    rng=random.Random(SEED);random_u={r['id']:r['utility'] for r in metrics['RANDOM']['rows']}
    diff=[r['utility']-random_u[r['id']] for r in metrics['SIGNED']['rows']]
    boot=sorted(st.mean(rng.choices(diff,k=len(diff))) for _ in range(2000))
    base=metrics['RANDOM']['total_ms']
    checks={a:abs(metrics[a]['total_ms']/base-1) for a in ['ENTROPY','SIGNED']}
    out=dict(metrics=metrics,primary_signed_minus_random=st.mean(diff),paired_bootstrap_descriptive=[boot[50],boot[1949]],time_mismatch_relative=checks,time_gate=all(v<=.1 for v in checks.values()),signed_minus_prefix_only=metrics['SIGNED']['utility']-metrics['PREFIX_ONLY']['utility'],n=24,labels=u,raw_sha256=sha(D/'C08_LIVE.json'),source_scope='constructed late 7x7, both plies odd (white to move); no general game population claim')
    dump('C08_SUMMARY.json',out)
    print(json.dumps({**{k:v for k,v in out.items() if k not in ['metrics','labels']},'metrics':{a:{k:v for k,v in z.items() if k!='rows'} for a,z in metrics.items()}},indent=2),flush=True)
def smoke15():
    eng=Engine('gomoku15');rr=[]
    try:
        ss=sources('SMOKE')
        warm(eng,ss[0])
        for s in ss:
            p,q=warm(eng,s)
            rr.append(dict(id=s['id'],ply=s['ply'],prefix=p,final=q))
    finally:eng.stop()
    dump('C08_GOMOKU15_SMOKE.json',rr)
    print('GOMOKU15 SMOKE',len(rr),'actual_nn_rows',sum(r['prefix']['actual_nn_rows']+r['final']['actual_nn_rows'] for r in rr),flush=True)
if __name__=='__main__':
    {'train':train,'heldout':heldout,'score':score_heldout,'smoke15':smoke15}[sys.argv[1]]()
