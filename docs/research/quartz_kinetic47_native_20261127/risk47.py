"""Separate offline proxy: only TRAIN bin probabilities, CAL admission, TEST untouched.
No games are generated here. Fixed child PUCT values are not exact WDL.
"""
import gzip,json,collections as C,sys,time,random
from concurrent.futures import ProcessPoolExecutor
from k47_plan import D,T,ROOT,load,save,sha
K45=D.parent/'quartz_kinetic45_sequential_20261127'
K46=D.parent/'quartz_kinetic46_confirm_20261127'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows():
    for mode in ('TRAIN','NN_CAL','NN_TEST'):
        lk=json.loads((K45/(mode+'_LOCK.json')).read_text())
        for f in lk['files']:
            p=K45/f['file'];assert sha(p)==f['sha256']
            for r in raw(p):
                if mode=='TRAIN':yield 'TRAIN',r['seed'],r['job'],r
                else:
                    for a in r['atlas']:yield mode[3:],r['seed'],r['job']['id'],a
def jobs():
    ct=C.Counter();out=[]
    for split,seed,job,r in rows():
        ct[split]+=1
        if r['proof']['cert'] is not None:continue
        a,b=r['options']['20']['move'],r['options']['32']['move'];bd={m:(l,u) for m,l,u in r['proof']['bounds']};l=r['L']['20']
        assert l['kind']!='CERTIFICATE' and a in bd and b in bd
        z=int(l['distance']<=3)+2*int(l['move']!=l['prior_best'])+4*int(any(u==-1 for _,u in bd.values()))
        out.append(dict(split=split,seed=seed,job=job,history=r['history'],actions=[a,b],bounds=[list(bd[a]),list(bd[b])],z=z,low=dict(distance=l['distance'],move=l['move'],prior=l['prior_best'],loss_count=sum(u==-1 for _,u in bd.values()),replacement=r['options']['20']['replacement']),archive_nn=[r['options'][str(k)]['nn'] for k in (20,32)]))
    assert dict(ct)=={'TRAIN':1709,'CAL':785,'TEST':748},ct
    return out,dict(ct)
def value(ev,h,move):
    from k46_engine import Node,from_history
    a,b=from_history(h+[move]);p=1-2*((len(h)+1)%2)
    t=time.perf_counter_ns();c=time.process_time_ns();nn=ev.rows
    root=Node(a,b,p);v,term=root.expand(ev);terminal=0
    if term:return dict(raw=-v,nn=0,visits=0,terminal=True,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
    for _ in range(32):
        node=root;path=[]
        while node.legal is not None:
            i=node.select();path.append((node,i));node=node.descend(i)
        v,term=node.expand(ev);terminal+=term
        for parent,i in reversed(path):
            v=-v;parent.n[i]+=1;parent.w[i]+=v
    i=int(root.n.argmax());assert root.n[i]>0 and int(root.n.sum())==32 and ev.rows-nn==33-terminal
    return dict(raw=-float(root.w[i]/root.n[i]),nn=ev.rows-nn,visits=32,terminal=False,terminal_leaves=int(terminal),leader=int(root.legal[i]),leader_n=int(root.n[i]),leader_w=float(root.w[i]),cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def worker(args):
    i,rs=args;sys.path.insert(0,str(K46));from k46_engine import load_eval
    evs={};out=[];ct=C.Counter()
    for j in rs:
        a,b=j['actions'];bd=j['bounds'];vals=[];teacher=[]
        if a==b:
            gain=0.;teacher=[None,None];vals=None
        else:
            for m,(l,u) in zip((a,b),bd):
                if l==u:v=float(l);t=None
                else:
                    if j['seed'] not in evs:evs[j['seed']]=load_eval(j['seed'])
                    t=value(evs[j['seed']],j['history'],m);v=max(l,min(u,t['raw']))
                    ct['teacher_calls']+=1;ct['nn']+=t['nn'];ct['cpu_ns']+=t['cpu_ns']
                vals.append(v);teacher.append(t)
            gain=(vals[0]-vals[1])/2
            assert (bd[0][0]-bd[1][1])/2<=gain<=(bd[0][1]-bd[1][0])/2
        out.append(dict(**j,values=vals,teacher=teacher,gain=gain,label=int(gain<0)));ct['roots']+=1;ct['negative']+=gain<0
    assert sum(e.rows for e in evs.values())==ct['nn']
    print('RISK_SHARD',i,dict(ct),flush=True)
    return out,dict(ct)
def main():
    js,source=jobs()
    # Group by seed; no model mixing within a pair. Each row is a separate archived observation.
    tasks=[(i,js[i::8]) for i in range(8)]
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=8,mp_context=get_context('spawn')) as ex:res=list(ex.map(worker,tasks))
    ct=sum((C.Counter(c) for _,c in res),C.Counter());allrows=[r for rs,c in res for r in rs]
    path=T/'RISK.jsonl.gz';assert not path.exists()
    with gzip.open(path,'wt',compresslevel=3) as f:
        for r in allrows:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    train=[r for r in allrows if r['split']=='TRAIN'];n=len(train);k=sum(r['label'] for r in train)
    base=(k+1)/(n+2);pr={}
    for z in range(8):
        rr=[r for r in train if r['z']==z];pr[str(z)]=(sum(r['label'] for r in rr)+1)/(len(rr)+2)
    out={}
    for split in ('TRAIN','CAL','TEST'):
        rr=[r for r in allrows if r['split']==split];n=len(rr)
        t=C.Counter();bins={}
        for z in range(8):
            r=[x for x in rr if x['z']==z]
            bins[str(z)]=dict(n=len(r),negative=sum(x['label'] for x in r),p=pr[str(z)],rate=sum(x['label'] for x in r)/len(r) if r else None,mean_gain=sum(x['gain'] for x in r)/len(r) if r else None)
        for r in rr:
            t['different']+=r['actions'][0]!=r['actions'][1];t['teacher_calls']+=sum(x is not None for x in r['teacher']);t['nn']+=sum(x['nn'] for x in r['teacher'] if x);t['negative']+=r['label']
        out[split]=dict(n=n,counts=dict(t),mean_gain=sum(r['gain'] for r in rr)/n,brier=sum((pr[str(r['z'])]-r['label'])**2 for r in rr)/n,constant_brier=sum((base-r['label'])**2 for r in rr)/n,bins=bins)
    admit=out['CAL']['brier']<=out['CAL']['constant_brier']
    s=dict(source_counts=source,counts=dict(ct),probabilities=pr,constant=base,cal_admitted=admit,results=out,scope='archived K45 staged hybrid Q20 vs Q32; finite-budget proof-clipped 32 child PUCT visit-leader values, not exact minimax, no games or live policy',raw_sha256=sha(path),source_locks={m:sha(K45/(m+'_LOCK.json')) for m in ('TRAIN','NN_CAL','NN_TEST')})
    (T/'RISK_SUMMARY.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print('RISK',json.dumps(s,separators=(',',':')),flush=True)
if __name__=='__main__':main()
