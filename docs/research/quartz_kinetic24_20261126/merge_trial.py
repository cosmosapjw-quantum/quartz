"""Actual deterministic board identity and MERGE trial."""
from source_gen import D,ROOT,R,BASE,SEED,PRICE,sha,read,save,raw,emit,ident,k20,p,old,BIN
import math,random,itertools,statistics as S,collections as C,gzip,sys
ARMS=('K24N','K24M','K24D')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def check(x,r,base=False):
    assert 'error' not in x,x
    assert x['history']==r['history'] and x['legal']==r['legal'] and x['best'] in r['legal']
    assert x['proof']==k20.proof(r['history'])
    a,b=x['prefix'],x['result'];assert a['root_n']==32 and b['root_n']==x['visits']<=512
    assert b['evaluator_calls']==x['NN_calls'] and b['evaluator_calls']-a['evaluator_calls']==x['incremental_nn']<=64
    cert=k20.cert(x['proof']);assert x['best']==(b['best'] if cert is None else cert)
    assert x['cause']==('CERTIFIED' if cert is not None else 'NN_TARGET' if x['incremental_nn']==64 else 'VISIT_CAP')
    if x['cause']=='VISIT_CAP':assert x['visits']==512
    for s in (a,b):assert sum(e['n'] for e in s['edges'])==s['root_n'] and all(abs(e['w'])<=e['n']+1e-6 and e['vl']==0 for e in s['edges'])
    if base:assert x['trace'] is None and x['counts'] is None;return
    assert len(x['trace'])==x['visits'] and x['counts']['iterations']==x['visits']
    assert x['prefix_counts']['iterations']==32
    assert x['counts']['actual_node_ids']==x['counts']['children']
    n=sorted(e['n'] for e in a['edges']);assert x['bin']==2*int(x['prefix_counts']['shared']>0)+int(n[-1]-n[-2]>=4)
    for i,t in enumerate(x['trace']):
        assert t['mode']==(0 if i<32 else x['mode']) and len(t['path'])==len(t['steps']) and len(t['path'])>0
        for s in t['steps']:
            assert 0<=s['novel_n']<=s['unique_n']<=s['raw_n']
            assert (s['raw_mean'] is not None)==(s['raw_n']>0) and (s['novel_mean'] is not None)==(s['novel_n']>0)
            if t['terminal'] or t['mode']==0:close(s['backup'],-s['incoming'])
def order(i):
    a=list(range(3));random.Random(SEED+900000+i).shuffle(a);return a
def trial(part):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    if part=='TEST':
        f=read('FROZEN.json');assert f['code']==sha(D/'merge_trial.py') and f['cohort']==sha(D/'COHORT.json') and f['identity']==ident()
    path=D/(part+'_ATLAS.jsonl.gz');assert not path.exists()
    cnt=vis=nn=0;caps=C.Counter();parity=0
    with k20.Server(BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            if (r['group']=='TEST')!=(part=='TEST'):continue
            seq=order(r['id']);arms=[None]*3
            for i in seq:
                x=srv.request(r['history'],ARMS[i]);check(x,r);assert x['mode']==i
                arms[i]=x;cnt+=1;vis+=x['visits'];nn+=x['NN_calls'];caps[x['cause']]+=1
            assert all(x['prefix']==arms[0]['prefix'] and x['prefix_counts']==arms[0]['prefix_counts'] and x['trace'][:32]==arms[0]['trace'][:32] for x in arms)
            t=dict(root=r,order=seq,logged=seq[0],propensity=[1/3]*3,arms=arms)
            if part=='TEST':
                b=srv.request(r['history'],'K24BASE');check(b,r,True);assert all(b[k]==arms[0][k] for k in ('prefix','result','best','visits','incremental_nn','cause'))
                x=srv.request(r['history'],'K24LIVE',merge_table=f['table']);check(x,r)
                c=f['table'][x['bin']];assert x['mode']==c and all(x[k]==arms[c][k] for k in ('prefix','prefix_counts','result','best','visits','incremental_nn','counts','trace'))
                t.update(base=b,live=x);parity+=1
                for a in (b,x):cnt+=1;vis+=a['visits'];nn+=a['NN_calls']
            emit(fd,t)
            if r['id']%30==0:print('PROGRESS',part,r['id'],cnt,flush=True)
    save(part+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),sha256=sha(path),identity=ident(),requests=cnt,visits=vis,NN=nn,causes=dict(caps),live_and_base_parity=parity,freeze=None if part!='TEST' else sha(D/'FROZEN.json')))
    print('TRIAL',part,cnt,vis,nn,dict(caps),'parity',parity,flush=True)
def labels(part):
    lk=read(part+'_LOCK.json');assert lk['sha256']==sha(D/(part+'_ATLAS.jsonl.gz'))
    co=read('COHORT.json');assert lk['cohort']==sha(D/'COHORT.json')
    if part=='TEST':assert lk['freeze']==sha(D/'FROZEN.json')
    out=[];cnt=C.Counter()
    for r in co['roots']:
        if (r['group']=='TEST')!=(part=='TEST'):continue
        q=old.oracle(r);item=dict(root=r,status=q['status'])
        if q['status']=='EXACT':
            try:
                v,n=p.ref.exact_wdl(r['history']);truth={a:(q['scores'][a]>0)-(q['scores'][a]<0) for a in r['legal']}
                item.update(scores=q['scores'],independent=v,nodes=n,agreement=v==truth)
                if v!=truth:item['status']='MISMATCH'
            except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
        else:item['oracle_result']=q
        cnt[item['status']]+=1;out.append(item)
    save(part+'_LABELS.json',dict(atlas=lk['sha256'],cohort=sha(D/'COHORT.json'),counts=dict(cnt),rows=out));print('LABELS',part,dict(cnt),flush=True)
def tele(a):
    z=C.Counter()
    for t in a['trace'][32:]:
        z['terminal_visits']+=int(t['terminal'])
        for s in t['steps']:
            z['edge_events']+=1
            if t['terminal']:continue
            z['nonterminal_edges']+=1
            z['raw_opportunities']+=int(s['raw_n']>0)
            z['novel_opportunities']+=int(s['novel_n']>0)
            z['duplicate_only']+=int(s['raw_n']>0 and s['novel_n']==0)
            z['cross_raw_count']+=s['raw_n'];z['cross_unique_count']+=s['unique_n'];z['cross_novel_count']+=s['novel_n']
            z['repeated_other_count']+=s['raw_n']-s['unique_n'];z['overlap_own_current']+=s['unique_n']-s['novel_n']
            z['changed_backup']+=int(abs(s['backup']+s['incoming'])>1e-7)
    z['shared_children']=a['counts']['shared'];z['prefix_shared']=a['prefix_counts']['shared']
    return dict(z)
def data(part):
    lab=read(part+'_LABELS.json');lk=read(part+'_LOCK.json');assert lab['atlas']==lk['sha256']==sha(D/(part+'_ATLAS.jsonl.gz'))
    ll={x['root']['id']:x for x in lab['rows'] if x['status']=='EXACT'}
    rows=[]
    for t in raw(D/(part+'_ATLAS.jsonl.gz')):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u={int(k):(v+1)/2 for k,v in ll[i]['independent'].items()};aa=t['arms']
        for a in aa:assert all(lo<=u[m]<=hi for m,lo,hi in a['proof'])
        y=[u[a['best']]-PRICE*a['incremental_nn'] for a in aa]
        q=dict(id=i,group=r['group'],b=aa[0]['bin'],y=y,us=[u[a['best']] for a in aa],n=[a['incremental_nn'] for a in aa],full=max(u.values()),pre=u[aa[0]['prefix']['best']],logged=t['logged'],vis=[a['visits'] for a in aa],cpu=[a['cpu_ns']/1e6 for a in aa],wall=[a['wall_ns']/1e6 for a in aa],tail=[a['tail_us']/1e3 for a in aa],tele=[tele(a) for a in aa])
        if part=='TEST':q['base']=dict(cpu=t['base']['cpu_ns']/1e6,wall=t['base']['wall_ns']/1e6,tail=t['base']['tail_us']/1e3)
        rows.append(q)
    return rows
def fit():
    assert not (D/'TEST_ATLAS.jsonl.gz').exists() and not (D/'TEST_LABELS.json').exists()
    rows=data('TRAIN_CAL');tr=[r for r in rows if r['group']=='TRAIN'];cal=[r for r in rows if r['group']=='CAL']
    assert tr and cal;table=[];stats=[]
    for b in range(4):
        rr=[r for r in tr if r['b']==b];scores=[]
        for c in (1,2):
            z=[r['y'][c]-r['y'][0] for r in rr];n=len(z);mu=S.mean(z) if n else 0.;se=S.stdev(z)/math.sqrt(n) if n>1 else 0.;pos=sum(x>1e-12 for x in z);neg=sum(x< -1e-12 for x in z)
            v=mu-2*se;eligible=n>=8 and pos>=3 and neg>=1 and v>0
            scores.append(dict(c=c,n=n,mean=mu,se=se,lower=v,positive=pos,negative=neg,eligible=eligible))
        best=max([dict(c=0,lower=0.)]+[x for x in scores if x['eligible']],key=lambda x:x['lower'])
        table.append(best['c']);stats.append(scores)
    calnet={'N':S.mean(r['y'][0] for r in cal),'guarded':S.mean(r['y'][table[r['b']]] for r in cal)}
    selected=max(('N','guarded'),key=calnet.__getitem__);chosen=[0]*4 if selected=='N' else table
    save('FROZEN.json',dict(cohort=sha(D/'COHORT.json'),trace=sha(D/'TRAIN_CAL_ATLAS.jsonl.gz'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'merge_trial.py'),identity=ident(),train_ids=[r['id'] for r in tr],cal_ids=[r['id'] for r in cal],stats=stats,guarded=table,table=chosen,selected=selected,calnet=calnet))
    print('FIT',len(tr),len(cal),selected,calnet,table,stats,flush=True)
def agg(rows,t):
    if not rows:return dict(n=0)
    v=[(r,t[r['b']]) for r in rows];u=S.mean(r['us'][c] for r,c in v)
    return dict(n=len(v),utility=u,regret=S.mean(r['full'] for r in rows)-u,net=S.mean(r['y'][c] for r,c in v),NN=S.mean(r['n'][c] for r,c in v),visits=S.mean(r['vis'][c] for r,c in v),cpu=S.mean(r['cpu'][c] for r,c in v),wall=S.mean(r['wall'][c] for r,c in v),tail=S.mean(r['tail'][c] for r,c in v),choices=dict(C.Counter(c for r,c in v)))
def pair(rows,t,base=0):
    if not rows:return dict(n=0)
    z=[r['us'][t[r['b']]]-r['us'][base] for r in rows];rng=random.Random(SEED+21)
    bs=sorted(S.mean(rng.choices(z,k=len(z))) for _ in range(4000))
    return dict(n=len(z),gain=S.mean(z),net_gain=S.mean(r['y'][t[r['b']]]-r['y'][base] for r in rows),better=sum(v>0 for v in z),worse=sum(v<0 for v in z),ci=[bs[99],bs[3899]])
def score():
    f=read('FROZEN.json');assert read('TEST_LOCK.json')['freeze']==sha(D/'FROZEN.json')
    rows=data('TEST');assert rows;tabs={a:[i]*4 for i,a in enumerate(ARMS)};tabs.update(guarded=f['guarded'],live=f['table'])
    matched=[r for r in rows if r['n']==[64]*3]
    telem={}
    for i,a in enumerate(ARMS):
        z=C.Counter()
        for r in rows:z.update(r['tele'][i])
        telem[a]=dict(totals=dict(z),raw_exposed=sum(r['tele'][i]['raw_opportunities']>0 for r in rows),novel_exposed=sum(r['tele'][i]['novel_opportunities']>0 for r in rows),changed_roots=sum(r['tele'][i]['changed_backup']>0 for r in rows))
    out=dict(n=len(rows),matched=len(matched),selected=f['selected'],full={k:agg(rows,t) for k,t in tabs.items()},pairs={k:pair(rows,t) for k,t in tabs.items()},matched_pairs={k:pair(matched,t) for k,t in tabs.items()},D_vs_M=pair(rows,[2]*4,1),telemetry=telem,base={k:S.mean(r['base'][k] for r in rows) for k in ('cpu','wall','tail')},prefix_shared_roots=sum(r['tele'][0]['prefix_shared']>0 for r in rows),rows=rows)
    save('SUMMARY.json',out);print('SUMMARY',__import__('json').dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('trial','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
