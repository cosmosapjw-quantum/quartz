"""Independent coordinate/proof replay, chronological root allocation and paired attribution."""
from plan65 import D,T,P,P58,SEEDS,EXTRA,NOVEL,ARMS,DEV,sha,read,raw
from collections import Counter,defaultdict
import sys,json,gzip,math,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P));import audit64 as old
a=old.a
def close(x,y):assert abs(x-y)<1e-12,(x,y)
def check_new(h,r,policy):
    assert r['history']==h and r['cap']==4 and r['lock'] is True
    aa,bb=a.v.orient(h,7);base,g=a.v.base(aa,bb,7);z=r['partial'];assert bool(z)==(g is None)
    if g is not None:
        a.vote(h,7,r,True,4);assert 'allocation65' not in r
    elif z['cert'] is not None:
        bd=a.replay_proof(h,7,z);assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None
        a.vote(h,7,r,True,4);assert 'allocation65' not in r
    else:
        bd=a.replay_proof(h,7,z);s=r['allocation65'];mode=DEV.index(policy);assert s['mode']==mode
        es=s['edges'];assert [x[0] for x in es]==list(bd) and all(x[3]>=0 for x in es);close(math.fsum(x[3] for x in es),1)
        p={m:pi for m,n,w,pi in es};N={m:0 for m in bd};W={m:0. for m in bd};safe=[m for m in bd if bd[m][1]>-1] or list(bd)
        assert safe==s['safe'] and len(s['steps'])==4
        k=r['cache_receipt'];keys=k['keys'];hits=set(k['hit_indices']);last=1;term=0
        ba,bb0,_=a.v.replay(h,7);player=1-2*(len(h)%2)
        for i,(m,v,t,nn,phys) in enumerate(s['steps']):
            allow=list(bd) if mode<2 else safe
            unseen=[u for u in allow if not N[u]]
            if mode==4 and unseen:
                exp=max(unseen,key=lambda u:(p[u],-u))
            else:
                exp=max(allow,key=lambda u:((W[u]/N[u] if N[u] else 0.)+2*p[u]*math.sqrt(max(i,1))/(1+N[u]),-u))
            assert m==exp and -1<=v<=1 and type(t) is bool
            assert nn==last+int(not t) and phys==nn-sum(x<nn for x in hits)
            if t:term+=1;assert v in (-1,0,1)
            else:
                x,y,turn=keys[last];assert (x if player==1 else y)&(1<<m)
                assert x&ba==ba and y&bb0==bb0
            N[m]+=1;W[m]+=v;last=nn
        assert term==r['terminal_leaves'] and last==r['nn']
        for m,n,w,pi in es:
            assert n==N[m];close(w,W[m]);assert abs(w)<=n+1e-12
        assert r['votes']==[[m,N[m]] for m in bd]
        lead=max(N,key=lambda m:(N[m],-m));fallback=a.v.changed(lead,r['votes'],bd,7);assert s['fallback']==fallback and r['replacement']==(fallback!=lead)
        vals={}
        for m in safe:
            l,u=bd[m]
            if l==u:vals[m]=float(l)
            elif N[m]:vals[m]=max(l,min(u,W[m]/N[m]))
        chosen=max(vals,key=lambda m:(vals[m],N[m],p[m],-m)) if vals and mode in (1,3,4) else fallback
        assert s['chosen']==r['move']==chosen and lead==r['original_move']
        assert r['prior_best']==max(p,key=lambda m:(p[m],-m))
        a.vote(h,7,r,True,4,lead)
        if mode>=2:assert all(N[m]==0 for m in bd if m not in safe)
        if mode==4:assert sum(N[m]>0 for m in safe)==min(4,len(safe))
        a.CT['allocation_roots']+=1;a.CT['allocation_steps']+=4
    a.cache_check(h,r)
def metrics(r):
    s=r['allocation65'];es={m:(n,w,p) for m,n,w,p in s['edges']};safe=s['safe'];bd={m:(l,u) for m,l,u in r['partial']['bounds']}
    return dict(visited=sum(es[m][0]>0 for m in safe),loss_visits=sum(n for m,(n,w,p) in es.items() if bd[m][1]==-1),unvisited_chosen=int(es[r['move']][0]==0),mass=sum(es[m][2] for m in safe if es[m][0])/sum(es[m][2] for m in safe),feasible=len(safe),physical=r['actual_nn'],logical=r['nn'],chosen=r['move'])
def prefix(r):return {k:v for k,v in r['allocation65'].items() if k not in ('mode','chosen')}
def check_game(g,start,color,policy):
    if policy in ('H4','FREE'):return old.audit_game(g,start,color,policy)
    h=list(start);ct=Counter()
    for r in g['rows']:
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,7);bd,g0=a.v.base(aa,bb,7)
        assert r['history']==h and r['played']==r['move'] and r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B') and r['lock']==q
        if q:check_new(h,r,policy)
        else:
            assert not r['partial'] and 'allocation65' not in r
            a.vote(h,7,r,False,32);a.cache_check(h,r)
        tag='Q' if q else 'B'
        for k,f in [('nn','actual_nn'),('logical','nn'),('cpu','cpu_ns'),('wall','wall_ns')]:
            assert type(r[f]) is int and r[f]>=0;ct[tag+k]+=r[f]
        ct[tag+'moves']+=1;ct['moves']+=1;h.append(r['move'])
    aa,bb,w=a.v.replay(h,7);assert h==g['history'] and (w or aa|bb==a.v.G[7][1]) and w==g['winner'] and g['score2']==1+color*w
    assert g['actual_nn']==ct['Qnn']+ct['Bnn'] and g['nn']==ct['Qlogical']+ct['Blogical'] and g['targets']==g['switches']==0
    ct.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),targets=0,switches=0);return dict(ct)
def shard(arg):
    mode,meta=arg;a.CT.clear();a.EX.clear();f=T/meta['file'];assert sha(f)==meta['sha256'];pl=read(D/'PLAN.json');ct=Counter();rows=[]
    for z in raw(f):
        if mode=='DEV':
            j=z['job'];assert j==pl['roots'][j['id']];rs=z['arms'];assert set(rs)==set(DEV)
            for arm,r in rs.items():
                check_new(j['history'],r,arm);assert r['partial']['bounds']==j['bounds'];ct['requests']+=1;ct['physical']+=r['actual_nn'];ct['logical']+=r['nn']
            r=rs['L4'];s=j['snapshot'];assert r['votes']==s['votes'] and r['allocation65']['edges']==s['edges'] and r['move']==s['move'] and r['actual_nn']==j['root_nn']
            for p,q in [('L4','A4'),('M4','MA4')]:
                assert prefix(rs[p])==prefix(rs[q]) and rs[p]['cache_receipt']==rs[q]['cache_receipt']
            ct['roots']+=1;rows.append(dict(id=j['id'],domain=j['domain'],cell=j['cell'],weight=j['population']/j['sample'],arms={k:metrics(v) for k,v in rs.items()}))
        else:
            g=z['H4'];j=pl['jobs'][g['job']['id']];seed=g['seed'];color=g['color'];ac={};assert j['panel']==mode
            for arm in ARMS:
                g=z[arm];assert g['job']==j and g['seed']==seed and g['color']==color and g['policy']==arm
                c=check_game(g,j['opening'],color,arm);ac[arm]=c
                ct['games']+=1;ct['moves']+=c['moves'];ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical']
            ct['episodes']+=1
            rows.append(dict(id=j['id'],seed=seed,color=color,geometry=j['geometry'],arms=ac,pairs={arm:dict(delta2=z[arm]['score2']-z['H4']['score2'],diverged=z[arm]['history']!=z['H4']['history']) for arm in ARMS if arm!='H4'}))
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return rows,dict(a.CT),[(list(k),v) for k,v in a.EX.items()]
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json') and lock['source']==sha(D/'games65.py') and lock['eval_source']==sha(D/'eval65.py')
    pl=read(D/'PLAN.json');rr=[];ct=Counter();ex=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for r,c,xx in pool.map(shard,[(mode,m) for m in lock['files']]):rr+=r;ct.update(c);ex+=xx
    if mode=='DEV':assert sorted(r['id'] for r in rr)==list(range(554))
    else:
        seeds=SEEDS if mode=='MAIN' else EXTRA if mode=='ARCHIVE' else NOVEL
        assert {(r['id'],r['seed'],r['color']) for r in rr}=={(j['id'],s,c) for j in pl['jobs'] if j['panel']==mode for s in seeds for c in (-1,1)}
        assert len(rr)==(336 if mode=='MAIN' else 72)
    assert ct['cache_physical']==lock['counts']['physical'] and ct['cache_logical']==lock['counts']['logical']
    out=dict(rows=rr,proof=dict(ct),exact=ex,lock_sha256=sha(T/(mode+'_LOCK.json')),counts=lock['counts'])
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print(mode,'AUDIT',len(rr),dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
