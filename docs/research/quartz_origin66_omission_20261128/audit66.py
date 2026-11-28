from plan66 import D,T,P,P58,SEEDS,EXTRA,NOVEL,ARMS,DEV,sha,read,raw
from collections import Counter,defaultdict
import sys,json,gzip,math,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P));import audit65 as old
a=old.a
def close(x,y):assert abs(x-y)<1e-12,(x,y)
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def check_operation(h,r,policy):
    assert r['history']==h and r['lock'] is True
    aa,bb=a.v.orient(h,7);base,g=a.v.base(aa,bb,7);z=r['partial'];assert bool(z)==(g is None)
    if g is not None or z['cert'] is not None:
        old.check_new(h,dict(r,cap=4),'MA4');assert r['cap']==5 and 'operation66' not in r;return
    bd=a.replay_proof(h,7,z);s=r['allocation65'];op=r['operation66'];mode=5 if policy=='FOCUS' else 6;assert s['mode']==mode
    es=op['prefix_edges'];p={m:pi for m,n,w,pi in es};N={m:0 for m in bd};W={m:0. for m in bd};safe=[m for m in bd if bd[m][1]>-1] or list(bd)
    assert [m for m,n,w,pi in es]==list(bd) and safe==s['safe'];close(math.fsum(p.values()),1)
    steps=s['steps'];k=r['cache_receipt'];keys=k['keys'];hits=set(k['hit_indices']);last=1;term=0
    ba,bb0,_=a.v.replay(h,7);player=1-2*(len(h)%2)
    def value(m):l,u=bd[m];return float(l) if l==u else max(l,min(u,W[m]/N[m])) if N[m] else None
    def key(m):return (value(m),N[m],p[m],-m)
    def readout():
        vals=[m for m in safe if value(m) is not None]
        return max(vals,key=key) if vals else max(safe,key=lambda m:(N[m],-m))
    def apply(step,exp):
        nonlocal last,term
        m,v,t,nn,phys=step;assert m==exp and -1<=v<=1 and type(t) is bool
        assert nn==last+int(not t) and phys==nn-sum(x<nn for x in hits)
        if t:term+=1;assert v in (-1,0,1)
        else:
            x,y,turn=keys[last];assert (x if player==1 else y)&(1<<m) and x&ba==ba and y&bb0==bb0
        N[m]+=1;W[m]+=v;last=nn
    for i,step in enumerate(steps[:4]):
        exp=max(safe,key=lambda m:((W[m]/N[m] if N[m] else 0.)+2*p[m]*math.sqrt(max(i,1))/(1+N[m]),-m));apply(step,exp)
    for m,n,w,pi in es:assert N[m]==n;close(W[m],w)
    inc=readout();assert inc==op['prefix_move'];assert op['prefix_logical']==last and op['prefix_physical']==last-sum(x<last for x in hits)
    eligible=[m for m in safe if bd[m][0]!=bd[m][1]]
    if mode==5:
        rivals=[m for m in eligible if N[m] and m!=inc];target=max(rivals,key=key) if rivals else inc if inc in eligible and N[inc] else None
    else:
        unvisited=[m for m in eligible if not N[m]];target=max(unvisited,key=lambda m:(p[m],-m)) if unvisited else None
    assert op['target']==target and len(steps)==4+int(target is not None)==r['cap']==r['visits']
    if target is not None:apply(steps[4],target)
    assert term==r['terminal_leaves'] and last==r['nn']
    for m,n,w,pi in s['edges']:assert N[m]==n and pi==p[m];close(W[m],w)
    assert r['votes']==[[m,N[m]] for m in bd];lead=max(N,key=lambda m:(N[m],-m));fallback=a.v.changed(lead,r['votes'],bd,7)
    assert s['fallback']==fallback and r['original_move']==lead and r['replacement']==(fallback!=lead)
    assert r['move']==s['chosen']==readout() and r['prior_best']==max(p,key=lambda m:(p[m],-m))
    assert op['extra_logical']==r['nn']-op['prefix_logical'] and op['extra_physical']==r['actual_nn']-op['prefix_physical']
    assert 0<=op['extra_physical']<=op['extra_logical']<=int(target is not None)
    assert all(not N[m] for m in bd if m not in safe)
    a.vote(h,7,r,True,r['cap'],lead);a.cache_check(h,r)
    a.CT['operation_roots']+=1;a.CT['operation_steps']+=int(target is not None);a.CT['operation_extra_physical']+=op['extra_physical'];a.CT['operation_noop']+=int(target is None)
def check_game(g,start,color,policy,dev=False,force=None):
    h=list(start);ct=Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,7);bd,g0=a.v.base(aa,bb,7)
        assert r['history']==h and r['move'] in bd and r['played'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B') and r['lock']==q
        pol=policy if not dev or i==0 else 'FREE'
        if q:
            if pol in ('FOCUS','EXPAND'):check_operation(h,r,pol)
            elif pol=='MA4':old.check_new(h,r,'MA4')
            else:
                # Reuse older independent H4/FREE per-root logic from audit64.
                z=r['partial'];assert bool(z)==(g0 is None);cap=20
                if z:
                    bds=a.replay_proof(h,7,z)
                    if z['cert'] is not None:
                        assert r['kind']=='PARTIAL47' and r['move']==z['cert'];a.vote(h,7,r,True,cap)
                    else:
                        rr=r
                        if pol=='FREE':
                            e=r['early'];assert e['op']==8 and ('select63' in r)==e['reached']
                            if e['reached']:rr=old.old.check_select(h,r,bds,'FREE')
                            rr=dict(rr,early=dict(rr['early'],op=1))
                        a.inspect_early(h,rr,bds,dict(horizon=4),'STOP')
                        a.vote(h,7,rr,True,rr['cap'],rr['original_move'])
                        assert rr['move']==a.v.changed(rr['original_move'],rr['votes'],bds,7)
                else:a.vote(h,7,r,True,cap)
                a.cache_check(h,r)
        else:
            assert not r['partial'] and 'allocation65' not in r;a.vote(h,7,r,False,32);a.cache_check(h,r)
        assert r['played']==(force if dev and i==0 and force is not None else r['move'])
        tag='Q' if q else 'B'
        for k,f in [('nn','actual_nn'),('logical','nn'),('cpu','cpu_ns'),('wall','wall_ns')]:
            assert type(r[f]) is int and r[f]>=0;ct[tag+k]+=r[f]
        ct[tag+'moves']+=1;ct['moves']+=1;h.append(r['played'])
    aa,bb,w=a.v.replay(h,7);assert h==g['history'] and (w or aa|bb==a.v.G[7][1]) and w==g['winner'] and g['score2']==1+color*w
    assert g['actual_nn']==ct['Qnn']+ct['Bnn'] and g['nn']==ct['Qlogical']+ct['Blogical'] and g['targets']==g['switches']==0
    ct.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),targets=0,switches=0);return dict(ct)
def check_dev(j,z):
    gs={int(m):g for m,g in z['forced'].items()};ops=z['operations'];bd={m:(l,u) for m,l,u in j['bounds']};assert set(gs)==set(bd) and set(ops)=={'FOCUS','EXPAND'}
    ac={};first=None
    for m,g in gs.items():
        assert g['policy']=='MA4';c=check_game(g,j['history'],j['color'],'MA4',True,m);ac[m]=c
        r=g['rows'][0];assert r['allocation65']==j['snapshot'] and r['move']==j['incumbent'] and r['actual_nn']==j['root_nn'] and r['partial']['bounds']==j['bounds']
        if first is None:first=r
        assert sem({k:v for k,v in r.items() if k!='played'})==sem({k:v for k,v in first.items() if k!='played'})
    oc={};y={m:g['score2'] for m,g in gs.items()}
    for arm,g in ops.items():
        assert g['policy']==arm;oc[arm]=check_game(g,j['history'],j['color'],arm,True)
        r=g['rows'][0];op=r['operation66'];assert op['prefix_edges']==j['snapshot']['edges'] and op['prefix_move']==j['incumbent'] and op['prefix_physical']==j['root_nn'] and r['allocation65']['steps'][:4]==j['snapshot']['steps']
        v=gs[r['played']];assert g['score2']==v['score2'] and g['history']==v['history'] and sem(g['rows'][1:])==sem(v['rows'][1:])
    safe=set(j['snapshot']['safe']);live={m for m,n,w,p in j['snapshot']['edges'] if m in safe and (n or bd[m][0]==bd[m][1])} or {j['incumbent']};inc=j['incumbent']
    assert inc in live<=safe<=set(gs)
    ma=max(y.values());mf=max(y[m] for m in safe);ml=max(y[m] for m in live);base=y[inc];assert ma-base==(ma-mf)+(mf-ml)+(ml-base)
    out=dict(id=j['id'],domain=j['domain'],cell=j['cell'],family=j['family_id'],weight_num=j['population'],weight_den=j['sample'],legal=len(gs),safe=len(safe),live=len(live),ma=ma,mf=mf,ml=ml,base=base,mask=ma-mf,omission=mf-ml,ranking=ml-base,regret=ma-base,base_cost=ac[inc]['Qnn'],ops={})
    for arm,g in ops.items():
        r=g['rows'][0];op=r['operation66'];t=op['target'];live2=live|({t} if t is not None else set());m2=max(y[m] for m in live2);sel=r['move'];assert sel in live2
        gain=y[sel]-base;omission_reduction=m2-ml;ranking_change=m2-y[sel]-(ml-base)
        assert gain==omission_reduction-ranking_change
        if arm=='FOCUS':assert omission_reduction==0
        out['ops'][arm]=dict(target=t,chosen=sel,gain=gain,omission_reduction=omission_reduction,ranking_change=ranking_change,extra=op['extra_physical'],whole_extra=oc[arm]['Qnn']-ac[inc]['Qnn'],changed=sel!=inc)
    return out,list(ac.values())+list(oc.values())
def shard(arg):
    mode,meta=arg;a.CT.clear();a.EX.clear();f=T/meta['file'];assert sha(f)==meta['sha256'];pl=read(D/'PLAN.json');ct=Counter();rows=[]
    def count(c):
        ct['games']+=1;ct['moves']+=c['moves'];ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical']
    for z in raw(f):
        if mode=='DEV':
            j=z['job'];assert j==pl['roots'][j['id']];r,cs=check_dev(j,z);rows.append(r)
            for c in cs:count(c)
            ct['roots']+=1;ct['forced_games']+=len(z['forced']);ct['operation_games']+=2
        else:
            g=z['H4'];j=pl['jobs'][g['job']['id']];seed=g['seed'];color=g['color'];ac={};assert j['panel']==mode
            for arm in ARMS:
                g=z[arm];assert g['job']==j and g['seed']==seed and g['color']==color and g['policy']==arm
                c=check_game(g,j['opening'],color,arm);ac[arm]=c;count(c)
            ct['episodes']+=1
            rows.append(dict(id=j['id'],seed=seed,color=color,geometry=j['geometry'],arms=ac,pairs={arm:dict(delta2=z[arm]['score2']-z['H4']['score2'],diverged=z[arm]['history']!=z['H4']['history']) for arm in ARMS if arm!='H4'}))
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return rows,dict(a.CT),[(list(k),v) for k,v in a.EX.items()]
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json') and lock['source']==sha(D/'games66.py') and lock['eval_source']==sha(D/'eval66.py')
    pl=read(D/'PLAN.json');rr=[];ct=Counter();ex=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for r,c,xx in pool.map(shard,[(mode,m) for m in lock['files']]):rr+=r;ct.update(c);ex+=xx
    if mode=='DEV':assert sorted(r['id'] for r in rr)==list(range(171))
    else:
        seeds=SEEDS if mode=='MAIN' else EXTRA if mode=='ARCHIVE' else NOVEL
        assert {(r['id'],r['seed'],r['color']) for r in rr}=={(j['id'],s,c) for j in pl['jobs'] if j['panel']==mode for s in seeds for c in (-1,1)}
        assert len(rr)==(336 if mode=='MAIN' else 72)
    assert ct['cache_physical']==lock['counts']['physical'] and ct['cache_logical']==lock['counts']['logical']
    out=dict(rows=rr,proof=dict(ct),exact=ex,lock_sha256=sha(T/(mode+'_LOCK.json')),counts=lock['counts'])
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print(mode,'AUDIT',len(rr),dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
