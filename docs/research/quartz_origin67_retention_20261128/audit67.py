from plan67 import D,T,P,P58,SEEDS,EXTRA,NOVEL,TRANSFER,ARMS,DEV,sha,read,raw
from collections import Counter,defaultdict
import sys,json,gzip,math,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P));import audit66 as old
a=old.a
CAL={'CALV':'VOTE','CALF':'FOCUS','CALX':'EXPAND'}
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def independent_features(r,op):
    s=r['allocation65'];bd={m:(l,u) for m,l,u in r['partial']['bounds']};post={m:(n,w,pi) for m,n,w,pi in s['edges']}
    pre=post if op=='VOTE' else {m:(n,w,pi) for m,n,w,pi in r['operation66']['prefix_edges']}
    a0=s['chosen'] if op=='VOTE' else r['operation66']['prefix_move'];b=s['fallback'] if op=='VOTE' else s['chosen']
    assert a0 in s['safe'] and b in s['safe']
    def v(tab,m):
        lo,hi=bd[m];n,w,p=tab[m]
        if lo==hi:return float(lo)
        if not n:return None
        return min(hi,max(lo,w/n))
    x0,x1,y0,y1=v(pre,a0),v(pre,b),v(post,a0),v(post,b)
    u0,u1=0 if y0 is None else y0,0 if y1 is None else y1
    z=[1,int(op=='VOTE'),int(op=='FOCUS'),(u1-u0)/2,u0,(post[b][0]-post[a0][0])/5,post[b][2]-post[a0][2],
       (y1-x1)/2 if x1 is not None and y1 is not None else 0,(y0-x0)/2 if x0 is not None and y0 is not None else 0,
       int(x1 is None),int(x0 is None),1-len(s['safe'])/len(bd),len(r['history'])/49,int(y1 is None),int(y0 is None)]
    return a0,b,z
def check_retention(r,op,model):
    if 'allocation65' not in r:assert 'retention67' not in r;return r['move']
    a0,b,x=independent_features(r,op);z=r['retention67'];assert z['a']==a0 and z['b']==b and z['op']==op
    if a0==b:assert z['prob'] is None and z['gain'] is None and z['chosen']==a0;return a0
    v=[(model['offset'][j]+sum(c*f for c,f in zip(model['coef'][j],x)))/model['temperature'] for j in range(5)]
    e=[math.exp(t-max(v)) for t in v];p=[t/sum(e) for t in e];g=(p[4]-p[0])+.5*(p[3]-p[1])
    assert all(abs(t-u)<1e-12 for t,u in zip(p,z['prob'])) and abs(g-z['gain'])<1e-12
    # Decision on serialized exact model prediction, independent prediction agrees away from numerical tie.
    m=b if z['gain']>0 else a0
    assert z['chosen']==m and (abs(g)<1e-14 or (g>0)==(z['gain']>0))
    a.CT['retention_discord']+=1;a.CT['retention_switch']+=m!=a0
    return m
def check_game(g,start,color,policy,dev=False,model=None):
    h=list(start);ct=Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,7);bd,g0=a.v.base(aa,bb,7)
        assert r['history']==h and r['move'] in bd and r['played'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B') and r['lock']==q
        actual=policy if not dev or i==0 else 'FREE'; pol=CAL.get(actual,actual)
        if pol=='VOTE':pol='MA4'
        if q:
            if pol in ('FOCUS','EXPAND'):old.check_operation(h,r,pol)
            elif pol=='MA4':old.old.check_new(h,r,'MA4')
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
                            if e['reached']:rr=old.old.old.check_select(h,r,bds,'FREE')
                            rr=dict(rr,early=dict(rr['early'],op=1))
                        a.inspect_early(h,rr,bds,dict(horizon=4),'STOP')
                        a.vote(h,7,rr,True,rr['cap'],rr['original_move'])
                        assert rr['move']==a.v.changed(rr['original_move'],rr['votes'],bds,7)
                else:a.vote(h,7,r,True,cap)
                a.cache_check(h,r)
        else:
            assert not r['partial'] and 'allocation65' not in r;a.vote(h,7,r,False,32);a.cache_check(h,r)
        played=r['move']
        if q and actual in CAL:
            played=check_retention(r,CAL[actual],model)
        elif q and actual=='VOTE' and 'allocation65' in r:played=r['allocation65']['fallback']
        else:assert 'retention67' not in r
        assert r['played']==played
        tag='Q' if q else 'B'
        for k,f in [('nn','actual_nn'),('logical','nn'),('cpu','cpu_ns'),('wall','wall_ns')]:
            assert type(r[f]) is int and r[f]>=0;ct[tag+k]+=r[f]
        ct[tag+'moves']+=1;ct['moves']+=1;h.append(r['played'])
    aa,bb,w=a.v.replay(h,7);assert h==g['history'] and (w or aa|bb==a.v.G[7][1]) and w==g['winner'] and g['score2']==1+color*w
    assert g['actual_nn']==ct['Qnn']+ct['Bnn'] and g['nn']==ct['Qlogical']+ct['Blogical'] and g['targets']==g['switches']==0
    ct.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),targets=0,switches=0);return dict(ct)

def check_dev(j,gs):
    assert set(gs)==set(DEV);ac={};a0=j['incumbent'];ref=gs['MA4']['rows'][0]
    assert ref['allocation65']==j['snapshot'] and ref['partial']['bounds']==j['bounds'] and ref['actual_nn']==j['root_nn'] and ref['move']==a0
    for arm,g in gs.items():
        assert g['policy']==arm;ac[arm]=check_game(g,j['history'],j['color'],arm,True)
        r=g['rows'][0]
        if arm in ('MA4','VOTE'):assert r['allocation65']==j['snapshot'] and r['actual_nn']==j['root_nn']
        else:assert r['operation66']['prefix_edges']==j['snapshot']['edges'] and r['operation66']['prefix_move']==a0 and r['allocation65']['steps'][:4]==j['snapshot']['steps']
        for y,v in gs.items():
            if r['played']==v['rows'][0]['played']:assert g['score2']==v['score2'] and g['history']==v['history'] and sem(g['rows'][1:])==sem(v['rows'][1:])
    return ac
def shard(arg):
    mode,meta=arg;a.CT.clear();a.EX.clear();f=T/meta['file'];assert sha(f)==meta['sha256'];pl=read(D/'PLAN.json');ct=Counter();rows=[];model=read(D/'MODEL.json') if mode!='DEV' else None
    def count(c):
        ct['games']+=1;ct['moves']+=c['moves'];ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical']
    for z in raw(f):
        if mode=='DEV':
            j=z['job'];assert j==pl['roots'][j['id']];gs=z['games'];ac=check_dev(j,gs)
            for c in ac.values():count(c)
            ct['roots']+=1
            rows.append(dict(id=j['id'],domain=j['domain'],split=j['split'],family=j['family_id'],scores={arm:g['score2'] for arm,g in gs.items()},arms=ac))
        else:
            g=z['H4'];j=pl['jobs'][g['job']['id']];seed=g['seed'];color=g['color'];ac={};assert j['panel']==mode
            for arm in ARMS:
                g=z[arm];assert g['job']==j and g['seed']==seed and g['color']==color and g['policy']==arm
                c=check_game(g,j['opening'],color,arm,False,model);ac[arm]=c;count(c)
            ct['episodes']+=1
            rows.append(dict(id=j['id'],seed=seed,color=color,geometry=j['geometry'],arms=ac,pairs={arm:dict(delta2=z[arm]['score2']-z['H4']['score2'],diverged=z[arm]['history']!=z['H4']['history']) for arm in ARMS if arm!='H4'}))
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return rows,dict(a.CT),[(list(k),v) for k,v in a.EX.items()]
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json') and lock['source']==sha(D/'games67.py') and lock['eval_source']==sha(D/'eval67.py') and lock['retention_source']==sha(D/'retention67.py')
    assert lock['model']==(None if mode=='DEV' else sha(D/'MODEL.json'))
    pl=read(D/'PLAN.json');rr=[];ct=Counter();ex=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for r,c,xx in pool.map(shard,[(mode,m) for m in lock['files']]):rr+=r;ct.update(c);ex+=xx
    if mode=='DEV':assert sorted(r['id'] for r in rr)==list(range(749))
    else:
        seeds={'MAIN':SEEDS,'ARCHIVE':EXTRA,'NOVEL':NOVEL,'TRANSFER':TRANSFER}[mode]
        assert {(r['id'],r['seed'],r['color']) for r in rr}=={(j['id'],s,c) for j in pl['jobs'] if j['panel']==mode for s in seeds for c in (-1,1)}
        assert len(rr)==(336 if mode=='MAIN' else 72)
    assert ct['cache_physical']==lock['counts']['physical'] and ct['cache_logical']==lock['counts']['logical']
    out=dict(rows=rr,proof=dict(ct),exact=ex,lock_sha256=sha(T/(mode+'_LOCK.json')),counts=lock['counts'])
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print(mode,'AUDIT',len(rr),dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
