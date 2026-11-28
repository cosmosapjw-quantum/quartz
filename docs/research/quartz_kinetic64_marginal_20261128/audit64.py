"""Independent replay: forced actions, paired child updates, proof and exact cache accounting."""
from plan64 import D,T,P58,SEEDS,EXTRA,NOVEL,ARMS,DEV,sha,read,raw
from collections import Counter,defaultdict
import sys,json,gzip,math,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P58));import audit58 as a
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def close(x,y):assert abs(x-y)<1e-12,(x,y)
def check_target(h,r,bd):
    t=r['target'];s=r['early']['at'];m=s['move'];b=s['rival'];k=1 if t['mode']=='pooled64' else 2;assert t['mode'] in ('pooled62','pooled64') and t['incumbent']==m and t['rival']==b and b is not None and t['reason']=='compared'
    ee={i:(n,w) for i,n,w,_ in s['edges']};values=[];nn=0
    assert len(t['children'])==2
    for i,z in zip((m,b),t['children']):
        l,u=bd[i];n,w=ee[i]
        if l==u:
            assert z is None;values.append(float(l));continue
        assert z['old_root_n']==n;close(z['old_root_w'],w)
        q=z['samples'];assert len(q)==z['visits']==k and all(-1<=v<=1 for v in q)
        aa,bb,win=a.v.replay(h+[i],7);term=bool(win) or aa|bb==a.v.G[7][1]
        pre,post=z['pre_edges'],z['edges']
        if term:
            assert z['starts_unexpanded'] and not pre and not post and z['terminal_leaves']==k and z['nn']==0
            assert q==([-1.]*k if win else [0.]*k)
        else:
            fresh=n==0;assert z['starts_unexpanded']==fresh
            legal=[i for i in range(49) if not (aa|bb)>>i&1]
            assert [x[0] for x in post]==legal
            if fresh:
                assert not pre
                before={i:(0,0.) for i in legal}
            else:
                assert [x[0] for x in pre]==legal and sum(x[1] for x in pre)==n-1
                before={i:(nn,w) for i,nn,w in pre}
            assert sum(x[1]-before[x[0]][0] for x in post)==k-int(fresh)
            assert all(v>=before[i][0] and abs(w)<=v+1e-12 and abs(w-before[i][1])<=v-before[i][0]+1e-12 for i,v,w in post)
            close(math.fsum(w-before[i][1] for i,v,w in post),math.fsum(q[int(fresh):]))
        assert z['nn']==k-z['terminal_leaves'] and 0<=z['nn']<=k
        raw0=(w-math.fsum(q))/(n+k);close(z['raw'],raw0);val=max(l,min(u,raw0));close(z['value'],val);values.append(z['value']);nn+=z['nn']
        a.CT[f'target{k}_children']+=1
    assert t['values']==values and t['chosen']==(b if values[1]>values[0] else m)==r['move']
    assert t['extra_nn']==nn<=2*k and r['nn']==r['cache_receipt']['root_logical']+nn and r['cache_receipt']['physical']-r['cache_receipt']['root_physical']<=2*k
    a.CT[f'target{k}_events']+=1;a.CT[f'target{k}_switches']+=int(r['move']!=m);a.CT[f'target{k}_logical']+=nn
    # Normalize only the prefix for the independent preexisting snapshot and vote checks.
    rr={k:v for k,v in r.items() if k!='target'};rr['move']=m;rr['nn']=r['cache_receipt']['root_logical'];rr['actual_nn']=r['cache_receipt']['root_physical']
    return rr
def check_select(h,r,bd,policy):
    z=r['select63'];s=r['early']['at'];m,b=s['move'],s['rival'];assert z['incumbent']==m and z['rival']==b
    es={i:(n,w) for i,n,w,p in s['edges']};n,w=es[m];v,t=es[b]
    cell=0 if not n else (1 if w<0 else 2) if not v else 3 if t*n>w*v else 4 if w<0 else 5
    assert z['cell']==cell
    op=1 if policy in ('FREE','OTHER') else [2,0,0,1,2,0][cell];assert z['op']==op
    vals=[]
    for i in (m,b):
        n,w=es[i];l,u=bd[i]
        vals.append(float(l) if l==u else max(l,min(u,w/n)) if n else None)
    for x,y in zip(vals,z['values']):
        if x is None:assert y is None
        else:close(x,y)
    assert ('target' in r)==(op==2)
    if op==2:rr=check_target(h,r,bd);chosen=r['target']['chosen']
    else:
        chosen=b if op==1 and all(x is not None for x in vals) and vals[1]>vals[0] else m
        rr=dict(r,move=m)
    assert r['move']==z['chosen']==chosen
    c=r['cache_receipt'];assert z['extra_logical']==c['logical']-c['root_logical'] and z['extra_physical']==c['physical']-c['root_physical']
    assert 0<=z['extra_physical']<=z['extra_logical']<=4 and (op==2 or z['extra_logical']==0)
    a.CT[f'select63_op{op}']+=1;a.CT[f'select63_cell{cell}_op{op}']+=1
    a.CT['select63_switches']+=int(chosen!=m)
    return rr
def check_marginal(h,r,bd,policy):
    z=r['select64'];s=r['early']['at'];m,b=s['move'],s['rival'];assert z['incumbent']==m and z['rival']==b
    es={i:(n,w) for i,n,w,p in s['edges']};n,w=es[m];v,t=es[b]
    cell=0 if not n else (1 if w<0 else 2) if not v else 3 if t*n>w*v else 4 if w<0 else 5
    assert z['cell']==cell
    op=1 if policy=='T2' else read(D/'TABLE.json')['table'][cell];assert z['op']==op
    vals=[]
    for i in (m,b):
        n,w=es[i];l,u=bd[i];vals.append(float(l) if l==u else max(l,min(u,w/n)) if n else None)
    for x,y in zip(vals,z['values']):
        if x is None:assert y is None
        else:close(x,y)
    f=b if all(x is not None for x in vals) and vals[1]>vals[0] else m;assert z['free']==f
    assert ('target' in r)==bool(op)
    if op:
        assert r['target']['mode']==('pooled64' if op==1 else 'pooled62')
        rr=check_target(h,r,bd);chosen=r['target']['chosen']
    else:rr=dict(r,move=m);chosen=f
    assert r['move']==z['chosen']==chosen
    c=r['cache_receipt'];assert z['extra_logical']==c['logical']-c['root_logical'] and z['extra_physical']==c['physical']-c['root_physical']
    assert 0<=z['extra_physical']<=z['extra_logical']<=2*op
    a.CT[f'select64_op{op}']+=1;a.CT[f'select64_cell{cell}_op{op}']+=1;a.CT['select64_change_free']+=int(chosen!=f)
    return rr
def audit_game(g,start,color,policy,dev=False,root=None):
    h=list(start);ct=Counter();tcount=scount=0
    for j,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,7);bd,g0=a.v.base(aa,bb,7)
        assert r['history']==h and r['played'] in bd and r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B') and r['lock']==q
        z=r.get('partial');assert bool(z)==bool(q and g0 is None);cap=20 if q else 32;op=policy if not dev or j==0 else 'FREE'
        if z:
            bds=a.replay_proof(h,7,z)
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                a.vote(h,7,r,True,cap);assert 'early' not in r and 'target' not in r
            else:
                assert 'early' in r
                rr=r
                if op=='T4':
                    e=r['early'];assert e['op']==7 and e['horizon']==4 and ('target' in r)==e['reached']
                    if e['reached']:
                        rr=check_target(h,r,bds);tcount+=1;scount+=int(r['move']!=r['target']['incumbent'])
                    rr=dict(rr,early=dict(rr['early'],op=1))
                    horizon=4
                elif op in ('FREE','OTHER','K63'):
                    e=r['early'];assert e['op']==(8 if op in ('FREE','OTHER') else 9) and e['horizon']==4 and ('select63' in r)==e['reached']
                    if e['reached']:
                        rr=check_select(h,r,bds,op)
                        if 'target' in r:tcount+=1;scount+=int(r['move']!=r['target']['incumbent'])
                    rr=dict(rr,early=dict(rr['early'],op=1));horizon=4
                elif op in ('T2','ROBUST'):
                    e=r['early'];assert e['op']==(10 if op=='T2' else 11) and e['horizon']==4 and ('select64' in r)==e['reached']
                    if e['reached']:
                        rr=check_marginal(h,r,bds,op)
                        if 'target' in r:tcount+=1;scount+=int(r['move']!=r['target']['incumbent'])
                    rr=dict(rr,early=dict(rr['early'],op=1));horizon=4
                else:
                    assert 'target' not in r;horizon=4
                a.inspect_early(h,rr,bds,dict(horizon=horizon),'STOP')
                a.vote(h,7,rr,True,rr['cap'],rr['original_move'])
                mv=a.v.changed(rr['original_move'],rr['votes'],bds,7);assert mv==rr['move'] and rr['replacement']==(mv!=rr['original_move'])
        else:
            assert 'early' not in r and 'target' not in r;a.vote(h,7,r,q,cap)
        a.cache_check(h,r)
        if dev and j==0 and policy=='OTHER':
            assert r['early']['reached'] and r['move']==root['free'];x=r['early']['at'];other=x['rival'] if r['move']==x['move'] else x['move']
            assert r['played']==other and bds[other][1]!=-1
        else:assert r['played']==r['move']
        tag='Q' if q else 'B'
        for k,f in [('nn','actual_nn'),('logical','nn'),('cpu','cpu_ns'),('wall','wall_ns')]:
            assert type(r[f]) is int and r[f]>=0;ct[tag+k]+=r[f]
        ct[tag+'moves']+=1;ct['moves']+=1;h.append(r['played'])
    aa,bb,w=a.v.replay(h,7);assert h==g['history'] and (w or aa|bb==a.v.G[7][1]) and w==g['winner'] and g['score2']==1+color*w
    assert g['actual_nn']==ct['Qnn']+ct['Bnn'] and g['nn']==ct['Qlogical']+ct['Blogical'] and (g['targets'],g['switches'])==(tcount,scount)
    ct.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),targets=tcount,switches=scount)
    return dict(ct)
def check_pair(j,gs,p):
    r=gs['FREE']['rows'][0];x=r['early']['at'];a0,b0=x['move'],x['rival'];f=r['move'];o=b0 if f==a0 else a0
    assert x==j['snapshot'] and r['partial']['bounds']==j['bounds'] and r['actual_nn']==j['root_nn'] and f==j['free']
    y={k:g['score2'] for k,g in gs.items()};m={k:g['rows'][0]['played'] for k,g in gs.items()}
    assert m['FREE']==f and m['OTHER']==o and gs['OTHER']['rows'][0]['actual_nn']==r['actual_nn']
    for k in gs:
        assert gs[k]['rows'][0]['early']['at']==x and m[k] in (a0,b0)
        for l in gs:
            if m[k]==m[l]:assert y[k]==y[l] and gs[k]['history']==gs[l]['history'] and sem(gs[k]['rows'][1:])==sem(gs[l]['rows'][1:])
    extra={k:gs[k]['rows'][0]['actual_nn']-r['actual_nn'] for k in ('FREE','T2','T4')}
    assert p==dict(a=a0,b=b0,free=f,other=o,moves=m,score2=y,oracle2=max(y['FREE'],y['OTHER']),extra=extra)
    assert extra['FREE']==0 and 0<=extra['T2']<=2 and 0<=extra['T4']<=4
def shard(arg):
    mode,meta=arg;a.CT.clear();a.EX.clear();f=T/meta['file'];assert sha(f)==meta['sha256'];pl=read(D/'PLAN.json');ct=Counter();rows=[]
    def count(c):
        ct['games']+=1;ct['moves']+=c['moves'];ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical'];ct['targets']+=c['targets'];ct['switches']+=c['switches']
    for z in raw(f):
        if mode=='DEV':
            j=z['job'];assert j==pl['roots'][j['id']];gs=z['games'];ac={}
            for arm in DEV:
                assert gs[arm]['policy']==arm
                c=audit_game(gs[arm],j['history'],j['color'],arm,True,j);count(c);ac[arm]=c
            check_pair(j,gs,z['pair']);ct['roots']+=1;rows.append(dict(id=j['id'],cell=j['cell'],family=j['family_id'],domain=j['domain'],pair=z['pair'],arms=ac))
        else:
            g=z['H4'];j=pl['jobs'][g['job']['id']];seed=g['seed'];color=g['color'];ac={};assert j['panel']==mode
            for arm in ARMS:
                g=z[arm];assert g['job']==j and g['seed']==seed and g['color']==color and g['policy']==arm
                c=audit_game(g,j['opening'],color,arm);count(c);ac[arm]=c
            ct['episodes']+=1;rows.append(dict(id=j['id'],seed=seed,color=color,geometry=j['geometry'],arms=ac,pairs={arm:dict(delta2=z[arm]['score2']-z['H4']['score2'],diverged=z[arm]['history']!=z['H4']['history']) for arm in ('FREE','T2','T4','K63','ROBUST')}))
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return rows,dict(a.CT),[(list(k),v) for k,v in a.EX.items()]
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json') and lock['source']==sha(D/'games64.py') and lock['eval_source']==sha(D/'eval64.py')
    pl=read(D/'PLAN.json');rr=[];ct=Counter();ex=[]
    assert lock['table']==(None if mode=='DEV' else sha(D/'TABLE.json'))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for r,c,xx in pool.map(shard,[(mode,m) for m in lock['files']]):rr+=r;ct.update(c);ex+=xx
    if mode=='DEV':assert sorted(r['id'] for r in rr)==list(range(554))
    else:
        seeds=SEEDS if mode=='MAIN' else EXTRA if mode=='ARCHIVE' else NOVEL
        assert len(rr)==(448 if mode=='MAIN' else 192 if mode=='ARCHIVE' else 144)
        assert {(r['id'],r['seed'],r['color']) for r in rr}=={(j['id'],s,c) for j in pl['jobs'] if j['panel']==mode for s in seeds for c in (-1,1)}
    assert ct['cache_physical']==lock['counts']['physical'] and ct['cache_logical']==lock['counts']['logical']
    out=dict(rows=rr,proof=dict(ct),exact=ex,lock_sha256=sha(T/(mode+'_LOCK.json')),counts=lock['counts'])
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print(mode,'AUDIT',len(rr),dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
