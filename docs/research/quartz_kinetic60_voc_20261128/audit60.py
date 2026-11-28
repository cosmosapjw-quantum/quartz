"""Chronological replay and coordinate proof audit; independently classify adaptive decisions."""
from plan60 import D,T,P58,SEEDS,HS,ARMS,sha,read,raw
import sys,collections as C,json,gzip,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P58))
import audit58 as a
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def adaptive(h,r,bds,table):
    e=r['early'];assert e['op']==5 and e['horizon']==16
    ss=e['snapshots'];sn=[];expected=None;counts=C.Counter()
    for x in ss:
        n=x['visits'];d=x['distance'];eligible=(d<=20-n and x['rival'] is not None)
        idx=(n//4-1)*4+2*int(2*d>=20-n)+int(x['move']==x['prior_best']) if eligible else None
        stop=idx is not None and table[idx]
        assert x['cell60']==idx and x['stop60']==stop
        if idx is not None:counts[str(idx)]+=1
        sn.append({k:v for k,v in x.items() if k not in ('cell60','stop60')})
        if stop:
            assert expected is None and x is ss[-1];expected=x
    assert e['reached']==(expected is not None) and e['at']==expected
    # Reuse K58 independent snapshot equations after removal of new decision metadata.
    rr=dict(r);rr['early']=dict(e,snapshots=sn)
    if expected is not None:
        hh=expected['visits'];rr['early'].update(horizon=hh,op=1,reason='eligible',at=sn[-1],reached=True)
        assert e['reason']=='voc60'
        a.inspect_early(h,rr,bds,dict(horizon=hh),'STOP')
    else:
        x=next((x for x in sn if x['visits']==16),None)
        reached=x is not None and not x['locked20'] and x['rival'] is not None
        reason='eligible' if reached else 'lock_before_horizon' if x is None else 'locked_at_horizon' if x['locked20'] else 'no_rival'
        rr['early'].update(horizon=16,op=0,reason=reason,at=x,reached=reached)
        a.inspect_early(h,rr,bds,dict(horizon=16),'CONTROL')
        assert e['reason']==('continue60' if ss else 'lock_before_horizon')
    return counts
def audit_game(g,start,color,rule,table):
    h=list(start);c=C.Counter();attempts=[];events=[];cells=C.Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,7);bd,g0=a.v.base(aa,bb,7)
        assert r['history']==h and r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B') and r['lock']==q
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=20 if q else 32
        if z:
            bds=a.replay_proof(h,7,z)
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                a.vote(h,7,r,True,cap);assert 'early' not in r
            else:
                mode,horizon=rule(i)
                assert ('early' in r)==(mode!='PLAIN')
                if mode!='PLAIN':
                    attempts.append(i)
                    if mode=='ADAPT':cells.update(adaptive(h,r,bds,table));reached=r['early']['reached']
                    else:reached,_=a.inspect_early(h,r,bds,dict(horizon=horizon),'CONTROL' if mode=='FULL' else 'STOP')
                    if reached:events.append(i)
                    cap=r['cap']
                a.vote(h,7,r,True,cap,r['original_move'])
                mv=a.v.changed(r['original_move'],r['votes'],bds,7);assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
        else:
            assert 'early' not in r;a.vote(h,7,r,q,cap)
        assert 'target' not in r;a.cache_check(h,r)
        tag='Q' if q else 'B'
        for key,f in [('nn','actual_nn'),('logical','nn'),('cpu','cpu_ns'),('wall','wall_ns')]:
            assert type(r[f]) is int and r[f]>=0;c[tag+key]+=r[f]
        c[tag+'moves']+=1;c['moves']+=1;h.append(r['move'])
    aa,bb,w=a.v.replay(h,7);assert h==g['history'] and (w or aa|bb==a.v.G[7][1])
    assert g['winner']==w and g['score2']==1+color*w
    assert g['nn']==c['Qlogical']+c['Blogical'] and g['actual_nn']==c['Qnn']+c['Bnn']
    if 'attempts' in g:assert attempts==g['attempts'] and events==g['events']
    c.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),attempts=len(attempts),events=len(events))
    return dict(c),dict(cells)
def shard(arg):
    mode,meta=arg;a.CT.clear();a.EX.clear();p=T/meta['file'];assert sha(p)==meta['sha256'];pl=read(D/'PLAN.json')
    table=read(D/'TABLE.json')['table'];ct=C.Counter();rows=[];cells=C.Counter()
    for z in raw(p):
        if mode=='TRAIN':
            j=z['job'];assert j==pl['roots'][j['id']];gs=z['games'];full=gs['0']
            for h in (0,*HS):
                g=gs[str(h)]
                c,_=audit_game(g,j['history'],j['color'],lambda i:('FULL' if h==0 else 'STOP',h or 16) if i==0 else ('STOP',j['h0']),table)
                ct['games']+=1;ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical'];ct['moves']+=c['moves']
                if h:
                    aa,bb=full['rows'][0],g['rows'][0];e=bb['early'];pa=z['pairs'][str(h)]
                    assert aa['early']['snapshots'][:len(e['snapshots'])]==e['snapshots']
                    assert pa==dict(eligible=e['reached'],gain2=full['score2']-g['score2'],root_cost=aa['actual_nn']-bb['actual_nn'],changed=aa['move']!=bb['move'])
                    assert pa['root_cost']>=0
                    if not e['reached']:assert pa['root_cost']==pa['gain2']==0 and not pa['changed']
                    if not pa['changed']:assert g['history']==full['history'] and sem(g['rows'][1:])==sem(full['rows'][1:])
                    ct['eligible']+=e['reached'];ct['changed']+=pa['changed']
            ct['roots']+=1;rows.append(j['id'])
        else:
            gs=z;g=gs['CONTROL'];j=pl['jobs'][g['job']['id']];seed=g['seed'];color=g['color'];ag={}
            for arm in (*ARMS,'H4'):
                g=gs[arm];assert g['job']==j and g['seed']==seed and g['color']==color and g['policy']==arm
                c,cl=audit_game(g,j['opening'],color,lambda i:('PLAIN',20) if arm=='CONTROL' else ('STOP',j['horizon'] if arm=='REPEAT' else 4) if arm in ('REPEAT','H4') else ('ADAPT',16),table)
                ag[arm]=c;cells.update(cl)
                ct['games']+=1;ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical'];ct['moves']+=c['moves']
                ct[arm+'_attempts']+=c['attempts'];ct[arm+'_events']+=c['events']
            ct['episodes']+=1
            rows.append(dict(id=j['id'],seed=seed,color=color,horizon=j['horizon'],geometry=j['geometry'],arms=ag,pair=dict(delta2=gs['ADAPT']['score2']-gs['REPEAT']['score2'],diverged=gs['ADAPT']['history']!=gs['REPEAT']['history'],h4_diverged=gs['ADAPT']['history']!=gs['H4']['history'],h4_delta2=gs['ADAPT']['score2']-gs['H4']['score2'])))
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return rows,dict(a.CT),dict(cells),[(list(k),v) for k,v in a.EX.items()]
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json')
    assert lock['source']==sha(D/('continuation60.py' if mode=='TRAIN' else 'arena60.py'))
    pl=read(D/'PLAN.json');rr=[];ct=C.Counter();cells=C.Counter();ex=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as p:
        for r,c,cl,xx in p.map(shard,[(mode,m) for m in lock['files']]):
            rr+=r;ct.update(c);cells.update(cl);ex+=xx
    if mode=='TRAIN':assert sorted(rr)==list(range(len(pl['roots'])))
    else:assert len(rr)==896 and {(r['id'],r['seed'],r['color']) for r in rr}=={(j['id'],s,c) for j in pl['jobs'] for s in SEEDS for c in (-1,1)}
    assert ct['cache_physical']==lock['counts']['physical'] and ct['cache_logical']==lock['counts']['logical']
    out=dict(rows=rr,proof=dict(ct),cells=dict(cells),exact=ex,lock_sha256=sha(T/(mode+'_LOCK.json')),counts=lock['counts'])
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print(mode,'AUDIT',len(rr),dict(ct),dict(cells),flush=True)
if __name__=='__main__':main(sys.argv[1])
