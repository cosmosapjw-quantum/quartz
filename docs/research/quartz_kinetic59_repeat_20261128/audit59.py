"""Independent chronological replay, native proof/visit/cache audit, family accounting."""
import collections as C, gzip,json,sys, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from plan59 import D,T,P,SEEDS,ARMS,sha,family
sys.path.insert(0,str(P))
import audit58 as a
def load(p):return json.loads(p.read_text())
def read(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def same(x,y):
    def sem(z):
        if isinstance(z,dict):return {k:sem(v) for k,v in z.items() if not k.endswith('_ns')}
        if isinstance(z,list):return [sem(v) for v in z]
        return z
    return sem(x)==sem(y)
def audit_game(g,j,policy,seed,color):
    assert g['job']==j and g['policy']==policy and g['seed']==seed and g['color']==color
    h=list(j['opening']); attempts=[]; events=[]; c=C.Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color
        aa,bb=a.v.orient(h,7);bd,g0=a.v.base(aa,bb,7)
        assert r['history']==h and r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B') and r['lock']==q
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=20 if q else 32
        if z:
            bds=a.replay_proof(h,7,z)
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                a.vote(h,7,r,True,cap)
                assert 'early' not in r
            else:
                should_attempt=policy=='REPEAT' or not attempts
                assert ('early' in r)==should_attempt
                if should_attempt:
                    attempts.append(i)
                    reached,_=a.inspect_early(h,r,bds,j,'CONTROL' if policy=='CONTROL' else 'STOP')
                    if reached:events.append(i)
                    cap=r['cap']
                else:assert 'low' not in r
                a.vote(h,7,r,True,cap,r['original_move'])
                mv=a.v.changed(r['original_move'],r['votes'],bds,7);assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
        else:
            assert 'early' not in r
            a.vote(h,7,r,q,cap)
        assert 'target' not in r
        a.cache_check(h,r)
        tag='Q' if q else 'B'
        for key,f in [('nn','actual_nn'),('logical','nn'),('cpu','cpu_ns'),('wall','wall_ns')]:
            assert type(r[f]) is int and r[f]>=0;c[tag+key]+=r[f]
        c[tag+'moves']+=1;c['moves']+=1;h.append(r['move'])
    aa,bb,w=a.v.replay(h,7)
    assert h==g['history'] and (w or (aa|bb)==a.v.G[7][1])
    assert g['winner']==w and g['score2']==1+color*w
    assert g['nn']==c['Qlogical']+c['Blogical'] and g['actual_nn']==c['Qnn']+c['Bnn']
    assert attempts==g['attempts'] and events==g['events']
    c.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),attempts=len(attempts),events=len(events))
    return dict(c)
def check_pairs(gs):
    f,r,c=gs['FIRST'],gs['REPEAT'],gs['CONTROL']
    i=f['attempts'][0] if f['attempts'] else None
    assert (r['attempts'][0] if r['attempts'] else None)==i
    assert (c['attempts'][0] if c['attempts'] else None)==i
    if i is None:
        assert same(f['rows'],r['rows']) and same(f['rows'],c['rows'])
    else:
        assert same(f['rows'][:i+1],r['rows'][:i+1]) and same(f['rows'][:i],c['rows'][:i])
        assert c['rows'][i]['early']['snapshots'][:len(f['rows'][i]['early']['snapshots'])]==f['rows'][i]['early']['snapshots']
        assert c['rows'][i]['early']['at']==f['rows'][i]['early']['at']
        assert c['events']==f['events']
    if len(r['attempts'])<=1:assert same(f['rows'],r['rows'])
    # First-only vs repeat: at common history, unchanged B moves and certificate decisions;
    # after extra STOP events, path may diverge. Measure observed difference, not counterfactual suffix.
    div=None;local=0;common_extra=0
    for k,(x,y) in enumerate(zip(f['rows'],r['rows'])):
        assert x['history']==y['history']
        if k!=i and 'early' in y:
            assert y['arm']=='Q' and 'early' not in x
            common_extra+=1
            assert y['actual_nn']<=x['actual_nn']
            local+=y['actual_nn']-x['actual_nn']
            if not y['early']['reached']:assert x['move']==y['move'] and x['votes']==y['votes'] and x['nn']==y['nn']
        else:assert same(x,y)
        if x['move']!=y['move']:
            assert k!=i and y['early']['reached'];div=k;break
    if div is None:assert f['history']==r['history'] and r['score2']==f['score2']
    return dict(divergence=div,common_extra=common_extra,common_local_nn=local,delta2=r['score2']-f['score2'],delta_nn=r['actual_nn']-f['actual_nn'],extra_attempts=len(r['attempts'])-len(f['attempts']),extra_events=len(r['events'])-len(f['events']))
def shard(meta):
    a.CT.clear();a.EX.clear()
    p=T/meta['file'];assert sha(p)==meta['sha256'];plan=load(D/'PLAN.json')
    ct=C.Counter();results=[];expected={j['id']:j for j in plan['jobs']}
    for gs in read(p):
        assert set(gs)==set(ARMS);g=gs['CONTROL'];j=expected[g['job']['id']];s=g['seed'];color=g['color']
        ag={arm:audit_game(gs[arm],j,arm,s,color) for arm in ARMS}
        pair=check_pairs(gs)
        results.append(dict(id=j['id'],seed=s,color=color,horizon=j['horizon'],geometry=j['geometry'],arms=ag,pair=pair))
        for arm,c in ag.items():
            ct['games']+=1;ct['physical']+=c['Qnn']+c['Bnn'];ct['logical']+=c['Qlogical']+c['Blogical'];ct['moves']+=c['moves'];ct[arm+'_attempts']+=c['attempts'];ct[arm+'_events']+=c['events']
        ct['episodes']+=1
    assert dict(ct)==meta['counts'],(ct,meta['counts'])
    return results,dict(a.CT)
def main():
    lock=load(T/'NN_LOCK.json');assert lock['plan']==sha(D/'PLAN.json') and lock['source']==sha(D/'experiment59.py') and lock['policy']==sha(D/'policy59.py')
    plan=load(D/'PLAN.json')
    assert len({family(j['opening']) for j in plan['jobs']})==64
    assert len(lock['files'])==28
    allrows=[];proof=C.Counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
        for rs,ct in ex.map(shard,lock['files']):
            allrows+=rs;proof.update(ct)
    assert len(allrows)==896 and {(r['id'],r['seed'],r['color']) for r in allrows}=={(j['id'],s,c) for j in plan['jobs'] for s in SEEDS for c in (-1,1)}
    assert proof['cache_physical']==lock['counts']['physical'] and proof['cache_logical']==lock['counts']['logical']
    out=dict(episodes=sorted(allrows,key=lambda r:(r['id'],r['seed'],r['color'])),proof=dict(proof),lock_sha256=sha(T/'NN_LOCK.json'),counts=lock['counts'])
    with gzip.open(T/'AUDIT.json.gz','wt',compresslevel=6) as f:json.dump(out,f,allow_nan=False,separators=(',',':'))
    print('AUDIT',len(allrows),dict(proof),lock['counts'],flush=True)
if __name__=='__main__':main()
