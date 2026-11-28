"""Independent Rapfi chronological, proof, wave, target and score/cost checks."""
from plan59 import D,T,P,sha
import sys,json,gzip,collections as C,random
sys.path.insert(0,str(P))
import audit58 as a
def load(p):return json.loads(p.read_text())
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def game(g,j,arm,color):
    assert g['job']==j and g['policy']==arm and g['color']==color
    h=j['opening'].copy();events=[];c=C.Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,15);bd,g0=a.v.base(aa,bb,15)
        assert r['move'] in bd and r['player']==side and r['ply']==len(h) and r['arm']==('Q' if q else 'B')
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=96 if q else 128;rr=r
        if z:
            bds=a.replay_proof(h,15,z)
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                a.vote(h,15,r,True,cap);assert 'target' not in r
            else:
                vs=r['votes'];j0=max(range(len(vs)),key=lambda k:(vs[k][1],k))
                di=min((vs[j0][1]-nv+int(k<j0) for k,(m,nv) in enumerate(vs) if k!=j0),default=10**9)
                eligible=di<=8 and vs[0][0]!=vs[j0][0] and (arm=='REPEAT' or not events)
                assert ('target' in r)==eligible
                if eligible:
                    events.append(i);rr=a.rapfi_target(h,r,bds)
                    c['switches']+=r['target']['incumbent']!=r['move'];c['extra_nn']+=r['target']['extra_nn']
                a.vote(h,15,rr,True,cap,rr['original_move'])
                mv=a.v.changed(rr['original_move'],rr['votes'],bds,15)
                assert mv==rr['move'] and rr['replacement']==(mv!=rr['original_move'])
        else:
            assert 'target' not in r;a.vote(h,15,r,q,cap)
        a.wave_audit(rr,96,q)
        tag='Q' if q else 'B'
        for k in ('nn','cpu_ns','wall_ns'):assert type(r[k]) is int and r[k]>=0;c[tag+k]+=r[k]
        c[tag+'moves']+=1;c['moves']+=1;h.append(r['move'])
    aa,bb,w=a.v.replay(h,15)
    assert h==g['history'] and (w or aa|bb==a.v.G[15][1]) and w==g['winner']
    assert g['score2']==1+color*w and g['nn']==c['Qnn']+c['Bnn'] and events==g['events']
    c.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0),events=len(events))
    return c
def main():
    lock=load(T/'RAPFI_LOCK.json');plan=load(D/'RAPFI_PLAN.json')
    assert lock['source']==sha(D/'rapfi59.py') and lock['plan']==sha(D/'RAPFI_PLAN.json')
    ids={j['id']:j for j in plan['jobs']};seen=set();fams={i:{arm:C.Counter() for arm in ('FIRST','REPEAT')} for i in ids};counts=C.Counter();div=0
    for meta in lock['files']:
        p=T/meta['file'];assert sha(p)==meta['sha256'];ct=C.Counter()
        for gs in raw(p):
            f,r=gs['FIRST'],gs['REPEAT'];i=f['job']['id'];j=ids[i];color=f['color']
            assert (i,color) not in seen and color in j['color_order'];seen.add((i,color))
            for arm,g in gs.items():
                c=game(g,j,arm,color);fams[i][arm].update(c);ct['games']+=1;ct['moves']+=c['moves'];ct['nn']+=c['Qnn']+c['Bnn'];ct[arm+'_events']+=c['events']
            ct['episodes']+=1
            n=f['events'][0] if f['events'] else len(f['rows'])-1
            assert (r['events'][0] if r['events'] else None)==(f['events'][0] if f['events'] else None)
            for x,y in zip(f['rows'][:n+1],r['rows'][:n+1]):assert x['move']==y['move'] and x['nn']==y['nn'] and x['arm']==y['arm']
            if not f['events']:assert f['history']==r['history']
            div+=f['history']!=r['history']
        assert dict(ct)==meta['counts'];counts.update(ct)
    assert counts==lock['counts'] and seen=={(i,c) for i in ids for c in (-1,1)}
    def calc(c):return dict(score=c['score2']/(2*c['games']),saving=1-(c['Qnn']/c['Qmoves'])/(c['Bnn']/c['Bmoves']),cpu_saving=1-(c['Qcpu_ns']/c['Qmoves'])/(c['Bcpu_ns']/c['Bmoves']))
    def aggregate(ix):
        agg={arm:C.Counter() for arm in ('FIRST','REPEAT')}
        for i in ix:
            for arm in agg:agg[arm].update(fams[i][arm])
        return agg
    agg=aggregate(list(ids));p={arm:calc(c) for arm,c in agg.items()};rng=random.Random(202611285904);ds=[];dv=[]
    for _ in range(8000):
        x={arm:calc(c) for arm,c in aggregate(rng.choices(list(ids),k=16)).items()}
        ds.append(x['REPEAT']['score']-x['FIRST']['score']);dv.append(x['REPEAT']['saving']-x['FIRST']['saving'])
    out=dict(scope='exploratory; repeated TARGET, not NN STOP, no confirmatory gate',counts={arm:dict(c) for arm,c in agg.items()},points=p,paired_delta_score=p['REPEAT']['score']-p['FIRST']['score'],paired_delta_saving=p['REPEAT']['saving']-p['FIRST']['saving'],score_CI=[sorted(ds)[199],sorted(ds)[7799]],saving_CI=[sorted(dv)[199],sorted(dv)[7799]],trajectory_divergences=div,proof=dict(a.CT),lock_sha256=sha(T/'RAPFI_LOCK.json'))
    (T/'RAPFI_SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
