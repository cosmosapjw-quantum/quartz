from plan60 import D,T,P58,sha,read,raw
import sys,json,collections as C,random,ast,subprocess
sys.path.insert(0,str(P58))
import audit58 as a
def game(g,j,arm,color):
    assert g['job']==j and g['policy']==arm and g['color']==color and not g['events']
    h=j['opening'].copy();c=C.Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==color;aa,bb=a.v.orient(h,15);bd,g0=a.v.base(aa,bb,15)
        assert r['move'] in bd and r['player']==side and r['ply']==len(h) and r['arm']==('Q' if q else 'B')
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=int(arm[1:]) if q else 128
        assert 'target' not in r
        if z:
            bds=a.replay_proof(h,15,z)
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                a.vote(h,15,r,True,cap)
            else:
                a.vote(h,15,r,True,cap,r['original_move'])
                mv=a.v.changed(r['original_move'],r['votes'],bds,15);assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
        else:a.vote(h,15,r,q,cap)
        a.wave_audit(r,int(arm[1:]),q)
        tag='Q' if q else 'B'
        for k in ('nn','cpu_ns','wall_ns'):assert type(r[k]) is int and r[k]>=0;c[tag+k]+=r[k]
        c[tag+'moves']+=1;c['moves']+=1;h.append(r['move'])
    aa,bb,w=a.v.replay(h,15);assert h==g['history'] and (w or aa|bb==a.v.G[15][1]) and w==g['winner']
    assert g['score2']==1+color*w and g['nn']==c['Qnn']+c['Bnn']
    c.update(games=1,score2=g['score2'],wins=int(g['score2']==2),draws=int(g['score2']==1),losses=int(g['score2']==0))
    return c
def main():
    plan=read(D/'RAPFI_PLAN.json');assert sha(D/'RAPFI_PLAN.json')==sha(T/'RAPFI_PLAN.json');assert sha(plan['binary'])==plan['binary_sha256']
    log=(T/'RAPFI.log').read_text();logged={}
    for l in log.splitlines():
        if l.startswith('RAPFI SHARD '):
            _,_,i,s=l.split(' ',3);logged[int(i)]=ast.literal_eval(s)
    assert set(logged)==set(range(4)) and 'FileNotFoundError' in log
    executed=subprocess.check_output(['git','show','f38eac27:docs/research/quartz_kinetic60_voc_20261128/rapfi60.py'])
    import hashlib
    assert hashlib.sha256(executed).hexdigest()==sha(T/'rapfi60.py')
    ids={j['id']:j for j in plan['jobs']};seen=set();fams={i:{arm:C.Counter() for arm in ('Q80','Q96')} for i in ids};counts=C.Counter();files=[];div=0
    for i in range(4):
        p=T/f'RAPFI_{i}.jsonl.gz';ct=C.Counter()
        for gs in raw(p):
            f,r=gs['Q80'],gs['Q96'];j=ids[f['job']['id']];color=f['color']
            assert set(gs)=={'Q80','Q96'} and (j['id'],color) not in seen;seen.add((j['id'],color))
            for arm,g in gs.items():
                c=game(g,j,arm,color);fams[j['id']][arm].update(c);ct['games']+=1;ct['moves']+=c['moves'];ct['nn']+=c['Qnn']+c['Bnn'];ct[arm+'_events']+=0
            ct['episodes']+=1;div+=f['history']!=r['history']
        assert dict(ct)==logged[i];counts.update(ct);files.append(dict(file=p.name,sha256=sha(p),counts=dict(ct)))
    assert seen=={(i,c) for i in ids for c in (-1,1)} and counts['games']==64
    lock=dict(files=files,counts=dict(counts),plan=sha(D/'RAPFI_PLAN.json'),executed_source=sha(T/'rapfi60.py'),executed_source_commit='f38eac27',status='reconstructed after original manifest-path error; all shard logs and complete game replay agree')
    (T/'RAPFI_LOCK.json').write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    def calc(c):return dict(score=c['score2']/(2*c['games']),saving=1-(c['Qnn']/c['Qmoves'])/(c['Bnn']/c['Bmoves']),cpu_saving=1-(c['Qcpu_ns']/c['Qmoves'])/(c['Bcpu_ns']/c['Bmoves']))
    def aggregate(ix):
        agg={arm:C.Counter() for arm in ('Q80','Q96')}
        for i in ix:
            for arm in agg:agg[arm].update(fams[i][arm])
        return agg
    agg=aggregate(list(ids));p={arm:calc(c) for arm,c in agg.items()};rng=random.Random(202611286005);ds=[];dv=[]
    for _ in range(8000):
        x={arm:calc(c) for arm,c in aggregate(rng.choices(list(ids),k=16)).items()}
        ds.append(x['Q80']['score']-x['Q96']['score']);dv.append(x['Q80']['saving']-x['Q96']['saving'])
    out=dict(scope='exploratory fixed cap Q80 vs Q96, each vs same B128',counts={arm:dict(c) for arm,c in agg.items()},points=p,paired_delta_score=p['Q80']['score']-p['Q96']['score'],paired_delta_saving=p['Q80']['saving']-p['Q96']['saving'],score_CI=[sorted(ds)[199],sorted(ds)[7799]],saving_CI=[sorted(dv)[199],sorted(dv)[7799]],trajectory_divergences=div,proof=dict(a.CT),lock_sha256=sha(T/'RAPFI_LOCK.json'))
    (T/'RAPFI_SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
