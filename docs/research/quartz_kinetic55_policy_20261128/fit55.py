"""Complete paired historical outcomes; fit only K53, other cohorts diagnostics."""
import gzip,json,collections as C,itertools
from k55_plan import D,load,save,sha
DIRS={51:'quartz_kinetic51_targeted_20261127',52:'quartz_kinetic52_confirm_20261127',53:'quartz_kinetic53_reuse_20261127',54:'quartz_kinetic54_cache_20261127'}
def cell(r):
    assert r['bin'] in (3,7)
    return 2*int(r['bin']==7)+int(r['ply']>8)
def read(cycle):
    d=D.parent/DIRS[cycle];p=d/'NN_LOCK.json';lk=json.loads(p.read_text())
    for f in lk['files']:
        q=d/f['file'];assert sha(q)==f['sha256']
        with gzip.open(q,'rt') as r:yield from map(json.loads,r)
def main():
    panels={};second=[C.Counter() for _ in range(4)]
    for cycle in DIRS:
        zs=[C.Counter() for _ in range(4)];ct=C.Counter()
        for x in read(cycle):
            c=x['control'];t=x['target' if cycle<=52 else 'cold'];e=c['event'];assert e==t['event']
            ct['episodes']+=1;ct['cold_delta2']+=t['score2']-c['score2']
            if e is None:
                assert c['history']==t['history']
                if cycle==53:assert c['history']==x['reuse']['history']
                continue
            a,b=c['rows'][e],t['rows'][e];assert a['low']==b['low'] and a['history']==b['history']
            z=zs[cell(a)];z['n']+=1;z['cold_gain2']+=t['score2']-c['score2'];z['cold_cost']+=b['target']['extra_nn']
            if cycle==53:
                r=x['reuse'];assert r['event']==e and r['rows'][e]['low']==a['low']
                z['reuse_gain2']+=r['score2']-c['score2'];z['reuse_cost']+=r['rows'][e]['target']['extra_nn']
            if cycle==54:
                g=x['conditional'];assert g['event']==e
                if len(g['events'])==2:
                    r=g['rows'][g['events'][1]];q=second[cell(r)];q['n']+=1;q['gain2']+=g['score2']-t['score2'];q['logical_cost']+=r['target']['extra_nn'];q['physical_cost']+=r['cache_receipt']['physical']-r['cache_receipt']['root_physical']
        panels[str(cycle)]=dict(counts=dict(ct),cells=[dict(z) for z in zs],lock=sha(D.parent/DIRS[cycle]/'NN_LOCK.json'))
    a=[C.Counter(z) for z in panels['53']['cells']];cap=sum(z['cold_cost'] for z in a)//2
    allrows=[]
    for t in itertools.product(range(3),repeat=4):
        if any(v and a[i]['n']<20 for i,v in enumerate(t)):continue
        gain=sum(a[i][('cold' if v==1 else 'reuse')+'_gain2'] for i,v in enumerate(t) if v)
        cost=sum(a[i][('cold' if v==1 else 'reuse')+'_cost'] for i,v in enumerate(t) if v)
        if cost<=cap:allrows.append(dict(table=t,gain2=gain,cost=cost,active=sum(v>0 for v in t)))
    best=min(allrows,key=lambda r:(-r['gain2'],r['cost'],r['active'],r['table']))
    assert best['gain2']>=0
    save('FROZEN.json',dict(table=best['table'],fit=best,cap=cap,feasible=len(allrows),candidates=allrows,panels=panels,second_cells=[dict(z) for z in second],scope='fit K53 logical local costs only; K51/52/54 cold and K54 second historical diagnostics, not fresh holdouts'))
    print('FIT',json.dumps(dict(best=best,cap=cap,feasible=len(allrows),panels=panels,second_cells=[dict(z) for z in second])),flush=True)
if __name__=='__main__':main()
