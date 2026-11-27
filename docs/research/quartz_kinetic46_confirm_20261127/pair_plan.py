import json,gzip,random,collections as C
from k46_plan import D,ROOT,SEED,load,save,sha
K45=D.parent/'quartz_kinetic45_sequential_20261127';K42=D.parent/'quartz_kinetic42_threshold_20261127'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def orient(h):
    a=sum(1<<m for m in h[::2]);b=sum(1<<m for m in h[1::2])
    return (b,a) if len(h)%2 else (a,b)
def make():
    lk=json.loads((K45/'NN_TEST_LOCK.json').read_text());seen={};allct=C.Counter();source={}
    for fi in lk['files']:
        p=K45/fi['file'];assert sha(p)==fi['sha256'];source[str(p.relative_to(ROOT))]=fi['sha256']
        for g in raw(p):
            for a in g['atlas']:
                h=a['history'];q=a['options']['20'];b=a['B']['move'];m=q['move'];l,u=q['gain2']
                allct['roots']+=1
                if b==m or l==u:continue
                allct['ambiguous_changed']+=1
                bd={i:(x,y) for i,x,y in a['proof']['bounds']};aa,bb=orient(h)
                oa=a['proof']['cert']==m;ob=all(bd[b][0]>=y for i,(x,y) in bd.items() if i!=b)
                key=(aa,bb,m,b,bd[m],bd[b],oa,ob)
                ref=dict(job=g['job']['id'],seed=g['seed'],ply=len(h),game_color=g['color'])
                if key not in seen:seen[key]=dict(lane='EARLY',a=aa,b=bb,history=h,actions=[m,b],input=[list(bd[m]),list(bd[b])],optimal=[oa,ob],gain2=[l,u],refs=[])
                seen[key]['refs'].append(ref)
    jobs=list(seen.values());rng=random.Random(SEED+62)
    lp=K42/'BENCHMARK.jsonl.gz';source[str(lp.relative_to(ROOT))]=sha(lp)
    for i,r in enumerate(raw(lp)):
        h=r['job']['opening'];a,b=orient(h);assert (a,b)==(r['a'],r['b'])
        E=[m for m in range(49) if not (a|b)>>m&1];assert len(E)>=2
        m,n=rng.sample(E,2)
        jobs.append(dict(lane='LATE',a=a,b=b,history=h,actions=[m,n],input=[[-1,1],[-1,1]],optimal=[False,False],gain2=[-2,2],ref=r['job']))
    for i,j in enumerate(jobs):j['id']=i
    assert len([j for j in jobs if j['lane']=='LATE'])==384
    save('PAIR_PLAN.json',dict(jobs=jobs,counts=dict(allct,early_unique=len(seen),late=384),source=source,query_cap=100000,independent_cap=200000,scope='historical TEST ambiguous changed Q20 vs B32, all eligible; dedup exact oriented pair+initial bounds+order; late validation independent truth not used for selection'))
    print('PAIR_PLAN',dict(allct),len(seen),384)
if __name__=='__main__':make()
