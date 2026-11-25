"""Prospective targeted follow-up: freeze distinct quiet families, no policy tuning."""
import pilot as p
import random, collections as C, json, sys
orig=p.D; orig_id=p.identity
p.D=orig/'quiet'
def identity():
    d=p.D;p.D=orig
    try:return orig_id()
    finally:p.D=d
p.identity=identity
def plan():
    old=p.old;fam,boards,locks=old.c4historical()
    fam.update(j['family'] for j in old.read('C06_PLAN.json')['c4'])
    for r in old.read('C06_COHORT.json')['roots']:
        if r['domain']=='C4':boards.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    broad=json.loads((orig/'COHORT.json').read_text())
    fam.update(j['family'] for j in broad['jobs'])
    for r in broad['roots']:boards.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    nf,nb=len(fam),len(boards);rng=random.Random(202611261701);jobs=[];roots=[]
    for i in range(192):
        while True:
            h=''.join(str(rng.randrange(1,8)) for _ in range(5));k=old.mirror(h)
            if k not in fam:break
        fam.add(k);seed=rng.randrange(2**63);r=random.Random(seed);op=h
        depth=(24,26,28)[i%3];group=('TRAIN','CAL','TEST')[i//64]
        b,z=old.c4replay(h)
        while len(h)<depth:
            side=len(h)%2;opts=[]
            for c in range(7):
                if z[c]==6:continue
                bit=1<<(7*c+z[c]);b[side]|=bit;z[c]+=1
                if not old.won(b[side]) and not any(old.canwin(b,z,x) for x in (0,1)):opts.append(c)
                b[side]^=bit;z[c]-=1
            if not opts:break
            c=r.choice(opts);b[side]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
        key=old.boardkey(old.c4flat(b));legal=[c for c in range(7) if z[c]<6]
        ok=len(h)==depth and len(legal)>=3 and key not in boards
        jobs.append(dict(id=i,group=group,family=k,opening=op,seed=seed,depth=depth,history=h,accepted=ok))
        if ok:
            assert not any(old.canwin(b,z,x) for x in (0,1))
            boards.add(key);roots.append(dict(id=i,group=group,family=k,history=h,legal=legal,depth=depth,tactical=False))
    p.save('COHORT.json',dict(identity=identity(),source_locks=locks,old_families=nf,old_boards=nb,broad_cohort=p.sha(orig/'COHORT.json'),seed=202611261701,jobs=jobs,roots=roots))
    print('QUIET_COHORT',len(roots),dict(C.Counter(r['group'] for r in roots)),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    else:getattr(p,sys.argv[1])()
