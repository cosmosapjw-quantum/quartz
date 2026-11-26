import pathlib,json,hashlib,random,importlib.util,collections
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
assert D.name=='quartz_kinetic41_exact_20261126'
SEED=2026112641;SEEDS=(101,102,103,105);FIT=(101,103,105);WORKERS=8
H=(20,26,32,64);BUDGET=25000
spec=importlib.util.spec_from_file_location('_k33_rules',D.parent/'quartz_kinetic33_20261126/proof.py')
rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def family(h):
    z=[]
    for t in range(8):
        out=[]
        for a in h:
            r,c=divmod(a,7)
            if t>=4:c=6-c
            for _ in range(t%4):r,c=c,6-r
            out.append(r*7+c)
        z.append(tuple(sorted(out[::2])+sorted(out[1::2])))
    return min(z)
def save(name,x):
    p=D/name;assert not p.exists(),p;p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(name):return json.loads((D/name).read_text())
lines=[tuple(rule.cells(l)) for l in rule.LINES]
by=[[i for i,l in enumerate(lines) if m in l] for m in range(49)]
def board_candidate(rng,e):
    n=49-e
    board=[1]*((n+1)//2)+[2]*(n//2)+[0]*e;rng.shuffle(board)
    def penalty(i):
        l=lines[i];x=[board[k] for k in l]
        # Penalize existing 4 and immediate win threats. No exact labels.
        a=x.count(1);b=x.count(2);z=x.count(0)
        return 6*int(a==4 or b==4)+int(z==1 and (a==3 or b==3))
    costs=[penalty(i) for i in range(len(lines))];total=sum(costs)
    for step in range(2500):
        if total==0:break
        i,j=rng.sample(range(49),2)
        if board[i]==board[j]:continue
        affected=set(by[i])|set(by[j]);old=sum(costs[k] for k in affected)
        board[i],board[j]=board[j],board[i];new={k:penalty(k) for k in affected};delta=sum(new.values())-old
        temp=0.5 if step<1500 else 0.1
        if delta<=0 or rng.random()<__import__('math').exp(-delta/temp):
            total+=delta
            for k,v in new.items():costs[k]=v
        else:board[i],board[j]=board[j],board[i]
    a=sum(1<<i for i,v in enumerate(board) if v==1);b=sum(1<<i for i,v in enumerate(board) if v==2)
    if rule.won(a) or rule.won(b):return None
    aa=[i for i,v in enumerate(board) if v==1];bb=[i for i,v in enumerate(board) if v==2]
    rng.shuffle(aa);rng.shuffle(bb);h=[]
    for i in range(len(aa)):
        h.append(aa[i])
        if i<len(bb):h.append(bb[i])
    return h,total

def plan():
    paths=[D.parent/f/'PLAN.json' for f in ('quartz_kinetic33_20261126','quartz_kinetic38_nnlock_20261126','quartz_kinetic39_voc_20261126','quartz_kinetic40_budget_20261126')]
    old=[json.loads(p.read_text()) for p in paths];seen={family(j['opening']) for x in old for j in x['jobs']}
    prior=len(seen);rng=random.Random(SEED);jobs=[];models=old[-1]['models'];ct=collections.Counter()
    for m in models.values():assert sha(pathlib.Path(m['path']))==m['sha256']
    for split,N in [('TRAIN',120),('CAL',60),('TEST',60),('LATE',64)]:
        for i in range(N):
            e=8+i%5
            while True:
                ct['attempts']+=1
                if ct['attempts']>6000:raise RuntimeError(f'frame admission ceiling {dict(ct)}')
                r=board_candidate(rng,e)
                if r is None:ct['placement_dead_end']+=1;continue
                h,penalty=r
                ct['quiet' if penalty==0 else 'nonquiet']+=1
                a,b=rule.from_history(h);f=family(h)
                if f in seen:ct['duplicate']+=1;continue
                p=rule.root_certificate(a,b) if len(h)%2==0 else rule.root_certificate(b,a)
                if p['move'] is not None:ct['gated']+=1;continue
                break
            seen.add(f);ct['admitted']+=1
            jobs.append(dict(id=len(jobs),split=split,opening=h,family=f,empties=e,color_order=rng.sample([1,-1],2)))
    for i in range(64):
        cells=[7*r+c for r in range(1,6) for c in range(1,6)] if i%2==0 else list(range(49))
        while True:
            h=rng.sample(cells,4);f=family(h)
            if f not in seen:break
        seen.add(f);jobs.append(dict(id=len(jobs),split='EARLY',opening=h,family=f,empties=45,color_order=rng.sample([1,-1],2)))
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,fit=list(FIT),heldout=102,workers=WORKERS,horizons=H,proof_node_budget=BUDGET,budget_fraction='7/10',margin=.05,prior_families=prior,prior_locks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],generation_counts=dict(ct),scope='count-preserving local repair of random legal nonterminal late boards, K33 unresolved filter independent of exact outcomes; 64 late and 64 early actual game starts; fixed checkpoints'))
    print('PLAN',len(jobs),dict(ct), 'prior',prior)
if __name__=='__main__':plan()
