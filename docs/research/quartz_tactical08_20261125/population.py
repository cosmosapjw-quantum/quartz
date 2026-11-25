"""Fresh independent constructed legal roots; old pilot only calibration."""
import hashlib,gzip,json,random,collections as C
from rules import replay,ReplayCursor,proof_reference,orbit_board,fixtures,ADMITTED
from experiment import D,ROOT,ST,write,classify
def family(h,rule='freestyle'):
    b,p,w=replay(h[:6],rule);assert w is None
    return hashlib.sha256(orbit_board(b,p,rule)).hexdigest()
def construct(rng,st,side):
    rule='freestyle'
    center=[r*15+c for r in range(4,11) for c in range(4,11)]
    h=rng.sample(center,6);cur=ReplayCursor(h,rule)
    needed={1:[],-1:[]};reserved=set()
    if st!='QUIET':
        lines=[]
        for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
            for r in range(15):
                for c in range(15):
                    pts=[(r+i*dr,c+i*dc) for i in range(6)]
                    if all(0<=a<15 and 0<=b<15 for a,b in pts):
                        cells=[15*a+b for a,b in pts]
                        if not any(cur.board[a] for a in cells):lines.append(cells)
        if not lines:return None
        seq=rng.choice(lines);owner=side if st=='WIN' else -side
        needed[owner]=seq[1:5];reserved=set(seq)
        if st=='BLOCK':
            needed[side]=[rng.choice([seq[0],seq[5]])]
        for v in needed.values():rng.shuffle(v)
    L=rng.randrange(16,37)
    if (1 if L%2==0 else -1)!=side:L+=1
    while len(h)<L:
        p=cur.player
        if needed[p]:m=needed[p].pop()
        else:
            choices=[i for i,v in enumerate(cur.board) if v==0 and i not in reserved]
            if not choices:return None
            m=rng.choice(choices)
        cur.push(m);h.append(m)
        if cur.winner is not None:return None
    if any(needed.values()):return None
    b,p,w=cur.snapshot();assert p==side and w is None
    if classify(proof_reference(b,p,rule))!=st:return None
    return h
def main():
    root=ROOT/'docs/research/quartz_preflight07_20261125/P2_CORPUS'
    priorfam=set();priorboards=set();groups=C.defaultdict(lambda:C.defaultdict(list))
    for path in sorted(root.glob('shard_*/*.json.gz')):
        with gzip.open(path,'rt') as f:g=json.load(f)
        fam=g['job']['family'];h=list(g['job']['opening']);assert family(h)==fam
        priorfam.add(fam)
        for t in g['moves']:
            b,p,w=replay(h,'freestyle');key=orbit_board(b,p,'freestyle');priorboards.add(key)
            st=classify(t['native']['proof'])
            groups[fam][st].append(dict(history=list(h),rule='freestyle',family=fam,stratum=st,player=p,source_game=g['job']['game_id'],source_ply=len(h)))
            h.append(t['played_move'])
        b,p,w=replay(h,'freestyle');priorboards.add(orbit_board(b,p,'freestyle'))
    # Fixed ordering and balanced retrospective CAL selection independent of NN ranks.
    f=sorted(priorfam)
    def assign(i,counts):
        if i==len(f):return []
        for st in ST:
            if counts[st]<2 and groups[f[i]][st]:
                cc=counts.copy();cc[st]+=1;r=assign(i+1,cc)
                if r is not None:return [st]+r
        return None
    choices=assign(0,C.Counter())
    assert choices is not None,[(k,list(groups[k])) for k in f]
    cal=[]
    for i,(fam,st) in enumerate(zip(f,choices)):
        v=sorted(groups[fam][st],key=lambda r:hashlib.sha256(json.dumps(r['history']).encode()).hexdigest())[0]
        cal.append(dict(v,id=f'CAL_{i:02d}'))
    rng=random.Random(2026112508)
    test=[];attempts=C.Counter();newfam=set();newboards=set()
    for h in fixtures().values():
        if len(h)>=6:priorfam.add(family(h))
        b,p,w=replay(h,'freestyle');priorboards.add(orbit_board(b,p,'freestyle'))
    for st in ST:
        for side in (1,-1):
            for i in range(2):
                for _ in range(10000):
                    attempts[st]+=1;h=construct(rng,st,side)
                    if h is None:continue
                    fam=family(h);b,p,w=replay(h,'freestyle');key=orbit_board(b,p,'freestyle')
                    if fam in priorfam|newfam or key in priorboards|newboards:continue
                    newfam.add(fam);newboards.add(key);test.append(dict(id=f'TEST_{st}_{side}_{i}',history=h,rule='freestyle',family=fam,stratum=st,player=p,source='constructed_legal_tactical_root'));break
                else:raise RuntimeError('construction exhausted')
    # Interleave strata in fixed random order, independent of search.
    rng.shuffle(test)
    write('C07_COHORT.json',dict(CAL=cal,TEST=test,attempts=dict(attempts),excluded_families=len(priorfam),excluded_boards=len(priorboards),scope='CAL old pilot eight families; TEST fresh constructed legal tactical roots, not natural games'))
    print('SOURCE',len(cal),len(test),dict(attempts),'excluded',len(priorfam),len(priorboards),flush=True)
if __name__=='__main__':main()
