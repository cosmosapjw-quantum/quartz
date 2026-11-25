from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
from itertools import product
import json,gzip,math,hashlib,subprocess
from publish import D,ROOT,BASE
# Integer threshold proof, actual finite domain n<=4096.
n=0
for r in range(1,226):
    k=1 if r==1 else (r*r+3)//4+1
    assert max(1,math.isqrt(4*(k-1)))>=r;n+=1
    if k>1:assert max(1,math.isqrt(4*(k-2)))<r;n+=1
    for v in (1,8,32,64,128,256,4096):
        assert (r<=max(1,math.isqrt(4*(v-1))))==(v>=k);n+=1
for e,v,s in product(range(1,8),repeat=3):
    if not s<=v<=e:continue
    a,b,c=F(e,8),F(v,e),F(s,v)
    assert a*b*c==F(s,8)
    assert 1-a*b*c==(1-a)+a*(1-b)+a*b*(1-c);n+=2
# Finite two-ply interval construction: unique block optimal; double threat all loss.
for num in range(2,10):
    for own in range(1<<num):
        for opp in (0,1,2,3, (1<<num)-1):
            a=[(1,1) if own>>i&1 else (-1,-1) if opp&~(1<<i) else (-1,1) for i in range(num)]
            w=[i for i,(l,u) in enumerate(a) if l>=max((v[1] for j,v in enumerate(a) if j!=i),default=-1)]
            if own:assert all(i in w for i in range(num) if own>>i&1)
            elif opp.bit_count()==1:assert w==[opp.bit_length()-1]
            elif opp.bit_count()>1:assert len(w)==num
            else:assert not w
            n+=1
P=ROOT/'docs/research/quartz_preflight07_20261125/P2_CORPUS'
by=defaultdict(Counter);ranks=defaultdict(list);sha={};family=set()
for p in sorted(P.glob('shard_*/*.json.gz')):
    rel=str(p.relative_to(ROOT));blob=subprocess.check_output(['git','show',BASE+':'+rel]);assert blob==p.read_bytes()
    sha[rel]=hashlib.sha256(blob).hexdigest()
    with gzip.open(p,'rt') as f:g=json.load(f)
    family.add(g['job']['family'])
    for row in g['moves']:
        a=row['native'];pr=a['proof'];own=pr['own'];opp=pr['opponent']
        st='WIN' if own else 'BLOCK' if len(opp)==1 else 'LOSS' if len(opp)>1 else 'QUIET'
        c=by[st,str(row['player'])];c['n']+=1
        t=set(own if st=='WIN' else opp if st=='BLOCK' else [])
        if not t:continue
        es=a['result']['edges'];rank=min(i+1 for i,(m,p0) in enumerate(a['result']['candidates']) if m in t)
        width=len(es);v=a['visits'];expected=min(len(a['legal']),max(1,math.isqrt(4*(v-1))))
        assert width==expected,(width,expected);n+=1
        E=any(e['move'] in t for e in es);V=any(e['move'] in t and e['n']>0 for e in es);S=a['best'] in t;played=row['played_move'] in t
        assert E==(rank<=width) and (not S or V) and (not V or E);n+=1
        c['E']+=E;c['V']+=V;c['S']+=S;c['played']+=played
        ranks[st,str(row['player'])].append(rank)
        for cap in (8,32,64,128,256):
            c[f'potential_exposed_{cap}']+=rank<=max(1,math.isqrt(4*(cap-1)))
out={'assertions':n,'sources':sha,'families':len(family),'rows':{'/'.join(k):dict(v) for k,v in by.items()},'ranks':{'/'.join(k):v for k,v in ranks.items()},'scope':'retrospective pilot, rank-based potential exposure only at larger budgets, not new actual search'}
(D/'R07_RETROSPECTIVE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('Exact finite assertions',n)
for k,v in out['rows'].items():print(k,v)
