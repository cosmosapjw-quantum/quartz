"""Generate pre-outcome canonical late Gomoku7 roots and Gomoku15 smoke boards."""
import random,hashlib,json
from pathlib import Path
from functools import lru_cache
D=Path(__file__).resolve().parent
def lines(size,k):
    ans=[]
    for y in range(size):
        for x in range(size):
            for dx,dy in [(1,0),(0,1),(1,1),(1,-1)]:
                if 0<=x+(k-1)*dx<size and 0<=y+(k-1)*dy<size:
                    ans.append(sum(1<<(size*(y+i*dy)+x+i*dx) for i in range(k)))
    return tuple(ans)
L=lines(7,4);FULL=(1<<49)-1
BY=tuple(tuple(mask for mask in L if mask&(1<<i)) for i in range(49))
def win(bits,a):
    return any(bits&m==m for m in BY[a])
def key(b,n):
    variants=[]
    for swap in (0,1):
        for rx in (0,1):
            for ry in (0,1):
                t=[]
                for y in range(n):
                    for x in range(n):
                        u,v=(y,x) if swap else (x,y)
                        if rx:u=n-1-u
                        if ry:v=n-1-v
                        t.append(b[v*n+u]+1)
                variants.append(bytes(t))
    return hashlib.sha256(min(variants)).hexdigest()
def snapshot(hist,n):
    b=[0]*(n*n)
    for j,i in enumerate(hist):
        assert b[i]==0
        b[i]=1 if j%2==0 else -1
    return b
def generate():
    seen=set();rr=[];attempts={}
    for split,seed in [('TRAIN',20261124071),('HELDOUT',20261124072)]:
        rng=random.Random(seed)
        for ply in [39,41]:
            n=0;tries=0
            while n<12:
                tries+=1
                if tries>200000:raise RuntimeError('source construction exhausted')
                h=[];bs=[0,0];empty=list(range(49))
                for j in range(ply):
                    p=j%2
                    moves=[a for a in empty if not win(bs[p]|(1<<a),a)]
                    if not moves:break
                    a=rng.choice(moves);bs[p]|=1<<a;empty.remove(a);h.append(a)
                if len(h)!=ply:continue
                b=snapshot(h,7);k=key(b,7)
                if k in seen:continue
                seen.add(k);n+=1
                rr.append(dict(id=k,split=split,game='gomoku7',ply=ply,history=h,board=b,player=-1 if ply%2 else 1))
            attempts[split+str(ply)]=tries
    rng=random.Random(20261124073)
    l15=lines(15,5)
    for i,ply in enumerate([8,16,24,32]):
        while True:
            h=rng.sample(range(225),ply);b=snapshot(h,15)
            masks=[sum(1<<a for a,v in enumerate(b) if v==p) for p in [1,-1]]
            if not any(bits&m==m for bits in masks for m in l15):break
        rr.append(dict(id=key(b,15),split='SMOKE',game='gomoku15',ply=ply,history=h,board=b,player=1))
    (D/'sources.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rr))
    print(json.dumps({'train':24,'heldout':24,'smoke':4,'attempts':attempts,'source_sha256':hashlib.sha256((D/'sources.jsonl').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':generate()
