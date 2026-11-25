"""Independent 15x15 placement-game reference; not the native rule engine.
Only unrestricted freestyle/standard are admitted for proof/collection.
"""
from itertools import groupby
import hashlib,random
SIZE=15;N=225
RULES={'freestyle':0,'standard':1,'omok':2,'renju':3,'caro':4}
ADMITTED=('freestyle','standard')
LINES=[]
for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
    for r in range(15):
        for c in range(15):
            if 0<=r-dr<15 and 0<=c-dc<15:continue
            rr,cc=r,c;line=[]
            while 0<=rr<15 and 0<=cc<15:line.append(rr*15+cc);rr+=dr;cc+=dc
            LINES.append(tuple(line))
AT=[[] for _ in range(N)]
for line in LINES:
    for i,a in enumerate(line):AT[a].append((line,i))
def win_at(b,a,p,rule):
    assert rule in ADMITTED
    if b[a] not in (0,p):return False
    for line,i in AT[a]:
        x=[b[j] if j!=a else p for j in line]
        start=0
        for v,g in groupby(x):
            n=sum(1 for _ in g)
            if start<=i<start+n and v==p and (n>=5 if rule=='freestyle' else n==5):return True
            start+=n
    return False
def replay(h,rule):
    assert rule in ADMITTED
    b=[0]*N;p=1;winner=None
    for i,a in enumerate(h):
        if type(a) is not int or not 0<=a<N or b[a] or winner is not None:raise ValueError('invalid/post-terminal history')
        b[a]=1 if p==1 else 2
        if win_at(b,a,b[a],rule):winner=p
        elif i+1==N:winner=0
        p=-p
    return b,p,winner
def proof_reference(b,p,rule):
    """All legal placements, independently compute each successor's reply.
    For scalable verification the line-local opponent set lemma is used;
    optional audit checks the direct child witness as well.
    """
    x=1 if p==1 else 2
    legal=[a for a in range(N) if not b[a]]
    assert legal
    own=[a for a in legal if win_at(b,a,x,rule)]
    opp=[a for a in legal if win_at(b,a,3-x,rule)]
    rows=[]
    for a in legal:
        if a in own:lo,hi,rep=1,1,None
        elif len(legal)==1:lo,hi,rep=0,0,None
        else:
            rep=next((j for j in opp if j!=a),None)
            lo,hi=(-1,-1) if rep is not None else (-1,1)
        rows.append([a,lo,hi,rep])
    L=max(x[1] for x in rows);U=max(x[2] for x in rows)
    d=[max(0,max((v[2] for v in rows if v[0]!=a),default=-1)-l) for a,l,u,rep in rows]
    risk2=min(d);witness=next((v[0] for v,z in zip(rows,d) if z==0),None)
    return dict(kind='rule_two_ply_v1',own=own,opponent=opp,actions=rows,interval=[L,U],witness=witness,risk2=risk2)
def rule_key(b,p,rule):
    k=sum(v<<(2*i) for i,v in enumerate(b))|((p==1)<<450)|(RULES[rule]<<451)
    return f'{k:0128x}'
def features(h):
    # Same current-player reference in every time plane, independent of native
    # feature scratch or Zobrist hash. Earlier-than-initial positions empty.
    p=1 if len(h)%2==0 else 2
    out=[]
    for t in range(8):
        b=[0]*N
        for i,a in enumerate(h[:max(0,len(h)-t)]):b[a]=1+i%2
        out.extend(float(v==p) for v in b);out.extend(float(v==3-p) for v in b)
    out.extend([float(p==1)]*N)
    return out
def transform(a,k):
    r,c=divmod(a,15)
    if k>=4:c=14-c
    for _ in range(k%4):r,c=c,14-r
    return r*15+c
def orbit_history(h):return min(tuple(transform(a,k) for a in h) for k in range(8))
def orbit_board(b,p,rule):
    arr=[]
    for k in range(8):
        a=bytearray(N+2);a[-2]=1 if p==1 else 2;a[-1]=RULES[rule]
        for i,v in enumerate(b):a[transform(i,k)]=v
        arr.append(bytes(a))
    return min(arr)
def draw_history():
    # Four-direction pattern: no run of five even on the entire board.
    for offset in range(4):
        b=[1 if (r+2*c+offset)%4<2 else 2 for r in range(15) for c in range(15)]
        if b.count(1)==113:break
    assert b.count(1)==113 and b.count(2)==112
    a=[i for i,v in enumerate(b) if v==1];w=[i for i,v in enumerate(b) if v==2]
    r=random.Random(2026112506);r.shuffle(a);r.shuffle(w)
    h=[x for pair in zip(a,w) for x in pair]+[a[-1]]
    assert replay(h,'freestyle')[2]==0
    return h
def fixtures():
    # Four black stones, one end blocked by white -> only 109 completes.
    immediate=[105,104,106,0,107,2,108,4]
    # Black would connect three+two into six, white remote placements.
    overline=[106,0,107,2,108,4,110,6,111,8]
    # Opponent white has an open four at 100..103; black to move.
    double_loss=[0,100,2,101,4,102,6,103]
    # Same board, different ordered histories. Both nonterminal.
    a=[112,113,97,127,98,126,82,143]
    b=[97,127,112,113,98,126,82,143]
    # Cross black double-three: candidate 112.
    forbidden=[111,0,113,2,97,4,127,6]
    return dict(immediate=immediate,overline=overline,double_loss=double_loss,trans_a=a,trans_b=b,forbidden=forbidden,long=draw_history()[:150],draw_pre=draw_history()[:-1],draw=draw_history())
