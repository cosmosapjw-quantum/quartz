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
    exact=rule=='standard'
    for line,i in AT[a]:
        n=1;j=i-1
        while j>=0 and b[line[j]]==p:n+=1;j-=1
        j=i+1
        while j<len(line) and b[line[j]]==p:n+=1;j+=1
        if n==5 or not exact and n>5:return True
    return False
class ReplayCursor:
    def __init__(self,h,rule):
        assert rule in ADMITTED
        self.rule=rule;self.board=[0]*N;self.player=1;self.winner=None;self.ply=0
        for a in h:self.push(a)
    def push(self,a):
        b=self.board;p=self.player
        if type(a) is not int or not 0<=a<N or b[a] or self.winner is not None:raise ValueError('invalid/post-terminal history')
        b[a]=1 if p==1 else 2;self.ply+=1
        if win_at(b,a,b[a],self.rule):self.winner=p
        elif self.ply==N:self.winner=0
        self.player=-p
    def snapshot(self):return self.board.copy(),self.player,self.winner
def replay(h,rule):return ReplayCursor(h,rule).snapshot()
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
    rows=[];own_set=set(own)
    for a in legal:
        if a in own_set:lo,hi,rep=1,1,None
        elif len(legal)==1:lo,hi,rep=0,0,None
        else:
            rep=opp[0] if opp and opp[0]!=a else opp[1] if len(opp)>1 else None
            lo,hi=(-1,-1) if rep is not None else (-1,1)
        rows.append([a,lo,hi,rep])
    L=max(x[1] for x in rows);U=max(x[2] for x in rows)
    # Track largest upper endpoint and largest endpoint on a different action.
    hi1=-2;hi2=-1;idx=-1
    for a,l,u,rep in rows:
        if u>hi1:hi2=hi1;hi1=u;idx=a
        elif u>hi2:hi2=u
    d=[max(0,(max(-1,hi2) if a==idx else hi1)-l) for a,l,u,rep in rows]
    risk2=min(d);witness=next((v[0] for v,z in zip(rows,d) if z==0),None)
    return dict(kind='rule_two_ply_v1',own=own,opponent=opp,actions=rows,interval=[L,U],witness=witness,risk2=risk2)
def rule_key(b,p,rule):
    k=sum(v<<(2*i) for i,v in enumerate(b))|((p==1)<<450)|(RULES[rule]<<451)
    return f'{k:0128x}'
def features(h):
    p=1 if len(h)%2==0 else 2
    out=[0.0]*(17*N)
    for i,a in enumerate(h):
        color=1+i%2;offset=0 if color==p else N
        for t in range(min(8,len(h)-i)):out[t*2*N+offset+a]=1.0
    if p==1:out[-N:]=[1.0]*N
    return out
def transform(a,k):
    r,c=divmod(a,15)
    if k>=4:c=14-c
    for _ in range(k%4):r,c=c,14-r
    return r*15+c
D4=tuple(tuple(transform(a,k) for a in range(N)) for k in range(8))
def orbit_history(h):return min(tuple(m[a] for a in h) for m in D4)
def orbit_board(b,p,rule):
    arr=[]
    for m in D4:
        a=bytearray(N+2);a[-2]=1 if p==1 else 2;a[-1]=RULES[rule]
        for i,v in enumerate(b):a[m[i]]=v
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
