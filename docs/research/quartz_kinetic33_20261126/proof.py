"""Exact unrestricted 7x7 four-in-row rule bounds. No evaluator dependency."""
N=7;K=4;FULL=(1<<49)-1
LINES=tuple(sum(1<<((r+d*dr)*N+c+d*dc) for d in range(K)) for r in range(N) for c in range(N)
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)) if 0<=r+(K-1)*dr<N and 0<=c+(K-1)*dc<N)
BYCELL=tuple(tuple(l for l in LINES if l>>i&1) for i in range(49))
def cells(b):
    while b:
        t=b&-b;yield t.bit_length()-1;b^=t
def from_history(h):
    a=b=0
    for i,m in enumerate(h):
        assert 0<=m<49 and not ((a|b)>>m&1)
        assert not won(a) and not won(b)
        if i%2==0:a|=1<<m
        else:b|=1<<m
    return a,b
def won(a):return any(a&l==l for l in LINES)
def immediate(a,b):
    out=0
    for l in LINES:
        if not l&b and (l&a).bit_count()==K-1:out|=l&~a
    return out
def root_certificate(a,b):
    assert not a&b and not won(a) and not won(b)
    empty=FULL^(a|b);assert empty
    rows=[];scans=0
    for i in cells(empty):
        x=a|1<<i;scans+=len(BYCELL[i])
        if sum(x&l==l for l in BYCELL[i]):lo=hi=1;kind='WIN1';threats=[]
        elif x|b==FULL:lo=hi=0;kind='DRAW';threats=[]
        else:
            scans+=len(LINES);opp=immediate(b,x)
            if opp:lo=hi=-1;kind='LOSE2';threats=list(cells(opp))
            else:
                scans+=len(LINES);own=immediate(x,b)
                threats=list(cells(own))
                if own.bit_count()>=2:lo=hi=1;kind='WIN3'
                else:lo,hi=-1,1;kind='UNKNOWN'
        rows.append([i,lo,hi,kind,threats])
    best_l=max(z[1] for z in rows);allowed=[z[0] for z in rows if z[2]>=best_l]
    w=[z for z in rows if all(t[0]==z[0] or z[1]>=t[2] for t in rows)]
    if w:
        def rank(z):
            m=z[0];return (-z[1],z[3]!='WIN1',abs(m//7-3)+abs(m%7-3),m)
        chosen=min(w,key=rank);move=chosen[0];kind=chosen[3]
    else:move=None;kind='UNRESOLVED'
    return dict(move=move,kind=kind,allowed=allowed,interval=[best_l,max(z[2] for z in rows)],bounds=rows,rule_scans=scans)
