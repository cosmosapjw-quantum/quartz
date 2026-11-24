"""Two independent offline exact Gomoku7 solvers; no runtime use."""
from functools import lru_cache
from population import BY, FULL, win
def solve_bit(board,player):
    p=sum(1<<i for i,x in enumerate(board) if x==player)
    q=sum(1<<i for i,x in enumerate(board) if x==-player)
    @lru_cache(None)
    def val(a,b):
        empty=FULL^(a|b)
        if not empty:return 0
        best=-1
        while empty:
            bit=empty&-empty;empty^=bit;i=bit.bit_length()-1
            z=1 if win(a|bit,i) else -val(b,a|bit)
            if z==1:return 1
            best=max(best,z)
        return best
    return {i:(1 if win(p|(1<<i),i) else -val(q,p|(1<<i))) for i,x in enumerate(board) if x==0}
def line_win(board,i,p):
    x=i%7;y=i//7
    for dx,dy in [(1,0),(0,1),(1,1),(1,-1)]:
        n=1
        for s in [-1,1]:
            u,v=x+s*dx,y+s*dy
            while 0<=u<7 and 0<=v<7 and board[7*v+u]==p:
                n+=1;u+=s*dx;v+=s*dy
        if n>=4:return True
    return False
def solve_grid(board,player):
    @lru_cache(None)
    def val(b,p):
        best=-1;found=False
        for i,x in enumerate(b):
            if x:continue
            found=True;after=b[:i]+(p,)+b[i+1:]
            z=1 if line_win(after,i,p) else -val(after,-p)
            if z==1:return 1
            best=max(best,z)
        return best if found else 0
    b=tuple(board)
    return {i:(1 if line_win(b[:i]+(player,)+b[i+1:],i,player) else -val(b[:i]+(player,)+b[i+1:],-player)) for i,x in enumerate(b) if not x}
def labels(s):
    a=solve_bit(s['board'],s['player']);b=solve_grid(s['board'],s['player'])
    assert a==b,(s['id'],a,b)
    return {str(k):(v+1)/2 for k,v in a.items()}
