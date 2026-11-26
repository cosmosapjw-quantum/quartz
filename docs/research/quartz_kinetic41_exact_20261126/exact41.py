"""Finite exact 7x7 four-in-row action values, bounded uncached-state count."""
from k41_plan import rule,BUDGET
class Limit(Exception):pass
def solve(a,b,budget=BUDGET):
    assert type(budget) is int and budget>=0 and not a&b
    assert 0<=a<=rule.FULL and 0<=b<=rule.FULL and not rule.won(a) and not rule.won(b)
    empty=rule.FULL^(a|b);assert 1<=empty.bit_count()<=12
    memo={};nodes=hits=0;reasons={}
    def count(k):reasons[k]=reasons.get(k,0)+1
    def tick():
        nonlocal nodes
        if nodes>=budget:raise Limit
        nodes+=1
    def v(x,y):
        nonlocal hits
        k=x,y
        if k in memo:hits+=1;return memo[k]
        tick();e=rule.FULL^(x|y)
        if rule.won(y):z=-1;count('terminal_win')
        elif not e:z=0;count('full')
        elif rule.immediate(x,y):z=1;count('immediate_win')
        else:
            threat=rule.immediate(y,x)
            if threat.bit_count()>=2:z=-1;count('double_threat')
            elif threat:z=-v(y,x|threat);count('forced_reply')
            elif all(l&y for l in rule.LINES) and all(l&x for l in rule.LINES):z=0;count('dead_lines')
            else:
                z=-1;count('branch')
                for m in rule.cells(e):
                    z=max(z,-v(y,x|1<<m))
                    if z==1:break
        memo[k]=z;return z
    try:
        tick();count('root_census')
        values={str(m):-v(b,a|1<<m) for m in rule.cells(empty)}
        best=max(values.values());move=min(int(m) for m,z in values.items() if z==best)
    except Limit:return dict(complete=False,nodes=nodes,hits=hits,reasons=reasons)
    return dict(complete=True,nodes=nodes,hits=hits,reasons=reasons,values=values,best_value=best,best=move)
