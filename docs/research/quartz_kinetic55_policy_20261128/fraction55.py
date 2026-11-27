"""Exact LP extreme points for a single expected local-budget constraint."""
from fractions import Fraction as F
from itertools import product,combinations
import collections as C,json
from k55_plan import D,load,save,sha
def solve(f):
    cells=[C.Counter(z) for z in f['panels']['53']['cells']];cap=f['cap']
    opts=[range(3) if z['n']>=20 else range(1) for z in cells]
    def val(i,a):
        if not a:return (0,0)
        k='cold' if a==1 else 'reuse';return cells[i][k+'_gain2'],cells[i][k+'_cost']
    rows=[]
    for t in product(*opts):
        gain=sum(val(i,a)[0] for i,a in enumerate(t));cost=sum(val(i,a)[1] for i,a in enumerate(t))
        if cost<=cap:rows.append((F(gain),F(cost),t,t,F(0)))
        for i,aa in enumerate(t):
            g,c=val(i,aa)
            for bb in opts[i]:
                gg,cc=val(i,bb)
                if cc<=c or not cost<cap<cost-c+cc:continue
                p=F(cap-cost,cc-c);u=list(t);u[i]=bb
                rows.append((F(gain)+p*(gg-g),F(cap),t,tuple(u),p))
    # max expected gain, min expected cost, then deterministic sparse tie break
    best=min(rows,key=lambda r:(-r[0],r[1],sum(x!=0 for x in r[2])+sum(x!=0 for x in r[3]),r[2],r[3],r[4]))
    g,c,t,u,p=best;assert g>=f['fit']['gain2'] and c<=cap
    return dict(low=list(t),high=list(u),prob_num=p.numerator,prob_den=p.denominator,expected_gain2=str(g),expected_logical_cost=str(c),vertices=len(rows),cap=cap)
def main():
    f=load('FROZEN.json');s=solve(f);s['integer_lock']=sha(D/'FROZEN.json');s['scope']='expected logical budget under randomized treatment; not hard realized budget or physical optimizer; independent fixed coin per episode';save('MIX_FROZEN.json',s);print('MIX',json.dumps(s),flush=True)
if __name__=='__main__':main()
