"""Fixed first compared episode per seed; independent recomputation using same reference Node."""
import sys,math,collections as C,json
from k51_plan import D,T,SEEDS,load,sha
from experiment51 import rows
from native51 import REF
from k46_engine import load_eval,Node,from_history
def evaluate(ev,h,m):
    a,b=from_history(h+[m]);p=1-2*((len(h)+1)%2);n0=ev.rows
    root=Node(a,b,p);v,term=root.expand(ev)
    if term:return dict(raw=-v,visits=0,nn=0,terminal=True,terminal_leaves=0)
    terminal=0
    for _ in range(5):
        node=root;path=[]
        while node.legal is not None:
            i=node.select();path.append((node,i));node=node.descend(i)
        v,term=node.expand(ev);terminal+=int(term)
        for parent,i in reversed(path):
            v=-v;parent.n[i]+=1;parent.w[i]+=v
    return dict(raw=-float(root.w.sum())/5,visits=5,nn=ev.rows-n0,terminal=False,terminal_leaves=terminal,edges=[[int(m),int(n),float(w)] for m,n,w in zip(root.legal,root.n,root.w)])
def main():
    first={}
    for x in rows('NN'):
        g=x['target'];e=g['event']
        if e is None:continue
        r=g['rows'][e];z=r['target']
        if z['reason']=='compared' and any(t is not None for t in z['children']) and g['seed'] not in first:first[g['seed']]=(r,z)
    out=[];ct=C.Counter()
    for s in SEEDS:
        if s not in first:continue
        r,z=first[s];ev=load_eval(s)
        for m,t in zip((z['incumbent'],z['rival']),z['children']):
            if t is None:continue
            a=evaluate(ev,r['history'],m)
            assert all(a[k]==t[k] for k in ('visits','nn','terminal','terminal_leaves'))
            assert abs(a['raw']-t['raw'])<1e-12
            if not a['terminal']:assert a['edges']==t['edges']
            out.append(dict(seed=s,history=r['history'],move=m,actual=a,reference=t));ct['children']+=1;ct['nn']+=a['nn']
        assert ev.rows==sum(x['actual']['nn'] for x in out if x['seed']==s)
    (T/'COLD.json').write_text(json.dumps(dict(counts=dict(ct),rows=out,scope='first compared event per seed, separate cold re-execution same reference Node, independent aggregation; not independent evaluator architecture'),indent=2,allow_nan=False)+'\n');print('COLD',dict(ct),flush=True)
if __name__=='__main__':main()
