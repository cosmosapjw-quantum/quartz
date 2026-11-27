"""Independent Python reference construction of retained child trees on fixed archived roots."""
import collections as C,json,gzip,math
from k53_plan import D,T,SEEDS,load,sha
from native53 import NNServer,BIN
from k46_engine import load_eval,Session
def receipt(node):
    return [[int(m),int(n),float(w)] for m,n,w in zip(node.legal,node.n,node.w)]
def child(ev,root,m):
    idx=list(map(int,root.legal)).index(m);n=ev.rows;old=int(root.n[idx]);node=root.descend(idx);init=node.legal is None
    if init:
        value,term=node.expand(ev)
        if term:return dict(raw=-value,nn=0,visits=0,terminal=True,terminal_leaves=0,samples=[],initial_eval=False,old_root_n=old,pre_edges=[],edges=[])
    pre=receipt(node);samples=[];ends=0
    for _ in range(5):
        t=node;path=[]
        while t.legal is not None:
            j=t.select();path.append((t,j));t=t.descend(j)
        v,term=t.expand(ev);ends+=int(term)
        for parent,j in reversed(path):
            v=-v;parent.n[j]+=1;parent.w[j]+=v
        samples.append(float(v))
    return dict(raw=-math.fsum(samples)/5,nn=ev.rows-n,visits=5,terminal=False,terminal_leaves=ends,samples=samples,initial_eval=init,old_root_n=old,pre_edges=pre,edges=receipt(node))
def eq(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:eq(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):eq(x,y)
    elif isinstance(a,float):assert abs(a-b)<=2e-15,(a,b)
    else:assert a==b,(a,b)
def main():
    lk=load('CACHE_LOCK.json');assert sha(D/lk['file'])==lk['sha256'];first={}
    with gzip.open(D/lk['file'],'rt') as f:
        for x in map(json.loads,f):first.setdefault(x['seed'],x['history'])
    ct=C.Counter();rows=[]
    for s in SEEDS:
        ev=load_eval(s);h=first[s]
        with NNServer(ev) as srv:r=srv.target(h,reuse=True)
        nr=ev.rows;ss=Session(ev,h,pr={'move':None});lo=ss.advance(20,True);a=r['low']['move']
        assert all(lo[k]==r['low'][k] for k in ('nn','visits','votes','distance','prior_best'))
        assert lo['move']==r['low']['original_move']
        bd={m:(l,u) for m,l,u in r['partial']['bounds']};vs=dict(lo['votes']);alts=[m for m in vs if m!=a and bd[m][1]!=-1];b=max(alts,key=lambda m:(vs[m],-m)) if alts else None
        if b is not None and bd[a][0]>=bd[b][1]:b=None
        t=r['target'];assert b==t['rival'] and a==t['incumbent'];vals=[];chs=[];chosen=a
        if b is not None:
            for m,z in zip((a,b),t['children']):
                l,u=bd[m];x=None if l==u else child(ev,ss.root,m)
                if x is None:assert z is None;vals.append(l)
                else:eq(x,{k:v for k,v in z.items() if k!='wall_ns'});vals.append(max(l,min(u,x['raw'])));ct['children']+=1
                chs.append(x)
            if vals[1]>vals[0]:chosen=b
        eq(vals,t['values']);assert chosen==t['chosen']==r['move'];assert ev.rows-nr==r['nn']==nr
        rows.append(dict(seed=s,history=h,native=r,reference=dict(low=lo,children=chs,values=vals,chosen=chosen,nn=ev.rows-nr)))
        ct['roots']+=1;ct['native_nn']+=nr;ct['reference_nn']+=ev.rows-nr
    (T/'REUSE_REFERENCE.json').write_text(json.dumps(dict(counts=dict(ct),rows=rows,scope='separate Python tree reconstruction, same evaluator, fixed first historical root per seed, not full independent rerun'),indent=2,allow_nan=False)+'\n')
    print('REUSE_REFERENCE',dict(ct),flush=True)
if __name__=='__main__':main()
