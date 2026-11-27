"""Bounded root rival comparison using separate equal-budget child PUCT; exact proof guards."""
import time,math
def candidate(low,bounds):
    a=low['move'];bd={m:(l,u) for m,l,u in bounds}
    vs={m:n for m,n in low['votes']};assert set(bd)==set(vs) and a in bd
    r=[m for m in bd if m!=a and bd[m][1]!=-1]
    if not r:return None
    b=max(r,key=lambda m:(vs[m],-m))
    assert bd[a][1]!=-1
    return None if bd[a][0]>=bd[b][1] else b
def choose(a,b,x,y):return b if y>x else a
def child(ev,h,m):
    from k46_engine import Node,from_history
    a,b=from_history(h+[m]);p=1-2*((len(h)+1)%2);nn=ev.rows;c=time.process_time_ns();t=time.perf_counter_ns()
    root=Node(a,b,p);init,term=root.expand(ev)
    if term:
        assert ev.rows==nn
        return dict(raw=-init,visits=0,nn=0,terminal=True,terminal_leaves=0,samples=[],cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
    samples=[];ends=0
    for _ in range(5):
        node=root;path=[]
        while node.legal is not None:
            i=node.select();path.append((node,i));node=node.descend(i)
        v,ended=node.expand(ev);ends+=int(ended)
        for parent,i in reversed(path):
            v=-v;parent.n[i]+=1;parent.w[i]+=v
        samples.append(float(v))
    assert int(root.n.sum())==5 and abs(float(root.w.sum())-math.fsum(samples))<1e-12
    assert ev.rows-nn==6-ends
    return dict(raw=-math.fsum(samples)/5,visits=5,nn=ev.rows-nn,terminal=False,terminal_leaves=ends,samples=samples,edges=[[int(m),int(n),float(w)] for m,n,w in zip(root.legal,root.n,root.w)],cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def probe(ev,h,r):
    a=r['move'];bd={m:(l,u) for m,l,u in r['partial']['bounds']};b=candidate(r['low'],r['partial']['bounds'])
    start=time.perf_counter_ns();c=time.process_time_ns();nn=ev.rows
    out=dict(incumbent=a,rival=b,chosen=a,children=[],values=[],extra_nn=0,reason='no_nondominated_rival')
    if b is not None:
        vals=[];children=[]
        for m in (a,b):
            l,u=bd[m];z=None if l==u else child(ev,h,m)
            vals.append(float(l) if z is None else max(l,min(u,z['raw'])))
            children.append(z)
        out.update(chosen=choose(a,b,*vals),children=children,values=vals,extra_nn=ev.rows-nn,reason='compared')
        assert out['extra_nn']==sum(z['nn'] for z in children if z) and out['extra_nn']<=12
    out.update(cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-start)
    return out
