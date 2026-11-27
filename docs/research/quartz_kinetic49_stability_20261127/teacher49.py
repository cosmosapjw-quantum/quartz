"""Nested fixed-budget child PUCT; no common root gate, no games."""
import time
H=(32,64,128)
def snapshots(ev,h,move,horizons=H):
    from k46_engine import Node,from_history
    assert tuple(sorted(set(horizons)))==tuple(horizons) and horizons[0]>0
    a,b=from_history(h+[move]);p=1-2*((len(h)+1)%2)
    start=time.perf_counter_ns();c=time.process_time_ns();nn=ev.rows
    root=Node(a,b,p);v,term=root.expand(ev);out={};terminal=0;n=0
    for cap in horizons:
        if term:
            out[str(cap)]=dict(raw=-v,nn=0,visits=0,terminal=True,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-start)
            continue
        while n<cap:
            node=root;path=[]
            while node.legal is not None:
                i=node.select();path.append((node,i));node=node.descend(i)
            v,ended=node.expand(ev);terminal+=ended;n+=1
            for parent,i in reversed(path):
                v=-v;parent.n[i]+=1;parent.w[i]+=v
        i=int(root.n.argmax());assert root.n[i]>0 and int(root.n.sum())==cap and ev.rows-nn==1+cap-terminal
        out[str(cap)]=dict(raw=-float(root.w[i]/root.n[i]),nn=ev.rows-nn,visits=cap,terminal=False,terminal_leaves=int(terminal),leader=int(root.legal[i]),leader_n=int(root.n[i]),leader_w=float(root.w[i]),cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-start)
    return out
def pair(ev,h,r):
    l,u=r['low'],r['high'];a,b=l['move'],u['move'];bd={m:(x,y) for m,x,y in r['partial']['bounds']}
    labels=[];vals={str(k):[] for k in H}
    if a==b:
        return dict(gains={str(k):0. for k in H},values=None,teacher=[None,None],cost=u['nn']-l['nn'],visits=u['visits']-l['visits'],changed=False)
    for m in (a,b):
        lo,hi=bd[m]
        t=None if lo==hi else snapshots(ev,h,m)
        labels.append(t)
        for k in H:vals[str(k)].append(float(lo) if t is None else max(lo,min(hi,t[str(k)]['raw'])))
    g={k:(y[1]-y[0])/2 for k,y in vals.items()}
    assert all((bd[b][0]-bd[a][1])/2<=x<=(bd[b][1]-bd[a][0])/2 for x in g.values())
    return dict(gains=g,values=vals,teacher=labels,cost=u['nn']-l['nn'],visits=u['visits']-l['visits'],changed=True)
def sem(z):return {k:v for k,v in z.items() if k not in ('cpu_ns','wall_ns')}
def physical(z):return sum(t['128']['nn'] for t in z['teacher'] if t is not None)
def calls(z):return sum(t is not None for t in z['teacher'])
