import ctypes as C,pathlib,time
from pair46 import gain
from inc46 import probe
B=pathlib.Path.home()/'Dropbox/personal_projects'
class Ex(C.Structure):
    _fields_=[('status',C.c_int32),('move',C.c_int32),('value',C.c_int32),('lo',C.c_int32*49),('hi',C.c_int32*49),('nodes',C.c_uint64),('hits',C.c_uint64),('states',C.c_uint64)]
ex=C.CDLL(str(B/'quartz_kinetic42_threshold_backups/threshold42.so'))
ex.solve42.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_int,C.POINTER(Ex)];ex.solve42.restype=C.c_int
def exact(h):
    a=sum(1<<m for m in h[::2]);b=sum(1<<m for m in h[1::2])
    if len(h)%2:a,b=b,a
    t=time.perf_counter_ns();c=time.process_time_ns();r=Ex();rc=ex.solve42(a,b,25000,0,C.byref(r))
    assert rc==r.status and rc in (0,1)
    z=dict(complete=rc==0,nodes=r.nodes,hits=r.hits,states=r.states,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
    if rc==0:z.update(best=r.move,value=r.value,bounds=[[m,r.lo[m],r.hi[m]] for m in range(49) if not (a|b)>>m&1])
    return z
def bundle(h,n):
    t=time.perf_counter_ns();c=time.process_time_ns();z=probe(h,n,7,4096);s=dict(receipt=z,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t);e=exact(h) if n==7 and 33<=len(h)<49 and z['cert'] is None else None
    bounds={m:(l,u) for m,l,u in z['bounds']};cert=z['cert']
    if e and e['complete']:
        cert=e['best']
        for m,l,u in e['bounds']:
            x,y=bounds[m];bounds[m]=(max(l,x),min(u,y));assert bounds[m][0]<=bounds[m][1]
    return dict(cold=s,exact=e,cert=cert,bounds=[[m,l,u] for m,(l,u) in bounds.items()],cpu_ns=s['cpu_ns']+(e['cpu_ns'] if e else 0),wall_ns=s['wall_ns']+(e['wall_ns'] if e else 0))
def replace(move,votes,bounds,n):
    bad={m for m,l,u in bounds if u==-1}
    if move not in bad:return move
    pool=[(v,i if n==15 else -m,m) for i,(m,v) in enumerate(votes) if m not in bad]
    return max(pool)[2] if pool else move
def compare(p,B,L,n):
    cert=p['cert'];a=cert if cert is not None else replace(L['move'],L['votes'],p['bounds'],n)
    bd={m:(l,u) for m,l,u in p['bounds']};b=B['move'];base_opt=all(bd[b][0]>=u for m,(l,u) in bd.items() if m!=b)
    lo,hi=gain([bd[a],bd[b]],[cert is not None,base_opt],a==b)
    return dict(move=a,gain2=[lo,hi],nn=0 if cert is not None else L['nn'],cost_cpu_ns=p['cpu_ns']+(0 if cert is not None else L['cpu_ns']),cert=cert is not None,replacement=cert is None and a!=L['move'])
