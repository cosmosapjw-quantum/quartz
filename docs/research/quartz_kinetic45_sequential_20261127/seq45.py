import ctypes as C,pathlib,time
from k45_plan import T,TIERS
class Receipt(C.Structure):
    _fields_=[('status',C.c_int32),('gate',C.c_int32),('cert',C.c_int32),('exhausted',C.c_int32),('lo',C.c_int32*225),('hi',C.c_int32*225),('bin',C.c_int32)]+[(k,C.c_uint64) for k in ('queries','scans','checks','skipped','hits','entries','positive','negative')]
lib=C.CDLL(str(T/'seq45.so'))
lib.start45.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_uint8)];lib.start45.restype=C.c_void_p
lib.advance45.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.POINTER(Receipt)];lib.advance45.restype=C.c_int
lib.destroy45.argtypes=[C.c_void_p];lib.destroy45.restype=None
class Session:
    def __init__(self,h,n):
        assert n in (7,15)
        b=(C.c_uint8*(n*n))();p=1+len(h)%2
        for i,m in enumerate(h):
            assert 0<=m<n*n and not b[m];b[m]=1 if 1+i%2==p else 2
        self.legal=[m for m in range(n*n) if not b[m]]
        self.ptr=lib.start45(n,4 if n==7 else 5,b);assert self.ptr,'missing native persistent session'
    def __enter__(self):return self
    def __exit__(self,*a):
        lib.destroy45(self.ptr);self.ptr=None
    def advance(self,d,b):
        r=Receipt();rc=lib.advance45(self.ptr,d,b,C.byref(r));assert rc==r.status==0,'invalid sequential receipt'
        return dict(gate=None if r.gate<0 else r.gate,cert=None if r.cert<0 else r.cert,exhausted=bool(r.exhausted),bin=r.bin,bounds=[[m,r.lo[m],r.hi[m]] for m in self.legal],depth=d,budget=b,mode='SEQ',**{k:getattr(r,k) for k in ('queries','scans','checks','skipped','hits','entries','positive','negative')})
def staged(h,n):
    t=time.perf_counter_ns();c=time.process_time_ns();rr=[]
    with Session(h,n) as s:
        for d,b in TIERS:
            r=s.advance(d,b);rr.append(r)
            if r['cert'] is not None or r['bin']==0:break
    return dict(stages=rr,receipt=rr[-1],cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def gain2(a,b,bounds,optimal=False,base_optimal=False):
    if a==b:return (0,0)
    l,u=bounds[a];p,q=bounds[b];lo,hi=l-q,u-p
    if optimal:lo=max(lo,0)
    if base_optimal:hi=min(hi,0)
    assert lo<=hi
    return lo,hi
