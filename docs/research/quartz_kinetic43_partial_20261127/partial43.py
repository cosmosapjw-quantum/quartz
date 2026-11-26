import ctypes as C
from k43_plan import T
class Receipt(C.Structure):
    _fields_=[('status',C.c_int32),('gate',C.c_int32),('cert',C.c_int32),('exhausted',C.c_int32),('lo',C.c_int32*225),('hi',C.c_int32*225),('queries',C.c_uint64),('scans',C.c_uint64),('checks',C.c_uint64)]
lib=C.CDLL(str(T/'partial43.so'));lib.probe43.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_uint8),C.c_int,C.c_uint64,C.POINTER(Receipt)];lib.probe43.restype=C.c_int
def probe(h,n,d,budget):
    assert n in (7,15) and type(d) is int and d>=3 and type(budget) is int and budget>=0
    b=(C.c_uint8*(n*n))();p=1+len(h)%2
    for i,m in enumerate(h):
        assert 0<=m<n*n and b[m]==0;b[m]=1 if 1+i%2==p else 2
    r=Receipt();rc=lib.probe43(n,4 if n==7 else 5,b,d,budget,C.byref(r))
    assert rc==r.status==0,'missing or invalid native proof receipt'
    return dict(gate=None if r.gate<0 else r.gate,cert=None if r.cert<0 else r.cert,exhausted=bool(r.exhausted),bounds=[[m,r.lo[m],r.hi[m]] for m in range(n*n) if not b[m]],queries=r.queries,scans=r.scans,checks=r.checks,depth=d,budget=budget)
