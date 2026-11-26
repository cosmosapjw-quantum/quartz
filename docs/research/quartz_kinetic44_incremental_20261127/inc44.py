import ctypes as C,pathlib
from k44_plan import T
class Old(C.Structure):
    _fields_=[('status',C.c_int32),('gate',C.c_int32),('cert',C.c_int32),('exhausted',C.c_int32),('lo',C.c_int32*225),('hi',C.c_int32*225),('queries',C.c_uint64),('scans',C.c_uint64),('checks',C.c_uint64)]
class Receipt(C.Structure):
    _fields_=Old._fields_[:6]+[('bin',C.c_int32)]+Old._fields_[6:]+[('skipped',C.c_uint64)]
old=C.CDLL(str(pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic43_partial_backups/partial43.so'))
old.probe43.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_uint8),C.c_int,C.c_uint64,C.POINTER(Old)];old.probe43.restype=C.c_int
lib=C.CDLL(str(T/'inc44.so'))
lib.probe44.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_uint8),C.c_int,C.c_uint64,C.c_int,C.POINTER(Receipt)];lib.probe44.restype=C.c_int
def probe(h,n,d,budget,mode='F'):
    assert n in (7,15) and mode in ('OLD','I','F') and type(budget) is int and budget>=0
    b=(C.c_uint8*(n*n))();p=1+len(h)%2
    for i,m in enumerate(h):
        assert 0<=m<n*n and b[m]==0;b[m]=1 if 1+i%2==p else 2
    k=4 if n==7 else 5
    if mode=='OLD':
        r=Old();rc=old.probe43(n,k,b,d,budget,C.byref(r))
    else:
        r=Receipt();rc=lib.probe44(n,k,b,d,budget,int(mode=='F'),C.byref(r))
    assert rc==r.status==0,'missing or invalid native incremental receipt'
    z=dict(gate=None if r.gate<0 else r.gate,cert=None if r.cert<0 else r.cert,exhausted=bool(r.exhausted),bounds=[[m,r.lo[m],r.hi[m]] for m in range(n*n) if not b[m]],queries=r.queries,scans=r.scans,checks=r.checks,depth=d,budget=budget)
    if mode!='OLD':z.update(bin=r.bin,skipped=r.skipped,mode=mode)
    return z
def semantic(z):
    return {k:z[k] for k in ('gate','cert','exhausted','bounds','queries','checks','depth','budget')}
