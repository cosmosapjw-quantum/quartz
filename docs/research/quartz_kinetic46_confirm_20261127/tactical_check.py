import ctypes as C,gzip,json,collections as Co,time,hashlib,pathlib
from concurrent.futures import ProcessPoolExecutor
T=pathlib.Path(__file__).resolve().parent;D=pathlib.Path.home()/'quantum/research/quartz_kinetic46_confirm_20261127/docs/research/quartz_kinetic46_confirm_20261127'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def one(x):
    j,k,expect=x;lib=C.CDLL(str(T/'independent46_tactical.so'))
    lib.check46t.argtypes=[C.c_uint64,C.c_uint64,C.c_int,C.c_int,C.c_uint64,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)];lib.check46t.restype=C.c_int
    n=C.c_uint64();h=C.c_uint64();c=time.process_time_ns()
    v=lib.check46t(j['a'],j['b'],j['actions'][k//2],k%2,2000000,C.byref(n),C.byref(h))
    assert v in (-1,0,1) and (v==-1 or v==expect)
    return dict(id=j['id'],k=k,expected=expect,value=v,nodes=n.value,hits=h.value,cpu_ns=time.process_time_ns()-c)
if __name__=='__main__':
    lk=json.loads((D/'PAIR_LOCK.json').read_text());p=D/lk['file'];assert sha(p)==lk['sha256']
    jobs=[(r['job'],i,v) for r in raw(p) if r['job']['lane']=='EARLY' for i,v in enumerate(r['result']['ask']) if v in (0,1)]
    assert len(jobs)==10
    c=Co.Counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        rs=list(ex.map(one,jobs))
    for r in rs:c['checks']+=1;c['confirmed']+=r['value']>=0;c['censored']+=r['value']==-1;c['nodes']+=r['nodes'];c['cpu_ns']+=r['cpu_ns']
    p=T/'PAIR_TACTICAL.json';assert not p.exists();p.write_text(json.dumps(dict(counts=dict(c),rows=rs,cap=2000000,source_sha256=sha(T/'independent46_tactical.cpp'),binary_sha256=sha(T/'independent46_tactical.so'),primary_lock_sha256=sha(D/'PAIR_LOCK.json'),scope='diagnosed follow-up after terminal-only cap: independent coordinate terminal, separately enumerated completion windows, exact forced-response lemma, reverse move order; no dead-window shortcut'),indent=2)+'\n');print(p.read_text())
