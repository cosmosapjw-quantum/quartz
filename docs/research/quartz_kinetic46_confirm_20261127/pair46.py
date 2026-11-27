import ctypes as C,time
from k46_plan import T
class Receipt(C.Structure):
    _fields_=[('status',C.c_int32),('lo',C.c_int32*2),('hi',C.c_int32*2),('ask',C.c_int32*4),('nodes',C.c_uint64*4),('hits',C.c_uint64),('states',C.c_uint64)]
lib=C.CDLL(str(T/'pair46.so'));lib.pair46.argtypes=[C.c_uint64,C.c_uint64,C.POINTER(C.c_int32),C.POINTER(C.c_int32),C.POINTER(C.c_int32),C.c_uint64,C.POINTER(Receipt)];lib.pair46.restype=C.c_int
def gain(bd,opt,same=False):
    v=[x-y for x in range(bd[0][0],bd[0][1]+1) for y in range(bd[1][0],bd[1][1]+1) if (not opt[0] or x>=y) and (not opt[1] or x<=y) and (not same or x==y)]
    assert v,'inconsistent sound bounds'
    return [min(v),max(v)]
def query(j,cap=100000):
    A=C.c_int32*2;m=A(*j['actions']);lo=A(*(x[0] for x in j['input']));hi=A(*(x[1] for x in j['input']));r=Receipt()
    t=time.perf_counter_ns();c=time.process_time_ns();rc=lib.pair46(j['a'],j['b'],m,lo,hi,cap,C.byref(r))
    assert rc==r.status==0,'native pair receipt missing or invalid'
    bd=[[r.lo[i],r.hi[i]] for i in range(2)]
    return dict(bounds=bd,gain2=gain(bd,j['optimal'],m[0]==m[1]),ask=list(r.ask),nodes=list(r.nodes),hits=r.hits,states=r.states,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t,cap=cap)
