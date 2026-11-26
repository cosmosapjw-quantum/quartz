import ctypes as C,pathlib
from k42_plan import ROOT,rule,BUDGET
T=ROOT.parent/'quartz_kinetic42_threshold_target_20261127'
class Receipt(C.Structure):
    _fields_=[('status',C.c_int32),('move',C.c_int32),('value',C.c_int32),('lo',C.c_int32*49),('hi',C.c_int32*49),('nodes',C.c_uint64),('hits',C.c_uint64),('states',C.c_uint64)]
lib=C.CDLL(str(T/'threshold42.so'))
lib.solve42.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_int,C.POINTER(Receipt)]
lib.solve42.restype=C.c_int
def solve(a,b,budget=BUDGET,census=False):
    assert type(budget) is int and budget>=0
    r=Receipt();rc=lib.solve42(a,b,budget,int(census),C.byref(r))
    assert rc==r.status and rc in (0,1),f'invalid native receipt {rc}'
    out=dict(complete=rc==0,nodes=r.nodes,hits=r.hits,states=r.states)
    if rc==0:
        out.update(best=r.move,best_value=r.value,bounds={str(m):[r.lo[m],r.hi[m]] for m in rule.cells(rule.FULL^(a|b))})
        if census:out['values']={m:x[0] for m,x in out['bounds'].items() if x[0]==x[1]}
    return out
