import ctypes as C,gzip,json,collections as Co,sys,time
from concurrent.futures import ProcessPoolExecutor
from k46_plan import D,ROOT,T,load,save,sha
from pair46 import query,gain
K45=D.parent/'quartz_kinetic45_sequential_20261127'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def run_one(j):
    z=query(j)
    assert gain(j['input'],j['optimal'],j['actions'][0]==j['actions'][1])==j['gain2']
    assert j['gain2'][0]<=z['gain2'][0]<=z['gain2'][1]<=j['gain2'][1]
    return dict(job=j,result=z)
def run():
    plan=load('PAIR_PLAN.json')
    for k,v in plan['source'].items():assert sha(ROOT/k)==v
    prior={(r['a'],r['b']):r['truth'] for r in raw(K45/'LATE.jsonl.gz')}
    cnt=Co.defaultdict(Co.Counter);p=D/'PAIR_RESULT.jsonl.gz';assert not p.exists()
    with ProcessPoolExecutor(max_workers=8) as ex,gzip.open(p,'wt',compresslevel=3) as f:
        for rr in ex.map(run_one,plan['jobs'],chunksize=2):
            j,z=rr['job'],rr['result'];ct=cnt[j['lane']];ct['rows']+=1
            mult=len(j['refs']) if j['lane']=='EARLY' else 1
            ct['observations']+=mult;ct['nodes']+=sum(z['nodes']);ct['cpu_ns']+=z['cpu_ns'];ct['hits']+=z['hits'];ct['censored_queries']+=z['ask'].count(-1);ct['completed_queries']+=z['ask'].count(0)+z['ask'].count(1)
            ct['width_before']+=mult*(j['gain2'][1]-j['gain2'][0]);ct['width_after']+=mult*(z['gain2'][1]-z['gain2'][0])
            ct['tightened']+=z['gain2']!=j['gain2'];ct['tightened_observations']+=mult*(z['gain2']!=j['gain2'])
            ct['point_gain']+=z['gain2'][0]==z['gain2'][1];ct['strict_positive']+=z['gain2'][0]>0;ct['strict_negative']+=z['gain2'][1]<0
            ct['nonnegative_lower']+=z['gain2'][0]>=0
            ct['sum_gain2_lo_before']+=mult*j['gain2'][0];ct['sum_gain2_hi_before']+=mult*j['gain2'][1];ct['sum_gain2_lo_after']+=mult*z['gain2'][0];ct['sum_gain2_hi_after']+=mult*z['gain2'][1]
            if j['lane']=='LATE':
                v=prior[(j['a'],j['b'])];rr['truth']=v
                ct['truth_known']+=v is not None
                if v is not None:
                    x,y=[v[str(m)] for m in j['actions']]
                    assert z['bounds'][0][0]<=x<=z['bounds'][0][1] and z['bounds'][1][0]<=y<=z['bounds'][1][1] and z['gain2'][0]<=x-y<=z['gain2'][1]
                    ct['truth_point']+=z['gain2'][0]==z['gain2'][1]
            f.write(json.dumps(rr,separators=(',',':'))+'\n')
    save('PAIR_LOCK.json',dict(file=p.name,sha256=sha(p),plan=sha(D/'PAIR_PLAN.json'),counts={k:dict(v) for k,v in cnt.items()},library=sha(T/'pair46.so'),source=sha(D/'pair46.cpp')))
    print('PAIR_RESULT',json.dumps({k:dict(v) for k,v in cnt.items()}),flush=True)
def independent_one(x):
    j,k,m,t,expected=x
    lib=C.CDLL(str(T/'independent46.so'));lib.check46.argtypes=[C.c_uint64,C.c_uint64,C.c_int,C.c_int,C.c_uint64,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)];lib.check46.restype=C.c_int
    nodes=C.c_uint64();hits=C.c_uint64();c=time.process_time_ns()
    val=lib.check46(j['a'],j['b'],m,t,200000,C.byref(nodes),C.byref(hits))
    assert val in (-1,0,1) and (val==-1 or val==expected)
    return dict(id=j['id'],k=k,m=m,t=t,expected=expected,value=val,nodes=nodes.value,hits=hits.value,cpu_ns=time.process_time_ns()-c)
def independent():
    lk=load('PAIR_LOCK.json');p=D/lk['file'];assert sha(p)==lk['sha256']
    tasks=[]
    for r in raw(p):
        j,z=r['job'],r['result']
        if j['lane']!='EARLY':continue
        for k,v in enumerate(z['ask']):
            if v in (0,1):tasks.append((j,k,j['actions'][k//2],k%2,v))
    cnt=Co.Counter();out=D/'PAIR_INDEPENDENT.jsonl.gz';assert not out.exists()
    with ProcessPoolExecutor(max_workers=8) as ex,gzip.open(out,'wt',compresslevel=3) as f:
        for r in ex.map(independent_one,tasks,chunksize=2):
            cnt['checks']+=1;cnt['confirmed']+=r['value']>=0;cnt['censored']+=r['value']==-1;cnt['nodes']+=r['nodes'];cnt['cpu_ns']+=r['cpu_ns']
            f.write(json.dumps(r,separators=(',',':'))+'\n')
    save('PAIR_INDEPENDENT_LOCK.json',dict(file=out.name,sha256=sha(out),counts=dict(cnt),pair=sha(D/'PAIR_LOCK.json'),library=sha(T/'independent46.so'),scope='independent coordinate terminal-only Boolean proof, cap200000 per query; censored is not disagreement'))
    print('INDEPENDENT',dict(cnt),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
