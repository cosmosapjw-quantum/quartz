"""Matched recommendation benchmark, alternating order; not strength or clean production cost."""
from experiment import *
def ticks(pid):
    a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return int(a[11])+int(a[12])
def main():
    co=read('C06_COHORT.json')['roots']
    lock=read('C07_RAW_LOCK.json');assert sha(D/'C07_RAW.jsonl.gz')==lock['raw_sha256']
    ref={(v['root']['id'],v['prefix'],v['response']['arm']):v['response'] for v in rows('C07_RAW.jsonl.gz')}
    n=D/'C08_COST_RAW.jsonl.gz';assert not n.exists();start=host()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        ch=Path(f'/proc/{s.p.pid}/task/{s.p.pid}/children').read_text().split()
        assert len(ch)==1,ch
        child=int(ch[0]);hz=os.sysconf('SC_CLK_TCK')
        for rep in range(5):
            for i,r in enumerate(co):
                for p in (8,16):
                    order=('DS0','DSSTOP') if (rep+i+p//8)%2==0 else ('DSSTOP','DS0')
                    for arm in order:
                        t0=ticks(s.p.pid);c0=ticks(child);py0=time.process_time_ns()
                        a=s.search(r['history'],arm,32,17,prefix=p)
                        py=time.process_time_ns()-py0;t=ticks(s.p.pid)-t0;c=ticks(child)-c0
                        z=ref[r['id'],p,arm]
                        assert a['result']==z['result'] and a['best']==z['best'] and a['NN_calls']==z['NN_calls']
                        emit(f,dict(id=r['id'],group=r['group'],stratum=r['stratum'],prefix=p,rep=rep,arm=arm,order=list(order),best=a['best'],visits=a['visits'],nn=a['NN_calls'],wall_ns=a['transport_ns'],native_us=a['whole_us'],rust_ticks=t,rapfi_ticks=c,python_cpu_ns=py,tick_hz=hz))
    write('C08_COST_LOCK.json',dict(identity=identity(),requests=len(co)*2*2*5,raw_sha256=sha(n),host=host_end(start),scope='CPU tick aggregate; coarse 10ms per process at 100Hz; Python includes validation; diagnostic snapshots included'))
if __name__=='__main__':
    from pathlib import Path
    main()
