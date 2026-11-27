"""Historical K48 teacher reproduction and nested-horizon sensitivity; no games."""
import collections as C,gzip,json,sys,math,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k49_plan import D,T,SEEDS,WORKERS,load,save,sha
from teacher49 import pair,sem,snapshots,physical,calls
OLD=D.parent/'quartz_kinetic48_voc_20261127'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def jobs():
    rows=[];ct=C.Counter();locks={}
    for mode in ('TRAIN','CAL','TEST'):
        p=OLD/(mode+'_LOCK.json');locks[mode]=sha(p)
        for fi in json.loads(p.read_text())['files']:
            pp=OLD/fi['file'];assert sha(pp)==fi['sha256']
            for g in raw(pp):
                for r in g['rows']:
                    if 'utility' not in r:continue
                    h=g['history'][:r['ply']];assert h==r['history']
                    rows.append(dict(split=mode,seed=g['seed'],job=g['job']['id'],history=h,root=r));ct[mode]+=1
    assert dict(ct)=={'TRAIN':351,'CAL':238,'TEST':165},ct
    return rows,locks
def worker(arg):
    seed,rs=arg
    from k46_engine import load_eval
    ev=load_eval(seed);ct=C.Counter();path=D/f'HIST_{seed}.jsonl.gz';assert not path.exists()
    tested=False
    with gzip.open(path,'wt',compresslevel=3) as f:
        for j in rs:
            r=j.pop('root');h=j['history'];z=pair(ev,h,r);u=r['utility']
            assert z['gains']['32']==u['gain'] and z['changed']==u['changed'] and z['cost']==u['cost']
            if z['changed']:assert z['values']['32']==u['values']
            else:assert z['values'] is None and u['values'] is None
            for i,(t,v) in enumerate(zip(z['teacher'],u['teacher'])):
                if t is None:assert v is None
                else:
                    assert v is not None and sem(t['32'])==sem(v),(seed,h,t['32'],v)
                    ct['H32_receipts']+=1
                    if not tested:
                        m=(r['low']['move'],r['high']['move'])[i]
                        for cap in (64,128):
                            a=snapshots(ev,h,m,(cap,))[str(cap)]
                            assert sem(a)==sem(t[str(cap)]),(seed,h,cap,a,t[str(cap)])
                            ct['cold_extra_nn']+=a['nn'];ct['cold_checks']+=1
                        tested=True
            ct['rows']+=1;ct['changed']+=z['changed'];ct['teacher_calls']+=calls(z);ct['teacher_nn']+=physical(z)
            ct['hypothetical_cold_nn']+=sum(t[str(cap)]['nn'] for t in z['teacher'] if t for cap in (32,64,128))
            f.write(json.dumps(dict(**j,bin=r['bin'],disagree=int(r['low']['original_move']!=r['low']['prior_best']),actions=[r['low']['move'],r['high']['move']],bounds=[r['partial']['bounds'],],utility=z),separators=(',',':'),allow_nan=False)+'\n')
    assert ev.rows==ct['teacher_nn']+ct['cold_extra_nn'];print('HIST',seed,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main():
    from native49 import REF
    js,locks=jobs();tasks=[(s,[r for r in js if r['seed']==s]) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=sum((C.Counter(x['counts']) for x in fs),C.Counter());assert ct['rows']==754 and ct['H32_receipts']==360 and ct['changed']==180
    save('HIST_LOCK.json',dict(files=fs,counts=dict(ct),historical_locks=locks,teacher=sha(D/'teacher49.py'),source=sha(D/'historical49.py'),scope='K48 archived source roots; nested H32/64/128 and one cold parity child per eligible seed; no new games'))
    print('HIST_TOTAL',dict(ct),flush=True)
if __name__=='__main__':
    from native49 import REF
    main()
