"""K53 frozen outer roots, old/new exact Rapfi semantic receipts for all three commands."""
import json,gzip,collections as C,pathlib
from k54_plan import D,T,sha,load
from native54 import rapfi_module,rsearch,BIN
def sem(z):
    if isinstance(z,dict):return {k:sem(v) for k,v in z.items() if k not in ('cpu_ns','wall_ns','proof_us','bridge_wait_ns','eval_ipc_ns','transport_ns','step_us','whole_us','proof_ns')}
    if isinstance(z,list):return [sem(v) for v in z]
    return z
def main():
    d=D.parent/'quartz_kinetic53_reuse_20261127';p=d/'R_T_LOCK.json';lk=json.loads(p.read_text());xs=[]
    for fi in lk['files']:
        q=d/fi['file'];assert sha(q)==fi['sha256']
        with gzip.open(q,'rt') as f:
            for g in map(json.loads,f):
                if g['event'] is not None:xs.append((g['job'],g['history'][:g['rows'][g['event']]['ply']]))
                if len(xs)>=8:break
        if len(xs)>=8:break
    assert len(xs)==8
    e=rapfi_module();ob=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic53_reuse_backups/mcts_demo'
    assert sha(ob)==load('BINARY_LOCK.json')['old_binary_sha256']
    ct=C.Counter();out=[]
    with e.Server(ob) as old,e.Server(BIN) as new:
        for j,h in xs:
            for q,cap in ((False,96),(True,96),(True,'T')):
                a=rsearch(e,old,h,e.seed(j,len(h)),q,cap);b=rsearch(e,new,h,e.seed(j,len(h)),q,cap);assert sem(a)==sem(b),(h,q,cap)
                ct['requests']+=1;ct['old_nn']+=a['nn'];ct['new_nn']+=b['nn'];ct['target_requests']+=cap=='T'
                if cap=='T':assert 'target' in b;ct['target_extra']+=b['target']['extra_nn']
                out.append(dict(history=h,quartz=q,cap=cap,old=a,new=b))
    (T/'RAPFI_PARITY.json').write_text(json.dumps(dict(counts=dict(ct),old_binary=sha(ob),new_binary=sha(BIN),source_lock=sha(p),rows=out),indent=2,allow_nan=False)+'\n')
    print('RAPFI_PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
