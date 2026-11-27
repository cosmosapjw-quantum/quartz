"""Old/new binary semantic parity on fixed old outer roots, no games."""
import json,gzip,collections as C,pathlib
from k52_plan import D,T,sha,load
from native52 import rapfi_module,rsearch,BIN
def sem(z):
    if isinstance(z,dict):return {k:sem(v) for k,v in z.items() if k not in ('cpu_ns','wall_ns','proof_us','bridge_wait_ns','eval_ipc_ns','transport_ns','step_us','whole_us')}
    if isinstance(z,list):return [sem(v) for v in z]
    return z
def main():
    d=D.parent/'quartz_kinetic51_targeted_20261127';p=d/'R_F96_LOCK.json';lk=json.loads(p.read_text());xs=[]
    for fi in lk['files']:
        q=d/fi['file'];assert sha(q)==fi['sha256']
        with gzip.open(q,'rt') as f:
            for g in map(json.loads,f):
                if g['job']['stratum']!='outer':continue
                for r in g['rows']:
                    if r['arm']=='Q' and r['kind'] not in ('CERTIFICATE','PARTIAL47'):
                        h=g['history'][:r['ply']];xs.append((g['job'],h));break
                if len(xs)>=8:break
        if len(xs)>=8:break
    assert len(xs)==8
    e=rapfi_module();ob=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic51_targeted_backups/mcts_demo'
    assert sha(ob)==load('BINARY_LOCK.json')['old_binary_sha256']
    out=[];ct=C.Counter()
    with e.Server(ob) as old,e.Server(BIN) as new:
        for j,h in xs:
            for arm in (False,True):
                a=rsearch(e,old,h,e.seed(j,len(h)),arm,96);b=rsearch(e,new,h,e.seed(j,len(h)),arm,96)
                assert sem(a)==sem(b),(h,arm)
                out.append(dict(history=h,seed=e.seed(j,len(h)),quartz=arm,old=a,new=b));ct['requests']+=1;ct['old_nn']+=a['nn'];ct['new_nn']+=b['nn']
    (T/'RAPFI_PARITY.json').write_text(json.dumps(dict(counts=dict(ct),old_binary=sha(ob),new_binary=sha(BIN),source_lock=sha(p),rows=out),indent=2,allow_nan=False)+'\n')
    print('RAPFI_PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
