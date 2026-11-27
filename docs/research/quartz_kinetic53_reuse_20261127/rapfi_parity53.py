"""Archived outer roots, same old/new base receipt, optional target arithmetic; no games."""
import json,gzip,collections as C,pathlib,math
from k53_plan import D,T,sha,load
from native53 import rapfi_module,rsearch,BIN
def sem(z):
    if isinstance(z,dict):return {k:sem(v) for k,v in z.items() if k not in ('cpu_ns','wall_ns','proof_us','bridge_wait_ns','eval_ipc_ns','transport_ns','step_us','whole_us')}
    if isinstance(z,list):return [sem(v) for v in z]
    return z
def main():
    d=D.parent/'quartz_kinetic52_confirm_20261127';p=d/'R_TEST_96_LOCK.json';lk=json.loads(p.read_text());xs=[]
    for fi in lk['files']:
        q=d/fi['file'];assert sha(q)==fi['sha256']
        with gzip.open(q,'rt') as f:
            for g in map(json.loads,f):
                for r in g['rows']:
                    if r['arm']!='Q' or r['kind'] in ('CERTIFICATE','PARTIAL47'):continue
                    vs=r['votes'];i=max(range(len(vs)),key=lambda j:(vs[j][1],j));dist=min((vs[i][1]-n+int(j<i) for j,(m,n) in enumerate(vs) if j!=i),default=10**9)
                    if dist<=8 and vs[0][0]!=vs[i][0]:xs.append((g['job'],g['history'][:r['ply']]))
                if len(xs)>=8:break
        if len(xs)>=8:break
    xs=xs[:8];assert len(xs)==8
    e=rapfi_module();ob=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic52_confirm_backups/mcts_demo'
    assert sha(ob)==load('BINARY_LOCK.json')['old_binary_sha256']
    out=[];ct=C.Counter()
    with e.Server(ob) as old,e.Server(BIN) as new:
        for j,h in xs:
            for q in (False,True):
                a=rsearch(e,old,h,e.seed(j,len(h)),q,96);b=rsearch(e,new,h,e.seed(j,len(h)),q,96);assert sem(a)==sem(b)
                ct['base_requests']+=1;ct['old_nn']+=a['nn'];ct['new_base_nn']+=b['nn']
            t=rsearch(e,new,h,e.seed(j,len(h)),True,'T');assert 'target' in t
            base={k:v for k,v in t.items() if k not in ('target','root_nn')};base['move']=t['target']['incumbent'];base['nn']=t['root_nn']
            assert sem(base)==sem(b)
            z=t['target'];assert z['extra_nn']<=12 and sum(c['nn'] for c in z['children'] if c)==z['extra_nn']
            for c in z['children']:
                if c and not c['terminal']:
                    assert c['visits']==5 and abs(c['raw']+math.fsum(e0['w'] for e0 in c['edges'])/5)<1e-12
            out.append(dict(history=h,seed=e.seed(j,len(h)),base=b,target=t));ct['target_requests']+=1;ct['new_target_nn']+=t['nn'];ct['extra_nn']+=z['extra_nn']
    (T/'RAPFI_PARITY.json').write_text(json.dumps(dict(counts=dict(ct),old_binary=sha(ob),new_binary=sha(BIN),source_lock=sha(p),rows=out),indent=2,allow_nan=False)+'\n')
    print('RAPFI_PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
