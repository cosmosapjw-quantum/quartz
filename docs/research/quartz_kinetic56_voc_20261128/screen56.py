"""Predecision screen using frozen K55 Q96; no full games or target computation."""
import json,collections as C,pathlib,gzip
from k56_plan import D,T,load,sha,save
from native56 import rapfi_module,rsearch
def main():
    p=load('PLAN.json');e=rapfi_module();old=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic55_policy_backups/mcts_demo'
    assert sha(old)==json.loads((D.parent/'quartz_kinetic55_policy_20261128/NATIVE_IDENTITY.json').read_text())['binary_sha256']
    jobs=[];ct=C.Counter();f=T/'SCREEN.jsonl.gz';assert not f.exists()
    with e.Server(old) as server,gzip.open(f,'wt',compresslevel=3) as out:
        for j in p['rapfi_candidates']:
            h=j['opening'];r=rsearch(e,server,h,e.seed(j,len(h)),True,96)
            assert 'target' not in r
            vs=r['votes'];ok=False;reason=r['kind']
            if r['kind'] not in ('CERTIFICATE','PARTIAL47'):
                i=max(range(len(vs)),key=lambda k:(vs[k][1],k));dist=min((vs[i][1]-v+int(k<i) for k,(m,v) in enumerate(vs) if k!=i),default=10**9)
                ok=dist<=8 and vs[0][0]!=vs[i][0];reason='eligible' if ok else 'unresolved_ineligible'
            ct['screened']+=1;ct['nn']+=r['nn'];ct[reason]+=1
            out.write(json.dumps(dict(job=j,root=r,selected=ok,reason=reason),separators=(',',':'),allow_nan=False)+'\n')
            if ok:jobs.append(j)
            if len(jobs)==64:break
    assert jobs
    # retain candidate IDs for seed identity; analysis uses sorted families, not contiguous ids.
    (T/'RAPFI_PLAN.json').write_text(json.dumps(dict(jobs=jobs,counts=dict(ct),screen_sha256=sha(f),binary=sha(old),candidate_plan=sha(D/'PLAN.json'),scope='broken-three synthetic predecision qualified roots; no target outcomes used'),indent=2,sort_keys=True)+'\n')
    print('SCREEN',dict(ct),flush=True)
if __name__=='__main__':main()
