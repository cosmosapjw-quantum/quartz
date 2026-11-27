"""Bounded followup for censored natural exact roots using independently verified K51 solver."""
import json,pathlib,collections as C,subprocess
from k52_plan import D,T,sha,load
def main():
    x=load('NATURAL_EXACT_AUDIT.json');missing=[r for r in x['rows'] if not r['complete']]
    solver=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic51_targeted_backups/independent51'
    lk=json.loads((D.parent/'quartz_kinetic51_targeted_20261127/EXACT_EXTENDED_LOCK.json').read_text())
    assert sha(solver)==lk['binary_sha256']
    ps=C.defaultdict(list)
    for r in load('AUDIT_NN.json')['natural_exact']:ps[r['a'],r['b']].extend(r['proofs'])
    # Hard bound to at most 8 additional exact roots, preserve any unrun remainder.
    jobs=missing[:8];results=[]
    if jobs:
        inp=''.join(f"{i} {r['a']} {r['b']}\n" for i,r in enumerate(jobs))
        out=subprocess.run([str(solver)],input=inp,text=True,capture_output=True,check=True,timeout=120)
        rows=list(map(json.loads,out.stdout.splitlines()));assert len(rows)==len(jobs)
        for r in rows:
            j=jobs[r['id']];a,b=j['a'],j['b']
            if r['complete']:
                for z in ps[a,b]:
                    vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
                    if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
            results.append(dict(a=a,b=b,**r))
    q=dict(initial=x['roots'],initial_complete=x['complete'],extended_attempted=len(jobs),extended_complete=sum(r['complete'] for r in results),not_extended=max(0,len(missing)-len(jobs)),solver_sha256=sha(solver),rows=results,scope='same independent coordinate terminal-only solver, 12M cap; 8 root ceiling; initial censor preserved')
    (T/'LATE_EXTENDED.json').write_text(json.dumps(q,indent=2,sort_keys=True)+'\n')
    print('LATE',q['initial'],q['initial_complete'],q['extended_attempted'],q['extended_complete'],q['not_extended'],flush=True)
if __name__=='__main__':main()
