"""Separate pre-outcome supplemental screening pool; same generator/criterion."""
import pathlib,random,json,gzip,collections as C
from k56_plan import D,T,load,sha
from native56 import rapfi_module,rsearch
def main():
    p=load('PLAN.json');original=json.loads((T/'RAPFI_PLAN.json').read_text());e=rapfi_module()
    old=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic55_policy_backups/mcts_demo';assert sha(old)==original['binary']
    fam,_=e.historical();fam.update(j['family'] for j in e.read('PLAN.json')['jobs'])
    fam.update(j['family'] for j in p['rapfi_candidates'])
    for x in p['prior_locks']:
        q=json.loads((D.parents[2]/x['path']).read_text())
        fam.update(j['family'] for j in q.get('rapfi_jobs',[]))
    rng=random.Random(p['seed']+5601);cs=[]
    while len(cs)<4096:
        dr,dc=rng.choice([(0,1),(1,0),(1,1),(1,-1)]);r,c=rng.randrange(4,11),rng.randrange(4,11);ks=rng.choice([(0,1,3),(0,2,3)])
        black=[(r+(k-1)*dr,c+(k-1)*dc) for k in ks]
        if not all(1<=x<14 and 1<=y<14 for x,y in black):continue
        white=rng.sample([(x,y) for x in range(r-3,r+4) for y in range(c-3,c+4) if (x,y) not in black],3)
        rng.shuffle(black);rng.shuffle(white);h=[15*x+y for pair in zip(black,white) for x,y in pair];f=e.family(h)
        if f in fam:continue
        fam.add(f);cs.append(dict(id=512+len(cs),split='ARENA',stratum='supplement',opening=h,family=f,seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
    cp=T/'SUPPLEMENT_CANDIDATES.json';assert not cp.exists()
    cp.write_text(json.dumps(dict(seed=p['seed']+5601,candidates=cs,original_plan=sha(D/'PLAN.json'),reason='original512 yielded8 eligible; separate pre-outcome supplemental 4096 ceiling, same screening criterion, stop after64 eligible'),indent=2,sort_keys=True)+'\n')
    f=T/'SUPPLEMENT_SCREEN.jsonl.gz';jobs=[];ct=C.Counter()
    with e.Server(old) as server,gzip.open(f,'wt',compresslevel=3) as out:
        for j in cs:
            h=j['opening'];r=rsearch(e,server,h,e.seed(j,len(h)),True,96);assert 'target' not in r
            vs=r['votes'];ok=False;reason=r['kind']
            if r['kind'] not in ('CERTIFICATE','PARTIAL47'):
                i=max(range(len(vs)),key=lambda k:(vs[k][1],k));dist=min((vs[i][1]-v+int(k<i) for k,(m,v) in enumerate(vs) if k!=i),default=10**9)
                ok=dist<=8 and vs[0][0]!=vs[i][0];reason='eligible' if ok else 'unresolved_ineligible'
            ct['screened']+=1;ct['nn']+=r['nn'];ct[reason]+=1
            out.write(json.dumps(dict(job=j,root=r,selected=ok,reason=reason),separators=(',',':'),allow_nan=False)+'\n')
            if ok:jobs.append(j)
            if len(jobs)==64:break
    orig=T/'RAPFI_ORIGINAL_PLAN.json';(T/'RAPFI_PLAN.json').rename(orig)
    o=dict(jobs=original['jobs']+jobs,counts=dict(ct),original=original,original_sha256=sha(orig),supplement_screen_sha256=sha(f),supplement_candidates_sha256=sha(cp),binary=sha(old),candidate_plan=sha(D/'PLAN.json'),scope='original8 plus independent supplemental up to64; separate strata; screen only before actual games')
    (T/'RAPFI_PLAN.json').write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
    print('SUPPLEMENT',dict(ct),'qualified total',len(o['jobs']),flush=True)
if __name__=='__main__':main()
