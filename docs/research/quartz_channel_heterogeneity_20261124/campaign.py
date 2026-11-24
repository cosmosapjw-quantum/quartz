"""Frozen C07 protocol: train validation, then independent heldout native campaign."""
import sys, json, gzip, math, hashlib, random, statistics as st, subprocess, copy, time
from pathlib import Path
from collections import defaultdict
from population import D, replay, apply, outcome
from model import rows, CUES, ADDS, add
from planner import plan
BASE=Path('/home/cosmosapjw/quantum/research')
NATIVE=BASE/'quartz_channel_heterogeneity_20261124_native'
SURVEY=BASE/'quartz_general_meta26_20261123/target/release/quartz_c26_game_metareasoning'
SEEDS=(0,1,2,3); REPS=5; LAMBDA=1/1024; MASK=2**64-1
GAMMA=0x9e3779b97f4a7c15
CI={h:i for i,h in enumerate(CUES)}; HI={h:i for i,h in enumerate(ADDS)}
POL=['INITIAL_STOP','BELIEF_STOP','G','F','A']
ZERO=(0,0,0)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (D/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def gz(name,raw): (D/name).write_bytes(gzip.compress(raw,mtime=0))
def encode(rs): return ''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs).encode()
def invoke(cmd,reqs):
    cp=subprocess.run([str(p) for p in cmd],input=encode(reqs),capture_output=True,check=True,timeout=180)
    rs=[json.loads(x) for x in cp.stdout.splitlines()]
    assert len(rs)==len(reqs)
    assert all('error' not in r and r['id']==q['id'] for q,r in zip(reqs,rs))
    return rs,cp.stdout
class RNG:
    def __init__(self,seed):self.s=seed
    def index(self,n):
        while True:
            self.s=(self.s+GAMMA)&MASK
            z=self.s
            z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK
            z=((z^(z>>27))*0x94d049bb133111eb)&MASK
            z=z^(z>>31)
            if z>=((-n)&MASK)%n:return z%n
def rollout(b,p,root,rng):
    n=1
    while outcome(b) is None:
        ms=[i for i,x in enumerate(b) if x==0]
        b=apply(b,ms[rng.index(len(ms))],p);p=-p;n+=1
    return outcome(b)*root+1,n
def cues(s,seed):
    b,p=replay(s['history']);cs=[]
    for c in rows(s):
        c=dict(c);m=c['move'];child=apply(b,m,p)
        rng=RNG(seed^(((m+1)*GAMMA)&MASK));h=ZERO;n=0
        if c['terminal']:
            h=tuple(4*(i==c['u']) for i in range(3));n=1
        else:
            for _ in range(4):
                y,k=rollout(child,-p,p,rng);h=add(h,y);n+=k
        c.update(child=child,rng=rng,cue=h,cue_nodes=n,player=p)
        cs.append(c)
    return cs
def survey_check(sources,prefix):
    jobs=[(s,seed) for s in sources for seed in SEEDS]
    req=[dict(id=s['id']+':'+str(seed),history=s['history'],cue_seed=seed,policy_seed=0,mode='survey',additional=9,repeats=1,warmup=0,trace=False,model=None) for s,seed in jobs]
    ans,raw=invoke([SURVEY],req);checks=0
    for (s,seed),r in zip(jobs,ans):
        cs=cues(s,seed)
        assert len(cs)==len(r['candidates'])
        assert sum(c['cue_nodes'] for c in cs)==r['cue_nodes']
        for c,a in zip(cs,r['candidates']):
            expected=dict(move=c['move'],terminal_known=c['terminal'],remaining=0 if c['terminal'] else c['rem'],counts=list(c['cue']),u=c['u']/2,solve_nodes=c['solve_nodes'],cue_nodes=c['cue_nodes'])
            assert all(a[k]==v for k,v in expected.items()),(s['id'],expected,a)
            checks+=1
    gz(prefix+'_SURVEY.jsonl.gz',raw)
    return dict(roots=len(sources),requests=len(req),candidate_checks=checks,survey_binary_sha256=sha(SURVEY))
def model():
    m=json.loads((D/'model.json').read_text())
    assert sha(D/'model.json')==json.loads((D/'TRAIN_FIT.json').read_text())['model_sha256']
    return m
def validate_table(m):
    count=0;diff=0
    for dep in m['table']:
        for cue in range(15):
            assert dep[0][cue][0][0]==dep[1][cue][0][0]==dep[2][cue][0][0]
            assert dep[1][cue][0]==dep[2][cue][0]
            for k in range(1,4):assert dep[1][cue][k][0]==dep[2][cue][k][0]
            for arm in dep:
                r=arm[cue]
                for h in ADDS[:4]:
                    x=r[HI[h]]
                    assert all(p>0 for p in x[1:]) and abs(sum(x[1:])-1)<1e-12
                    assert abs(x[0]-sum(x[1+y]*r[HI[add(h,y)]][0] for y in range(3)))<1e-12
                    count+=1
            diff+=sum(dep[1][cue][k]!=dep[2][cue][k] for k in range(1,10))
    return dict(normalization_martingale_rows=count,F_A_different_later_rows=diff)
def rowfn(m,arm):
    def row(c):
        k,h,rem,cu=c
        return [k,0.,0.,0.] if k is not None else m['table'][rem][arm][CI[cu]][HI[h]]
    return row
def best(ss,row,known_only=False):
    val=-1;idx=None
    for i,s in enumerate(ss):
        if known_only and s[0] is None:continue
        v=row(s)[0]
        if v>val+1e-12 or (abs(v-val)<=1e-12 and s[0] is not None and ss[idx][0] is None):
            val,idx=v,i
    return val,idx
def ref_run(s,seed,m,policy,horizon,cs0):
    cs=copy.deepcopy(cs0)
    ss=[(c['u']/2 if c['terminal'] else None,ZERO,c['rem'],c['cue']) for c in cs]
    exact_nodes=0;solved=None
    if not any(x[0]==1 for x in ss):
        unknown=[i for i,x in enumerate(ss) if x[0] is None]
        if unknown:
            i=max(unknown,key=lambda j:cs[j]['cue'][1]+2*cs[j]['cue'][2])
            ss[i]=(cs[i]['u']/2,*ss[i][1:]);exact_nodes=cs[i]['solve_nodes'];solved=cs[i]['move']
    initial,ii=best(ss,rowfn(m,0),True)
    arm=max(0,policy-2);row=rowfn(m,arm)
    pv=best(ss,row,policy==0)[0];qm=[None]*2;yy=[None]*2;qn=[0]*2
    if policy>=2:
        costs=[LAMBDA*m['costs'][c['rem']] for c in cs]
        for step in range(horizon):
            v,i=plan(ss,row,costs,horizon-step)
            if step==0:pv=v
            if i is None:break
            c=cs[i];y,n=rollout(c['child'],-c['player'],c['player'],c['rng'])
            k,h,rem,cu=ss[i];ss[i]=(k,add(h,y),rem,cu)
            qm[step],yy[step],qn[step]=c['move'],y,n
    terminal,fi=best(ss,row,policy==0)
    return dict(initial_move=cs[ii]['move'],initial_u=initial,solved=solved,final_move=cs[fi]['move'],terminal_mean=terminal,plan_value=pv,query_moves=qm,observations=yy,query_nodes=qn,cue_nodes=sum(c['cue_nodes'] for c in cs),exact_nodes=exact_nodes)
def check_run(r,ref):
    for k,v in ref.items():
        assert (abs(r[k]-v)<1e-10 if k in ['terminal_mean','plan_value'] else r[k]==v),(k,r,ref)
def repeat_nll(m,cs):
    ll=[[] for _ in range(3)]
    for c in cs:
        if c['terminal']:continue
        r=[float(x) for x in c['rate']]
        for arm in range(3):
            table=m['table'][c['rem']][arm][CI[c['cue']]]
            loss=-sum(r[y]*r[z]*math.log2(table[HI[add(ZERO,y)]][1+z]) for y in range(3) for z in range(3) if r[y]*r[z]>0)
            ll[arm].append(loss)
    return [st.mean(x) for x in ll] if ll[0] else None
def stats(values):
    x=list(values); rng=random.Random(20261124)
    if not x:return None
    boot=sorted(st.mean(rng.choices(x,k=len(x))) for _ in range(2000))
    return dict(n=len(x),mean=st.mean(x),median=st.median(x),min=min(x),max=max(x),bootstrap95_descriptive=[boot[50],boot[1949]],positive=sum(v>1e-12 for v in x),negative=sum(v< -1e-12 for v in x),zero=sum(abs(v)<=1e-12 for v in x))
def aggregate(rr):
    result={}
    for p in POL:
        result[p]={k:st.mean(r['policies'][p][k] for r in rr) for k in rr[0]['policies'][p]}
    pred=[r['repeat_nll'] for r in rr if r['repeat_nll'] is not None]
    return dict(roots=len(rr),policy=result,primary_net_A_minus_F=stats(r['policies']['A']['net']-r['policies']['F']['net'] for r in rr),primary_predictive_A_minus_F_bits=stats(p[2]-p[1] for p in pred),predictive_mean_bits={a:st.mean(p[i] for p in pred) if pred else None for i,a in enumerate(['G','F','A'])})
def main():
    mode=sys.argv[1];src=[json.loads(l) for l in (D/'sources.jsonl').read_text().splitlines()]
    m=model()
    if mode=='traincheck':
        v=validate_table(m);s=survey_check([s for s in src if s['split']=='TRAIN'],'C07_TRAIN')
        subprocess.run([str(NATIVE),str(D/'model.json')],input=b'',check=True)
        dump('C07_TRAIN_CHECK.json',dict(table=v,survey=s,seeds=SEEDS,repeats=REPS,native_sha256=sha(NATIVE),model_sha256=sha(D/'model.json'),source_sha256=sha(D/'sources.jsonl'),protocol_sha256=sha(Path(__file__))))
        print((D/'C07_TRAIN_CHECK.json').read_text());return
    assert mode=='heldout'
    frozen=json.loads((D/'C07_TRAIN_CHECK.json').read_text())
    assert all(frozen[k]==v for k,v in [('model_sha256',sha(D/'model.json')),('source_sha256',sha(D/'sources.jsonl')),('protocol_sha256',sha(Path(__file__))),('native_sha256',sha(NATIVE))])
    sources=[s for s in src if s['split']=='HELDOUT']
    jobs=[(s,seed) for s in sources for seed in SEEDS]
    req=[dict(id=s['id'],history=s['history'],cue_seed=seed,repeats=REPS) for s,seed in jobs]
    gz('C08_REQUESTS.jsonl.gz',encode(req))
    ans,raw=invoke([NATIVE,D/'model.json'],req)
    gz('C08_NATIVE.jsonl.gz',raw)
    survey=survey_check(sources,'C08_HELDOUT')
    root_seeds=defaultdict(list);path_changes=defaultdict(int);checks=0
    for (s,seed),resp in zip(jobs,ans):
        assert resp['cue_seed']==seed
        cs=cues(s,seed); by={c['move']:c for c in cs}
        rr=resp['runs'];assert len(rr)==2+5*REPS
        for p,h in [(3,1),(4,1)]+[(p,2) for p in range(5)]:
            ref=ref_run(s,seed,m,p,h,cs)
            these=[r for r in rr if r['policy']==p and r['horizon']==h]
            assert len(these)==(1 if h==1 else REPS)
            for r in these:check_run(r,ref);checks+=1
        h1=[r for r in rr if r['horizon']==1]
        assert all(h1[0][k]==h1[1][k] for k in ['final_move','query_moves','observations','query_nodes','terminal_mean','plan_value'])
        pol={}; trajectories={}
        for p,label in enumerate(POL):
            these=[r for r in rr if r['policy']==p and r['horizon']==2];r=these[0]
            assert sorted(x['repeat'] for x in these)==list(range(REPS))
            u=by[r['final_move']]['u']/2; extra=sum(r['query_nodes']);total=r['cue_nodes']+r['exact_nodes']+extra
            q=sum(x is not None for x in r['query_moves'])
            pol[label]=dict(utility=u,total_nodes=total,extra_nodes=extra,net=u-LAMBDA*total,elapsed_ns=st.median(x['elapsed_ns'] for x in these),queries=q,repeat_same_candidate=int(q==2 and r['query_moves'][0]==r['query_moves'][1]),degrade=int(u<r['initial_u']),improve=int(u>r['initial_u']),different_move=int(r['final_move']!=r['initial_move']))
            trajectories[label]=(r['final_move'],r['query_moves'],r['observations'])
        for a,b in [('A','F'),('F','G'),('A','BELIEF_STOP'),('BELIEF_STOP','INITIAL_STOP')]:
            path_changes[a+'_vs_'+b+'_move']+=trajectories[a][0]!=trajectories[b][0]
            path_changes[a+'_vs_'+b+'_trajectory']+=trajectories[a]!=trajectories[b]
        root_seeds[s['id']].append(dict(policies=pol,repeat_nll=repeat_nll(m,cs)))
    rr=[]
    for s in sources:
        seeds=root_seeds[s['id']];assert len(seeds)==4
        p=[z['repeat_nll'] for z in seeds if z['repeat_nll'] is not None]
        rr.append(dict(id=s['id'],ply=s['ply'],policies={a:{k:st.mean(z['policies'][a][k] for z in seeds) for k in seeds[0]['policies'][a]} for a in POL},repeat_nll=[st.mean(x[i] for x in p) for i in range(3)] if p else None))
    dump('C08_ROOTS.json',rr)
    summary=aggregate(rr);summary.update(dict(by_ply={str(p):aggregate([r for r in rr if r['ply']==p]) for p in sorted({r['ply'] for r in rr})},survey=survey,native_rows_verified=checks,h1_equal_requests=len(jobs),root_seed_count=len(jobs),path_changes=dict(path_changes),seeds=SEEDS,repeats=REPS,model_sha256=sha(D/'model.json'),native_sha256=sha(NATIVE),primary_definition='root mean of seed mean of per-candidate true r_y*r_z predictive loss; A-F lower is better; root mean paired node-priced net A-F higher is better; no iid coverage claim',timing='median five repeated native one() per seed, then equal seed/root averages; no matched time, wall timing excludes process and model parsing'))
    dump('C08_SUMMARY.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='by_ply'},indent=2))
if __name__=='__main__':main()
