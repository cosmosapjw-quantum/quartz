"""Independent two-phase, exact contrast, native trace and paired arena audit."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
from itertools import product
import hashlib,json,gzip,random,statistics as S,math,sys,importlib.util
D=Path(__file__).resolve().parent;ROOT=D.parents[2];R=D.parent
p=R/'quartz_kinetic36_20261126/audit_k36.py'
spec=importlib.util.spec_from_file_location('k36_exact_audit',p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
H=(80,96,112,128,256);GROUPS=('TRAIN','CAL','TEST','NATURAL')
REFS=((128,128),(256,256),(128,256),(256,128))
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(n):return json.loads((D/n).read_text())
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def near(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)
def mean(x):return sum(x,F(0))/len(x)
def audit_files(lk):
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];yield f,p
def gain(z,l,h,ref=None):
    a=z['arms']['L'+str(h)]['best'];b=z['arms']['L128']['best']
    if a==b:return F(0)
    x,y=l[str(a)],l[str(b)]
    return mean(x)-mean(y) if ref is None else x[ref]-y[ref]
def audit_sample(fr,sp):
    assert sp['seed']==2026112737 and sp['frame']==sha(D/'FRAME.json')
    for i,g in enumerate(GROUPS):
        rr=[r for r in fr if r['root']['job']['group']==g];q=sp['groups'][g]
        hard=[r['root']['job']['id'] for r in rr if len(r['actions'])>1]
        stable=[r['root']['job']['id'] for r in rr if len(r['actions'])==1]
        n=len(stable) if g=='NATURAL' else min(len(stable),max(2,(len(stable)+3)//4))
        ss=stable if g=='NATURAL' else sorted(random.Random(2026112737+i).sample(stable,n))
        assert q==dict(N=len(rr),stable_N=len(stable),stable_n=n,hard_N=len(hard),stable_ids=ss,hard_ids=hard,stable_pi=[n,len(stable)] if stable else None)
def fit(fr,lab):
    rr=[r for r in fr if r['root']['job']['group']=='TRAIN'];n=len(rr)
    opts=[]
    for tab in product((80,128),repeat=8):
        g=sum(gain(r,lab.get(r['root']['job']['id']),tab[r['bin']]) for r in rr)
        if g>=-F(n,40):opts.append((sum(r['arms']['L'+str(tab[r['bin']])]['NN_calls'] for r in rr),-g,tab))
    cost,ng,tab=min(opts)
    return dict(table=list(tab),cost=cost,gain=float(-ng/n),n=n,feasible=len(opts),bin_n=[sum(r['bin']==b for r in rr) for b in range(8)])
def audit_abs(fr,lab,sp,summary):
    for g in GROUPS:
        rr=[r for r in fr if r['root']['job']['group']==g];N=len(rr);q=sp['groups'][g];z=summary['groups'][g]
        assert z['N']==N and z['sampled']==q['stable_n']+q['hard_N'] and z['hard']==q['hard_N']
        labels=[lab[i] for i in q['stable_ids']+q['hard_ids']]
        assert z['actions']==sum(len(l) for l in labels) and z['reference_disagreement']==sum(len(set(u))>1 for l in labels for u in l.values())
        assert z['nonconstant']==sum(len({mean(u) for u in l.values()})>1 for l in labels)
        for h in H:
            a=z['arms'][str(h)]
            d=[gain(r,lab.get(r['root']['job']['id']),h) for r in rr]
            near(a['gain'],mean(d));assert a['nonzero']==sum(x!=0 for x in d)
            for k in range(4):near(a['ref_gains'][k],mean([gain(r,lab.get(r['root']['job']['id']),h,k) for r in rr]))
            near(a['NN'],S.mean(r['arms']['L'+str(h)]['NN_calls'] for r in rr));near(a['B_NN'],S.mean(r['arms']['B'+str(h)]['NN_calls'] for r in rr))
            xs=[[],[]];hard=[]
            for r in rr:
                i=r['root']['job']['id'];sel=r['arms']['L'+str(h)]['best']
                if i in lab:xs[int(len(r['actions'])>1)].append(mean(lab[i][str(sel)]))
                if len(r['actions'])>1:hard.append(gain(r,lab[i],h))
            wt=[F(q['stable_N'],q['stable_n']) if q['stable_n'] else F(0),F(1)]
            w=[wt[k] for k in range(2) for _ in xs[k]];u=xs[0]+xs[1];assert sum(w)==N
            near(a['absolute'],sum(x*y for x,y in zip(w,u))/N);near(a['naive'],mean(u));near(a['ESS'],F(N*N)/sum(x*x for x in w))
            var=F(0)
            for M,k,ys in [(q['stable_N'],q['stable_n'],xs[0]),(q['hard_N'],q['hard_N'],xs[1])]:
                if 0<k<M:var+=F(M*(M-k),N*N)*sum((y-mean(ys))**2 for y in ys)/(k*(k-1))
            near(a['design_se']**2,var)
            if hard:near(a['hard_gain'],mean(hard))
        assert z['positive_hard']==sum(any(gain(r,lab[r['root']['job']['id']],h)>0 for h in H) for r in rr if len(r['actions'])>1)
def main():
    pl=read('PLAN.json');co=read('COHORT.json');fr=read('FRAME.json');sp=read('SAMPLE.json')
    jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==1280 and len({j['family'] for j in jobs.values()})==1280
    for j in jobs.values():
        b,p,w=v.replay(j['opening'],'freestyle');assert w is None and hashlib.sha256(v.orbit_board(b,p,'freestyle')).hexdigest()==j['family']
    for a in pl['prior_locks']:assert sha(ROOT/a['path'])==a['sha256']
    assert sha(ROOT.parent/'quartz_kinetic37_target_20261126/mcts_demo')==pl['identity']['binary']==sha(ROOT.parent/'quartz_kinetic36_target_20261126/mcts_demo')
    assert sha(ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi')==pl['identity']['bridge']
    for a in read('RAPFI_LOCK.json')['files']:assert sha(Path(a['path']))==a['sha256']
    roots={};seen=set();c=Counter()
    assert co['plan']==sha(D/'PLAN.json')
    for f in co['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];q=Counter()
        for z in raw(p):
            j=z['job'];i=j['id'];assert j==jobs[i] and i<1088 and i not in seen;seen.add(i);h=list(j['opening'])
            for a in z['rows']:v.gate_small(a,h,j,q);h.append(a['best'])
            b,p1,w=v.replay(h,'freestyle');reason='TERMINAL' if w is not None else 'GATED' if v.proof(tuple(b),p1)[0]['gate'] is not None else 'ADMITTED'
            assert reason==z['reason'] and h==z['history'] and z['side']==p1 and (len(h)==j['target'] or w is not None)
            q[reason]+=1
            if reason=='ADMITTED':roots[i]=dict(job=j,history=h,side=p1)
        for k,x in f.items():
            if k not in ('file','sha256'):assert q[k]==x
        c.update(q)
    assert len(seen)==1088 and [roots[i] for i in sorted(roots)]==co['roots']
    for k,x in co['counts'].items():assert c[k]==x
    out={'SOURCE':dict(c)};print('AUDIT SOURCE',dict(c),flush=True)
    lk=read('SCREEN_LOCK.json');assert lk['cohort']==sha(D/'COHORT.json') and lk['frame']==sha(D/'FRAME.json') and lk['code']==sha(D/'experiment.py')
    c=Counter();fmap={r['root']['job']['id']:r for r in fr};seen=set()
    for f,p in audit_files(lk):
        q=Counter()
        for z in raw(p):
            r=z['root'];j=r['job'];i=j['id'];h=r['history'];assert roots[i]==r and i not in seen;seen.add(i)
            order=[(h,lk) for h in H for lk in (False,True)];random.Random(pl['seed']+i).shuffle(order);assert z['order']==[list(x) for x in order]
            base=z['results']['B128'];assert base['proof']['gate'] is None
            for cap,lock in order:
                a=z['results'][('L' if lock else 'B')+str(cap)];v.audit_native(a,h,j,q,cap,lock)
                assert a['probe']==base['probe'] and a['prefix']['result']==base['prefix']['result'] and a['root_eval']==base['root_eval']
            for cap in H:
                a,b=z['results']['B'+str(cap)],z['results']['L'+str(cap)]
                assert a['best']==b['best'] and b['NN_calls']<=a['NN_calls'] and b['visits']<=a['visits'];q['parity']+=1
            aa=sorted({a['best'] for a in z['results'].values()});assert aa==z['actions']
            assert fmap[i]==dict(root=r,actions=aa,bin=base['bin'],arms={k:{x:a[x] for x in ('best','NN_calls','visits','cause')} for k,a in z['results'].items()})
            q['roots']+=1;q['disagree']+=len(aa)>1
        for k,x in f.items():
            if k not in ('file','sha256'):assert q[k]==x,(k,q[k],x)
        c.update(q)
    assert seen==set(roots)==set(fmap)
    for k,x in lk['counts'].items():assert c[k]==x
    out['SCREEN']=dict(c);print('AUDIT SCREEN',dict(c),flush=True)
    audit_sample(fr,sp);lab={}
    frozen=read('FROZEN.json')
    for mode,gs in [('TRAIN_CAL',('TRAIN','CAL')),('HELDOUT',('TEST','NATURAL'))]:
        lk=read(mode+'_LOCK.json');assert lk['sample']==sha(D/'SAMPLE.json') and lk['frame']==sha(D/'FRAME.json')
        if mode=='HELDOUT':assert lk['freeze']==sha(D/'FROZEN.json')
        else:assert lk['freeze'] is None
        expected={i for g in gs for i in sp['groups'][g]['stable_ids']+sp['groups'][g]['hard_ids']};seen=set();c=Counter()
        for f,p in audit_files(lk):
            q=Counter()
            for z in raw(p):
                r=z['root'];j=r['job'];i=j['id'];assert r==roots[i] and i in expected and i not in seen;seen.add(i)
                assert set(z['labels'])=={str(x) for x in fmap[i]['actions']}
                ll={}
                for a,ls in z['labels'].items():
                    assert [x['ref'] for x in ls]==[list(x) for x in REFS];vv=[]
                    for x,ref in zip(ls,REFS):
                        assert x['action']==int(a);h=r['history']+[int(a)]
                        for move in x['rows']:
                            b,p1,w=v.replay(h,'freestyle');assert w is None
                            cap=ref[0] if p1==r['side'] else ref[1]
                            v.gate_small(move,h,j,q,cap);h.append(move['best'])
                        b,p1,w=v.replay(h,'freestyle');assert w is not None and w==x['winner'] and h==x['history'] and F(x['utility'])==F(1+r['side']*w,2)
                        vv.append(F(x['utility']));q['labels']+=1
                    ll[a]=vv;q['actions']+=1
                assert i not in lab;lab[i]=ll;q['roots']+=1
            for k,x in f.items():
                if k not in ('file','sha256'):assert q[k]==x,(k,q[k],x)
            c.update(q)
        assert seen==expected
        for k,x in lk['counts'].items():assert c[k]==x
        out[mode]=dict(c);print('AUDIT LABEL',mode,dict(c),flush=True)
    model=read('MODEL.json');assert model['frame']==sha(D/'FRAME.json') and model['labels']==sha(D/'TRAIN_CAL_LOCK.json')
    f=fit(fr,lab);assert all(model[k]==x for k,x in f.items()),(model,f)
    for mode,gs in [('CAL',('CAL',)),('HELDOUT',('TEST','NATURAL'))]:
        lk=read(mode+'_LIVE_LOCK.json');p=D/f'{mode}_LIVE.jsonl.gz';assert sha(p)==lk['sha256']
        tabs={'L128':[128]*8,'L80':[80]*8,'table':model['table'] if mode=='CAL' else frozen['table']}
        c=Counter();vals=defaultdict(list);seen=set()
        for z in raw(p):
            r=z['root'];j=r['job'];i=j['id'];assert r==roots[i] and j['group'] in gs and i not in seen;seen.add(i);assert set(z['policies'])==set(tabs)
            for k,x in z['policies'].items():
                a=x['response'];v.gate_small(a,r['history'],j,c,128,True,tabs[k])
                ref=fmap[i]['arms']['L'+str(a['cap'])];assert all(a[t]==ref[t] for t in ('best','NN_calls','visits','cause'))
                d=gain(fmap[i],lab.get(i),a['cap']);near(x['gain'],d);near(x['net'],d-F(a['NN_calls'],500));vals[k].append((d,a['NN_calls']))
        assert seen=={i for i,r in roots.items() if r['job']['group'] in gs} and len(seen)==lk['n']
        for k,x in lk['counts'].items():assert c[k]==x
        def low(x):return float(mean(x))-2*S.stdev(x)/math.sqrt(len(x))
        stats={}
        for k,x in vals.items():
            u=[a for a,b in x];dn=[(a-F(b,500))-(aa-F(bb,500)) for (a,b),(aa,bb) in zip(x,vals['L80'])]
            y=dict(gain=mean(u),u_lower=low(u),NN=S.mean(b for a,b in x),net_gain=mean(dn),net_lower=low(dn))
            for t,vv in y.items():near(lk['stats'][k][t],vv)
            stats[k]=y
        if mode=='CAL':
            admitted=model['table'] not in ([80]*8,[128]*8) and stats['table']['net_lower']>0 and stats['table']['u_lower']>=-.05
            assert frozen['admitted']==admitted and frozen['table']==(model['table'] if admitted else [80]*8)
            assert frozen['model']==sha(D/'MODEL.json') and frozen['cal']==sha(D/'CAL_LIVE_LOCK.json')
        out[mode+'_LIVE']=dict(c)
    audit_abs(fr,lab,sp,read('ATLAS_SUMMARY.json'))
    dc=read('DECOMPOSITION.json')
    for g in GROUPS:
        rr=[r for r in fr if r['root']['job']['group']==g]
        for h in (96,112,128,256):
            d=[gain(r,lab.get(r['root']['job']['id']),h)-gain(r,lab.get(r['root']['job']['id']),80) for r in rr]
            k=[i for i,r in enumerate(rr) if r['arms']['L'+str(h)]['best']!=r['arms']['L80']['best']]
            z=dc['groups'][g][str(h)]
            assert z['flip']==len(k) and z['n']==len(rr)
            near(z['p_flip'],F(len(k),len(rr)));near(z['mean_gain'],mean(d))
            if k:near(z['conditional_gain'],mean([d[i] for i in k]))
            assert z['positive']==sum(d[i]>0 for i in k) and z['negative']==sum(d[i]<0 for i in k) and z['zero']==sum(d[i]==0 for i in k)
            cp=mean([r['arms']['L'+str(h)]['NN_calls']-r['arms']['L80']['NN_calls'] for r in rr]);near(z['extra_NN'],cp)
            if cp>0:near(z['signed_gain_per_NN'],mean(d)/cp)
    rr=[r for r in fr if r['root']['job']['group']=='NATURAL']
    yy=lambda r:mean(lab[r['root']['job']['id']][str(r['arms']['L80']['best'])])
    sy=[yy(r) for r in rr if len(r['actions'])==1];hy=[yy(r) for r in rr if len(r['actions'])>1];N=len(rr);M=len(sy);k=min(M,max(2,(M+3)//4))
    z=dc['natural_resampling'];truth=mean(sy+hy);near(z['true'],truth);assert z['N']==N and z['stable']==M and z['sample']==k and z['hard']==len(hy)
    var=F(M*(M-k),N*N)*sum((y-mean(sy))**2 for y in sy)/((M-1)*k);near(z['theoretical_variance'],var)
    rng=random.Random(pl['seed']+12345);es=[];ns=[];vs=[]
    for j in range(z['reps']):
        x=rng.sample(sy,k);xm=mean(x);es.append((M*xm+sum(hy))/N);ns.append(mean(x+hy));vs.append(F(M*(M-k),N*N)*sum((y-xm)**2 for y in x)/(k*(k-1)))
    near(z['HT_mean'],mean(es));near(z['HT_empirical_bias'],mean(es)-truth);near(z['naive_mean'],mean(ns));near(z['empirical_variance'],mean([(y-truth)**2 for y in es]));near(z['mean_estimated_variance'],mean(vs))
    print('AUDIT HT, COMPLETE CONTRASTS AND NATURAL RESAMPLING',flush=True)
    lk=read('ARENA_LOCK.json');ss=read('ARENA_SUMMARY.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['freeze']==sha(D/'FROZEN.json') and ss['alias']==lk['alias']
    pols={'B128':(128,False,None),'B80':(80,False,None),'L80':(80,True,None),'L128':(128,True,None),'ADAPT':(128,True,frozen['table'])}
    games=Counter();seen=set();hist={};groups=defaultdict(lambda:defaultdict(list))
    for f,p in audit_files(lk):
        c=Counter()
        for g in raw(p):
            j=g['job'];arm=g['arm'];color=g['color'];i=j['id'];assert jobs[i]==j and j['group']=='ARENA' and f['arm']==arm
            k=(arm,i,color);assert k not in seen;seen.add(k);h=list(j['opening']);cost=Counter()
            for a in g['rows']:
                b,p1,w=v.replay(h,'freestyle');who=arm if p1==color else 'B128';assert a['arm']==who
                v.gate_small(a,h,j,c,*pols[who]);tag='candidate' if p1==color else 'base'
                cost[tag+'_moves']+=1
                for key,x in [('NN',a['NN_calls']),('visits',a['visits']),('wall_ns',a['transport_ns']),('cpu_ns',sum(a['cpu_ns'])),('wait_ns',a['bridge_wait_ns'])]:cost[tag+'_'+key]+=x
                cost[tag+'_'+a['cause']]+=1;h.append(a['best'])
            b,p1,w=v.replay(h,'freestyle');assert w is not None and w==g['winner'] and h==g['history'] and F(g['score'])==F(1+color*w,2)
            groups[arm][i].append((F(g['score']),color,cost));hist[k]=g;c['games']+=1
        for k in ('games','requests','NN','visits'):assert c[k]==f[k]
        games.update(c)
    for k,x in lk['counts'].items():assert games[k]==x
    for arm,gg in groups.items():
        y=ss['arms'][arm];assert len(gg)==192 and all(len(x)==2 and {a[1] for a in x}=={-1,1} for x in gg.values())
        d={i:mean([a[0] for a in x]) for i,x in gg.items()};cs={}
        for i,x in gg.items():
            c=Counter()
            for _,_,q in x:c.update(q)
            cs[i]=c
        cc=Counter()
        for c in cs.values():cc.update(c)
        for k,x in cc.items():assert y['counts'][k]==x
        near(y['score'],mean(list(d.values())));assert y['wins']==sum(a[0]==1 for x in gg.values() for a in x) and y['losses']==sum(a[0]==0 for x in gg.values() for a in x)
        def red(c):return 1-F(c['candidate_NN']*c['base_moves'],c['candidate_moves']*c['base_NN'])
        rng=random.Random(pl['seed']+42);ids=sorted(gg);bs=[];bn=[]
        for _ in range(4000):
            ix=rng.choices(ids,k=len(ids));bs.append(mean([d[i] for i in ix]));q=Counter()
            for i in ix:q.update(cs[i])
            bn.append(red(q))
        bs.sort();bn.sort();near(y['NI_lower'],bs[199]);near(y['score_ci'][0],bs[99]);near(y['score_ci'][1],bs[3899]);near(y['NN_reduction'],red(cc));near(y['NN_ci'][0],bn[99]);near(y['NN_ci'][1],bn[3899])
        assert y['joint']==(y['NI_lower']>=.45 and y['NN_reduction']>=.3)
    parity80=Counter();parity128=Counter()
    for i in range(1088,1280):
        for color in (-1,1):
            a,b=hist['B80',i,color],hist['L80',i,color];assert a['history']==b['history'] and a['winner']==b['winner'];parity80['games']+=1
            for x,y in zip(a['rows'],b['rows']):
                assert x['history']==y['history'] and x['best']==y['best']
                if x['player']==color:
                    assert x['arm']=='B80' and y['arm']=='L80' and y['NN_calls']<=x['NN_calls']
                    parity80.update(moves=1,before=x['NN_calls'],after=y['NN_calls'],stops=int(y['cause']=='VOTE_LOCK'))
                else:assert x['NN_calls']==y['NN_calls'] and x['visits']==y['visits']
        a,b=hist['L128',i,1],hist['L128',i,-1];assert a['history']==b['history'] and a['winner']==b['winner'];parity128['games']+=2
        for x,y in zip(a['rows'],b['rows']):
            assert x['history']==y['history'] and x['best']==y['best']
            if x['arm']=='B128':x,y=y,x
            assert x['arm']=='L128' and y['arm']=='B128' and x['NN_calls']<=y['NN_calls']
            parity128.update(moves=1,before=y['NN_calls'],after=x['NN_calls'],stops=int(x['cause']=='VOTE_LOCK'))
    for z in (parity80,parity128):z['saved']=z['before']-z['after']
    out['ARENA']=dict(games);print('AUDIT ARENA',dict(games),dict(parity80),dict(parity128),flush=True)
    lk=read('DIAGNOSTIC_LOCK.json');p=D/'DIAGNOSTIC.jsonl.gz';assert sha(p)==lk['sha256'];c=Counter();n=0
    for z in raw(p):
        r=z['root'];j=r['job'];assert r==roots[j['id']]
        for k,a in z['results'].items():v.audit_native(a,r['history'],j,c,int(k[1:]),k[0]=='L',trace=n<3)
        n+=1
    assert n==lk['roots']==12
    for k,x in lk['counts'].items():assert c[k]==x
    out['DIAGNOSTIC']=dict(c)
    (D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    (D/'PARITY_AUDIT.json').write_text(json.dumps(dict(same80=dict(parity80),same128=dict(parity128),same80_reduction=parity80['saved']/parity80['before'],same128_reduction=parity128['saved']/parity128['before']),indent=2,sort_keys=True)+'\n')
    print('AUDIT_OK',json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
