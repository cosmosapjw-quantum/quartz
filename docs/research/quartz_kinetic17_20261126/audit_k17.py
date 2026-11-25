"""Independent grid replay, raw-observation/label and aggregate checks; no pilot imports."""
import collections as C, gzip, hashlib, json, math, pathlib, random, statistics as S, subprocess as SP, os
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def won(b,p):
    for x in range(7):
        for y,v in enumerate(b[x]):
            if v!=p:continue
            for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
                if all(0<=x+i*dx<7 and 0<=y+i*dy<len(b[x+i*dx]) and b[x+i*dx][y+i*dy]==p for i in (1,2,3)):return True
    return False
def canwin(b,p):
    for col in b:
        if len(col)==6:continue
        col.append(p)
        w=won(b,p)
        col.pop()
        if w:return True
    return False
def replay(h):
    b=[[] for _ in range(7)]
    for i,s in enumerate(h):
        c=int(s)-1;assert 0<=c<7 and len(b[c])<6
        b[c].append(i%2+1);assert not won(b,i%2+1)
    return b
def canonical(h):return min(h,''.join(str(8-int(x)) for x in h))
def boardkey(b):
    k=tuple(tuple(c) for c in b);return min(k,k[::-1])
def moments(s):
    ca=[a for a,_ in s['candidates']]
    e={a['move']:a for a in s['edges']}
    assert len(ca)==len(set(ca)) and set(e)<=set(ca)
    n={a:e.get(a,{}).get('n',0) for a in ca};w={a:e.get(a,{}).get('w',0.) for a in ca}
    assert sum(n.values())==s['root_n'] and abs(sum(w.values())-s['root_w'])<1e-5
    assert all(n[a]>=0 and abs(w[a])<=n[a]+1e-6 for a in ca)
    return ca,n,w,e
def audit(p,quiet,allfam,allboards):
    co=read(p/'COHORT.json');lock=read(p/'ATLAS_LOCK.json');lab=read(p/'LABELS.json');su=read(p/'SUMMARY.json')
    assert sha(p/'COHORT.json')==lock['cohort']==lab['cohort']==su['cohort']
    assert sha(p/'ATLAS.jsonl.gz')==lock['sha256']==lab['atlas']==su['atlas']
    assert sha(p/'LABELS.json')==su['labels']
    roots={r['id']:r for r in co['roots']}
    assert len(co['jobs'])==(192 if quiet else 96)
    for j in co['jobs']:
        i=j['id'];h=j['opening'];fam=canonical(h)
        assert fam==j['family'] and fam not in allfam;allfam.add(fam)
        assert j['group']==('TRAIN','CAL','TEST')[i//(64 if quiet else 32)]
        assert j['depth']==((24,26,28) if quiet else (26,28,30))[i%3]
        b=replay(h);rng=random.Random(j['seed'])
        while len(h)<j['depth']:
            side=len(h)%2+1;opts=[]
            for c,col in enumerate(b):
                if len(col)==6:continue
                col.append(side)
                ok=not won(b,side) and (not quiet or not any(canwin(b,k) for k in (1,2)))
                col.pop()
                if ok:opts.append(c)
            if not opts:break
            c=rng.choice(opts);b[c].append(side);h+=str(c+1)
        assert h==j['history']
        legal=[c for c in range(7) if len(b[c])<6]
        assert j['accepted']==(i in roots)
        if i in roots:
            r=roots[i];key=boardkey(b)
            assert key not in allboards;allboards.add(key)
            assert r['history']==h and r['group']==j['group'] and r['family']==fam and len(h)==j['depth'] and len(legal)>=3
            assert r['legal']==legal and r['tactical']==any(canwin(b,k) for k in (1,2))
            if quiet:assert not r['tactical']
    labels={z['root']['id']:z for z in lab['rows']}
    assert labels.keys()==roots.keys()
    u={}
    for i,z in labels.items():
        assert z['status']=='EXACT' and z['agreement'] and z['root']==roots[i]
        v={int(a):k for a,k in z['independent'].items()}
        assert set(v)==set(roots[i]['legal'])
        assert all(k==(1 if z['scores'][a]>0 else -1 if z['scores'][a]<0 else 0) for a,k in v.items())
        u[i]={a:(k+1)/2 for a,k in v.items()}
    stored={(x['id'],x['cap'],x['arm']):x for x in su['rows']}
    seen=set();pre={};coverage={};vis=C.Counter();tot=C.Counter();result=[]
    for line in gzip.open(p/'ATLAS.jsonl.gz','rt'):
        t=json.loads(line);r=t['root'];i=r['id'];cap=t['cap'];arm=t['arm'];a=t['response'];e=a['evidence'];key=(i,cap,arm)
        assert key not in seen and r==roots[i];seen.add(key)
        assert a['visits']==cap and a['best'] in r['legal'] and a['history']==r['history']
        x=e['prefix'];ca,n,w,edges=moments(x)
        assert x['root_n']==8 and set(ca)==set(r['legal'])
        if i in pre:assert x==pre[i]
        else:pre[i]=x
        pair=e['pair'];assert len(pair)==2 and len(set(pair))==2 and pair[0]==x['best']
        third=next(m for m in ca if m not in pair);panel=pair+[third]
        if i not in coverage:
            coverage[i]=panel
            vis[(r['group'],'roots')]+=1
            vis[(r['group'],'third_visible')]+=int(third in edges)
            vis[(r['group'],'third_visited')]+=int(n[third]>0)
            vp=max(u[i][m] for m in pair);vt=max(u[i][m] for m in panel);vl=max(u[i].values())
            vis[(r['group'],'pair_omission')]+=int(vp<vl)
            vis[(r['group'],'third_recovery_any')]+=int(vt>vp)
            vis[(r['group'],'third_recovery_full')]+=int(vp<vl and vt==vl)
        else:assert coverage[i]==panel
        q=(cap-16)//2
        targets=[None,None,None] if arm=='K16N' else [pair[0],pair[1] if arm=='K16BAL' else third,None]
        if arm=='K17EXP':assert e['reservoir']==panel
        prev=n;pw=w;step=8
        for ix,(phase,k,target) in enumerate(zip(e['phases'],(q,q,8),targets,strict=True)):
            y=phase['result'];cs,nn,ww,_=moments(y);step+=k
            assert cs==ca and y['root_hash']==x['root_hash'] and y['root_n']==step
            diff={m:nn[m]-prev[m] for m in ca}
            assert [diff[m] for m in ca]==phase['delta'] and min(diff.values())>=0 and sum(diff.values())==k and phase['target']==target
            assert target is None or diff[target]==k
            assert all(abs(ww[m]-pw[m])<=diff[m]+1e-6 for m in ca)
            if ix==1:pulse={m:(ww[m]-w[m]+4*w[m]/(n[m]+4))/(nn[m]-n[m]+4) for m in ca}
            prev,pw=nn,ww
        assert y==a['result'] and y['evaluator_calls']==a['NN_calls']
        final={m:pw[m]/(prev[m]+4) for m in ca}
        o={'pair':pair,'panel':panel,'native':a['best']}
        for tag,sc in [('pulse',pulse),('final',final)]:
            for kind,cands in [('pair',pair),('triple',panel)]:o[tag+'_'+kind]=max(cands,key=lambda m:sc[m])
        assert o==t['readout']
        z=stored[key];v=u[i];L=max(v.values());A=max(v[m] for m in pair);B=max(v[m] for m in panel)
        assert z['omission_pair']==L-A and z['omission_triple']==L-B and z['oracle_access']==B-A
        for typ in ('native','pulse_pair','pulse_triple','final_pair','final_triple'):
            assert z['regret_'+typ]==L-v[o[typ]]
            if typ!='native':assert z['rank_'+typ]==(A if typ.endswith('pair') else B)-v[o[typ]]
        assert z['NN']==a['NN_calls'] and z['cpu_ms']==a['cpu_ns']/1e6 and z['wall_ms']==a['wall_ns']/1e6
        result.append(z);tot['visits']+=cap;tot['NN']+=a['NN_calls']
    assert seen==stored.keys() and len(seen)==len(roots)*6==lock['requests'] and lock['baseline_parity']==len(roots)*2
    for row in su['summary']:
        xs=[z for z in result if z['cap']==row['cap'] and z['arm']==row['arm'] and (row['group']=='ALL' or z['group']==row['group'])]
        assert len(xs)==row['n']
        for k in row.keys()-{'group','cap','arm','n'}:assert math.isclose(S.mean(z[k] for z in xs),row[k],abs_tol=1e-12)
    return dict(roots=len(roots),jobs=len(co['jobs']),requests=len(seen),exact_actions=sum(map(len,u.values())),independent_nodes=sum(z['nodes'] for z in labels.values()),baseline_parity_recorded=lock['baseline_parity'],aggregate_rows=len(su['summary']),group_diagnostics={g:{k:v for (gg,k),v in vis.items() if gg==g} for g in ('TRAIN','CAL','TEST')},total=dict(tot))
def main():
    fam=set();boards=set()
    a=audit(D,False,fam,boards);b=audit(D/'quiet',True,fam,boards)
    # Verify native boundary requests using executable identity and unmodified request schema.
    bid=read(D/'RUST_BUILD_IDENTITY.json');bin=ROOT.parent/'quartz_kinetic17_target_20261126/mcts_demo'
    assert sha(bin)==bid['binary_sha256']
    co=read(D/'COHORT.json');h=co['roots'][0]['history']
    model=ROOT/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/evidence/mlp_conv_train_20261121/conv_model.json'
    request=dict(cmd='research_connect4_evidence_cpu',history=h,arm='K17EXP',cap=32,seed=17,model_path=str(model),model_identity=co['identity']['model'])
    # Deterministically find nonterminal two-legal root, independent grid rules.
    rng=random.Random(1709);two=None
    for _ in range(2000):
        board=[[] for _ in range(7)];hist=''
        while len(hist)<42:
            if sum(len(c)<6 for c in board)==2:two=hist;break
            p=len(hist)%2+1;opts=[]
            for c,col in enumerate(board):
                if len(col)==6:continue
                col.append(p);ok=not won(board,p);col.pop()
                if ok:opts.append(c)
            if not opts:break
            c=rng.choice(opts);board[c].append(p);hist+=str(c+1)
        if two:break
    assert two
    qs=[dict(request,cap=48),dict(request,history='88888'),dict(request,history=two),dict(request,oracle=[1,2,3])]
    z=SP.run([str(bin),'--server'],input=''.join(json.dumps(q)+'\n' for q in qs),capture_output=True,text=True,timeout=30,env=dict(os.environ,RAYON_NUM_THREADS='1'))
    replies=[json.loads(l) for l in z.stdout.splitlines()]
    assert z.returncode==0 and len(replies)==len(qs) and all('error' in x for x in replies),replies
    return dict(broad=a,quiet=b,distinct_attempt_families=len(fam),distinct_accepted_boards=len(boards),invalid_requests=replies,scope='independent grid replay of frozen jobs and fresh-family disjointness within K17, exact WDL sign readback, raw counts/means/readouts, all aggregate means; historical exclusions and independent negamax computed in previous stages; not full selection/backup trace')
if __name__=='__main__':print(json.dumps(main(),indent=2,sort_keys=True))
