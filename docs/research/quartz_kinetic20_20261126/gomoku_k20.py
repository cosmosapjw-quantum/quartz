"""Two actual hidden-edge Rapfi pilot; distinct candidate labels, no model fit."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,random,collections as C,statistics as S,time,concurrent.futures as CF
from acquire import D,ROOT,p,old,sha,save,read,raw,emit,BIN,identity,lock
from pathlib import Path
import arena
from run_k19 import D as P19
from gomoku_k19 import historical as previous_historical
GBASE=old.identity()
GID=dict(parent=GBASE,native=identity())
arena.BIN=BIN
arena.identity=lambda:GID
GSEED=202611262001
P16=ROOT.parent/'quartz_kinetic16_20261126/docs/research/quartz_kinetic16_20261126'
ARMS=('DS0','DSX','DSX2')
POLICIES=(('STANDARD_PW',128),('STANDARD_FULL',256))
def historical():
    fam,boards,sources=previous_historical()
    q=ROOT/'docs/research/quartz_kinetic19_20261126'
    for name in ('G_PLAN.json','G_SOURCE.jsonl.gz','G_LABELS.jsonl.gz'):
        sources.append(lock(q/name))
    plan=json.loads((q/'G_PLAN.json').read_text())
    fam.update(j['family'] for j in plan['jobs'])
    for row in raw(q/'G_SOURCE.jsonl.gz'):
        h=list(row['job']['opening'])
        for a in row['rows']:
            b,side,w=old.replay(h,'freestyle');boards.add(old.orbit_board(b,side,'freestyle'));h.append(a['move'])
        b,side,w=old.replay(h,'freestyle');boards.add(old.orbit_board(b,side,'freestyle'))
    panel=json.loads((q/'G_PANEL.json').read_text());roots={x['root']['id']:x['root'] for x in panel['panels']}
    for row in raw(q/'G_LABELS.jsonl.gz'):
        h=roots[row['id']]['history']+[row['action']]
        for a in row['rows']:
            b,side,w=old.replay(h,'freestyle');boards.add(old.orbit_board(b,side,'freestyle'));h.append(a['move'])
        b,side,w=old.replay(h,'freestyle');boards.add(old.orbit_board(b,side,'freestyle'))
    return fam,boards,sources
def plan():
    fam,boards,src=historical();n=len(fam);rng=random.Random(GSEED)
    cells=[15*i+j for i in range(4,11) for j in range(4,11)];jobs=[]
    for i in range(12):
        while True:
            h=rng.sample(cells,6);k=old.family(h)
            if k not in fam:break
        fam.add(k);jobs.append(dict(id=i,opening=h,family=k))
    save('G_PLAN.json',dict(seed=GSEED,identity=GID,old_families=n,old_boards=len(boards),source_locks=src,jobs=jobs))
    print('G PLAN',len(jobs),'historical',n,len(boards),flush=True)
def source():
    pl=read('G_PLAN.json');assert pl['identity']==GID
    fam,boards,src=historical();assert src==pl['source_locks']
    path=D/'G_SOURCE.jsonl.gz';assert not path.exists()
    roots=[];nn=0;plies=0
    with old.Server() as s,gzip.open(path,'wt',compresslevel=3) as fd:
        for j in pl['jobs']:
            assert j['family'] not in fam;fam.add(j['family'])
            h=list(j['opening']);rr=[];eligible=[]
            while True:
                b,p0,w=old.replay(h,'freestyle')
                if w is not None:break
                proof=old.proof_reference(b,p0,'freestyle');key=old.orbit_board(b,p0,'freestyle')
                if 8<=len(h)<=14 and not proof['own'] and not proof['opponent'] and key not in boards:eligible.append((list(h),p0,key))
                if len(h)>=14:break
                a=s.search(h,'STANDARD_PW',128,17)
                rr.append(dict(move=a['best'],visits=a['visits'],NN=a['NN_calls'],cause=a['cause'],wall_ns=a['transport_ns']));nn+=a['NN_calls'];plies+=1;h.append(a['best'])
            root=None
            if eligible:
                hist,side,key=eligible[-1];boards.add(key)
                root=dict(id=j['id'],family=j['family'],history=hist,player=side);roots.append(root)
            emit(fd,dict(job=j,history=h,rows=rr,selected=root))
            print('G SOURCE',j['id'],None if root is None else len(root['history']),flush=True)
    save('G_COHORT.json',dict(plan=sha(D/'G_PLAN.json'),source=sha(path),roots=roots,source_plies=plies,source_NN=nn,missing=[j['id'] for j in pl['jobs'] if j['id'] not in {r['id'] for r in roots}]))
def atlas():
    co=read('G_COHORT.json');assert co['source']==sha(D/'G_SOURCE.jsonl.gz')
    path=D/'G_ATLAS.jsonl.gz';assert not path.exists()
    panels=[];nn=0;req=0;hidden=0
    with old.Server() as s,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            a={arm:s.search(r['history'],arm,32,17,prefix=8) for arm in ARMS}
            assert all(x['decision'] for x in a.values())
            pre=a['DS0']['decision']['phases'][0]['result'];assert pre['root_n']==8
            assert all(x['decision']['phases'][0]['result']==pre for x in a.values())
            ca,n,w=p.pr.moments(pre);visible={x['move'] for x in pre['edges']};panel=[pre['best']]
            targets={};proof= a['DS0']['proof']
            for arm in ('DSX','DSX2'):
                t=a[arm]['decision']['target'];targets[arm]=None if t is None else t['move']
                if t is not None:
                    m=t['move'];assert m in ca
                    if m not in panel:panel.append(m)
                    nxt=a[arm]['decision']['phases'][1]['result'];c1,n1,w1=p.pr.moments(nxt);assert c1==ca
                    assert n1[ca.index(m)]-n[ca.index(m)]==4
                    if arm in ('DSX','DSX2'):
                        assert t['materialized'] is False and m not in visible and m in {e['move'] for e in nxt['edges']} and n[ca.index(m)]==0
                        hidden+=1
                    else:raise AssertionError('unexpected target')
            if targets['DSX2'] is not None:
                assert targets['DSX2']!=targets['DSX']
                assert next(row[2] for row in proof['actions'] if row[0]==targets['DSX2'])> -1
            native={arm:x['best'] for arm,x in a.items()}
            qbest={}
            for arm,x in a.items():
                c,nf,wf=p.pr.moments(x['result'])
                qbest[arm]=max(panel,key=lambda m:wf[c.index(m)]/(nf[c.index(m)]+4))
                emit(fd,dict(root=r,arm=arm,response=x));nn+=x['NN_calls'];req+=1
            moves=sorted(set(panel)|set(native.values()))
            panels.append(dict(root=r,anchor=pre['best'],prefix_panel=panel,targets=targets,initial_visible=len(visible),final_visible={arm:len(x['result']['edges']) for arm,x in a.items()},native=native,qbest=qbest,moves=moves))
            print('G ATLAS',r['id'],targets,'visible',len(visible),flush=True)
    save('G_PANEL.json',dict(cohort=sha(D/'G_COHORT.json'),atlas=sha(path),panels=panels,requests=req,NN=nn,hidden_materialized=hidden))
def label_one(z):
    r=z['root'];out=[]
    with old.Server() as s:
        for a in z['moves']:
            for arm,cap in POLICIES:
                h=r['history']+[a];rr=[]
                while True:
                    b,side,w=old.replay(h,'freestyle')
                    if w is not None:break
                    x=s.search(h,arm,cap,17)
                    rr.append(dict(move=x['best'],NN=x['NN_calls'],visits=x['visits'],cause=x['cause'],wall_ns=x['transport_ns']))
                    h=h+[x['best']]
                out.append(dict(id=r['id'],action=a,arm=arm,cap=cap,history=h,winner=w,utility=(1+r['player']*w)/2,rows=rr))
    return out
def labels():
    pan=read('G_PANEL.json');assert pan['atlas']==sha(D/'G_ATLAS.jsonl.gz')
    path=D/'G_LABELS.jsonl.gz';assert not path.exists();n=nn=plies=0
    # Two separate single-worker native evaluator sessions; label timing not a performance comparison.
    with gzip.open(path,'wt',compresslevel=3) as fd, CF.ProcessPoolExecutor(max_workers=2) as pool:
        for rr in pool.map(label_one,pan['panels']):
            for x in rr:
                emit(fd,x);n+=1;nn+=sum(a['NN'] for a in x['rows']);plies+=len(x['rows'])
            print('G LABEL',rr[0]['id'],len(rr),'games',n,flush=True)
    save('G_LABELS_LOCK.json',dict(panel=sha(D/'G_PANEL.json'),sha256=sha(path),games=n,NN=nn,plies=plies,policies=POLICIES))
def score():
    pan=read('G_PANEL.json');lk=read('G_LABELS_LOCK.json');assert lk['sha256']==sha(D/'G_LABELS.jsonl.gz') and lk['panel']==sha(D/'G_PANEL.json')
    u={};pol={}
    for z in raw(D/'G_LABELS.jsonl.gz'):
        k=(z['id'],z['action']);pol.setdefault(k,{})[z['arm']]=z['utility']
    for k,v in pol.items():
        assert set(v)=={x[0] for x in POLICIES};u[k]=S.mean(v.values())
    rows=[]
    for z in pan['panels']:
        i=z['root']['id'];U={m:u[i,m] for m in z['moves']};ref=max(U.values());a=z['anchor']
        q=dict(id=i,anchor_u=U[a],labeled_best=ref,hidden_u=None if z['targets']['DSX'] is None else U[z['targets']['DSX']],hidden2_u=None if z['targets']['DSX2'] is None else U[z['targets']['DSX2']],u=U,targets=z['targets'],native={c:U[m] for c,m in z['native'].items()},q={c:U[m] for c,m in z['qbest'].items()},initial_visible=z['initial_visible'],final_visible=z['final_visible'])
        rows.append(q)
    native={};qb={}
    for c in ARMS:
        native[c]=dict(utility=S.mean(z['native'][c] for z in rows),panel_regret=S.mean(z['labeled_best']-z['native'][c] for z in rows),gain_vs_DS0=S.mean(z['native'][c]-z['native']['DS0'] for z in rows))
        qb[c]=dict(utility=S.mean(z['q'][c] for z in rows),panel_regret=S.mean(z['labeled_best']-z['q'][c] for z in rows),gain_vs_DS0=S.mean(z['q'][c]-z['native']['DS0'] for z in rows))
    h=[z['hidden_u']-z['anchor_u'] for z in rows if z['hidden_u'] is not None]
    h2=[z['hidden2_u']-z['anchor_u'] for z in rows if z['hidden2_u'] is not None]
    paired=[z['hidden2_u']-z['hidden_u'] for z in rows if z['hidden2_u'] is not None and z['hidden_u'] is not None]
    resp={}
    for c in ARMS:
        rr=[t['response'] for t in raw(D/'G_ATLAS.jsonl.gz') if t['arm']==c]
        resp[c]=dict(mean_wall_ms=S.mean(x['transport_ns']/1e6 for x in rr),mean_NN=S.mean(x['NN_calls'] for x in rr),mean_phase_ms=S.mean(sum(k['step_us'] for k in x['decision']['phases'][1:])/1000 for x in rr))
    s=dict(n=len(rows),requests=pan['requests'],native_NN=pan['NN'],hidden_materialized=pan['hidden_materialized'],hidden_positive=sum(x>0 for x in h),hidden_negative=sum(x<0 for x in h),hidden_equal=sum(x==0 for x in h),hidden_mean_gain=S.mean(h) if h else None,hidden2_positive=sum(x>0 for x in h2),hidden2_negative=sum(x<0 for x in h2),hidden2_equal=sum(x==0 for x in h2),hidden2_mean_gain=S.mean(h2) if h2 else None,second_vs_first_mean=S.mean(paired) if paired else None,action_pairs=len(pol),policy_disagreements=sum(len(set(v.values()))>1 for v in pol.values()),games=lk['games'],label_NN=lk['NN'],label_plies=lk['plies'],native=native,q=qb,cost=resp,rows=rows)
    save('G_SUMMARY.json',s);print('G SUMMARY',json.dumps({k:v for k,v in s.items() if k!='rows'}),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
