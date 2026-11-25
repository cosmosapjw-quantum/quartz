"""Independent row-major WDL oracle and replay of actual rule/candidate evidence."""
from pair_run import *
from publish import B,BASE,BR,git
from functools import lru_cache
from fractions import Fraction as F
import tempfile
import pair_ref as pr
import score_pair as sc
p=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
sp=importlib.util.spec_from_file_location('old_rule_reference',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
# Independent row-major representation (not Pons sentinel bitboard).
lines=[sum(1<<(7*(y+k*dy)+x+k*dx) for k in range(4))
       for y in range(6) for x in range(7) for dx,dy in ((1,0),(0,1),(1,1),(-1,1))
       if 0<=x+3*dx<7 and 0<=y+3*dy<6]
assert len(lines)==69 and len(set(lines))==69
through=[[m for m in lines if m>>i&1] for i in range(42)]
def independent_c4(h):
    p=[0,0];ht=[0]*7
    for i,m in enumerate(h):
        c=int(m)-1;assert 0<=c<7 and ht[c]<6
        bit=1<<(7*ht[c]+c);p[i%2]|=bit
        assert not any(p[i%2]&x==x for x in through[7*ht[c]+c])
        ht[c]+=1
    return p,tuple(ht)
def exact_wdl(h):
    p,ht=independent_c4(h);cnt=0
    def children(heights):
        for c in (3,2,4,1,5,0,6):
            if heights[c]==6:continue
            idx=heights[c]*7+c;hh=list(heights);hh[c]+=1
            yield c,idx,tuple(hh)
    @lru_cache(None)
    def V(a,b,heights):
        nonlocal cnt
        cnt+=1
        if cnt>2_000_000:raise RuntimeError('bounded independent oracle node cap')
        if sum(heights)==42:return 0
        cs=list(children(heights))
        for c,i,hh in cs:
            x=a|(1<<i)
            if any(x&m==m for m in through[i]):return 1
        result=-1
        for c,i,hh in cs:
            v=-V(b,a|(1<<i),hh);result=max(result,v)
            if v==1:break
        return result
    a,b=p[len(h)%2],p[1-len(h)%2]
    out={}
    for c,i,hh in children(ht):
        x=a|(1<<i);out[c]=1 if any(x&m==m for m in through[i]) else -V(b,x,hh)
    return out,cnt
cache={}
def grule(h):
    k=tuple(h)
    if k not in cache:
        b,p,w=old.replay(h,'freestyle')
        cache[k]=(b,p,w,None if w is not None else old.proof_reference(b,p,'freestyle'))
    return cache[k]
def verify_g(a,h,arm,cap):
    b,p,w,proof=grule(h);assert w is None and a['board']==b and a['player']==p and a['proof']==proof
    check(a,h,arm,cap,17)
    return len(a['legal'])
def game(g,h,arm,cap):
    h=list(h);c=C.Counter()
    for t in g['rows']:
        a=t['response'];b,p,w,pf=grule(h)
        assert w is None and t['ply']==len(h) and t['player']==p and t['move']==a['best']
        c.update(plies=1,intervals=verify_g(a,h,arm,cap),NN=a['NN_calls'])
        h.append(t['move'])
    b,p,w,_=grule(h);assert w is not None and w==g['winner_black'] and h==g['final_history']
    return c
def audit():
    plan=read('C06_PLAN.json');co=read('C06_COHORT.json');pan=read('C07_PANEL.json');fit=read('C08_FROZEN.json');su=read('C08_SUMMARY.json')
    assert plan['identity']==co['identity']==fit['identity']==su['identity']==ident()
    assert sha(D/'C06_GOMOKU_SOURCE.jsonl.gz')==co['source_sha256']
    assert sha(D/'C07_ATLAS.jsonl.gz')==fit['atlas_sha256']==pan['atlas_sha256']==read('C07_ATLAS_LOCK.json')['sha256']
    assert sha(D/'C08_TRAIN_LABELS.jsonl.gz')==fit['train_sha256']==read('C08_TRAIN_LOCK.json')['sha256']
    cf,cb,locks=c4historical();assert locks==plan['c4_sources']
    gp,gb=ghistorical();assert len(cb)==plan['c4_old_boards'] and len(cf)==plan['c4_old_families'] and len(gb)==plan['g_old_boards']
    rr={r['id']:r for r in co['roots']};c=C.Counter()
    seen=set()
    for j in plan['c4']:
        assert mirror(j['opening'])==j['family'] and j['family'] not in cf|seen;seen.add(j['family'])
    attempts={x['job']['id']:x for x in co['c4_attempts']}
    count=C.Counter();used=[]
    for j in plan['c4']:
        if count[j['group']]>=co['c4_targets'][j['group']]:continue
        a=attempts[j['id']];used.append(j['id']);assert a['job']==j
        h=j['opening'];p,ht=independent_c4(h);ht=list(ht);rng=random.Random(j['seed'])
        while len(h)<28:
            i=len(h)%2;opts=[]
            for col,n in enumerate(ht):
                if n<6:
                    ix=7*n+col;x=p[i]|(1<<ix)
                    if not any(x&m==m for m in through[ix]):opts.append(col)
            if not opts:break
            col=rng.choice(opts);p[i]|=1<<(7*ht[col]+col);ht[col]+=1;h+=str(col+1)
        b=[1 if p[0]>>i&1 else -1 if p[1]>>i&1 else 0 for i in range(42)]
        key=boardkey(b);leg=[i for i,v in enumerate(ht) if v<6]
        win=lambda x:any(any((x|(1<<(7*ht[k]+k)))&m==m for m in through[7*ht[k]+k]) for k in leg)
        ok=len(h)==28 and len(leg)>=2 and not any(win(x) for x in p) and key not in cb
        assert a['history']==h and a['accepted']==ok
        if ok:
            cb.add(key);count[j['group']]+=1
            assert rr[j['id']]==dict(id=j['id'],domain='C4',group=j['group'],family=j['family'],history=h,legal=leg,player=1)
        else:assert j['id'] not in rr
        c['c4_source_attempts']+=1
    assert set(used)==set(attempts)
    gs=list(rows('C06_GOMOKU_SOURCE.jsonl.gz'));gmiss=[]
    for g,j in zip(gs,plan['gomoku'],strict=True):
        assert g['job']==j and family(j['opening'])==j['family'] and j['family'] not in gp;gp.add(j['family'])
        z=game(g,j['opening'],'STANDARD_PW',128);c.update({'source_'+k:v for k,v in z.items()})
        elig=[]
        for row in g['rows']:
            a=row['response'];b,p,w,pf=grule(a['history']);k=old.orbit_board(b,p,'freestyle')
            if not pf['own'] and not pf['opponent'] and len(a['legal'])>=2 and k not in gb:elig.append((a,k))
        sel=None
        if elig:
            a,k=min(elig,key=lambda x:(abs(len(x[0]['history'])-14),len(x[0]['history'])))
            gb.add(k);sel=dict(id=j['id'],domain='GOMOKU',group=j['group'],family=j['family'],history=a['history'],legal=a['legal'],player=a['player'])
        assert sel==g['selected']
        if sel:assert rr[j['id']]==sel
        else:gmiss.append(j['id'])
    assert gmiss==co['missing_gomoku']
    aa={};needs=C.defaultdict(set)
    for z in atlas_rows():
        r=z['root'];i=r['id'];a=z['response'];arm=z['arm'];cap=z['cap'];k=i,cap,arm
        assert r==rr[i] and k not in aa;aa[k]=a
        if r['domain']=='GOMOKU':c['atlas_gomoku_intervals']+=verify_g(a,r['history'],arm,cap)
        else:
            p,ht=independent_c4(r['history']);assert a['history']==r['history'] and a['legal']==[j for j,n in enumerate(ht) if n<6] and a['model_identity']==MID
        c['forced_visits']+=pr.verify(a);c.update(atlas_requests=1,visits=a['visits'],NN=a['NN_calls'],phase_checks=3)
        e=a['evidence'];needs[i].update(e['pair']+[a['best'],e['prefix_best'],e['pulse_best'],e['final_best']])
    assert set(aa)=={(i,b,a) for i in rr for b in CAPS for a in ARMS}
    for i,r in rr.items():
        p0=aa[i,32,'K16N']['evidence']['prefix']
        for cap in CAPS:
            for arm in ARMS:
                assert aa[i,cap,arm]['evidence']['prefix']==p0;c['prefix_equalities']+=1
        assert set(pan['panels'][i]['moves'])==needs[i]
    u={};gval=C.defaultdict(dict);orows=[]
    for part in ('TRAIN','HELDOUT'):
        fn=f'C08_{part}_LABELS.jsonl.gz';assert sha(D/fn)==read(f'C08_{part}_LOCK.json')['sha256']
        for g in rows(fn):
            r=g['root'];i=r['id'];assert r==rr[i] and ((r['group']=='TRAIN')==(part=='TRAIN'))
            if r['domain']=='C4':
                assert g['status']=='EXACT' and i not in u
                wdl,nodes=exact_wdl(r['history'])
                assert set(wdl)==set(r['legal'])
                assert wdl=={k:(g['scores'][k]>0)-(g['scores'][k]<0) for k in r['legal']},(i,wdl,g['scores'])
                u[i]={k:F(v+1,2) for k,v in wdl.items()};c['oracle_independent_roots']+=1;c['oracle_independent_actions']+=len(wdl);c['oracle_independent_nodes']+=nodes
                orows.append(dict(id=i,wdl=wdl,nodes=nodes))
            else:
                m=g['action'];arm=g['arm'];cap=g['cap'];assert m in needs[i] and (arm,cap) in (('STANDARD_PW',128),('STANDARD_FULL',256)) and (m,arm) not in gval[i]
                z=game(g,r['history']+[m],arm,cap);c.update({'continuation_'+k:v for k,v in z.items()})
                assert g['utility']==(1+r['player']*g['winner_black'])/2;gval[i][m,arm]=F(g['utility']);c['continuation_games']+=1
    for i,z in gval.items():
        assert set(z)=={(m,a) for m in needs[i] for a in ('STANDARD_PW','STANDARD_FULL')}
        u[i]={m:(z[m,'STANDARD_PW']+z[m,'STANDARD_FULL'])/2 for m in needs[i]}
    assert set(u)==set(rr) and fit['models']=={} and not fit['missing']
    for dom in ('C4','GOMOKU'):
        ids=[i for i,r in rr.items() if r['group']=='TRAIN' and r['domain']==dom]
        d=[]
        for i in ids:
            a,b=aa[i,32,'K16N']['evidence']['pair'];d.append(u[i][b]-u[i][a])
        gate=dict(n=len(ids),nonzero=sum(x!=0 for x in d),positive=sum(x>0 for x in d),negative=sum(x<0 for x in d))
        gate['pass']=gate['nonzero']>=4 and gate['positive']>=1 and gate['negative']>=1
        assert not gate['pass'] and fit['gates'][dom]==gate
        for grp in ('TRAIN','CAL','TEST'):
            z=su['groups'][dom][grp];ids=[i for i,r in rr.items() if r['group']==grp and r['domain']==dom]
            for cap in CAPS:
                for arm in ARMS:
                    for kind in ('NATIVE','PREFIX_VOTE','PREFIX','PULSE','FINAL'):
                        v=[];loss=[];gains=[]
                        for i in ids:
                            a=aa[i,cap,arm];e=a['evidence'];pair0=e['pair']
                            move=a['best'] if kind=='NATIVE' else pair0[0] if kind=='PREFIX_VOTE' else e[kind.lower()+'_best']
                            v.append(u[i][move]);loss.append(max(u[i].values())-u[i][move]);gains.append(u[i][move]-u[i][aa[i,cap,'K16N']['best']])
                        q=z['by_cap'][str(cap)]['arms'][arm]['readouts'][kind]
                        assert math.isclose(q['mean_U'],float(S.mean(v))) and math.isclose(q['regret']['mean'],float(S.mean(loss))) and math.isclose(q['gain']['mean'],float(S.mean(gains)))
                        c['aggregate_checks']+=1
    # Actual standard baseline replay, full recommendation/visit-count/NN parity.
    parity=[]
    with Server() as s:
        for i,r in sorted(rr.items()):
            for cap in CAPS:
                a=aa[i,cap,'K16N'];e=a['result']
                if r['domain']=='GOMOKU':
                    b=s.search(r['history'],'STANDARD_PW',cap);assert b['result']==e
                    result=dict(best=b['best'],visits=b['visits'],NN=b['NN_calls'])
                else:
                    b=s.request(dict(cmd='research_connect4_cpu',history=r['history'],eval_mode='conv3x3',model_path=str(MODEL),model_identity=MID,cache_model=True,seed=17,force_stop_visits=cap))
                    assert 'error' not in b,b
                    t=b['steps'][-1];ca,n,w=pr.moments(e)
                    assert t['best_action']==a['best'] and t['evaluator_calls']==a['NN_calls'] and t['root_visits']==cap
                    assert dict(t['visit_policy'])=={m:v/cap for m,v in zip(ca,n,strict=True)}
                    result=dict(best=t['best_action'],visits=t['root_visits'],NN=t['evaluator_calls'])
                parity.append(dict(id=i,domain=r['domain'],cap=cap,**result));c['actual_baseline_parities']+=1
    write('C09_INDEPENDENT_WDL.json',dict(rows=orows,scope='independent row-major negamax WDL, not win-distance scores'))
    write('C09_BASELINE_PARITY.json',dict(rows=parity,scope='actual standard re-search'))
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident(),distinct_gomoku_histories=len(cache),**dict(c),scope='independent row-major exact WDL, old rule replay, observed root moments and flux/paired readout checks, actual baseline re-search. No full internal selection/backup replay.')
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0] and git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts]
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines()
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-kinetic16-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists()
        assert SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head
        assert gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for f in files:assert hashlib.sha256(gg(r,'show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(head=head,remote=rec['remote'],tree=rec['tree'],stage_count=len(receipts),cleanup_count=len(clean),restored_files=len(files),external_base=BASE,original_ssot='87c15dc828d6eea519c219fd8a412fa21dfc464d',bundle=bundle.name,bundle_sha256=sha(bundle),bytes=rec['bytes'],actual_restore=True,connectivity_fsck=True,identity=ident())
if __name__=='__main__':print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2,allow_nan=False),flush=True)
