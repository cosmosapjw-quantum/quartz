"""Independent grid-history, rational gain/cost, plan/scoring and actual bundle readback."""
from pathlib import Path
from fractions import Fraction as F
import collections as C,gzip,json,math,statistics as S,hashlib,subprocess as SP,tempfile
from publish import ROOT,D,B,BASE,BR,sha,git
import native as n
from population import exclusions
from experiment import ST,SEEDS,ARMS,snap
OLD=ROOT.parent/'quartz_nn_multiquanta_20261124/docs/research/quartz_nn_multiquanta_20261124'
BIN=B/'mcts_demo'
LINES=[tuple((y+j*dy)*7+x+j*dx for j in range(4)) for y in range(6) for x in range(7) for dx,dy in ((1,0),(0,1),(1,1),(-1,1)) if 0<=x+3*dx<7 and 0<=y+3*dy<6]
def read(f):return json.loads((D/f).read_text())
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8),(a,b)
def win(b,p):return any(all(b[k]==p for k in line) for line in LINES)
def replay(h):
    b=[0]*42;z=[0]*7
    for i,x in enumerate(h):
        c=int(x)-1;p=1-2*(i%2)
        assert 0<=c<7 and z[c]<6 and not win(b,p) and not win(b,-p)
        b[7*z[c]+c]=p;z[c]+=1
    return b,z
def immediate(b,z,p):
    return any(win(b[:7*z[c]+c]+[p]+b[7*z[c]+c+1:],p) for c in range(7) if z[c]<6)
def canon(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def cb(b):return min(tuple(b),tuple(b[7*y+x] for y in range(6) for x in range(6,-1,-1)))
def u(x):return F(1+(x>0)-(x<0),2)
def root_cost(r,j):return sum(s['wall_us'] for s in r['steps'][1:j+1])
def rk(s,seed):
    m=(1<<64)-1;z=seed^0xcbf29ce484222325
    for b in s.encode():z=((z^b)*0x100000001b3)&m
    z^=z>>30;z=z*0xbf58476d1ce4e5b9&m;z^=z>>27;z=z*0x94d049bb133111eb&m;return z^(z>>31)
def host_ok(a,b):
    dt=b['mono']-a['mono'];rate=sum(b['swap_pages'][k]-a['swap_pages'][k] for k in a['swap_pages'])/dt
    return rate<=512 and max(a['load'][0],b['load'][0])<=12,rate
def audit():
    assert sha(BIN)=='e49d6200008d3dcda7237560fe99696ad3d15fbd2977f88f77e906a8a98b374f'
    assert git('diff','--name-only','3d22dadcade29f8cb96b44e7ef40571c4d5c1db3','HEAD','--','src','Cargo.toml','Cargo.lock')==''
    co=read('C07_COHORT.json');g=read('GAIN_LOCK.json');m=read('C07_COST_MODEL.json');t=read('C07_TIMING_RAW.json');r=read('C08_LIVE_RAW.json');s=read('C08_SUMMARY.json')
    assert sha(D/'GAIN_LOCK.json')==m['gain_sha'];assert sha(D/'C07_COHORT.json')==m['cohort_sha'];assert sha(D/'C07_TIMING_RAW.json')==m['timing_sha'];assert sha(D/'C08_LIVE_RAW.json')==s['raw_sha'];assert sha(D/'C07_ATTEMPTS.jsonl.gz')==co['attempt_sha']
    # Reconstruct prior gain values from exact outcome traces without using any prior costs.
    for name,v in g['inputs'].items():
        f=OLD/name;assert sha(f)==v['sha256'] and git('hash-object',str(f))==v['blob']
    prior=json.loads((OLD/'C07_TRAIN_RAW.json').read_text());lab={a['id']:a['scores'] for a in json.loads((OLD/'C07_TRAIN_LABELS.json').read_text()) if a['status']=='EXACT'}
    obs=[[] for _ in range(3)];ex=set()
    for a in prior:
        for row in a['rows']:
            zz=lab[row['id']];us=[u(zz[x['best']]) for x in row['steps']]
            for j in (1,2,3):
                delta=us[j]-us[0];obs[j-1].append((row['cell'],delta))
                if delta:ex.add(row['id'])
    assert len(lab)==264 and len(ex)==64 and all(len(o)==528 for o in obs)
    for j,o in enumerate(obs):
        mean=sum(v for _,v in o)/len(o)
        for c in range(16):
            v=[x for a,x in o if a==c]
            close((sum(v)+8*mean)/(len(v)+8),g['gain'][c][j])
    # Exclusion readback and independent gravity grid replay of every new attempt.
    oldp,oldb,meta=exclusions();assert len(oldp)==co['old_families'] and len(oldb)==co['old_boards'] and meta==co['sources']
    ids={a['id']:a for a in co['rows']};assert len(ids)==len(co['rows'])
    seen=set();fam=set();boards=set();accepted=set();nevents=0;attempts=C.Counter();counts=C.Counter()
    with gzip.open(D/'C07_ATTEMPTS.jsonl.gz','rt') as fp:
        for line in fp:
            a=json.loads(line);i=a['id'];h=a['history'];p0=a['opening'];f=canon(p0)
            assert i not in seen and f not in oldp|fam and h.startswith(p0)
            seen.add(i);fam.add(f);attempts[a['stratum']+'/'+a['group']]+=1
            if a['stratum']=='SELFPLAY':
                hist=p0
                for ev in a['events']:
                    b,z=replay(hist);assert not win(b,1) and not win(b,-1)
                    assert ev['ply']==len(hist) and all(0<=c<7 and z[c]<6 for c in (ev['action'],ev['recommendation']))
                    assert ev['explore'] or ev['action']==ev['recommendation'];assert ev['eval_calls']>=1
                    hist+=str(ev['action']+1);nevents+=1
                assert hist==h
            else:
                b,z=replay(h);assert not win(b,1) and not win(b,-1)
            if a['candidate'] is None:continue
            x=ids[i];assert a['candidate']==h and a['status']=='ACCEPT' and i not in accepted;accepted.add(i)
            b,z=replay(h);p=1-2*(len(h)%2);assert not win(b,1) and not win(b,-1)
            assert x['board']==b and x['legal']==[c for c in range(7) if z[c]<6] and len(x['legal'])>=2 and x['immediate']==immediate(b,z,p)
            assert x['history']==h and x['family']==f and x['group']==a['group'] and x['stratum']==a['stratum'] and x['player']==p and x['ply']==len(h) and len(h)%2==a['parity']
            assert (18<=len(h)<=25 and not x['immediate']) if x['stratum']=='TACTICAL' else (16<=len(h)<=26)
            k=cb(b);assert k not in oldb|boards;boards.add(k);counts[x['stratum']+'/'+x['group']+'/'+str(p)]+=1
    assert accepted==set(ids) and len(seen)==co['attempts']
    # Native trace invariants, independently recomputed on all new runs.
    def check(a,split):
        assert len(a['rows'])==8 and a['threads']==2 and a['model_identity']==n.IDENT
        assert a['selected']==a['planned'][:len(a['selected'])]
        sels=dict(a['selected']);assert len(sels)==len(a['selected'])
        assert a['whole_us']+3>=a['prefix_us']+a['optional_us']+a['drop_us']
        assert a['setup_total_us']==sum(x['setup_us'] for x in a['rows'])
        assert a['prefix_step_us']==sum(x['steps'][0]['wall_us'] for x in a['rows'] if x['steps'])
        assert a['continuation_step_us']==sum(v['wall_us'] for x in a['rows'] for v in x['steps'][1:])
        for x in a['rows']:
            y=ids[x['id']];assert x['history']==y['history'] and x['legal']==y['legal'] and y['group']==split and y['stratum']==a['stratum']
            b,z=replay(y['history']);assert x['threat']==immediate(b,z,-y['player'])
            if a['mode']=='SETUP':assert not x['steps'];continue
            ss=x['steps'];assert [v['visits'] for v in ss]==[j for j in (4,8,16,32) if j<=sels.get(x['id'],4)]
            for v in ss:
                assert v['root_hash']==x['root_hash'] and v['best'] in y['legal'] and v['eval_calls']>=1 and v['wall_us']>=v['search_us']>=1
                assert {c for c,p in v['policy']}==set(y['legal']) and all(0<=p<=1 for c,p in v['policy'])
                close(sum(p for c,p in v['policy']),1)
            assert all(v['eval_calls']<=w['eval_calls'] for v,w in zip(ss,ss[1:]))
            ps=[p for c,p in ss[0]['policy']];en=-sum(p*math.log(p,7) for p in ps if p>0)
            close(en,x['entropy']);close(max(ps),x['top'])
            assert x['cell']==int(en>=.55)+2*int(max(ps)>=.75)+4*int(x['threat'])+8*int(len(y['legal'])<=4)
    for split,rr in t['raw'].items():
        for a in rr:check(a,'CHECK' if split=='PROFILE' else split)
    # Independent exact cost ranks/coverage; confirm all gain values unchanged.
    for st in ST:
        v=m['strata'][st]
        fit=[x for a in t['raw']['FIT'] if a['stratum']==st for x in a['rows']]
        d=[math.ceil(S.median(root_cost(x,j) for x in fit)) for j in (1,2,3)];assert d==v['point_us']
        cal=[a for a in t['raw']['CAL'] if a['stratum']==st];ck=[a for a in t['raw']['CHECK'] if a['stratum']==st]
        scores=lambda a:max(F(root_cost(x,j),d[j-1]) for x in a['rows'] for j in (1,2,3))
        sc=[scores(a) for a in cal];k=math.ceil(F(9*(len(cal)+1),10));a=max(F(1),sorted(sc)[k-1]) if k<=len(sc) else None
        assert len(cal)==19 and len(ck)==4 and k==v['rank'];close(a,v['multiplier'])
        for aa,bb in zip(sc,v['cal_scores']):close(aa,bb)
        cc=[scores(run) for run in ck]
        close(sum(x<=a for x in cc)/len(cc),v['check_pool_coverage'])
        flags=[F(root_cost(x,j),d[j-1])<=a for run in ck for x in run['rows'] for j in (1,2,3)]
        close(sum(flags)/len(flags),v['check_coordinate_coverage'])
        assert v['budget_us']==min(100000,max(1000,4*d[1],2*d[2]))
        for cell in range(16):
            for j in range(3):
                close(v['point'][cell][j][0],g['gain'][cell][j]);close(v['robust'][cell][j][0],g['gain'][cell][j])
                close(v['point'][cell][j][1],d[j]);close(v['robust'][cell][j][1],a*d[j])
        for name,table in [('point',v['point']),('robust',v['robust'])]:
            avail=[sorted({j for row in run['rows'] for j,(gain,cost) in zip((8,16,32),table[row['cell']]) if gain>0 and math.ceil(cost/50)<=v['budget_us']//50}) for run in ck]
            assert avail==v[name+'_availability']
            close(sum(len(x)>=2 for x in avail)/len(avail),v[name+'_availability_rate'])
        for mode,p in v['profiles'].items():
            a1=[a['whole_us'] for a in t['raw']['PROFILE'] if a['stratum']==st and a['mode']==mode];assert a1==p['whole_us']
            med=S.median(a1);close(med,p['median']);close(S.median(abs(x-med) for x in a1)/med,p['relative_mad'])
    # Fresh exact solver reflection spot checks; no timing experiments concurrent.
    labels=read('C08_LABELS.json');labs={x['id']:x['scores'] for x in labels if x['status']=='EXACT'}
    assert {x['id'] for x in labels}=={i for i,y in ids.items() if y['group']=='TEST'}
    for i,zz in labs.items():assert len(zz)==7 and all((q!=-1000)==(j in ids[i]['legal']) for j,q in enumerate(zz))
    mirrors=0
    for st in ST:
        for x in sorted((y for y in ids.values() if y['stratum']==st and y['group']=='TEST'),key=lambda y:y['id'])[:3]:
            h=x['history'].translate(str.maketrans('1234567','7654321'))
            zz=SP.run([str(n.EXT/'c4solver'),'-a','-b',str(n.EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=n.EXT,timeout=15,check=True).stdout.split()
            assert zz[0]==h and list(map(int,zz[1:]))==labs[x['id']][::-1];mirrors+=1
    by={};plan_checks=C.Counter();forecast={};blocks={};rob_cov=C.Counter()
    for a in r['runs']:
        check(a,'TEST');st=a['stratum'];arm=a['arm'];rows=a['rows'];v=m['strata'][st]
        assert a['budget_us']==v['budget_us'];assert a['overrun_us']==(0 if arm=='PREFIX' else max(0,a['optional_us']-a['budget_us']))
        blocks.setdefault((st,a['seed'],a['pool']),{})[arm]=a
        if arm in ('DP','RATIO','ROBUST_DP'):
            table=v['robust'] if arm=='ROBUST_DP' else v['point'];rr=sorted(rows,key=lambda x:x['id'])
            opt=[[(0,0.)]+[(math.ceil(c/50),g0) for g0,c in table[x['cell']]] for x in rr];K=a['budget_us']//50
            p=dict(a['planned']);js=[(4,8,16,32).index(p.get(x['id'],4)) for x in rr]
            c=sum(o[j][0] for o,j in zip(opt,js));gg=sum(o[j][1] for o,j in zip(opt,js))
            assert c==a['predicted_ticks']<=K;close(gg,a['predicted_gain'])
            if arm=='RATIO':
                items=sorted((-g0/c0,i,j,c0) for i,o in enumerate(opt) for j,(c0,g0) in enumerate(o) if j and g0>0)
                want=[0]*len(rr);left=K
                for _,i,j,c0 in items:
                    if want[i]==0 and c0<=left:want[i]=j;left-=c0
                assert want==js
            else:
                # Independent sparse cost frontier, not native dense DP/backpointer.
                f={0:0.}
                for o in opt:
                    q={}
                    for c0,v0 in f.items():
                        for c1,v1 in o:
                            if c0+c1<=K:q[c0+c1]=max(q.get(c0+c1,float('-inf')),v0+v1)
                    f=q
                close(max(f.values()),gg)
            plan_checks[arm]+=1
        elif arm=='PREFIX':assert not a['planned']
        else:
            order=sorted(rows,key=lambda x:x['id']) if arm=='FIXED8' else sorted(rows,key=lambda x:(-x['entropy'],x['id'])) if arm=='ENTROPY16' else sorted(rows,key=lambda x:(rk(x['id'],a['seed']),x['id']))
            assert a['planned']==[[x['id'],8 if arm=='FIXED8' else 16] for x in order]
        dest=by.setdefault((st,arm,a['seed']),{})
        for row in rows:
            i=row['id'];ss=row['steps'];z=labs[i];u0=u(z[ss[0]['best']]);u1=u(z[ss[-1]['best']]);opt=max(u(z[c]) for c in row['legal'])
            assert i not in dest;dest[i]=(opt-u1,u1-u0)
            if arm in ('DP','RATIO','ROBUST_DP') and len(ss)>1:
                j=len(ss)-2;f=forecast.setdefault(st+'/'+arm,dict(n=0,predicted_total=0.,realized_total=0.,positive=0,negative=0,unchanged=0))
                f['n']+=1;f['predicted_total']+=g['gain'][row['cell']][j];f['realized_total']+=float(u1-u0)
                f['positive']+=u1>u0;f['negative']+=u1<u0;f['unchanged']+=u1==u0
                if arm=='ROBUST_DP':
                    for j in range(1,len(ss)):
                        rob_cov[st+'/n']+=1;rob_cov[st+'/covered']+=root_cost(row,j)<=v['robust'][row['cell']][j-1][1]
    assert len(blocks)==2*6*8 and all(set(v)==set(ARMS) for v in blocks.values())
    # When actual prefix feature/decision records are identical, compare predicted objectives on the same input.
    paired_plan={}
    for st in ST:
        p=dict(blocks=0,same_prefix=0,dp_predicted_ge_ratio=0,dp_predicted_strictly_gt_ratio=0,predicted_difference_sum=0.)
        for (ss,seed,k),a in blocks.items():
            if ss!=st:continue
            p['blocks']+=1
            signature=lambda run:sorted((x['id'],x['cell'],x['steps'][0]['best'],tuple(sorted(tuple(q) for q in x['steps'][0]['policy']))) for x in run['rows'])
            if signature(a['DP'])==signature(a['RATIO']):
                p['same_prefix']+=1
                z=a['DP']['predicted_gain']-a['RATIO']['predicted_gain'];assert z>=-1e-9
                p['dp_predicted_ge_ratio']+=1;p['dp_predicted_strictly_gt_ratio']+=z>1e-9;p['predicted_difference_sum']+=z
        paired_plan[st]=p
    # Score readback from raw exact WDL with rational accumulation.
    for st,x in s['groups'].items():
        root_ids={i for i,y in ids.items() if y['group']=='TEST' and y['stratum']==st}
        for arm,v in x['arms'].items():
            vals=[]
            for seed in SEEDS:
                d=by[st,arm,seed];assert set(d)==root_ids;vals.extend(d.values())
            close(sum(v[0] for v in vals)/len(vals),x['arms'][arm]['regret']);close(sum(v[1] for v in vals)/len(vals),x['arms'][arm]['gain'])
            rr=[a for a in r['runs'] if a['arm']==arm and a['stratum']==st]
            for field in ('whole_us','transport_us','optional_us','continuation_step_us','plan_us','drop_us','setup_total_us','overrun_us'):
                close(sum(a[field] for a in rr)/6,x['arms'][arm][field])
            assert dict(C.Counter(str(y) for a in rr for i,y in a['selected']))==x['arms'][arm]['selected']
        for name,v in x['paired'].items():
            a,b=name.split('/');d=[sum(by[st,b,seed][i][0]-by[st,a,seed][i][0] for seed in SEEDS)/6 for i in root_ids]
            close(sum(d)/len(d),v['improvement']);close(abs(x['arms'][a]['whole_us']/x['arms'][b]['whole_us']-1),v['time_mismatch'])
        rr=x['arms']['ROBUST_DP'];assert rob_cov[st+'/n']==rr['selected_coordinates'];close(rob_cov[st+'/covered']/rob_cov[st+'/n'],rr['selected_coordinate_coverage'])
    ok1,rate1=host_ok(t['start'],t['end']);ok2,rate2=host_ok(r['start'],r['end'])
    assert (ok1 and max(a['host']['load'][0] for rr in t['raw'].values() for a in rr)<=12)==m['host_ok']
    assert (ok2 and max(a['host']['load'][0] for a in r['runs'])<=12)==s['host_ok']
    return dict(verified_attempts=len(seen),verified_roots=len(ids),attempt_groups=dict(attempts),cohort=dict(counts),selfplay_events=nevents,prior_gain_recomputed=True,prior_exposure=len(ex),median_rank_coverage_recomputed=True,exact_labels=len(labs),mirror_vectors=mirrors,plan_checks=dict(plan_checks),paired_same_prefix_plan=paired_plan,actual_selected_gain_diagnostic=forecast,robust_live_selected_coverage=dict(rob_cov),live_requests=len(r['runs']),live_root_traces=sum(len(a['rows']) for a in r['runs']),score_recomputed=True,cal_swap_rate=rate1,live_swap_rate=rate2,scientific_promotion=s['promotion'])
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in ('src/mcts_server.rs','src/mcts_server/multiquanta_cpu_server.rs','src/mcts_server/multiquanta_plan.rs') for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-nn04-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head and gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for p in files:assert hashlib.sha256(gg(r,'show',f'{head}:{p}')).hexdigest()==sha(ROOT/p),p
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(BIN)=='e49d6200008d3dcda7237560fe99696ad3d15fbd2977f88f77e906a8a98b374f'
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(clean),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,actual_restore=True,connectivity_fsck=True,binary_sha256=sha(BIN))
if __name__=='__main__':
    import sys
    print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2),flush=True)
