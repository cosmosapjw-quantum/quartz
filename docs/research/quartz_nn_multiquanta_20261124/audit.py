"""Independent readback of actual NN-DECISION-03 source, fit, live plans, metrics and backup."""
import collections as C, gzip, hashlib, itertools, json, math, statistics as S, subprocess as SP, tempfile
from fractions import Fraction as F
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,sha,git
from campaign import lock,EXT,BIN,IDENT,SEEDS,ARMS,exclusions,host
def read(n):return json.loads((D/n).read_text())
def close(x,y,tol=1e-8):assert math.isclose(float(x),float(y),rel_tol=1e-9,abs_tol=tol),(x,y)
def four(b,p):
    for y in range(6):
        for x in range(7):
            for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
                if 0<=x+3*dx<7 and 0<=y+3*dy<6 and all(b[(y+i*dy)*7+x+i*dx]==p for i in range(4)):return True
    return False
def replay(h):
    b=[0]*42;z=[0]*7
    for i,a in enumerate(h):
        c=int(a)-1;p=1-2*(i%2);assert 0<=c<7 and z[c]<6 and not four(b,-p) and not four(b,p)
        b[7*z[c]+c]=p;z[c]+=1
    return b,z
def immediate(b,z,p):
    return any(four(b[:7*z[c]+c]+[p]+b[7*z[c]+c+1:],p) for c in range(7) if z[c]<6)
def canon(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def cb(b):return min(tuple(b),tuple(b[7*y+x] for y in range(6) for x in range(6,-1,-1)))
def u(s):return F(1+(s>0)-(s<0),2)
def kkey(s,seed):
    m=(1<<64)-1;z=seed^0xcbf29ce484222325
    for b in s.encode():z=((z^b)*0x100000001b3)&m
    z^=z>>30;z=z*0xbf58476d1ce4e5b9&m;z^=z>>27;z=z*0x94d049bb133111eb&m;return z^(z>>31)
def audit():
    lock();co=read('C07_COHORT.json');m=read('C07_MODEL.json');su=read('C08_SUMMARY.json');rr=read('C08_LIVE_RAW.json')
    for f,k in [('C07_COHORT.json','source_sha'),('C07_TRAIN_RAW.json','raw_sha'),('C07_TRAIN_LABELS.json','labels_sha'),('C07_CAL_RAW.json','cal_sha'),('C07_PROFILE.json','profile_sha')]:assert sha(D/f)==m[k]
    assert sha(D/'C07_ATTEMPTS.jsonl.gz')==co['attempt_sha'] and sha(D/'C08_LIVE_RAW.json')==su['raw_sha']
    oldp,oldb,meta=exclusions();assert len(oldp)==co['old_families'] and len(oldb)==co['old_boards'] and meta==co['sources']
    ids={r['id']:r for r in co['rows']};seen=set();families=set();boards=set();nevents=0;accepted=set();attempts=C.Counter();counts=C.Counter()
    with gzip.open(D/'C07_ATTEMPTS.jsonl.gz','rt') as fp:
        for line in fp:
            a=json.loads(line);i=a['id'];h=a['history'];opening=a['opening'];fam=canon(opening)
            assert i not in seen and fam not in oldp|families and h.startswith(opening);seen.add(i);families.add(fam);attempts[a['stratum']+'/'+a['group']]+=1
            if a['stratum']=='SELFPLAY':
                q=opening
                for ev in a['events']:
                    b,z=replay(q);assert ev['ply']==len(q) and all(0<=c<7 and z[c]<6 for c in (ev['action'],ev['recommendation']))
                    assert ev['explore'] or ev['action']==ev['recommendation'];assert ev['eval_calls']>=1
                    q+=str(ev['action']+1);nevents+=1
                assert q==h
            else:
                b,z=replay(h);assert not four(b,1) and not four(b,-1)
            if a['candidate'] is None:continue
            assert a['candidate']==h and a['status']=='ACCEPT';r=ids[i];assert i not in accepted;accepted.add(i)
            b,z=replay(h);p=1-2*(len(h)%2)
            assert not four(b,p) and not four(b,-p)
            assert r['board']==b and r['legal']==[c for c in range(7) if z[c]<6] and len(r['legal'])>=2
            assert r['family']==fam and r['player']==p and r['ply']==len(h) and r['group']==a['group'] and r['stratum']==a['stratum']
            assert r['immediate']==immediate(b,z,p)
            assert (18<=len(h)<=25 and not r['immediate']) if r['stratum']=='TACTICAL' else (16<=len(h)<=26)
            k=cb(b);assert k not in boards|oldb;boards.add(k)
            counts[r['stratum']+'/'+r['group']+'/'+str(p)]+=1
    assert accepted==set(ids) and len(seen)==co['attempts']
    labels={};label_status=C.Counter()
    for n,group in [('C07_TRAIN_LABELS.json','TRAIN'),('C08_LABELS.json','TEST')]:
        labs=read(n);assert {r['id'] for r in labs}=={i for i,r in ids.items() if r['group']==group}
        for r in labs:
            label_status[group+'/'+r['status']]+=1
            if r['status']!='EXACT':continue
            z=r['scores'];assert len(z)==7 and all((x!=-1000)==(a in ids[r['id']]['legal']) for a,x in enumerate(z))
            labels[r['id']]=z
    # Exact solver mirror spot checks, independent query path.
    mirrors=0
    for st in ('TACTICAL','SELFPLAY'):
        for group in ('TRAIN','TEST'):
            a=next(r for r in co['rows'] if r['stratum']==st and r['group']==group and r['id'] in labels)
            h=a['history'].translate(str.maketrans('1234567','7654321'))
            z=SP.run([str(EXT/'c4solver'),'-a','-b',str(EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=EXT,timeout=15,check=True).stdout.split()
            assert z[0]==h and list(map(int,z[1:]))==labels[a['id']][::-1];mirrors+=1
    obs=[[],[],[]];hist=[C.Counter() for _ in range(3)];exposed=[set() for _ in range(3)]
    def check(a):
        assert a['model_identity']==IDENT and a['threads']==2 and len(a['rows'])<=8
        assert a['selected']==a['planned'][:len(a['selected'])]
        sels=dict(a['selected']);assert len(sels)==len(a['selected'])
        assert a['whole_us']+3>=a['prefix_us']+a['optional_us']+a['drop_us']
        assert a['setup_total_us']==sum(r['setup_us'] for r in a['rows'])
        for r in a['rows']:
            s=ids[r['id']];assert r['history']==s['history'] and r['legal']==s['legal']
            b,z=replay(s['history']);assert r['threat']==immediate(b,z,-s['player'])
            if a['mode']=='SETUP':assert not r['steps'];continue
            ss=r['steps'];assert [x['visits'] for x in ss]==[v for v in (4,8,16,32) if v<=sels.get(r['id'],4)]
            assert all(x['root_hash']==r['root_hash'] and x['best'] in s['legal'] for x in ss)
            assert all(x['eval_calls']>=1 and x['wall_us']>=x['search_us']>=1 for x in ss)
            assert all(x['eval_calls']<=y['eval_calls'] for x,y in zip(ss,ss[1:]))
            p=[v for _,v in ss[0]['policy']];e=-sum(v*math.log(v,7) for v in p if v);close(e,r['entropy']);close(max(p),r['top'])
            assert r['cell']==int(e>=.55)+2*int(max(p)>=.75)+4*int(r['threat'])+8*int(len(s['legal'])<=4)
        assert a['prefix_step_us']==sum(r['steps'][0]['wall_us'] for r in a['rows'] if r['steps'])
        assert a['continuation_step_us']==sum(x['wall_us'] for r in a['rows'] for x in r['steps'][1:])
    train=read('C07_TRAIN_RAW.json');cal=read('C07_CAL_RAW.json');profiles=read('C07_PROFILE.json')
    for a in train+cal+profiles:check(a)
    assert all(ids[r['id']]['group']=='TRAIN' for a in train for r in a['rows'])
    assert all(ids[r['id']]['group']=='CAL' for a in cal+profiles for r in a['rows'])
    for a in train:
        for r in a['rows']:
            i=r['id']
            if i not in labels:continue
            z=labels[i];s=r['steps'];assert len(s)==4;v=[u(z[x['best']]) for x in s]
            for j in range(1,4):
                g=int(2*(v[j]-v[0]));t=sum(x['wall_us'] for x in s[1:j+1]);obs[j-1].append((r['cell'],g,t));hist[j-1][g]+=1
                if g:exposed[j-1].add(i)
    assert [len(x) for x in exposed]==m['exposed_by'] and len(set.union(*exposed))==m['exposed']
    for j,o in enumerate(obs):
        assert {str(k):v for k,v in hist[j].items()}==m['histograms'][j]
        gp=[F(hist[j][k],len(o)) for k in range(-2,3)];mean=F(sum(t for c,g,t in o),len(o))
        for c in range(16):
            sub=[(g,t) for a,g,t in o if a==c];n=len(sub);probs=[(sum(g==k for g,t in sub)+8*gp[k+2])/(n+8) for k in range(-2,3)]
            for a,b in zip(probs,m['fitted'][j]['cells'][c]):close(a,b)
            close(sum(F(k,2)*probs[k+2] for k in range(-2,3)),m['cells'][c][j][0]);close((sum(t for g,t in sub)+8*mean)/(n+8),m['cells'][c][j][1])
        cost=[sum(s['wall_us'] for s in r['steps'][1:j+2]) for a in cal for r in a['rows']]
        close(S.mean(cost),m['cal_cost_mean'][j]);assert max(1,math.ceil(S.median(cost)))==m['floors'][j]
    by={};dpchecks=0;ratchecks=0;plan_targets=C.Counter()
    for a in rr['runs']:
        check(a);mode=a['mode'];rows=a['rows'];st=a['stratum'];assert all(ids[r['id']]['group']=='TEST' and ids[r['id']]['stratum']==st for r in rows)
        assert a['budget_us']==min(100000,max(500,round(len(rows)*.5*m['cal_cost_mean'][1])))
        assert a['overrun_us']==(0 if mode=='PREFIX' else max(0,a['optional_us']-a['budget_us']))
        if mode in ('DP','RATIO'):
            rs=sorted(rows,key=lambda r:r['id']);options=[[(0,0.)]+[(math.ceil(c/50),g) for g,c in m['cells'][r['cell']]] for r in rs];K=a['budget_us']//50
            planned=dict(a['planned']);choices=[(4,8,16,32).index(planned.get(r['id'],4)) for r in rs];cs=sum(o[j][0] for o,j in zip(options,choices));g=sum(o[j][1] for o,j in zip(options,choices))
            assert cs<=K and cs==a['predicted_ticks'];close(g,a['predicted_gain'])
            if mode=='DP':
                # Independent sparse exact cost frontier, not the native dense/backpointer algorithm.
                frontier={0:0.}
                for o in options:
                    nxt={}
                    for c,v in frontier.items():
                        for d,w in o:
                            if c+d<=K:nxt[c+d]=max(nxt.get(c+d,float('-inf')),v+w)
                    frontier=nxt
                close(max(frontier.values()),g);dpchecks+=1
            else:
                items=sorted((-g/c,i,j,c) for i,o in enumerate(options) for j,(c,g) in enumerate(o) if j and g>0)
                ans=[0]*len(rs);left=K
                for _,i,j,c in items:
                    if ans[i]==0 and c<=left:ans[i]=j;left-=c
                assert ans==choices;ratchecks+=1
            plan_targets.update((mode,v) for _,v in a['selected'])
        elif mode=='PREFIX':assert not a['planned']
        else:
            if mode=='FIXED8':rs=sorted(rows,key=lambda r:r['id']);v=8
            elif mode=='ENTROPY16':rs=sorted(rows,key=lambda r:(-r['entropy'],r['id']));v=16
            else:rs=sorted(rows,key=lambda r:(kkey(r['id'],a['seed']),r['id']));v=16
            assert a['planned']==[[r['id'],v] for r in rs]
        d=by.setdefault((st,mode,a['seed']),{})
        for r in rows:
            i=r['id']
            if i not in labels:continue
            z=labels[i];opt=max(u(z[c]) for c in r['legal']);v0=u(z[r['steps'][0]['best']]);v1=u(z[r['steps'][-1]['best']]);assert i not in d
            d[i]=(opt-v1,v1-v0)
    for st,gr in su['groups'].items():
        test={i for i,r in ids.items() if r['group']=='TEST' and r['stratum']==st and i in labels}
        for arm,v in gr['arms'].items():
            for seed in SEEDS:assert set(by[st,arm,seed])==test
            x=[t for seed in SEEDS for t in by[st,arm,seed].values()]
            close(sum(t[0] for t in x)/len(x),v['regret']);close(sum(t[1] for t in x)/len(x),v['gain'])
            runs=[a for a in rr['runs'] if a['stratum']==st and a['mode']==arm]
            for field in ('whole_us','setup_total_us','drop_us','optional_us','plan_us','continuation_step_us','transport_us'):
                close(F(sum(a[field] for a in runs),4) if field!='transport_us' else sum(a[field] for a in runs)/4,v[field])
        for arm,v in gr['paired'].items():
            d=[sum(by[st,arm,s][i][0]-by[st,'DP',s][i][0] for s in SEEDS)/4 for i in test]
            close(sum(d)/len(d),v['improvement']);assert len(d)==v['n']
            close(abs(gr['arms']['DP']['whole_us']/gr['arms'][arm]['whole_us']-1),v['time_mismatch'])
    return dict(attempts=dict(attempts),cohort=dict(counts),selfplay_events=nevents,independent_grid_replay=True,labels=dict(label_status),mirror_exact_vectors=mirrors,exposed=m['exposed'],exposed_by=m['exposed_by'],fit_recomputed=True,dp_plan_checks=dpchecks,ratio_plan_checks=ratchecks,live_requests=len(rr['runs']),live_root_traces=sum(len(a['rows']) for a in rr['runs']),selected_targets={str(k):v for k,v in plan_targets.items()},summary_recomputed=True)
def restore():
    rec=read_latest=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[r['head'] for r in receipts] and all(r['head']==r['remote'] for r in receipts)
    cleanup=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(cleanup)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in ('src/mcts_server.rs','src/mcts_server/multiquanta_cpu_server.rs','src/mcts_server/multiquanta_plan.rs') for p in files)
    def g(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-nn03-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        g(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        g(r,'bundle','verify',str(bundle));g(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert g(r,'rev-parse','refs/heads/restored').decode().strip()==head and g(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert g(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for p in files:assert hashlib.sha256(g(r,'show',f'{head}:{p}')).hexdigest()==sha(ROOT/p),p
        g(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(cleanup),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,actual_restore=True,connectivity_fsck=True,binary_sha256=sha(BIN))
if __name__=='__main__':
    import sys
    print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2),flush=True)
