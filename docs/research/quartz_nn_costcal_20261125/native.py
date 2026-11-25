"""Prospective multi-option experiment. Reused native generation and external exact labels."""
import collections as C,concurrent.futures,gzip,hashlib,json,math,os,random,statistics as S,subprocess,sys,time
from pathlib import Path
from prior_campaign import D,ROOT,HIST,EXT,MODEL,IDENT,Server as PriorServer,lock as prior_lock,oracle,old_sources
from hurdle import shrink
BIN=ROOT.parent/'quartz_nn_multiquanta_target_20261124/mcts_demo'
PREV=ROOT.parent/'quartz_nn_intervention_20261124/docs/research/quartz_nn_intervention_20261124'
TARGETS={'TACTICAL':{'TRAIN':240,'CAL':16,'TEST':32},'SELFPLAY':{'TRAIN':24,'CAL':8,'TEST':16}}
CAPS={'TRAIN':300,'CAL':100,'TEST':200}
SEEDS=(17,43,71,101)
ARMS=('PREFIX','FIXED8','RANDOM16','ENTROPY16','RATIO','DP')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(n,x):
    p=D/n;assert not p.exists(),p;p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def read(n):return json.loads((D/n).read_text())
def lock():
    prior_lock();a=read('C06_BUILD_IDENTITY.json')
    assert a['binary_sha256']==sha(BIN) and sha(MODEL)==IDENT
    assert sha(D/'prior_campaign.py')=='e044e47f028d589a281d8ee2e2b15310c81cdb39aad8dbad3f9e4c05b1d52c1c'
def host():
    v={k:int(x) for k,x in (l.split() for l in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}
    return dict(load=list(os.getloadavg()),cpus=os.cpu_count(),swap_pages=v)
def won(x):return any((m:=x&(x>>d))&(m>>(2*d)) for d in (1,7,6,8))
def replay(h):
    b=[0,0];z=[0]*7
    for i,ch in enumerate(h):
        c=int(ch)-1;assert 0<=c<7 and z[c]<6 and not any(won(x) for x in b)
        b[i%2]|=1<<(7*c+z[c]);z[c]+=1
    return b,z
def flat(b):
    return [1 if b[0]>>(7*x+y)&1 else -1 if b[1]>>(7*x+y)&1 else 0 for y in range(6) for x in range(7)]
def canwin(b,z,p):return any(won(b[p]|(1<<(7*c+z[c]))) for c in range(7) if z[c]<6)
def canon(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def cb(b):return min(tuple(b),tuple(b[y*7+x] for y in range(6) for x in range(6,-1,-1)))
def chunks(a,k=8):return [a[i:i+k] for i in range(0,len(a),k)]
class Server:
    def __init__(self):
        lock()
        self.p=subprocess.Popen(['nice','-n','10',str(BIN),'--server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS='2'))
    def call(self,mode,roots,seed=17,budget=0,floors=None,cells=None):
        q=dict(cmd='research_connect4_multiquanta_cpu',mode=mode,seed=seed,model_path=str(MODEL),model_identity=IDENT,jobs=[dict(id=r['id'],history=r['history']) for r in roots],budget_us=budget)
        if floors is not None:q['floors']=floors
        if cells is not None:q['cells']=cells
        t=time.perf_counter_ns();self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush();line=self.p.stdout.readline()
        assert line,('EOF',self.p.poll());a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000
        assert 'error' not in a,a
        assert a['threads']==2 and a['model_identity']==IDENT and len(a['rows'])==len(roots)
        assert a['whole_us']+3>=a['prefix_us']+a['optional_us']+a['drop_us']
        assert a['setup_total_us']==sum(r['setup_us'] for r in a['rows'])
        assert a['prefix_step_us']==sum(r['steps'][0]['wall_us'] for r in a['rows'] if r['steps'])
        assert a['continuation_step_us']==sum(s['wall_us'] for r in a['rows'] for s in r['steps'][1:])
        assert a['selected']==a['planned'][:len(a['selected'])]
        sels=dict(a['selected'])
        for r,src in zip(a['rows'],roots):
            assert r['id']==src['id'] and r['history']==src['history']
            b,z=replay(src['history']);leg=[i for i in range(7) if z[i]<6];p=len(src['history'])%2
            assert not any(won(x) for x in b) and r['legal']==leg and r['threat']==canwin(b,z,1-p)
            ss=r['steps']
            if mode=='SETUP':assert not ss;continue
            assert [s['visits'] for s in ss]==[j for j in (4,8,16,32) if j<=sels.get(r['id'],4)]
            for s in ss:
                assert s['root_hash']==r['root_hash'] and s['best'] in leg and s['wall_us']>=s['search_us']>=1 and s['eval_calls']>=1
                assert {c for c,_ in s['policy']}==set(leg) and abs(sum(v for _,v in s['policy'])-1)<1e-5
            assert all(x['eval_calls']<=y['eval_calls'] for x,y in zip(ss,ss[1:]))
            prob=[v for _,v in ss[0]['policy']];ent=-sum(v*math.log(v,7) for v in prob if v>0)
            assert abs(ent-r['entropy'])<1e-6 and abs(max(prob)-r['top'])<1e-6
            assert r['cell']==int(r['entropy']>=.55)+2*int(r['top']>=.75)+4*int(r['threat'])+8*int(len(leg)<=4)
        return a
    def close(self):
        self.p.stdin.close();assert self.p.wait(timeout=30)==0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke():
    jobs=[dict(id='a',history='324252'),dict(id='b',history='4433226223')]
    with Server() as s:
        a=s.call('TRACE',jobs);b=s.call('SETUP',jobs);c=s.call('DP',jobs,budget=100000,floors=[1,1,1],cells=[[[.05,100.],[.1,150.],[.2,200.]]]*16)
        d=s.call('RATIO',jobs,budget=100000,floors=[1,1,1],cells=[[[.05,100.],[.1,150.],[.2,200.]]]*16)
        assert a['rows'][0]['threat'] and all(len(r['steps'])==4 for r in a['rows'])
        assert c['predicted_ticks']==8 and c['predicted_gain']==.4 and all(v==32 for _,v in c['selected']) and len(c['selected'])==2
        assert len(d['selected'])==2
        neg=s.call('DP',jobs,budget=100000,floors=[1,1,1],cells=[[[-.05,100.],[-.1,150.],[-.2,200.]]]*16)
        assert not neg['selected'] and not neg['planned']
        q=dict(cmd='research_connect4_multiquanta_cpu',mode='TRACE',seed=17,model_path=str(MODEL),model_identity=IDENT,jobs=jobs,oracle=[1])
        s.p.stdin.write(json.dumps(q)+'\n');s.p.stdin.flush();bad=json.loads(s.p.stdout.readline());assert 'error'in bad
        write('C06_NATIVE_SMOKE.json',dict(trace=a,setup=b,dp=c,ratio=d,negative=neg,reject=bad))
    print('Native smoke: actual resident CNN 4/8/16/32, setup, DP/ratio, negative STOP, no oracle',flush=True)
def exclusions():
    oldp,oldb,meta=old_sources()
    # Check prior actual source identity by recorded exact commit.
    for n in ('C07_GENERATION.jsonl.gz','C07_RESUME_GENERATION.jsonl.gz','C07_COHORT.json'):
        p=PREV/n
        obj=subprocess.check_output(['git','hash-object',str(p)],cwd=ROOT,text=True).strip()
        ref=subprocess.check_output(['git','rev-parse','543e12dd3a759ebef1e7163a80a1f57c45c653d5:docs/research/quartz_nn_intervention_20261124/'+n],cwd=ROOT,text=True).strip()
        assert obj==ref
        meta['NN02/'+n]={'sha256':sha(p)}
        if n.endswith('.gz'):
            with gzip.open(p,'rt') as f:
                for line in f:oldp.add(canon(json.loads(line)['opening']))
        else:
            for r in json.loads(p.read_text())['rows']:oldb.add(cb(r['board']))
    return oldp,oldb,meta
def make_row(g,h):
    b,z=replay(h);assert not any(won(x) for x in b)
    return dict(id=g['id'],family=canon(g['opening']),group=g['group'],stratum=g['stratum'],history=h,board=flat(b),ply=len(h),player=1-2*(len(h)%2),legal=[c for c in range(7) if z[c]<6],immediate=canwin(b,z,len(h)%2))
def generate():
    prior_lock();before=host();oldp,oldb,meta=exclusions();usedp=set(oldp);usedb=set(oldb);rows=[]
    rng=random.Random(202611240303);serial=0
    path=D/'C07_ATTEMPTS.jsonl.gz';assert not path.exists()
    def new(group,stratum,parity):
        nonlocal serial
        while True:
            h=''.join(str(rng.randrange(1,8)) for _ in range(5));c=canon(h)
            if c not in usedp:usedp.add(c);break
        serial+=1
        return dict(id=f'{stratum}_{group}_{serial:05d}',opening=h,history=h,group=group,stratum=stratum,parity=parity,events=[],candidate=None)
    def accept(g,h,quota,count,immediate_allowed):
        r=make_row(g,h);p=r['ply']%2;b=cb(r['board'])
        if p!=g['parity'] or len(r['legal'])<2 or count[p]>=quota//2 or b in usedb or (not immediate_allowed and r['immediate']):return False
        usedb.add(b);count[p]+=1;rows.append(r);g['candidate']=h;return True
    with gzip.open(path,'wt',compresslevel=6) as log:
        for group,quota in TARGETS['TACTICAL'].items():
            count=[0,0];n=0
            while sum(count)<quota and n<4000:
                n+=1;p=0 if count[0]<count[1] else 1
                if count[p]>=quota//2:p=1-p
                g=new(group,'TACTICAL',p);h=g['history'];b,z=replay(h)
                target=rng.choice([x for x in range(18,26) if x%2==p])
                for i in range(5,target):
                    opts=[c for c in range(7) if z[c]<6 and not won(b[i%2]|(1<<(7*c+z[c])))]
                    if not opts:break
                    c=rng.choice(opts);b[i%2]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
                g['history']=h;g['target']=target
                if len(h)==target:accept(g,h,quota,count,False)
                g['status']='ACCEPT' if g['candidate'] else 'REJECT'
                log.write(json.dumps(g,separators=(',',':'))+'\n')
            print('TACTICAL',group,count,'attempts',n,flush=True)
        # Real previous trained CNN and parallel resident engine; only source generation.
        with PriorServer() as srv:
            for group,quota in TARGETS['SELFPLAY'].items():
                count=[0,0];n=0
                while sum(count)<quota and n<CAPS[group]:
                    batch=[new(group,'SELFPLAY',(n+i)%2) for i in range(min(8,CAPS[group]-n))]
                    n+=len(batch)
                    for ply in range(5,27):
                        active=[]
                        for g in batch:
                            if g.get('status'):continue
                            h=g['history'];assert len(h)==ply;b,z=replay(h)
                            if any(won(x) for x in b) or all(t==6 for t in z):g['status']='TERMINAL';continue
                            if ply>=16 and ply%2==g['parity'] and accept(g,h,quota,count,True):g['status']='ACCEPT';continue
                            if ply==26:g['status']='CAP';continue
                            active.append(g)
                        if not active:break
                        a=srv.call('GEN',active,seed=17)
                        for g,r in zip(active,a['rows']):
                            rec=r['steps'][0]['best'];explore=rng.random()<.45;move=rng.choice(r['legal']) if explore else rec
                            g['events'].append(dict(ply=ply,action=move,recommendation=rec,explore=explore,eval_calls=r['steps'][0]['eval_calls'],root_hash=r['root_hash']))
                            g['history']+=str(move+1)
                    for g in batch:log.write(json.dumps(g,separators=(',',':'))+'\n')
                print('SELFPLAY',group,count,'attempts',n,flush=True)
    write('C07_COHORT.json',dict(rows=rows,targets=TARGETS,caps=CAPS,attempts=serial,old_families=len(oldp),old_boards=len(oldb),sources=meta,attempt_sha=sha(path),host_start=before,host_end=host()))
    print('COHORT',dict(C.Counter(f"{r['stratum']}/{r['group']}/{r['player']}" for r in rows)),flush=True)
def utility(s):return (1+(s>0)-(s<0))/2
def trace_pools(s,roots,seed,mode='TRACE'):
    result=[]
    for pool in chunks(roots):
        a=s.call(mode,pool,seed);a['stratum']=pool[0]['stratum'];result.append(a)
    return result
def train():
    co=read('C07_COHORT.json');start=host();roots=[r for r in co['rows'] if r['group']=='TRAIN'];cal=[r for r in co['rows'] if r['group']=='CAL']
    assert roots and cal
    raw=[];calraw=[];profile=[]
    with Server() as s:
        s.call('SETUP',[roots[0]])
        for seed in (17,43):
            for st in TARGETS:
                raw.extend(trace_pools(s,[r for r in roots if r['stratum']==st],seed))
                calraw.extend(trace_pools(s,[r for r in cal if r['stratum']==st],seed))
        for rep in range(5):
            for st in TARGETS:
                pool=[r for r in cal if r['stratum']==st][:8]
                if not pool:continue
                for mode in (('SETUP','PREFIX') if rep%2==0 else ('PREFIX','SETUP')):
                    a=s.call(mode,pool,17+rep);a['stratum']=st;a['rep']=rep;profile.append(a)
    write('C07_TRAIN_RAW.json',raw);write('C07_CAL_RAW.json',calraw);write('C07_PROFILE.json',profile)
    print('TRAIN traces',sum(len(a['rows']) for a in raw),'CAL',sum(len(a['rows']) for a in calraw),flush=True)
    labs=oracle(roots);write('C07_TRAIN_LABELS.json',labs);lab={x['id']:x['scores'] for x in labs if x['status']=='EXACT'}
    obs=[[],[],[]];exposed=set();exposed_by=[set() for _ in range(3)];hist=[C.Counter() for _ in range(3)];head=C.Counter();down=C.Counter()
    for a in raw:
        for r in a['rows']:
            if r['id'] not in lab:continue
            z=lab[r['id']];ss=r['steps'];v=[utility(z[x['best']]) for x in ss];opt=max(utility(z[c]) for c in r['legal'])
            head[str(opt-v[0])]+=1;down[str(v[0]-min(utility(z[c]) for c in r['legal']))]+=1
            for j in (1,2,3):
                g=int(2*(v[j]-v[0]));t=sum(x['wall_us'] for x in ss[1:j+1]);obs[j-1].append((r['cell'],g,t));hist[j-1][g]+=1
                if g:exposed.add(r['id']);exposed_by[j-1].add(r['id'])
    assert all(obs)
    fitted=[shrink(x) for x in obs]
    cells=[[fitted[j]['gain_cost'][c] for j in range(3)] for c in range(16)]
    costs=[[sum(s['wall_us'] for s in r['steps'][1:j+1]) for a in calraw for r in a['rows']] for j in (1,2,3)]
    floors=[max(1,math.ceil(S.median(x))) for x in costs]
    m=dict(cells=cells,fitted=fitted,train_labeled=len(lab),train_total=len(roots),exposed=len(exposed),exposed_by=[len(x) for x in exposed_by],histograms=[dict(x) for x in hist],headroom=head,downside=down,
      cal_cost_mean=[S.mean(x) for x in costs],floors=floors,source_sha=sha(D/'C07_COHORT.json'),raw_sha=sha(D/'C07_TRAIN_RAW.json'),labels_sha=sha(D/'C07_TRAIN_LABELS.json'),cal_sha=sha(D/'C07_CAL_RAW.json'),profile_sha=sha(D/'C07_PROFILE.json'),binary_sha=sha(BIN),host_start=start,host_end=host())
    write('C07_MODEL.json',m);print('TRAIN',json.dumps({k:m[k] for k in ('exposed','exposed_by','histograms','train_labeled','cal_cost_mean','floors')}),flush=True)
def live():
    m=read('C07_MODEL.json');co=read('C07_COHORT.json');assert sha(D/'C07_COHORT.json')==m['source_sha'] and sha(D/'C07_TRAIN_RAW.json')==m['raw_sha'] and sha(BIN)==m['binary_sha']
    start=host();raw=[]
    with Server() as s:
        s.call('SETUP',[next(r for r in co['rows'] if r['group']=='TRAIN')])
        for st in TARGETS:
            roots=[r for r in co['rows'] if r['group']=='TEST' and r['stratum']==st]
            assert roots
            for rep,seed in enumerate(SEEDS):
                for k,pool in enumerate(chunks(roots)):
                    budget=min(100000,max(500,round(len(pool)*.5*m['cal_cost_mean'][1])))
                    rotate=(rep+k)%len(ARMS)
                    for arm in ARMS[rotate:]+ARMS[:rotate]:
                        a=s.call(arm,pool,seed,budget,m['floors'] if arm!='PREFIX' else None,m['cells'] if arm in ('DP','RATIO') else None)
                        a.update(stratum=st,pool=k,rep=rep,host=host());raw.append(a)
                print('LIVE',st,seed,len(raw),flush=True)
    write('C08_LIVE_RAW.json',dict(host_start=start,host_end=host(),runs=raw));print('BLIND',sha(D/'C08_LIVE_RAW.json'),flush=True)
def score():
    co=read('C07_COHORT.json');m=read('C07_MODEL.json');roots=[r for r in co['rows'] if r['group']=='TEST'];raw=read('C08_LIVE_RAW.json');runs=raw['runs']
    labs=oracle(roots);write('C08_LABELS.json',labs);label={x['id']:x['scores'] for x in labs if x['status']=='EXACT'}
    by={};summary={};rng=random.Random(202611240304)
    for a in runs:
        d=by.setdefault((a['stratum'],a['mode'],a['seed']),{})
        for r in a['rows']:
            if r['id'] not in label:continue
            z=label[r['id']];opt=max(utility(z[c]) for c in r['legal']);v0=utility(z[r['steps'][0]['best']]);v=utility(z[r['steps'][-1]['best']])
            assert r['id'] not in d;d[r['id']]=dict(regret=opt-v,gain=v-v0,visit=r['steps'][-1]['visits'])
    for st in TARGETS:
        arms={}
        ids=sorted(x['id'] for x in roots if x['stratum']==st and x['id'] in label)
        for arm in ARMS:
            rr=[a for a in runs if a['stratum']==st and a['mode']==arm];assert rr
            rows=[x for seed in SEEDS for x in by[st,arm,seed].values()]
            vals={k:S.mean(r[k] for r in rows) for k in ('regret','gain','visit')}
            vals.update({k:S.mean(sum(a[k] for a in rr if a['seed']==seed) for seed in SEEDS) for k in ('whole_us','transport_us','setup_total_us','prefix_step_us','plan_us','optional_us','continuation_step_us','drop_us','overrun_us')})
            vals.update(selected=S.mean(sum(len(a['selected']) for a in rr if a['seed']==seed) for seed in SEEDS),planned=S.mean(sum(len(a['planned']) for a in rr if a['seed']==seed) for seed in SEEDS),targets=dict(C.Counter(v for a in rr for _,v in a['selected'])),pools=len(rr))
            arms[arm]=vals
        diffs={}
        for arm in ARMS[:-1]:
            d=[S.mean(by[st,arm,s][i]['regret']-by[st,'DP',s][i]['regret'] for s in SEEDS) for i in ids]
            boot=sorted(S.mean(rng.choices(d,k=len(d))) for _ in range(3000)) if d else []
            diffs[arm]=dict(improvement=S.mean(d),bootstrap95=[boot[75],boot[2924]],n=len(d),positive=sum(x>0 for x in d),negative=sum(x<0 for x in d),time_mismatch=abs(arms['DP']['whole_us']/arms[arm]['whole_us']-1))
        summary[st]=dict(n=len(ids),arms=arms,paired=diffs)
    prof=read('C07_PROFILE.json')
    profile={f'{st}/{mode}':{k:S.median(a[k] for a in prof if a['stratum']==st and a['mode']==mode) for k in ('whole_us','setup_total_us','prefix_step_us','drop_us','transport_us')} for st in TARGETS for mode in ('SETUP','PREFIX') if any(a['stratum']==st and a['mode']==mode for a in prof)}
    max_load=max(a['host']['load'][0] for a in runs);cpus=raw['host_start']['cpus'];stable=max_load<=2*cpus and max(raw['host_start']['load'][0],raw['host_end']['load'][0])<=2*cpus
    result=dict(groups=summary,profile=profile,labels=dict(C.Counter(r['status'] for r in labs)),raw_sha=sha(D/'C08_LIVE_RAW.json'),host_max_load1=max_load,host_cpus=cpus,stable_host=stable,exposure_pass=m['exposed']>=20,requests=len(runs),root_traces=sum(len(a['rows']) for a in runs))
    write('C08_SUMMARY.json',result);print('SUMMARY',json.dumps(result),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='generate':prior_lock()
    else:lock()
    {'smoke':smoke,'generate':generate,'train':train,'live':live,'score':score}[sys.argv[1]]()
