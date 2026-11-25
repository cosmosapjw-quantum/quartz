"""Frozen actual parallel neural-MCTS experiment; no exact labels in online requests."""
import collections,concurrent.futures,gzip,hashlib,json,math,os,random,statistics,subprocess,sys,time
from pathlib import Path
from hurdle import shrink
D=Path(__file__).resolve().parent; ROOT=D.parents[2]
HIST=ROOT/'docs/research/cognitive_metareasoning_20261120'
BIN=Path('/home/cosmosapjw/quantum/research/quartz_nn_intervention_target_20261124/release/mcts_demo')
MODEL=HIST/'connect4_generalization_07/evidence/mlp_conv_train_20261121/conv_model.json'
EXT=Path('/home/cosmosapjw/quantum/research/quartz_external_engines_20261120/pascalpons_connect4')
IDENT='161f9e6e26840754384cb47335d1082d837fa014735cbfaa2517209422e29f24'
BIN_ID='053646f92f1acf03cbd07d56815a9ea8de26cc58c60ade395a6f612be1c13819'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    p=D/name;assert not p.exists(),p
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def read(name):return json.loads((D/name).read_text())
def lock():
    assert sha(MODEL)==IDENT and sha(BIN)==BIN_ID
    assert sha(EXT/'c4solver')=='fca2b665b1b3dbd88f2f50a7e0ba86fd8ae253f2861da1ad51cf110c69f741ed'
    assert sha(EXT/'7x6_small.book')=='38f9834317c37d9516e45a21da598569a5d1556595686593d14c2e63f59c1f38'
def board(h):
    b=[0]*42;z=[0]*7
    for i,a in enumerate(h):
        c=int(a)-1;p=1-2*(i%2)
        assert 0<=c<7 and z[c]<6
        b[7*z[c]+c]=p;z[c]+=1
        if i+1<len(h):assert not won(b,p)
    return b,z
def won(b,p):
    return any(all(b[(y+k*dy)*7+x+k*dx]==p for k in range(4))
        for y in range(6) for x in range(7) for dx,dy in ((1,0),(0,1),(1,1),(-1,1))
        if 0<=x+3*dx<7 and 0<=y+3*dy<6)
def immediate(b,z,p):
    for c in range(7):
        if z[c]<6:
            u=b.copy();u[z[c]*7+c]=p
            if won(u,p):return True
    return False
def mirror(h):return ''.join(str(8-int(a)) for a in h)
def canon(h):return min(h,mirror(h))
def bkey(b):return min(tuple(b),tuple(b[y*7+6-x] for y in range(6) for x in range(7)))
def old_sources():
    refs=['connect4_generalization_07/COHORT_1024_FROZEN.json','connect4_recognizer_06/COHORT_FROZEN.json','standard_external_04/POSITIONS_FIXED.json',
      'voc_calibration_09/COHORT_384_FROZEN.json','dynamic_voc_10/COHORT_384_FROZEN.json','dynamic_costmatch_11/COHORT_384_FROZEN.json',
      'einstellung_reopening_12/COHORT_384_FROZEN.json','resident_reopening_13/COHORT_384_FROZEN.json']
    prefixes=set(); boards=set(); meta={}
    for ref in refs:
        p=HIST/ref;j=json.loads(p.read_text());rows=j.get('rows') or j.get('connect4')
        assert rows
        for r in rows:
            h=r.get('history',r.get('sequence'));b,z=board(h)
            assert b==r['board']
            prefixes.add(canon(h[:5]));boards.add(bkey(b))
        meta[ref]={'sha256':sha(p),'rows':len(rows)}
    return prefixes,boards,meta
class Server:
    def __init__(self):
        lock()
        env=dict(os.environ,RAYON_NUM_THREADS='2')
        self.p=subprocess.Popen([str(BIN),'--server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1,env=env)
    def call(self,mode,jobs,seed=17,budget_us=0,floor_us=0,cells=None):
        q=dict(cmd='research_connect4_intervention_cpu',mode=mode,seed=seed,model_path=str(MODEL),model_identity=IDENT,jobs=[dict(id=r['id'],history=r['history']) for r in jobs],budget_us=budget_us,floor_us=floor_us)
        if cells is not None:q['cells']=cells
        t=time.perf_counter_ns()
        self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('server EOF '+str(self.p.poll()))
        a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000
        assert 'error' not in a,a
        assert a['threads']==2 and a['model_identity']==IDENT and len(a['rows'])==len(jobs)
        for r,j in zip(a['rows'],jobs):
            assert r['id']==j['id'] and r['history']==j['history']
            b,z=board(j['history']);leg=[c for c in range(7) if z[c]<6];p=1-2*(len(j['history'])%2)
            assert r['legal']==leg and r['threat']==immediate(b,z,-p)
            ss=r['steps'];assert ss[0]['visits']==4
            for s in ss:
                assert s['best'] in leg and s['root_hash']==r['root_hash'] and s['wall_us']>=1
                assert all(c in leg and v>=0 for c,v in s['policy'])
                assert abs(sum(v for _,v in s['policy'])-1)<1e-5
            assert all(x['eval_calls']<=y['eval_calls'] for x,y in zip(ss,ss[1:]))
            e=-sum(v*math.log(v) for _,v in ss[0]['policy'] if v>0)/math.log(7)
            top=max(v for _,v in ss[0]['policy'])
            assert abs(e-r['entropy'])<1e-6 and abs(top-r['top'])<1e-6
            c=int(r['entropy']>=.55)+2*int(r['top']>=.75)+4*int(r['threat'])+8*int(len(leg)<=4)
            assert r['cell']==c
        return a
    def close(self):
        self.p.stdin.close();assert self.p.wait(timeout=10)==0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke():
    with Server() as s:
        rows=[{'id':'a','history':'324252'},{'id':'b','history':'4433226223'}]
        t=s.call('TRACE',rows)
        assert all([x['visits'] for x in r['steps']]==[4,8,16,32] for r in t['rows'])
        assert t['rows'][0]['threat']
        assert all(r['steps'][0]['eval_calls']>0 for r in t['rows'])
        p=s.call('SIGNED16',rows,budget_us=1_000_000,floor_us=1,cells=[[.05,100.] for _ in range(16)])
        assert all([x['visits'] for x in r['steps']]==[4,8,16] for r in p['rows'])
        # Strict serde request rejects a label field.
        q=dict(cmd='research_connect4_intervention_cpu',mode='GEN',seed=17,model_path=str(MODEL),model_identity=IDENT,jobs=rows,oracle=[1])
        s.p.stdin.write(json.dumps(q)+'\n');s.p.stdin.flush();bad=json.loads(s.p.stdout.readline());assert 'error' in bad
        write('C06_NATIVE_SMOKE.json',dict(trace=t,signed=p,oracle_rejection=bad))
    print('native smoke: real CNN, 2 threads, two roots, resident 4/8/16/32, signed pool, oracle rejected',flush=True)
def generate():
    oldp,oldb,meta=old_sources();rng=random.Random(2026112402)
    usedp=set(oldp);usedb=set(oldb);rows=[];attempts=0
    path=D/'C07_GENERATION.jsonl.gz';assert not path.exists()
    targets=[('TRAIN',144),('STRESS',48),('NATURAL',48)]
    with Server() as s,gzip.open(path,'wt',compresslevel=6) as log:
        for group,target in targets:
            count=[0,0]
            while sum(count)<target and attempts<1800:
                batch=[]
                # A unique mirror-canonical opening for every attempted game.
                for _ in range(min(32,1800-attempts)):
                    while True:
                        h=''.join(str(rng.randrange(1,8)) for _ in range(5))
                        k=canon(h)
                        if k not in usedp:usedp.add(k);break
                    i=attempts;attempts+=1
                    batch.append(dict(id=f'{group}_{i:04d}',group=group,opening=h,history=h,parity=i%2,events=[],candidate=None,status='RUNNING'))
                for ply in range(5,31):
                    active=[]
                    for g in batch:
                        if g['status']!='RUNNING':continue
                        h=g['history'];assert len(h)==ply
                        b,z=board(h);p=1-2*(ply%2);legal=[c for c in range(7) if z[c]<6]
                        if won(b,-p) or not legal:g['status']='TERMINAL';continue
                        if 20<=ply<=30 and ply%2==g['parity'] and g['candidate'] is None and len(legal)>=2 and (group=='NATURAL' or not immediate(b,z,p)):
                            if bkey(b) not in usedb and count[ply%2]<target//2:
                                row=dict(id=g['id'],family=canon(g['opening']),group=group,history=h,board=b,ply=ply,player=p,legal=legal,immediate=immediate(b,z,p))
                                usedb.add(bkey(b));count[ply%2]+=1;rows.append(row);g['candidate']=h
                        if ply==30:g['status']='DEPTH_CAP';continue
                        active.append(g)
                    if not active:break
                    a=s.call('GEN',active,seed=17)
                    for g,r in zip(active,a['rows']):
                        recommendation=r['steps'][0]['best'];explore=rng.random()<.45
                        move=rng.choice(r['legal']) if explore else recommendation
                        g['events'].append(dict(ply=ply,recommendation=recommendation,action=move,explore=explore,eval_calls=r['steps'][0]['eval_calls'],root_hash=r['root_hash']))
                        g['history']+=str(move+1)
                for g in batch:log.write(json.dumps(g,separators=(',',':'))+'\n')
                print('generation',group,count,'attempts',attempts,flush=True)
            # Keep cap shortfalls explicit. Do not fill using outcome-selected positions.
    write('C07_COHORT.json',dict(rows=rows,attempts=attempts,targets=dict(targets),old_prefixes=len(oldp),old_boards=len(oldb),old_sources=meta,gen_sha256=sha(path)))
    print('COHORT',collections.Counter((x['group'],x['player']) for x in rows),flush=True)
def oracle_one(row):
    h=row['history'];t=time.perf_counter()
    try:
        p=subprocess.run([str(EXT/'c4solver'),'-a','-b',str(EXT/'7x6_small.book')],input=h+'\n',capture_output=True,text=True,timeout=8,cwd=EXT)
        vals=p.stdout.strip().split()
        if p.returncode or len(vals)!=8 or vals[0]!=h:return dict(id=row['id'],status='ERROR',returncode=p.returncode,stdout=p.stdout[-500:])
        v=list(map(int,vals[1:]))
        assert all((x!=-1000)==(i in row['legal']) and (x==-1000 or -21<=x<=21) for i,x in enumerate(v))
        return dict(id=row['id'],status='EXACT',scores=v,wall_s=time.perf_counter()-t)
    except subprocess.TimeoutExpired:return dict(id=row['id'],status='TIMEOUT')
def oracle(rows):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:return list(ex.map(oracle_one,rows))
def utility(score):return (1+(score>0)-(score<0))/2
def train():
    data=read('C07_COHORT.json');roots=[r for r in data['rows'] if r['group']=='TRAIN']
    with Server() as s:
        raw=[s.call('TRACE',roots,seed=k) for k in (17,43)]
    write('C07_TRAIN_RAW.json',raw)
    labels=oracle(roots);write('C07_TRAIN_LABELS.json',labels)
    m={r['id']:r for r in labels if r['status']=='EXACT'}
    observations=[];cost8=[];cost16=[];exposed=set();exposed16=set();hist=collections.Counter();hist32=collections.Counter()
    for run in raw:
        for r in run['rows']:
            if r['id'] not in m:continue
            z=m[r['id']]['scores'];steps=r['steps'];assert [x['visits'] for x in steps]==[4,8,16,32]
            v=[utility(z[x['best']]) for x in steps];g=int(2*(v[2]-v[0]));g32=int(2*(v[3]-v[0]))
            c8=steps[1]['wall_us'];c16=c8+steps[2]['wall_us']
            observations.append((r['cell'],g,c16));cost8.append(c8);cost16.append(c16);hist[g]+=1;hist32[g32]+=1
            if g or g32:exposed.add(r['id'])
            if g:exposed16.add(r['id'])
    model=shrink(observations);model.update(exposed=len(exposed),exposed16=len(exposed16),train_labeled=len(m),observations=len(observations),
      histogram=dict(hist),histogram32=dict(hist32),cost16_mean=statistics.mean(cost16),cost16_median=statistics.median(cost16),cost8_median=statistics.median(cost8),
      raw_sha256=sha(D/'C07_TRAIN_RAW.json'),label_sha256=sha(D/'C07_TRAIN_LABELS.json'),cohort_sha256=sha(D/'C07_COHORT.json'),binary_sha256=BIN_ID,model_sha256=IDENT)
    write('C07_MODEL.json',model)
    print('TRAIN',json.dumps({k:model[k] for k in ('exposed','exposed16','train_labeled','observations','histogram','histogram32','cost16_mean','counts')}),flush=True)
def live():
    m=read('C07_MODEL.json');assert m['cohort_sha256']==sha(D/'C07_COHORT.json') and m['raw_sha256']==sha(D/'C07_TRAIN_RAW.json') and m['label_sha256']==sha(D/'C07_TRAIN_LABELS.json')
    data=read('C07_COHORT.json');raw=[]
    arms=['PREFIX','FIXED8','RANDOM16','ENTROPY16','SIGNED16']
    with Server() as s:
        # Warm model and thread pool; use training root, never a held-out root.
        s.call('GEN',[next(r for r in data['rows'] if r['group']=='TRAIN')])
        for group in ('STRESS','NATURAL'):
            roots=[r for r in data['rows'] if r['group']==group];assert roots
            budget=max(1000,round(len(roots)*.5*m['cost16_mean']))
            for rep,seed in enumerate((17,43)):
                rotation=(rep+(0 if group=='STRESS' else 2))%len(arms)
                for arm in arms[rotation:]+arms[:rotation]:
                    floor=math.ceil(m['cost8_median'] if arm=='FIXED8' else m['cost16_median'])
                    a=s.call(arm,roots,seed,budget,floor,m['gain_cost'] if arm=='SIGNED16' else None);a['group']=group;raw.append(a)
                    print('LIVE',group,seed,arm,len(a['selected']),a['whole_us'],flush=True)
    write('C08_LIVE_RAW.json',raw)
    print('BLIND_RAW',sha(D/'C08_LIVE_RAW.json'),flush=True)
def score():
    roots=[r for r in read('C07_COHORT.json')['rows'] if r['group']!='TRAIN']
    labels=oracle(roots);write('C08_LABELS.json',labels)
    label={r['id']:r['scores'] for r in labels if r['status']=='EXACT'};raw=read('C08_LIVE_RAW.json')
    def avg(x):return sum(x)/len(x) if x else None
    rows=[];groups={}
    for a in raw:
        gains=[];regrets=[];initial=[];by={}
        for r in a['rows']:
            if r['id'] not in label:continue
            z=label[r['id']];opt=max(utility(z[c]) for c in r['legal']);v0=utility(z[r['steps'][0]['best']]);v1=utility(z[r['steps'][-1]['best']])
            gains.append(v1-v0);regrets.append(opt-v1);initial.append(v0);by[r['id']]=[opt-v1,v1-v0,opt-v0]
        rows.append(dict(group=a['group'],seed=a['seed'],arm=a['mode'],n=len(gains),regret=avg(regrets),gain=avg(gains),initial=avg(initial),selected=len(a['selected']),
           whole_us=a['whole_us'],transport_us=a['transport_us'],optional_us=a['optional_us'],budget_us=a['budget_us'],overrun_us=a['overrun_us'],
           eval_calls=sum(r['steps'][-1]['eval_calls'] for r in a['rows']),visits=sum(r['steps'][-1]['visits'] for r in a['rows']),
           by=by))
    rng=random.Random(2026112403)
    for group in ('STRESS','NATURAL'):
        out={}
        for arm in ('PREFIX','FIXED8','RANDOM16','ENTROPY16','SIGNED16'):
            rr=[r for r in rows if r['group']==group and r['arm']==arm];assert len(rr)==2
            out[arm]={k:avg([r[k] for r in rr]) for k in ('regret','gain','whole_us','transport_us','selected','optional_us','overrun_us','eval_calls','visits')}
        for arm in out:
            out[arm]['time_mismatch_vs_random']=abs(out[arm]['whole_us']/out['RANDOM16']['whole_us']-1)
        # One family unit = mean of the two actual independent search-arm replicates.
        a=[r for r in rows if r['group']==group and r['arm']=='SIGNED16'];b=[r for r in rows if r['group']==group and r['arm']=='RANDOM16'];a.sort(key=lambda r:r['seed']);b.sort(key=lambda r:r['seed'])
        ids=sorted(set(a[0]['by'])&set(a[1]['by'])&set(b[0]['by'])&set(b[1]['by']))
        diff=[avg([b[k]['by'][i][0]-a[k]['by'][i][0] for k in (0,1)]) for i in ids]
        boot=sorted(avg(rng.choices(diff,k=len(diff))) for _ in range(4000)) if diff else []
        groups[group]=dict(arms=out,paired_regret_improvement=avg(diff),paired_n=len(diff),bootstrap95=[boot[100],boot[3900]] if boot else None,positive=sum(x>0 for x in diff),negative=sum(x<0 for x in diff))
    write('C08_SUMMARY.json',dict(groups=groups,rows=[{k:v for k,v in r.items() if k!='by'} for r in rows],labels=collections.Counter(r['status'] for r in labels),raw_sha256=sha(D/'C08_LIVE_RAW.json')))
    print('SUMMARY',json.dumps(groups),flush=True)
if __name__=='__main__':
    lock();{'smoke':smoke,'generate':generate,'train':train,'live':live,'score':score}[sys.argv[1]]()
