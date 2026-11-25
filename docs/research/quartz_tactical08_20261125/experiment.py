"""Bounded actual CPU tactical calibration; every point is resident MCTS."""
import collections as C,gzip,hashlib,json,math,os,random,statistics as S,struct,sys,time
from pathlib import Path
from engine import D,ROOT,BIN,Engine,identity,sha,MODELS
from rules import replay,proof_reference,rule_key,features,fixtures
ST=('WIN','BLOCK','LOSS','QUIET')
FULL=(1,8,32,64,128,256);LOCAL=(1,8,32,64,128);CAL=(1,8,32)
def write(n,o):
    p=D/n;assert not p.exists(),p;p.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')
def read(n):return json.loads((D/n).read_text())
def rows(n):
    with gzip.open(D/n,'rt') as f:
        for x in f:yield json.loads(x)
def emit(f,o):f.write(json.dumps(o,separators=(',',':'),allow_nan=False)+'\n')
def host():
    a={}
    for line in Path('/proc/vmstat').read_text().splitlines():
        k,v=line.split()
        if k in ('pswpin','pswpout'):a[k]=int(v)
    return dict(mono=time.monotonic(),load=os.getloadavg(),swap=a)
def classify(p):
    return 'WIN' if p['own'] else 'BLOCK' if len(p['opponent'])==1 else 'LOSS' if len(p['opponent'])>=2 else 'QUIET'
def targets(p):return p['own'] if p['own'] else p['opponent'] if len(p['opponent'])==1 else []
def eq(a,b,tol=1e-6):assert abs(a-b)<=tol*max(1,abs(a),abs(b)),(a,b)
def check(a,h,rule,arm,budgets,seed,model,ref=None):
    if ref is None:
        b,p,w=replay(h,rule);pr=proof_reference(b,p,rule);key=rule_key(b,p,rule);feat=features(h)
    else:
        b,p,w=ref.replay(h,rule);pr=ref.proof_reference(b,p,rule);key=ref.rule_key(b,p,rule);feat=ref.features(h)
    assert w is None and a['schema']=='quartz_gomoku_tactical_v1'
    assert a['history']==h and a['rule']==rule and a['arm']==arm and a['budgets']==list(budgets) and a['seed']==seed and a['threads']==1
    assert a['board']==b and a['player']==p and a['proof']==pr and a['targets']==targets(pr) and a['rule_key']==key and a['recent']==h[-7:]
    legal=[m for m,v in enumerate(b) if v==0];assert a['legal']==legal
    t=set(targets(pr));guard=pr['own'][0] if pr['own'] else pr['opponent'][0] if len(pr['opponent'])==1 else legal[0] if len(legal)==1 else None
    stop=pr['witness'] if arm=='CERT' else guard if arm=='GUARD' else None
    assert a['model_identity']==identity(model)
    assert a['transport_ns']>0 and a['python_cpu_ns']>=0 and a['rust_cpu_ticks']>=0 and a['tick_hz']>0
    assert a['whole_us']>=a['drop_us']+a['setup_us']+a['plan_us'] and a['nn_rows']==a['evaluator_calls']
    if stop is not None:
        assert a['cause']==('CERTIFIED' if arm=='CERT' else 'GUARDED') and a['best']==stop
        assert a['initial'] is None and a['target_rank'] is None and a['target'] is None and a['forced_first'] is None and a['root_floor']==0
        assert a['visits']==a['nn_rows']==a['nn_batches']==0 and a['root_nn_feature_sha256'] is None
        x=a['points'];assert len(x)==1 and x[0]['budget']==8 and x[0]['visits']==0 and x[0]['best']==stop and x[0]['result'] is None
        return
    assert a['cause']=='VISITS' and a['visits']==budgets[-1] and a['nn_rows']>=1
    assert a['root_nn_feature_sha256']==hashlib.sha256(struct.pack('<3825f',*feat)).hexdigest()
    assert 1<=a['nn_batches']<=a['nn_rows']
    init=a['initial'];ca=init['candidates'];assert len(ca)==len(legal) and {m for m,v in ca}==set(legal)
    assert all(type(m) is int and math.isfinite(v) and v>=0 for m,v in ca);eq(sum(v for _,v in ca),1,1e-5)
    assert all(x[1]>=y[1] for x,y in zip(ca,ca[1:]))
    assert init['root_n']==0 and init['root_w']==0 and init['evaluator_calls']==1 and len(init['edges'])==1
    rank=next((i+1 for i,(m,v) in enumerate(ca) if m in t),None);target=ca[rank-1][0] if rank else None
    floor=rank if arm in ('EXPOSE','FOCUS') else 0
    assert a['target_rank']==rank and a['target']==target and a['root_floor']==floor and a['forced_first']==(target if arm=='FOCUS' else None)
    before={e['move']:e for e in init['edges']};prev=0;nn=1;cum=0
    for budget,x in zip(budgets,a['points'],strict=True):
        r=x['result'];es=r['edges']
        assert x['budget']==budget and x['visits']==budget and r['root_n']==budget and x['best']==r['best']
        assert r['root_hash']==a['tt_hash']==init['root_hash'] and r['candidates']==ca
        expected=min(len(legal),max(1,math.isqrt(4*(budget-1)),floor))
        assert len(es)==expected,(arm,budget,len(es),expected,rank)
        assert [e['move'] for e in es]==[m for m,v in ca[:expected]]
        assert sum(e['n'] for e in es)==budget;eq(sum(e['w'] for e in es),r['root_w'])
        assert max(enumerate(es),key=lambda it:(it[1]['n'],it[0]))[1]['move']==x['best']
        assert r['evaluator_calls']>=nn and r['evaluator_calls']<=nn+budget-prev
        assert x['cumulative_us']>=cum+x['step_us']
        for e,(m,v) in zip(es,ca):
            n=e['n'];W=e['w'];M=e['m2'];assert type(n) is int and n>=0 and math.isfinite(W) and math.isfinite(M)
            assert e['vl']==0 and abs(e['vvalue'])<1e-6 and abs(W)<=n+1e-6 and -1e-7<=M<=(n-W*W/n if n else 0)+1e-5
            eq(e['prior'],v)
            if m in before:assert n>=before[m]['n']
            if m in pr['own'] and n>0:assert abs(W-n)<1e-8 and abs(M)<1e-8
        if t:
            E=any(e['move'] in t for e in es);V=any(e['move'] in t and e['n']>0 for e in es);G=x['best'] in t
            assert E==(rank<=expected) and (not V or E) and (not G or V)
            if arm in ('EXPOSE','FOCUS'):assert E
            if arm=='FOCUS':assert V
        if budget==1 and arm=='FOCUS':
            assert next(e['n'] for e in es if e['move']==target)==1
        before={e['move']:e for e in es};prev=budget;nn=r['evaluator_calls'];cum=x['cumulative_us']
    assert a['best']==a['points'][-1]['best'] and a['nn_rows']==nn and a['whole_us']>=cum
def smoke():
    f=fixtures();f['block']=[0,100,2,101,4,102,99,103];rows0=[];reject=[];parity=[]
    with Engine(1,'F1') as e:
        for nm in ('immediate','block','double_loss','trans_a'):
            h=f[nm];arms=['BASE','GUARD','CERT']+(['EXPOSE','FOCUS'] if nm in ('immediate','block') else [])
            for arm in arms:
                budget=[8] if arm in ('GUARD','CERT') else [1,8]
                a=e.ladder(h,arm,budget);check(a,h,'freestyle',arm,budget,17,'F1');a['fixture']=nm;rows0.append(a)
                print('SMOKE',nm,arm,a['cause'],a['target_rank'],a['visits'],a['nn_rows'],flush=True)
            if nm=='trans_a':
                old=e.search(h,'freestyle','BASE',8,1,17)
                b=next(x for x in rows0 if x['fixture']==nm and x['arm']=='BASE')
                assert old['result']==b['points'][-1]['result']
                parity.append(dict(old=old,ladder=b['points'][-1]))
        for q in [
            dict(cmd='research_gomoku15_tactical_ladder',rule='renju',history=[],arm='BASE',budgets=[8],seed=17),
            dict(cmd='research_gomoku15_tactical_ladder',rule='freestyle',history=[0,0],arm='BASE',budgets=[8],seed=17),
            dict(cmd='research_gomoku15_tactical_ladder',rule='freestyle',history=[],arm='EXPOSE',budgets=[8],seed=17),
            dict(cmd='research_gomoku15_tactical_ladder',rule='freestyle',history=[],arm='BASE',budgets=[8,1],seed=17),
            dict(cmd='research_gomoku15_tactical_ladder',rule='freestyle',history=[],arm='BASE',budgets=[8],seed=17,oracle=1)]:
            a=e.request(q);assert 'error' in a;reject.append(dict(request=q,response=a))
        params=e.model_parameters
    with Engine(1,'F2') as e:
        h=f['immediate'];a=e.ladder(h,'BASE',[1,8]);check(a,h,'freestyle','BASE',[1,8],17,'F2');a['fixture']='immediate';rows0.append(a)
        assert e.model_parameters==params
    # Exact same full candidates for search arms on the same actual model/state.
    for nm in f:
        x=[a['initial']['candidates'] for a in rows0 if a['fixture']==nm and a['initial'] and a['model_identity']['model']=='F1']
        assert not x or all(a==x[0] for a in x)
    write('C06_SMOKE.json',dict(rows=rows0,rejected=reject,parity=parity,parameters=params))
def campaign(model,group):
    data=read('C07_COHORT.json');rr=data[group]
    assert len(rr)==(8 if group=='CAL' else 16)
    name=f'C07_CAL_{model}.jsonl.gz' if group=='CAL' else f'C08_TEST_{model}.jsonl.gz'
    assert not (D/name).exists()
    before=host();count=0;maxload=0.
    with Engine(1,model) as eng,gzip.open(D/name,'wt',compresslevel=3) as f:
        for i,r in enumerate(rr):
            h=r['history'];rule=r['rule'];seed=17
            arms=[('BASE',CAL if group=='CAL' else FULL)]
            if group=='TEST' and model=='F1':
                if r['stratum'] in ('WIN','BLOCK'):arms += [('EXPOSE',LOCAL),('FOCUS',LOCAL)]
                arms += [('GUARD',(8,)),('CERT',(8,))]
            off=i%len(arms);arms=arms[off:]+arms[:off]
            for arm,budgets in arms:
                h0=host();a=eng.ladder(h,arm,budgets,rule,seed);h1=host()
                check(a,h,rule,arm,budgets,seed,model)
                assert classify(a['proof'])==r['stratum']
                a.update(id=r['id'],family=r['family'],group=group,stratum=r['stratum'],host_start=h0,host_end=h1)
                emit(f,a);f.flush();count+=1;maxload=max(maxload,h0['load'][0],h1['load'][0])
            print('ROOT',model,group,r['id'],r['stratum'],count,flush=True)
    end=host();dt=end['mono']-before['mono'];sd={k:end['swap'][k]-before['swap'][k] for k in before['swap']}
    write(name.replace('.jsonl.gz','_HOST.json'),dict(model=model,group=group,requests=count,raw_sha256=sha(D/name),start=before,end=end,seconds=dt,swap_delta=sd,swap_pages_s=sum(sd.values())/dt,max_load=maxload,timing_gate=sum(sd.values())/dt<=512 and maxload<=12,concurrent_model_workers=True))
    print('RAW_FROZEN',model,group,count,sha(D/name),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='smoke':smoke()
    else:campaign(sys.argv[1],sys.argv[2])
