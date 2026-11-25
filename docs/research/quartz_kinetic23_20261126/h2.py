"""K23 prospective H2 factorial interventions, sequential policy OPE and channel shrinkage."""
from source_gen import D,ROOT,R,BASE,SEED,PRICE,sha,read,save,raw,emit,ident,k20,p,old,BIN
import numpy as np
import math,random,itertools,statistics as S,collections as C,gzip,sys
PTH=list(itertools.product(range(3),repeat=2));NAMES=[''.join('NFX'[x] for x in z) for z in PTH]
U=np.array(list(itertools.product((0.,.5,1.),repeat=3)))
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
def sc(s):return .5+.5*np.asarray(s['w'][:s['len']])/(np.asarray(s['n'][:s['len']])+4)
def phi(x):a=np.exp(4*x);return math.log(a.mean())/4,a/a.sum()
def panel(s):
    n=s['n'];m=s['moves'][:s['len']];a=m.index(s['best']);b=max((k for k in range(len(m)) if k!=a),key=lambda k:(n[k],-k));c=next(k for k in range(len(m)) if k not in (a,b))
    return [m[k] for k in (a,b,c)]
def check(x,r):
    assert 'error' not in x,x
    assert x['history']==r['history'] and x['legal']==r['legal'] and x['best'] in r['legal'] and x['proof']==k20.proof(r['history'])
    pre,mid,end=x['states'];assert pre['visits']==8 and mid['visits']<=64 and end['visits']==x['visits']==x['result']['root_n']<=128
    assert pre['nn']+sum(x['incremental_nn'])==end['nn']==x['NN_calls']==x['result']['evaluator_calls']
    assert x['panels']==[panel(pre),panel(mid)] and all(s['moves']==pre['moves'] for s in x['states']+x['pulses'])
    cert=k20.cert(x['proof']);assert x['best']==(end['best'] if cert is None else cert)
    for t,(s,pl,en) in enumerate(zip(x['states'],x['pulses'],x['states'][1:])):
        c=x['chosen'][t];inc=en['nn']-s['nn'];assert inc==x['incremental_nn'][t] and 0<=inc<=6
        assert x['causes'][t]==('CERTIFIED' if cert is not None else 'NN_TARGET' if inc==6 else 'VISIT_CAP')
        assert s['visits']<=pl['visits']<=min(s['visits']+4,en['visits'])
        if cert is None and inc<6:assert en['visits']==(64 if t==0 else 128)
        if c:
            m=x['panels'][t][c];j=s['moves'].index(m);assert x['forced'][t]==m and pl['n'][j]-s['n'][j]==pl['visits']-s['visits']
        else:assert x['forced'][t] is None
        assert all(a<=b<=c for a,b,c in zip(s['n'],pl['n'],en['n']))
    for j,s in enumerate(x['states']):
        n=np.array(s['n'][:s['len']]);assert sum(n)==s['visits'] and set(s['moves'][:s['len']])==set(r['legal'])
        assert np.all(np.abs(s['w'][:s['len']])<=n+1e-6)
        a=sc(s);b=np.full(len(a),.5) if j==0 else sc(x['states'][j-1])
        f0,q=phi(b);f1,v=phi(a);kl=q@np.log(q/v);dr=q@(a-b)
        near([s['phi'],s['kl'],s['drift'],s['bregman']],[f1,kl,dr,.25*kl])
        n.sort();assert s['bin']==4*int(n[-1]/n.sum()>=.75)+2*int(n[-1]-n[-2]>=2)+int(kl>=.02)
    assert end['best']==x['result']['best']
    es={e['move']:e for e in x['result']['edges']}
    for j,m in enumerate(end['moves'][:end['len']]):
        e=es.get(m,{});assert end['n'][j]==e.get('n',0);near([end['w'][j],end['m2'][j]],[e.get('w',0),e.get('m2',0)])
def order(i):
    a=list(range(9));random.Random(SEED+900000+i).shuffle(a);return a
def pick(r,t):
    a=t['first'][r['b']];return 3*a+t['second'][a][r['mid'][a]]
def trial(part):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    if part=='TEST':
        f=read('FROZEN.json');assert f['code']==sha(D/'h2.py') and f['cohort']==sha(D/'COHORT.json') and f['identity']==ident()
    path=D/(part+'_ATLAS.jsonl.gz');assert not path.exists();cnt=vis=nn=0;caps=C.Counter();parity=0
    with k20.Server(BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            if (r['group']=='TEST')!=(part=='TEST'):continue
            seq=order(r['id']);arms=[None]*9
            for i in seq:
                x=srv.request(r['history'],'K23PATH',h2_path=list(PTH[i]));check(x,r);assert x['chosen']==list(PTH[i])
                arms[i]=x;cnt+=1;vis+=x['visits'];nn+=x['NN_calls'];caps.update(x['causes'])
            assert all(x['states'][0]==arms[0]['states'][0] for x in arms)
            assert all(arms[3*a+b]['states'][1]==arms[3*a]['states'][1] and arms[3*a+b]['pulses'][0]==arms[3*a]['pulses'][0] for a,b in PTH)
            t=dict(root=r,order=seq,logged=seq[0],e=[1/3]*3,f=[[1/3]*3]*3,arms=arms)
            if part=='TEST':
                table=f['tables'][f['selected']];x=srv.request(r['history'],'K23LIVE',h2_table=table);check(x,r);a,b=x['chosen'];q=arms[3*a+b]
                assert a==table['first'][x['states'][0]['bin']] and b==table['second'][a][x['states'][1]['bin']]
                assert all(x[k]==q[k] for k in ('states','pulses','best','visits','panels','incremental_nn','causes','result'))
                t['live']=x;parity+=1;cnt+=1;vis+=x['visits'];nn+=x['NN_calls']
            emit(fd,t)
    save(part+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),sha256=sha(path),identity=ident(),requests=cnt,visits=vis,NN=nn,causes=dict(caps),live_parity=parity,freeze=None if part!='TEST' else sha(D/'FROZEN.json')))
    print('TRIAL',part,cnt,vis,nn,dict(caps),'live parity',parity,flush=True)
def labels(part):
    lk=read(part+'_LOCK.json');assert lk['sha256']==sha(D/(part+'_ATLAS.jsonl.gz'))
    co=read('COHORT.json');assert lk['cohort']==sha(D/'COHORT.json')
    if part=='TEST':assert lk['freeze']==sha(D/'FROZEN.json')
    out=[];cnt=C.Counter()
    for r in co['roots']:
        if (r['group']=='TEST')!=(part=='TEST'):continue
        q=old.oracle(r);item=dict(root=r,status=q['status'])
        if q['status']=='EXACT':
            try:
                v,n=p.ref.exact_wdl(r['history']);truth={a:(q['scores'][a]>0)-(q['scores'][a]<0) for a in r['legal']}
                item.update(scores=q['scores'],independent=v,nodes=n,agreement=v==truth)
                if v!=truth:item['status']='MISMATCH'
            except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
        else:item['oracle_result']=q
        cnt[item['status']]+=1;out.append(item)
    save(part+'_LABELS.json',dict(atlas=lk['sha256'],cohort=sha(D/'COHORT.json'),counts=dict(cnt),rows=out));print('LABELS',part,dict(cnt),flush=True)
def data(part):
    lab=read(part+'_LABELS.json');lk=read(part+'_LOCK.json');assert lab['atlas']==lk['sha256']==sha(D/(part+'_ATLAS.jsonl.gz'))
    ll={x['root']['id']:x for x in lab['rows'] if x['status']=='EXACT'};rows=[]
    for t in raw(D/(part+'_ATLAS.jsonl.gz')):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u={int(k):(v+1)/2 for k,v in ll[i]['independent'].items()}
        pre=t['arms'][0]['states'][0];pa=panel(pre);index=[pre['moves'].index(m) for m in pa]
        x=[];nn=[];us=[];z2=[];cpu=[];wall=[];vv=[];mid=[];z1=[]
        def z(s):return 2*int(np.argmax(sc(s)[index]))+int(s['kl']>=.02)
        for a in range(3):
            s=t['arms'][3*a]['states'][1];mid.append(s['bin']);z1.append(z(s))
        for a in t['arms']:
            assert all(lo<=u[m]<=hi for m,lo,hi in a['proof'])
            us.append(u[a['best']]);nn.append(a['incremental_nn']);x.append(us[-1]-PRICE*sum(nn[-1]));z2.append(z(a['states'][2]))
            cpu.append(a['cpu_ns']/1e6);wall.append(a['wall_ns']/1e6);vv.append(a['visits'])
        uid=int(np.where(np.all(U==[u[m] for m in pa],axis=1))[0][0])
        rows.append(dict(id=i,group=r['group'],b=pre['bin'],mid=mid,y=x,us=us,n=nn,pre=u[pre['best']],proxy=float(sc(pre)[index[0]]),full=max(u.values()),u=uid,z1=z1,z2=z2,logged=t['logged'],cpu=cpu,wall=wall,vis=vv))
    return rows
def static(k):
    a,b=PTH[k];return dict(first=[a]*8,second=[[b]*8 for _ in range(3)])
def shrink(v,pooled):
    return (sum(v)+8*pooled)/(len(v)+8)
def q1fit(tr,t):
    q=[]
    for b in range(8):
        q.append([shrink([r['y'][3*a+t['second'][a][r['mid'][a]]] for r in tr if r['b']==b],S.mean(r['y'][3*a+t['second'][a][r['mid'][a]]] for r in tr)) for a in range(3)])
    return q
def fit():
    assert not (D/'TEST_ATLAS.jsonl.gz').exists() and not (D/'TEST_LABELS.json').exists()
    rows=data('TRAIN_CAL');tr=[r for r in rows if r['group']=='TRAIN'];cal=[r for r in rows if r['group']=='CAL'];assert len(tr)>60 and len(cal)>20
    q2=[[[shrink([r['y'][3*a+c] for r in tr if r['mid'][a]==b],S.mean(r['y'][3*a+c] for r in tr)) for c in range(3)] for b in range(8)] for a in range(3)]
    adap=dict(first=[0]*8,second=[[int(np.argmax(z)) for z in q] for q in q2])
    adap['first']=[int(np.argmax(z)) for z in q1fit(tr,adap)]
    bst=int(np.argmax(np.mean([r['y'] for r in tr],axis=0)));tables={n:static(i) for i,n in enumerate(NAMES)}
    tables.update(best_static=static(bst),adaptive=adap)
    calnet={k:S.mean(r['y'][pick(r,t)] for r in cal) for k,t in tables.items()}
    selected=max(('NN','XN','XX','best_static','adaptive'),key=calnet.__getitem__)
    q1={k:q1fit(tr,t) for k,t in tables.items()}
    anchors={}
    for k,t in tables.items():
        rr=[r for r in tr if pick(r,t)!=0];v=[(r['y'][pick(r,t)]+r['y'][0])/2 for r in rr];pooled=S.mean(v) if v else .5
        anchors[k]=[shrink([(r['y'][pick(r,t)]+r['y'][0])/2 for r in rr if r['b']==b],pooled) for b in range(8)]
    # Shared latent prior and arm-specific likelihood, alpha chosen with CAL log response loss.
    cnt=np.zeros(27);lc=np.zeros((3,27,6))
    for r in tr:
        cnt[r['u']]+=1
        for a,z in enumerate(r['z1']):lc[a,r['u'],z]+=1
    prior=(cnt+.5)/(len(tr)+13.5);gl=(lc+.5)/(lc+.5).sum(2,keepdims=True)
    models=[]
    for b in range(8):
        rr=[r for r in tr if r['b']==b];pr=8*prior;L=8*gl
        pr=pr.copy();L=L.copy()
        for r in rr:
            pr[r['u']]+=1
            for a,z in enumerate(r['z1']):L[a,r['u'],z]+=1
        pr/=pr.sum();L/=L.sum(2,keepdims=True)
        models.append(dict(pr=pr.tolist(),L=L.tolist(),marg=np.einsum('u,auz->az',pr,L).tolist()))
    alphas=[0,.25,.5,.75,1.]
    losses=[]
    for a in range(3):
        losses.append([S.mean(-math.log((1-al)*models[r['b']]['marg'][a][r['z1'][a]]+al*models[r['b']]['L'][a][r['u']][r['z1'][a]]) for r in cal) for al in alphas])
    alpha=[alphas[int(np.argmin(z))] for z in losses]
    save('FROZEN.json',dict(cohort=sha(D/'COHORT.json'),trace=sha(D/'TRAIN_CAL_ATLAS.jsonl.gz'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'h2.py'),identity=ident(),train_ids=[r['id'] for r in tr],cal_ids=[r['id'] for r in cal],tables=tables,calnet=calnet,selected=selected,best_static=NAMES[bst],q1=q1,q2=q2,anchors=anchors,models=models,alpha=alpha,cal_channel_loss=losses))
    print('FIT',len(tr),len(cal),'static',NAMES[bst],'selected',selected,'alpha',alpha,'CAL',calnet,'adaptive',adap,flush=True)
def dr(r,f,k):
    t=f['tables'][k];q1=f['q1'][k][r['b']];a=t['first'][r['b']];ans=np.full(9,q1[a])
    for i,j in PTH:
        b=t['second'][i][r['mid'][i]];q2=f['q2'][i][r['mid'][i]]
        ans[3*i+j]+=3*int(i==a)*(q2[b]-q1[i])+9*int(i==a and j==b)*(r['y'][3*i+j]-q2[j])
    return ans
def ope(rows,f,k):
    t=f['tables'][k];n=len(rows);values={name:[] for name in ('raw','proxy','learned','prefix_offline','oracle_offline','sequential_DR')};v={key:0. for key in values};truth=[];between=within=0.;disagree=0;gaps=[]
    for r in rows:
        c=pick(r,t);d=np.eye(9)[c]-np.eye(9)[0];y=np.array(r['y']);tar=d@y;truth.append(float(tar));disagree+=int(c!=0)
        opt=(y[c]+y[0])/2
        scores={name:9*d*(y-m) for name,m in [('raw',0),('proxy',r['proxy']),('learned',f['anchors'][k][r['b']]),('prefix_offline',r['pre']),('oracle_offline',opt)]}
        scores['sequential_DR']=dr(r,f,k)-dr(r,f,'NN')
        for name,x in scores.items():
            near(x.mean(),tar);v[name]+=float(x.var());values[name].append(float(x[r['logged']]))
        a=scores['sequential_DR'].reshape(3,3);between+=float(a.mean(1).var());within+=float(a.var(1).mean())
        near(scores['learned'].var()-scores['oracle_offline'].var(),18*int(c!=0)*(f['anchors'][k][r['b']]-opt)**2)
        gaps.append(float(tar))
    near(v['sequential_DR'],between+within)
    return dict(full=S.mean(truth),disagreement=disagree,actual={k:S.mean(a) for k,a in values.items()},sd={k:math.sqrt(x)/n for k,x in v.items()},DR_between=between/n**2,DR_within=within/n**2)
def agg(rows,t):
    vals=[(r,pick(r,t)) for r in rows];u=S.mean(r['us'][c] for r,c in vals)
    return dict(utility=u,regret=S.mean(r['full'] for r in rows)-u,net=S.mean(r['y'][c] for r,c in vals),NN=S.mean(sum(r['n'][c]) for r,c in vals),visits=S.mean(r['vis'][c] for r,c in vals),cpu=S.mean(r['cpu'][c] for r,c in vals),wall=S.mean(r['wall'][c] for r,c in vals),paths=dict(C.Counter(NAMES[c] for r,c in vals)))
def pair(rows,t):
    if not rows:return dict(n=0)
    x=[r['us'][pick(r,t)]-r['us'][0] for r in rows];rng=random.Random(SEED+17)
    bs=sorted(S.mean(rng.choices(x,k=len(x))) for _ in range(2000))
    return dict(n=len(x),gain=S.mean(x),better=sum(v>0 for v in x),worse=sum(v<0 for v in x),ci=[bs[49],bs[1949]])
def mi_counts(x,y):
    p=C.Counter(zip(x,y));a=C.Counter(x);b=C.Counter(y);n=len(x)
    return sum(v/n*math.log(v*n/(a[i]*b[j])) for (i,j),v in p.items())
def score():
    f=read('FROZEN.json');assert read('TEST_LOCK.json')['freeze']==sha(D/'FROZEN.json')
    rows=data('TEST');assert rows;tables=f['tables'];matched=[r for r in rows if all(x==[6,6] for x in r['n'])]
    result=dict(n=len(rows),selected=f['selected'],matched_n=len(matched),full={k:agg(rows,t) for k,t in tables.items()},pairs={k:pair(rows,t) for k,t in tables.items()},matched_pairs={k:pair(matched,t) for k,t in tables.items()},ope={k:ope(rows,f,k) for k in tables},log_counts=dict(C.Counter(NAMES[r['logged']] for r in rows)))
    inter={}
    for a,b in PTH:
        x=[r['us'][3*a+b]-r['us'][3*a]-r['us'][b]+r['us'][0] for r in matched]
        inter[NAMES[3*a+b]]=dict(mean=S.mean(x) if x else None,positive=sum(v>0 for v in x),negative=sum(v<0 for v in x))
    result['interaction_matched']=inter
    # predictive channel shrinkage, latent posterior coherence and model EVSI
    cs=[]
    for a,al in enumerate(f['alpha']):
        metrics=[]
        for r in rows:
            m=f['models'][r['b']];pr=np.array(m['pr']);L=np.array(m['L'][a]);marg=np.array(m['marg'][a]);Q=(1-al)*marg[None,:]+al*L
            near(pr@Q,marg);J=pr[:,None]*Q
            mi=float(np.sum(J*np.log(Q/marg[None,:])));g=float(np.max(J.T@U,axis=1).sum()-np.max(pr@U))
            assert -1e-10<=g<=math.sqrt(max(0,mi)/2)+1e-10
            u,z=r['u'],r['z1'][a];metrics.append([-math.log(marg[z]),-math.log(L[u,z]),-math.log(Q[u,z]),mi,g])
        cs.append(dict(arm='NFX'[a],alpha=al,**dict(zip(['marginal_log','raw_conditional_log','shrink_log','model_MI','model_EVSI'],np.mean(metrics,axis=0).tolist()))))
    result['channel']=cs;chain={}
    for i,(a,b) in enumerate(PTH):
        # within prefix-bin plug-in finite table; chain exact algebra, biased population estimates
        first=second=joint=0.
        for h in range(8):
            rr=[r for r in rows if r['b']==h]
            if not rr:continue
            u=[r['u'] for r in rr];z1=[r['z1'][a] for r in rr];z2=[r['z2'][i] for r in rr];w=len(rr)/len(rows)
            first+=w*mi_counts(u,z1);joint+=w*mi_counts(u,list(zip(z1,z2)))
            for z in set(z1):
                q=[r for r in rr if r['z1'][a]==z];second+=len(q)/len(rows)*mi_counts([r['u'] for r in q],[r['z2'][i] for r in q])
        near(first+second,joint);chain[NAMES[i]]=[first,second,joint]
    result['plugin_chain']=chain;result['rows']=rows
    save('SUMMARY.json',result)
    print('SUMMARY',__import__('json').dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('trial','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
