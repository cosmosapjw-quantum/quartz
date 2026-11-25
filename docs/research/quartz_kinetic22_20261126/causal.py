"""K22: prospective paired native computation interventions and randomized sparse policy evaluation."""
from source_gen import D,ROOT,R,BASE,SEED,PRICE,sha,read,save,raw,emit,ident,k20,p,old,BIN
import math,random,itertools,statistics as S,collections as C,gzip,sys
ARMS=('K22N','K22F','K22X')
U=list(itertools.product((0.,.5,1.),repeat=3))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)
def phi(x,t=.25):
    m=max(x);z=[math.exp((v-m)/t) for v in x];s=sum(z)
    return m+t*math.log(s/len(x)),[a/s for a in z]
def kl(p,q):return sum(a*math.log(a/b) for a,b in zip(p,q) if a)
def scores(s):return [(1+w/(n+4))/2 for n,w in zip(s['n'][:s['len']],s['w'][:s['len']])]
def check(a,r):
    assert 'error' not in a,a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['best'] in r['legal'] and a['proof']==k20.proof(r['history'])
    pre,pulse,end=a['states'];assert pre['visits']==8 and end['visits']==a['visits']==a['result']['root_n']
    assert pre['nn']+a['incremental_nn']==a['NN_calls']==end['nn']==a['result']['evaluator_calls']
    assert a['goal_nn']==pre['nn']+12 and 0<=a['incremental_nn']<=12 and a['visits']<=128
    assert pre['moves']==pulse['moves']==end['moves']
    n=pre['n'];moves=pre['moves'];i=moves[:pre['len']].index(pre['best'])
    j=max((k for k in range(pre['len']) if k!=i),key=lambda k:(n[k],-k));k=next(k for k in range(pre['len']) if k not in (i,j))
    assert a['panel']==[moves[x] for x in (i,j,k)]
    cert=k20.cert(a['proof']);assert a['best']==(end['best'] if cert is None else cert)
    if cert is None:
        assert pulse['visits']==12 and a['cause']==('NN_TARGET' if a['incremental_nn']==12 else 'VISIT_CAP')
        if a['chosen']:
            assert a['forced']==a['panel'][a['chosen']]
            ix=moves[:pre['len']].index(a['forced']);assert pulse['n'][ix]-pre['n'][ix]==4
        else:assert a['forced'] is None
        if a['cause']=='VISIT_CAP':assert a['visits']==128
    else:assert a['cause']=='CERTIFIED' and a['incremental_nn']==0
    for s in (pre,pulse,end):
        nn=s['n'][:s['len']];ww=s['w'][:s['len']]
        assert sum(nn)==s['visits'] and set(s['moves'][:s['len']])==set(r['legal'])
        assert all(abs(w)<=n+1e-6 for n,w in zip(nn,ww))
        x=scores(s);y=[.5]*s['len'] if s is pre else scores(pre)
        f0,q=phi(y);f1,v=phi(x);kld=kl(q,v);dr=sum(qi*(xi-yi) for qi,xi,yi in zip(q,x,y))
        close(s['kl'],kld);close(s['drift'],dr);close(s['phi'],f1);close(s['bregman'],f1-f0-dr);close(s['bregman'],.25*kld)
        cnt=sorted(nn,reverse=True);assert s['bin']==4*int(cnt[0]/s['visits']>=.75)+2*int(cnt[0]-cnt[1]>=2)+int(kld>=.02)
    assert all(x<=y<=z for x,y,z in zip(pre['n'],pulse['n'],end['n']))
    es={e['move']:e for e in a['result']['edges']};assert end['best']==a['result']['best']
    for i,m in enumerate(end['moves'][:end['len']]):
        e=es.get(m,{});assert end['n'][i]==e.get('n',0);close(end['w'][i],e.get('w',0));close(end['m2'][i],e.get('m2',0))
def order(i):
    z=list(range(3));random.Random(SEED+900000+i).shuffle(z);return z
def trial(part):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    if part=='TEST':
        f=read('FROZEN.json');assert f['code']==sha(D/'causal.py') and f['cohort']==sha(D/'COHORT.json') and f['identity']==ident()
    path=D/(part+'_ATLAS.jsonl.gz');assert not path.exists()
    cnt=vis=nn=0;caps=C.Counter();parity=0
    with k20.Server(BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            if (r['group']=='TEST')!=(part=='TEST'):continue
            seq=order(r['id']);arms=[None]*3
            for i in seq:
                a=srv.request(r['history'],ARMS[i]);check(a,r);assert a['chosen']==i
                arms[i]=a;cnt+=1;vis+=a['visits'];nn+=a['NN_calls'];caps[a['cause']]+=1
            assert all(x['states'][0]==arms[0]['states'][0] and x['panel']==arms[0]['panel'] for x in arms)
            t=dict(root=r,order=seq,logged=seq[0],propensity=[1/3]*3,arms=arms)
            if part=='TEST':
                table=f['tables'][f['selected']];a=srv.request(r['history'],'K22LIVE',causal_table=table);check(a,r)
                i=table[a['states'][0]['bin']];b=arms[i]
                assert a['chosen']==i and all(a[k]==b[k] for k in ('states','best','visits','panel','incremental_nn','cause','result'))
                t['live']=a;parity+=1;cnt+=1;vis+=a['visits'];nn+=a['NN_calls']
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
    ll={x['root']['id']:x for x in lab['rows'] if x['status']=='EXACT'}
    rows=[]
    for t in raw(D/(part+'_ATLAS.jsonl.gz')):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u={int(k):(v+1)/2 for k,v in ll[i]['independent'].items()}
        pre=t['arms'][0]['states'][0];panel=t['arms'][0]['panel'];z=[];out=[];n=[];us=[];geo=[];dr=[]
        for a in t['arms']:
            assert all(lo<=u[m]<=hi for m,lo,hi in a['proof'])
            s=a['states'][-1];x=scores(s);q=[x[s['moves'][:s['len']].index(m)] for m in panel]
            z.append(2*max(range(3),key=q.__getitem__)+int(s['kl']>=.02))
            us.append(u[a['best']]);n.append(a['incremental_nn']);out.append(us[-1]-PRICE*n[-1])
            geo.append(.25*s['kl']);dr.append(s['drift'])
        rows.append(dict(id=i,group=r['group'],family=r['family'],b=pre['bin'],panel=panel,u=U.index(tuple(u[m] for m in panel)),y=out,us=us,n=n,pre_u=u[pre['best']],full=max(u.values()),z=z,logged=t['logged'],geo=geo,drift=dr,cpu=[a['cpu_ns']/1e6 for a in t['arms']],wall=[a['wall_ns']/1e6 for a in t['arms']],vis=[a['visits'] for a in t['arms']],third_visited=pre['n'][pre['moves'][:pre['len']].index(panel[2])]>0,third_visible=panel[2] in pre['moves'][:pre['len']]))
    return rows
def normalize(v):
    z=sum(v);assert z>0;return [x/z for x in v]
def channel(pr,L,t=.25):
    J=[[p*x for x in l] for p,l in zip(pr,L)]
    pz=[sum(row[z] for row in J) for z in range(6)];m=[sum(p*u[a] for p,u in zip(pr,U)) for a in range(3)]
    post=[[sum(row[z]*u[a] for row,u in zip(J,U))/pz[z] for a in range(3)] for z in range(6)]
    hard=sum(p*max(q) for p,q in zip(pz,post))-max(m)
    mi=sum(x*math.log(x/(p*pz[z])) for p,row in zip(pr,J) for z,x in enumerate(row) if x)
    coarse=[[row[2*z]+row[2*z+1] for z in range(3)] for row in J];pc=[pz[2*z]+pz[2*z+1] for z in range(3)]
    cm=sum(x*math.log(x/(p*pc[z])) for p,row in zip(pr,coarse) for z,x in enumerate(row) if x)
    hard_s=sum(max(sum(row[z]*u[a] for row,u in zip(coarse,U)) for a in range(3)) for z in range(3))-max(m)
    loss=hard-hard_s;f,q=phi(m,t);smooth=0.;geom=0.
    for p,mp in zip(pz,post):
        f1,q1=phi(mp,t);smooth+=p*(f1-f);geom+=p*t*kl(q,q1)
    close(smooth,geom);assert -1e-11<=loss<=math.sqrt(max(0,mi-cm)/2)+1e-10
    a=max(range(3),key=m.__getitem__)
    var=[sum(p*((mp[b]-m[b])-(mp[a]-m[a]))**2 for p,mp in zip(pz,post)) for b in range(3)]
    gap=[m[a]-v for v in m]
    moment=sum(v/(2*(math.sqrt(d*d+v)+d)) if v or d else 0. for d,v in zip(gap,var))
    evpi=sum(p*max(u) for p,u in zip(pr,U))-max(m)
    bound=min(evpi,moment,math.sqrt(max(0,mi)/2));assert -1e-11<=hard<=bound+1e-10
    return dict(pr=pr,L=L,hard=hard,mi=mi,cmi=mi-cm,loss=loss,coarse=hard_s,evpi=evpi,bound=bound,moment=moment,smooth=smooth,pz=pz,m=m,post=post,response_variance=var)
def fit():
    assert not (D/'TEST_ATLAS.jsonl.gz').exists() and not (D/'TEST_LABELS.json').exists()
    rows=data('TRAIN_CAL');tr=[r for r in rows if r['group']=='TRAIN'];cal=[r for r in rows if r['group']=='CAL']
    assert len(tr)>60 and len(cal)>20
    global_pr=normalize([sum(r['u']==i for r in tr)+.5 for i in range(27)])
    global_l=[[normalize([sum(r['u']==u and r['z'][c]==z for r in tr)+.5 for z in range(6)]) for u in range(27)] for c in range(3)]
    pooled=[S.mean(r['y'][c] for r in tr) for c in range(3)];cost=[S.mean(r['n'][c] for r in tr) for c in range(3)]
    tables={m:[] for m in ('normal','direct','evsi','mi')};mu=[];mods=[];counts=[]
    for b in range(8):
        rr=[r for r in tr if r['b']==b];counts.append(len(rr))
        pr=normalize([sum(r['u']==u for r in rr)+8*global_pr[u] for u in range(27)])
        mm=[(sum(r['y'][c] for r in rr)+8*pooled[c])/(len(rr)+8) for c in range(3)]
        cc=[(sum(r['n'][c] for r in rr)+8*cost[c])/(len(rr)+8) for c in range(3)]
        mu.append(mm);chs=[]
        for c in range(3):
            L=[normalize([sum(r['u']==u and r['z'][c]==z for r in rr)+8*global_l[c][u][z] for z in range(6)]) for u in range(27)]
            z=channel(pr,L);z['cost']=cc[c];chs.append(z)
        mods.append(chs)
        tables['normal'].append(0);tables['direct'].append(max(range(3),key=mm.__getitem__))
        tables['evsi'].append(max(range(3),key=lambda c:chs[c]['hard']-PRICE*cc[c]));tables['mi'].append(max(range(3),key=lambda c:chs[c]['mi']))
    calnet={m:S.mean(r['y'][table[r['b']]] for r in cal) for m,table in tables.items()}
    selected=max(('normal','direct','evsi'),key=calnet.__getitem__)
    save('FROZEN.json',dict(cohort=sha(D/'COHORT.json'),trace=sha(D/'TRAIN_CAL_ATLAS.jsonl.gz'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'causal.py'),identity=ident(),train_ids=[r['id'] for r in tr],cal_ids=[r['id'] for r in cal],counts=counts,price=PRICE,tables=tables,mu=mu,models=mods,calnet=calnet,selected=selected))
    print('FIT',len(tr),len(cal),'counts',counts,'calnet',calnet,'selected',selected,'tables',tables,flush=True)
def ope(rows,f,tab,baseline=None):
    # Conditional finite-population assignment moments; nuisance is frozen independently of TEST.
    ips=[];dr=[];truth=[];reg=[];var=[0.,0.];masks=[]
    for r in rows:
        b=r['b'];j=tab[b];k=None if baseline is None else baseline[b]
        d=[int(i==j)-int(i==k) for i in range(3)];y=r['y'];m=f['mu'][b];base=sum(a*b for a,b in zip(d,m));tar=sum(a*b for a,b in zip(d,y))
        ip=[3*d[c]*y[c] for c in range(3)];dd=[base+3*d[c]*(y[c]-m[c]) for c in range(3)]
        close(S.mean(ip),tar);close(S.mean(dd),tar)
        truth.append(tar);reg.append(base);ips.append(ip[r['logged']]);dr.append(dd[r['logged']]);var[0]+=S.mean((x-tar)**2 for x in ip);var[1]+=S.mean((x-tar)**2 for x in dd);masks.append((ip,dd))
    n=len(rows);t=S.mean(truth);rng=random.Random(SEED+61);sq=[0.,0.]
    for _ in range(2000):
        v=[0.,0.]
        for a,b in masks:
            c=rng.randrange(3);v[0]+=a[c]/n;v[1]+=b[c]/n
        for j in range(2):sq[j]+=(v[j]-t)**2
    return dict(full=t,DM=S.mean(reg),IPS=S.mean(ips),DR=S.mean(dr),IPS_sd=math.sqrt(var[0])/n,DR_sd=math.sqrt(var[1])/n,mask_IPS_rmse=math.sqrt(sq[0]/2000),mask_DR_rmse=math.sqrt(sq[1]/2000),disagreement=sum(1 for r in rows if baseline is None or tab[r['b']]!=baseline[r['b']]))
def aggregate(rows,tab):
    def avg(k):return S.mean(r[k][tab[r['b']]] for r in rows)
    u=avg('us');return dict(utility=u,regret=S.mean(r['full'] for r in rows)-u,net=avg('y'),NN_increment=avg('n'),visits=avg('vis'),CPU_ms=avg('cpu'),wall_ms=avg('wall'),choices=dict(C.Counter(str(tab[r['b']]) for r in rows)),gain_vs_prefix=S.mean(r['us'][tab[r['b']]]-r['pre_u'] for r in rows))
def pair(rows,t1,t0):
    a=[r['us'][t1[r['b']]]-r['us'][t0[r['b']]] for r in rows];rng=random.Random(SEED+1)
    if not a:return dict(n=0)
    bs=sorted(S.mean(rng.choices(a,k=len(a))) for _ in range(4000))
    return dict(n=len(a),gain=S.mean(a),better=sum(x>0 for x in a),worse=sum(x<0 for x in a),ci=[bs[99],bs[3899]])
def score():
    f=read('FROZEN.json');lk=read('TEST_LOCK.json');assert lk['freeze']==sha(D/'FROZEN.json')
    rows=data('TEST');assert rows
    tables=dict(f['tables'],F=[1]*8,X=[2]*8)
    selected=f['selected'];full={m:aggregate(rows,t) for m,t in tables.items()}
    matched=[r for r in rows if r['n']==[12]*3]
    pairs={m:pair(rows,t,tables['normal']) for m,t in tables.items()}
    mp={m:pair(matched,t,tables['normal']) for m,t in tables.items()}
    estimates={m:dict(value=ope(rows,f,t),contrast=ope(rows,f,t,tables['normal'])) for m,t in tables.items()}
    # Held-out model scores and model-vs-realized panel decision value.
    pred=[]
    for r in rows:
        for c,ch in enumerate(f['models'][r['b']]):
            z=r['z'][c];u=r['u'];m=ch['m'];post=ch['post'][z];act=max(range(3),key=post.__getitem__);a=max(range(3),key=m.__getitem__)
            pred.append(dict(c=c,hard=ch['hard'],mi=ch['mi'],cmi=ch['cmi'],loss=ch['loss'],bound=ch['bound'],smooth=ch['smooth'],real_panel_gain=U[u][act]-U[u][a],native_gain=r['us'][c]-r['pre_u'],log_joint=-math.log(ch['pr'][u]*ch['L'][u][z]),log_response=-math.log(ch['L'][u][z]),log_marginal=-math.log(ch['pz'][z]),raw_geo=r['geo'][c],raw_drift=r['drift'][c]))
    model={ARMS[c]:{k:S.mean(x[k] for x in pred if x['c']==c) for k in pred[0] if k!='c'} for c in range(3)}
    out=dict(n=len(rows),selected=selected,all_roots=len(read('TEST_LABELS.json')['rows']),full=full,pairs=pairs,matched_n=len(matched),matched_pairs=mp,estimates=estimates,model=model,logged_counts=dict(C.Counter(str(r['logged']) for r in rows)),third_visited=sum(r['third_visited'] for r in rows),third_visible=sum(r['third_visible'] for r in rows),rows=rows)
    save('SUMMARY.json',out);print('SUMMARY',__import__('json').dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('trial','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
