"""Finite measured schedule responses, not physical thermodynamic production."""
from kinetic_run import *
from itertools import combinations
from math import fsum
mean=S.mean
def utility(u,i,a):return mean(u[i,a,k] for k in (128,512))
def counter(a):return a['kinetic']['steps'][-1]['counters']
def phases(a):return [a['kinetic']['prefix']]+a['kinetic']['steps']
def occ(r):return dict(zip((x[0] for x in r['candidates']),kr.probs(kr.counts(r)),strict=True))
def panel_regret(r,p):
    a=r['best'];v={x['move'] for x in r['edges']};z=max(p.values());h=max(p[m] for m in p.keys()&v)
    assert a in p and a in v
    out=(z-p[a],z-h,h-p[a])
    assert abs(out[0]-out[1]-out[2])<1e-12 and min(out)>=-1e-12
    return out
def bounds(r,s,p):
    x=occ(r);y=occ(s);assert x.keys()==y.keys()
    j={m:y[m]-v for m,v in x.items()};known=fsum(j[m]*p[m] for m in p)
    unk=[v for m,v in j.items() if m not in p]
    return (known+fsum(min(0,v) for v in unk),known+fsum(max(0,v) for v in unk))
def sensitivity():
    """Retrospective only. Gibbs q argmax diagnostic under arbitrary beta/kappa."""
    p=ROOT/'docs/research/quartz_metacog13_20261126/C07_PRIMARY_RAW.jsonl.gz'
    rr=[]
    with gzip.open(p,'rt') as f:
        for l in f:
            a=json.loads(l)['response']
            if a['arm']!='META':continue
            for k,t in enumerate([a['meta']['prefix']]+a['meta']['steps']):
                r=t['result'];n=kr.counts(r);w={x['move']:x['w'] for x in r['edges']}
                rr.append((k,r,n,w,a['root_eval']['value']))
    def q(r,n,w,v,b,k):
        pi=[a[1]+1e-8 for a in r['candidates']]
        z=[math.log(p)+b*(w.get(a[0],0)+k*v)/(n0+k) for p,a,n0 in zip(pi,r['candidates'],n,strict=True)]
        z=[math.exp(a-max(z)) for a in z];s=fsum(z);return [a/s for a in z]
    out=[]
    for b,k in product((.5,1.,4.,16.,32.),(1.,4.,16.)):
        vals=[]
        for stage,r,n,w,v in rr:
            z=q(r,n,w,v,b,k);z0=q(r,n,w,v,4.,4.)
            m=lambda z:r['candidates'][max(range(len(z)),key=lambda i:(z[i],i))][0]
            vals.append((stage,m(z)!=m(z0),m(z)!=r['best'],kr.tv(z,z0),fsum(z[len(r['edges']):])))
        out.append(dict(beta=b,kappa=k,states=len(vals),argmax_changed=sum(x[1] for x in vals),final_argmax_changed=sum(x[1] for x in vals if x[0]==15),argmax_disagrees_native=sum(x[2] for x in vals),mean_TV_from_reference=mean(x[3] for x in vals),mean_hidden_mass=mean(x[4] for x in vals)))
    return out
def main():
    aa=atlas_rows();live=list(rows('C08_LIVE_RAW.jsonl.gz'));games=list(rows('C06_SOURCE_GAMES.jsonl.gz'))
    labs=[g for name in ('TRAIN','HELDOUT') for g in rows('C08_'+name+'_LABELS.jsonl.gz')]
    for name,file in [('LIVE','C08_LIVE_RAW.jsonl.gz'),('TRAIN','C08_TRAIN_LABELS.jsonl.gz'),('HELDOUT','C08_HELDOUT_LABELS.jsonl.gz')]:assert sha(D/file)==read('C08_'+name+'_LOCK.json')['raw_sha256']
    u={(g['root']['id'],g['action'],g['cap']):g['utility'] for g in labs};assert len(u)==len(labs)
    co=read('C06_COHORT.json');pan=read('C07_PANEL.json')['panels'];model=read('C08_MODELS.json')
    idx={(x['root']['id'],*x['schedule']):x['response'] for x in aa}
    U=lambda i,c,d:utility(u,i,idx[i,c,d]['best'])
    out=dict(identity=identity(),source_games=len(games),source_plies=sum(len(g['rows']) for g in games),probes=sum(len(g['probes']) for g in games),missing=co['missing'],roots=len(co['roots']),atlas_requests=len(aa),live_requests=len(live),atlas_visits=sum(x['response']['visits'] for x in aa),atlas_NN=sum(x['response']['NN_calls'] for x in aa),continuations=len(labs),continuation_plies=sum(len(g['rows']) for g in labs),continuation_NN=sum(x['response']['NN_calls'] for g in labs for x in g['rows']),panel_moves=sum(len(z['moves']) for z in pan.values()),opposite_strength_labels=sum(u[i,a,128]!=u[i,a,512] for i,a,_ in u if _==128),groups={},historical_Gibbs=sensitivity())
    for group in ('TRAIN','CAL','TEST'):
        ids=[r['id'] for r in co['roots'] if r['group']==group];n=len(ids);v=[x for x in aa if x['root']['id'] in ids];cost=[cpu(x['response']) for x in v]
        steps=[s for x in v for s in x['response']['kinetic']['steps']]
        g=dict(roots=n,strata=dict(C.Counter(r['stratum'] for r in co['roots'] if r['group']==group)),requests=len(v),final_diverse_families=sum(len({idx[i,c,d]['best'] for c,d in PAIRS})>1 for i in ids),distinct_utility_families=sum(len({U(i,c,d) for c,d in PAIRS})>1 for i in ids),panel_utility_diverse_families=sum(len({utility(u,i,a) for a in pan[i]['moves']})>1 for i in ids),label_strength_disagreement=sum(u[i,a,128]!=u[i,a,512] for i in ids for a in pan[i]['moves']),mean_cpu_ms=mean(cost),mean_wall_ms=mean(x['response']['transport_ns']/1e6 for x in v),NN=sum(x['response']['NN_calls'] for x in v),flow_steps=len(steps),KL_increase=sum(s['flux'][4]>1e-12 for s in steps),KL_decrease=sum(s['flux'][4]<-1e-12 for s in steps),KL_unchanged=sum(abs(s['flux'][4])<=1e-12 for s in steps),mean_activity=mean(s['flux'][1] for s in steps),mean_mixing=mean(s['flux'][3] for s in steps),counters={k:sum(counter(x['response'])[k] for x in v) for k in ('merge_blends','backup_changed','select_changed','width_mass_added','extra_evals','proofs')},schedules=[],live=[],tilt=[],prediction=[])
        order=[(i,c,d) for i in ids for c,d in combinations(range(5),2)]
        om=[U(i,c,d)-U(i,d,c) for i,c,d in order]
        inter=[U(i,c,d)-U(i,c,0)-U(i,0,d)+U(i,0,0) for i in ids for c,d in product(range(1,5),repeat=2)]
        g['order']=dict(pairs=len(order),changed_best=sum(idx[i,c,d]['best']!=idx[i,d,c]['best'] for i,c,d in order),different_final_occupation=sum(kr.counts(idx[i,c,d]['result'])!=kr.counts(idx[i,d,c]['result']) for i,c,d in order),different_final_snapshot=sum(idx[i,c,d]['result']!=idx[i,d,c]['result'] for i,c,d in order),nonzero_utility=sum(x!=0 for x in om),mean_abs_utility=mean(map(abs,om)),mean_abs_CPU_difference=mean(abs(cpu(idx[i,c,d])-cpu(idx[i,d,c])) for i,c,d in order))
        g['interaction']=dict(count=len(inter),nonzero=sum(x!=0 for x in inter),mean_abs=mean(map(abs,inter)))
        for c,d in PAIRS:
            a=[idx[i,c,d] for i in ids];b=[idx[i,0,0] for i in ids];du=[U(i,c,d)-U(i,0,0) for i in ids];dc=[cpu(x)-cpu(y) for x,y in zip(a,b)]
            g['schedules'].append(dict(schedule=[c,d],mean_U=mean(U(i,c,d) for i in ids),delta_U=mean(du),delta_U128=mean(u[i,x['best'],128]-u[i,y['best'],128] for i,x,y in zip(ids,a,b)),delta_U512=mean(u[i,x['best'],512]-u[i,y['best'],512] for i,x,y in zip(ids,a,b)),improved=sum(z>0 for z in du),worse=sum(z<0 for z in du),different_best=sum(x['best']!=y['best'] for x,y in zip(a,b)),cpu_ms=mean(map(cpu,a)),wall_ms=mean(x['transport_ns']/1e6 for x in a),delta_cpu_ms=mean(dc),net=mean(z-.002*t for z,t in zip(du,dc))))
        for k in range(5):
            ds=[];finite=[];err=[];ent=[]
            for i in ids:
                f=[int(c==k)+int(d==k) for c,d in PAIRS];y=[U(i,c,d) for c,d in PAIRS]
                def ev(e):
                    z=[math.exp(e*t) for t in f];p=[v/fsum(z) for v in z]
                    kl=fsum(a*math.log(a*25) for a in p);assert abs(kl-(e*fsum(a*b for a,b in zip(p,f))-math.log(fsum(z)/25)))<1e-12
                    return fsum(a*b for a,b in zip(p,y)),kl
                der=mean(a*b for a,b in zip(y,f))-mean(y)*mean(f);ds.append(der);err.append(abs((ev(1e-4)[0]-ev(-1e-4)[0])/2e-4-der))
                finite.append([ev(e)[0]-mean(y) for e in (-1.,-.1,.1,1.)]);ent.append(ev(1.)[1])
            g['tilt'].append(dict(channel=k,mean_derivative=mean(ds),nonzero_derivative=sum(abs(x)>1e-12 for x in ds),finite_delta=[mean(x[j] for x in finite) for j in range(4)],max_derivative_error=max(err),KL_e1=mean(ent)))
        regs=[];bs=[];offset=0;rt=0
        for x in v:
            i=x['root']['id'];p={m:utility(u,i,m) for m in pan[i]['moves']};r=[q['result'] for q in phases(x['response'])]
            # intermediate 80 visits not always in frozen panel; do not create a label.
            rr=[panel_regret(r[j],p) for j in (0,1,3)]
            regs.append(rr);offset+=rr[2][1]<rr[0][1] and rr[2][2]>rr[0][2]
            for a,b in zip(r,r[1:]):bs.append(bounds(a,b,p))
        g['panel']=dict(mean_regret_omission_ranking=[[mean(t[j][k] for t in regs) for k in range(3)] for j in range(3)],stages=[16,48,96],offset_cases=offset,flows=len(bs),positive_bounds=sum(a>1e-12 for a,b in bs),negative_bounds=sum(b<-1e-12 for a,b in bs),mean_width=mean(b-a for a,b in bs),max_width=max(b-a for a,b in bs))
        for arm in ARMS:
            k=ARMS.index(arm);m=model['models'][arm];a=[x for x in live if x['root']['id'] in ids and x['response']['arm']==arm]
            if a:
                du=[utility(u,x['root']['id'],x['response']['best'])-U(x['root']['id'],0,0) for x in a];dc=[cpu(x['response'])-cpu(idx[x['root']['id'],0,0]) for x in a]
                g['live'].append(dict(arm=arm,n=len(a),schedules=dict(C.Counter(str(x['schedule']) for x in a)),delta_U=mean(du),delta_U128=mean(u[x['root']['id'],x['response']['best'],128]-u[x['root']['id'],idx[x['root']['id'],0,0]['best'],128] for x in a),delta_U512=mean(u[x['root']['id'],x['response']['best'],512]-u[x['root']['id'],idx[x['root']['id'],0,0]['best'],512] for x in a),cpu_ms=mean(cpu(x['response']) for x in a),base_cpu_ms=mean(cpu(idx[x['root']['id'],0,0]) for x in a),wall_ms=mean(x['response']['transport_ns']/1e6 for x in a),base_wall_ms=mean(idx[x['root']['id'],0,0]['transport_ns']/1e6 for x in a),NN=mean(x['response']['NN_calls'] for x in a),base_NN=mean(idx[x['root']['id'],0,0]['NN_calls'] for x in a),changed=sum(x['response']['best']!=idx[x['root']['id'],0,0]['best'] for x in a),net=mean(z-.002*t for z,t in zip(du,dc))))
            pred=[]
            for stage in ('first','second'):
                z=[]
                for i in ids:
                    for c in range(1,5):
                        if stage=='first':
                            x=idx[i,0,0]['kinetic']['prefix']['features'][k]
                            z.append((fsum(a*b for a,b in zip(x,m['first_gain'][c])),mean(U(i,c,d)-U(i,0,d) for d in range(5))))
                        else:
                            for d in range(5):
                                x=idx[i,d,0]['kinetic']['steps'][0]['features'][k]
                                z.append((fsum(a*b for a,b in zip(x,m['second_gain'][c])),U(i,d,c)-U(i,d,0)))
                pred.append(dict(stage=stage,n=len(z),nonzero_targets=sum(y!=0 for p,y in z),MSE=mean((p-y)**2 for p,y in z),zero_MSE=mean(y*y for p,y in z)))
            g['prediction'].append(dict(arm=arm,stages=pred))
        g['informative_cases']=[dict(id=i,baseline=idx[i,0,0]['best'],actions=[dict(move=m,utility128=u[i,m,128],utility512=u[i,m,512],schedules=[list(z) for z in PAIRS if idx[i,*z]['best']==m]) for m in pan[i]['moves']]) for i in ids if len({idx[i,*z]['best'] for z in PAIRS})>1]
        out['groups'][group]=g
    write('C08_SUMMARY.json',out)
    print('KINETIC_SCORE', {k:v for k,v in out.items() if k not in ('identity','groups','historical_Gibbs')},flush=True)
    for name,g in out['groups'].items():
        print(name,{k:v for k,v in g.items() if k not in ('schedules','tilt','prediction','informative_cases')},flush=True)
        print('TILT',g['tilt'],'CASES',g['informative_cases'],'PREDICTION',g['prediction'],flush=True)
    print('GIBBS_SENSITIVITY',out['historical_Gibbs'],flush=True)
if __name__=='__main__':main()
