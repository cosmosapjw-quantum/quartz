"""Fixed readouts, separate acquisition and recommendation effects, family statistics."""
from pair_run import *
from fractions import Fraction as F
def stats(v):
    return dict(n=len(v),mean=S.mean(v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v)) if v else dict(n=0,mean=None,positive=0,negative=0,zero=0)
def ci(v):
    if not v:return None
    rng=random.Random(2026112616);z=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(5000));return [z[125],z[4874]]
def pred(a,kind):
    e=a['evidence']
    if kind=='NATIVE':return a['best']
    if kind=='PREFIX_VOTE':return e['pair'][0]
    return e[kind.lower()+'_best']
def summary():
    raw=atlas_rows();co=read('C06_COHORT.json');pan=read('C07_PANEL.json');f=read('C08_FROZEN.json')
    assert f['atlas_sha256']==sha(D/'C07_ATLAS.jsonl.gz') and f['train_sha256']==sha(D/'C08_TRAIN_LABELS.jsonl.gz')
    train,m1=label_index('TRAIN');test,m2=label_index('HELDOUT');assert not train.keys()&test.keys();u={**train,**test}
    aa={(x['root']['id'],x['cap'],x['arm']):x['response'] for x in raw};assert len(aa)==len(raw)==6*len(co['roots'])
    lab=[z for part in ('TRAIN','HELDOUT') for z in rows('C08_'+part+'_LABELS.jsonl.gz')]
    gg=[z for z in lab if z['root']['domain']=='GOMOKU'];src=list(rows('C06_GOMOKU_SOURCE.jsonl.gz'))
    pol={(x['root']['id'],x['action'],x['arm']):x['utility'] for x in gg}
    out=dict(identity=ident(),families=len(co['roots']),atlas_requests=len(raw),atlas_visits=sum(x['response']['visits'] for x in raw),atlas_NN=sum(x['response']['NN_calls'] for x in raw),forced_visits=sum(sum(sum(t['delta']) for t in x['response']['evidence']['phases'] if t['target'] is not None) for x in raw),source_games=len(src),source_plies=sum(len(x['rows']) for x in src),source_NN=sum(y['response']['NN_calls'] for x in src for y in x['rows']),gomoku_continuations=len(gg),continuation_plies=sum(len(x['rows']) for x in gg),continuation_NN=sum(y['response']['NN_calls'] for x in gg for y in x['rows']),gomoku_panel_actions=len(pol)//2,label_disagreements=sum(v!=pol[i,a,'STANDARD_FULL'] for (i,a,k),v in pol.items() if k=='STANDARD_PW'),c4_attempts=len(co['c4_attempts']),c4_oracle_status=dict(C.Counter(z['status'] for z in lab if z['root']['domain']=='C4')),missing=m1+m2,gates=f['gates'],fitted_domains=list(f['models']),groups={})
    for dom in ('C4','GOMOKU'):
        out['groups'][dom]={}
        for grp in ('TRAIN','CAL','TEST'):
            ids=sorted(r['id'] for r in co['roots'] if r['domain']==dom and r['group']==grp and r['id'] in u);assert ids
            ps={i:aa[i,32,'K16N']['evidence']['pair'] for i in ids}
            dv=[u[i][b]-u[i][a] for i,(a,b) in ps.items()]
            omission=[max(u[i].values())-max(u[i][a],u[i][b]) for i,(a,b) in ps.items()]
            g=dict(n=len(ids),ids=ids,pair_gap=stats(dv),pair_omission=stats(omission),by_cap={})
            for cap in CAPS:
                base=[aa[i,cap,'K16N'] for i in ids];baseU=[u[i][a['best']] for i,a in zip(ids,base,strict=True)]
                z=dict(arms={},order={},acquisition={},pairs={})
                for arm in ARMS:
                    arr=[aa[i,cap,arm] for i in ids];d=dict(cpu_ms=S.mean(map(cpu,arr)),wall_ms=S.mean(a['transport_ns']/1e6 for a in arr),NN=S.mean(a['NN_calls'] for a in arr),readouts={},positive_KL=sum(t['flux'][4]>1e-12 for a in arr for t in a['evidence']['phases']),negative_KL=sum(t['flux'][4]<-1e-12 for a in arr for t in a['evidence']['phases']),forced=sum(sum(sum(t['delta']) for t in a['evidence']['phases'] if t['target'] is not None) for a in arr))
                    for kind in ('NATIVE','PREFIX_VOTE','PREFIX','PULSE','FINAL'):
                        moves=[pred(a,kind) for a in arr];vv=[u[i][m] for i,m in zip(ids,moves,strict=True)]
                        regret=[max(u[i].values())-v for i,v in zip(ids,vv,strict=True)]
                        gain=[v-b for v,b in zip(vv,baseU,strict=True)]
                        x=dict(mean_U=S.mean(vv),regret=stats(regret),gain=stats(gain),gain_CI95=ci(gain),changed_vs_standard=sum(m!=a['best'] for m,a in zip(moves,base,strict=True)),changed_vs_prefix=sum(m!=ps[i][0] for i,m in zip(ids,moves,strict=True)))
                        if kind!='NATIVE':
                            loss=[max(u[i][ps[i][0]],u[i][ps[i][1]])-v for i,v in zip(ids,vv,strict=True)]
                            assert all(abs(r-l-o)<1e-12 for r,l,o in zip(regret,loss,omission,strict=True));x['pair_regret']=stats(loss)
                        if kind=='PULSE':
                            signed=[v-u[i][pred(a,'PREFIX')] for i,v,a in zip(ids,vv,arr,strict=True)];x['gain_vs_prefix_score']=stats(signed)
                            p=[a['evidence']['pulse_score'][1]-a['evidence']['pulse_score'][0] for a in arr]
                            x['gap_MSE']=S.mean((v/2-d0)**2 for v,d0 in zip(p,dv,strict=True));x['zero_gap_MSE']=S.mean(d0*d0 for d0 in dv)
                        if dom=='GOMOKU':
                            x['gain_by_policy']={k:stats([pol[i,m,k]-pol[i,a['best'],k] for i,m,a in zip(ids,moves,base,strict=True)]) for k in ('STANDARD_PW','STANDARD_FULL')}
                        d['readouts'][kind]=x
                    z['arms'][arm]=d
                for kind in ('NATIVE','PULSE','FINAL'):
                    x=[aa[i,cap,'K16BAL'] for i in ids];y=[aa[i,cap,'K16REV'] for i in ids];n0=[aa[i,cap,'K16N'] for i in ids]
                    z['order'][kind]=dict(move_changes=sum(pred(a,kind)!=pred(b,kind) for a,b in zip(x,y,strict=True)),gain=stats([u[i][pred(a,kind)]-u[i][pred(b,kind)] for i,a,b in zip(ids,x,y,strict=True)]))
                    for arm,v in [('BAL',x),('REV',y)]:
                        gain=[u[i][pred(a,kind)]-u[i][pred(b,kind)] for i,a,b in zip(ids,v,n0,strict=True)]
                        delta=[cpu(a)-cpu(b) for a,b in zip(v,n0,strict=True)]
                        z['acquisition'][arm+'_'+kind]=dict(gain=stats(gain),gain_CI95=ci(gain),delta_CPU_ms=stats(delta),net=stats([x-.002*y for x,y in zip(gain,delta,strict=True)]))
                z['order']['state_changes']=sum(aa[i,cap,'K16BAL']['result']!=aa[i,cap,'K16REV']['result'] for i in ids)
                z['pairs']={i:dict(pair=ps[i],D=d,utility={str(m):u[i][m] for m in pan['panels'][i]['moves']},readouts={a:{k:pred(aa[i,cap,a],k) for k in ('NATIVE','PREFIX_VOTE','PREFIX','PULSE','FINAL')} for a in ARMS}) for i,d in zip(ids,dv,strict=True)}
                # No learned predictor if the frozen gate fails; deliberately do not
                # compute a post-hoc conditional fit from these heldout outcomes.
                z['model_status']='NOT_FIT_EXPOSURE_GATE' if dom not in f['models'] else 'FROZEN_AVAILABLE'
                g['by_cap'][str(cap)]=z
            out['groups'][dom][grp]=g
    return out
if __name__=='__main__':
    s=summary();write('C08_SUMMARY.json',s)
    print('TOTAL', {k:v for k,v in s.items() if k not in ('identity','groups')})
    for dom,groups in s['groups'].items():
        for grp,g in groups.items():
            print(dom,grp,'n',g['n'],'gap',g['pair_gap'],'omission',g['pair_omission'])
            for cap,z in g['by_cap'].items():
                print('CAP',cap,'order',z['order'])
                for a,v in z['arms'].items():
                    print(a,'CPU/wall/NN',v['cpu_ms'],v['wall_ms'],v['NN'],'U/gain/regret',[(k,x['mean_U'],x['gain']['mean'],x['regret']['mean'],x['changed_vs_standard']) for k,x in v['readouts'].items()])
                print('ACQUISITION',[(k,x['gain']['mean'],x['net']['mean']) for k,x in z['acquisition'].items()])
