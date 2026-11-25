"""Paired descriptive scores; never refit or promote on TEST."""
from response_run import *
import response_ref as ref
from fractions import Fraction as F
from itertools import combinations
def stats(v):
    return dict(n=len(v),mean=S.mean(v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v))
def boot(v):
    rng=random.Random(2026112615);n=len(v)
    z=sorted(S.mean(rng.choices(v,k=n)) for _ in range(5000))
    return [z[125],z[4874]]
def root_utility(u,i,a,kind):
    return U(u,i,a) if kind=='mean' else u[i,a,kind]
def main():
    raw=atlas_rows();src=list(rows('C06_SOURCE_RAW.jsonl.gz'));co=read('C06_COHORT.json')
    assert sha(D/'C06_SOURCE_RAW.jsonl.gz')==co['source_sha256']
    train=label_index('TRAIN');held=label_index('HELDOUT');assert not train.keys()&held.keys()
    u={**train,**held};aa={(x['root']['id'],x['cap'],x['arm']):x['response'] for x in raw};assert len(aa)==len(raw)==len(co['roots'])*20
    labels0=[g for part in ('TRAIN','HELDOUT') for g in rows('C08_'+part+'_LABELS.jsonl.gz')]
    frozen=read('C08_FROZEN.json');assert frozen['atlas_sha256']==sha(D/'C07_ATLAS_RAW.jsonl.gz') and frozen['train_labels_sha256']==sha(D/'C08_TRAIN_LABELS.jsonl.gz')
    screens=[z for g in src for x in g['attempts'] for z in x['screen']];probes=[x['prefix'] for g in src for x in g['attempts']]
    out=dict(identity=identity(),source_games=len(src),source_plies=sum(len(g['rows']) for g in src),source_draws=sum(g['winner_black']==0 for g in src),source_min_max_length=[min(len(g['final_history']) for g in src),max(len(g['final_history']) for g in src)],screen_root_attempts=sum(bool(x['screen']) for g in src for x in g['attempts']),screen_games=len(screens),screen_plies=sum(len(g['rows']) for g in screens),prefix_probes=len(probes),roots=len(co['roots']),missing=co['missing'],atlas_requests=len(raw),atlas_NN=sum(x['response']['NN_calls'] for x in raw),atlas_visits=sum(x['response']['visits'] for x in raw),source_NN=sum(x['response']['NN_calls'] for g in src for x in g['rows']),screen_NN=sum(x['response']['NN_calls'] for g in screens for x in g['rows']),prefix_NN=sum(x['NN_calls'] for x in probes),continuation_NN=sum(x['response']['NN_calls'] for g in labels0 for x in g['rows']),continuations=len(labels0),continuation_plies=sum(len(g['rows']) for g in labels0),panel_actions=len(u)//2,label_disagreement=sum(u[i,a,'STANDARD_PW']!=u[i,a,'STANDARD_FULL'] for i,a,k in u if k=='STANDARD_PW'),gate=frozen['gate'],model_status=frozen['model_status'],groups={})
    for grp in ('TRAIN','CAL','TEST'):
        ids=[r['id'] for r in co['roots'] if r['group']==grp];n=len(ids)
        g=dict(families=n,strata=dict(C.Counter(r['stratum'] for r in co['roots'] if r['group']==grp)),exposure=expo(aa,u,ids),by_cap={},stratified={})
        for st in ('UNFILTERED','SCREEN'):
            it=[r['id'] for r in co['roots'] if r['group']==grp and r['stratum']==st]
            g['stratified'][st]=expo(aa,u,it)
        for cap in CAPS:
            a0=[aa[i,cap,'K15N'] for i in ids];base=[U(u,i,a['best']) for i,a in zip(ids,a0)]
            stop=[U(u,i,aa[i,cap,'K15STOP']['best']) for i in ids]
            d=dict(arms={},policies={},different_final_families=sum(len({aa[i,cap,a]['best'] for a in ARMS})>1 for i in ids),different_sub_families=sum(len({aa[i,cap,a]['exposure']['sub_best'] for a in ARMS})>1 for i in ids),nonzero_utility_families=sum(len({U(u,i,aa[i,cap,a]['best']) for a in ARMS})>1 for i in ids),nonzero_sub_utility_families=sum(len({U(u,i,aa[i,cap,a]['exposure']['sub_best']) for a in ARMS})>1 for i in ids))
            for arm in ARMS:
                z=[aa[i,cap,arm] for i in ids];r=[U(u,i,a['best']) for i,a in zip(ids,z)];s=[U(u,i,a['exposure']['sub_best']) for i,a in zip(ids,z)]
                gain=[x-y for x,y in zip(r,base)];sub=[x-y for x,y in zip(s,base)];dc=[cpu(x)-cpu(y) for x,y in zip(z,a0)]
                raw_sub=[x-y for x,y in zip(r,s)];net=[x-.002*c for x,c in zip(gain,dc)]
                bounds=[];phase=[]
                for i,a in zip(ids,z):
                    e=a['exposure'];counts=ref.kr.counts(a['result']);subcounts=[x-y for x,y in zip(counts,e['forced'])]
                    ca=[x[0] for x in a['result']['candidates']];J=[F(x,sum(counts))-F(y,sum(subcounts)) for x,y in zip(counts,subcounts)]
                    known=F(0);unknown=[]
                    for mv,j in zip(ca,J):
                        if (i,mv,'STANDARD_PW') in u:known+=j*F(U(u,i,mv))
                        else:unknown.append(j)
                    lo=known+sum(min(0,j) for j in unknown);hi=known+sum(max(0,j) for j in unknown)
                    assert -F(sum(e['forced']),sum(counts))<=lo<=hi<=F(sum(e['forced']),sum(counts))
                    bounds.append((float(lo),float(hi)))
                    phase+=e['phases']
                a=dict(mean_utility=S.mean(r),mean_sub_utility=S.mean(s),gain=stats(gain),sub_gain=stats(sub),gain_vs_stop=stats([x-y for x,y in zip(r,stop)]),raw_minus_sub=stats(raw_sub),changed=sum(x['best']!=y['best'] for x,y in zip(z,a0)),sub_changed=sum(x['exposure']['sub_best']!=y['best'] for x,y in zip(z,a0)),raw_sub_changed=sum(x['best']!=x['exposure']['sub_best'] for x in z),cpu_ms=S.mean(map(cpu,z)),wall_ms=S.mean(x['transport_ns']/1e6 for x in z),mean_visits=S.mean(x['visits'] for x in z),mean_NN=S.mean(x['NN_calls'] for x in z),mean_forced=S.mean(sum(x['exposure']['forced']) for x in z),mean_injection_tv=S.mean(x['exposure']['injection_tv'] for x in z),locked=sum(x['exposure']['locked'] for x in z),net=stats(net),gain_CI95=boot(gain),net_CI95=boot(net),bounds=dict(positive=sum(l>0 for l,h in bounds),negative=sum(h<0 for l,h in bounds),mean_width=S.mean(h-l for l,h in bounds)),phase_count=len(phase),positive_KL=sum(x['flux'][4]>1e-12 for x in phase),negative_KL=sum(x['flux'][4]<-1e-12 for x in phase),gain_by_policy={k:stats([u[i,x['best'],k]-u[i,y['best'],k] for i,x,y in zip(ids,z,a0)]) for k in ('STANDARD_PW','STANDARD_FULL')},counters={k:sum((x['exposure']['phases'][-1] if x['exposure']['phases'] else x['exposure']['prefix'])['counters'][k] for x in z) for k in ('select_changed','width_mass_added','extra_evals','proofs','merge_blends')})
                d['arms'][arm]=a
            if frozen['models']:
                for kind,name in enumerate(('RAW','GIBBS','OCCUPATION')):
                    m=frozen['models'][str(cap)][name]
                    chosen=[];pred=[]
                    for i in ids:
                        f=aa[i,cap,'K15N']['exposure']['prefix']['features'][kind]
                        gains={a:math.fsum(x*y for x,y in zip(f,m[a]['gain'],strict=True)) for a in ARMS}
                        costs={a:math.fsum(x*y for x,y in zip(f,m[a]['cost'],strict=True)) for a in ARMS}
                        a=max(ARMS,key=lambda k:gains[k]-.002*costs[k]);chosen.append(a)
                        for arm in ARMS:
                            y=U(u,i,aa[i,cap,arm]['best'])-U(u,i,aa[i,cap,'K15N']['best']);pred.append((gains[arm],y,m[arm]['mean_gain']))
                    actual=[aa[i,cap,a] for i,a in zip(ids,chosen)]
                    dg=[U(u,i,x['best'])-b for i,x,b in zip(ids,actual,base)]
                    dc=[cpu(x)-cpu(y) for x,y in zip(actual,a0)]
                    d['policies'][name]=dict(selected=dict(C.Counter(chosen)),gain=stats(dg),net=stats([x-.002*c for x,c in zip(dg,dc)]),MSE=S.mean((p-y)**2 for p,y,_ in pred),zero_MSE=S.mean(y*y for p,y,_ in pred),mean_predictor_MSE=S.mean((m0-y)**2 for p,y,m0 in pred),scope='offline measured atlas lookup, not live execution')
                m=frozen['models'][str(cap)]['RAW'];arm=max(ARMS,key=lambda a:m[a]['mean_gain']-.002*m[a]['mean_cost'])
                dg=[U(u,i,aa[i,cap,arm]['best'])-b for i,b in zip(ids,base)]
                dc=[cpu(aa[i,cap,arm])-cpu(a) for i,a in zip(ids,a0)]
                d['policies']['MEAN']=dict(selected={arm:n},gain=stats(dg),net=stats([x-.002*c for x,c in zip(dg,dc)]),scope='offline fixed TRAIN mean control')
            g['by_cap'][str(cap)]=d
        out['groups'][grp]=g
    assert out['groups']['TRAIN']['exposure']==frozen['gate']
    write('C08_SUMMARY.json',out)
    print('TOTAL',{k:v for k,v in out.items() if k not in ('groups','gate','identity')})
    for g,z in out['groups'].items():
        print('GROUP',g,z['families'],z['strata'],z['exposure']['counts'],'strata',[(k,v['counts']) for k,v in z['stratified'].items()])
        for cap,v in z['by_cap'].items():
            print('CAP',cap,{k:x for k,x in v.items() if k not in ('arms','policies')})
            for k,x in v['arms'].items():print(k,'gain/sub',x['gain']['mean'],x['sub_gain']['mean'],'signs',x['gain']['positive'],x['gain']['negative'],'change raw/sub/raw-sub',x['changed'],x['sub_changed'],x['raw_sub_changed'],'CPU/wall',x['cpu_ms'],x['wall_ms'],'visits/NN',x['mean_visits'],x['mean_NN'],'locked',x['locked'],'net',x['net']['mean'],'CI',x['gain_CI95'])
            print('POLICIES',v['policies'])
if __name__=='__main__':main()
