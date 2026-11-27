"""Family-resampled paired effects and actual arena metrics. Replay is not new games."""
import collections as C,math,json,sys
import numpy as np
from k50_plan import D,SEED,SEEDS,load,save,sha
from experiment50 import rows
OLD={101,102,103,105}
K=('n','delta2','reached','changed','improved','worsened','local_nn','delta_Q','delta_B','delta_total','downstream_Q','downstream_B','same_history')
G=('games','score2','wins','draws','losses','Qmoves','Qnn','Qcpu','Qwall','Bmoves','Bnn','Bcpu','Bwall','cert','queries','replacement')
def sum_pair(rr):
    c=C.Counter(n=len(rr))
    for r in rr:
        for k in ('delta2','reached','changed','local_nn','delta_Q','delta_B','delta_total','downstream_Q','downstream_B','same_history'):c[k]+=r[k]
        c['improved']+=r['delta2']>0;c['worsened']+=r['delta2']<0
    return c
def sum_game(gs):
    c=C.Counter()
    for g in gs:
        c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['draws']+=g['score2']==1;c['losses']+=g['score2']==0
        for r in g['rows']:
            tag=r['arm'];c[tag+'moves']+=1;c[tag+'nn']+=r['nn'];c[tag+'cpu']+=r['cpu_ns'];c[tag+'wall']+=r['wall_ns']
            z=r.get('partial')
            if z:c['cert']+=z['cert'] is not None;c['queries']+=z['cold']['receipt']['queries'];c['replacement']+=r['replacement']
    return c
def ci(x):
    s=np.sort(x);return [float(s[199]),float(s[7799])]
def game_metric(x):
    i={k:j for j,k in enumerate(G)}
    def v(k):return x[...,i[k]]
    d=dict(score=v('score2')/(2*v('games')))
    for k,l in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
        d[l+'_reduction']=1-(v('Q'+k)/v('Qmoves'))/(v('B'+k)/v('Bmoves'))
        d[l+'_game_reduction']=1-v('Q'+k)/v('B'+k)
    return d
def game(gs,idx):
    groups=C.defaultdict(list)
    for g in gs:groups[g['job']['id']].append(g)
    c=sum_game(gs);a=np.array([[sum_game(groups[j])[k] for k in G] for j in sorted(groups)],dtype=float)
    assert idx.shape[1]==len(groups)
    z=game_metric(a[idx].sum(axis=1));d={k:float(v) for k,v in game_metric(np.array([c[k] for k in G],dtype=float)).items()}
    lower={k:float(np.sort(v)[399]) for k,v in z.items()}
    return dict(counts={k:int(c[k]) for k in G},metrics=d,CI={k:ci(v) for k,v in z.items()},lower95=lower,gate=dict(score=lower['score']>=.45,NN=lower['NN_reduction']>=.30,CPU=d['CPU_reduction']>0)),z
def pair_metric(a):
    i={k:j for j,k in enumerate(K)}
    def v(k):return a[...,i[k]]
    n=v('n')
    return dict(tau=v('delta2')/(2*n),local_nn=v('local_nn')/n,delta_Q=v('delta_Q')/n,delta_B=v('delta_B')/n,delta_total=v('delta_total')/n,downstream_Q=v('downstream_Q')/n,downstream_B=v('downstream_B')/n,reach=v('reached')/n,change=v('changed')/n)
def effect(rr,idx):
    groups=C.defaultdict(list)
    for r in rr:groups[r['job']].append(r)
    c=sum_pair(rr);a=np.array([[sum_pair(groups[j])[k] for k in K] for j in sorted(groups)],dtype=float);assert idx.shape[1]==len(groups)
    z=pair_metric(a[idx].sum(axis=1));d={k:float(v) for k,v in pair_metric(np.array([c[k] for k in K],dtype=float)).items()}
    out=dict(counts={k:int(c[k]) for k in K},metrics=d,CI={k:ci(v) for k,v in z.items()},lower95={k:float(np.sort(v)[399]) for k,v in z.items()},delta2_hist=dict(C.Counter(r['delta2'] for r in rr)),conditional_reached=c['delta2']/(2*c['reached']) if c['reached'] else None,conditional_changed=c['delta2']/(2*c['changed']) if c['changed'] else None)
    for k in ('local_nn','delta_Q','delta_total'):
        out['ratio_'+k]=dict(point=d['tau']/d[k],CI=ci(z['tau']/z[k])) if d[k]>0 and out['CI'][k][0]>0 else None
    return out
def describe(rr):
    c=sum_pair(rr)
    return dict(n=len(rr),reached=c['reached'],changed=c['changed'],delta2=c['delta2'],gain=c['delta2']/2,local_nn=c['local_nn'],delta_Q=c['delta_Q'],delta_total=c['delta_total'],improved=c['improved'],worsened=c['worsened'],mean_gain=c['delta2']/(2*len(rr)) if rr else None)
def policy(xs,mask,idx):
    rr=[x['pair'] for x in xs];selected=[r for r in rr if r['bin'] is not None and mask>>r['bin']&1]
    total=describe(rr);sel=describe(selected);p=sel['local_nn']/total['local_nn'] if total['local_nn'] else 0.
    groups=C.defaultdict(list)
    for r in rr:groups[r['job']].append(r)
    ar=[]
    for j in sorted(groups):
        a=groups[j];b=[r for r in a if r['bin'] is not None and mask>>r['bin']&1]
        ar.append([len(a),sum(r['delta2'] for r in a),sum(r['local_nn'] for r in a),sum(r['delta2'] for r in b),sum(r['local_nn'] for r in b)])
    ar=np.array(ar,dtype=float)[idx].sum(axis=1);pp=np.divide(ar[:,4],ar[:,2],out=np.zeros(len(ar)),where=ar[:,2]>0);ex=(ar[:,3]-pp*ar[:,1])/(2*ar[:,0])
    chosen=[x['extend'] if x['pair']['bin'] is not None and mask>>x['pair']['bin']&1 else x['control'] for x in xs]
    g,_=game(chosen,idx)
    return dict(mask=mask,total=total,selected=sel,constant_fraction=p,constant_gain=p*total['gain'],excess=sel['gain']-p*total['gain'],mean_excess_CI=ci(ex),admitted=bool(mask) and sel['gain']>0 and sel['gain']>p*total['gain'],replay=g,scope='selection of observed actual arm outcome, not newly played games; timing is observed-arm time, not deterministic counterfactual')
def main():
    xs={m:list(rows(m)) for m in ('TRAIN','CAL','TEST')};xs['ALL']=sum(xs.values(),[])
    rgs=list(rows('R96'));f=load('FROZEN.json');ad=load('ADMISSION.json')
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);out={};models={};pairs={}
    for m,xx in xs.items():
        rr=[x['pair'] for x in xx];N=len({r['job'] for r in rr});idx=rng.integers(0,N,(8000,N))
        assert N=={'TRAIN':48,'CAL':24,'TEST':24,'ALL':96}[m]
        panels={'all':xx,'old':[x for x in xx if x['pair']['seed'] in OLD],'new':[x for x in xx if x['pair']['seed'] not in OLD]}
        out[m]={};pairs[m]={}
        for p,v in panels.items():
            a,b=game([x['control'] for x in v],idx),game([x['extend'] for x in v],idx)
            out[m][p]={'control':a[0],'extend':b[0]}
            pairs[m][p]=effect([x['pair'] for x in v],idx)
            pairs[m][p]['baseline_relative_cost_saving_delta']={k:dict(point=b[0]['metrics'][k]-a[0]['metrics'][k],CI=ci(b[1][k]-a[1][k])) for k in ('NN_reduction','CPU_reduction','wall_reduction')}
        pairs[m]['by_bin']={str(z):describe([r for r in rr if r['bin']==z]) for z in range(8)}
        pairs[m]['by_ordinal']={str(z):describe([r for r in rr if r['ordinal']==z]) for z in (1,2)}
        pairs[m]['by_seed']={str(s):describe([r for r in rr if r['seed']==s]) for s in SEEDS}
        if m!='ALL':models[m]=policy(xx,f['mask'],idx)
        print(m,'pairs',pairs[m]['all'],'control',out[m]['all']['control']['metrics'],'extend',out[m]['all']['extend']['metrics'],flush=True)
    assert models['CAL']['admitted']==ad['admitted']
    rp={}
    for p,gs in [('all',rgs),('compact',[g for g in rgs if g['job']['stratum']=='compact']),('outer',[g for g in rgs if g['job']['stratum']=='outer'])]:
        n=len({g['job']['id'] for g in gs});assert n==(128 if p=='all' else 64)
        rp[p],_=game(gs,rng.integers(0,n,(8000,n)));print('Rapfi',p,rp[p],flush=True)
    save('SUMMARY.json',dict(nn=out,paired=pairs,replay=models,rapfi=rp,plan=sha(D/'PLAN.json'),locks={m:sha(D/(m+'_LOCK.json')) for m in ('TRAIN','CAL','TEST','R96')},scope='family bootstrap8000; actual separate same-evaluator Q-v-B arms; paired episode ITT and single-event deterministic replay'))
if __name__=='__main__':main()
