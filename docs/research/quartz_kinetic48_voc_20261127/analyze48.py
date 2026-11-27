"""Independent family-cluster summary with same family samples for distinct Q vs B policies."""
import collections as C,gzip,json,sys,math
import numpy as np
from k48_plan import D,T,load,save,sha,SEEDS,SEED
OLD={101,102,103,105};NEW={41,42,43}
KEYS=('games','score2','wins','draws','losses','Qmoves','Qnn','Qcpu','Qwall','Bmoves','Bnn','Bcpu','Bwall','attempts','cert','replacement','queries','exact_attempt','exact_complete','exact_nodes','proof_wall','extensions','extra_nn')
def rows(mode):
    l=load(mode+'_LOCK.json')
    for f in l['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as fd:yield from map(json.loads,fd)
def accum(gs):
    c=C.Counter()
    for g in gs:
        c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['draws']+=g['score2']==1;c['losses']+=g['score2']==0
        for r in g['rows']:
            t=r['arm'];c[t+'moves']+=1;c[t+'nn']+=r['nn'];c[t+'cpu']+=r['cpu_ns'];c[t+'wall']+=r['wall_ns']
            z=r.get('partial')
            if z:
                c['attempts']+=1;c['cert']+=z['cert'] is not None;c['replacement']+=r['replacement'];c['queries']+=z['cold']['receipt']['queries'];c['proof_wall']+=z['wall_ns']
                if z['exact']:c['exact_attempt']+=1;c['exact_complete']+=z['exact']['complete'];c['exact_nodes']+=z['exact']['nodes']
            if r.get('extended'):c['extensions']+=1;c['extra_nn']+=r['nn']-r['low']['nn']
    return c
def metric(a):
    ix={k:i for i,k in enumerate(KEYS)}
    def val(k):return a[...,ix[k]]
    d=dict(score=val('score2')/(2*val('games')))
    for k,n in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
        d[n+'_reduction']=1-(val('Q'+k)/val('Qmoves'))/(val('B'+k)/val('Bmoves'))
        d[n+'_game_reduction']=1-val('Q'+k)/val('B'+k)
    return d
def calc(gs,idx):
    groups=C.defaultdict(list)
    for g in gs:groups[g['job']['id']].append(g)
    assert len(groups)==32
    a=np.asarray([[accum(groups[i])[k] for k in KEYS] for i in sorted(groups)],dtype=float);c=accum(gs)
    d={k:float(v) for k,v in metric(np.asarray([c[k] for k in KEYS],dtype=float)).items()};z=metric(a[idx].sum(axis=1))
    ci={k:[float(np.sort(v)[199]),float(np.sort(v)[7799])] for k,v in z.items()};lo={k:float(np.sort(v)[399]) for k,v in z.items()}
    gate=dict(score=lo['score']>=.45,NN=lo['NN_reduction']>=.30,CPU=d['CPU_reduction']>0)
    return dict(counts={k:int(c[k]) for k in KEYS},metrics=d,CI=ci,lower95=lo,gate=gate,all_gate=all(gate.values())),z
def main():
    p=load('PLAN.json');rng=np.random.default_rng(p['bootstrap_seed']);ix=rng.integers(0,32,(8000,32));ri=rng.integers(0,32,(8000,32))
    modes=['NN_FIXED']+(['NN_ADAPT'] if (D/'NN_ADAPT_LOCK.json').exists() else [])+['R80','R96','R112','R128']
    out={};boot={};games={}
    for m in modes:
        gs=list(rows(m));games[m]={(g['job']['id'],g['seed'],g['color']):g for g in gs};assert len(gs)==len(games[m])==(64 if m.startswith('R') else 448)
        if m.startswith('R'):pan={'Rapfi':gs}
        else:pan={'old':[g for g in gs if g['seed'] in OLD],'new':[g for g in gs if g['seed'] in NEW],'all':gs,**{str(s):[g for g in gs if g['seed']==s] for s in SEEDS}}
        out[m]={};boot[m]={}
        for k,rr in pan.items():out[m][k],boot[m][k]=calc(rr,ri if m.startswith('R') else ix)
        print(m,{p:{'WDL':[z['counts'][x] for x in ('wins','draws','losses')],'metrics':z['metrics'],'lower95':z['lower95'],'gate':z['gate']} for p,z in out[m].items() if p in ('old','new','all','Rapfi')},flush=True)
    pairs={}
    for m,b in ([('NN_ADAPT','NN_FIXED')] if 'NN_ADAPT' in out else [])+[(m,'R128') for m in ('R80','R96','R112')]:
        assert set(games[m])==set(games[b]);pairs[m]={}
        for p in (('Rapfi',) if m.startswith('R') else ('old','new','all')):
            ds={k:boot[m][p][k]-boot[b][p][k] for k in boot[m][p]}
            ks=[k for k in games[m] if p in ('all','Rapfi') or k[1] in (OLD if p=='old' else NEW)]
            pairs[m][p]=dict(reference=b,identical_histories=sum(games[m][k]['history']==games[b][k]['history'] for k in ks),pairs=len(ks),metrics={k:dict(point=out[m][p]['metrics'][k]-out[b][p]['metrics'][k],CI=[float(np.sort(v)[199]),float(np.sort(v)[7799])]) for k,v in ds.items()})
    save('SUMMARY.json',dict(arena=out,paired=pairs,plan=sha(D/'PLAN.json'),locks={m:sha(D/(m+'_LOCK.json')) for m in modes},scope='fixed-family bootstrap 8000, same evaluator Q vs B; pair contrast of separately played games against common baseline, not direct Q-v-Q'))
if __name__=='__main__':main()
