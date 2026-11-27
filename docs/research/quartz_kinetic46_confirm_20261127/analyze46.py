"""Predeclared family-cluster bootstrap; fixed model panel, paired colors and policies."""
import collections as C,gzip,json,pathlib,sys,hashlib
import numpy as np
from k46_plan import D,load,save,sha,SEEDS
OLD={101,102,103,105};NEW={41,42,43}
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as fd:yield from map(json.loads,fd)
KEYS=('games','score2','wins','draws','losses','Qmoves','Qnn','Qcpu','Qwall','Bmoves','Bnn','Bcpu','Bwall','attempts','cert','replacement','queries','exact_attempt','exact_complete','exact_nodes','proof_cpu')
def accum(gs):
    c=C.Counter()
    for g in gs:
        c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['draws']+=g['score2']==1;c['losses']+=g['score2']==0
        for r in g['rows']:
            t=r['arm'];c[t+'moves']+=1;c[t+'nn']+=r['nn'];c[t+'cpu']+=r['cpu_ns'];c[t+'wall']+=r['wall_ns']
            z=r.get('partial')
            if z:
                c['attempts']+=1;c['cert']+=z['cert'] is not None;c['replacement']+=r['replacement'];c['queries']+=z['cold']['receipt']['queries'];c['proof_cpu']+=z['cpu_ns']
                if z['exact']:c['exact_attempt']+=1;c['exact_complete']+=z['exact']['complete'];c['exact_nodes']+=z['exact']['nodes']
    return c
def metric(a):
    ix={k:i for i,k in enumerate(KEYS)}
    def val(k):return a[...,ix[k]]
    d=dict(score=val('score2')/(2*val('games')))
    for k,n in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
        d[n+'_reduction']=1-(val('Q'+k)/val('Qmoves'))/(val('B'+k)/val('Bmoves'))
        d[n+'_game_reduction']=1-val('Q'+k)/val('B'+k)
    return d
def point(c):
    return {k:float(x) for k,x in metric(np.asarray([c[k] for k in KEYS],dtype=float)).items()}
def run(mode,gs,idx):
    groups=C.defaultdict(list)
    for g in gs:groups[g['job']['id']].append(g)
    ids=sorted(groups);assert len(ids)==(64 if mode=='R_ARENA' else 96)
    ar=np.array([[accum(groups[i])[k] for k in KEYS] for i in ids],dtype=np.float64)
    c=accum(gs);d=point(c);z=metric(ar[idx].sum(axis=1));ci={}
    for k,v in z.items():
        s=np.sort(v);ci[k]=[float(s[199]),float(s[7799])]
    one={k:float(np.sort(v)[399]) for k,v in z.items()}
    gate=dict(score=one['score']>=.45,NN=one['NN_reduction']>=.30,CPU=d['CPU_reduction']>0)
    return dict(counts={k:int(c[k]) for k in KEYS},metrics=d,CI=ci,lower95=one,gate=gate,all_gate=all(gate.values())),z
def main(modes,write=True):
    p=load('PLAN.json');assert p['bootstrap_replicates']==8000
    rng=np.random.default_rng(p['bootstrap_seed']);nnidx=rng.integers(0,96,(8000,96));ridx=rng.integers(0,64,(8000,64));out={};z={};games={}
    for mode in modes:
        gs=list(rows(mode));games[mode]={ (g['job']['id'],g['seed'],g['color']):g for g in gs};assert len(games[mode])==len(gs)
        assert len(gs)==(128 if mode=='R_ARENA' else 1344)
        if mode=='R_ARENA':panels={'Rapfi':gs}
        else:
            panels={'old':[g for g in gs if g['seed'] in OLD],'new':[g for g in gs if g['seed'] in NEW],'all':gs}
            panels.update({str(s):[g for g in gs if g['seed']==s] for s in SEEDS})
        out[mode]={};z[mode]={}
        for panel,gg in panels.items():
            out[mode][panel],z[mode][panel]=run(mode,gg,ridx if mode=='R_ARENA' else nnidx)
        print(mode,{k:{'games':v['counts']['games'],'WDL':[v['counts'][a] for a in ('wins','draws','losses')],'point':v['metrics'],'lower95':v['lower95'],'gate':v['gate']} for k,v in out[mode].items() if k in ('old','new','all','Rapfi')},flush=True)
    paired={}
    if 'NN_LOW' in out and 'NN_FIXED' in out:
        a,b=games['NN_LOW'],games['NN_FIXED'];assert set(a)==set(b)
        assert all(a[k]['job']==b[k]['job'] for k in a)
        for panel in ('old','new','all'):
            ids=[k for k in a if panel=='all' or k[1] in (OLD if panel=='old' else NEW)]
            diff={k:z['NN_LOW'][panel][k]-z['NN_FIXED'][panel][k] for k in z['NN_LOW'][panel]}
            paired[panel]=dict(same_history=sum(a[k]['history']==b[k]['history'] for k in ids),pairs=len(ids),score2_diff=dict(C.Counter(a[k]['score2']-b[k]['score2'] for k in ids)),metrics={k:dict(point=out['NN_LOW'][panel]['metrics'][k]-out['NN_FIXED'][panel]['metrics'][k],CI=[float(np.sort(v)[199]),float(np.sort(v)[7799])],lower95=float(np.sort(v)[399])) for k,v in diff.items()})
        print('PAIRED',paired,flush=True)
    if write:save('SUMMARY.json',dict(arena=out,paired=paired,plan_sha256=sha(D/'PLAN.json'),locks={m:sha(D/(m+'_LOCK.json')) for m in modes},scope='predeclared fixed-size family bootstrap with 8000 resamples; fixed checkpoint panels, per move and path-dependent per game costs; compare distinct Q-v-B runs against common baseline only'))
if __name__=='__main__':main(sys.argv[1:] or ['NN_LOW','NN_FIXED','R_ARENA'])
