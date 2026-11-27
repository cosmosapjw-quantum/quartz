"""Prespecified family bootstrap; actual separate same-evaluator policy games."""
import sys,collections as C,json
import numpy as np
from k52_plan import D,SEED,SEEDS,load,save,sha
from experiment52 import rows
sys.path.append(str(D.parent/'quartz_kinetic50_intervention_20261127'))
from analyze50 import game,effect,describe,ci
CANDIDATES=('104','112','AD')
def aligned(a,b):
    x={(g['job']['id'],g['seed'],g['color']):g for g in a};y={(g['job']['id'],g['seed'],g['color']):g for g in b}
    assert len(x)==len(a)==len(b)==len(y) and set(x)==set(y)
    assert all(x[k]['job']==y[k]['job'] for k in x)
    return x,y
def contrast(a,b,idx):
    x,y=aligned(a,b);aa,za=game(a,idx);bb,zb=game(b,idx)
    return dict(a=aa,b=bb,pairs=len(x),identical_histories=sum(x[k]['history']==y[k]['history'] for k in x),delta={k:dict(point=bb['metrics'][k]-aa['metrics'][k],CI=ci(zb[k]-za[k])) for k in za})
def rng():return np.random.default_rng(load('PLAN.json')['bootstrap_seed'])
def counts_cap(gs):
    c=C.Counter()
    for g in gs:
        for r in g['rows']:
            if r['arm']=='Q' and r['kind'] not in ('CERTIFICATE','PARTIAL47'):
                cap=r['selected_cap'];c[str(cap)]+=1
    return dict(c)
def select():
    r=rng();idx=r.integers(0,32,(8000,32));out={}
    for cap in CANDIDATES:
        gs=list(rows('R_CAL_'+cap));assert len(gs)==64
        z,_=game(gs,idx);z['cap_counts']=counts_cap(gs);out[cap]=z
    ok=[c for c in CANDIDATES if all(out[c]['gate'].values())]
    selected=max(ok,key=lambda c:(out[c]['lower95']['NN_reduction'],out[c]['lower95']['score'],-CANDIDATES.index(c))) if ok else '112'
    result=dict(selected=selected,admitted=bool(ok),eligible=ok,cal=out,locks={c:sha(D/('R_CAL_'+c+'_LOCK.json')) for c in CANDIDATES},scope='CAL-only fixed rule; if none pass, Q112 diagnostic; no TEST access')
    save('SELECTION.json',result);print('SELECT',json.dumps(result),flush=True)
def main():
    f=load('SELECTION.json');c=f['selected']
    r=rng();r.integers(0,32,(8000,32));xs=list(rows('NN'));assert len(xs)==1344
    out={}
    for panel,xx in [('all',xs),('old',[x for x in xs if x['control']['seed'] in (101,102,103,105)]),('new',[x for x in xs if x['control']['seed'] in (41,42,43)])]:
        idx=r.integers(0,96,(8000,96));arms={a:[x[a] for x in xx] for a in ('control','global','target')}
        out[panel]=dict(arms={a:game(v,idx)[0] for a,v in arms.items()},effects={a:effect([x['pairs'][a] for x in xx],idx) for a in ('global','target')},target_minus_global=contrast(arms['global'],arms['target'],idx))
        print('NN',panel,{a:z['metrics'] for a,z in out[panel]['arms'].items()},{a:(z['counts'],z['CI']['tau'],z['ratio_local_nn']) for a,z in out[panel]['effects'].items()},flush=True)
    td=C.Counter();bins=C.defaultdict(list)
    for x in xs:
        a,b=x['pairs']['global'],x['pairs']['target'];assert a['event']==b['event']
        if b['reached']:
            t=x['target']['rows'][b['event']]['target'];td['reached']+=1;td['compared']+=t['reason']=='compared';td['children']+=sum(z is not None for z in t['children']);td['physical_nn']+=t['extra_nn'];td['point_values']+=sum(z is None for z in t['children']);td['target_switches']+=b['changed'];td['global_switches']+=a['changed'];td['same_action_global_target']+=x['global']['rows'][a['event']]['move']==t['chosen'];bins[b['bin']].append(b)
    out['target_receipts']=dict(td);out['target_bins']={str(k):describe(v) for k,v in bins.items()}
    # Separate immutable historical comparison, never combine samples for primary confirmation.
    a=out['all'];effect0=a['effects']['target'];g=a['arms']['target']['gate'];rat=effect0['ratio_local_nn']
    confirmation=dict(score=effect0['CI']['tau'][0]>0,local_value=rat is not None and rat['CI'][0]>0,baseline=all(g.values()))
    aa=list(rows('R_TEST_96'));bb=list(rows('R_TEST_'+c));assert len(aa)==len(bb)==96
    z=contrast(aa,bb,r.integers(0,48,(8000,48)));z['a']['cap_counts']=counts_cap(aa);z['b']['cap_counts']=counts_cap(bb)
    print('R_TEST',c,json.dumps(z),flush=True)
    save('SUMMARY.json',dict(nn=out,confirmation=confirmation,confirmed=all(confirmation.values()),rapfi=dict(selection=f,test=z,admitted=f['admitted'] and all(z['b']['gate'].values())),plan=sha(D/'PLAN.json'),locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_CAL_104','R_CAL_112','R_CAL_AD','R_TEST_96','R_TEST_'+c)},scope='8000 family bootstrap; actual separate Q-v-B games; independent native TARGET confirmation and Rapfi CAL-only selection/holdout'))
if __name__=='__main__':
    if sys.argv[1]=='select':select()
    else:main()
