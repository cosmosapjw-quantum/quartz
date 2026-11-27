"""Family bootstrap of actual paired games; compact reuse is policy replay, not new games."""
import sys,collections as C,json,math
import numpy as np
from k51_plan import D,SEED,SEEDS,load,save,sha
from experiment51 import rows,raw
OLD=D.parent/'quartz_kinetic50_intervention_20261127'
sys.path.append(str(OLD))
from analyze50 import game,effect,describe,ci
def archived():
    p=OLD/'R96_LOCK.json';assert sha(p)==load('PLAN.json')['historical_rapfi_lock_sha256']
    for f in json.loads(p.read_text())['files']:
        q=OLD/f['file'];assert sha(q)==f['sha256'];yield from raw(q)
def aligned(a,b):
    x={(g['job']['id'],g['seed'],g['color']):g for g in a};y={(g['job']['id'],g['seed'],g['color']):g for g in b}
    assert len(x)==len(a)==len(b)==len(y) and set(x)==set(y)
    assert all(x[k]['job']==y[k]['job'] for k in x)
    return x,y
def contrast(a,b,idx):
    x,y=aligned(a,b);aa,za=game(a,idx);bb,zb=game(b,idx)
    return dict(a=aa,b=bb,pairs=len(x),identical_histories=sum(x[k]['history']==y[k]['history'] for k in x),delta={k:dict(point=bb['metrics'][k]-aa['metrics'][k],CI=ci(zb[k]-za[k])) for k in za})
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed'])
    xs=list(rows('NN'));assert len(xs)==672
    out={}
    for panel,xx in [('all',xs),('old',[x for x in xs if x['control']['seed'] in (101,102,103,105)]),('new',[x for x in xs if x['control']['seed'] in (41,42,43)])]:
        idx=rng.integers(0,48,(8000,48))
        arms={a:[x[a] for x in xx] for a in ('control','global','target')}
        out[panel]=dict(arms={a:game(v,idx)[0] for a,v in arms.items()},effects={a:effect([x['pairs'][a] for x in xx],idx) for a in ('global','target')},target_minus_global=contrast(arms['global'],arms['target'],idx))
        print('NN',panel,{a:z['metrics'] for a,z in out[panel]['arms'].items()},{a:z['counts'] for a,z in out[panel]['effects'].items()},flush=True)
    td=C.Counter();bins=C.defaultdict(list)
    for x in xs:
        a,b=x['pairs']['global'],x['pairs']['target'];assert a['event']==b['event']
        if b['reached']:
            t=x['target']['rows'][b['event']]['target'];td['reached']+=1;td['compared']+=t['reason']=='compared';td['children']+=sum(z is not None for z in t['children']);td['physical_nn']+=t['extra_nn'];td['point_values']+=sum(z is None for z in t['children']);td['target_switches']+=b['changed'];td['global_switches']+=a['changed'];td['same_action_global_target']+=x['global']['rows'][a['event']]['move']==t['chosen'];bins[b['bin']].append(b)
    out['target_receipts']=dict(td);out['target_bins']={str(k):describe(v) for k,v in bins.items()}
    hist96=list(archived());hist128=list(rows('R_H128'));fresh96=list(rows('R_F96'));outer128=list(rows('R_F128'));assert len(hist96)==len(hist128)==256 and len(fresh96)==192 and len(outer128)==96
    guard=[g for g in fresh96 if g['job']['stratum']=='compact']+outer128
    r={}
    for cohort,a,b in [('historical',hist96,hist128),('fresh',fresh96,guard)]:
        r[cohort]={}
        for p in ('all','compact','outer'):
            aa=[g for g in a if p=='all' or g['job']['stratum']==p];bb=[g for g in b if p=='all' or g['job']['stratum']==p]
            n=len({g['job']['id'] for g in aa});idx=rng.integers(0,n,(8000,n));z=contrast(aa,bb,idx)
            if cohort=='historical':
                z['delta'].pop('CPU_reduction',None);z['delta'].pop('CPU_game_reduction',None);z['delta'].pop('wall_reduction',None);z['delta'].pop('wall_game_reduction',None)
            r[cohort][p]=z
            print('Rapfi',cohort,p,'q96',z['a']['metrics'],'other',z['b']['metrics'],'lower',z['b']['lower95'],'gate',z['b']['gate'],'delta',z['delta']['score'],z['delta']['NN_reduction'],flush=True)
    save('SUMMARY.json',dict(nn=out,rapfi=r,plan=sha(D/'PLAN.json'),locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_H128','R_F96','R_F128')},scope='8000 family bootstrap; all actual Q-v-B games, fresh guard compact reuses identical Q96 arm; historical timing contrasts omitted'))
if __name__=='__main__':main()
