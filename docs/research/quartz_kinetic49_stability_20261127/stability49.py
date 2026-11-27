"""Finite-horizon sensitivity, not statistical or minimax error bounds."""
import collections as C,gzip,json,math,sys
from k49_plan import D,T,SEEDS,SEED,load,save,sha
H=('32','64','128')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    for fi in load(mode+'_LOCK.json')['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];yield from raw(p)
def atlas(mode):
    if mode=='HIST':
        for g in rows(mode):yield dict(split=g['split'],job=g['job'],seed=g['seed'],bin=g['bin'],disagree=g['disagree'],history=g['history'],**g['utility'])
    else:
        for g in rows(mode):
            for r in g['rows']:
                if 'utility' in r:
                    yield dict(split=mode,job=g['job']['id'],seed=g['seed'],bin=r['bin'],disagree=int(r['low']['original_move']!=r['low']['prior_best']),history=r['history'],**r['utility'])
def sig(x):return int(x>0)-int(x<0)
def describe(rr):
    n=len(rr);ch=[r for r in rr if r['changed']]
    def stat(rs):
        nn=len(rs);w=[max(r['gains'].values())-min(r['gains'].values()) for r in rs];ss=[tuple(sig(r['gains'][h]) for h in H) for r in rs]
        q=sorted(w)
        def quant(p):return q[min(len(q)-1,math.floor(p*len(q)))] if q else None
        return dict(n=nn,strict_reversal=sum(min(s)<0<max(s) for s in ss),sign_unstable=sum(len(set(s))>1 for s in ss),constant=sum(x==0 for x in w),diameter_mean=math.fsum(w)/nn if nn else None,diameter_median=quant(.5),diameter_p90=quant(.9),diameter_max=max(w) if w else None,gains={h:dict(mean=math.fsum(r['gains'][h] for r in rs)/nn if nn else None,total=math.fsum(r['gains'][h] for r in rs),pos=sum(r['gains'][h]>0 for r in rs),neg=sum(r['gains'][h]<0 for r in rs),zero=sum(r['gains'][h]==0 for r in rs)) for h in H},changes={a+'_'+b:dict(mean_abs=math.fsum(abs(r['gains'][a]-r['gains'][b]) for r in rs)/nn if nn else None,strict_reversal=sum(r['gains'][a]*r['gains'][b]<0 for r in rs),sign_diff=sum(sig(r['gains'][a])!=sig(r['gains'][b]) for r in rs)) for a,b in (('32','64'),('64','128'),('32','128'))})
    teacher=[t for r in rr for t in r['teacher'] if t is not None]
    cost={h:sum(t[h]['nn'] for t in teacher) for h in H};lead={}
    for a,b in (('32','64'),('64','128'),('32','128')):
        active=[t for t in teacher if not t[a]['terminal']]
        lead[a+'_'+b]=dict(n=len(active),leader_changed=sum(t[a]['leader']!=t[b]['leader'] for t in active),same_leader_value_changed=sum(t[a]['leader']==t[b]['leader'] and t[a]['raw']!=t[b]['raw'] for t in active))
    return dict(all=stat(rr),changed=stat(ch),teacher_calls=len(teacher),physical_teacher_nn=cost['128'],hypothetical_cold_nn=sum(cost.values()),teacher_nn_by_h=cost,leader=lead,by_seed={str(s):stat([r for r in ch if r['seed']==s]) for s in SEEDS},top={h:[dict(seed=r['seed'],job=r['job'],ply=len(r['history']),gain=r['gains'][h]) for r in sorted(ch,key=lambda r:abs(r['gains'][h]),reverse=True)[:5]] for h in H},by_bin={str(z):stat([r for r in rr if r['bin']==z]) for z in range(8)},loocv_mean={h:{str(s):math.fsum(r['gains'][h] for r in rr if r['seed']!=s)/sum(r['seed']!=s for r in rr) for s in SEEDS} for h in H})
def main():
    old=list(atlas('HIST'));fresh={m:list(atlas(m)) for m in ('TRAIN','CAL','TEST')}
    d=dict(historical={m:describe([r for r in old if r['split']==m]) for m in ('TRAIN','CAL','TEST')},fresh={m:describe(r) for m,r in fresh.items()},historical_all=describe(old),fresh_all=describe(sum(fresh.values(),[])),scope='sampled H32/64/128 sensitivity, not true minimax error or confidence interval; conditional changed-action subset separate')
    save('STABILITY.json',d)
    for name,z in [('historical_all',d['historical_all']),('fresh_all',d['fresh_all']),*d['fresh'].items()]:
        print(name,'changed',z['changed'],'teacher_nn',z['teacher_nn_by_h'],'leader',z['leader'],flush=True)
if __name__=='__main__':main()
