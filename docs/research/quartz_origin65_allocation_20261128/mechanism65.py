from plan65 import D,T,ARMS,DEV,read,raw,sha
from collections import Counter,defaultdict
from fractions import Fraction
import math,json,sys
from audit65 import metrics,prefix
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json')
    counters={arm:Counter() for arm in DEV}
    contrasts=[('L4','H4'),('A4','L4'),('M4','L4'),('MA4','M4'),('MA4','A4'),('MA4','FREE'),('X4','MA4')]
    pairs={f'{x}-{y}':Counter() for x,y in contrasts}
    for m in lock['files']:
        f=T/m['file'];assert sha(f)==m['sha256']
        for gs in raw(f):
            for arm in DEV:
                c=counters[arm]
                for r in gs[arm]['rows']:
                    if r['arm']!='Q':continue
                    c['moves']+=1
                    if 'allocation65' not in r:c['cert']+=1;continue
                    z=metrics(r);c['unresolved']+=1
                    for key in ('visited','loss_visits','unvisited_chosen','physical','logical','feasible'):c[key]+=z[key]
                    c['all_loss']+=z['loss_visits']==4;c['mass']+=z['mass']
            for x,y in contrasts:
                ga,gb=gs[x],gs[y];c=pairs[f'{x}-{y}']
                c['episodes']+=1;c['same_history']+=ga['history']==gb['history'];c['better']+=ga['score2']>gb['score2'];c['worse']+=ga['score2']<gb['score2'];c['delta_score2']+=ga['score2']-gb['score2']
                c['delta_both_physical']+=ga['actual_nn']-gb['actual_nn']
                div=False
                for a,b in zip(ga['rows'],gb['rows']):
                    assert a['history']==b['history'];c['common_roots']+=1
                    c['common_both_physical_delta']+=a['actual_nn']-b['actual_nn']
                    if a['arm']=='Q':
                        c['common_q_roots']+=1;c['common_q_physical_delta']+=a['actual_nn']-b['actual_nn']
                        if 'allocation65' in a and 'allocation65' in b:
                            u,v=metrics(a),metrics(b);c['common_unresolved']+=1
                            c['common_coverage_delta']+=u['visited']-v['visited']
                            c['common_loss_visits_delta']+=u['loss_visits']-v['loss_visits']
                            c['common_unvisited_choice_delta']+=u['unvisited_chosen']-v['unvisited_chosen']
                            if (x,y) in [('A4','L4'),('MA4','M4')]:
                                assert prefix(a)==prefix(b) and a['cache_receipt']==b['cache_receipt']
                                c['readout_only_prefix']+=1
                    if a['played']!=b['played']:
                        assert a['arm']=='Q';c['first_divergence']+=1;div=True;break
                if not div:assert ga['history']==gb['history']
    n={'MAIN':336,'ARCHIVE':72,'NOVEL':72}[mode]
    assert all(c['episodes']==n and c['same_history']+c['first_divergence']==n for c in pairs.values())
    for arm in ('M4','MA4','X4'):assert counters[arm]['loss_visits']==0 and counters[arm]['all_loss']==0
    out=dict(mode=mode,arms={a:dict(c,mean_coverage=c['visited']/c['unresolved'],mean_prior_mass=c['mass']/c['unresolved']) for a,c in counters.items()},pairs={a:dict(c) for a,c in pairs.items()},lock_sha256=sha(T/(mode+'_LOCK.json')))
    (T/(mode+'_MECHANISM.json')).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(mode,json.dumps(out),flush=True)
if __name__=='__main__':main(sys.argv[1])
