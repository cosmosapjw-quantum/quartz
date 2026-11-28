"""Historical matched outcomes only at the actually played assigned STOP horizon."""
import collections as C,json,gzip
from k58_plan import D,T,sha
from parity58 import stable
def main():
    d=D.parent/'quartz_kinetic57_early_20261128';lk=json.loads((d/'NN_LOCK.json').read_text());o=C.defaultdict(C.Counter)
    for f in lk['files']:
        p=d/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as r:
            for x in map(json.loads,r):
                g=x['control'];e=g['event']
                if e is None:continue
                s=g['rows'][e]['early']['snapshots'];h=g['job']['horizon']
                match=next(((a,b) for a,b in zip(s,s[1:]) if b['visits']==h),None)
                first=next((b for a,b in zip(s,s[1:]) if stable(a,b)),None)
                status='stable_at_assigned' if match is not None and stable(*match) else 'not_stable_at_assigned'
                r0=x['pairs']['stop'];assert r0['horizon']==h and r0['reached']
                for k in (status,f'{status}:{h}'):
                    c=o[k];c['n']+=1;c['delta2']+=r0['delta2'];c['local_nn']+=r0['local_nn'];c['changed']+=r0['changed'];c['improved']+=r0['delta2']>0;c['worsened']+=r0['delta2']<0
                if first is not None and first['visits']==h:
                    c=o['earliest_matches_assigned'];c['n']+=1;c['delta2']+=r0['delta2'];c['local_nn']+=r0['local_nn'];c['changed']+=r0['changed'];c['improved']+=r0['delta2']>0;c['worsened']+=r0['delta2']<0
    assert o['stable_at_assigned']['n']+o['not_stable_at_assigned']['n']==586
    assert o['stable_at_assigned']['delta2']+o['not_stable_at_assigned']['delta2']==4
    for c in o.values():assert abs(c['delta2'])<=2*c['changed']
    x=dict(counts={k:dict(v) for k,v in o.items()},source=sha(d/'NN_LOCK.json'),scope='Retrospective actual matched STOP-Q20 only at assigned horizon; no unplayed counterfactuals, no tuning or test admission. Stability at h4 unavailable by rule; stratify rather than causal comparison.')
    (T/'HISTORICAL_EFFECT.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x),flush=True)
if __name__=='__main__':main()
