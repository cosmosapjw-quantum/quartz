from plan67 import D,T,read,raw,sha
from collections import Counter
import json,sys
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns') and k not in ('retention67','played')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json') and lock['model']==sha(D/'MODEL.json')
    out={}
    for arm,ref in [('CALV','VOTE'),('CALF','FOCUS'),('CALX','EXPAND')]:
        c=Counter()
        for m in lock['files']:
            f=T/m['file'];assert sha(f)==m['sha256']
            for gs in raw(f):
                a,b=gs[arm],gs[ref];c['episodes']+=1;c['same_history']+=a['history']==b['history'];c['better']+=a['score2']>b['score2'];c['worse']+=a['score2']<b['score2'];c['delta_score2']+=a['score2']-b['score2']
                c['delta_both_physical']+=a['actual_nn']-b['actual_nn']
                for r in a['rows']:
                    if 'retention67' not in r:continue
                    z=r['retention67'];c['eligible']+=1;c['discord']+=z['a']!=z['b'];c['switch']+=z['chosen']!=z['a'];c['veto']+=z['a']!=z['b'] and z['chosen']==z['a']
                    if z['gain'] is not None:c['pred_gain_sum']+=z['gain']
                div=False
                for x,y in zip(a['rows'],b['rows']):
                    assert x['history']==y['history'] and sem(x)==sem(y)
                    c['common_roots']+=1;c['common_physical_delta']+=x['actual_nn']-y['actual_nn']
                    if x['arm']=='Q':
                        c['common_q_roots']+=1
                        if 'retention67' in x:
                            z=x['retention67'];assert z['b']==y['played'] and z['chosen']==x['played'];c['common_discord']+=z['a']!=z['b'];c['common_veto']+=z['a']!=z['b'] and z['chosen']==z['a']
                    if x['played']!=y['played']:
                        assert x['arm']=='Q';c['first_divergence']+=1;div=True;break
                if not div:assert a['history']==b['history']
        assert c['episodes']==(336 if mode=='MAIN' else 72) and c['same_history']+c['first_divergence']==c['episodes'] and c['common_physical_delta']==0
        out[arm]=dict(c)
    z=dict(mode=mode,arms=out,lock_sha256=sha(T/(mode+'_LOCK.json')))
    (T/(mode+'_MECHANISM.json')).write_text(json.dumps(z,indent=2,sort_keys=True)+'\n');print(mode,json.dumps(z),flush=True)
if __name__=='__main__':main(sys.argv[1])
