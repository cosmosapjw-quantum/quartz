from plan66 import D,T,read,raw,sha
from collections import Counter
import json,sys
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['plan']==sha(D/'PLAN.json')
    out={}
    for arm in ('FOCUS','EXPAND'):
        c=Counter()
        for m in lock['files']:
            f=T/m['file'];assert sha(f)==m['sha256']
            for gs in raw(f):
                a,b=gs[arm],gs['MA4'];c['episodes']+=1;c['same_history']+=a['history']==b['history'];c['better']+=a['score2']>b['score2'];c['worse']+=a['score2']<b['score2'];c['delta_score2']+=a['score2']-b['score2'];c['delta_both_physical']+=a['actual_nn']-b['actual_nn']
                for r in a['rows']:
                    if 'operation66' not in r:continue
                    z=r['operation66'];c['eligible']+=1;c['noop']+=z['target'] is None;c['extra_physical']+=z['extra_physical'];c['changed']+=r['move']!=z['prefix_move']
                div=False
                for x,y in zip(a['rows'],b['rows']):
                    assert x['history']==y['history'];c['common_roots']+=1;c['common_both_physical_delta']+=x['actual_nn']-y['actual_nn']
                    if x['arm']=='Q':
                        c['common_q_roots']+=1;c['common_q_physical_delta']+=x['actual_nn']-y['actual_nn']
                        if 'operation66' in x:
                            z=x['operation66'];s=x['allocation65'];t=y['allocation65']
                            assert z['prefix_edges']==t['edges'] and z['prefix_move']==y['move'] and s['steps'][:4]==t['steps'] and z['prefix_physical']==y['actual_nn']
                            assert x['actual_nn']-y['actual_nn']==z['extra_physical']
                            c['common_operation']+=1;c['common_extra']+=z['extra_physical'];c['common_noop']+=z['target'] is None
                    if x['played']!=y['played']:
                        assert x['arm']=='Q';c['first_divergence']+=1;div=True;break
                if not div:assert a['history']==b['history']
        assert c['episodes']=={'MAIN':336,'ARCHIVE':72,'NOVEL':72}[mode] and c['same_history']+c['first_divergence']==c['episodes']
        out[arm]=dict(c)
    r=dict(mode=mode,arms=out,lock_sha256=sha(T/(mode+'_LOCK.json')))
    (T/(mode+'_MECHANISM.json')).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(mode,json.dumps(r),flush=True)
if __name__=='__main__':main(sys.argv[1])
