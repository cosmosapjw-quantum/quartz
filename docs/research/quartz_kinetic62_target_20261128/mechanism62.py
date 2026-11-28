from plan62 import D,T,read,raw,sha,ARMS
from collections import Counter
import json
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));out={}
    for arm in ('H8','TARGET'):
        c=Counter()
        for meta in lock['files']:
            p=T/meta['file'];assert sha(p)==meta['sha256']
            for g in raw(p):
                a,b=g[arm],g['H4'];c['episodes']+=1;c['equal_history']+=a['history']==b['history'];c['delta_score2']+=a['score2']-b['score2']
                c['delta_both_physical']+=a['actual_nn']-b['actual_nn']
                for r in a['rows']:
                    if 'target' in r:
                        c['target_events']+=1;c['target_switch']+=r['target']['chosen']!=r['target']['incumbent'];c['target_extra_physical']+=r['actual_nn']-r['cache_receipt']['root_physical']
                div=False
                for x,y in zip(a['rows'],b['rows']):
                    assert x['history']==y['history'];c['common_roots']+=1
                    c['common_physical_delta']+=x['actual_nn']-y['actual_nn']
                    if x['arm']=='Q':
                        c['common_q_physical_delta']+=x['actual_nn']-y['actual_nn']
                        if 'target' in x:
                            assert x['early']['at']==y['early']['at']
                            c['common_target_events']+=1;c['common_target_extra']+=x['actual_nn']-x['cache_receipt']['root_physical']
                    if x['played']!=y['played']:
                        assert x['arm']=='Q';c['first_divergence']+=1;div=True;break
                if not div:assert a['history']==b['history']
        assert c['episodes']==(896 if mode=='MAIN' else 96) and c['equal_history']+c['first_divergence']==c['episodes']
        out[arm]=dict(c)
    x=dict(mode=mode,arms=out,lock_sha256=sha(T/(mode+'_LOCK.json')),scope='Common prefix accounting only, downstream path dependent')
    (T/(mode+'_MECHANISM.json')).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(mode,json.dumps(x),flush=True)
if __name__=='__main__':
    import sys;main(sys.argv[1])
