from plan64 import D,T,read,raw,sha
from collections import Counter
import json
def main(mode):
    lock=read(T/(mode+'_LOCK.json'));assert lock['table']==sha(D/'TABLE.json');out={}
    for arm in ('H4','T2','T4','K63','ROBUST'):
        c=Counter();cells=Counter();ops=Counter()
        for meta in lock['files']:
            p=T/meta['file'];assert sha(p)==meta['sha256']
            for g in raw(p):
                a,b=g[arm],g['FREE'];c['episodes']+=1;c['equal_history']+=a['history']==b['history'];c['delta_score2']+=a['score2']-b['score2'];c['delta_both_physical']+=a['actual_nn']-b['actual_nn']
                c['improved']+=a['score2']>b['score2'];c['worsened']+=a['score2']<b['score2']
                for r in a['rows']:
                    if 'select64' in r:
                        z=r['select64'];ops[z['op']]+=1;cells[f"{z['cell']}:{z['op']}"]+=1;c['eligible']+=1;c['changed_free']+=z['chosen']!=z['free'];c['extra_physical']+=z['extra_physical']
                    if 'target' in r:
                        c['target_events']+=1;c['target_extra_physical']+=r['actual_nn']-r['cache_receipt']['root_physical']
                div=False
                for x,y in zip(a['rows'],b['rows']):
                    assert x['history']==y['history'];c['common_roots']+=1;c['common_physical_delta']+=x['actual_nn']-y['actual_nn']
                    if x['arm']=='Q':
                        c['common_q_physical_delta']+=x['actual_nn']-y['actual_nn']
                        if 'select64' in x:
                            z=x['select64'];assert x['early']['at']==y['early']['at'] and z['free']==y['move'];c['common_select_events']+=1
                            assert x['actual_nn']-y['actual_nn']==z['extra_physical'];c['common_select_extra']+=z['extra_physical']
                        if 'target' in x:
                            assert x['early']['at']==y['early']['at'];c['common_target_events']+=1;c['common_target_extra']+=x['actual_nn']-x['cache_receipt']['root_physical']
                    if x['played']!=y['played']:
                        assert x['arm']=='Q';c['first_divergence']+=1;div=True;break
                if not div:assert a['history']==b['history']
        assert c['episodes']=={'MAIN':448,'ARCHIVE':192,'NOVEL':144}[mode] and c['equal_history']+c['first_divergence']==c['episodes']
        out[arm]=dict(counts=dict(c),ops=dict(ops),cells=dict(cells))
    x=dict(mode=mode,arms=out,lock_sha256=sha(T/(mode+'_LOCK.json')),scope='common-prefix accounting vs FREE; downstream path-dependent')
    (T/(mode+'_MECHANISM.json')).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(mode,json.dumps(x),flush=True)
if __name__=='__main__':
    import sys;main(sys.argv[1])
