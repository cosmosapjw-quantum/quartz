"""Mechanism audit: compare adaptive policy to fixed H4, excluding timing/diagnostic-only fields."""
from plan60 import D,T,read,raw,sha
from collections import Counter
import json
def norm(r):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_ns')}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    return clean({k:v for k,v in r.items() if k not in ('early','root_nn','low','atlas','mask','physical_nn','native_physical_nn')})
def main():
    lock=read(T/'ARENA_LOCK.json');assert lock['table']==sha(D/'TABLE.json')
    ct=Counter();hist=Counter();cells=Counter()
    for meta in lock['files']:
        p=T/meta['file'];assert sha(p)==meta['sha256']
        for gs in raw(p):
            a,b=gs['ADAPT'],gs['H4'];ct['episodes']+=1
            ct['equal_history']+=a['history']==b['history'];ct['equal_score']+=a['score2']==b['score2'];ct['equal_physical']+=a['actual_nn']==b['actual_nn']
            ct['delta_score2']+=a['score2']-b['score2'];ct['delta_physical']+=a['actual_nn']-b['actual_nn']
            for r in a['rows']:
                if 'early' in r:
                    e=r['early'];ct['adaptive_attempts']+=1
                    if e['reached']:hist[e['at']['visits']]+=1;cells[e['at']['cell60']]+=1
                    else:ct['adaptive_no_stop']+=1
            diverged=False
            for x,y in zip(a['rows'],b['rows']):
                assert x['history']==y['history']
                ct['common_roots']+=1
                if x['move']!=y['move']:
                    ct['first_different_move']+=1;diverged=True;break
                if norm(x)!=norm(y):ct['semantic_nonparity']+=1
                else:ct['semantic_parity']+=1
            if not diverged:assert a['history']==b['history']
    assert ct['episodes']==896 and ct['equal_history']+ct['first_different_move']==896
    out=dict(counts=dict(ct),stop_hist=dict(hist),stop_cells=dict(cells),lock_sha256=sha(T/'ARENA_LOCK.json'))
    (T/'MECHANISM.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
