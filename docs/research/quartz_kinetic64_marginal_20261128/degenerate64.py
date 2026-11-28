from plan64 import D,T,read,raw,sha
from collections import Counter
import json
def norm(z):
    if isinstance(z,dict):return {k:norm(v) for k,v in z.items() if not k.endswith('_ns') and k not in ('select63','select64')}
    if isinstance(z,list):return [norm(v) for v in z]
    return z
def row(r):
    z=norm(r)
    if 'early' in z:z['early']['op']=1
    return z
def main():
    tab=read(D/'TABLE.json')['table'];assert tab==[0]*6
    out={}
    for mode in ('MAIN','ARCHIVE','NOVEL'):
        lock=read(T/(mode+'_LOCK.json'));assert lock['table']==sha(D/'TABLE.json');ct=Counter()
        for meta in lock['files']:
            f=T/meta['file'];assert sha(f)==meta['sha256']
            for gs in raw(f):
                a,b=gs['FREE'],gs['ROBUST'];assert a['history']==b['history'] and a['score2']==b['score2'] and a['winner']==b['winner']
                assert a['nn']==b['nn'] and a['actual_nn']==b['actual_nn'] and a['targets']==b['targets']==0
                assert len(a['rows'])==len(b['rows'])
                for x,y in zip(a['rows'],b['rows']):
                    assert row(x)==row(y)
                    if 'select64' in y:
                        z=y['select64'];assert z['op']==0 and z['chosen']==z['free']==x['move'] and z['extra_logical']==z['extra_physical']==0
                        ct['eligible']+=1
                    ct['rows']+=1;ct['physical']+=y['actual_nn']
                ct['games']+=1
        assert ct['games']=={'MAIN':448,'ARCHIVE':192,'NOVEL':144}[mode]
        out[mode]=dict(ct)
    (T/'DEGENERACY.json').write_text(json.dumps(dict(table=tab,panels=out,scope='full semantic receipts excluding timings and policy metadata'),indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
