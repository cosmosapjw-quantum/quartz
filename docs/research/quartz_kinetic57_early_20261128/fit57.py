"""Fixed family TRAIN fit, actual predecision flip and signed effect."""
import collections as C,json
from fractions import Fraction as F
from k57_plan import D,T,load,save,sha
from experiment57 import rows
def cell(r):return f"{r['horizon']}:{int(r['distance']>2)}"
def main():
    out={};n=0;nr=0
    for z in rows('NN'):
        c=z['control']
        if c['job']['split']!='TRAIN':continue
        n+=1;nr+=c['event'] is not None
    assert n==32*14
    for arm in ('cold','reuse'):
        cells=C.defaultdict(C.Counter);total=C.Counter()
        for z in rows('NN'):
            if z['control']['job']['split']!='TRAIN':continue
            r=z['pairs'][arm]
            if not r['reached']:continue
            assert not r['delta2'] or r['changed']
            for a in (cells[cell(r)],total):
                a['n']+=1;a['flip']+=r['changed'];a['gain2']+=r['delta2'];a['cost']+=r['local_nn']
        assert total['n']==nr and nr>0
        def model(a):
            p=F(a['flip']+1,a['n']+2)
            m=F(a['gain2'],2*(a['flip']+4))
            return dict(n=a['n'],flip=a['flip'],gain2=a['gain2'],cost=a['cost'],p=float(p),m=float(m),mu=float(p*m))
        out[arm]=dict(total=dict(total),constant=dict(p=total['flip']/nr,mu=total['gain2']/(2*nr)),cells={f'{h}:{k}':model(cells[f'{h}:{k}']) for h in (4,8,12,16) for k in (0,1)})
    o=dict(arms=out,episodes=n,reached=nr,lock=sha(D/'NN_LOCK.json'),scope='TRAIN only. Beta(1,1) flip; signed changed-score mean with4 zero pseudo-observations; replay choose highest predicted positive mu else STOP, tie lower op; no live deployment or CAL/TEST tuning')
    save('FROZEN.json',o);print('FIT',json.dumps(o),flush=True)
if __name__=='__main__':main()
