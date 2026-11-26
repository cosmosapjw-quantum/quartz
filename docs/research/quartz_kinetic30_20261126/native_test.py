from core import *
r=json.loads((R/'quartz_kinetic29_20261126/COHORT.json').read_text())['roots'][0]
with Server(pathlib.Path(sys.argv[1]),8) as s:
    for c in (3,4,5,7,5):
        a=s.request(r['history'],arm=c,trace=True);assert a.get('schema')=='quartz_k30_c4_v1',a
        q=a['queue'];assert q['logical']==a['NN_calls']-a['mid']['nn']==q['physical']+q['coalesced']
        assert q['parity_checks']==q['physical'] and q['peak_pending']<=8
        assert q['kernel_lanes']==sum(i*x for i,x in enumerate(q['kernel_hist']))
        assert q['physical']<=q['kernel_lanes']<=8*q['batches']
        assert a['response']==2*a['z']+a['gap_flag'];assert q['specialized']==(c!=3)
        if c==3:assert q['kernel_lanes']==8*q['batches']
        print('batch',c,q['epoch'],q['logical'],q['physical'],q['kernel_lanes'],q['batches'],flush=True)
