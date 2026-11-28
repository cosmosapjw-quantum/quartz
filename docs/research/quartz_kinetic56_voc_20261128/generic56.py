"""Actual frozen NN, history planes from strict native generic MctsEngine."""
import json,collections as C,numpy as np,torch,pathlib
from k56_plan import D,T,SEEDS,load,sha
from native56 import NNServer
from k46_engine import load_eval
def expected(h):
    p=1-2*(len(h)%2);x=np.zeros((17,49),np.float32)
    for t in range(8):
        for i,m in enumerate(h[:max(0,len(h)-t)]):x[2*t+int((1-2*(i%2))!=p),m]=1.
    x[16]=float(p==1);return x.reshape(-1).tolist()
def main():
    ct=C.Counter();rs=[]
    for seed in SEEDS:
        ev=load_eval(seed);identity=load('PLAN.json')['models'][str(seed)]['sha256']
        hs=[[16,17,23,24],[23,17,16,24]]
        for h in hs:
            with NNServer(None) as s,torch.inference_mode():
                s.send(dict(cmd='research_gomoku7_generic_k56',history=h,identity=identity));n=ev.rows;calls=0;reqs=[]
                while True:
                    a=s.read()
                    if a.get('type')!='k56_generic_eval':break
                    calls+=1;assert a['id']==calls and a['identity']==identity and len(a['planes'])==833
                    x=np.array(a['planes'],dtype=np.float32);assert np.isin(x,[0,1]).all()
                    legal=a['legal'];assert len(set(legal))==len(legal)
                    logits,v=ev.model(torch.from_numpy(x.reshape(1,17,7,7)))
                    q=logits[0,legal].softmax(0).tolist();v=float(v.item())
                    assert np.isfinite(q).all() and abs(sum(q)-1)<1e-5 and -1<=v<=1
                    s.send(dict(cmd='k56_generic_reply',id=calls,identity=identity,prior=q,value=v));reqs.append(a)
                assert 'error' not in a,a
                assert ev.rows-n==calls==a['physical'] and a['root_planes']==expected(h)
                assert a['plain']==a['first']==a['second'] and a['first_counts'][1]==a['second_counts'][1]
                assert a['second_counts'][0]-a['first_counts'][0]==a['second_counts'][2]-a['first_counts'][2]
                assert all(x['planes']==y['planes'] and x['legal']==y['legal'] for x,y in zip(reqs,a['trace'],strict=True))
                ct['roots']+=1;ct['physical']+=calls;ct['plain_calls']+=a['plain_calls'];ct['logical_cached']+=a['second_counts'][0];ct['cache_hits']+=a['second_counts'][2]
                rs.append(dict(seed=seed,**a))
        assert rs[-1]['root_planes']!=rs[-2]['root_planes']
    (T/'GENERIC.json').write_text(json.dumps(dict(counts=dict(ct),rows=rs,scope='actual full-history input, TT off, strict successful sync inference; no games or strength claim'),separators=(',',':'),allow_nan=False)+'\n')
    print('GENERIC',dict(ct),flush=True)
if __name__=='__main__':main()
