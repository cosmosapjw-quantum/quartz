"""Actual strict history-plane protocol across model requests and isolated malformed reply.
No claim that the EpochMemo57 seam is integrated into Strict56 or async training."""
import json,collections as C,numpy as np,torch
from k58_plan import D,T,load,sha
from native58 import NNServer
from k46_engine import load_eval
def infer(ev,req):
    x=np.array(req['planes'],dtype=np.float32);legal=req['legal']
    assert len(x)==833 and len(set(legal))==len(legal)
    with torch.inference_mode():
        logits,v=ev.model(torch.from_numpy(x.reshape(1,17,7,7)))
        q=logits[0,legal].softmax(0).tolist();v=float(v.item())
    assert np.isfinite(q).all() and abs(sum(q)-1)<1e-5 and -1<=v<=1
    return dict(cmd='k56_generic_reply',id=req['id'],identity=req['identity'],prior=q,value=v)
def run(s,h,seed,ev):
    model=load('PLAN.json')['models'][str(seed)];assert sha(__import__('pathlib').Path(model['path']))==model['sha256']
    identity=model['sha256'];s.send(dict(cmd='research_gomoku7_generic_k56',history=h,identity=identity));n=0
    while True:
        a=s.read()
        if a.get('type')!='k56_generic_eval':break
        n+=1;assert a['id']==n and a['identity']==identity
        s.send(infer(ev,a))
    assert 'error' not in a and a['identity']==identity and a['physical']==n==len(a['trace'])
    assert a['plain']==a['first']==a['second'] and a['first_counts'][1]==a['second_counts'][1]
    return dict(seed=seed,**a)
def main():
    h=[16,17,23,24];ev={s:load_eval(s) for s in (101,102)}
    with NNServer(None) as srv:
        a,b,c=[run(srv,h,k,ev[k]) for k in (101,102,101)]
    assert a['root_planes']==b['root_planes']==c['root_planes']
    assert a['trace']==c['trace'] and a['plain']==c['plain']
    assert a['trace'][0]['prior']!=b['trace'][0]['prior'] and a['identity']!=b['identity']
    # An isolated process receives a genuine forward but a deliberately wrong identity in its reply.
    s=NNServer(None);identity=a['identity'];bad=None
    try:
        s.send(dict(cmd='research_gomoku7_generic_k56',history=h,identity=identity));r=s.read()
        assert r['type']=='k56_generic_eval' and r['id']==1
        q=infer(ev[101],r);q['identity']=b['identity'];s.send(q)
        try:
            bad=s.read()
        except RuntimeError as e:
            bad=str(e)
        assert isinstance(bad,str) and 'EOF' in bad, bad
        rc=s.p.wait(timeout=5);assert rc!=0
    finally:s.abort()
    with NNServer(None) as srv:d=run(srv,h,101,ev[101])
    assert d['trace']==a['trace'] and d['plain']==a['plain']
    o=dict(rows=[a,b,c,d],counts=dict(successful_requests=4,physical_success=sum(x['physical'] for x in (a,b,c,d)),physical_rejected=1,cache_hits=sum(x['second_counts'][2] for x in (a,b,c,d))),wrong_identity=dict(expected=identity,sent=b['identity'],observed=bad,returncode=rc),scope='request-local strict generation A/B/A isolation, rejected wrong identity terminates isolated server; fresh-process restart parity. EpochMemo57 Rust seam not wired into strict endpoint; not in-process recovery or async.')
    (T/'GENERIC.json').write_text(json.dumps(o,separators=(',',':'),allow_nan=False)+'\n')
    print('GENERIC',o['counts'],o['wrong_identity'],flush=True)
if __name__=='__main__':main()
