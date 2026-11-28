from plan62 import D,T,read,sha,SEEDS,EXTRA
from eval62 import n,load_eval,target
from games62 import sem
import json,collections as C
def main():
    p=read(D/'PLAN.json');ev=load_eval(SEEDS[0]);ct=C.Counter();js=[j for j in p['roots'] if j['seed']==SEEDS[0]][:16]
    assert len(js)>=8
    with n.NNServer(ev,bin=p['old_binary']) as old,n.NNServer(ev,bin=T/'mcts_demo') as new:
        for j in js:
            h=j['history']
            for q in (False,True):
                x=old.search(h,q);y=new.search(h,q);assert sem(x)==sem(y);ct['old_new']+=1
            for hh in (4,8):
                x=old.early(h,hh,1);y=new.early(h,hh,1);assert sem(x)==sem(y);ct['old_new']+=1
            x=new.early(h,4,1);y=target(new,h)
            assert y['early']['at']==x['early']['at'] and y['target']['incumbent']==x['move'] and y['cache_receipt']['root_physical']==x['actual_nn'] and y['actual_nn']-x['actual_nn']<=4
            ct['target_prefix']+=1
        new.send(dict(cmd='research_gomoku7_k62',history=js[0]['history'],cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=8,op=7)));assert 'error' in new.read();ct['invalid_rejected']+=1
        new.send(dict(cmd='research_gomoku7_k57',history=js[0]['history'],cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=4,op=7)));assert 'error' in new.read();ct['invalid_rejected']+=1
    extras=[]
    for s in EXTRA:
        e=load_eval(s);pr,v=e(1<<24,1<<25,1,[i for i in range(49) if i not in (24,25)])
        assert e.rows==1;extras.append(dict(seed=s,sha256=p['models'][str(s)]['sha256'],value=v,prior_sum=float(pr.sum()),strict_load=True))
    out=dict(counts=dict(ct),physical_nn=ev.rows,extra_panel=extras,old_sha256=sha(p['old_binary']),new_sha256=sha(T/'mcts_demo'))
    (T/'PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
