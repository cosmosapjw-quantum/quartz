from plan64 import D,T,read,sha,SEEDS,EXTRA,NOVEL,feature
from eval64 import n,load_eval,target,select,marginal
from games64 import sem
import json,collections as C
def normal(x):
    z=sem(x)
    if 'early' in z:z['early']['op']=1
    z.pop('select63',None);z.pop('select64',None)
    return z
def main():
    p=read(D/'PLAN.json');ct=C.Counter();physical=0
    js=[next(j for j in p['roots'] if j['domain']=='MAIN' and j['cell']==i) for i in range(6)]
    for j in js:
        ev=load_eval(j['seed']);h=j['history']
        with n.NNServer(ev,bin=p['old_binary']) as old,n.NNServer(ev,bin=T/'mcts_demo') as new:
            for q in (False,True):
                assert sem(old.search(h,q))==sem(new.search(h,q));ct['old_new']+=1
            for hh in (4,8):
                assert sem(old.early(h,hh,1))==sem(new.early(h,hh,1));ct['old_new']+=1
            for table in (None,[2,0,0,1,2,0]):
                assert sem(select(old,h,table))==sem(select(new,h,table));ct['old_new']+=1
            assert sem(target(old,h))==sem(target(new,h));ct['old_new']+=1
            f=select(new,h);z=marginal(new,h,[0]*6);assert normal(z)==normal(f);ct['free_parity']+=1
            y=marginal(new,h);w=marginal(new,h,[1]*6);assert normal(y)==normal(w) and y['actual_nn']-f['actual_nn']<=2;ct['t2_parity']+=1
            t=target(new,h);v=marginal(new,h,[2]*6);assert normal(v)==normal(t);ct['t4_parity']+=1
            assert y['select64']['cell']==j['cell']==feature(f['early']['at'],f['partial']['bounds'])
            for q in [dict(cmd='research_gomoku7_k64',history=h,cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=4,op=11)),dict(cmd='research_gomoku7_k64',history=h,cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=4,op=10),table64=[0]*6),dict(cmd='research_gomoku7_k64',history=h,cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=4,op=11),table64=[3]*6)]:
                new.send(q);assert 'error' in new.read();ct['invalid_rejected']+=1
        physical+=ev.rows
    extra=[]
    for s in EXTRA+NOVEL:
        ev=load_eval(s);pr,v=ev(1<<24,1<<25,1,[i for i in range(49) if i not in (24,25)])
        extra.append(dict(seed=s,strict_load=True,sha256=p['models'][str(s)]['sha256'],value=v,prior_sum=float(pr.sum())))
    out=dict(counts=dict(ct),physical_nn=physical,archive=extra,old_sha256=sha(p['old_binary']),new_sha256=sha(T/'mcts_demo'))
    (T/'PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
