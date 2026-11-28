from plan66 import D,T,read,sha,SEEDS,EXTRA,NOVEL
from eval66 import n,load_eval,select,target,marginal,allocation,operation
from audit66 import check_operation,old,sem
import collections as C,json
def main():
    p=read(D/'PLAN.json');ct=C.Counter();physical=0
    jobs=[next(r for r in p['roots'] if r['domain']==d and r['cell']==c) for d in ('MAIN','ARCHIVE') for c in ('01','10','11')]
    for j in jobs:
        ev=load_eval(j['seed']);h=j['history']
        with n.NNServer(ev,bin=p['old_binary']) as a,n.NNServer(ev,bin=T/'mcts_demo') as b:
            for f in (lambda s:s.search(h,False),lambda s:s.search(h,True),lambda s:s.early(h,4,1),lambda s:select(s,h),*(lambda s,i=i:allocation(s,h,i) for i in range(5))):
                assert sem(f(a))==sem(f(b));ct['old_new']+=1
            m=allocation(b,h,3);assert m['allocation65']==j['snapshot']
            for arm,mode in [('FOCUS',5),('EXPAND',6)]:
                r=operation(b,h,mode);check_operation(h,r,arm);ct['operation']+=1
                assert r['operation66']['prefix_edges']==m['allocation65']['edges'] and r['allocation65']['steps'][:4]==m['allocation65']['steps'] and r['operation66']['prefix_move']==m['move'];ct['prefix']+=1
                if r['operation66']['target'] is None:assert r['move']==m['move'];ct['noop']+=1
            q=dict(cmd='research_gomoku7_origin66',history=h,cap=5,quartz=True,cache=True,allocation=5)
            for patch in (dict(cap=4),dict(quartz=False),dict(cache=False),dict(allocation=4),dict(allocation=7),dict(bogus=1)):
                b.send(dict(q,**patch));assert 'error' in b.read();ct['invalid_rejected']+=1
        physical+=ev.rows
    extras=[]
    for s in SEEDS+EXTRA+NOVEL:
        ev=load_eval(s);pr,v=ev(1<<24,1<<25,1,[i for i in range(49) if i not in (24,25)])
        extras.append(dict(seed=s,strict_load=True,sha256=p['models'][str(s)]['sha256'],value=v,prior_sum=float(pr.sum())))
    out=dict(counts=dict(ct),physical_nn=physical,models=extras,old_sha256=sha(p['old_binary']),new_sha256=sha(T/'mcts_demo'))
    (T/'PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
