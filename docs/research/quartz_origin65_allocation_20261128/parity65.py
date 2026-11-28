from plan65 import D,T,read,sha,SEEDS,EXTRA,NOVEL,DEV
from eval65 import n,load_eval,select,target,marginal,allocation
from audit65 import check_new,prefix,a
import collections as C,json
def norm(x):
    if isinstance(x,dict):return {k:norm(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [norm(v) for v in x]
    return x
def main():
    p=read(D/'PLAN.json');ct=C.Counter();physical=0
    jobs=[next(r for r in p['roots'] if r['domain']=='MAIN' and r['cell']==i) for i in range(6)]
    for j in jobs:
        ev=load_eval(j['seed']);h=j['history']
        with n.NNServer(ev,bin=p['old_binary']) as old,n.NNServer(ev,bin=T/'mcts_demo') as new:
            for f in (lambda s:s.search(h,False),lambda s:s.search(h,True),lambda s:s.early(h,4,1),lambda s:select(s,h),lambda s:target(s,h),lambda s:marginal(s,h),lambda s:marginal(s,h,[0]*6)):
                assert norm(f(old))==norm(f(new));ct['old_new']+=1
            rs={k:allocation(new,h,i) for i,k in enumerate(DEV)}
            for k,r in rs.items():check_new(h,r,k);ct['new']+=1
            assert rs['L4']['allocation65']['edges']==j['snapshot']['edges'] and rs['L4']['move']==j['snapshot']['move']
            for x,y in [('L4','A4'),('M4','MA4')]:
                assert prefix(rs[x])==prefix(rs[y]) and rs[x]['cache_receipt']==rs[y]['cache_receipt'];ct['paired_prefix']+=1
            q=dict(cmd='research_gomoku7_origin65',history=h,cap=4,quartz=True,cache=True,allocation=0)
            for patch in (dict(cap=8),dict(quartz=False),dict(cache=False),dict(allocation=5),dict(bogus=1)):
                new.send(dict(q,**patch));assert 'error' in new.read();ct['invalid_rejected']+=1
        physical+=ev.rows
    extras=[]
    for s in SEEDS+EXTRA+NOVEL:
        ev=load_eval(s);pr,v=ev(1<<24,1<<25,1,[i for i in range(49) if i not in (24,25)])
        extras.append(dict(seed=s,strict_load=True,sha256=p['models'][str(s)]['sha256'],value=v,prior_sum=float(pr.sum())))
    out=dict(counts=dict(ct),physical_nn=physical,models=extras,old_sha256=sha(p['old_binary']),new_sha256=sha(T/'mcts_demo'))
    (T/'PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
