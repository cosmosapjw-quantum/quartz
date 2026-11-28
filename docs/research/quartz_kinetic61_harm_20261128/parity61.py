from plan61 import D,T,P58,read,sha,SEEDS,feature
import sys,json,collections as C
sys.path.insert(0,str(P58));import native58 as n
n.T=T
from continuation61 import sem
def norm(x):return {k:v for k,v in sem(x).items() if k not in ('early','root_nn','low','atlas','mask','physical_nn','native_physical_nn')}
def adaptive(s,h,table):
    return s.cached_request(dict(cmd='research_gomoku7_k61',history=h,cap=20,quartz=True,trace=False,atlas=False,mask=0,early=dict(horizon=8,op=6),extend61=table),True,True)
def main():
    from k46_engine import load_eval
    p=read(D/'PLAN.json');ev=load_eval(SEEDS[0]);ct=C.Counter()
    js=[]
    for i in range(5):
        z=[j for j in p['roots'] if j['seed']==SEEDS[0] and j['cell']==i][:4];js+=z
    assert len(js)>=10
    with n.NNServer(ev,bin=p['old_binary']) as old,n.NNServer(ev,bin=T/'mcts_demo') as new:
        for j in js:
            h=j['history']
            for q in (False,True):
                x=old.search(h,q);y=new.search(h,q);assert sem(x)==sem(y);ct['old_new']+=1
            for hh in (4,8):
                x=old.early(h,hh,1);y=new.early(h,hh,1);assert sem(x)==sem(y);ct['old_new']+=1
            for table,hh,label in [([False]*5,4,'all_false_h4'),([True]*5,8,'all_true_h8')]:
                x=new.early(h,hh,1);y=adaptive(new,h,table)
                assert norm(x)==norm(y),(x,y)
                assert y['early']['snapshots'][0]['cell61']==j['cell'] and y['early']['snapshots'][0]['extend61']==table[j['cell']]
                ct[label]+=1
        for q in [dict(cmd='research_gomoku7_k61',history=js[0]['history'],cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=8,op=6)),dict(cmd='research_gomoku7_k57',history=js[0]['history'],cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=4,op=1),extend61=[False]*5)]:
            new.send(q);a=new.read();assert 'error' in a and a.get('type')!='k47_eval';ct['invalid_rejected']+=1
    out=dict(counts=dict(ct),physical_nn=ev.rows,old_sha256=sha(p['old_binary']),new_sha256=sha(T/'mcts_demo'))
    (T/'PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
