from plan60 import D,T,P58,read,sha,SEEDS
import sys,json,collections as C
sys.path.insert(0,str(P58))
import native58 as n
n.T=T
from continuation60 import sem
def norm(x):
    return {k:v for k,v in sem(x).items() if k not in ('early','root_nn','low','atlas','mask','physical_nn','native_physical_nn')}
def adaptive(s,h,table):
    return s.cached_request(dict(cmd='research_gomoku7_k60',history=h,cap=20,quartz=True,trace=False,atlas=False,mask=0,early=dict(horizon=16,op=5),stop60=table),True,True)
def main():
    from k46_engine import load_eval
    p=read(D/'PLAN.json');ev=load_eval(SEEDS[0]);ct=C.Counter()
    # Real roots sampled from training panel, not heldout; same frozen NN used in both native binaries.
    js=[j for j in p['roots'] if j['seed']==SEEDS[0]][:12]
    assert len(js)>=6
    with n.NNServer(ev,bin=p['old_binary']) as old,n.NNServer(ev,bin=T/'mcts_demo') as new:
        for j in js:
            h=j['history']
            for q in (False,True):
                a=old.search(h,q);b=new.search(h,q);assert sem(a)==sem(b);ct['old_new']+=1
            for hh in (4,16):
                a=old.early(h,hh,1);b=new.early(h,hh,1);assert sem(a)==sem(b);ct['old_new']+=1
            a=new.search(h,True);b=adaptive(new,h,[False]*16)
            assert norm(a)==norm(b),(a,b);ct['all_false']+=1
            a=new.early(h,4,1);b=adaptive(new,h,[True]*16)
            if 'early' in a and a['early']['reached']:
                assert norm(a)==norm(b);ct['all_true_h4']+=1
            else:ct['h4_not_eligible']+=1
        for q in [dict(cmd='research_gomoku7_k60',history=js[0]['history'],cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=16,op=5)),dict(cmd='research_gomoku7_k57',history=js[0]['history'],cap=20,quartz=True,cache=True,mask=0,early=dict(horizon=4,op=1),stop60=[False]*16)]:
            new.send(q);a=new.read();assert 'error' in a and a.get('type')!='k47_eval';ct['invalid_rejected']+=1
    out=dict(counts=dict(ct),physical_nn=ev.rows,old_sha256=sha(p['old_binary']),new_sha256=sha(T/'mcts_demo'))
    (T/'PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
