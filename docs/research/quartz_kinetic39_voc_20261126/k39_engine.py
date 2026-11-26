"""Reference PUCT using original trained Quartz NN, with exact finite-vote STOP."""
import pathlib,sys,os,time,math
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',QUARTZ_NO_COMPILE='1')
from k39_plan import D,ROOT,load,sha
sys.path.insert(0,str(ROOT))
# Load precisely the frozen rule module without modifying the child import path.
import importlib.util
assert D.name=='quartz_kinetic39_voc_20261126', D
spec=importlib.util.spec_from_file_location('_k33_frozen_rule',D.parent/'quartz_kinetic33_20261126/proof.py')
rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
proof,from_history,won,FULL=rule.root_certificate,rule.from_history,rule.won,rule.FULL
import numpy as np
import torch
from quartz.models_torch import AlphaZeroNet
from quartz.encoders import GomokuEncoder
from quartz.training_catalog import GAME_CONFIGS
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
ENC=GomokuEncoder(7,4)
def lock_distance(actions,n):
    l=max(range(len(n)),key=lambda i:(int(n[i]),-int(actions[i])))
    return min((int(n[l])-int(n[i])+int(actions[i]>actions[l]) for i in range(len(n)) if i!=l),default=10**9)
class Evaluator:
    def __init__(self,model):
        self.model=model;self.rows=0
        self.model.register_forward_pre_hook(lambda m,x:setattr(self,'rows',self.rows+int(x[0].shape[0])))
    @torch.inference_mode()
    def __call__(self,a,b,p,legal):
        board=np.zeros(49,dtype=np.int8)
        for i in range(49):
            if a>>i&1:board[i]=1
            elif b>>i&1:board[i]=-1
        logits,v=self.model(torch.from_numpy(ENC.encode(board,p)).unsqueeze(0))
        q=logits[0,legal].to(torch.float64).softmax(0).numpy()
        v=float(v.item())
        assert np.isfinite(q).all() and abs(q.sum()-1)<1e-10 and -1<=v<=1
        return q,v
def load_eval(seed):
    lock=load('PLAN.json')['models'][str(seed)]
    p=pathlib.Path(lock['path']);assert sha(p)==lock['sha256']
    raw=torch.load(p,map_location='cpu',weights_only=True)
    state=raw.get('model_state_dict',raw)
    cfg=dict(GAME_CONFIGS['gomoku7'])
    meta=raw.get('cfg') or {}
    for k in ('blocks','filters','vh','ch','board','actions'):
        if k in meta:cfg[k]=meta[k]
    assert (cfg['ch'],cfg['board'],cfg['actions'])==(17,7,49)
    model=AlphaZeroNet(cfg);model.load_state_dict(state,strict=True);model.eval()
    return Evaluator(model)
class Node:
    def __init__(self,a,b,p):
        self.a=a;self.b=b;self.p=p;self.legal=None;self.n=None;self.w=None;self.prior=None;self.child={}
    def expand(self,ev):
        if won(self.b if self.p==1 else self.a):return -1.,True
        if self.a|self.b==FULL:return 0.,True
        self.legal=np.array([i for i in range(49) if not ((self.a|self.b)>>i&1)],dtype=np.int64)
        self.prior,v=ev(self.a,self.b,self.p,self.legal.tolist())
        self.n=np.zeros(len(self.legal),dtype=np.int64)
        self.w=np.zeros(len(self.legal),dtype=np.float64)
        return v,False
    def select(self):
        q=np.divide(self.w,self.n,out=np.zeros_like(self.w),where=self.n>0)
        return int(np.argmax(q+2.0*self.prior*math.sqrt(max(1,int(self.n.sum())))/(1+self.n)))
    def descend(self,i):
        if i not in self.child:
            m=int(self.legal[i]);a,b=self.a,self.b
            if self.p==1:a|=1<<m
            else:b|=1<<m
            self.child[i]=Node(a,b,-self.p)
        return self.child[i]
class Puct:
    def __init__(self,ev):self.ev=ev
    def search(self,h,cap,lock):
        assert cap>0
        t=time.perf_counter_ns();c=time.process_time_ns();initial=self.ev.rows
        a,b=from_history(h);p=1-2*(len(h)%2);pr=proof(a,b) if p==1 else proof(b,a)
        rule_ns=time.perf_counter_ns()-t
        if pr['move'] is not None:
            return dict(move=pr['move'],kind='CERTIFICATE',proof=pr['kind'],cap=cap,lock=lock,nn=0,visits=0,terminal_leaves=0,votes=[],distance=None,wall_ns=time.perf_counter_ns()-t,cpu_ns=time.process_time_ns()-c,rule_ns=rule_ns)
        root=Node(a,b,p);_,term=root.expand(self.ev);assert not term
        visits=terminal=0
        while visits<cap:
            d=lock_distance(root.legal,root.n)
            if lock and d>cap-visits:break
            node=root;path=[]
            while node.legal is not None:
                i=node.select();path.append((node,i));node=node.descend(i)
            v,term=node.expand(self.ev);terminal+=int(term)
            for parent,i in reversed(path):
                v=-v;parent.n[i]+=1;parent.w[i]+=v
            visits+=1
        assert int(root.n.sum())==visits
        d=lock_distance(root.legal,root.n);leader=int(root.legal[np.argmax(root.n)])
        nn=self.ev.rows-initial;assert nn==1+visits-terminal
        return dict(move=leader,kind='VOTE_LOCK' if visits<cap else 'CAP',proof='UNRESOLVED',cap=cap,lock=lock,nn=nn,visits=visits,terminal_leaves=terminal,votes=[[int(i),int(n)] for i,n in zip(root.legal,root.n)],distance=d,wall_ns=time.perf_counter_ns()-t,cpu_ns=time.process_time_ns()-c,rule_ns=rule_ns)
