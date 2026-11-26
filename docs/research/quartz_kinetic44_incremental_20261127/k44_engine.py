"""Reference PUCT using original trained Quartz NN, with exact finite-vote STOP."""
import pathlib,sys,os,time,math
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',QUARTZ_NO_COMPILE='1')
from k44_plan import D,ROOT,load,sha
sys.path.insert(0,str(ROOT))
# Load precisely the frozen rule module without modifying the child import path.
import importlib.util
assert D.name=='quartz_kinetic44_incremental_20261127', D
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

class Session:
    def __init__(self,ev,h,pr=None):
        self.ev=ev;self.t=time.perf_counter_ns();self.c=time.process_time_ns()
        self.initial=ev.rows;self.visits=self.terminal=0
        a,b=from_history(h);p=1-2*(len(h)%2)
        self.pr=(proof(a,b) if p==1 else proof(b,a)) if pr is None else pr
        self.rule_ns=time.perf_counter_ns()-self.t
        self.root=None;self.root_value=None
        if self.pr['move'] is None:
            self.root=Node(a,b,p);self.root_value,term=self.root.expand(ev);assert not term
    def advance(self,cap,lock):
        assert cap>0 and self.visits<=cap
        if self.root is None:
            assert self.ev.rows==self.initial
            return dict(move=self.pr['move'],kind='CERTIFICATE',proof=self.pr['kind'],cap=cap,lock=lock,nn=0,visits=0,terminal_leaves=0,votes=[],distance=None,prior_best=None,wall_ns=time.perf_counter_ns()-self.t,cpu_ns=time.process_time_ns()-self.c,rule_ns=self.rule_ns)
        root=self.root
        while self.visits<cap:
            if lock and lock_distance(root.legal,root.n)>cap-self.visits:break
            node=root;path=[]
            while node.legal is not None:
                i=node.select();path.append((node,i));node=node.descend(i)
            v,term=node.expand(self.ev);self.terminal+=int(term)
            for parent,i in reversed(path):
                v=-v;parent.n[i]+=1;parent.w[i]+=v
            self.visits+=1
        assert int(root.n.sum())==self.visits
        d=lock_distance(root.legal,root.n);leader=int(root.legal[np.argmax(root.n)])
        nn=self.ev.rows-self.initial;assert nn==1+self.visits-self.terminal
        return dict(move=leader,kind='VOTE_LOCK' if self.visits<cap else 'CAP',proof='UNRESOLVED',cap=cap,lock=lock,nn=nn,visits=self.visits,terminal_leaves=self.terminal,votes=[[int(i),int(n)] for i,n in zip(root.legal,root.n)],distance=d,prior_best=int(root.legal[np.argmax(root.prior)]),wall_ns=time.perf_counter_ns()-self.t,cpu_ns=time.process_time_ns()-self.c,rule_ns=self.rule_ns)
def bin_of(low):
    assert low['kind']!='CERTIFICATE'
    return int(low['distance']<=3)+2*int(low['move']!=low['prior_best'])
class Puct:
    def __init__(self,ev):self.ev=ev
    def search(self,h,cap,lock):
        return Session(self.ev,h).advance(cap,lock)
    def atlas(self,h):
        before=self.ev.rows
        b20=self.search(h,20,False);b32=self.search(h,32,False)
        ss=Session(self.ev,h);lo=ss.advance(20,True);hi=ss.advance(32,True)
        assert b20['move']==lo['move'] and b32['move']==hi['move']
        assert lo['nn']<=hi['nn']<=b32['nn'] and lo['nn']<=b20['nn']
        assert self.ev.rows-before==b20['nn']+b32['nn']+hi['nn']
        return dict(B20=b20,B32=b32,L20=lo,L32=hi,bin=None if lo['kind']=='CERTIFICATE' else bin_of(lo),actual_nn=self.ev.rows-before)
    def adaptive(self,h,mask):
        assert type(mask) is int and 0<=mask<16
        ss=Session(self.ev,h);lo=ss.advance(20,True)
        z=None if lo['kind']=='CERTIFICATE' else bin_of(lo)
        extend=z is not None and bool(mask>>z&1)
        out=ss.advance(32,True) if extend else lo.copy()
        out.update(bin=z,extended=extend,mask=mask,low_nn=lo['nn'],low_visits=lo['visits'],low_move=lo['move'],low_distance=lo['distance'],low_prior=lo['prior_best'])
        out['wall_ns']=time.perf_counter_ns()-ss.t;out['cpu_ns']=time.process_time_ns()-ss.c
        return out

def feature40(ss,lo):
    assert ss.root is not None and lo['kind']!='CERTIFICATE'
    pr=ss.root.prior
    positive=pr[pr>0]
    entropy=float(-(positive*np.log(positive)).sum()/math.log(len(pr))) if len(pr)>1 else 0.
    assert -1e-12<=entropy<=1+1e-12
    v=ss.root_value
    z=int(entropy>=.8)+2*int(lo['move']!=lo['prior_best'])+4*int(abs(v)>=.5)
    return z,entropy,v

def draw_h(h,policy,z):
    from fractions import Fraction as F
    import hashlib
    hh=policy.get(str(z),{'20':'1'})
    a,b=from_history(h)
    key=f"K40-2026112640:{a}:{b}:{len(h)%2}".encode()
    r=int.from_bytes(hashlib.sha256(key).digest(),'big')
    u=F(r,1<<256);cum=F(0)
    assert sum(F(v) for v in hh.values())==1
    for cap,p in sorted(hh.items(),key=lambda x:int(x[0])):
        cum+=F(p)
        if u<cum:return int(cap)
    raise AssertionError("probability mass")

def _atlas40(self,h):
    before=self.ev.rows
    b=self.search(h,32,False)
    ss=Session(self.ev,h);lo=ss.advance(20,True)
    if lo['kind']=='CERTIFICATE':
        return dict(B32=b,L20=lo,L26=ss.advance(26,True),L32=ss.advance(32,True),bin=None,entropy=None,root_value=None,actual_nn=0)
    z,e,v=feature40(ss,lo);mid=ss.advance(26,True);hi=ss.advance(32,True)
    assert hi['move']==b['move'] and lo['nn']<=mid['nn']<=hi['nn']<=b['nn']
    assert self.ev.rows-before==b['nn']+hi['nn']
    return dict(B32=b,L20=lo,L26=mid,L32=hi,bin=z,entropy=e,root_value=v,actual_nn=self.ev.rows-before)
def _adaptive40(self,h,policy):
    ss=Session(self.ev,h);lo=ss.advance(20,True)
    z=e=v=None;H=20
    if lo['kind']!='CERTIFICATE':
        z,e,v=feature40(ss,lo);H=draw_h(h,policy,z);assert H in (20,26,32)
    out=lo.copy() if H==20 else ss.advance(H,True)
    out.update(bin=z,entropy=e,root_value=v,selected_cap=H,low_nn=lo['nn'],low_visits=lo['visits'],low_move=lo['move'],low_prior=lo['prior_best'])
    out['wall_ns']=time.perf_counter_ns()-ss.t;out['cpu_ns']=time.process_time_ns()-ss.c
    return out
Puct.atlas40=_atlas40
Puct.adaptive40=_adaptive40

# K43 proof-aware policy: common-gate parity, sufficient partial proof,
# then ordinary locked MCTS with loss-only weak replacement.
def choose43(move,votes,partial):
    bad={m for m,l,u in partial['bounds'] if u==-1}
    if move not in bad:return move
    pool=[(m,v) for m,v in votes if m not in bad]
    if not pool:return move
    return max(pool,key=lambda z:(z[1],-z[0]))[0]
def _quartz44(self,h,depth=7,budget=4096):
    from inc44 import probe
    t=time.perf_counter_ns();c=time.process_time_ns()
    # This common gate has exactly the K33 selection priority.
    a,b=from_history(h);p=1-2*(len(h)%2);pr=proof(a,b) if p==1 else proof(b,a)
    gate_ns=time.perf_counter_ns()-t
    if pr['move'] is not None:
        r=Session(self.ev,h,pr).advance(32,True)
        r.update(partial=None,original_move=r['move'],replacement=False,proof_cpu_ns=0)
    else:
        zt=time.process_time_ns();z=probe(h,7,depth,budget);dt=time.process_time_ns()-zt;assert z['gate'] is None
        if z['cert'] is not None:
            r=dict(move=z['cert'],kind='PARTIAL44',proof='INTERVAL',cap=32,lock=True,nn=0,visits=0,terminal_leaves=0,votes=[],distance=None,prior_best=None,original_move=None,replacement=False)
        else:
            r=Session(self.ev,h,pr).advance(32,True);o=r['move'];r['move']=choose43(o,r['votes'],z);r.update(original_move=o,replacement=r['move']!=o)
        r.update(partial=z,proof_cpu_ns=dt)
    r.update(wall_ns=time.perf_counter_ns()-t,cpu_ns=time.process_time_ns()-c,rule_ns=gate_ns)
    return r
Puct.quartz44=_quartz44
