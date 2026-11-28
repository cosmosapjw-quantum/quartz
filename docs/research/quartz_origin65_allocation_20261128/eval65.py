from plan65 import D,P58,T,read,sha,SEEDS,EXTRA,NOVEL
import sys
sys.path.insert(0,str(P58))
import native58 as n
n.T=T
from k46_engine import Evaluator,load_eval as old_load,AlphaZeroNet,GAME_CONFIGS,from_history,won,FULL,torch
def load_eval(seed):
    p=read(D/'PLAN.json')['models'][str(seed)]
    assert sha(p['path'])==p['sha256']
    if seed in SEEDS:return old_load(seed)
    assert seed in EXTRA+NOVEL and sha(p['backup'])==p['sha256']
    raw=torch.load(p['path'],map_location='cpu',weights_only=True)
    cfg=dict(GAME_CONFIGS['gomoku7']);meta=raw.get('cfg') or {}
    for k in ('blocks','filters','vh','ch','board','actions'):
        if k in meta:cfg[k]=meta[k]
    assert (cfg['ch'],cfg['board'],cfg['actions'])==(17,7,49)
    model=AlphaZeroNet(cfg);model.load_state_dict(raw.get('model_state_dict',raw),strict=True);model.eval()
    return Evaluator(model)
def target(s,h):
    return s.cached_request(dict(cmd='research_gomoku7_k62',history=h,cap=20,quartz=True,trace=False,atlas=False,mask=0,early=dict(horizon=4,op=7)),True,True)

def select(s,h,table=None):
    q=dict(cmd='research_gomoku7_k63',history=h,cap=20,quartz=True,trace=False,atlas=False,mask=0,early=dict(horizon=4,op=8 if table is None else 9))
    if table is not None:q['table63']=table
    return s.cached_request(q,True,True)

def marginal(s,h,table=None):
    q=dict(cmd='research_gomoku7_k64',history=h,cap=20,quartz=True,trace=False,atlas=False,mask=0,early=dict(horizon=4,op=10 if table is None else 11))
    if table is not None:q['table64']=table
    return s.cached_request(q,True,True)

def allocation(s,h,mode):
    return s.cached_request(dict(cmd='research_gomoku7_origin65',history=h,cap=4,quartz=True,allocation=mode),True)
