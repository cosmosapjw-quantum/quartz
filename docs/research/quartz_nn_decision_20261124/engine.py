"""Real SSOT resident MCTS sessions with trained neural evaluation and diagnostics."""
import os,time,json,hashlib
from pathlib import Path
os.environ['QUARTZ_DISABLE_COMPILE']='1'
os.environ['QUARTZ_DISABLE_QIPC_SHM']='1'
os.environ['QUARTZ_DISABLE_ASYNC_PIPELINE']='1'
os.environ['QUARTZ_DISABLE_NN_CACHE']='1'  # isolate actual neural work across arms
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import torch
from quartz.models_torch import AlphaZeroNet
from quartz.training_catalog import GAME_CONFIGS
from quartz.encoders import get_encoder
from quartz.runtime_support import NNSearchClient
from quartz.selfplay_runtime import rust_search_options
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
B=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_nn_decision_backups')
BIN=Path('/home/cosmosapjw/quantum/research/quartz_nn_decision_target_20261124/release/mcts_demo')
MODELS={'gomoku7':B/'gomoku7_seed41_latest.pt','gomoku15':B/'gomoku15_F1_latest.pt'}
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Engine:
    def __init__(self,game):
        torch.set_num_threads(2)
        # Set once per process before first inference, or preserve established value.
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        self.game=game
        raw=torch.load(MODELS[game],map_location='cpu',weights_only=True)
        cfg=dict(GAME_CONFIGS[game])
        for k in ('filters','blocks','vh','ch'):
            cfg[k]=raw['cfg'][k]
        cfg.update(_name=game,_encoder=get_encoder(game),search_profile='baseline_strict',penalty_mode='None',vl_mode='disabled',root_only_shaping=True,seed=17,n_threads=2,batch_size=4,iters=8)
        self.cfg=cfg
        self.model=AlphaZeroNet(cfg).cpu()
        self.model.load_state_dict(raw['model_state_dict'],strict=True);self.model.eval()
        self.nn_batches=0;self.nn_rows=0
        def hook(m,args,out):
            self.nn_batches+=1;self.nn_rows+=int(args[0].shape[0])
        self.hook=self.model.register_forward_hook(hook)
        self.client=NNSearchClient(self.model,cfg,torch.device('cpu'),str(BIN))
        self.client.start()
        self.active=set()
    def _counts(self):return self.nn_batches,self.nn_rows
    def request(self,q):
        before=self._counts();t=time.perf_counter_ns()
        ans=self.client._exchange_search_request(q)
        ns=time.perf_counter_ns()-t
        if not isinstance(ans,dict) or ans.get('error') or not ans.get('results'):
            raise RuntimeError(f'bad resident reply {str(ans)[:600]}')
        if ans.get('response_schema')!='quartz_resident_diag_json_v1':
            raise RuntimeError(f'missing native diagnostic schema: {list(ans)}')
        if len(ans['results'])!=1 or not isinstance(ans['results'][0],dict):
            raise RuntimeError(f'expected single active resident result: {ans}')
        after=self._counts();row=dict(ans['results'][0])
        row.update(wall_ns=ns,actual_nn_batches=after[0]-before[0],actual_nn_rows=after[1]-before[1])
        assert row['root_visits']>=0 and row['evaluator_calls']>=0
        return int(ans['session_id']),row
    def open(self,s):
        q=dict(cmd='search_nn_multi_engine_session_open',game=self.game,iters=8,jobs=[dict(board=s['board'],player=s['player'])],diagnostic_json=True)
        q.update(rust_search_options(self.cfg,penalty_mode='None'))
        sid,row=self.request(q);self.active.add(sid);self.check(s,row,8)
        return sid,row
    def step(self,s,sid,old):
        new_sid,row=self.request(dict(cmd='search_nn_multi_engine_session_step',session_id=sid,updates=[],iters=16,diagnostic_json=True))
        assert new_sid==sid and row['root_hash']==old['root_hash']
        self.check(s,row,24)
        assert row['additional_root_visits']==row['root_visits']-old['root_visits']
        assert row['additional_evaluator_calls']==row['evaluator_calls']-old['evaluator_calls']
        return row
    @staticmethod
    def check(s,row,visits):
        assert row['root_visits']==visits,(row['root_visits'],visits,row)
        assert 0<=row['best_move']<len(s['board']) and s['board'][row['best_move']]==0
        assert row['actual_nn_batches']>=0 and row['actual_nn_rows']>=row['actual_nn_batches']
    def close(self,sid):
        if sid in self.active:
            res=self.client.close_search_session(sid)
            self.active.remove(sid)
            return res
    def stop(self):
        for sid in list(self.active):self.close(sid)
        self.hook.remove();self.client.stop()
