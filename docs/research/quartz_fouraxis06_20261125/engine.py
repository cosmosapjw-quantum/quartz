"""Prior real 17-plane Torch checkpoint and native research endpoint."""
import os,time,json,hashlib
from pathlib import Path
from rules import features,ADMITTED
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
MODEL=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz/results/smoke_torch/gomoku15/models/F1_legacy_base/latest.pt')
MODEL_SHA='57b93ba867ff8f9b74f91adef6b38f02f94de0bd7dc177785d668d32e1dcc3ae'
BIN=ROOT.parent/'quartz_fouraxis06_target_20261125/mcts_demo'
PYTHON='/home/cosmosapjw/wssg_p2_20261120/venv/bin/python'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity():
    x=json.loads((D/'C06_BUILD_IDENTITY.json').read_text())
    assert sha(BIN)==x['binary_sha256'] and sha(MODEL)==MODEL_SHA
    return dict(model_sha256=MODEL_SHA,binary_sha256=x['binary_sha256'],source_head=x['source_head'])
class Engine:
    def __init__(self,threads=1):
        identity();assert threads in (1,2)
        # Disable transport/model caches for independent observed NN counts.
        for k in ('QUARTZ_DISABLE_COMPILE','QUARTZ_DISABLE_QIPC_SHM','QUARTZ_DISABLE_ASYNC_PIPELINE','QUARTZ_DISABLE_NN_CACHE'):os.environ[k]='1'
        os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
        os.environ['RAYON_NUM_THREADS']=str(threads)
        import torch
        from quartz.models_torch import AlphaZeroNet
        from quartz.training_catalog import GAME_CONFIGS
        from quartz.encoders import get_encoder
        from quartz.runtime_support import NNSearchClient
        torch.set_num_threads(2)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        raw=torch.load(MODEL,map_location='cpu',weights_only=True)
        cfg=dict(GAME_CONFIGS['gomoku15'])
        for k in ('filters','blocks','vh','ch'):cfg[k]=raw['cfg'][k]
        cfg.update(_name='gomoku15',_encoder=get_encoder('gomoku15'),search_profile='baseline_strict',penalty_mode='None',vl_mode='disabled',root_only_shaping=True,seed=17,n_threads=threads,batch_size=4,iters=8)
        self.torch=torch;self.cfg=cfg;self.model=AlphaZeroNet(cfg).cpu()
        self.model.load_state_dict(raw['model_state_dict'],strict=True);self.model.eval()
        self.model_parameters=sum(p.numel() for p in self.model.parameters())
        self.rows=0;self.batches=0;self.first=None
        def hook(m,args,out):
            x=args[0].detach().cpu();self.batches+=1;self.rows+=int(x.shape[0])
            if self.first is None:self.first=x[0].reshape(-1).clone()
        self.hook=self.model.register_forward_hook(hook)
        self.client=NNSearchClient(self.model,cfg,torch.device('cpu'),str(BIN))
        self.client.start()
    def request(self,q):
        self.first=None;b=self.batches;r=self.rows
        def ticks():
            s=Path(f'/proc/{self.client.proc.pid}/stat').read_text().rsplit(')',1)[1].split()
            return int(s[11])+int(s[12])
        r0=ticks();p0=time.process_time_ns();t=time.perf_counter_ns()
        a=self.client._exchange_search_request(q)
        elapsed=time.perf_counter_ns()-t
        cpu_ns=time.process_time_ns()-p0;rust_ticks=ticks()-r0
        if not isinstance(a,dict):raise RuntimeError('native non-dict reply')
        if a.get('error'):return a
        if a.get('schema')!='quartz_gomoku_long_v1':raise RuntimeError(f'bad schema: {str(a)[:500]}')
        a['transport_ns']=elapsed;a['nn_batches']=self.batches-b;a['nn_rows']=self.rows-r
        a['python_cpu_ns']=cpu_ns;a['rust_cpu_ticks']=rust_ticks;a['tick_hz']=os.sysconf('SC_CLK_TCK')
        if 'evaluator_calls' in a:
            assert a['nn_rows']==a['evaluator_calls'],('NN boundary mismatch',a['nn_rows'],a['evaluator_calls'])
            if a['nn_rows']:
                expected=self.torch.tensor(features(q['history']),dtype=self.first.dtype)
                assert self.first.shape==expected.shape and self.torch.equal(self.first,expected),('actual root input differs',q['history'][:15])
                a['root_nn_feature_sha256']=hashlib.sha256(self.first.numpy().tobytes()).hexdigest()
            else:
                assert self.first is None;a['root_nn_feature_sha256']=None
        else:assert a['nn_rows']==a['nn_batches']==0
        return a
    def inspect(self,h,rule='freestyle'):
        return self.request(dict(cmd='research_gomoku15_long_inspect',history=list(h),rule=rule))
    def search(self,h,rule='freestyle',arm='BASE',visits=8,threads=1,seed=17):
        assert rule in ADMITTED
        a=self.request(dict(cmd='research_gomoku15_long_search',history=list(h),rule=rule,arm=arm,visits=visits,threads=threads,seed=seed))
        if a.get('error'):raise RuntimeError(a['error'])
        return a
    def close(self):
        self.hook.remove();self.client.stop()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
