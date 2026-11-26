import pathlib,sys,os,json
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(QUARTZ_NO_COMPILE='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,torch
from scripts.controller_sweep import build_base_cfg,build_anchor_candidates,apply_runtime_overrides,_new_search_client
torch.set_num_threads(1);torch.set_num_interop_threads(1)
p=json.loads((D/'PLAN.json').read_text());cfg,dev=build_base_cfg('gomoku7','cpu',32);cfg=apply_runtime_overrides(cfg,build_anchor_candidates(cfg)[0]['overrides'])
c=_new_search_client(p['models']['101']['path'],cfg,dev,p['binary']['path'])
try:
    b=np.zeros(49,dtype=np.int8)
    for i,m in enumerate([23,0,25,48]):b[m]=1 if i%2==0 else -1
    r=c.search_move(b,1,penalty_mode=cfg['penalty_mode']);print(json.dumps(r['realized_budget']),flush=True)
    assert r['realized_budget']['evaluator_calls']==0,'old A1 does not implement exact pre-search WIN3 certificate'
finally:c.stop()
