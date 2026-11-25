"""Deterministic tests on real prior snapshots; no model execution."""
import copy,gzip,json,sys,tempfile,hashlib
from pathlib import Path
from native_smoke import check
from collector import load_record,z_label
from engine import D,ROOT
OLD=ROOT/'docs/research/quartz_fouraxis06_20261125'
def rejects(fn):
    try:fn()
    except (AssertionError,ValueError,TypeError,KeyError):return
    raise AssertionError('corruption accepted')
def record():
    return json.loads((OLD/'C06_NATIVE_SMOKE.json').read_text())['search'][0]
def B():
    a=record();check(a,a['history'],a['rule'])
    mutations=[]
    v=copy.deepcopy(a);v['result']['edges'][0]['m2']=1e6;mutations.append(('impossible M2',v))
    v=copy.deepcopy(a);v['result']['candidates'].append(v['result']['candidates'][-1]);mutations.append(('duplicate candidate',v))
    v=copy.deepcopy(a);v['result']['candidates'][0][1]=-1.;mutations.append(('negative prior',v))
    v=copy.deepcopy(a);v['best']=next(m for m in v['legal'] if m!=v['result']['best']);mutations.append(('changed BASE recommendation',v))
    v=copy.deepcopy(a);v['nn_batches']=True;mutations.append(('boolean count',v))
    for nm,v in mutations:
        rejects(lambda:check(v,v['history'],v['rule']));print('rejected',nm)
def C():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        for nm,raw in [('duplicate field',b'{"job":{"game_id":"X"},"winner_black":1,"winner_black":0}'),('nonfinite',b'{"job":{"game_id":"X"},"v":NaN}')]:
            z=gzip.compress(raw,mtime=0);f=p/('X.'+hashlib.sha256(z).hexdigest()+'.json.gz');f.write_bytes(z)
            rejects(lambda:load_record(f));print('rejected',nm)
        rejects(lambda:z_label(False,1));rejects(lambda:z_label(1,True));print('rejected boolean WDL/player')
if __name__=='__main__':{'B':B,'C':C}[sys.argv[1]]()
