import json,gzip,collections as C,pathlib
from k48_plan import D,T,SEEDS,load,save,sha
from native48 import NNServer,BIN
from k46_engine import load_eval
OLD=D.parent/'quartz_kinetic47_native_20261127';OB=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic47_native_backups/mcts_demo'
def raw(p):
 with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def main():
 lk=json.loads((OLD/'NN_PARITY_LOCK.json').read_text());p=OLD/lk['file'];assert sha(p)==lk['sha256'] and sha(OB)==lk['binary']
 xs=list(raw(p));ct=C.Counter();out=T/'PARITY.jsonl.gz';assert not out.exists()
 fields=('move','kind','proof','cap','lock','nn','visits','terminal_leaves','votes','distance','prior_best','original_move','replacement')
 with gzip.open(out,'wt',compresslevel=3) as f:
  for seed in SEEDS:
   hs=[r['history'] for r in xs if r['seed']==seed and r['quartz']][:8]
   ev=load_eval(seed)
   with NNServer(ev,OB) as old,NNServer(ev) as new:
    for h in hs:
     low=old.search(h,True,trace=True,cap=20);high=old.search(h,True,trace=True,cap=32);a=new.nested(h,atlas=True,trace=True)
     for k in fields:assert a[k]==low[k],(seed,h,k,a[k],low[k])
     if a.get('low'):
      for k in ('move','kind','nn','visits','terminal_leaves','votes','distance','prior_best','original_move','replacement','cap'):assert a['high'][k]==high[k],(h,k,a['high'][k],high[k])
      assert a['physical_nn']==high['nn'] and a['eval_trace']==high['eval_trace'] and a['eval_trace'][:low['nn']]==low['eval_trace']
      for mask in (0,255):
       r=new.nested(h,mask=mask)
       exp=low if mask==0 else high
       for k in fields:assert r[k]==exp[k],(mask,k,r[k],exp[k])
       ct['adaptive_checks']+=1
       ct['native_nn']+=r['nn']
      ct['nested']+=1;ct['additional_nn']+=high['nn']-low['nn']
     else:assert low['nn']==high['nn']==a['nn']==0
     ct['roots']+=1;ct['ref_nn']+=low['nn']+high['nn'];ct['native_nn']+=a.get('physical_nn',a['nn'])
     f.write(json.dumps(dict(seed=seed,history=h,low=low,high=high,native=a),separators=(',',':'))+'\n')
 assert ct['roots']==56
 (T/'PARITY_LOCK.json').write_text(json.dumps(dict(counts=dict(ct),file=out.name,sha256=sha(out),binary=sha(BIN),old_binary=sha(OB)),indent=2)+'\n');print('PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
