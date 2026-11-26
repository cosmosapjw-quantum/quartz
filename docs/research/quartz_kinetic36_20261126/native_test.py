import json,sys,pathlib,os,subprocess
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2]
env=dict(os.environ,RAYON_NUM_THREADS='4',QUARTZ_DISABLE_QIPC_SHM='1',QUARTZ_RAPFI_EVAL_EXE=str(R.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'),QUARTZ_RAPFI_EVAL_WEIGHT=json.loads((D/'RAPFI_LOCK.json').read_text())['files'][3]['path'])
h=[112,113,97,127,98,126,82,143]
q=dict(cmd='research_rapfi_k36',history=h,cap=96,horizons=None,lock=True,seed=17,trace=True)
qs=[dict(cmd='research_rapfi_shared_open'),q,dict(q,cap=None,horizons=[96]*8),dict(q,lock=False),dict(cmd='research_rapfi_shared_close')]
p=subprocess.run([sys.argv[1],'--server'],input='\n'.join(map(json.dumps,qs))+'\n',text=True,capture_output=True,env=env,timeout=30)
a=[json.loads(x) for x in p.stdout.splitlines() if x.startswith('{')]
assert len(a)==5 and a[1].get('schema')=='quartz_rapfi_k36_v1',('K36 horizon/probe endpoint unavailable',len(a),p.returncode)
x,y,z=a[1:4]
assert x['cap']==y['cap']==z['cap']==96 and x['best']==y['best']==z['best']
assert x['result']==y['result'] and x['NN_calls']==y['NN_calls']<=z['NN_calls']<=96
assert x['probe']==y['probe']==z['probe'] and x['probe']['root_n']<=39
print('GREEN actual cap96, common probe/table and lock parity',x['bin'],x['NN_calls'],z['NN_calls'],x['best'])
