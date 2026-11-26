import json,sys,pathlib,os,subprocess
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2]
env=dict(os.environ,RAYON_NUM_THREADS='4',QUARTZ_DISABLE_QIPC_SHM='1',QUARTZ_RAPFI_EVAL_EXE=str(R.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'),QUARTZ_RAPFI_EVAL_WEIGHT=json.loads((D/'RAPFI_LOCK.json').read_text())['files'][3]['path'])
h=[112,113,97,127,98,126,82,143]
q=dict(cmd='research_rapfi_k35',history=h,mask=4,cap=80,threads=1,lock=False,seed=17,trace=True)
qs=[dict(cmd='research_rapfi_shared_open'),q,dict(q,mask=None,table=[4,4,4,4]),dict(cmd='research_rapfi_shared_close')]
p=subprocess.run([sys.argv[1],'--server'],input='\n'.join(map(json.dumps,qs))+'\n',text=True,capture_output=True,env=env,timeout=30)
a=[json.loads(x) for x in p.stdout.splitlines() if x.startswith('{')]
assert len(a)==4 and a[1].get('schema')=='quartz_rapfi_k35_v1',('K35 native mask/table unavailable',len(a),p.returncode)
x,y=a[1:3]
assert x['selected_mask']==y['selected_mask']==4 and x['result']==y['result'] and x['NN_calls']==y['NN_calls']<=80
assert x['prefix']['result']==y['prefix']['result'] and x['bridge_wait_ns']>=0
print('GREEN real mask4 and table parity',x['bin'],x['NN_calls'],x['best'])
