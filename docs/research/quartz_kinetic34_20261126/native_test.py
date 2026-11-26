import sys,subprocess,json,os,pathlib
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
p=ROOT.parent
env=dict(os.environ,QUARTZ_RAPFI_EVAL_EXE=str(p/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'),QUARTZ_RAPFI_EVAL_WEIGHT=json.loads((D/'RAPFI_LOCK.json').read_text())['files'][3]['path'],RAYON_NUM_THREADS='4',QUARTZ_DISABLE_QIPC_SHM='1')
qs=[dict(cmd='research_rapfi_shared_open'),dict(cmd='research_rapfi_k34',history=[110,0,111,2,113,4],arm='F80L',seed=17,trace=True),dict(cmd='research_rapfi_shared_close')]
r=subprocess.run([sys.argv[1],'--server'],input='\n'.join(map(json.dumps,qs))+'\n',text=True,capture_output=True,env=env,timeout=30)
a=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')]
assert len(a)==3,('missing K34 response',len(a),r.returncode)
z=a[1];assert z.get('schema')=='quartz_rapfi_k34_v1',('missing K34 native certificate path',z.get('schema'),z.get('error'))
assert z['cause']=='WIN3' and z['NN_calls']==z['visits']==0
print('GREEN exact WIN3 bypass',z['best'],z['proof']['windows'])
