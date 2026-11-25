import pathlib,sys,json,subprocess as SP,os
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
sys.path.insert(0,str(ROOT.parent/'quartz_kinetic17_20261126/docs/research/quartz_kinetic17_20261126'));import pilot as p
r=json.loads((ROOT/'docs/research/quartz_kinetic24_20261126/COHORT.json').read_text())['roots'][0]
q=dict(cmd='research_connect4_k25_cpu',history=r['history'],method='SERIAL',profile=15,seed=17,model_path=str(p.MODEL),model_identity=p.MID,trace=True)
s=SP.run([sys.argv[1],'--server'],input=json.dumps(q)+'\n',text=True,capture_output=True,timeout=30,env=dict(os.environ,RAYON_NUM_THREADS='2',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
assert s.returncode==0,s.stderr
a=json.loads(s.stdout.splitlines()[0]);print(a if 'schema' not in a else {k:a[k] for k in ('schema','method','profile','visits','NN_calls','best')})
assert a.get('schema')=='quartz_k25_c4_v1',a
assert a['profile']==15 and a['visits']==128 and sum(a['result']['n'])==128 and len(a['lanes'])==1
assert a['lanes'][0]['tail_hooks']['ledger']['repeated_delivery']==0
