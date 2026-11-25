import pathlib,sys,json,subprocess as SP,os
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
sys.path.insert(0,str(ROOT.parent/'quartz_kinetic17_20261126/docs/research/quartz_kinetic17_20261126'));import pilot as p
r=json.loads((ROOT/'docs/research/quartz_kinetic25_20261126/COHORT.json').read_text())['roots'][0]
q=dict(cmd='research_connect4_k26_cpu',history=r['history'],method='TREE8',path=[15,8],seed=17,model_path=str(p.MODEL),model_identity=p.MID,trace=True)
s=SP.run([sys.argv[1],'--server'],input=json.dumps(q)+'\n',text=True,capture_output=True,timeout=30,env=dict(os.environ,RAYON_NUM_THREADS='8',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
assert s.returncode==0,s.stderr
a=json.loads(s.stdout.splitlines()[0]);print(a if 'schema' not in a else {k:a[k] for k in ('schema','method','chosen','visits','NN_calls','best')})
assert a.get('schema')=='quartz_k26_c4_v1',a
assert a['chosen']==[15,8] and a['prefix']['visits']==8 and a['visits']<=72
prev=a['prefix']['nn']
for z in a['stages']:
    b=z['budgets'][0][0];assert 0<=b['used_nn']<=8 and b['used_nn']+b['unused']==8 and z['state']['nn']-prev==b['used_nn']
    prev=z['state']['nn']
assert a['stages'][1]['hooks'][0]['ledger']['repeated_delivery']==0
