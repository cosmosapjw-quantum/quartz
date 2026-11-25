import pathlib,sys,json,subprocess as SP,os
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
P=ROOT.parent/'quartz_kinetic17_20261126/docs/research/quartz_kinetic17_20261126'
sys.path.insert(0,str(P));import pilot as p
co=json.loads((ROOT/'docs/research/quartz_kinetic18_20261126/COHORT.json').read_text());h=co['roots'][0]['history']
q=dict(cmd='research_connect4_evidence_cpu',history=h,arm='K21LIVE',cap=64,seed=17,model_path=str(p.MODEL),model_identity=p.MID,seq_table=[[0.]*8 for _ in range(7)])
r=SP.run([sys.argv[1],'--server'],input=json.dumps(q)+'\n',text=True,capture_output=True,timeout=30,env=dict(os.environ,RAYON_NUM_THREADS='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
assert r.returncode==0
a=json.loads(r.stdout.splitlines()[0]);print(a if 'error' in a else {k:a[k] for k in ('arm','visits','best','cause')})
assert 'error' not in a,a
assert a['visits']==8 and a['cause']=='STOP' and a['best']==a['states'][-1]['best']
