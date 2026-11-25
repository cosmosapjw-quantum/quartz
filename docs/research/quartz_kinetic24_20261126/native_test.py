import pathlib,sys,json,subprocess as SP,os
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
sys.path.insert(0,str(ROOT.parent/'quartz_kinetic17_20261126/docs/research/quartz_kinetic17_20261126'));import pilot as p
r=json.loads((ROOT/'docs/research/quartz_kinetic23_20261126/COHORT.json').read_text())['roots'][0]
q=dict(cmd='research_connect4_evidence_cpu',history=r['history'],arm='K24D',cap=64,seed=17,model_path=str(p.MODEL),model_identity=p.MID)
s=SP.run([sys.argv[1],'--server'],input=json.dumps(q)+'\n',text=True,capture_output=True,timeout=30,env=dict(os.environ,RAYON_NUM_THREADS='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
assert s.returncode==0,s.stderr
a=json.loads(s.stdout.splitlines()[0]);print(a if 'error' in a else {k:a[k] for k in ('arm','visits','mode','incremental_nn','prefix_counts','counts')})
assert 'error' not in a,a
assert a['mode']==2 and a['prefix']['root_n']==32 and len(a['trace'])==a['visits'] and a['incremental_nn']<=64
