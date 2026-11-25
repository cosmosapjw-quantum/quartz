"""Actual endpoint red/green probe on a neutral all-zero acquisition table."""
import pathlib,sys,json,subprocess as SP
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
BIN=pathlib.Path(sys.argv[1]);P=ROOT.parent/'quartz_kinetic17_20261126/docs/research/quartz_kinetic17_20261126'
sys.path.insert(0,str(P));import pilot as p
q=dict(cmd='research_connect4_evidence_cpu',history='4433226223',arm='K20LIVE',cap=64,seed=17,model_path=str(p.MODEL),model_identity=p.MID,controller=dict(gains=[[0.]*6 for _ in range(4)],costs=[[1.]*6 for _ in range(4)],switch=[[[False]*2 for _ in range(3)] for _ in range(6)]))
# quiet known K18 root instead of a tactical fixture
co=json.loads((ROOT/'docs/research/quartz_kinetic18_20261126/COHORT.json').read_text())
q['history']=co['roots'][0]['history']
r=SP.run([str(BIN),'--server'],input=json.dumps(q)+'\n',text=True,capture_output=True,timeout=30)
assert r.returncode==0,r.stderr
a=json.loads(r.stdout.splitlines()[0]);print(a if 'error' in a else {k:a[k] for k in ('arm','visits','best','decision')})
assert 'error' not in a,a
assert a['visits']==8 and a['decision']['channel']=='STOP' and a['best']==a['prefix']['best']
