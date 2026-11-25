"""Direct release rustc with read-only frozen Cargo release dependencies from NN-DECISION-02."""
from pathlib import Path
import subprocess,hashlib,json,os
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
DEP=ROOT.parent/'quartz_nn_intervention_target_20261124/release/deps'
OUT=ROOT.parent/'quartz_kinetic24_target_20261126/mcts_demo'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
args=['nice','-n','10','rustc','--edition=2021','--crate-name','mcts_demo','--crate-type','bin','-C','opt-level=3','-C','codegen-units=4','-C','debuginfo=0','-L',f'dependency={DEP}','src/main.rs','-o',str(OUT)]
inputs={}
for name in ('rand','rand_distr','rayon','serde','serde_json','crossbeam_channel','parking_lot','bumpalo','smallvec','arc_swap'):
    paths=list(DEP.glob(f'lib{name}-*.rlib'));assert len(paths)==1,(name,paths)
    p=paths[0];args+=['--extern',f'{name}={p}'];inputs[str(p)]=sha(p)
assert not OUT.exists()
p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(D/'RUST_BUILD.log').write_text(p.stdout.rstrip()+'\nexit='+str(p.returncode)+'\n')
if p.returncode:print(p.stdout[-9000:]);raise SystemExit(p.returncode)
info=dict(command=args,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),rustc=subprocess.check_output(['rustc','-Vv'],text=True),dependencies=inputs,binary=str(OUT),binary_sha256=sha(OUT),bytes=OUT.stat().st_size)
(D/'RUST_BUILD_IDENTITY.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
print(info['binary_sha256'],info['bytes'],flush=True)
