from pathlib import Path
import os,subprocess as sp,hashlib,json,shutil
D=Path(__file__).resolve().parent;ROOT=D.parents[2];BIN=ROOT.parent/'quartz_kinetic34_target_20261126/mcts_demo'
target=BIN.parent/'cargo';BIN.parent.mkdir(exist_ok=True);assert not BIN.exists()
cmd=['nice','-n','5','cargo','build','--offline','--locked','--release','--bin','mcts_demo','--target-dir',str(target)]
env=dict(os.environ,CARGO_BUILD_JOBS='4',CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='4')
with (D/'RUST_BUILD.log').open('w') as f:
    p=sp.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=sp.STDOUT);f.write(f'\nexit={p.returncode}\n')
if p.returncode:raise SystemExit(p.returncode)
shutil.copy2(target/'release/mcts_demo',BIN)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(D/'RUST_BUILD_IDENTITY.json').write_text(json.dumps(dict(command=cmd,source_head=sp.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),rustc=sp.check_output(['rustc','-Vv'],text=True),cargo_lock=sha(ROOT/'Cargo.lock'),binary_sha256=sha(BIN),bytes=BIN.stat().st_size),indent=2)+'\n')
print('BUILD',sha(BIN),BIN.stat().st_size,flush=True)
