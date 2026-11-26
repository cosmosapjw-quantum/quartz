from core import D,ROOT,BIN,sha,save,SP
import os
target=BIN.parent/'cargo'
BIN.parent.mkdir(exist_ok=True)
assert not BIN.exists()
cmd=['nice','-n','5','cargo','build','--offline','--locked','--release','--bin','mcts_demo','--target-dir',str(target)]
env=dict(os.environ,CARGO_BUILD_JOBS='6',CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='4')
with (D/'RUST_BUILD.log').open('w') as f:
    p=SP.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=SP.STDOUT);f.write(f'\nexit={p.returncode}\n')
if p.returncode:raise SystemExit(p.returncode)
import shutil
shutil.copy2(target/'release/mcts_demo',BIN)
save('RUST_BUILD_IDENTITY.json',dict(command=cmd,env={k:env[k] for k in ('CARGO_BUILD_JOBS','CARGO_PROFILE_RELEASE_LTO','CARGO_PROFILE_RELEASE_CODEGEN_UNITS')},source_head=SP.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),rustc=SP.check_output(['rustc','-Vv'],text=True),cargo_lock=sha(ROOT/'Cargo.lock'),binary_sha256=sha(BIN),bytes=BIN.stat().st_size))
print('BUILD',sha(BIN),BIN.stat().st_size,flush=True)
