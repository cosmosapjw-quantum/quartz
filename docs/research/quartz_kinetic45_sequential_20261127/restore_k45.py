"""Independent incremental restore with K44 shallow base; preserve verified dependencies."""
import json,pathlib,subprocess as SP,hashlib,shutil
from publish import ROOT,B,BASE,BR,git,sha
D=pathlib.Path(__file__).resolve().parent
T=ROOT.parent/'quartz_kinetic45_sequential_target_20261127'
R=T/'restore_base.git'
def gg(*a):return SP.check_output(['git','-C',str(R),*a])
def main():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'] and not git('status','--porcelain')
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    exp=[f'R{i:02d}' for i in range(1,11)]+[f'C{i:02d}' for i in range(1,7)]+['C06A','C07','C07A','C07B','C07C','C08','C08A','C08B','C09','C09A','C10']
    assert [x['stage'] for x in receipts]==exp
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts] and all(x['remote']==x['head'] and x['parent']==p for x,p in zip(receipts,[BASE]+chain[:-1]))
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and sorted(p.name for p in B.glob('*.bundle'))==[bundle.name]
    assert all(c['old_head']==x['head'] and c['new_head']==y['head'] and c['old_sha256']==x['sha256'] and c['superseding']==y['bundle'] for c,x,y in zip(clean,receipts,receipts[1:]))
    models=json.loads((D/'PLAN.json').read_text())['models'];old=B.parent/'quartz_kinetic33_backups'/'models'
    for s,m in models.items():
        path=B.parent/'quartz_kinetic39_voc_backups'/'model_105_seed103_gen5.pt' if s=='105' else old/f'seed_{s}.pt'
        assert sha(pathlib.Path(m['path']))==m['sha256']==sha(path)
    native=json.loads((D.parent/'quartz_kinetic40_budget_20261126/NATIVE_IDENTITY.json').read_text())
    assert sha(B.parent/'quartz_kinetic40_budget_backups'/'mcts_demo')==native['binary_sha256']
    independent=json.loads((D.parent/'quartz_kinetic42_threshold_20261127/AUDIT.json').read_text())['independent_binary_sha256']
    assert sha(B.parent/'quartz_kinetic41_exact_backups'/'independent')==independent
    library=json.loads((D/'NATIVE_IDENTITY.json').read_text())['binary_sha256']
    assert sha(B/'seq45.so')==sha(T/'seq45.so')==library
    assert sha(B.parent/'quartz_kinetic44_incremental_backups'/'inc44.so')==json.loads((D/'NATIVE_IDENTITY.json').read_text())['old_incremental_library_sha256']
    assert sha(B.parent/'quartz_kinetic42_threshold_backups'/'threshold42.so')==json.loads((D/'NATIVE_IDENTITY.json').read_text())['threshold_library_sha256']
    assert sha(T/'mcts_demo')==native['binary_sha256']
    assets=json.loads((D.parent/'quartz_kinetic37_20261126/RAPFI_LOCK.json').read_text())['files']
    asset_backup=B.parent/'quartz_kinetic37_backups'/'rapfi_assets'
    for a in assets:assert sha(pathlib.Path(a['path']))==a['sha256']==sha(asset_backup/pathlib.Path(a['path']).name)
    bridge=ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125'/'pbrain-rapfi'
    assert sha(bridge)==sha(asset_backup/'pbrain-rapfi-bridge')=='bacbcfc90f6e4bfd0d9d035455afa3d6667ec32e7fd80cbc1d65a4001b30f64e'
    if not R.exists():
        SP.run(['git','init','--bare','-q',str(R)],check=True)
        gg('fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
    assert not (R/'objects/info/alternates').exists()
    assert gg('rev-parse','FETCH_HEAD').decode().strip()==BASE
    assert SP.run(['git','-C',str(R),'cat-file','-e',head],capture_output=True).returncode!=0
    gg('bundle','verify',str(bundle));gg('fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
    assert gg('rev-parse','refs/heads/restored').decode().strip()==head
    assert gg('rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
    assert gg('rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
    files=git('diff','--name-only',BASE,head).splitlines()
    for f in files:assert hashlib.sha256(gg('show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
    gg('fsck','--connectivity-only','--no-dangling')
    assert R.parent==T and R.name=='restore_base.git' and R.is_dir() and not R.is_symlink()
    restore_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    out=dict(head=head,tree=rec['tree'],remote=rec['remote'],bundle=bundle.name,bundle_sha256=sha(bundle),bytes=rec['bytes'],restored_files=len(files),stage_count=len(receipts),cleanup_count=len(clean),removed_stage_bundle_bytes=sum(x['bytes'] for x in receipts[:-1]),external_base=BASE,actual_restore=True,connectivity_fsck=True,checkpoint_backups_verified=len(models),rapfi_assets_verified=len(assets)+1,native_binary_sha256=native['binary_sha256'],independent_binary_sha256=independent,sequential_library_sha256=library,temporary_restore_bytes=restore_bytes,cloud_sync='NOT_VERIFIED')
    p=B/'FINAL_VERIFY.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    shutil.rmtree(R)
    out['temporary_restore_removed']=not R.exists()
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
