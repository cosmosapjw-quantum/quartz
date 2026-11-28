"""Restore K64 from its incremental bundle in an independent shallow repository."""
import json,pathlib,subprocess as SP,hashlib,shutil
from publish import ROOT,B,BASE,BR,git,sha
D=pathlib.Path(__file__).resolve().parent
T=ROOT.parent/'quartz_kinetic64_marginal_run_20261128'
R=T/'restore_base.git'
def gg(*a):return SP.check_output(['git','-C',str(R),*a])
def main():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'] and not git('status','--porcelain')
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    exp=[f'R{i:02d}' for i in range(1,11)]+['C01','C02','C03','C04','C05','C06','C06A','C07','C08','C09','C10']
    assert [x['stage'] for x in receipts]==exp
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts] and all(x['remote']==x['head'] and x['parent']==p for x,p in zip(receipts,[BASE]+chain[:-1]))
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and sorted(p.name for p in B.glob('*.bundle'))==[bundle.name]
    assert all(c['old_head']==x['head'] and c['new_head']==y['head'] and c['old_sha256']==x['sha256'] and c['superseding']==y['bundle'] for c,x,y in zip(clean,receipts,receipts[1:]))
    parent=B.parent/'quartz_kinetic63_select_backups';pr=json.loads((parent/'LATEST.json').read_text())
    assert pr['head']==BASE and sha(parent/pr['bundle'])==pr['sha256']
    assert json.loads((parent/'FINAL_VERIFY.json').read_text())['actual_restore']
    plan=json.loads((D/'PLAN.json').read_text());models=plan['models']
    for s,m in models.items():
        path=pathlib.Path(m['backup']) if s in ('211','222','233','311','322','333') else B.parent/'quartz_kinetic46_confirm_backups'/'models'/f'seed_{s}_gen2.pt' if s in ('41','42','43') else B.parent/'quartz_kinetic39_voc_backups'/'model_105_seed103_gen5.pt' if s=='105' else B.parent/'quartz_kinetic33_backups'/'models'/f'seed_{s}.pt'
        assert sha(pathlib.Path(m['path']))==m['sha256']==sha(path)
    lock=json.loads((D/'BINARY_LOCK.json').read_text())
    assert sha(T/'mcts_demo')==sha(B/'mcts_demo')==lock['binary_sha256']
    assert sha(pathlib.Path(plan['old_binary']))==sha(parent/'mcts_demo')==plan['old_binary_sha256']
    assert sha(D/'PLAN.json')==lock['plan_sha256']
    for mode in ('DEV','MAIN','ARCHIVE','NOVEL','RAPFI'):
        m=json.loads((D/(mode+'_LOCK.json')).read_text())
        for x in m['files']:assert sha(D/x['file'])==x['sha256']
    for p,digest in lock['source_sha256'].items():assert sha(ROOT/p)==digest
    rp=json.loads((D/'RAPFI_PLAN.json').read_text());asset_backup=B.parent/'quartz_kinetic37_backups'/'rapfi_assets'
    for a in rp['assets']:assert sha(pathlib.Path(a['path']))==a['sha256']==sha(asset_backup/pathlib.Path(a['path']).name)
    assert sha(pathlib.Path(rp['bridge']))==sha(asset_backup/'pbrain-rapfi-bridge')==rp['bridge_sha256']
    exact=json.loads((D/'EXACT_ALL.json').read_text())
    assert sha(B.parent/'quartz_kinetic51_targeted_backups'/'independent51')==exact['binary_sha256']
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
    out=dict(head=head,tree=rec['tree'],remote=rec['remote'],bundle=bundle.name,bundle_sha256=sha(bundle),bytes=rec['bytes'],restored_files=len(files),stage_count=len(receipts),cleanup_count=len(clean),removed_stage_bundle_bytes=sum(x['bytes'] for x in receipts[:-1]),external_base=BASE,parent_bundle_verified=True,actual_restore=True,connectivity_fsck=True,checkpoint_backups_verified=len(models),rapfi_assets_verified=len(rp['assets'])+1,native_binary_sha256=lock['binary_sha256'],old_native_binary_sha256=plan['old_binary_sha256'],independent_binary_sha256=exact['binary_sha256'],temporary_restore_bytes=restore_bytes,cloud_sync='NOT_VERIFIED')
    p=B/'FINAL_VERIFY.json';assert not p.exists()
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    shutil.rmtree(R)
    out['temporary_restore_removed']=not R.exists()
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
