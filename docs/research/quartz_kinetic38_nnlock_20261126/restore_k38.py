"""Restore K38 from its verified incremental bundle and a shallow K37 base."""
import json,pathlib,subprocess as SP,tempfile,hashlib
from publish import ROOT,B,BASE,BR,git,sha
D=pathlib.Path(__file__).resolve().parent
def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
def main():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree']
    assert not git('status','--porcelain')
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    expected=[f'R{i:02d}' for i in range(1,11)]+[f'C{i:02d}' for i in range(1,7)]+['C06A']+[f'C{i:02d}' for i in range(7,11)]
    assert [x['stage'] for x in receipts]==expected
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts]
    assert all(x['remote']==x['head'] and x['parent']==p for x,p in zip(receipts,[BASE]+chain[:-1]))
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and sorted(p.name for p in B.glob('*.bundle'))==[bundle.name]
    assert all(c['old_head']==x['head'] and c['new_head']==y['head'] and c['old_sha256']==x['sha256'] and c['superseding']==y['bundle'] for c,x,y in zip(clean,receipts,receipts[1:]))
    models=json.loads((D/'PLAN.json').read_text())['models']
    model_backup=B.parent/'quartz_kinetic33_backups'/'models'
    for s,m in models.items():
        assert sha(pathlib.Path(m['path']))==m['sha256']==sha(model_backup/f'seed_{s}.pt')
    files=git('diff','--name-only',BASE,head).splitlines()
    with tempfile.TemporaryDirectory(prefix='quartz-kinetic38-restore-') as tmp:
        r=pathlib.Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists()
        assert SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head
        assert gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for f in files:assert hashlib.sha256(gg(r,'show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
        gg(r,'fsck','--connectivity-only','--no-dangling')
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],bundle=bundle.name,bundle_sha256=sha(bundle),bytes=rec['bytes'],restored_files=len(files),stage_count=len(receipts),cleanup_count=len(clean),external_base=BASE,actual_restore=True,connectivity_fsck=True,checkpoint_backups_verified=len(models),checkpoint_backup_location=str(model_backup),cloud_sync='NOT_VERIFIED')
if __name__=='__main__':
    x=main();p=B/'FINAL_VERIFY.json';assert not p.exists();p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x,indent=2,sort_keys=True),flush=True)
