"""Restore incremental K17 bundle into an isolated shallow base object store."""
import json, pathlib, subprocess as SP, tempfile, hashlib
from publish import ROOT,B,BASE,BR,git,sha
D=pathlib.Path(__file__).resolve().parent
def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
def main():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts]
    assert all(x['remote']==x['head'] and x['parent']==p for x,p in zip(receipts,[BASE]+chain[:-1]))
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and sorted(p.name for p in B.glob('*.bundle'))==[bundle.name]
    assert all(c['old_head']==x['head'] and c['new_head']==y['head'] and c['old_sha256']==x['sha256'] and c['superseding']==y['bundle'] for c,x,y in zip(clean,receipts,receipts[1:]))
    files=git('diff','--name-only',BASE,head).splitlines()
    with tempfile.TemporaryDirectory(prefix='quartz-kinetic20-restore-') as tmp:
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
    bi=json.loads((D/'RUST_BUILD_IDENTITY.json').read_text());ei=json.loads((D/'RUST_EXT_BUILD_IDENTITY.json').read_text())
    binary=ROOT.parent/'quartz_kinetic20_target_20261126/mcts_demo'
    extended=ROOT.parent/'quartz_kinetic20_ext_target_20261126/mcts_demo'
    assert sha(B/'mcts_demo')==sha(binary)==bi['binary_sha256']
    assert sha(B/'mcts_demo_matchx')==sha(extended)==ei['binary_sha256']
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],bundle=bundle.name,bundle_sha256=sha(bundle),bytes=rec['bytes'],binary_sha256=sha(binary),extended_binary_sha256=sha(extended),restored_files=len(files),stage_count=len(receipts),cleanup_count=len(clean),external_base=BASE,actual_restore=True,connectivity_fsck=True,cloud_sync='NOT_VERIFIED')
if __name__=='__main__':print(json.dumps(main(),indent=2,sort_keys=True),flush=True)
