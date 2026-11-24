"""Read-only audit of exact base, incremental bundle, reconstructed tree and frozen results."""
import hashlib,json,os,subprocess,tempfile,shutil,sys
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[3]
D=Path(__file__).resolve().parent
B=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_channel_heterogeneity_backups')
BASE='87c15dc828d6eea519c219fd8a412fa21dfc464d'
BR='research/quartz-channel-heterogeneity-20261124'
def git(*args,repo=ROOT,raw=False):
    x=subprocess.check_output(['git','-C',str(repo),*args])
    return x if raw else x.decode().strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def restore():
    latest=json.loads((B/'LATEST.json').read_text());head=latest['head']
    assert head==git('rev-parse','HEAD')
    assert latest['tree']==git('rev-parse','HEAD^{tree}')
    bundle=B/latest['bundle'];assert sha(bundle)==latest['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    assert [r['head'] for r in receipts]==git('rev-list','--reverse',BASE+'..'+head).splitlines()
    assert all(r['parent']==(BASE if i==0 else receipts[i-1]['head']) and r['remote']==r['head'] for i,r in enumerate(receipts))
    cleanup=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(cleanup)==len(receipts)-1
    assert len(list(B.glob('*.bundle')))==1
    changed=git('diff','--name-only',BASE,head).splitlines()
    assert all(p.startswith('docs/research/quartz_channel_heterogeneity_20261124/') or p=='src/bin/quartz_channel_heterogeneity.rs' for p in changed)
    # Actual independent restore from a shallow exact base snapshot, no alternates.
    with tempfile.TemporaryDirectory(prefix='quartz-channel-restore-') as td:
        repo=Path(td)/'restore.git'
        subprocess.run(['git','init','--bare','-q',str(repo)],check=True)
        print('restore: fetching exact base snapshot',flush=True)
        git('fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE,repo=repo)
        git('update-ref','refs/heads/base',BASE,repo=repo)
        assert not (repo/'objects/info/alternates').exists()
        assert subprocess.run(['git','-C',str(repo),'cat-file','-e',head],capture_output=True).returncode!=0
        print('restore: unbundling incremental research history',flush=True)
        git('bundle','verify',str(bundle),repo=repo)
        git('fetch','--no-tags',str(bundle),'refs/heads/'+BR+':refs/heads/restored',repo=repo)
        assert git('rev-parse','refs/heads/restored',repo=repo)==head
        assert git('rev-parse','refs/heads/restored^{tree}',repo=repo)==latest['tree']
        assert git('rev-list','--reverse',BASE+'..refs/heads/restored',repo=repo).splitlines()==[r['head'] for r in receipts]
        assert not (repo/'objects/info/alternates').exists()
        for p in changed:
            actual=git('show',head+':'+p,repo=repo,raw=True)
            assert hashlib.sha256(actual).hexdigest()==sha(ROOT/p),p
        git('fsck','--connectivity-only','--no-dangling',repo=repo)
    frozen=json.loads((D/'C07_TRAIN_CHECK.json').read_text())
    for k,p in [('model_sha256',D/'model.json'),('protocol_sha256',D/'campaign.py'),('source_sha256',D/'sources.jsonl'),('native_sha256',B/'quartz_channel_heterogeneity_native')]:
        assert frozen[k]==sha(p)
    # Independent integer/rational aggregation of primary policy statistic from per-root evidence.
    rr=json.loads((D/'C08_ROOTS.json').read_text())
    total=Fraction(0)
    for r in rr:
        a,f=(r['policies'][p] for p in ('A','F'))
        total+=Fraction(a['utility'])-Fraction(f['utility'])-(Fraction(a['total_nodes'])-Fraction(f['total_nodes']))/1024
    exact=total/len(rr)
    summary=json.loads((D/'C08_SUMMARY.json').read_text())
    assert float(exact)==summary['primary_net_A_minus_F']['mean']
    assert git('ls-remote','origin','refs/heads/'+BR).split()[0]==head
    assert all(line.startswith('?? ') and line[3:]==str(Path(__file__).relative_to(ROOT)) for line in git('status','--porcelain').splitlines())
    result=dict(restored_head=head,tree=latest['tree'],bundle_sha256=latest['sha256'],stage_count=len(receipts),cleanup_count=len(cleanup),changed_files_byte_verified=len(changed),base=BASE,base_mode='independent shallow snapshot fetched locally; no alternates; target absent before import',connectivity='exit0',exact_primary_net=str(exact),heldout_count=len(rr),production_changes=0,cloud_sync='not verified; local Dropbox filesystem only')
    print(json.dumps(result,indent=2),flush=True)
    return result
if __name__=='__main__':
    result=restore()
    (D/'C09_RESTORE.json').write_text(json.dumps(result,indent=2)+'\n')
