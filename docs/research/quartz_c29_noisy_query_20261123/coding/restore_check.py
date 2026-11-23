#!/usr/bin/env python3
"""Restore C29 C08 or final C10 from retained bundles and independently verify bytes.

No remote checkout or mutable working tree is used as the source of restoration.
Failure leaves a marked disposable directory for diagnosis; success removes it.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,stat,subprocess,tarfile,time

BACKUP=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_c29_noisy_query_backups')
ROOT=Path('/home/cosmosapjw/quantum/research/quartz_c29_noisy_query_20261123')
RESTORE_PARENT=Path('/home/cosmosapjw/quantum/research')
BRANCH='research/quartz-c29-noisy-query-20261123'
C26_REF='refs/heads/research/quartz-general-c26-game-metareasoning-20261123'
C28_REF='refs/heads/research/quartz-c28-action-geometry-20261123-486e'
C29_REF='refs/heads/'+BRANCH
DOC='docs/research/quartz_c29_noisy_query_20261123'
STAGES=[f'R{i:02}' for i in range(1,11)]+[f'C{i:02}' for i in range(1,11)]

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(args):
    p=subprocess.run(args,capture_output=True)
    if p.returncode:raise RuntimeError(f'COMMAND FAILED {args[:5]}: '+p.stderr.decode(errors='replace')[-1800:])
    return p.stdout

def git(repo,*args):return run(['git','-C',str(repo),*args])

def blob_hash(path,mode):
    if mode=='120000':
        data=os.fsencode(os.readlink(path))
        return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),len(data)
    length=path.stat().st_size
    h=hashlib.sha1(b'blob '+str(length).encode()+b'\0')
    with path.open('rb') as inp:
        for block in iter(lambda:inp.read(1024*1024),b''):h.update(block)
    return h.hexdigest(),length

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--expect-stage',choices=STAGES,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tag',required=True)
    args=ap.parse_args()
    assert args.tag in ('C08','FINAL')
    assert not args.output.exists()
    begin=time.perf_counter()
    base=json.loads((BACKUP/'BASE.json').read_text())
    last_bytes=(BACKUP/'LATEST.json').read_bytes()
    latest=json.loads(last_bytes)
    records=[json.loads(x) for x in (BACKUP/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    n=STAGES.index(args.expect_stage)+1
    assert latest['stage']==args.expect_stage
    assert [x['stage'] for x in records]==STAGES[:n]
    assert records[-1]==latest
    assert base['user_ssot']=='87c15dc828d6eea519c219fd8a412fa21dfc464d'
    assert base['science_parent']=='ebbf21a6a77c0133525c1a9f16f0ee36375a1440'
    files=[Path(base['c28_ancestor_bundle']),Path(base['c28_cumulative_bundle']),
           Path(latest['bundle']),Path(latest['archive'])]
    expect=[(base['c28_ancestor_sha256'],base['c28_ancestor_bytes']),
            (base['c28_cumulative_sha256'],base['c28_cumulative_bytes']),
            (latest['bundle_sha256'],latest['bundle_bytes']),
            (latest['archive_sha256'],latest['archive_bytes'])]
    for path,(want,size) in zip(files,expect):
        assert path.is_file() and path.stat().st_size==size
        assert digest(path)==want,path
    restored=RESTORE_PARENT/f'quartz_c29_restore_{args.tag}_{latest["commit"][:12]}'
    restored.mkdir(exist_ok=False)
    marker={'type':'DISPOSABLE_C29_RESTORE','stage':args.expect_stage,'head':latest['commit']}
    (restored/'RESTORE_OWNER.json').write_text(json.dumps(marker)+'\n')
    work=restored/'repo'
    run(['git','init','--quiet',str(work)])
    git(work,'fetch','--quiet',str(files[0]),C26_REF+':refs/heads/base_c26')
    assert git(work,'rev-parse','refs/heads/base_c26').decode().strip()==base['user_ssot']
    git(work,'fetch','--quiet',str(files[1]),C28_REF+':refs/heads/base_c28')
    assert git(work,'rev-parse','refs/heads/base_c28').decode().strip()==base['science_parent']
    git(work,'fetch','--quiet',str(files[2]),C29_REF+':refs/heads/c29')
    head=git(work,'rev-parse','refs/heads/c29').decode().strip()
    assert head==latest['commit']
    git(work,'checkout','--quiet','--detach',head)
    assert git(work,'rev-parse','HEAD^{tree}').decode().strip()==latest['tree']
    previous=base['science_parent']
    for receipt in records:
        assert receipt['parent']==previous and receipt['commit']==receipt['remote']
        assert git(work,'show','-s','--format=%P',receipt['commit']).decode().strip()==previous
        assert git(work,'rev-parse',receipt['commit']+'^{tree}').decode().strip()==receipt['tree']
        previous=receipt['commit']
    git(work,'fsck','--connectivity-only','--no-dangling')
    entries={}
    total_bytes=0
    for raw in git(work,'ls-tree','-rz',head).split(b'\0'):
        if not raw:continue
        metadata,rel=raw.split(b'\t',1)
        mode,kind,oid=metadata.decode().split()
        name=os.fsdecode(rel)
        path=work/name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        st=path.lstat()
        if mode=='120000':
            assert stat.S_ISLNK(st.st_mode)
        else:
            assert mode in ('100644','100755') and kind=='blob' and stat.S_ISREG(st.st_mode)
            assert bool(st.st_mode & 0o111)==(mode=='100755'),name
        observed,size=blob_hash(path,mode)
        assert observed==oid,name
        entries[name]=(mode,oid)
        total_bytes+=size
    assert git(work,'status','--porcelain','--untracked-files=all')==b''
    changed={os.fsdecode(name) for name in git(work,'diff','--name-only','-z','--diff-filter=ACMRT',
                                               base['science_parent'],head).split(b'\0') if name}
    with tarfile.open(files[3],'r:gz') as archive:
        members={m.name:m for m in archive.getmembers() if m.isfile() or m.issym()}
        assert set(members)==changed
        for name in changed:
            member=members[name]
            mode,oid=entries[name]
            if member.issym():
                assert mode=='120000'
                data=os.fsencode(member.linkname)
                got=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            else:
                assert member.isfile() and mode in ('100644','100755')
                assert bool(member.mode & 0o111)==(mode=='100755')
                h=hashlib.sha1(b'blob '+str(member.size).encode()+b'\0')
                with archive.extractfile(member) as reader:
                    for block in iter(lambda:reader.read(1024*1024),b''):h.update(block)
                got=h.hexdigest()
            assert got==oid,name
    locked=json.loads((work/DOC/'evidence/C01_BASELINE.json').read_text())['source_hashes']
    protected={'C28':'src/bin/quartz_c28_action_geometry.rs',
               'C26':'src/bin/quartz_c26_game_metareasoning.rs',
               'source':'docs/research/quartz_general_game_metareasoning_20261123/coding/data/public_sources.jsonl',
               'model':'docs/research/quartz_general_game_metareasoning_20261123/coding/data/model.json',
               'cargo_lock':'Cargo.lock'}
    for key,rel in protected.items():
        assert digest(work/rel)==locked[key],rel
    binary=json.loads((BACKUP/'MEASURED_BINARY.json').read_text())
    assert digest(binary['backup'])==binary['sha256']
    assert Path(binary['backup']).stat().st_size==binary['bytes']
    assert oct(Path(binary['backup']).stat().st_mode&0o777)==binary['mode']
    if n>=19:
        assert digest(work/'src/bin/quartz_c29_noisy_query.rs')==binary['science_source_sha256']
    origin=git(ROOT,'ls-remote','--heads','origin',C29_REF).decode().split()[0]
    assert origin==head
    assert (BACKUP/'LATEST.json').read_bytes()==last_bytes
    assert sorted(x.name for x in BACKUP.glob('C29_*.bundle'))==[files[2].name]
    assert sorted(x.name for x in BACKUP.glob('C29_*.tar.gz'))==[files[3].name]
    cleanup=[json.loads(x) for x in (BACKUP/'CLEANUP_RECEIPTS.jsonl').read_text().splitlines()]
    assert [x['after_stage'] for x in cleanup]==STAGES[:n]
    removed=[entry for c in cleanup for entry in c['removed']]
    assert len(removed)==2*(n-1)
    assert all(not Path(x['path']).exists() for x in removed)
    assert json.loads((restored/'RESTORE_OWNER.json').read_text())==marker
    shutil.rmtree(restored)
    result={'status':'VERIFIED','audit_stage':args.expect_stage,'SSOT':base['user_ssot'],
       'C28_parent':base['science_parent'],'restored_commit':head,'remote_head':origin,
       'ordered_stage_count':n,'ordered_stages':STAGES[:n],'tracked_blobs_and_modes_verified':len(entries),
       'tracked_payload_bytes':total_bytes,'C29_changed_archive_files_verified':len(changed),
       'protected_input_hashes_verified':locked,'measured_binary_sha256':binary['sha256'],
       'measured_binary_bytes':binary['bytes'],'upstream_base_and_two_incremental_bundles_restored':True,
       'retained_backups':[{'path':str(p),'sha256':h,'bytes':b} for p,(h,b) in zip(files,expect)],
       'superseded_C29_archives_removed':len(removed),
       'superseded_C29_backup_bytes_removed':sum(x['bytes'] for x in removed),
       'disposable_restored_directory_removed':str(restored),
       'cloud_sync':'NOT_INDEPENDENTLY_VERIFIED',
       'elapsed_seconds':time.perf_counter()-begin,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('protected_input_hashes_verified','retained_backups','ordered_stages')},indent=2))

if __name__=='__main__':main()
