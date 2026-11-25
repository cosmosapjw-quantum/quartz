"""Remove only verified published inactive clean worktree copies, never branches or targets."""
import subprocess as s,os,json,hashlib
from pathlib import Path
from publish import ROOT,D,B,git
P=ROOT.parent;BACK=B.parent
def sh(*a):return s.check_output(a,text=True).strip()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def paths_in_use(p):
    out=[]
    for x in Path('/proc').iterdir():
        if not x.name.isdigit():continue
        try:
            vals=[os.readlink(x/'cwd'),os.readlink(x/'exe')]
            vals+=((x/'cmdline').read_bytes().decode(errors='replace').split('\0'))
            try:
                for fd in (x/'fd').iterdir():
                    try:vals.append(os.readlink(fd))
                    except OSError:pass
            except OSError:pass
            if any(v==str(p) or str(p)+'/' in v or v.startswith(str(p)+' ') for v in vals):out.append(x.name)
        except OSError:pass
    return out
candidates={'quartz_c28_action_geometry_20261123_486e':'quartz_c28_action_geometry','quartz_c29_noisy_query_20261123':'quartz_c29_noisy_query','quartz_channel_heterogeneity_20261124':'quartz_channel_heterogeneity','quartz_fouraxis_20261125':'quartz_fouraxis','quartz_fouraxis02_20261125':'quartz_fouraxis02','quartz_fouraxis03_20261125':'quartz_fouraxis03','quartz_fouraxis04_20261125':'quartz_fouraxis04','quartz_nn_costcal_20261125':'quartz_nn_costcal','quartz_nn_decision_20261124':'quartz_nn_decision','quartz_nn_intervention_20261124':'quartz_nn_intervention','quartz_nn_multiquanta_20261124':'quartz_nn_multiquanta'}
raw=git('worktree','list','--porcelain')
remote={r:h for h,r in (x.split('\t') for x in git('ls-remote','--heads','origin').splitlines())}
rows=[];free0=os.statvfs(P).f_bavail*os.statvfs(P).f_frsize
for block in raw.split('\n\n'):
    d=dict((line.split(' ',1)+[''])[:2] for line in block.splitlines())
    p=Path(d['worktree']);name=p.name
    if name not in candidates:continue
    row=dict(path=str(p),head=d['HEAD'],branch=d.get('branch'),status='RETAINED')
    rows.append(row)
    try:
        assert p.parent==P and p.is_dir() and p.resolve()==p and 'locked' not in d and 'prunable' not in d
        assert remote.get(d.get('branch'))==d['HEAD']
        assert sh('git','-C',str(p),'status','--porcelain','--untracked-files=all')==''
        ignored=sh('git','-C',str(p),'ls-files','--others','--ignored','--exclude-standard').splitlines()
        assert all('/__pycache__/' in x and x.endswith('.pyc') for x in ignored)
        assert not paths_in_use(p)
        b=BACK/(candidates[name]+'_backups')
        if (b/'FINAL_VERIFY.json').exists():
            r=json.loads((b/'FINAL_VERIFY.json').read_text())
            assert r.get('head',r.get('restored_head'))==d['HEAD'] and r.get('remote',r.get('remote_head'))==d['HEAD']
            f=b/r.get('bundle',r.get('bundle_file'));assert sha(f)==r.get('sha256',r.get('bundle_sha256'))
            assert r.get('actual_restore') or r.get('connectivity')=='exit0'
            bundles=[str(f)]
        else:
            r=json.loads((b/'FINAL_RESTORE_AUDIT.json').read_text())
            assert r.get('commit',r.get('restored_commit'))==d['HEAD'] and r['remote_head']==d['HEAD'] and r['status']=='VERIFIED'
            bundles=[]
            for v in r['retained_backups']:
                f=Path(v['path']);assert f.is_file() and sha(f)==v['sha256']
                if f.suffix=='.bundle':bundles.append(str(f))
            assert bundles
        assert any(d['HEAD'] in git('bundle','list-heads',f) for f in bundles)
        for f in bundles:git('bundle','verify',f)
        # Immediately recheck branch/worktree/active state, no force.
        assert sh('git','-C',str(p),'rev-parse','HEAD')==d['HEAD'] and sh('git','-C',str(p),'status','--porcelain','--untracked-files=all')=='' and not paths_in_use(p)
        row['bytes_du']=int(sh('du','-sB1',str(p)).split()[0]);row['ignored_cache_files']=len(ignored);row['verified_bundles']=bundles
        row['restore_command']=['git','worktree','add',str(p),d['branch'].removeprefix('refs/heads/')]
        s.run(['git','-C',str(ROOT),'worktree','remove',str(p)],check=True)
        assert not p.exists() and git('rev-parse',d['branch'])==d['HEAD']
        row['status']='REMOVED_RESTORABLE'
    except (AssertionError,KeyError,OSError,s.CalledProcessError,TypeError) as exc:row['reason']=str(exc) or 'precondition not satisfied'
    print(row['status'],name,row.get('bytes_du'),row.get('reason',''),flush=True)
free1=os.statvfs(P).f_bavail*os.statvfs(P).f_frsize
(D/'R01_WORKTREE_CLEANUP.json').write_text(json.dumps(dict(rows=rows,free_before=free0,free_after=free1,free_delta=free1-free0,other_worktrees='retained; includes base, FA05/06, main and detached branches, separate targets untouched'),indent=2,sort_keys=True)+'\n')
