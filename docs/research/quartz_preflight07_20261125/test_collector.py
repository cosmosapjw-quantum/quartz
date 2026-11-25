"""No production game generation: manifest/split/replay/fault tests on 2 stored fixtures."""
import copy,fcntl,gzip,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from collector import *
from rules import fixtures,orbit_board,transform
def rejects(fn,typ=(ValueError,AssertionError)):
    try:fn()
    except typ:return
    raise AssertionError('expected rejection')
def main():
    p=validate_plan(json.loads((D/'PLAN_DISABLED.json').read_text()));assert p['generation_enabled'] is False
    # Tiny metadata-only plan, never enabled or executed.
    t=dict(p,games=20,shards=3);jobs=list(job_stream(t));assert len(jobs)==20
    groups=collections.defaultdict(list)
    for j in jobs:groups[j['family']].append(j)
    assert len(groups)==10 and all(len(v)==2 and len({j['split'] for j in v})==1 and len({j['shard'] for j in v})==1 and v[0]['seed']!=v[1]['seed'] for v in groups.values())
    assert collections.Counter(j['split'] for j in jobs)=={'TRAIN':12,'CAL':4,'TEST':4}
    assert jobs==list(job_stream(t))
    f=fixtures();assert family_id('freestyle',f['trans_a'])==family_id('freestyle',f['trans_b'])
    assert family_id('freestyle',f['trans_a'])==family_id('freestyle',[transform(a,3) for a in f['trans_a']])
    assert family_id('freestyle',f['trans_a'])!=family_id('standard',f['trans_a'])
    arena=list(job_stream(dict(t,mode='PAIRED_ARENA')))
    for i in range(0,len(arena),2):
        a,b=arena[i:i+2];assert a['family']==b['family'] and a['opening']==b['opening'] and a['seed']==b['seed']
        assert (a['black_arm'],a['white_arm'],b['black_arm'],b['white_arm'])==('CERT','BASE','BASE','CERT')
    assert search_visits(dict(t,mode='PAIRED_ARENA',candidate_visits=256),'CERT')==256
    assert search_visits(dict(t,mode='PAIRED_ARENA',candidate_visits=256),'BASE')==128
    src=D/'C07_SMOKE';sp=json.loads((src/'manifest.json').read_text())['plan']
    j=sp['jobs'][0];orig=existing(src,sp,j);r=load_record(orig)
    assert r['winner_black']==0 and all(x['z']==0 for x in r['moves'])
    j2=sp['jobs'][1];r2=load_record(existing(src,sp,j2));assert r2['winner_black'] is None and all(x['z'] is None for x in r2['moves'])
    assert all(z_label(w,p0)==(None if w is None else w*p0) for w in (None,-1,0,1) for p0 in (-1,1))
    a=copy.deepcopy(r2);a['winner_black']=0;rejects(lambda:validate_record(a,sp,j2))
    a=copy.deepcopy(r);a['moves'][0]['z']=1;rejects(lambda:validate_record(a,sp,j))
    a=copy.deepcopy(r);a['moves'][0]['native']['rule_key']='0'*128;rejects(lambda:validate_record(a,sp,j))
    a=copy.deepcopy(r);a['moves'][0]['behavior_probability']=[0,1];rejects(lambda:validate_record(a,sp,j))
    bad=dict(p,collection_lock={});rejects(lambda:validate_plan(bad))
    with tempfile.TemporaryDirectory(prefix='quartz_long_fault_') as tmp:
        tmp=Path(tmp);dest=tmp/'never';rejects(lambda:authorize(p,digest(p),1,0,dest));assert not dest.exists()
        # Authorization hash and disk preflight independently blocked, no engine.
        active=dict(p,generation_enabled=True,min_free_gib=10000)
        rejects(lambda:authorize(active,'0'*64,1,0,dest));assert not dest.exists()
        rejects(lambda:authorize(active,digest(active),1,0,dest));assert not dest.exists()
        root=tmp/'store';init_store(root,sp)
        fp=save_record(root,sp,j,r);assert existing(root,sp,j)==fp
        rejects(lambda:save_record(root,sp,j,r))
        assert load_record(fp)==r
        # Filename remains valid format but bytes changed, explicit checksum error.
        badfile=fp.parent/(j['game_id']+'.'+'0'*64+'.json.gz')
        badfile.write_bytes(fp.read_bytes());rejects(lambda:load_record(badfile));rejects(lambda:existing(root,sp,j))
        badfile.unlink()
        q=tmp/'immutable';atomic_create(q,b'a');atomic_create(q,b'a');rejects(lambda:atomic_create(q,b'b'));assert q.read_bytes()==b'a'
        lock=tmp/'lock'
        with lock.open('a') as f1,lock.open('a') as f2:
            fcntl.flock(f1,fcntl.LOCK_EX|fcntl.LOCK_NB)
            rejects(lambda:fcntl.flock(f2,fcntl.LOCK_EX|fcntl.LOCK_NB),(BlockingIOError,))
        # Resume verifier validates only committed records; missing not draw.
        audit=verify_store(root);assert audit['committed']==1 and audit['missing']==1
        # Simulate orphan file from interruption, no auto-deletion.
        orphan=fp.parent/'.pending-interrupted.tmp';orphan.write_bytes(b'partial')
        audit=verify_store(root);assert audit['orphan_temp_files']==1 and orphan.exists()
    print('PASS: plan code lock, 20 metadata-only jobs, D4+transposition family, exact split/arena pairing, disabled/hash/disk gates, 2 fixture records, censor/label/feature/behavior corruption, duplicate/checksum, atomic nonoverwrite, lock, partial resume and orphan preservation.')
if __name__=='__main__':main()
