"""Explicitly authorized, resumable Gomoku15 game collector.
Default CLI is read-only dry-run; imports Torch only after a valid execution gate.
The smoke path is separate and cannot launch a production plan.
"""
import argparse,collections,fcntl,gzip,hashlib,json,math,os,random,shutil,sqlite3,sys,tempfile,time,subprocess
from functools import lru_cache
from pathlib import Path
from rules import ADMITTED,replay,orbit_history,orbit_board,rule_key
from engine import D,ROOT,MODEL,BIN,MODEL_SHA,PYTHON,identity,sha
from native_smoke import check as check_native

PLAN_SCHEMA='quartz_long_plan_v1'
SMOKE_SCHEMA='quartz_long_smoke_plan_v1'
GAME_SCHEMA='quartz_long_game_v1'
MANIFEST_SCHEMA='quartz_long_manifest_v1'
MAX_RECORD_BYTES=64*1024*1024
GIB=1024**3

def canonical(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(o):return hashlib.sha256(canonical(o)).hexdigest()
def derive_seed(*parts):return int.from_bytes(hashlib.sha256(canonical(parts)).digest()[:8],'big')
def family_id(rule,h):
    b,p,w=replay(h,rule);require(w is None,'terminal opening family')
    # Same board under D4 or reordered compatible histories belongs to one family.
    return hashlib.sha256(orbit_board(b,p,rule)).hexdigest()
@lru_cache(maxsize=1)
def collection_lock():
    paths=[D/n for n in ('collector.py','engine.py','rules.py','native_smoke.py','prepare_plan.py')]
    paths += [ROOT/'quartz'/n for n in ('models_torch.py','training_catalog.py','encoders.py','runtime_support.py','selfplay_runtime.py','eval_runtime.py','system_runtime.py')]
    env=subprocess.check_output([PYTHON,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"torch":m.version("torch"),"numpy":m.version("numpy")}))'],text=True)
    return dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},interpreter=PYTHON,interpreter_sha256=sha(Path(PYTHON).resolve()),versions=json.loads(env))
def search_visits(p,arm):
    return p['candidate_visits'] if p['mode']=='PAIRED_ARENA' and arm=='CERT' else p['visits']

def integer(x,lo,hi):return type(x) is int and lo<=x<=hi
def require(test,message):
    if not test:raise ValueError(message)
def validate_plan(p):
    fields={'schema','generation_enabled','mode','rule','games','replicates_per_family','seed','opening_plies','opening_radius','visits','threads','temperature_plies','max_ply','shards','arm','candidate_arm','min_free_gib','min_available_gib','estimated_compressed_bytes_per_ply','identity','collection_lock','candidate_visits'}
    require(set(p)==fields,'unexpected/missing plan fields')
    require(p['schema']==PLAN_SCHEMA and type(p['generation_enabled']) is bool,'plan schema')
    require(p['mode'] in ('SELFPLAY','PAIRED_ARENA') and p['rule'] in ADMITTED,'unsupported mode or rules')
    for k,lo,hi in [('games',1,1000000),('replicates_per_family',1,16),('seed',0,2**64-1),('opening_plies',4,8),('opening_radius',2,6),('visits',1,4096),('candidate_visits',1,4096),('threads',1,2),('temperature_plies',0,225),('max_ply',5,225),('shards',1,64),('min_free_gib',5,10000),('min_available_gib',2,1000),('estimated_compressed_bytes_per_ply',1024,262144)]:
        require(integer(p[k],lo,hi),'bad '+k)
    require(p['games']%p['replicates_per_family']==0,'fractional families')
    require(p['opening_plies']<p['max_ply'] and p['visits']%p['threads']==0 and p['candidate_visits']%p['threads']==0,'invalid ply/visit budget')
    require(p['arm'] in ('BASE','CERT') and p['candidate_arm'] in ('BASE','CERT'),'unknown arm')
    if p['mode']=='PAIRED_ARENA':require(p['replicates_per_family']==2 and p['arm']=='BASE' and p['candidate_arm']=='CERT','paired arena must have BASE vs CERT color swaps')
    require(p['identity']==identity(),'engine/model/source identity mismatch')
    require(p['collection_lock']==collection_lock(),'collector/source/interpreter identity mismatch')
    return p
def job_stream(p):
    validate_plan(p)
    r=random.Random(p['seed']);N=p['games']//p['replicates_per_family'];seen=set()
    cells=[15*i+j for i in range(7-p['opening_radius'],8+p['opening_radius']) for j in range(7-p['opening_radius'],8+p['opening_radius'])]
    for i in range(N):
        for _ in range(10000):
            h=r.sample(cells,p['opening_plies']);fam=family_id(p['rule'],h)
            if fam not in seen and replay(h,p['rule'])[2] is None:break
        else:raise RuntimeError('opening family generator exhausted')
        seen.add(fam);split=('TRAIN' if i%5<3 else 'CAL' if i%5==3 else 'TEST')
        for k in range(p['replicates_per_family']):
            paired=p['mode']=='PAIRED_ARENA'
            seed=derive_seed(p['seed'],fam,0 if paired else k)
            black,white=(p['candidate_arm'],p['arm']) if paired and k==0 else (p['arm'],p['candidate_arm']) if paired else (p['arm'],p['arm'])
            yield dict(game_id=f'F{i:07d}_R{k:02d}',family=fam,family_index=i,replicate=k,split=split,shard=i%p['shards'],opening=h,seed=seed,black_arm=black,white_arm=white)
def mem_available():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')
def disk_ancestor(path):
    p=Path(path).expanduser().absolute()
    while not p.exists():p=p.parent
    return p
def preflight(p,out,remaining=None):
    n=p['games'] if remaining is None else remaining
    worst=n*p['max_ply']*p['estimated_compressed_bytes_per_ply']
    disk=shutil.disk_usage(disk_ancestor(out));mem=mem_available()
    return dict(free_bytes=disk.free,available_memory_bytes=mem,estimated_max_ply_bytes=worst,estimated_bytes_per_ply=p['estimated_compressed_bytes_per_ply'],reserve_bytes=p['min_free_gib']*GIB,disk_ok=disk.free>=worst+p['min_free_gib']*GIB,memory_ok=mem>=p['min_available_gib']*GIB,estimate_not_guarantee=True)
def authorize(p,plan_sha,limit,shard,out):
    validate_plan(p)
    require(p['generation_enabled'] is True,'generation_enabled=false: no production collection')
    require(plan_sha==digest(p),'explicit exact plan hash authorization required')
    require(integer(shard,0,p['shards']-1),'invalid shard')
    require(integer(limit,1,p['games']),'explicit bounded max-new-games required')
    path=Path(out)
    remaining=p['games']
    if (path/'manifest.json').is_file():
        require(json.loads((path/'manifest.json').read_text())==manifest(p),'resume plan mismatch')
        remaining-=verify_store(path)['committed']
    elif path.exists():require(path.is_dir() and not any(path.iterdir()),'nonempty output without manifest')
    pf=preflight(p,out,remaining)
    require(pf['disk_ok'] and pf['memory_ok'],'bulk preflight blocked: insufficient disk reserve or memory')
    return pf
def manifest(p):
    return dict(schema=MANIFEST_SCHEMA,plan=p,plan_sha256=digest(p),identity=p['identity'])
def fsync_dir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def atomic_create(p,raw):
    """fsync then hard-link without overwrite; caller owns only its temp file."""
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.pending-',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,p)
        except FileExistsError:
            require(p.is_file() and p.read_bytes()==raw,'existing immutable file differs')
        fsync_dir(p.parent)
    finally:os.unlink(tmp)
def init_store(root,p):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    mf=root/'manifest.json';data=canonical(manifest(p))+b'\n'
    atomic_create(mf,data)
    require(mf.read_bytes()==data,'manifest identity mismatch')
    return root
def shard_dir(root,shard):return Path(root)/f'shard_{shard:03d}'
def record_bytes(record):
    return gzip.compress(canonical(record)+b'\n',compresslevel=5,mtime=0)
def load_record(p):
    p=Path(p);require(p.stat().st_size<=MAX_RECORD_BYTES,'compressed record cap');raw=p.read_bytes()
    require(len(raw)<=MAX_RECORD_BYTES,'compressed record cap')
    parts=p.name.split('.');require(len(parts)==4 and parts[-2:]==['json','gz'],'invalid record filename')
    require(hashlib.sha256(raw).hexdigest()==parts[1],'record checksum mismatch')
    # Cap decompressed read as well; no arbitrary gzip expansion.
    with gzip.GzipFile(fileobj=__import__('io').BytesIO(raw)) as f:data=f.read(MAX_RECORD_BYTES+1)
    require(len(data)<=MAX_RECORD_BYTES,'decompressed record cap')
    x=json.loads(data);require(x.get('job',{}).get('game_id')==parts[0],'record id/filename mismatch')
    return x
def existing(root,p,job):
    folder=shard_dir(root,job['shard']);files=list(folder.glob(job['game_id']+'.*.json.gz'))
    require(len(files)<=1,'duplicate record id')
    if not files:return None
    rec=load_record(files[0]);validate_record(rec,p,job)
    return files[0]
def save_record(root,p,job,rec,production=False):
    validate_record(rec,p,job)
    require(existing(root,p,job) is None,'record already committed; do not overwrite')
    data=record_bytes(rec);require(len(data)<=MAX_RECORD_BYTES and len(canonical(rec))+1<=MAX_RECORD_BYTES,'record exceeds cap')
    if production:
        free=shutil.disk_usage(disk_ancestor(root)).free
        require(free>=p['min_free_gib']*GIB+len(data),'disk reserve before record commit')
    folder=shard_dir(root,job['shard']);name=f"{job['game_id']}.{hashlib.sha256(data).hexdigest()}.json.gz"
    target=folder/name;atomic_create(target,data)
    require(load_record(target)==rec,'committed record readback mismatch')
    return target
def policy_target(a):
    if a['cause']=='CERTIFIED':return None
    counts=sorted([int(x['move']),int(x['n'])] for x in a['result']['edges'] if x['n']>0)
    total=sum(x[1] for x in counts);require(total==a['visits'] and total>0,'native visit counts mismatch')
    return dict(counts=counts,total=total)
def choose(a,job,ply,p):
    if a['cause']=='CERTIFIED':return a['best'],[1,1],None
    pi=policy_target(a)
    if ply>=p['temperature_plies']:return a['best'],[1,1],pi
    allowed=set(a['legal'])
    if a['arm']=='CERT':
        lo=a['proof']['interval'][0]
        allowed={m for m,l,u,rep in a['proof']['actions'] if u>=lo}
    counts=[(m,n) for m,n in pi['counts'] if m in allowed];total=sum(n for _,n in counts)
    if total==0:return a['best'],[1,1],pi
    j=random.Random(derive_seed(job['seed'],ply,'behavior')).randrange(total)
    for m,n in counts:
        if j<n:return m,[n,total],pi
        j-=n
    raise AssertionError('unreachable visit sampler')
def z_label(w,player):
    require(w is None or w in (-1,0,1),'invalid winner')
    require(player in (-1,1),'invalid player')
    return None if w is None else w*player
def validate_record(rec,p,job,positions=None):
    require(set(rec)=={'schema','plan_sha256','identity','job','rule','status','reason','winner_black','moves','final_history','record_type'},'game schema keys')
    require(rec['schema']==GAME_SCHEMA and rec['plan_sha256']==digest(p) and rec['identity']==p['identity'] and rec['job']==job and rec['rule']==p['rule'],'record identity mismatch')
    is_smoke=p['schema']==SMOKE_SCHEMA;require(rec['record_type']==('SMOKE_FIXTURE' if is_smoke else 'SELFPLAY' if p['mode']=='SELFPLAY' else 'PAIRED_ARENA'),'record type mismatch')
    h=list(job['opening']);_,_,w=replay(h,p['rule']);require(w is None,'initial terminal history')
    limit=job.get('max_ply',p['max_ply']) if is_smoke else p['max_ply']
    require(len(h)<limit<=225,'invalid game cap')
    for row in rec['moves']:
        require(set(row)=={'ply','player','played_move','behavior_probability','policy_target','z','native'},'move schema')
        b,player,w0=replay(h,p['rule']);require(w0 is None and len(h)<limit,'search after terminal/cap')
        require(row['ply']==len(h) and row['player']==player,'wrong ply/player')
        a=dict(row['native'],history=list(h),board=b,rule=p['rule'])
        require(a['arm']==(job['black_arm'] if player==1 else job['white_arm']) and a['requested_visits']==search_visits(p,a['arm']) and a['threads']==p['threads'] and a['seed']==derive_seed(job['seed'],len(h),'search'),'search config mismatch')
        check_native(a,h,p['rule'])
        if not is_smoke:require(all(integer(a.get(k),0,2**63-1) for k in ('python_cpu_ns','rust_cpu_ticks')) and integer(a.get('tick_hz'),1,1000000),'missing CPU accounting')
        m,pr,pi=choose(a,job,len(h),p)
        require(row['played_move']==m and row['behavior_probability']==pr and row['policy_target']==pi and m in a['legal'],'behavior/visit target mismatch')
        require(row['z']==z_label(rec['winner_black'],player),'wrong terminal label')
        if positions is not None:positions(b,player,row,h)
        h.append(m)
    b,p1,w=replay(h,p['rule'])
    require(h==rec['final_history'] and rec['winner_black']==w,'terminal replay mismatch')
    require(rec['status']==('COMPLETED' if w is not None else 'CENSORED'),'censoring mismatch')
    require(rec['reason']==('RULE_TERMINAL' if w is not None else 'PLY_CAP'),'termination reason mismatch')
    require(w is not None or len(h)==limit,'incomplete record without cap')
    return dict(plies=len(h),searched=len(rec['moves']),winner=w)
def play_game(engine,p,job):
    h=list(job['opening']);moves=[];limit=job.get('max_ply',p['max_ply']) if p['schema']==SMOKE_SCHEMA else p['max_ply']
    while True:
        b,player,w=replay(h,p['rule'])
        if w is not None or len(h)>=limit:break
        arm=job['black_arm'] if player==1 else job['white_arm']
        a=engine.search(h,p['rule'],arm,search_visits(p,arm),p['threads'],derive_seed(job['seed'],len(h),'search'))
        check_native(a,h,p['rule'])
        m,pr,pi=choose(a,job,len(h),p)
        # Exact history/board reconstructible from opening + chosen actions.
        compact={k:v for k,v in a.items() if k not in ('history','board','rule')}
        moves.append(dict(ply=len(h),player=player,played_move=m,behavior_probability=pr,policy_target=pi,z=None,native=compact))
        h.append(m)
    _,_,w=replay(h,p['rule'])
    for row in moves:row['z']=z_label(w,row['player'])
    r=dict(schema=GAME_SCHEMA,plan_sha256=digest(p),identity=p['identity'],job=job,rule=p['rule'],status='COMPLETED' if w is not None else 'CENSORED',reason='RULE_TERMINAL' if w is not None else 'PLY_CAP',winner_black=w,moves=moves,final_history=h,record_type='SMOKE_FIXTURE' if p['schema']==SMOKE_SCHEMA else 'SELFPLAY' if p['mode']=='SELFPLAY' else 'PAIRED_ARENA')
    validate_record(r,p,job)
    return r
def execute(p,out,shard,limit,plan_sha):
    authorize(p,plan_sha,limit,shard,out) # MUST precede model import or directory creation
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and sys.prefix==str(Path(PYTHON).parent.parent),'use locked Torch virtualenv')
    from engine import Engine
    root=init_store(out,p);folder=shard_dir(root,shard);folder.mkdir(exist_ok=True)
    new=skipped=0
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine=None
        try:
            for job in job_stream(p):
                if job['shard']!=shard:continue
                if existing(root,p,job):skipped+=1;continue
                if new>=limit:break
                # Check real resource state before each game, no autonomous retry.
                require(shutil.disk_usage(root).free>=p['min_free_gib']*GIB+MAX_RECORD_BYTES and mem_available()>=p['min_available_gib']*GIB,'resource reserve crossed')
                if engine is None:engine=Engine(p['threads'])
                r=play_game(engine,p,job)
                path=save_record(root,p,job,r,production=True);new+=1
                print(json.dumps(dict(game=job['game_id'],status=r['status'],plies=len(r['final_history']),path=path.name)),flush=True)
        finally:
            if engine is not None:engine.close()
    return dict(new=new,skipped=skipped)
def planned_jobs(p):
    if p['schema']==SMOKE_SCHEMA:
        require(not p['generation_enabled'] and len(p['jobs'])<=2,'invalid smoke plan')
        return iter(p['jobs'])
    return job_stream(p)
def verify_store(root,probe_manifest=None):
    """Read-only data validation; disk-backed exact D4 board leakage index.
    A probe manifest is metadata only, not MCTS execution or labels for alternatives.
    """
    root=Path(root);mf=json.loads((root/'manifest.json').read_text());p=mf['plan']
    require(mf==manifest(p) and p['identity']==identity(),'manifest/model/binary mismatch')
    if p['schema']==PLAN_SCHEMA:validate_plan(p)
    jobs={j['game_id']:j for j in planned_jobs(p)}
    counts=collections.Counter();ids=set();pairs=collections.defaultdict(list);probe_rows=[]
    length_hist=collections.Counter();ply_hist=collections.Counter();rule_proofs=collections.Counter();arm_cost=collections.defaultdict(collections.Counter)
    with tempfile.TemporaryDirectory(prefix='quartz_long_verify_') as tmp:
        db=sqlite3.connect(str(Path(tmp)/'positions.sqlite'))
        db.execute('CREATE TABLE states (key BLOB PRIMARY KEY, mask INTEGER NOT NULL, count INTEGER NOT NULL)')
        # Per-game records are committed atomically. Exact cross-split state
        # repeats are flagged, not silently reassigned after outcomes.
        for file in sorted(root.glob('shard_*/*.json.gz')):
            r=load_record(file);j=jobs.get(r['job']['game_id']);require(j is not None and j['game_id'] not in ids,'unplanned or duplicate game')
            require(file.parent==shard_dir(root,j['shard']),'wrong shard')
            ids.add(j['game_id']);mask={'TRAIN':1,'CAL':2,'TEST':4,'SMOKE':8}[j['split']]
            def position(b,pl,row,h):
                key=orbit_board(b,pl,p['rule']);db.execute('INSERT INTO states VALUES(?,?,1) ON CONFLICT(key) DO UPDATE SET mask=mask|excluded.mask,count=count+1',(key,mask))
                ply_hist[f"{j['split']}/{len(h)//20*20:03d}"]+=1
                if row['native']['proof']['witness'] is not None:rule_proofs[j['split']]+=1
                if probe_manifest is not None and p['schema']==PLAN_SCHEMA and j['split']=='TEST' and derive_seed(j['game_id'],len(h),'probe')%32==0:
                    # Export only eligible post-opening histories, deterministic
                    # hash inclusion independent of outcome. Later exact leak
                    # filter from all observed cross-split boards.
                    probe_rows.append((key,dict(game_id=j['game_id'],family=j['family'],ply=len(h),history=list(h),rule=p['rule'],player=pl,assignment_seed=derive_seed(j['game_id'],len(h),'assignment'))))
            m=validate_record(r,p,j,position);db.commit()
            counts[j['split']]+=1;counts[r['status']]+=1;counts['searched_plies']+=m['searched'];counts['NN_rows']+=sum(row['native']['nn_rows'] for row in r['moves']);counts['compressed_bytes']+=file.stat().st_size
            length_hist[f"{j['split']}/{m['plies']//20*20:03d}"]+=1
            for row in r['moves']:
                a=row['native'];cc=arm_cost[a['arm']];cc['moves']+=1;cc['wall_ns']+=a['transport_ns'];cc['visits']+=a['visits'];cc['NN_rows']+=a['nn_rows']
                if 'python_cpu_ns' in a:
                    cc['python_cpu_ns']+=a['python_cpu_ns'];cc['rust_cpu_ns_tick_estimate']+=a['rust_cpu_ticks']*10**9//a['tick_hz'];cc['CPU_measured_moves']+=1
            if p['schema']==PLAN_SCHEMA and p['mode']=='PAIRED_ARENA':
                pairs[j['family']].append((j,r['winner_black'],m['searched'],sum(x['native']['transport_ns'] for x in r['moves'])))
        state_count,obs=db.execute('SELECT COUNT(*),COALESCE(SUM(count),0) FROM states').fetchone()
        cross=db.execute('SELECT COUNT(*) FROM states WHERE (mask & (mask-1)) != 0').fetchone()[0]
        if probe_manifest is not None:
            require(p['schema']==PLAN_SCHEMA,'no smoke probes')
            require(len(ids)==len(jobs),'probe plan requires complete corpus; avoid changing leakage status')
            probes=[r for key,r in probe_rows if db.execute('SELECT mask FROM states WHERE key=?',(key,)).fetchone()[0]==4]
            atomic_create(Path(probe_manifest),canonical(dict(schema='quartz_probe_plan_v1',plan_sha256=digest(p),inclusion='hash mod32, TEST, exclude exact D4 state shared across splits',positions=probes))+b'\n')
        else:probes=[]
        db.close()
    # Paired game candidate WDL score: color-adjusted, complete pairs only.
    paired_scores=[]
    for fam,rr in pairs.items():
        if len(rr)==2 and all(x[1] is not None for x in rr):
            rr=sorted(rr,key=lambda x:x[0]['replicate'])
            require(rr[0][0]['black_arm']=='CERT' and rr[1][0]['white_arm']=='CERT','invalid arena swap')
            paired_scores.append((2+rr[0][1]-rr[1][1])/4)
    return dict(plan_sha256=digest(p),arm_cost={k:dict(v) for k,v in arm_cost.items()},planned=len(jobs),committed=len(ids),missing=len(jobs)-len(ids),counts=dict(counts),length_bins=dict(length_hist),searched_ply_bins=dict(ply_hist),rule_witnesses=dict(rule_proofs),unique_D4_states=state_count,observed_positions=obs,cross_split_D4_states=cross,probe_count=len(probes),completed_arena_pairs=len(paired_scores),candidate_score=sum(paired_scores)/len(paired_scores) if paired_scores else None,orphan_temp_files=len(list(root.glob('shard_*/.pending-*.tmp'))))
def main():
    require(__debug__,'validation requires assertions enabled; do not run Python -O')
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=D/'PLAN_DISABLED.json')
    parser.add_argument('--output',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--execute',action='store_true');group.add_argument('--verify',action='store_true');group.add_argument('--dry-run',action='store_true')
    parser.add_argument('--authorize-hash');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--max-new-games',type=int)
    parser.add_argument('--probe-manifest',type=Path)
    args=parser.parse_args()
    if args.verify:
        require(args.output is not None,'--output required for verify')
        print(json.dumps(verify_store(args.output,args.probe_manifest),indent=2,sort_keys=True));return
    p=validate_plan(json.loads(args.plan.read_text()));out=args.output or ROOT.parent/'quartz_long_corpus_NOT_STARTED'
    if args.execute:
        require(args.output is not None,'explicit output required')
        print(json.dumps(execute(p,out,args.shard,args.max_new_games,args.authorize_hash),indent=2));return
    pf=preflight(p,out);N=p['games']//p['replicates_per_family'];splits={s:0 for s in ('TRAIN','CAL','TEST')}
    for i in range(N):splits['TRAIN' if i%5<3 else 'CAL' if i%5==3 else 'TEST']+=p['replicates_per_family']
    print(json.dumps(dict(mode='DRY_RUN_NO_ENGINE',generation_enabled=p['generation_enabled'],plan_sha256=digest(p),planned_games=p['games'],families=N,planned_splits=splits,shards=p['shards'],max_plies=p['max_ply'],output=str(out),output_created=False,preflight=pf,ready_for_authorized_launch=p['generation_enabled'] and pf['disk_ok'] and pf['memory_ok']),indent=2,sort_keys=True))
if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,BlockingIOError) as exc:
        print('BLOCKED:',exc,file=sys.stderr);raise SystemExit(2)
