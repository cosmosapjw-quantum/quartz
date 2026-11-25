"""Independent grid replay, exact WDL, permutation Shapley and durable restore."""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import collections as C,gzip,json,math,statistics as S,hashlib,subprocess as SP,tempfile
from publish import ROOT,D,B,BASE,BR,sha,git
from population import exclusions
import experiment as e
import native as n
LINES=[tuple((y+j*dy)*7+x+j*dx for j in range(4)) for y in range(6) for x in range(7) for dx,dy in ((1,0),(0,1),(1,1),(-1,1)) if 0<=x+3*dx<7 and 0<=y+3*dy<6]
def read(f):return json.loads((D/f).read_text())
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-7),(a,b)
def win(b,p):return any(all(b[k]==p for k in line) for line in LINES)
def replay(h):
    b=[0]*42;z=[0]*7
    for i,x in enumerate(h):
        c=int(x)-1;p=1-2*(i%2)
        assert 0<=c<7 and z[c]<6 and not win(b,p) and not win(b,-p)
        b[7*z[c]+c]=p;z[c]+=1
    return b,z
def immediate(b,z,p):
    return any(win(b[:7*z[c]+c]+[p]+b[7*z[c]+c+1:],p) for c in range(7) if z[c]<6)
def canon(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def cb(b):return min(tuple(b),tuple(b[7*y+x] for y in range(6) for x in range(6,-1,-1)))
def u(x):return F(1+(x>0)-(x<0),2)
def encode(b,p):
    return '%022x'%((int(p==1)<<84)|sum((v+1)<<(2*i) for i,v in enumerate(b)))
def decode(k):
    n=int(k,16);assert n>>85==0
    b=[((n>>(2*i))&3)-1 for i in range(42)];p=1 if n>>84 else -1
    assert all(v in (-1,0,1) for v in b)
    z=[]
    for c in range(7):
        col=[b[y*7+c] for y in range(6)];h=sum(x!=0 for x in col)
        assert all(v!=0 for v in col[:h]) and all(v==0 for v in col[h:]);z.append(h)
    assert b.count(1)-b.count(-1)==(p==-1)
    return b,p,z
def outcome(b,p):
    w1,w2=win(b,1),win(b,-1);assert not(w1 and w2)
    if w1:return p
    if w2:return -p
    if all(v!=0 for v in b):return 0
    return None
def fnv(b,p):
    z=0xcbf29ce484222325
    for v in b+[p]:z=((z^(v+2))*0x100000001b3)&((1<<64)-1)
    return '%016x'%z
def rb(a,move):
    iv=dict(a)
    return F(max(0,max((u for j,(_,u) in iv.items() if j!=move),default=-1)-iv[move][0]),2)
def replay_schedule(a):
    """Independent reference: replay every actual selected block, retirements and credits."""
    arm=a['arm'];mo=a['mode'];rows=a['rows'];stop=arm in e.STOP;fixed=arm in e.FIXED
    assert len(rows)==4 and a['total_blocks']==len(a['trace'])
    st=[dict(r['initial']) for r in rows]
    dead=set();cert_events={};ret=a['retired'];assert len(ret)==4 and len({x['i'] for x in ret})==4
    assert all(0<=x['i']<4 and 0<=x['after']<=a['total_blocks'] for x in ret)
    for x in st:
        assert x['blocks']==x['spent_us']==0 and x['visits']==8
        if arm!='BASE':assert (x['witness'] is not None)==(x['risk2']==0)
    for i,x in enumerate(st):
        if stop and x['witness'] is not None:dead.add(i);cert_events[i]=0
    cur=0
    def eligible():
        return [i for i in range(4) if i not in dead and st[i]['blocks']<63 and (not fixed or (st[i]['blocks']<15 if mo=='WORK' else st[i]['spent_us']<a['allowance_us']))]
    for k,t in enumerate(a['trace'],1):
        z=eligible();assert z
        if arm=='BOUND':
            # independent ranking, ascending lexicographic negative keys
            i=min(z,key=lambda i:(-st[i]['risk2'],-st[i]['contenders'],st[i]['blocks'],(i-cur)%4))
        else:i=min(z,key=lambda i:(i-cur)%4)
        assert t['i']==i and st[i]==t['before'],(a['arm'],k,i,t['i'])
        x=t['before'];y=t['after'];charge=t['charge_us']
        assert charge>=1 and y['blocks']==x['blocks']+1 and y['visits']==x['visits']+8 and y['evals']>=x['evals']
        assert y['spent_us']==x['spent_us']+charge and 0<=y['risk2']<=x['risk2']<=2
        if arm!='BASE':assert (y['witness'] is not None)==(y['risk2']==0)
        st[i]=y;cur=(i+1)%4
        if stop and y['witness'] is not None:dead.add(i);cert_events[i]=k
    actual={x['i']:x['after'] for x in ret if x['reason']=='CERTIFIED'}
    assert cert_events==actual
    for x in ret:
        r=rows[x['i']];s=st[x['i']];assert r['reason']==x['reason']
        assert r['blocks']==s['blocks'] and r['spent_us']==s['spent_us'] and r['result']['root_n']==s['visits'] and r['result']['evaluator_calls']==s['evals']
        if x['reason']!='CERTIFIED':
            assert x['after']==a['total_blocks']
            reason='CAP' if s['blocks']>=63 else 'QUOTA' if fixed and (s['blocks']>=15 if mo=='WORK' else s['spent_us']>=a['allowance_us']) else a['cause']
            assert x['reason']==reason
        g=r['proof']
        if g is not None:
            d=min(rb(g['actions'],m) for m,_ in g['actions'])*2
            h=sum(u>=g['interval'][0] and l<u for _,(l,u) in g['actions'])
            assert d==s['risk2'] and h==s['contenders'] and g['witness']==s['witness']
    assert a['total_blocks']==sum(s['blocks'] for s in st)
    assert a['drop_total_us']==sum(r['drop_us'] for r in rows)
    assert a['whole_us']>=a['before_final_us']>=a['mandatory_us']
    assert a['whole_us']>=a['mandatory_us']+sum(s['spent_us'] for s in st)+a['drop_total_us']-a['total_blocks']
    if a['cause']=='NO_ELIGIBLE':assert not eligible()
    elif a['cause']=='WORK':assert mo=='WORK' and a['total_blocks']==60
    elif a['cause']=='TIME':assert mo=='TIME' and a['before_final_us']>=a['target_us']-a['reserve_us']
    else:raise AssertionError(a['cause'])
    if mo=='WORK':
        b=[s['blocks'] for s in st];C0=sum(max(0,15-b[i]) for i in cert_events);S0=sum(max(0,15-v) for v in b);E=sum(max(0,v-15) for v in b);U=60-sum(b)
        assert S0-E==U>=0 and C0<=S0 and a['account']==dict(credit=C0,extra=E,shortfall=S0,unused=U)
        if arm in e.FIXED:assert E==0
        if arm in ('BASE','ACTIVE_FIXED'):assert b==[15]*4
        if arm not in e.FIXED and arm!='BOUND':assert C0==S0
        return dict(credit=C0,extra=E,unused=U,noncert_shortfall=S0-C0,cert_reuse_lb=max(0,C0-U),cert_reuse_ub=min(C0,E))
    assert a['account'] is None
    assert a['allowance_us']==max(0,a['target_us']-a['reserve_us']-a['mandatory_us'])//4
    return dict(credit=0,extra=0,unused=0,noncert_shortfall=0,cert_reuse_lb=0,cert_reuse_ub=0)
def audit():
    e.lock()
    co=read('C07_COHORT.json');ps=read('C07_POOLS.json');sm=e.read_gz('C06_SMOKE.json.gz');cal=e.read_gz('C07_CAL_RAW.json.gz');cost=read('C07_COST_LOCK.json');su=read('C08_SUMMARY.json');host=read('C08_LIVE_HOST.json')
    assert sha(D/'C07_ATTEMPTS.jsonl.gz')==co['attempt_sha'] and sha(D/'C08_LIVE_RAW.jsonl.gz')==su['raw_sha']==host['raw_sha']
    assert cost['pools_sha']==sha(D/'C07_POOLS.json') and ps['cohort_sha']==sha(D/'C07_COHORT.json') and cost['raw_sha']==sha(D/'C07_CAL_RAW.json.gz') and cost['binary_sha']==sha(e.BIN)
    oldp,oldb,meta=exclusions();assert len(oldp)==co['old_families'] and len(oldb)==co['old_boards'] and meta==co['sources']
    ids={r['id']:r for r in co['rows']};seen=set();families=set();boards=set();accepted=set();events=0;counts=C.Counter();attempts=C.Counter()
    with gzip.open(D/'C07_ATTEMPTS.jsonl.gz','rt') as fp:
        for line in fp:
            a=json.loads(line);i=a['id'];h=a['history'];op=a['opening'];f=canon(op)
            assert i not in seen and f not in oldp|families and h.startswith(op);seen.add(i);families.add(f);attempts[a['stratum']+'/'+a['group']]+=1
            if a['stratum']=='SELFPLAY':
                hist=op
                for ev in a['events']:
                    b,z=replay(hist);assert not win(b,1) and not win(b,-1)
                    assert ev['ply']==len(hist) and all(0<=c<7 and z[c]<6 for c in (ev['action'],ev['recommendation']))
                    assert ev['explore'] or ev['action']==ev['recommendation'];assert ev['eval_calls']>=1
                    hist+=str(ev['action']+1);events+=1
                assert hist==h
            else:
                b,z=replay(h);assert not win(b,1) and not win(b,-1)
            if a['candidate'] is None:continue
            r=ids[i];b,z=replay(h);p=1-2*(len(h)%2)
            assert a['status']=='ACCEPT' and a['candidate']==h and i not in accepted and not win(b,p) and not win(b,-p);accepted.add(i)
            assert r['history']==h and r['board']==b and r['legal']==[j for j in range(7) if z[j]<6] and len(r['legal'])>=2
            assert r['family']==f and r['player']==p and r['ply']==len(h) and r['stratum']==a['stratum'] and len(h)%2==a['parity'] and r['group']==a['group']
            assert r['immediate']==immediate(b,z,p)
            if r['stratum']=='TACTICAL':assert 18<=len(h)<=25 and not r['immediate']
            elif r['stratum']=='HARD':assert 20<=len(h)<=25 and len(r['legal'])>=4 and not r['immediate'] and not immediate(b,z,-p)
            else:assert 16<=len(h)<=26
            k=cb(b);assert k not in oldb|boards;boards.add(k);counts[r['stratum']+'/'+r['group']+'/'+str(p)]+=1
    assert accepted==set(ids) and len(seen)==co['attempts'] and len(ids)==132
    assert all(counts[st+'/'+g+'/'+str(p)]==({'TRAIN':6,'CAL':4,'TEST':12}[g]) for st in e.ST for g in ('TRAIN','CAL','TEST') for p in (-1,1))
    pools={p['id']:p for p in ps['pools']};assert len(pools)==33
    all_ids=[]
    for p in pools.values():
        assert len(p['rows'])==4
        for r in p['rows']:assert r==ids[r['id']] and r['stratum']==p['stratum'] and r['group']==p['group'];all_ids.append(r['id'])
    assert len(all_ids)==len(set(all_ids))==len(ids)
    base=sm['tests'][0];assert all(r['prefix']==a['result'] for r,a in zip(base['rows'],sm['parity']))
    for a in sm['tests']+cal['timing']+cal['checks']:replay_schedule(a)
    for st in e.ST:
        rr=[a for a in cal['timing'] if a['stratum']==st];v=cost['cost'][st]
        assert len(rr)==6 and all(pools[a['pool']]['group']=='TRAIN' for a in rr)
        med=S.median(a['whole_us'] for a in rr);drop=S.median(a['drop_total_us'] for a in rr)
        assert v['median_whole_us']==med and v['median_drop_us']==drop and v['target']==min(40000,max(6000,math.ceil(med))) and v['reserve']==min(v['target']-1,math.ceil(drop)+500)
    assert len(cal['checks'])==54 and len({(a['pool'],a['arm']) for a in cal['checks']})==54 and all(pools[a['pool']]['group']=='CAL' for a in cal['checks'])
    test={i:r for i,r in ids.items() if r['group']=='TEST'}
    labs={r['id']:r['scores'] for r in read('C08_LABELS.json') if r['status']=='EXACT'};assert set(labs)==set(test)
    for i,z in labs.items():assert len(z)==7 and all((v!=-1000)==(j in test[i]['legal']) for j,v in enumerate(z))
    mirrors=0
    for st in e.ST:
        for r in sorted((x for x in test.values() if x['stratum']==st),key=lambda x:x['id'])[:3]:
            h=r['history'].translate(str.maketrans('1234567','7654321'))
            z=SP.run([str(n.EXT/'c4solver'),'-a','-b',str(n.EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=n.EXT,timeout=15,check=True).stdout.split()
            assert z[0]==h and list(map(int,z[1:]))==labs[r['id']][::-1];mirrors+=1
    results={};prefix={};pref_n=0;cert=0;action_n=0;stops=0;value_unsolved_cert=0;blocks=0;hist=C.Counter();worker=C.Counter();exposure=C.Counter();zero_unresolved=C.Counter();zero_regret=C.Counter()
    for a in e.records('C08_LIVE_RAW.jsonl.gz'):
        pid=a['pool'];s=a['seed'];mo=a['mode'];arm=a['arm'];p0=pools[pid];st=p0['stratum']
        assert p0['group']=='TEST' and a['stratum']==st and a['model_identity']==n.IDENT and (pid,s,mo,arm) not in results
        account=replay_schedule(a);blocks+=a['total_blocks'];rr=[]
        if mo=='TIME':
            cfg=cost['cost'][st];assert a['target_us']==cfg['target'] and a['reserve_us']==cfg['reserve']
        for r,r0 in zip(a['rows'],p0['rows']):
            assert r['id']==r0['id'] and r['history']==r0['history']
            i=r['id'];v=r['result'];p=r['prefix'];h=r['hooks'];ph=r['prefix_hooks']
            assert p['root_n']==8 and v['root_n']==8+8*r['blocks']<=512 and v['root_hash']==p['root_hash']==fnv(r0['board'],r0['player'])
            k=(i,s,arm=='BASE')
            if k in prefix:assert prefix[k]==p;pref_n+=1
            else:prefix[k]=p
            for x in (p,v):
                assert {j[0] for j in x['candidates']}==set(r0['legal']) and x['best'] in r0['legal']
                assert [j['move'] for j in x['edges']]==[j[0] for j in x['candidates'][:len(x['edges'])]]
                assert sum(j['n'] for j in x['edges'])==x['root_n'];close(sum(j['w'] for j in x['edges']),x['root_w'])
                for j in x['edges']:assert j['vl']==0 and abs(j['vvalue'])<1e-6 and abs(j['w'])<=j['n']+1e-6
            assert ph['base_evals']+ph['terminal_leaves']==8 and p['evaluator_calls']==1+ph['base_evals']
            assert h['base_evals']+h['terminal_leaves']+h['graph_solved_skips']==v['root_n']-8 and v['evaluator_calls']==p['evaluator_calls']+h['base_evals'] and h['extra_evals']==0
            if r['blocks']>0:worker[(mo,arm,'roots')]+=1;worker[(mo,arm,'two')]+=h['worker_mask'].bit_count()>=2
            z=labs[i];opt=max(u(j) for j in z if j!=-1000);assert r['best'] in r0['legal'];reg=opt-u(z[r['best']]);g=r['proof']
            if arm!='BASE':
                assert g['key']==encode(r0['board'],r0['player']);d=dict(g['actions']);assert set(d)==set(r0['legal']) and r['best']==g['best']
                assert g['interval']==[max(l for l,_ in d.values()),max(u0 for _,u0 in d.values())]
                assert g['allowed']==[j for j,(l,u0) in g['actions'] if u0>=g['interval'][0]] and r['best'] in g['allowed']
                for j,(l,u0) in g['actions']:assert -1<=l<=(z[j]>0)-(z[j]<0)<=u0<=1;action_n+=1
                assert reg<=rb(g['actions'],r['best'])
                w=g['witness'];assert (w is None)==all(rb(g['actions'],j)>0 for j in d)
                if w is not None:
                    assert reg==0 and r['best']==w and rb(g['actions'],w)==0;cert+=1
                    value_unsolved_cert+=g['interval'][0]!=g['interval'][1]
                if r['reason']=='CERTIFIED':assert arm in e.STOP and w is not None and reg==0;stops+=1
                if r['blocks']==0 and w is None:zero_unresolved[(st,mo,arm)]+=1;zero_regret[(st,mo,arm)]+=reg
            else:assert g is None and r['best']==v['best']
            if arm=='RR_WRITER':assert h['ledger_observations']==h['temporal_blends']==h['merge_blends']==0
            elif arm=='RR_TEMPORAL':assert h['ledger_observations']==h['backup_edges'] and h['merge_blends']==0
            elif arm=='RR_MERGE':assert h['ledger_observations']==h['backup_edges'] and h['temporal_blends']==0
            else:assert h['ledger_observations']==h['backup_edges']==h['temporal_blends']==h['merge_blends']==0
            assert h['backup_changed']<=h['temporal_blends']+h['merge_blends']<=h['backup_edges']-h['terminal_edges']
            for k in ('graph_filtered','graph_width_extra','graph_solved_skips','graph_rule_checks','graph_b_skip','temporal_blends','merge_blends','backup_changed'):
                exposure[(mo,arm,k)]+=h[k]
            hist[(st,mo,arm,r['reason'])]+=1
            rr.append(dict(id=i,regret=reg,blocks=r['blocks'],evals=v['evaluator_calls'],stop=r['reason']=='CERTIFIED',cert=g is not None and g['witness'] is not None,unsolved_cert=g is not None and g['witness'] is not None and g['interval'][0]!=g['interval'][1],cap=r['reason']=='CAP'))
        results[pid,s,mo,arm]=dict(regret=sum(r['regret'] for r in rr)/4,whole_us=a['whole_us'],transport_us=a['transport_us'],blocks=a['total_blocks'],evals=sum(r['evals'] for r in rr),roots=rr,**account)
    assert len(results)==648 and 4*len(results)==su['root_decisions'] and action_n==su['root_intervals'] and cert==su['valid_certificates'] and stops==su['certified_stops']
    diagnostic={}
    for st in (*e.ST,'ALL'):
        pp=[i for i,p in pools.items() if p['group']=='TEST' and (st=='ALL' or p['stratum']==st)]
        for mo in e.MODES:
            g=su['groups'][st][mo]
            for arm,x in g['arms'].items():
                a=[results[i,s,mo,arm] for i in pp for s in e.SEEDS];rr=[r for y in a for r in y['roots']]
                for name,k in [('regret','regret'),('whole_us','whole_us'),('transport_us','transport_us'),('mean_blocks','blocks'),('mean_evals','evals')]:close(sum(y[k] for y in a)/len(a),x[name])
                for k in ('credit','extra','unused','noncert_shortfall','cert_reuse_lb','cert_reuse_ub'):assert sum(y[k] for y in a)==x[k]
                for name,k in [('stops','stop'),('certs','cert'),('unsolved_cert','unsolved_cert'),('caps','cap')]:assert sum(r[k] for r in rr)==x[name]
                assert sum(y['cert_reuse_lb']>0 for y in a)==x['pools_cert_reused']
            for k,v in g['paired'].items():
                a,b=k.split('/');diff=[sum(results[i,s,mo,b]['regret']-results[i,s,mo,a]['regret'] for s in e.SEEDS)/2 for i in pp]
                close(sum(diff)/len(diff),v['delta'])
            if mo=='WORK':
                for arm in ('RR','BOUND','RR_SAFE','RR_TEMPORAL','RR_MERGE'):
                    buckets=C.defaultdict(list)
                    for i in pp:
                        for s in e.SEEDS:
                            rr=results[i,s,mo,arm]['roots'];ref=results[i,s,mo,'STOP_FIXED']['roots']
                            for a,b in zip(rr,ref):
                                assert a['id']==b['id'];bucket='above15' if a['blocks']>15 else 'at_most15'
                                buckets[bucket].append(b['regret']-a['regret'])
                    diagnostic[st+'/'+arm]={k:dict(n=len(v),gain=float(sum(v)),mean=float(sum(v)/len(v)),better=sum(x>0 for x in v),worse=sum(x<0 for x in v)) for k,v in buckets.items()}
    dt=host['end']['mono']-host['start']['mono'];swap={k:host['end']['swap_pages'][k]-host['start']['swap_pages'][k] for k in host['start']['swap_pages']}
    close(sum(swap.values())/dt,host['interval']['swap_pages_s'])
    def flatten(x):return {'/'.join(k):float(v) if isinstance(v,F) else v for k,v in x.items()}
    return dict(source_attempts=len(seen),attempt_strata=dict(attempts),roots=len(ids),cohort=dict(counts),selfplay_events=events,pools=len(pools),smoke_pool_checks=len(sm['tests']),train_pool_checks=len(cal['timing']),cal_pool_checks=len(cal['checks']),native_pool_requests=len(results),root_decisions=4*len(results),scheduler_blocks=blocks,prefix_groups=len(prefix),prefix_equality_comparisons=pref_n,root_action_intervals=action_n,valid_certificates=cert,certified_stop=stops,certificates_with_unsolved_value=value_unsolved_cert,mirror_vectors=mirrors,worker_counts=flatten(worker),exposure=flatten(exposure),termination_counts=flatten(hist),zero_optional_uncertified=flatten(zero_unresolved),zero_optional_uncertified_regret_sum=flatten(zero_regret),posthoc_extra_work_outcome=diagnostic,swap_pages_delta=swap,host_ok=su['host_ok'],scientific_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();allowed=('src/mcts/mod.rs','src/mcts/select.rs','src/mcts/four_axis.rs','src/mcts/four_axis_math.rs','src/mcts/proof_graph.rs','src/mcts_server/four_axis_pool_plan.rs','src/mcts_server/four_axis_pool_cpu_server.rs','src/mcts_server.rs','src/mcts_server/four_axis_cpu_server.rs')
    assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in allowed for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-fouraxis04-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head and gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for p in files:assert hashlib.sha256(gg(r,'show',f'{head}:{p}')).hexdigest()==sha(ROOT/p),p
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(e.BIN)==read('C06_BUILD_IDENTITY.json')['binary_sha256']
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(clean),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,actual_restore=True,connectivity_fsck=True,binary_sha256=sha(e.BIN))
if __name__=='__main__':
    import sys
    print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2),flush=True)
