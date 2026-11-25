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
def graph_check(run):
    g=run['proof'];nodes={v['key']:v for v in g['graph']};assert len(nodes)==g['nodes']
    b,z=replay(run['history']);p=1-2*(len(run['history'])%2);root=encode(b,p);assert root==g['key']
    ordered=sorted(nodes,key=lambda k:-sum(x!=0 for x in decode(k)[0]))
    ref={};arcs=0;terminal_count=0;shared=C.Counter()
    for k in ordered:
        v=nodes[k];b,p,z=decode(k);term=outcome(b,p)
        if v['terminal'] is not None:
            assert v['terminal']==term and v['actions'] is None;iv=(term,term);terminal_count+=1
        elif v['actions'] is None:
            assert term is None;iv=(-1,1)
        else:
            assert term is None and len(v['actions'])==len({m for m,_ in v['actions']})
            assert {m for m,_ in v['actions']}=={i for i in range(7) if z[i]<6}
            aa=[]
            for m,c in v['actions']:
                cb0=b.copy();cb0[z[m]*7+m]=p
                assert c==encode(cb0,-p) and c in ref
                l,u0=ref[c];aa.append((-u0,-l));arcs+=1;shared[c]+=1
            iv=(max(x[0] for x in aa),max(x[1] for x in aa))
        assert list(iv)==v['interval'];ref[k]=iv
    reachable={root};todo=[root]
    while todo:
        k=todo.pop()
        for _,c in nodes[k]['actions'] or []:
            if c not in reachable:reachable.add(c);todo.append(c)
    assert reachable==set(nodes)
    assert g['interval']==list(ref[root])
    aa=[[m,[-ref[c][1],-ref[c][0]]] for m,c in nodes[root]['actions']]
    assert aa==g['actions']
    assert g['solved']==sum(l==u0 for l,u0 in ref.values())
    assert g['registered']==sum(v['terminal'] is not None or v['actions'] is not None for v in nodes.values())
    return dict(nodes=len(nodes),arcs=arcs,terminal=terminal_count,shared_nodes=sum(v>1 for v in shared.values()),solved=g['solved'])
def audit():
    e.lock()
    co=read('C07_COHORT.json');raw=e.gzread('C08_LIVE_RAW.json.gz');sm=e.gzread('C06_SMOKE.json.gz');diag=e.gzread('C07_DIAG_RAW.json.gz');cost=read('C07_COST_LOCK.json');bd=read('C07_B_DIAGNOSIS.json');su=read('C08_SUMMARY.json')
    assert sha(D/'C07_ATTEMPTS.jsonl.gz')==co['attempt_sha'] and sha(D/'C08_LIVE_RAW.json.gz')==su['raw_sha']
    assert cost['cohort_sha']==sha(D/'C07_COHORT.json') and cost['diagnostic_raw_sha']==sha(D/'C07_DIAG_RAW.json.gz')==bd['raw_sha'] and cost['binary_sha']==sha(e.BIN)
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
    assert accepted==set(ids) and len(seen)==co['attempts'] and len(ids)==144
    assert all(counts[st+'/'+g+'/'+str(p)]==(12 if g=='TEST' else 6) for st in e.ST for g in ('TRAIN','CAL','TEST') for p in (-1,1))
    assert all(p['standard']['result']==p['research']['result'] for p in sm['parity'])
    # Verify actual proof graphs independently by legal gravity and backward topological reduction.
    graph_tot=C.Counter()
    for a in sm['graphs']+diag['graphs']:graph_tot.update(graph_check(a))
    assert sm['stop']['cause']=='CERTIFIED' and sm['stop']['result']['root_n']==8 and sm['stop']['best']==0
    for st in e.ST:
        rr=[a for a in diag['timing'] if a['stratum']==st]
        assert len(rr)==24 and {a['id'] for a in rr}=={r['id'] for r in ids.values() if r['stratum']==st and r['group']=='TRAIN'}
        v=cost['cost'][st];med=S.median(a['whole_us'] for a in rr);drop=S.median(a['drop_us'] for a in rr)
        assert v['median_whole_us']==med and v['median_drop_us']==drop and v['target']==min(10000,max(2000,math.ceil(med))) and v['reserve']==min(v['target']-1,math.ceil(drop)+100)
    # Reconstruct B observation provenance and labels; exact rational moment audit.
    bq=read('C07_B_QUERIES.json');bl=read('C07_B_LABELS.json');assert sha(D/'C07_B_QUERIES.json')==bd['query_sha']
    labels={r['id']:r['scores'] for r in bl if r['status']=='EXACT'};qq={r['id']:r for r in bq['queries']};assert set(labels)==set(qq)
    raw_events={(a['id'],a['arm'],x['child_key']):x for a in diag['btrace'] for x in a['hooks']['b_events']}
    btrain=set();bcal=set();used=set();by=C.defaultdict(list);mse_ci={}
    for r in bq['rows']:
        root=ids[r['source']];h=r['history'];ev=r['event']
        assert r['stratum']==root['stratum'] and root['group']==r['group'] in ('TRAIN','CAL')
        assert h==root['history']+''.join(str(x+1) for x in ev['path'])
        b,z=replay(h);p=1-2*(len(h)%2);pb,pz=replay(h[:-1])
        assert outcome(b,p) is None
        assert encode(b,p)==ev['child_key']==r['query'] and encode(pb,-p)==ev['parent_key']
        assert fnv(b,p)==ev['child_hash'] and fnv(pb,-p)==ev['parent_hash']
        assert ev==raw_events[r['source'],r['arm'],ev['child_key']] and ev['other']==(r['arm']=='MERGE')
        can=cb(b);assert can not in boards|oldb
        k=(r['source'],r['arm'],can);assert k not in used;used.add(k)
        (btrain if r['group']=='TRAIN' else bcal).add(can)
        q=qq[r['query']];qb,qz=replay(q['history']);assert qb==b and q['legal']==[i for i,v in enumerate(z) if v<6]
        sc=labels[r['query']];assert all((v!=-1000)==(i in q['legal']) for i,v in enumerate(sc))
        theta=max((v>0)-(v<0) for v in sc if v!=-1000)
        y=F(str(ev['y']))-theta;m=F(str(ev['m']))-theta;a=min(F(1,4),F(ev['n'],ev['n']+4))
        by[r['arm'],r['group'],r['source']].append((y*y,m*m,y*m,((1-a)*y+a*m)**2,(F(3,4)*y+F(1,4)*m)**2))
    assert not btrain&bcal and len(bq['rows'])==971 and len(labels)==965
    import random
    rng=random.Random(202611250709)
    for arm in e.BMODES:
        gs=bd['results'][arm];mse_ci[arm]={}
        for group in ('TRAIN','CAL'):
            fam=[f for (a,g,_),f in by.items() if a==arm and g==group];v=gs['groups'][group]
            assert len(fam)==v['families'] and sum(len(f) for f in fam)==v['events']
            s,t,r,actual,m025=[sum(sum(x[j] for x in f)/len(f) for f in fam)/len(fam) for j in range(5)]
            for k,x in [('s',s),('t',t),('r',r),('mse_fixed025',m025)]:close(v[k],x)
            if group=='TRAIN':
                d=s+t-2*r;eta=min(F(1,4),max(F(0),(s-r)/d)) if d else F(0)
                close(gs['eta'],eta)
            delta=[float(sum(x[0]-x[4] for x in f)/len(f)) for f in fam]
            boot=sorted(S.mean(rng.choices(delta,k=len(delta))) for _ in range(4000))
            mse_ci[arm][group]=dict(families=len(fam),events=v['events'],mse=float(s),mse_025=float(m025),mse_actual_eta=float(actual),gain=float(s-m025),ci95=[boot[100],boot[3899]])
    # Independent score readback, root certificate checks and resource accounting.
    test={i:r for i,r in ids.items() if r['group']=='TEST'}
    labs={r['id']:r['scores'] for r in read('C08_LABELS.json') if r['status']=='EXACT'};assert set(labs)==set(test)
    for i,z in labs.items():assert len(z)==7 and all((v!=-1000)==(j in test[i]['legal']) for j,v in enumerate(z))
    mirrors=0
    for st in e.ST:
        for r in sorted((x for x in test.values() if x['stratum']==st),key=lambda x:x['id'])[:3]:
            h=r['history'].translate(str.maketrans('1234567','7654321'))
            z=SP.run([str(n.EXT/'c4solver'),'-a','-b',str(n.EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=n.EXT,timeout=15,check=True).stdout.split()
            assert z[0]==h and list(map(int,z[1:]))==labs[r['id']][::-1];mirrors+=1
    results={};prefix={};pref_n=0;cert=0;action_n=0;stops=0;value_unsolved_cert=0;hist=C.Counter();expo=C.Counter()
    for a in raw['runs']:
        i=a['id'];r=test[i];v=a['result'];p=a['prefix'];h=a['hooks'];ph=a['prefix_hooks'];arm=a['arm'];mo=a['mode'];s=a['seed'];st=r['stratum']
        assert a['history']==r['history'] and a['stratum']==st and a['model_identity']==n.IDENT and (i,s,mo,arm) not in results
        assert p['root_n']==8 and v['root_hash']==p['root_hash'] and v['root_n']==8+8*a['blocks']<=512
        k=(i,s,arm=='BASE')
        if k in prefix:assert prefix[k]==p;pref_n+=1
        else:prefix[k]=p
        for x in (p,v):
            assert {j[0] for j in x['candidates']}==set(r['legal']) and x['best'] in r['legal']
            assert [j['move'] for j in x['edges']]==[j[0] for j in x['candidates'][:len(x['edges'])]]
            assert sum(j['n'] for j in x['edges'])==x['root_n'];close(sum(j['w'] for j in x['edges']),x['root_w'])
            for j in x['edges']:assert j['vl']==0 and abs(j['vvalue'])<1e-6 and abs(j['w'])<=j['n']+1e-6
        assert ph['base_evals']+ph['terminal_leaves']==8 and p['evaluator_calls']==1+ph['base_evals']
        assert h['base_evals']+h['terminal_leaves']+h['graph_solved_skips']==v['root_n']-8 and v['evaluator_calls']==p['evaluator_calls']+h['base_evals']+h['extra_evals']
        z=labs[i];opt=max(u(j) for j in z if j!=-1000);reg=opt-u(z[a['best']]);g=a['proof']
        if arm in e.GRAPH:
            assert g['key']==encode(r['board'],r['player']);d=dict(g['actions'])
            assert set(d)==set(r['legal']) and a['best']==g['best']
            assert g['interval']==[max(l for l,_ in d.values()),max(u0 for _,u0 in d.values())]
            assert g['allowed']==[j for j,(l,u0) in g['actions'] if u0>=g['interval'][0]]
            assert a['best'] in g['allowed']
            for j,(l,u0) in g['actions']:assert -1<=l<=(z[j]>0)-(z[j]<0)<=u0<=1;action_n+=1
            assert reg<=rb(g['actions'],a['best'])
            w=g['witness'];assert (w is None)==all(rb(g['actions'],j)>0 for j in d)
            if w is not None:
                assert reg==0 and a['best']==w and rb(g['actions'],w)==0;cert+=1
                value_unsolved_cert+=g['interval'][0]!=g['interval'][1]
            if a['cause']=='CERTIFIED':assert arm=='ACTIVE_STOP' and w is not None;stops+=1
        else:assert g is None and a['best']==v['best']
        if mo=='TIME':
            cfg=cost['cost'][st];assert a['target_us']==cfg['target'] and a['reserve_us']==cfg['reserve']
            assert a['before_drop_us']>=cfg['target']-cfg['reserve'] or v['root_n']==512 or a['cause']=='CERTIFIED'
        else:assert v['root_n']==128 or a['cause']=='CERTIFIED'
        hist[st+'/'+mo+'/'+a['cause']]+=1
        for k in ('graph_filtered','graph_width_extra','graph_solved_skips','graph_rule_checks'):expo[k]+=h[k]
        if g is not None:expo['graph_shared_notices']+=g['shared_notices']
        results[i,s,mo,arm]=(reg,a)
    assert len(results)==72*2*2*8 and action_n==su['checked_root_action_intervals'] and cert==su['certificates_valid']
    for st in e.ST:
        rr=[i for i,r in test.items() if r['stratum']==st]
        for mo in e.MODES:
            v=su['groups'][st][mo]
            for arm,x in v['arms'].items():
                a=[results[i,s,mo,arm] for i in rr for s in e.SEEDS];runs=[j[1] for j in a];gg=[j for j in runs if j['proof'] is not None]
                close(sum(j[0] for j in a)/len(a),x['regret'])
                for k in ('whole_us','transport_us'):close(sum(j[k] for j in runs)/len(runs),x[k])
                close(sum(j['result']['evaluator_calls'] for j in runs)/len(runs),x['evals']);close(sum(j['result']['root_n'] for j in runs)/len(runs),x['visits'])
                assert sum(j['cause']=='CERTIFIED' for j in runs)==x['stops'] and sum(j['proof']['witness'] is not None for j in gg)==x['certificates']
                if gg:close(sum(min(rb(j['proof']['actions'],m) for m,_ in j['proof']['actions']) for j in gg)/len(gg),x['mean_best_bound'])
            for key,x in v['paired'].items():
                a,b=key.split('/');diffs=[sum(results[i,s,mo,b][0]-results[i,s,mo,a][0] for s in e.SEEDS)/2 for i in rr]
                close(sum(diffs)/len(diffs),x['delta']);close(abs(v['arms'][a]['whole_us']/v['arms'][b]['whole_us']-1),x['time_mismatch'])
    dt=raw['end']['mono']-raw['start']['mono'];swap={k:raw['end']['swap_pages'][k]-raw['start']['swap_pages'][k] for k in raw['start']['swap_pages']};close(sum(swap.values())/dt,su['host_interval']['swap_pages_s'])
    return dict(source_attempts=len(seen),attempt_strata=dict(attempts),roots=len(ids),cohort=dict(counts),selfplay_events=events,full_diagnostic_graphs=len(sm['graphs']+diag['graphs']),graph_totals=dict(graph_tot),known_transposition_key_equal=sm['graphs'][2]['proof']['key']==sm['graphs'][3]['proof']['key'],b_events=len(bq['rows']),b_exact=len(labels),b_split_disjoint=True,b_mse_independent=mse_ci,test_exact=len(labs),mirror_vectors=mirrors,native_runs=len(results),prefix_groups=len(prefix),prefix_equality_comparisons=pref_n,root_action_intervals=action_n,valid_certificates=cert,certified_stop=stops,certificates_with_unsolved_value=value_unsolved_cert,exposure=dict(expo),termination_counts=dict(hist),swap_pages_delta=swap,host_ok=su['host_ok'],scientific_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();allowed=('src/mcts/mod.rs','src/mcts/select.rs','src/mcts/four_axis.rs','src/mcts/four_axis_math.rs','src/mcts/proof_graph.rs','src/mcts_server.rs','src/mcts_server/four_axis_cpu_server.rs')
    assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in allowed for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-fouraxis03-restore-') as tmp:
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
