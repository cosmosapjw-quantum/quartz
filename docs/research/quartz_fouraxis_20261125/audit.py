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
def audit():
    e.lock();co=read('C07_COHORT.json');sm=read('C06_SMOKE.json');su=read('C08_SUMMARY.json');raw=e.gzread('C08_LIVE_RAW.json.gz')
    assert sha(D/'C07_ATTEMPTS.jsonl.gz')==co['attempt_sha'] and sha(D/'C08_LIVE_RAW.json.gz')==su['raw_sha']
    oldp,oldb,meta=exclusions();assert len(oldp)==co['old_families'] and len(oldb)==co['old_boards'] and meta==co['sources']
    ids={r['id']:r for r in co['rows']};seen=set();families=set();boards=set();accepted=set();events=0;counts=C.Counter();attempts=C.Counter()
    with gzip.open(D/'C07_ATTEMPTS.jsonl.gz','rt') as fp:
        for line in fp:
            a=json.loads(line);i=a['id'];h=a['history'];op=a['opening'];f=canon(op)
            assert i not in seen and f not in oldp|families and h.startswith(op);seen.add(i);families.add(f);attempts[a['stratum']]+=1
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
            assert r['family']==f and r['player']==p and r['ply']==len(h) and r['stratum']==a['stratum'] and len(h)%2==a['parity'] and r['group']=='TEST'
            assert r['immediate']==immediate(b,z,p)
            assert (18<=len(h)<=25 and not r['immediate']) if r['stratum']=='TACTICAL' else (16<=len(h)<=26)
            k=cb(b);assert k not in oldb|boards;boards.add(k);counts[r['stratum']+'/'+str(p)]+=1
    assert accepted==set(ids) and len(seen)==co['attempts']
    labs={r['id']:r['scores'] for r in read('C08_LABELS.json') if r['status']=='EXACT'}
    assert set(labs)==set(ids)
    for i,z in labs.items():assert len(z)==7 and all((v!=-1000)==(j in ids[i]['legal']) for j,v in enumerate(z))
    mirrors=0
    for st in e.ST:
        for r in sorted((x for x in ids.values() if x['stratum']==st),key=lambda x:x['id'])[:3]:
            h=r['history'].translate(str.maketrans('1234567','7654321'))
            z=SP.run([str(n.EXT/'c4solver'),'-a','-b',str(n.EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=n.EXT,timeout=15,check=True).stdout.split()
            assert z[0]==h and list(map(int,z[1:]))==labs[r['id']][::-1];mirrors+=1
    # Independent known transposition witness using two legal histories.
    with e.Server() as srv:
        h1,h2='1234','3214';assert replay(h1)==replay(h2)
        a=srv.call(h1,8,threads=1);b=srv.call(h2,8,threads=1)
        assert a['result']['root_hash']==b['result']['root_hash']
        assert a['result']==b['result']
        witness_hash=a['result']['root_hash']
    by={};exposure=C.Counter();activations=C.Counter();hooks={}
    for a in raw['runs']:
        i=a['id'];r=ids[i];v=a['result'];ini=a['initial'];h=a['hooks'];mask=a['mask'];s=a['seed']
        assert a['history']==r['history'] and a['stratum']==r['stratum'] and a['threads']==2 and a['visits']==64 and not a['standard'] and a['model_identity']==n.IDENT
        assert (i,s,mask) not in by;assert v['root_hash']==ini['root_hash'] and v['root_n']==64 and ini['root_n']==0
        assert {x[0] for x in v['candidates']}==set(r['legal']);assert [x['move'] for x in v['edges']]==[x[0] for x in v['candidates'][:len(v['edges'])]]
        assert v['best'] in r['legal'] and sum(x['n'] for x in v['edges'])==64
        close(sum(x['w'] for x in v['edges']),v['root_w'])
        for x in v['edges']:
            assert x['vl']==0 and abs(x['vvalue'])<1e-6 and abs(x['w'])<=x['n']+1e-6
            if mask&8:assert -1e-7<=x['m2']<=x['n']+1e-6
        assert h['base_evals']+h['terminal_leaves']==64 and v['evaluator_calls']==1+h['base_evals']+h['extra_evals']
        assert h['extra_evals']+h['exact_child_evals']==2*h['refinements']
        assert h['select_changed']<=h['select_nodes'] and h['select_deep']<=h['select_nodes']
        assert h['width_mass_added']<=h['width_nodes'] and h['width_limited']<=h['width_nodes']
        assert h['backup_changed']<=h['merge_blends']+h['temporal_blends']<=h['backup_edges']-h['terminal_edges']
        assert h['ledger_contexts']>=h['ledger_nodes']>=h['ledger_shared_children']
        if not mask&1:assert h['select_nodes']==0
        if not mask&2:assert h['width_nodes']==0
        if not mask&4:assert h['refinements']==0
        if not mask&8:assert h['backup_edges']==0
        else:assert h['backup_edges']==h['ledger_observations']>=64
        for k in ('select_changed','select_deep','width_mass_added','width_limited','refinements','extra_evals','backup_changed','merge_blends','temporal_blends','ledger_shared_children'):
            exposure[r['stratum']+'/'+k]+=h[k];activations[r['stratum']+'/'+k]+=h[k]>0
        if mask==15:
            for k in ('select_changed','select_deep','width_mass_added','width_limited','refinements','extra_evals','backup_changed','merge_blends','temporal_blends','ledger_shared_children'):
                hooks.setdefault(r['stratum'],C.Counter())[k]+=h[k]
        z=labs[i];val=u(z[v['best']]);opt=max(u(x) for x in z if x!=-1000);by[i,s,mask]=(opt-val,val,F(a['whole_us']),F(v['evaluator_calls']),F(a['search_us']))
    assert len(by)==64*2*16
    attribution_checks=0;report={}
    for st in e.ST:
        rr=[i for i,r in ids.items() if r['stratum']==st];v=su['groups'][st]
        assert len(rr)==32
        for mask in range(16):
            x=[by[i,s,mask] for i in rr for s in e.SEEDS]
            for name,j in [('regret',0),('utility',1),('whole_us',2),('evals',3),('search_us',4)]:close(sum(a[j] for a in x)/len(x),v['arms'][str(mask)][name])
        for name,col in [('utility',1),('whole_us',2),('evals',3)]:
            f=[sum(by[i,s,a][col] for i in rr for s in e.SEEDS)/(len(rr)*2) for a in range(16)]
            # Independent inclusion inversion via recursive subtraction, and
            # Shapley via all 24 ordered insertions, not Mobius coefficients.
            m=[]
            for a in range(16):m.append(f[a]-sum(m[j] for j in range(a) if j&a==j))
            p=[F(0)]*4
            for order in permutations(range(4)):
                a=0
                for j in order:
                    b=a|(1<<j);p[j]+=f[b]-f[a];a=b
            p=[x/24 for x in p]
            assert sum(p)==f[15]-f[0] and sum(m[1:])==f[15]-f[0]
            for a,b in zip(m,v['attribution'][name]['mobius']):close(a,b)
            for a,b in zip(p,v['attribution'][name]['shapley']):close(a,b)
            attribution_checks+=1
            if name=='utility':report[st]=dict(full_delta=str(f[15]-f[0]),shapley_exact=[str(x) for x in p],mobius_exact=[str(x) for x in m])
        ds=[sum(by[i,s,0][0]-by[i,s,15][0] for s in e.SEEDS)/2 for i in rr]
        close(sum(ds)/len(ds),v['full_vs_baseline']['improvement'])
        close(abs(v['arms']['15']['whole_us']/v['arms']['0']['whole_us']-1),v['full_vs_baseline']['time_mismatch'])
        close(v['arms']['15']['evals']/v['arms']['0']['evals'],v['full_vs_baseline']['eval_ratio'])
    assert all(p['standard']['result']==p['research']['result'] for p in sm['parity'])
    dt=raw['end']['mono']-raw['start']['mono'];swap=sum(raw['end']['swap_pages'][k]-raw['start']['swap_pages'][k] for k in raw['start']['swap_pages'])/dt
    close(swap,su['host_interval']['swap_pages_s'])
    assert (swap<=512 and max(x['host_load1'] for x in raw['runs'])<=12 and max(raw['start']['load'][0],raw['end']['load'][0])<=12)==su['host_ok']
    return dict(attempts=len(seen),attempt_strata=dict(attempts),roots=len(ids),cohort=dict(counts),selfplay_events=events,exact_labels=len(labs),mirror_vectors=mirrors,known_transposition=dict(histories=[h1,h2],root_hash=witness_hash),native_runs=len(by),all_root_visits=64,actual_axis_exposure=dict(exposure),runs_with_axis_events=dict(activations),full_mask_hooks={k:dict(v) for k,v in hooks.items()},factorial_checks=attribution_checks,exact_utility_attribution=report,deep_merge_blends=sum(a['hooks']['merge_blends'] for a in sm['deep']),parity=len(sm['parity']),host_ok=su['host_ok'],scientific_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();allowed=('src/mcts/mod.rs','src/mcts/select.rs','src/mcts/four_axis.rs','src/mcts/four_axis_math.rs','src/mcts_server.rs','src/mcts_server/four_axis_cpu_server.rs')
    assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in allowed for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-fouraxis-restore-') as tmp:
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
