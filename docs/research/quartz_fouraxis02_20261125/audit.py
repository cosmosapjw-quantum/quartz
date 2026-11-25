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
    e.lock();co=read('C07_COHORT.json');su=read('C08_SUMMARY.json');raw=e.gzread('C08_LIVE_RAW.json.gz');sm=read('C06_SMOKE.json')
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
    # Actual diagnostic selected leaves reconstructed from recorded path and prior TRAIN root.
    dr=read('C06_DIAGNOSTIC_RAW.json');ds=read('C07_LEAF_DIAGNOSIS.json');dl={x['id']:x['scores'] for x in read('C07_LEAF_LABELS.json') if x['status']=='EXACT'}
    old=ROOT.parent/'quartz_nn_multiquanta_20261124/docs/research/quartz_nn_multiquanta_20261124/C07_COHORT.json'
    assert sha(old)==dr['source_identity'] and sha(D/'C06_DIAGNOSTIC_RAW.json')==ds['raw_sha']
    tr={x['source']:x['trace'] for x in dr['traces']};source={x['id']:x for x in json.loads(old.read_text())['rows'] if x['group']=='TRAIN'}
    diag_by={x['id']:x for x in ds['rows']};db=set();diag_data=[]
    sign=lambda x:int(x>0)-int(x<0)
    for x in dr['rows']:
        h=x['history'];p0=source[x['source']]['history'];assert h.startswith(p0)
        path=[int(j)-1 for j in h[len(p0):]];assert path in tr[x['source']]['prefix_hooks']['leaf_paths']
        b,z=replay(h);p=1-2*(len(h)%2);assert not win(b,p) and not win(b,-p) and x['board']==b
        assert cb(b) not in db|boards;db.add(cb(b))
        a=x['diag'];assert a['history']==h and {j['move'] for j in a['rows']}=={c for c in range(7) if z[c]<6}
        assert a['nn_calls']==1+sum(not j['terminal'] for j in a['rows'])
        assert [j['move'] for j in a['rows']]==[j['move'] for j in sorted(a['rows'],key=lambda j:(-j['prior'],j['move']))]
        vals=[]
        for j in a['rows']:
            c=j['move'];c0=b.copy();c0[z[c]*7+c]=p;term=win(c0,p) or all(v!=0 for v in c0)
            assert term==j['terminal']
            if term:assert j['h']==(1 if win(c0,p) else 0)
            if x['id'] in dl:
                t=sign(dl[x['id']][c]);assert not term or t==j['h'];vals.append((t,F(str(j['h'])),term))
        if x['id'] not in dl:continue
        T=max(v[0] for v in vals);Tk=max(v[0] for v in vals[:2]);Hk=max(v[1] for v in vals[:2]);Ha=max(v[1] for v in vals);v=F(str(a['base']))
        lo=max(t if term else -1 for t,_,term in vals);hi=max(t if term else 1 for t,_,term in vals);assert lo<=T<=hi
        proof=lo if lo==hi else None;gate=abs(v)<=F(65,100) and len(vals)>=2
        pred={'base':v,'legacy':(v+Hk)/2 if gate else v,'proof':F(proof) if proof is not None else v,'top2':F(proof) if proof is not None else (v+Hk)/2 if gate else v,'full':F(proof) if proof is not None else (v+Ha)/2 if gate else v}
        d=diag_by[x['id']];assert d['T']==T and d['omission']==Tk-T and d['proof']==proof and d['gate']==gate
        close(d['max_error'],Hk-Tk);close(d['full_max_error'],Ha-T)
        for k,y in pred.items():close(d['pred'][k],y);close(d['errors'][k],y-T)
        diag_data.append((x['stratum'],T,pred))
    for st,v in ds['summary'].items():
        a=[x for x in diag_data if st=='ALL' or x[0]==st];assert len(a)==v['n']
        for k in a[0][2]:close(sum(abs(p[k]-t) for _,t,p in a)/len(a),v['mae'][k]);close(sum((p[k]-t)**2 for _,t,p in a)/len(a),v['mse'][k])
    labs={r['id']:r['scores'] for r in read('C08_LABELS.json') if r['status']=='EXACT'};assert set(labs)==set(ids)
    for i,z in labs.items():assert len(z)==7 and all((v!=-1000)==(j in ids[i]['legal']) for j,v in enumerate(z))
    mirrors=0
    for st in e.ST:
        for r in sorted((x for x in ids.values() if x['stratum']==st),key=lambda x:x['id'])[:3]:
            h=r['history'].translate(str.maketrans('1234567','7654321'))
            z=SP.run([str(n.EXT/'c4solver'),'-a','-b',str(n.EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=n.EXT,timeout=15,check=True).stdout.split()
            assert z[0]==h and list(map(int,z[1:]))==labs[r['id']][::-1];mirrors+=1
    by={};prefix={};pref_n=0;exposure=C.Counter();act=C.Counter();caps=C.Counter();workers=C.Counter()
    for a in raw['runs']:
        i=a['id'];r=ids[i];v=a['result'];p=a['prefix'];h=a['hooks'];ph=a['prefix_hooks'];arm=a['arm'];mo=a['mode'];s=a['seed'];st=r['stratum']
        assert a['history']==r['history'] and a['stratum']==st and a['model_identity']==n.IDENT
        assert (i,s,mo,arm) not in by
        assert p['root_n']==8 and v['root_hash']==p['root_hash'] and v['root_n']==8+8*a['blocks']<=512
        k=(i,s,arm=='BASE')
        if k in prefix:assert prefix[k]==p,(k,mo,arm);pref_n+=1
        else:prefix[k]=p
        for x in (p,v):
            assert {j[0] for j in x['candidates']}==set(r['legal']) and x['best'] in r['legal']
            assert [j['move'] for j in x['edges']]==[j[0] for j in x['candidates'][:len(x['edges'])]]
            assert sum(j['n'] for j in x['edges'])==x['root_n'];close(sum(j['w'] for j in x['edges']),x['root_w'])
            for j in x['edges']:assert j['vl']==0 and abs(j['vvalue'])<1e-6 and abs(j['w'])<=j['n']+1e-6
        assert ph['base_evals']+ph['terminal_leaves']==8 and p['evaluator_calls']==1+ph['base_evals']
        assert h['base_evals']+h['terminal_leaves']==v['root_n']-8 and v['evaluator_calls']==p['evaluator_calls']+h['base_evals']+h['extra_evals']
        if arm=='WRITER':assert h['backup_edges']>=v['root_n']-8 and h['ledger_observations']==0
        if arm in ('TEMPORAL','MERGE','COMBINED','JOINT'):assert h['ledger_observations']==h['backup_edges'] and h['backup_edges']>=v['root_n']-8
        if arm not in ('TEMPORAL','COMBINED'):assert h['temporal_blends']==0
        if arm not in ('MERGE','COMBINED','JOINT'):assert h['merge_blends']==0
        if arm not in ('PROOF_V','TOP2_V','FULL_V','JOINT'):assert h['proofs']==0
        else:assert h['proof_scans']==h['base_evals'] and h['proof_win']<=h['proofs']<=h['proof_scans']
        assert h['backup_changed']<=h['temporal_blends']+h['merge_blends']<=h['backup_edges']-h['terminal_edges']
        assert h['extra_evals']==0 or arm in ('LEGACY_V','TOP2_V','FULL_V','JOINT')
        if mo=='VISITS':assert v['root_n']==64
        elif mo=='NN':assert v['evaluator_calls']>=48 or v['root_n']==512
        else:assert a['before_drop_us']>=2500 or v['root_n']==512
        for k in ('proofs','proof_win','merge_blends','temporal_blends','backup_changed','extra_evals','select_changed','width_mass_added'):
            exposure[st+'/'+mo+'/'+arm+'/'+k]+=h[k];act[st+'/'+mo+'/'+arm+'/'+k]+=h[k]>0
        caps[st+'/'+mo]+=a['cause']=='CAP';workers[st+'/'+mo]+=h['worker_mask'].bit_count()>=2
        z=labs[i];val=u(z[v['best']]);opt=max(u(j) for j in z if j!=-1000);by[i,s,mo,arm]=(opt-val,a)
    assert len(by)==48*2*3*11
    summary_checks=0
    for st,g in su['groups'].items():
        rr=[i for i,r in ids.items() if r['stratum']==st];assert len(rr)==24
        for mo,v in g.items():
            for arm,x in v['arms'].items():
                a=[by[i,s,mo,arm] for i in rr for s in e.SEEDS];runs=[j[1] for j in a]
                close(sum(j[0] for j in a)/len(a),x['regret'])
                for name in ('whole_us','transport_us'):close(sum(j[name] for j in runs)/len(runs),x[name])
                close(sum(j['result']['evaluator_calls'] for j in runs)/len(runs),x['evals'])
                close(sum(j['result']['root_n'] for j in runs)/len(runs),x['visits'])
                assert sum(j['cause']=='CAP' for j in runs)==x['caps']
                assert sum(j['hooks']['worker_mask'].bit_count()>=2 for j in runs)==x['workers_two']
                for k,val in x['hooks'].items():assert sum(j['hooks'][k] for j in runs)==val
                summary_checks+=1
            for key,x in v['paired'].items():
                a,b=key.split('/');d=[sum(by[i,s,mo,b][0]-by[i,s,mo,a][0] for s in e.SEEDS)/2 for i in rr]
                close(sum(d)/len(d),x['delta']);close(abs(v['arms'][a]['whole_us']/v['arms'][b]['whole_us']-1),x['time_mismatch']);close(abs(v['arms'][a]['evals']/v['arms'][b]['evals']-1),x['nn_mismatch'])
    assert all(p['standard']['result']==p['research']['result'] for p in sm['parity'])
    dt=raw['end']['mono']-raw['start']['mono'];swap=sum(raw['end']['swap_pages'][k]-raw['start']['swap_pages'][k] for k in raw['start']['swap_pages'])/dt
    close(swap,su['host_interval']['swap_pages_s'])
    return dict(attempts=len(seen),attempt_strata=dict(attempts),roots=len(ids),cohort=dict(counts),selfplay_events=events,diagnostic_leaves=len(dr['rows']),diagnostic_exact=len(dl),diagnostic_independent_error_and_certificate=True,exact_labels=len(labs),mirror_vectors=mirrors,all_native_runs=len(by),prefix_equality_comparisons=pref_n,prefix_groups=len(prefix),summary_groups_verified=summary_checks,exposure=dict(exposure),runs_with_exposure=dict(act),cap_counts=dict(caps),two_actual_worker_runs=dict(workers),parity=len(sm['parity']),deep_merge_blends=sum(a['hooks']['merge_blends'] for a in sm['deep']),host_ok=su['host_ok'],scientific_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD');assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines();allowed=('src/mcts/mod.rs','src/mcts/select.rs','src/mcts/four_axis.rs','src/mcts/four_axis_math.rs','src/mcts_server.rs','src/mcts_server/four_axis_cpu_server.rs')
    assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in allowed for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-fouraxis02-restore-') as tmp:
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
