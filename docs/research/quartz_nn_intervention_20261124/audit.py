"""Independent board/trace/fit/score audit and actual incremental bundle restoration."""
from pathlib import Path
from fractions import Fraction as F
import collections as C, gzip, json, math, statistics as S, hashlib, subprocess as SP, tempfile
from publish import ROOT,B,BASE,BR,D,sha,git
from campaign import EXT,MODEL,BIN,IDENT,BIN_ID,lock
def read(n):return json.loads((D/n).read_text())
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-9,rel_tol=1e-12),(a,b)
# Independent bitboard: column-stride 7 with a sentinel bit at the top.
def won(x):return any((m:=x&(x>>d))&(m>>(2*d)) for d in (1,7,6,8))
def replay(h):
    bits=[0,0]; heights=[0]*7
    for i,ch in enumerate(h):
        c=int(ch)-1
        assert 0<=c<7 and heights[c]<6 and not any(won(x) for x in bits)
        bits[i%2]|=1<<(7*c+heights[c]);heights[c]+=1
    return bits,heights
def flat(bits):
    return [1 if bits[0]>>(7*x+y)&1 else -1 if bits[1]>>(7*x+y)&1 else 0 for y in range(6) for x in range(7)]
def canwin(bits,h,p):
    return any(won(bits[p]|(1<<(7*c+h[c]))) for c in range(7) if h[c]<6)
def canon(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def cb(b):return min(tuple(b),tuple(b[y*7+x] for y in range(6) for x in range(6,-1,-1)))
def u(x):return F(1+(x>0)-(x<0),2)
def randomkey(s,seed):
    mask=(1<<64)-1;z=seed^0xcbf29ce484222325
    for b in s.encode():z=((z^b)*0x100000001b3)&mask
    z^=z>>30;z=z*0xbf58476d1ce4e5b9&mask;z^=z>>27;z=z*0x94d049bb133111eb&mask;return z^(z>>31)
def audit():
    lock();co=read('C07_COHORT.json');m=read('C07_MODEL.json');su=read('C08_SUMMARY.json')
    for f,k in [('C07_COHORT.json','cohort_sha256'),('C07_TRAIN_RAW.json','raw_sha256'),('C07_TRAIN_LABELS.json','label_sha256')]:assert sha(D/f)==m[k]
    assert sha(D/'C08_LIVE_RAW.json')==su['raw_sha256']
    for f,k in [('C07_GENERATION.jsonl.gz','gen_sha256'),('C07_RESUME_GENERATION.jsonl.gz','resume_gen_sha256'),('C07_GENERATION_AMENDMENT.md','amendment_sha256')]:assert sha(D/f)==co[k]
    oldf=set();oldb=set()
    for f,meta in co['old_sources'].items():
        p=ROOT/'docs/research/cognitive_metareasoning_20261120'/f;assert sha(p)==meta['sha256']
        j=json.loads(p.read_text());rr=j.get('rows') or j['connect4'];assert len(rr)==meta['rows']
        for r in rr:
            h=r.get('history',r.get('sequence'));b,z=replay(h);assert flat(b)==r['board'];oldf.add(canon(h[:5]));oldb.add(cb(flat(b)))
    assert len(oldf)==co['old_prefixes'] and len(oldb)==co['old_boards']
    ids={r['id']:r for r in co['rows']};assert len(ids)==len(co['rows'])
    checked_attempts=set();newf=set();selected=set();newb=set();nevents=0;status=C.Counter();groups=C.Counter()
    for f in ['C07_GENERATION.jsonl.gz','C07_RESUME_GENERATION.jsonl.gz']:
        with gzip.open(D/f,'rt') as fp:
            for line in fp:
                g=json.loads(line);i=g['id'];fam=canon(g['opening']);assert i not in checked_attempts and fam not in newf|oldf
                checked_attempts.add(i);newf.add(fam);h=g['opening']
                for e in g['events']:
                    bs,hs=replay(h);assert not any(won(x) for x in bs)
                    assert e['ply']==len(h) and all(0<=a<7 and hs[a]<6 for a in [e['action'],e['recommendation']])
                    assert e['explore'] or e['action']==e['recommendation'];assert e['eval_calls']>=1
                    h+=str(e['action']+1);nevents+=1
                assert h==g['history'];bs,hs=replay(h)
                assert (g['status']=='TERMINAL')==(any(won(x) for x in bs) or all(x==6 for x in hs))
                status[g['status']]+=1;groups[g['group']]+=1
                if g['candidate'] is not None:
                    r=ids[i];assert i not in selected;selected.add(i)
                    q=g['candidate'];assert h.startswith(q) and r['history']==q and r['family']==fam and r['group']==g['group']
                    b,z=replay(q);assert not any(won(x) for x in b)
                    assert 20<=len(q)<=30 and len(q)%2==g['parity'] and r['ply']==len(q) and r['player']==1-2*(len(q)%2)
                    assert r['board']==flat(b) and r['legal']==[c for c in range(7) if z[c]<6] and len(r['legal'])>=2
                    assert r['immediate']==canwin(b,z,len(q)%2)
                    assert r['group']=='NATURAL' or not r['immediate']
                    bk=cb(r['board']);assert bk not in oldb|newb;newb.add(bk)
    assert selected==set(ids) and len(checked_attempts)==co['completed_attempts']==co['attempts']-co['interrupted_unrecorded_attempts']
    labs={}
    for f,group in [('C07_TRAIN_LABELS.json','TRAIN'),('C08_LABELS.json','HELDOUT')]:
        rr=read(f); expected={i for i,r in ids.items() if (r['group']=='TRAIN')==(group=='TRAIN')}
        assert {r['id'] for r in rr}==expected and len(rr)==len(expected)
        for r in rr:
            assert r['status']=='EXACT';z=r['scores'];assert len(z)==7 and all((x!=-1000)==(a in ids[r['id']]['legal']) for a,x in enumerate(z));labs[r['id']]=z
    # New independent solver calls on mirrored histories, compare action-vector reversal.
    mirror_checks=0
    for group in ['TRAIN','STRESS','NATURAL']:
        for r in sorted((r for r in ids.values() if r['group']==group),key=lambda r:r['id'])[:3]:
            h=canon(r['history']) # use actual reflection, not canonicalization
            h=r['history'].translate(str.maketrans('1234567','7654321'))
            q=SP.run([str(EXT/'c4solver'),'-a','-b',str(EXT/'7x6_small.book')],input=h+'\n',text=True,capture_output=True,cwd=EXT,timeout=8,check=True).stdout.strip().split()
            assert q[0]==h and list(map(int,q[1:]))==labs[r['id']][::-1];mirror_checks+=1
    def check(a,train=False):
        group='TRAIN' if train else a['group'];mode=a['mode'];rows=a['rows'];expect={i for i,r in ids.items() if r['group']==group};assert {r['id'] for r in rows}==expect and len(rows)==len(expect)
        assert a['threads']==2 and a['model_identity']==IDENT
        order=[r['id'] for r in rows]
        if mode=='FIXED8':order.sort()
        if mode=='RANDOM16':order.sort(key=lambda i:(randomkey(i,a['seed']),i))
        if mode=='ENTROPY16':order=[r['id'] for r in sorted(rows,key=lambda r:(-r['entropy'],r['id']))]
        if mode=='SIGNED16':order=[r['id'] for r in sorted(rows,key=lambda r:(-m['gain_cost'][r['cell']][0]/m['gain_cost'][r['cell']][1],r['id']))]
        if mode=='PREFIX':assert not a['selected']
        assert a['selected']==order[:len(a['selected'])]
        assert a['whole_us']>=a['prefix_us']+a['optional_us']-1 and a['overrun_us']==max(0,a['optional_us']-a['budget_us'])
        if not train:assert a['budget_us']==max(1000,round(len(rows)*.5*m['cost16_mean']))
        out={}
        for r in rows:
            i=r['id'];z=labs[i];root=ids[i];ss=r['steps'];b,h=replay(root['history'])
            assert r['history']==root['history'] and r['legal']==root['legal'];assert r['threat']==canwin(b,h,1-len(root['history'])%2)
            want=[4,8,16,32] if train else [4] if i not in a['selected'] else [4,8] if mode=='FIXED8' else [4,8,16]
            assert [s['visits'] for s in ss]==want
            for j,s in enumerate(ss):
                assert s['root_hash']==r['root_hash'] and s['best'] in root['legal'] and s['wall_us']>=1
                assert {v[0] for v in s['policy']}==set(root['legal'])
                assert all(0<=v[1]<=1 for v in s['policy'])
                close(sum(v for _,v in s['policy']),1)
                if j:assert s['eval_calls']>=ss[j-1]['eval_calls']
            p=[v for _,v in ss[0]['policy']];ent=-sum(v*math.log(v,7) for v in p if v);close(ent,r['entropy']);close(max(p),r['top'])
            assert r['cell']==(int(ent>=.55)|2*int(max(p)>=.75)|4*int(r['threat'])|8*int(len(p)<=4))
            vv=[u(z[s['best']]) for s in ss];opt=max(u(z[c]) for c in root['legal'])
            out[i]=(opt-vv[-1],vv[-1]-vv[0],opt-vv[0],r['cell'],vv,ss)
        return out
    tt=[check(a,True) for a in read('C07_TRAIN_RAW.json')];obs=[];ex16=set();ex=set();hist=C.Counter();hist32=C.Counter();cost8=[]
    for t in tt:
        for i,(_,_,_,cell,v,s) in t.items():
            d=int(2*(v[2]-v[0]));d32=int(2*(v[3]-v[0]));cost=s[1]['wall_us']+s[2]['wall_us'];obs.append((cell,d,cost));cost8.append(s[1]['wall_us']);hist[d]+=1;hist32[d32]+=1
            if d:ex16.add(i)
            if d or d32:ex.add(i)
    assert len(obs)==m['observations'] and len(ex)==m['exposed'] and len(ex16)==m['exposed16']
    assert {str(k):v for k,v in hist.items()}==m['histogram'] and {str(k):v for k,v in hist32.items()}==m['histogram32']
    mean=F(sum(t for _,_,t in obs),len(obs));close(mean,m['cost16_mean']);close(S.median([t for _,_,t in obs]),m['cost16_median']);close(S.median(cost8),m['cost8_median'])
    gp=[F(hist[k],len(obs)) for k in range(-2,3)]
    for c in range(16):
        sub=[(d,t) for b,d,t in obs if b==c];assert len(sub)==m['counts'][c]
        prob=[(sum(d==k for d,t in sub)+8*gp[k+2])/(len(sub)+8) for k in range(-2,3)]
        for a,b in zip(prob,m['cells'][c]):close(a,b)
        close(sum(F(k,2)*prob[k+2] for k in range(-2,3)),m['gain_cost'][c][0]);close((sum(t for d,t in sub)+8*mean)/(len(sub)+8),m['gain_cost'][c][1])
    live=read('C08_LIVE_RAW.json');re={(a['group'],a['mode'],a['seed']):(a,check(a)) for a in live};assert len(re)==20
    for group in ['STRESS','NATURAL']:
        arms=su['groups'][group]['arms'];n=len([r for r in ids.values() if r['group']==group])
        for mode in arms:
            ab=[re[group,mode,seed] for seed in (17,43)]
            for metric,col in [('regret',0),('gain',1)]:close(sum(sum(v[col] for v in d.values()) for a,d in ab)/(2*n),arms[mode][metric])
            for k in ('whole_us','transport_us','optional_us','overrun_us'):close(S.mean(a[k] for a,d in ab),arms[mode][k])
            close(S.mean(len(a['selected']) for a,d in ab),arms[mode]['selected'])
            close(abs(arms[mode]['whole_us']/arms['RANDOM16']['whole_us']-1),arms[mode]['time_mismatch_vs_random'])
        diff=[]
        for i in ids:
            if ids[i]['group']==group:diff.append(sum(re[group,'RANDOM16',s][1][i][0]-re[group,'SIGNED16',s][1][i][0] for s in (17,43))/2)
        close(sum(diff)/n,su['groups'][group]['paired_regret_improvement'])
        assert len(diff)==su['groups'][group]['paired_n']
    return dict(attempts=len(checked_attempts),interrupted=co['interrupted_unrecorded_attempts'],attempt_groups=dict(groups),attempt_status=dict(status),generation_events=nevents,cohort_groups=dict(C.Counter(r['group'] for r in ids.values())),immediate_roots=dict(C.Counter(r['group'] for r in ids.values() if r['immediate'])),mirror_oracle_vectors=mirror_checks,exact_labeled=len(labs),exposed=len(ex),exposed16=len(ex16),exposure_pass=len(ex)>=20,nonzero_trace16=sum(v for k,v in hist.items() if k),nonzero_trace32=sum(v for k,v in hist32.items() if k),occupied_cells=sum(n>0 for n in m['counts']),fit_recomputed=True,live_arms=len(re),live_root_traces=sum(len(d) for a,d in re.values()),score_recomputed=True,elapsed_gate={g:su['groups'][g]['arms']['SIGNED16']['time_mismatch_vs_random']<=.1 for g in ('STRESS','NATURAL')})
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[r['head'] for r in receipts] and all(r['head']==r['remote'] for r in receipts)
    assert all(r['parent']==p for r,p in zip(receipts,[BASE]+chain[:-1]))
    cleanup=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(cleanup)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    changed=git('diff','--name-only',BASE,head).splitlines()
    assert all(p.startswith('docs/research/quartz_nn_intervention_20261124/') or p in ('src/mcts_server.rs','src/mcts_server/intervention_cpu_server.rs') for p in changed)
    def g(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-intervention-restore-') as t:
        r=Path(t)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        g(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        g(r,'bundle','verify',str(bundle));g(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert g(r,'rev-parse','refs/heads/restored').decode().strip()==head
        assert g(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert g(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for p in changed:assert hashlib.sha256(g(r,'show',f'{head}:{p}')).hexdigest()==sha(ROOT/p),p
        g(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==BIN_ID
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(cleanup),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(changed),external_base=BASE,actual_restore=True,connectivity_fsck=True,binary_sha256=BIN_ID)
if __name__=='__main__':
    import sys
    print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),indent=2,sort_keys=True),flush=True)
