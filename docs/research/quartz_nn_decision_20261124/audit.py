"""Independent readback of frozen native results; real incremental-bundle restore."""
import collections, hashlib, json, math, random, statistics, subprocess, tempfile, sys
from fractions import Fraction
from pathlib import Path
from exact import solve_bit, solve_grid, line_win
ROOT=Path(__file__).resolve().parents[3]
D=Path(__file__).resolve().parent
B=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_nn_decision_backups')
BASE='87c15dc828d6eea519c219fd8a412fa21dfc464d'
BR='research/quartz-nn-decision-20261124'
def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def read(n): return json.loads((D/n).read_text())
def git(*a,cwd=ROOT,raw=False):
    x=subprocess.check_output(['git','-C',str(cwd),*a])
    return x if raw else x.decode().strip()
def audit():
    ss=[json.loads(l) for l in (D/'sources.jsonl').read_text().splitlines()]
    by={s['id']:s for s in ss}; assert len(ss)==len(by)==52
    groups=collections.Counter(s['split'] for s in ss); assert groups=={'TRAIN':24,'HELDOUT':24,'SMOKE':4}
    def win_grid(b,n,a,p,k):
        x,y=a%n,a//n
        for dx,dy in [(0,1),(1,0),(1,1),(1,-1)]:
            count=1
            for sign in [-1,1]:
                i,j=x+sign*dx,y+sign*dy
                while 0<=i<n and 0<=j<n and b[j*n+i]==p:
                    count+=1;i+=sign*dx;j+=sign*dy
            if count>=k:return True
        return False
    for s in ss:
        n=7 if s['game']=='gomoku7' else 15; k=4 if n==7 else 5
        b=[0]*(n*n)
        for j,a in enumerate(s['history']):
            p=(-1)**j;assert 0<=a<len(b) and b[a]==0;b[a]=p
            assert not win_grid(b,n,a,p,k)
        assert b==s['board'] and len(s['history'])==s['ply'] and s['player']==(-1)**s['ply']
        # independent canonical rotation/reflection via matrix transforms
        grid=[b[j*n:(j+1)*n] for j in range(n)]; variants=[]
        for _ in range(4):
            variants.extend(bytes(x+1 for row in g for x in row) for g in (grid,[row[::-1] for row in grid]))
            grid=[list(row) for row in zip(*grid[::-1])]
        assert hashlib.sha256(min(variants)).hexdigest()==s['id']
    m=read('C07_MODEL.json'); t=read('C07_TRAIN.json'); raw=read('C08_LIVE.json'); summ=read('C08_SUMMARY.json')
    for field,p in [('source_sha256',D/'sources.jsonl'),('model_sha256',B/'gomoku7_seed41_latest.pt'),('binary_sha256',B/'mcts_demo'),('campaign_sha256',D/'campaign.py'),('engine_sha256',D/'engine.py'),('exact_sha256',D/'exact.py'),('scheduler_sha256',D/'scheduler.py')]: assert sha(p)==m[field],field
    assert sha(B/'gomoku15_F1_latest.pt')=='57b93ba867ff8f9b74f91adef6b38f02f94de0bd7dc177785d668d32e1dcc3ae'
    assert sha(D/'C08_LIVE.json')==summ['raw_sha256']==(D/'C08_BLIND_SHA.txt').read_text().split()[0]
    labs={}; immediate=collections.Counter()
    for s in ss:
        if s['game']!='gomoku7':continue
        a=solve_bit(s['board'],s['player']); b=solve_grid(s['board'],s['player']);assert a==b
        labs[s['id']]={str(k):Fraction(v+1,2) for k,v in a.items()}
        immediate[s['split']]+=any(win_grid(s['board'][:i]+[s['player']]+s['board'][i+1:],7,i,s['player'],4) for i,x in enumerate(s['board']) if x==0)
    def check(s,r,selected):
        p,f=r['prefix'],r['final']; assert p['root_hash']==f['root_hash']
        assert p['root_visits']==8 and f['root_visits']==(24 if selected else 8)
        for row in [p,f]:
            assert s['board'][row['best_move']]==0
            assert all(s['board'][a]==0 and v>=0 for a,v in row['policy'])
            assert math.isclose(sum(v for a,v in row['policy']),1.,abs_tol=1e-6)
        assert p['evaluator_calls']==p['actual_nn_rows']>0
        if selected:
            assert f['additional_root_visits']==16
            assert f['evaluator_calls']-p['evaluator_calls']==f['additional_evaluator_calls']==f['actual_nn_rows']
        else: assert f==p
    train_ids={r['id'] for r in t}; assert train_ids==set(m['train_ids'])=={s['id'] for s in ss if s['split']=='TRAIN'}
    train_opt=0;train_rows=0;gains=[]
    for r in t:
        check(by[r['id']],r,True); u=labs[r['id']];assert r['labels']==u
        b=u[str(r['prefix']['best_move'])];f=u[str(r['final']['best_move'])]
        assert f-b==r['gain'];gains.append(f-b); train_opt+=b==max(u.values())
        train_rows+=r['final']['evaluator_calls']
    assert len(t)==24 and set(gains)=={0} and m['beta']==[0.,0.,0.,0.] and m['intercept']==0.
    assert m['median_chunk_ns']==int(statistics.median(r['final']['wall_ns'] for r in t))
    assert m['q75_chunk_ns']==sorted(r['final']['wall_ns'] for r in t)[17] and m['optional_budget_ns']==m['K']*m['q75_chunk_ns']
    assert summ['labels']=={s['id']:labs[s['id']] for s in ss if s['split']=='HELDOUT'}
    met={}; improved={}; finals={}
    for arm in raw:
        name=arm['arm']; rr=arm['rows']; assert {r['id'] for r in rr}==set(summ['labels']) and len(rr)==24
        selected={r['id'] for r in rr if r['selected']}
        assert len(selected)==arm['selected_count']<=m['K']
        if name=='SIGNED':assert arm['order']==sorted(summ['labels'])
        if name=='ENTROPY':assert arm['order']==[r['id'] for r in sorted(rr,key=lambda r:(-r['x'][0],r['id']))]
        if name=='RANDOM':
            order=[r['id'] for r in rr];random.Random(20261124081).shuffle(order);assert arm['order']==order
        if name=='PREFIX_ONLY':assert not selected and not arm['order']
        assert selected==set(arm['order'][:len(selected)])
        ui=[];uf=[];reg=[];improved[name]=[];finals[name]={}; nn=0
        for r in rr:
            s=by[r['id']];check(s,r,r['selected']);u=labs[r['id']]
            b=u[str(r['prefix']['best_move'])];f=u[str(r['final']['best_move'])]
            assert r['predicted_gain']==0
            ui.append(b);uf.append(f);reg.append(max(u.values())-f);finals[name][r['id']]=f
            if f>b:improved[name].append(r['id'])
            nn+=r['final']['evaluator_calls']
        q={'initial':float(sum(ui)/24),'utility':float(sum(uf)/24),'regret':float(sum(reg)/24),'gain':float((sum(uf)-sum(ui))/24),'improved':sum(f>b for f,b in zip(uf,ui)),'degraded':sum(f<b for f,b in zip(uf,ui)),'nn_rows':nn,'native_evals':nn,'selected':len(selected)}
        for k,v in q.items():assert summ['metrics'][name][k]==v,(name,k,v)
        met[name]=q
    times={a['arm']:a['total_ns'] for a in raw}
    mismatch={name:abs(times[name]/times['RANDOM']-1) for name in ('SIGNED','ENTROPY')}
    assert all(math.isclose(v,summ['time_mismatch_relative'][k],abs_tol=1e-12) for k,v in mismatch.items())
    assert summ['time_gate']==all(v<=.1 for v in mismatch.values())==False
    smoke=read('C08_GOMOKU15_SMOKE.json');assert len(smoke)==4
    for r in smoke:check(by[r['id']],r,True)
    return {'source_counts':dict(groups),'legal_history_and_D4_verified':len(ss),'dual_solver_agreement':len(labs),'immediate_win_roots':dict(immediate),'all_winning_roots':{split:sum(max(labs[s['id']].values())==1 for s in ss if s['split']==split) for split in ('TRAIN','HELDOUT')},'train_prefix_optimal':train_opt,'train_nonzero_gain':sum(x!=0 for x in gains),'train_nn_rows':train_rows,'metrics':met,'improved_ids':improved,'time_mismatch_relative':mismatch,'time_gate':False,'gomoku15_roots':len(smoke),'gomoku15_nn_rows':sum(r['final']['evaluator_calls'] for r in smoke),'gomoku15_nn_batches':sum(r['prefix']['actual_nn_batches']+r['final']['actual_nn_batches'] for r in smoke),'disposition':'NOT_ADMITTED: no train gain variation and actual time mismatch'}
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[r['head'] for r in receipts]
    assert all(r['head']==r['remote'] for r in receipts)
    cleanup=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(cleanup)==len(receipts)-1
    assert [p.name for p in B.glob('*.bundle')]==[bundle.name]
    changed=git('diff','--name-only',BASE,head).splitlines()
    assert all(p.startswith('docs/research/quartz_nn_decision_20261124/') for p in changed)
    with tempfile.TemporaryDirectory(prefix='quartz-nn-restore-') as tmp:
        r=Path(tmp)/'restore.git';subprocess.run(['git','init','--bare','-q',str(r)],check=True)
        git('fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE,cwd=r)
        assert not (r/'objects/info/alternates').exists()
        assert subprocess.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        git('bundle','verify',str(bundle),cwd=r)
        git('fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored',cwd=r)
        assert git('rev-parse','refs/heads/restored',cwd=r)==head
        assert git('rev-parse','refs/heads/restored^{tree}',cwd=r)==rec['tree']
        assert git('rev-list','--reverse',f'{BASE}..{head}',cwd=r).splitlines()==chain
        for p in changed:assert hashlib.sha256(git('show',f'{head}:{p}',cwd=r,raw=True)).hexdigest()==sha(ROOT/p),p
        git('fsck','--connectivity-only','--no-dangling',cwd=r)
    return {'head':head,'tree':rec['tree'],'remote':rec['remote'],'stage_count':len(receipts),'cleanup_count':len(cleanup),'bundle':bundle.name,'sha256':rec['sha256'],'bytes':rec['bytes'],'restored_files':len(changed),'external_base':BASE,'actual_restore':True,'connectivity_fsck':True,'assets':{p.name:sha(p) for p in [B/'gomoku7_seed41_latest.pt',B/'gomoku15_F1_latest.pt',B/'mcts_demo']}}
if __name__=='__main__':
    print(json.dumps({'audit':audit(),'restore':restore()},indent=2,sort_keys=True),flush=True)
