"""Independent scalar recursive PUCT on a deterministic bounded replay sample.
Shares only the frozen evaluator interface, not the tree/selection/backup code."""
import collections, fractions, gzip, importlib.util, math
from plan import D,load,save,sha
from engine import load_eval, Puct
spec=importlib.util.spec_from_file_location('k33_grid_audit',D.parent/'quartz_kinetic33_20261126/audit_k33.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
F=fractions.Fraction
def ref(ev,h,cap,lock):
    board,w=grid.replay(h);assert not w
    p=1-2*(len(h)%2);assert grid.cert(board,p)['move'] is None
    tree={};start=ev.rows;term=0
    def expand(b,p,path):
        legal=[i for i,v in enumerate(b) if not v]
        if not legal:return 0.
        a=sum(1<<i for i,v in enumerate(b) if v==1);c=sum(1<<i for i,v in enumerate(b) if v==-1)
        pri,v=ev(a,c,p,legal)
        tree[path]={m:[0,0.,float(q)] for m,q in zip(legal,pri)}
        return v
    def descend(b,p,path):
        nonlocal term
        if path not in tree:
            if path and grid.won(b,path[-1],-p):
                term+=1;return -1.
            if not any(v==0 for v in b):
                term+=1;return 0.
            return expand(b,p,path)
        edges=tree[path];s=math.sqrt(max(1,sum(v[0] for v in edges.values())))
        def score(m):
            n,w,pr=edges[m]
            return ((w/n if n else 0.)+2.*pr*s/(n+1),-m)
        m=max(edges,key=score);b[m]=p
        v=-descend(b,-p,path+(m,));b[m]=0
        edges[m][0]+=1;edges[m][1]+=v
        return v
    expand(board,p,());t=0;edges=tree[()]
    while t<cap:
        n={m:r[0] for m,r in edges.items()};a=max(n,key=lambda i:(n[i],-i))
        if lock and all(n[a]>n[i]+cap-t or (n[a]==n[i]+cap-t and a<i) for i in n if i!=a):break
        descend(board,p,());t+=1
    votes=[[m,r[0]] for m,r in edges.items()]
    a=max(edges,key=lambda i:(edges[i][0],-i))
    assert ev.rows-start==1+t-term
    return dict(move=a,votes=votes,visits=t,nn=ev.rows-start,terminal_leaves=term)
def run():
    lock=load('GAMES_LOCK.json');summary=load('SUMMARY.json')['arms']
    g=[]
    for fi in lock['files']:
        assert sha(D/fi['file'])==fi['sha256']
        with gzip.open(D/fi['file'],'rt') as f:
            import json
            g.extend(json.loads(l) for l in f)
    g.sort(key=lambda x:(x['job']['seed'],x['job']['id'],x['arm'],x['candidate_side']))
    sample=collections.defaultdict(dict);counts={};results={}
    for arm in ('L32','L20'):
        gg=[x for x in g if x['arm']==arm];c=collections.Counter()
        for x in gg:
            c['score2']+=1+x['candidate_side']*x['winner'];c['games']+=1
            c['wins']+=x['score']==1;c['losses']+=x['score']==0
            h=x['job']['opening'].copy()
            for r in x['rows']:
                tag='L' if r['player']==x['candidate_side'] else 'B'
                c[tag+'_moves']+=1;c[tag+'_nn']+=r['nn']
                c[tag+'_cpu_ns']+=r['cpu_ns'];c[tag+'_wall_ns']+=r['wall_ns']
                if r['kind']!='CERTIFICATE':
                    sample[x['job']['seed']].setdefault(tuple(h),[]).append(r)
                h.append(r['move'])
        a=summary[arm];assert (c['wins'],c['losses'],c['games'])==(a['wins'],a['losses'],a['games'])
        sc=F(c['score2'],2*c['games']);assert abs(float(sc)-a['score'])<1e-14
        for k,field in [('nn','NN_saving'),('cpu_ns','CPU_saving'),('wall_ns','wall_saving')]:
            saving=1-F(c['L_'+k]*c['B_moves'],c['B_'+k]*c['L_moves'])
            assert abs(float(saving)-a[field])<1e-14
        results[arm]={'exact_score':str(sc),'exact_nn_saving':str(1-F(c['L_nn']*c['B_moves'],c['B_nn']*c['L_moves'])),'counts':dict(c)}
    nroot=0;records=0;checks=0;nnrows=0;lock20_saving=0;changes20=0;terminal=0
    for seed,hs in sorted(sample.items()):
        ev=load_eval(seed)
        for h in list(hs)[:12]:
            nroot+=1;v={}
            for cap,lock in [(32,False),(32,True),(20,False),(20,True)]:
                got=ref(ev,list(h),cap,lock)
                original=Puct(ev).search(list(h),cap,lock)
                assert all(original[k]==got[k] for k in got),(seed,h,cap,lock,got,original)
                for r in hs[h]:
                    if (r['cap'],r['lock'])==(cap,lock):
                        assert all(r[k]==got[k] for k in got),(seed,h,cap,lock)
                        records+=1
                v[cap,lock]=got;checks+=1;terminal+=got['terminal_leaves']
            for cap in (20,32):
                assert v[cap,False]['move']==v[cap,True]['move']
                assert v[cap,True]['nn']<=v[cap,False]['nn']
            lock20_saving+=v[20,False]['nn']-v[20,True]['nn']
            changes20+=v[20,True]['move']!=v[32,False]['move']
        nnrows+=ev.rows
    out=dict(scope='deterministic first 12 unique unresolved recorded histories per checkpoint, not random or inferential; scalar tree recursion/selection/backup independently reimplemented with shared actual NN evaluator; exact raw aggregates independently reconstructed; bootstrap CIs not independently recomputed',roots=nroot,scalar_vs_vector=checks,recorded_replay_comparisons=records,actual_NN_rows_in_review=nnrows,terminal_leaves_in_scalar=terminal,lock20_extra_rows_saved=lock20_saving,cap20_vs32_action_changes=changes20,aggregates=results,limitations=['common exact root gate shared by all arms','reference serial Python PUCT, not production Rust','fixed checkpoint panel; 103 performance warning','native subagent unavailable'])
    assert nroot==36 and checks==144
    save('REVIEW.json',out);print(out,flush=True)
if __name__=='__main__':run()
