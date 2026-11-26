"""Independent grid rule and paired game audit; no imported proof/arena."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
import gzip,hashlib,json,random,statistics as S,math
D=Path(__file__).resolve().parent
def load(n):return json.loads((D/n).read_text())
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def raw(p):
    with gzip.open(p,'rt') as fd:
        for l in fd:yield json.loads(l)
def won(b,m,p):
    r,c=divmod(m,7);assert b[m]==p
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        n=1
        for sg in (-1,1):
            i,j=r+sg*dx,c+sg*dy
            while 0<=i<7 and 0<=j<7 and b[7*i+j]==p:
                n+=1;i+=sg*dx;j+=sg*dy
        if n>=4:return True
    return False
def replay(h):
    b=[0]*49;winner=0
    for i,m in enumerate(h):
        assert not winner and 0<=m<49 and b[m]==0
        p=1-2*(i%2);b[m]=p
        if won(b,m,p):winner=p
    return b,winner
def wins(b,p):
    z=[]
    for m in range(49):
        if b[m]:continue
        b[m]=p
        if won(b,m,p):z.append(m)
        b[m]=0
    return z
def cert(b,p):
    rows=[];empty=[i for i,v in enumerate(b) if not v]
    for m in empty:
        b[m]=p
        if won(b,m,p):rows.append([m,1,1,'WIN1',[]])
        elif len(empty)==1:rows.append([m,0,0,'DRAW',[]])
        else:
            op=wins(b,-p)
            if op:rows.append([m,-1,-1,'LOSE2',op])
            else:
                ow=wins(b,p);z=len(ow)>=2
                rows.append([m,1 if z else -1,1,'WIN3' if z else 'UNKNOWN',ow])
        b[m]=0
    # Independently choose from certified minimax comparisons.
    lo=max(z[1] for z in rows);hi=max(z[2] for z in rows)
    allowed=[x[0] for x in rows if x[2]>=lo]
    w=[x for x in rows if all(y[0]==x[0] or x[1]>=y[2] for y in rows)]
    chosen=min(w,key=lambda x:(-x[1],x[3]!='WIN1',abs(x[0]//7-3)+abs(x[0]%7-3),x[0])) if w else None
    return dict(move=chosen[0] if chosen else None,kind=chosen[3] if chosen else 'UNRESOLVED',allowed=allowed,interval=[lo,hi],bounds=rows)
def audit_cert(b,p,z,ct):
    v=cert(b,p)
    for k,x in v.items():assert z[k]==x,(k,z[k],x,b,p)
    ct['proofs']+=1;ct['actions']+=len(z['bounds'])
    # Verify all opponent responses for each claimed WIN3, not only sampled reply.
    for m,l,u,k,threat in z['bounds']:
        ct[k]+=1
        if k!='WIN3':continue
        b[m]=p
        for r,x in enumerate(b):
            if x:continue
            b[r]=-p;assert not won(b,r,-p)
            # A certified winning square must remain and win.
            y=[q for q in threat if not b[q]];assert y
            q=y[0];b[q]=p;assert won(b,q,p);b[q]=0;b[r]=0;ct['win3_replies']+=1
        b[m]=0
    # Independently reconstruct scheduled line scans (every mask evaluated).
    cnt=0
    lines=[]
    for i in range(49):
        x,y=divmod(i,7)
        for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
            if 0<=x+3*dx<7 and 0<=y+3*dy<7:lines.append(tuple(7*(x+j*dx)+y+j*dy for j in range(4)))
    for m,l,u,k,threat in z['bounds']:
        cnt+=sum(m in t for t in lines)
        if k not in ('WIN1','DRAW'):cnt+=len(lines)
        if k not in ('WIN1','DRAW','LOSE2'):cnt+=len(lines)
    assert z['rule_scans']==cnt;ct['rule_scans']+=cnt
def validate_move(b,p,a,arm,ct):
    assert 0<=a['move']<49 and b[a['move']]==0
    assert isinstance(a['NN'],int) and a['NN']>=0 and a['physical']==sum(a['batches'])
    assert a['wall_ns']>=a['rule_ns']>=0
    if arm.startswith('proof'):
        assert a['proof'] is not None;audit_cert(b,p,a['proof'],ct)
    else:assert a['proof'] is None
    if a['kind']=='CERTIFICATE':
        assert a['proof']['move']==a['move'] and a['NN']==a['physical']==0 and a['budget'] is None and a['manifest'] is None
        ct['certified']+=1
    else:
        assert a['kind']=='NATIVE'
        assert a['proof'] is None or a['proof']['move'] is None
        q=a['budget'];m=a['manifest'];cap=20 if arm.endswith('20') else 32
        assert m['benchmark_safe'] and m['profile']=='quartz' and m['requested_iteration_limit']==cap and q['requested_iteration_limit']==cap
        assert a['NN']==0 or a['physical']>0
        assert q['evaluator_calls']==a['NN']
        ct['native']+=1;ct['overshoot_iters']+=max(0,q['realized_iterations']-cap)
        ct['logical_physical_difference']+=a['physical']-a['NN'];ct['unequal_native']+=a['physical']!=a['NN']
    ct['NN']+=a['NN'];ct['physical']+=a['physical'];ct['moves']+=1
def main():
    plan=load('PLAN.json');lk=load('GAMES_LOCK.json');assert sha(D/'PLAN.json')==lk['plan'] and sha(D/'arena.py')==lk['code'] and sha(D/'proof.py')==lk['proof']
    jobs={(j['seed'],j['id']):j for j in plan['jobs']};assert len(jobs)==36 and len({tuple(j['family']) for j in jobs.values()})==36
    assert sha(Path(plan['binary']['path']))==plan['binary']['sha256']
    for m in plan['models'].values():assert sha(Path(m['path']))==m['sha256']
    counters=Counter();pairs=defaultdict(list);seen=set()
    for fi in lk['files']:
        path=D/fi['file'];assert sha(path)==fi['sha256'];ct=Counter()
        for g in raw(path):
            j=g['job'];arm=g['arm'];side=g['candidate_side'];assert j==jobs[(j['seed'],j['id'])]
            k=(j['seed'],j['id'],arm,side);assert k not in seen;seen.add(k)
            b,w=replay(j['opening']);assert not w;hh=list(j['opening'])
            for a in g['rows']:
                assert not w and a['ply']==len(hh);p=1-2*(len(hh)%2);who=arm if p==side else 'base32'
                assert a['player']==p and a['arm']==who and a['seed']==(j['game_seed']+len(hh)*0x9e3779b97f4a7c15)&((1<<64)-1)
                validate_move(b,p,a,who,ct)
                m=a['move'];b[m]=p;hh.append(m)
                if won(b,m,p):w=p
            assert hh==g['history'] and g['winner']==w and (w or all(b))
            assert g['score']==(1+side*w)/2
            pairs[arm].append(g);ct['games']+=1
        for k in ('games','moves','NN','physical'):assert ct[k]==fi[k]
        counters.update(ct)
    assert len(seen)==216 and all(counters[k]==lk[k] for k in ('games','moves','NN','physical'))
    summary=load('SUMMARY.json')
    # Use exact rational per-pair sufficient statistics; preserve stored shard order.
    for arm,gg in pairs.items():
        groups=defaultdict(list)
        for g in gg:groups[(g['job']['seed'],g['job']['id'])].append(g)
        data={}
        for k,gs in groups.items():
            assert len(gs)==2 and {g['candidate_side'] for g in gs}=={-1,1}
            c=[0]*6
            for g in gs:
                for a in g['rows']:
                    offset=0 if a['player']==g['candidate_side'] else 3
                    c[offset]+=1;c[offset+1]+=a['NN'];c[offset+2]+=a['physical']
            data[k]=(sum(F(g['score']) for g in gs)/2,c)
        strata=[[k for k in groups if k[0]==seed] for seed in sorted({k[0] for k in groups})]
        rng=random.Random(plan['seed']+321);scores=[];nn=[];phys=[]
        for _ in range(4000):
            ids=[k for g in strata for k in rng.choices(g,k=len(g))]
            scores.append(sum(data[k][0] for k in ids)/len(ids))
            c=[sum(data[k][1][j] for k in ids) for j in range(6)]
            nn.append(1-F(c[1]*c[3],c[0]*c[4]))
            phys.append(1-F(c[2]*c[3],c[0]*c[5]))
        v=summary['arms'][arm]
        for x,key in [(scores,'score_ci'),(nn,'NN_reduction_ci'),(phys,'physical_reduction_ci')]:
            a=sorted(x);assert all(abs(float(z)-q)<1e-12 for z,q in zip((a[99],a[3899]),v[key]))
        assert abs(float(sorted(scores)[199])-v['NI_lower'])<1e-12
        assert v['NI_point_gate']==(v['NI_lower']>=0.45)

    for arm,gg in pairs.items():
        v=summary['arms'][arm];assert len(gg)==72 and len({(x['job']['seed'],x['job']['id']) for x in gg})==36
        assert sum(x['score']==1 for x in gg)==v['wins'] and sum(x['score']==.5 for x in gg)==v['draws'] and sum(x['score']==0 for x in gg)==v['losses']
        assert S.mean(x['score'] for x in gg)==v['score']
        c=Counter()
        for x in gg:
            for a in x['rows']:
                who='candidate' if a['player']==x['candidate_side'] else 'base'
                for k in ('NN','physical','wall_ns','rule_ns'):c[who+'_'+k]+=a[k]
                c[who+'_moves']+=1;c[who+'_cert']+=a['kind']=='CERTIFICATE'
        for k,x in c.items():assert x==v['count'][k]
        for k,key in [('NN','NN_reduction'),('physical','physical_reduction')]:
            x=1-(c['candidate_'+k]/c['candidate_moves'])/(c['base_'+k]/c['base_moves'])
            assert abs(x-v[key])<1e-12
    l=load('SAME_POSITION_LOCK.json');assert sha(D/'SAME_POSITION.jsonl.gz')==l['sha256'] and l['games']==sha(D/'GAMES_LOCK.json')
    ct=Counter();n=0;diag_seen=set();ref={ (g['job']['seed'],g['job']['id']):g for g in pairs['proof32'] if g['candidate_side']==1 };diag_cost=defaultdict(Counter)
    for x in raw(D/'SAME_POSITION.jsonl.gz'):
        j=(x['model'],x['id']);k=(*j,x['ply']);assert k not in diag_seen and x['ply'] in (4,10);diag_seen.add(k)
        g=ref[j];assert len(g['history'])>x['ply'] and x['history']==g['history'][:x['ply']]
        assert x['seed']==(g['job']['game_seed']+x['ply']*0x9e3779b97f4a7c15)&((1<<64)-1)
        assert set(x['results'])=={'base32','proof32','blind20','proof20'}
        b,w=replay(x['history']);assert not w and len(x['history'])==x['ply'];p=1-2*(x['ply']%2)
        for arm,a in x['results'].items():
            validate_move(b,p,a,arm,ct)
            c=diag_cost[arm];c['NN']+=a['NN'];c['physical']+=a['physical'];c['wall_ns']+=a['wall_ns'];c['count']+=1;c['certificates']+=a['kind']=='CERTIFICATE'

        n+=1
    missing={(*k,ply) for k,g in ref.items() for ply in (4,10) if len(g['history'])<=ply}
    expected={(*k,ply) for k,g in ref.items() for ply in (4,10)}-missing
    assert diag_seen==expected and missing=={tuple(k) for k in l['missing']}
    assert n==l['n'] and n+len(l['missing'])==72
    for arm,c in diag_cost.items():
        v=l['summary'][arm]
        for k,key in [('NN','NN'),('physical','physical'),('wall_ns','wall_ns'),('count','n'),('certificates','cert')]:assert c[k]==v[key]
        for k in ('NN','physical'):
            assert abs(float(1-F(c[k],diag_cost['base32'][k]))-v[k+'_reduction'])<1e-12
    out=dict(game=dict(counters),diagnostic=dict(ct),n=216,scope='Independent grid directional rules, all WIN3 opponent replies with remaining winning square, complete per-move accounting and outcomes; no model inference rerun')
    (D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
