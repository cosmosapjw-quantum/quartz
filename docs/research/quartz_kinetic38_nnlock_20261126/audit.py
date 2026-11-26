"""Independent grid legality, terminal, common gate, vote and paired-replay checks."""
import sys,json,gzip,pathlib,collections
from plan import D,load,sha,save,family
sys.path.insert(0,str(D.parent/'quartz_kinetic33_20261126'))
from audit_k33 import replay,cert
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def run():
    lk=load('GAMES_LOCK.json');pl=load('PLAN.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['engine']==sha(D/'engine.py') and lk['arena']==sha(D/'arena.py')
    jobs={(j['seed'],j['id']):j for j in pl['jobs']};assert len(jobs)==len({family(j['opening']) for j in jobs.values()})
    prior=load('PLAN.json')['prior_sha256'];assert sha(D.parent/'quartz_kinetic33_20261126/PLAN.json')==prior
    old=json.loads((D.parent/'quartz_kinetic33_20261126/PLAN.json').read_text());assert not ({family(j['opening']) for j in old['jobs']}&{family(j['opening']) for j in jobs.values()})
    ct=collections.Counter();pairs=collections.defaultdict(list);seen=set();cache={}
    for fi in lk['files']:
        assert sha(D/fi['file'])==fi['sha256']
        fc=collections.Counter()
        for g in raw(D/fi['file']):
            j=g['job'];k=(j['seed'],j['id']);assert j==jobs[k]
            arm=g['arm'];side=g['candidate_side'];key=(k,arm,side);assert key not in seen;seen.add(key)
            pairs[k,arm].append(g);h=list(j['opening']);assert g['history'][:len(h)]==h
            for r in g['rows']:
                b,w=replay(h);p=1-2*(len(h)%2);assert not w and r['ply']==len(h) and r['player']==p and b[r['move']]==0
                who=arm if p==side else 'B32';assert r['arm']==who and r['cap']==(20 if who=='L20' else 32) and r['lock']==(who!='B32')
                ck=(tuple(b),p)
                if ck not in cache:cache[ck]=cert(b,p)
                z=cache[ck];ct['rule_checks']+=1
                if r['kind']=='CERTIFICATE':
                    assert z['move']==r['move'] and z['kind']==r['proof'] and r['nn']==r['visits']==0 and not r['votes'];ct['certificates']+=1
                else:
                    assert z['move'] is None and r['proof']=='UNRESOLVED'
                    v=r['votes'];legal=[i for i,x in enumerate(b) if x==0];assert [a for a,n in v]==legal
                    n=dict(v);L=min(legal,key=lambda a:(-n[a],a));assert r['move']==L
                    d=min((n[L]-n[a]+int(a>L) for a in legal if a!=L),default=10**9)
                    assert r['distance']==d and sum(n.values())==r['visits'] and 0<=r['visits']<=r['cap']
                    assert r['nn']==1+r['visits']-r['terminal_leaves'] and 0<=r['terminal_leaves']<=r['visits']
                    if r['kind']=='VOTE_LOCK':assert r['lock'] and r['visits']<r['cap'] and d>r['cap']-r['visits'];ct['locks']+=1
                    else:assert r['kind']=='CAP' and r['visits']==r['cap']
                    ct['legal_edges']+=len(v)
                assert r['cpu_ns']>=0 and r['wall_ns']>=r['rule_ns']>=0
                h.append(r['move']);fc.update(moves=1,nn=r['nn'])
            b,w=replay(h);assert h==g['history'] and (w or all(b)) and g['winner']==w and g['score']==(1+side*w)/2
            fc['games']+=1
        assert all(fc[k]==fi[k] for k in ('games','moves','nn'));ct.update(fc)
    assert all(ct[k]==lk[k] for k in ('games','moves','nn'))
    for (k,arm),gg in pairs.items():
        assert len(gg)==2 and {g['candidate_side'] for g in gg}=={-1,1}
        if arm=='L32':
            assert gg[0]['history']==gg[1]['history'];ct['exact_same_horizon_pairs']+=1
            for a,b in zip(gg[0]['rows'],gg[1]['rows']):
                assert a['move']==b['move']
                x,y=(a,b) if a['arm']=='L32' else (b,a)
                assert x['nn']<=y['nn'] and x['visits']<=y['visits'];ct['exact_same_horizon_moves']+=1
    assert len(seen)==len(jobs)*4
    save('AUDIT.json',dict(counts=dict(ct),unique_rule_positions=len(cache),scope='independent grid certificate and terminal replay, all recorded moves and paired history; no second independent neural forward rerun'))
    print('AUDIT',dict(ct),flush=True)
if __name__=='__main__':run()
