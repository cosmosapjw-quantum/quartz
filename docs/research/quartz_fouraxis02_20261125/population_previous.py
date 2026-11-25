"""New prospective source generation, adapted from locked NN03 generator."""
from native import *
TARGETS={st:{'TEST':32} for st in ('TACTICAL','SELFPLAY')}
CAPS={'TEST':400}
OLD=ROOT.parent/'quartz_nn_multiquanta_20261124/docs/research/quartz_nn_multiquanta_20261124'
from native import exclusions as prev_exclusions
def exclusions():
    oldp,oldb,meta=prev_exclusions()
    for n in ('C07_ATTEMPTS.jsonl.gz','C07_COHORT.json'):
        p=OLD/n;ref='3d22dadcade29f8cb96b44e7ef40571c4d5c1db3:docs/research/quartz_nn_multiquanta_20261124/'+n
        obj=subprocess.check_output(['git','hash-object',str(p)],cwd=ROOT,text=True).strip()
        assert obj==subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()
        meta['NN03/'+n]={'sha256':sha(p)}
        if n.endswith('.gz'):
            with gzip.open(p,'rt') as f:
                for line in f:oldp.add(canon(json.loads(line)['opening']))
        else:
            for r in json.loads(p.read_text())['rows']:oldb.add(cb(r['board']))
    p4=ROOT.parent/'quartz_nn_costcal_20261125/docs/research/quartz_nn_costcal_20261125'
    for name in ('C07_ATTEMPTS.jsonl.gz','C07_COHORT.json'):
        p=p4/name;ref='d302bd0008c21a7f7bfd848a0d964d3f94662d78:docs/research/quartz_nn_costcal_20261125/'+name
        assert subprocess.check_output(['git','hash-object',str(p)],cwd=ROOT,text=True).strip()==subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()
        meta['NN04/'+name]={'sha256':sha(p)}
        if name.endswith('.gz'):
            with gzip.open(p,'rt') as f:
                for line in f:oldp.add(canon(json.loads(line)['opening']))
        else:
            for r in json.loads(p.read_text())['rows']:oldb.add(cb(r['board']))
    return oldp,oldb,meta
def generate():
    prior_lock();before=host();oldp,oldb,meta=exclusions();usedp=set(oldp);usedb=set(oldb);rows=[]
    rng=random.Random(202611250505);serial=0
    path=D/'C07_ATTEMPTS.jsonl.gz';assert not path.exists()
    def new(group,stratum,parity):
        nonlocal serial
        while True:
            h=''.join(str(rng.randrange(1,8)) for _ in range(5));c=canon(h)
            if c not in usedp:usedp.add(c);break
        serial+=1
        return dict(id=f'{stratum}_{group}_{serial:05d}',opening=h,history=h,group=group,stratum=stratum,parity=parity,events=[],candidate=None)
    def accept(g,h,quota,count,immediate_allowed):
        r=make_row(g,h);p=r['ply']%2;b=cb(r['board'])
        if p!=g['parity'] or len(r['legal'])<2 or count[p]>=quota//2 or b in usedb or (not immediate_allowed and r['immediate']):return False
        usedb.add(b);count[p]+=1;rows.append(r);g['candidate']=h;return True
    with gzip.open(path,'wt',compresslevel=6) as log:
        for group,quota in TARGETS['TACTICAL'].items():
            count=[0,0];n=0
            while sum(count)<quota and n<4000:
                n+=1;p=0 if count[0]<count[1] else 1
                if count[p]>=quota//2:p=1-p
                g=new(group,'TACTICAL',p);h=g['history'];b,z=replay(h)
                target=rng.choice([x for x in range(18,26) if x%2==p])
                for i in range(5,target):
                    opts=[c for c in range(7) if z[c]<6 and not won(b[i%2]|(1<<(7*c+z[c])))]
                    if not opts:break
                    c=rng.choice(opts);b[i%2]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
                g['history']=h;g['target']=target
                if len(h)==target:accept(g,h,quota,count,False)
                g['status']='ACCEPT' if g['candidate'] else 'REJECT'
                log.write(json.dumps(g,separators=(',',':'))+'\n')
            print('TACTICAL',group,count,'attempts',n,flush=True)
        # Real previous trained CNN and parallel resident engine; only source generation.
        with PriorServer() as srv:
            for group,quota in TARGETS['SELFPLAY'].items():
                count=[0,0];n=0
                while sum(count)<quota and n<CAPS[group]:
                    batch=[new(group,'SELFPLAY',(n+i)%2) for i in range(min(8,CAPS[group]-n))]
                    n+=len(batch)
                    for ply in range(5,27):
                        active=[]
                        for g in batch:
                            if g.get('status'):continue
                            h=g['history'];assert len(h)==ply;b,z=replay(h)
                            if any(won(x) for x in b) or all(t==6 for t in z):g['status']='TERMINAL';continue
                            if ply>=16 and ply%2==g['parity'] and accept(g,h,quota,count,True):g['status']='ACCEPT';continue
                            if ply==26:g['status']='CAP';continue
                            active.append(g)
                        if not active:break
                        a=srv.call('GEN',active,seed=17)
                        for g,r in zip(active,a['rows']):
                            rec=r['steps'][0]['best'];explore=rng.random()<.45;move=rng.choice(r['legal']) if explore else rec
                            g['events'].append(dict(ply=ply,action=move,recommendation=rec,explore=explore,eval_calls=r['steps'][0]['eval_calls'],root_hash=r['root_hash']))
                            g['history']+=str(move+1)
                    for g in batch:log.write(json.dumps(g,separators=(',',':'))+'\n')
                print('SELFPLAY',group,count,'attempts',n,flush=True)
    write('C07_COHORT.json',dict(rows=rows,targets=TARGETS,caps=CAPS,attempts=serial,old_families=len(oldp),old_boards=len(oldb),sources=meta,attempt_sha=sha(path),host_start=before,host_end=host()))
    print('COHORT',dict(C.Counter(f"{r['stratum']}/{r['group']}/{r['player']}" for r in rows)),flush=True)
if __name__=="__main__":generate()
