"""Resume after the bounded interrupted TRAIN generation, without relabeling or rerolling it."""
import collections,gzip,json,random
from campaign import D,Server,board,won,immediate,canon,bkey,old_sources,sha,write,lock
def row(g,h):
    b,z=board(h);p=1-2*(len(h)%2);legal=[c for c in range(7) if z[c]<6]
    assert not won(b,-p) and len(legal)>=2 and 20<=len(h)<=30 and len(h)%2==g['parity']
    assert g['group']=='NATURAL' or not immediate(b,z,p)
    return dict(id=g['id'],family=canon(g['opening']),group=g['group'],history=h,board=b,ply=len(h),player=p,legal=legal,immediate=immediate(b,z,p))
def main():
    lock();oldp,oldb,meta=old_sources();usedp=set(oldp);usedb=set(oldb);rows=[]
    original=D/'C07_GENERATION.jsonl.gz'
    with gzip.open(original,'rt') as f: attempts=[json.loads(x) for x in f]
    assert len(attempts)==1184 and all(g['group']=='TRAIN' for g in attempts)
    for g in attempts:
        k=canon(g['opening']);assert k not in usedp;usedp.add(k)
        if g['candidate'] is not None:
            r=row(g,g['candidate']);k=bkey(r['board']);assert k not in usedb;usedb.add(k);rows.append(r)
    assert len(rows)==104
    rng=random.Random(2026112404);path=D/'C07_RESUME_GENERATION.jsonl.gz';assert not path.exists()
    total=1216
    with Server() as s,gzip.open(path,'wt',compresslevel=6) as log:
        for group in ('STRESS','NATURAL'):
            count=[0,0];n=0
            while n<292 and sum(count)<48:
                batch=[]
                for _ in range(min(32,292-n)):
                    while True:
                        h=''.join(str(rng.randrange(1,8)) for _ in range(5));k=canon(h)
                        if k not in usedp:usedp.add(k);break
                    i=total;total+=1;n+=1
                    batch.append(dict(id=f'{group}_{i:04d}',group=group,opening=h,history=h,parity=i%2,events=[],candidate=None,status='RUNNING'))
                for ply in range(5,31):
                    active=[]
                    for g in batch:
                        if g['status']!='RUNNING':continue
                        h=g['history'];assert len(h)==ply
                        b,z=board(h);p=1-2*(ply%2);legal=[c for c in range(7) if z[c]<6]
                        if won(b,-p) or not legal:g['status']='TERMINAL';continue
                        if 20<=ply<=30 and ply%2==g['parity'] and g['candidate'] is None and len(legal)>=2 and (group=='NATURAL' or not immediate(b,z,p)):
                            if bkey(b) not in usedb and count[ply%2]<24:
                                r=row(g,h);usedb.add(bkey(b));count[ply%2]+=1;rows.append(r);g['candidate']=h
                        if ply==30:g['status']='DEPTH_CAP';continue
                        active.append(g)
                    if not active:break
                    a=s.call('GEN',active,seed=17)
                    for g,r in zip(active,a['rows']):
                        recommendation=r['steps'][0]['best'];explore=rng.random()<.45
                        move=rng.choice(r['legal']) if explore else recommendation
                        g['events'].append(dict(ply=ply,recommendation=recommendation,action=move,explore=explore,eval_calls=r['steps'][0]['eval_calls'],root_hash=r['root_hash']))
                        g['history']+=str(move+1)
                for g in batch:log.write(json.dumps(g,separators=(',',':'))+'\n')
                print('generation resumed',group,count,'attempts',total,flush=True)
    write('C07_COHORT.json',dict(rows=rows,attempts=total,completed_attempts=total-32,interrupted_unrecorded_attempts=32,targets={'TRAIN':144,'STRESS':48,'NATURAL':48},old_prefixes=len(oldp),old_boards=len(oldb),old_sources=meta,gen_sha256=sha(original),resume_gen_sha256=sha(path),amendment_sha256=sha(D/'C07_GENERATION_AMENDMENT.md')))
    print('COHORT',collections.Counter((r['group'],r['player']) for r in rows),flush=True)
if __name__=='__main__':main()
