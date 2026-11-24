import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

D=Path(__file__).resolve().parent
LINES=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
EMPTY=(0,)*9
def outcome(b):
    for i,j,k in LINES:
        if b[i] and b[i]==b[j]==b[k]: return b[i]
    return 0 if 0 not in b else None
def apply(b,i,p):
    assert b[i]==0 and outcome(b) is None
    return b[:i]+(p,)+b[i+1:]
def replay(hist):
    b=EMPTY; p=1
    for i in hist:
        b=apply(b,i,p); p=-p
    return b,p
def canonical(b):
    a=[]
    for swap in (False,True):
        for fx in (False,True):
            for fy in (False,True):
                coords=[]
                for x in range(3):
                    for y in range(3):
                        u,v=(y,x) if swap else (x,y)
                        if fx:u=2-u
                        if fy:v=2-v
                        coords.append(b[3*u+v])
                a.append(tuple(coords))
    return min(a)
def key(b):
    return hashlib.sha256(bytes(x+1 for x in canonical(b))).hexdigest()
def gitrows(ref,path,compressed=False):
    data=subprocess.check_output(['git','show',f'{ref}:{path}'])
    if compressed:data=gzip.decompress(data)
    return [json.loads(s) for s in data.splitlines()]
def generate():
    r26=gitrows('87c15dc828d6eea519c219fd8a412fa21dfc464d','docs/research/quartz_general_game_metareasoning_20261123/coding/data/public_sources.jsonl')
    r27=gitrows('7dfa3c96dc99c57ec5b52054211bf0280de412f5','docs/research/quartz_general_joint_posterior_20261123/coding/data/public_sources.jsonl')
    r29=gitrows('576cbe554ea0e1ea52d011e6a92b6ad567ca2501','docs/research/quartz_c29_noisy_query_20261123/evidence/C09_CORRECTED_BENCHMARK/requests.jsonl.gz',True)
    old=[{key(replay(r['history'])[0]) for r in rows} for rows in (r26,r27,r29)]
    assert len(old[0])==300 and len(old[1])==144 and old[2]==old[0] and not old[0]&old[1]
    excluded=set.union(*old)
    seen=set(); roots={}
    def visit(b,p,h):
        if b in seen or outcome(b) is not None:return
        seen.add(b)
        if h:roots.setdefault(key(b),h)
        for i in range(9):
            if b[i]==0:visit(apply(b,i,p),-p,h+[i])
    visit(EMPTY,1,[])
    groups=defaultdict(list)
    for k,h in roots.items():
        if k not in excluded:groups[len(h)].append((k,h))
    new=[]
    for ply,entries in sorted(groups.items()):
        for i,(k,h) in enumerate(sorted(entries)):
            new.append({'id':k,'history':h,'ply':ply,'split':'HELDOUT' if len(entries)>1 and i%3==0 else 'TRAIN'})
    assert len({r['id'] for r in new})==len(new)
    assert not {r['id'] for r in new}&excluded
    assert all(key(replay(r['history'])[0])==r['id'] for r in new)
    (D/'sources.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in new))
    manifest={'canonical_nonterminal_plies1_8':len(roots),'excluded_c26':len(old[0]),'excluded_c27':len(old[1]),'c29_equals_c26':old[0]==old[2],
              'excluded_in_population':len(excluded & roots.keys()),'new':len(new),
              'counts':{s:dict(Counter(str(r['ply']) for r in new if r['split']==s)) for s in ('TRAIN','HELDOUT')},
              'source_sha256':hashlib.sha256((D/'sources.jsonl').read_bytes()).hexdigest(),
              'scope':'canonical distinct C26-C29 policy roots, possible overlapping descendants and prior exhaustive oracle use'}
    (D/'population.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':generate()
