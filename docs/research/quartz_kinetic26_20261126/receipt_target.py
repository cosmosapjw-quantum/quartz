"""Posthoc mechanism-only deterministic positive trace; no policy changes."""
from trial import *
candidates=[]
for t in raw(D/'TRAIN_CAL_ATLAS.jsonl.gz'):
    for i,a in enumerate(t['arms']['SERIAL']):
        p=a['stages'][0]['hooks'][0]['ledger'];q=a['stages'][1]['hooks'][0]['ledger']
        if p['transferred'] and q['blocked_receipt']>p['blocked_receipt']:
            candidates.append((t['root'],PATHS[i],a))
assert candidates
selected=[]
for pred in (lambda p:p[0]!=p[1] and p[1]&8,lambda p:p[0]!=p[1] and not p[1]&8,lambda p:p[0]==p[1] and p[1]&8):
    x=next((x for x in candidates if pred(x[1])),None)
    if x is not None:selected.append(x)
assert selected
with Server() as s,gzip.open(D/'TARGET_RECEIPT.jsonl.gz','wt',compresslevel=3) as fd:
    for r,path,b in selected:
        a=req(s,r,'SERIAL',path=path,trace=True)
        assert a['result']==b['result'] and a['chosen']==b['chosen']
        assert [x['hooks'][0]['ledger'] for x in a['stages']]==[x['hooks'][0]['ledger'] for x in b['stages']]
        emit(fd,dict(root=r,response=a))
        print(r['id'],path,'mid/final receipt',*[x['hooks'][0]['ledger']['transferred'] for x in a['stages']], 'mid/final blocked',*[x['hooks'][0]['ledger']['blocked_receipt'] for x in a['stages']],flush=True)
save('TARGET_RECEIPT_LOCK.json',dict(sha256=sha(D/'TARGET_RECEIPT.jsonl.gz'),n=len(selected),scope='posthoc mechanism check on TRAIN/CAL, no policy fitting or TEST selection'))
