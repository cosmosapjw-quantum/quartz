"""Independent readback: grid legality, raw path snapshots, fixed-design exact OPE."""
import pathlib,subprocess as SP,hashlib,json,gzip,random,collections as C,math,itertools,importlib.util,statistics as S
from fractions import Fraction as F
import numpy as np
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];P=D.parent
def read(n):return json.loads((D/n).read_text())
def lines(n):
    with gzip.open(D/n,'rt') as f:return [json.loads(x) for x in f]
def sha(x):return hashlib.sha256(x).hexdigest()
def fs(n):return sha((D/n).read_bytes())
def git(*a):return SP.check_output(['git','-C',str(R),*a])
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
spec=importlib.util.spec_from_file_location('grid17',P/'quartz_kinetic17_20261126/audit_k17.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def same(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(a,b)
import struct
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def boardkey(b):return tuple(tuple(c) for c in b)
trace_edges=trace_iters=trace_lanes=trace_receipts=trace_blocked=0
def verify_trace(t):
    global trace_edges,trace_iters,trace_lanes,trace_receipts,trace_blocked
    r=t['root'];a=t['response'];root=g.replay(r['history'])
    for k,trace_hook in enumerate(a['stages'][1]['hooks']):
        trace_lanes+=1;ledger={};payloads={};samples=C.defaultdict(list);c=C.Counter()
        pre=a['prefix_lanes'][k];mid=a['stages'][0]['lanes'][k];end=a['stages'][1]['lanes'][k]
        boundary={pre['visits']:(pre,a['prefix_hooks'][k]),mid['visits']:(mid,a['stages'][0]['hooks'][k]),end['visits']:(end,trace_hook)}
        trace=trace_hook['trace'];assert len(trace)==end['visits']
        def summary(i):
            return dict(iterations=i,edges=c['edges'],children=len(ledger),contexts=sum(len(x) for x in ledger.values()),shared=sum(len(x)>1 for x in ledger.values()),unique_base_leaf=len(payloads),eligible=c['eligible'],novel=c['novel'],blocked_receipt=c['blocked_receipt'],transferred=c['transferred'],receipt_items=sum(len(z[1]) for x in ledger.values() for z in x.values()),applied=c['applied'],changed=c['changed'],repeated_delivery=0)
        for i,z in enumerate(trace):
            trace_iters+=1;b=[v[:] for v in root];side=len(r['history'])%2+1;keys=[]
            assert z['mask']==(0 if i<pre['visits'] else a['chosen'][0] if i<mid['visits'] else a['chosen'][1])
            for mv in z['path']:
                assert not g.won(b,1) and not g.won(b,2) and len(b[mv])<6
                p0=boardkey(b);b[mv].append(side);side=3-side;keys.append((boardkey(b),(p0,mv)))
            terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
            assert z['terminal']==terminal;leaf=(boardkey(b),terminal);raw=f32(z['raw']);y=f32(z['leaf'])
            if terminal:assert raw==y==(-1 if g.won(b,3-side) else 0)
            if not z['mask']&4:assert raw==y
            if leaf in payloads:assert payloads[leaf]==raw
            else:payloads[leaf]=raw
            for j in reversed(range(len(keys))):
                trace_edges+=1;c['edges']+=1;child,ctx=keys[j];cc=ledger.setdefault(child,{})
                own,received=cc.setdefault(ctx,({},{}));other={}
                for p0,(dd,rr) in cc.items():
                    if p0==ctx:continue
                    for key,val in dd.items():
                        if key in other:assert other[key]==val
                        other[key]=val
                other.pop(leaf,None);elig={k:v for k,v in other.items() if k not in own};new={k:v for k,v in elig.items() if k not in received}
                s=z['steps'][j];assert (s['eligible'],s['novel'],s['blocked'])==(len(elig),len(new),len(elig)-len(new))
                c['eligible']+=len(elig);c['novel']+=len(new);c['blocked_receipt']+=len(elig)-len(new)
                mean=S.mean(new.values()) if new else None
                if mean is None:assert s['mean'] is None
                else:same(mean,s['mean'])
                assert f32(s['incoming'])==y
                if leaf in own:assert own[leaf]==raw
                if leaf in received:assert received[leaf]==raw
                own[leaf]=raw
                if not terminal and z['mask']&8 and new:
                    assert not received.keys()&new.keys();received.update(new);c['transferred']+=len(new);c['applied']+=1
                    incoming=y;alpha=min(.25,len(new)/(len(new)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*mean)));c['changed']+=abs(y-incoming)>1e-7
                raw=-raw;y=-y;assert f32(s['backup'])==y
            samples[z['path'][0]].append(y)
            if i+1 in boundary:
                l,h=boundary[i+1];assert summary(i+1)==h['ledger']
                for m,mv in enumerate(l['moves'][:l['len']]):
                    v=samples[mv];assert len(v)==l['n'][m];mu=S.mean(v) if v else 0.
                    same(sum(v),l['w'][m]);same(sum((x-mu)**2 for x in v),l['m2'][m])
        trace_receipts+=c['transferred'];trace_blocked+=c['blocked_receipt']

dd=lines('TARGET_RECEIPT.jsonl.gz');lock=read('TARGET_RECEIPT_LOCK.json');assert fs('TARGET_RECEIPT.jsonl.gz')==lock['sha256'] and len(dd)==lock['n']
for t in dd:verify_trace(t)
assert trace_receipts>0 and trace_blocked>0
out=dict(requests=len(dd),lanes=trace_lanes,iterations=trace_iters,edges=trace_edges,receipts=trace_receipts,blocked=trace_blocked,scope=lock['scope'])
(D/'TARGET_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
