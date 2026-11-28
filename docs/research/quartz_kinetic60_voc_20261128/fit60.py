"""Exact, family-equal conditional paired continuation estimates; no held-out access."""
from plan60 import D,T,HS,sha,read,raw
import collections as C,json
from fractions import Fraction as F
def cell(s):
    h=s['visits'];assert h in HS and not s['locked20'] and s['rival'] is not None
    return (h//4-1)*4+2*int(2*s['distance']>=20-h)+int(s['move']==s['prior_best'])
def fit():
    p=read(D/'PLAN.json');lock=read(T/'TRAIN_LOCK.json')
    assert sha(D/'PLAN.json')==lock['plan'] and sha(D/'continuation60.py')==lock['source']
    assert lock['binary']==p['old_binary_sha256']
    families=C.defaultdict(lambda:C.defaultdict(list));seen=set();counts=C.Counter()
    for m in lock['files']:
        f=T/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];assert j==p['roots'][j['id']] and j['id'] not in seen;seen.add(j['id'])
            full=z['games']['0'];a=full['rows'][0]
            for h in HS:
                g=z['games'][str(h)];b=g['rows'][0];e=b['early'];pair=z['pairs'][str(h)]
                gain=F(full['score2']-g['score2'],2);cost=a['actual_nn']-b['actual_nn']
                assert gain==F(pair['gain2'],2) and cost==pair['root_cost']>=0
                counts['comparisons']+=1;counts['eligible']+=e['reached']
                if not e['reached']:assert gain==0 and cost==0;continue
                s=e['at'];idx=cell(s)
                assert s['visits']==h and s['physical']==b['actual_nn']
                dq=sum(r['actual_nn'] for r in full['rows'] if r['arm']=='Q')-sum(r['actual_nn'] for r in g['rows'] if r['arm']=='Q')
                families[idx][j['family_id']].append((gain,F(cost),max(gain,0),F(gain>0),F(pair['changed']),F(dq)))
    assert seen==set(range(len(p['roots'])))
    rows=[];table=[]
    for i in range(16):
        fs=families[i];n=sum(map(len,fs.values()));nf=len(fs)
        means=[]
        if nf:
            means=[sum(sum((r[k] for r in z),F(0))/len(z) for z in fs.values())/nf for k in range(6)]
            net=means[0]-means[1]*F(1,500)
        else:net=None
        stop=nf>=p['min_family'] and net is not None and net<=0
        table.append(stop)
        rows.append(dict(cell=i,n=n,families=nf,mean_gain=float(means[0]) if means else None,mean_root_cost=float(means[1]) if means else None,mean_downside=float(means[2]) if means else None,p_harm=float(means[3]) if means else None,p_flip=float(means[4]) if means else None,mean_total_Qcost=float(means[5]) if means else None,net_exact=str(net) if net is not None else None,stop=stop))
    out=dict(table=table,cells=rows,counts=dict(counts),train_lock_sha256=sha(T/'TRAIN_LOCK.json'),price='1/500',min_family=p['min_family'],root_count=len(seen))
    (T/'TABLE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':fit()
