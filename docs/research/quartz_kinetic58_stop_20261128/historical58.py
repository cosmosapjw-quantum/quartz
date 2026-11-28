import collections as C,gzip,json
from k58_plan import D,T,sha
from parity58 import stable
def main():
    d=D.parent/'quartz_kinetic57_early_20261128';lk=json.loads((d/'NN_LOCK.json').read_text());ct=C.Counter()
    for f in lk['files']:
        p=d/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as r:
            for x in map(json.loads,r):
                c=x['control'];i=c['attempt'];ct['episodes']+=1
                if i is None:continue
                ct['attempts']+=1;a=c['rows'][i];ss=a['early']['snapshots'];q=next((v for u,v in zip(ss,ss[1:]) if stable(u,v)),None)
                if q is not None:
                    ct['stable']+=1;ct['h'+str(q['visits'])]+=1;ct['saved_root']+=a['cache_receipt']['root_physical']-q['physical'];ct['changed']+=a['move']!=q['move']
    (T/'HISTORICAL.json').write_text(json.dumps(dict(counts=dict(ct),source=sha(d/'NN_LOCK.json'),scope='predecision snapshot applicability and local root calls only; no inferred terminal outcomes for unplayed horizons'),indent=2,sort_keys=True)+'\n')
    print('HISTORICAL',dict(ct),flush=True)
if __name__=='__main__':main()
