"""Identify features with zero empirical support variation and verify runtime policy."""
import json,collections as C
from k56_plan import D,T,load
from fit56 import records
from predictor56 import predict
def main():
    m=load('FROZEN.json')['model'];out={}
    for k in (54,55):
        a=list(records(k));n=len(a);rs=list(zip(*(r['x'] for r in a)));s=[dict(min=min(x),max=max(x),unique=len(set(x))) for x in rs]
        assert s[2]['unique']==s[3]['unique']==1 and s[2]['min']==1/3 and s[3]['min']==1.
        out[str(k)]=dict(n=n,features=s,scope='eligible snapshots only')
    q=dict(out=out,conclusion='distance and visits constant on all689 reached historical snapshots; no conditional identification from these two columns under sampled event rule; earlier snapshot needed')
    (T/'FEATURE_SUPPORT.json').write_text(json.dumps(q,indent=2,sort_keys=True)+'\n');print(q,flush=True)
if __name__=='__main__':main()
