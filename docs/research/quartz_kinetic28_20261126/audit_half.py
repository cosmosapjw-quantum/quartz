"""Independent readback of post-freeze one-stage comparison."""
from audit_k28 import *
p=D/'HALF_DIAGNOSTIC.jsonl.gz';h=read('HALF_DIAGNOSTIC.json')
assert sha(p)==h['sha256'];total=C.Counter();by=C.defaultdict(C.Counter)
rows0={};rows1={};nn=vis=n=0
for t in raw(p):
    r=t['root'];i=r['id'];assert r==roots[i] and i in lab and i not in rows0
    oo=[(a,j) for a in (0,2) for j in range(2)];random.Random(SEED+771001+i).shuffle(oo)
    assert t['order']==[list(z) for z in oo]
    z={}
    for a,j in oo:
        x=t['results'][str(a)+':'+str(j)]
        native_verify(x,r,[a,STOP],64,total,by);n+=1;nn+=x['NN_calls'];vis+=x['visits']
        z[str(a)+str(j)]=small(x,lab[i])
    rows0[i]=z
for t in rows('TEST_LIVE',64):
    i=t['root']['id']
    if i in rows0:rows1[i]={k:small(x,lab[i]) for k,x in t['results'].items()}
assert set(rows0)==set(rows1) and (n,nn,vis)==(h['requests'],h['NN'],h['visits'])
def get(k,i,j):return (rows0 if k in ('0','2') else rows1)[i][k+str(j)]
for k,z in h['means'].items():
    v=[get(k,i,j) for i in rows0 for j in range(2)]
    near(S.mean(x['u'] for x in v),z['utility']);near(S.mean(x['net'] for x in v),z['net']);near(S.mean(x['nn'] for x in v),z['NN']);near(S.mean(x['opt']-x['u'] for x in v),z['regret'])
for k,z in h['pairs'].items():
    a,b=k.split(':');gains=[];cost=[];net=[]
    for i in rows0:
        x=[get(a,i,j) for j in range(2)];y=[get(b,i,j) for j in range(2)]
        gains.append(S.mean(xx['u']-yy['u'] for xx,yy in zip(x,y)))
        cost.append(S.mean(xx['nn']-yy['nn'] for xx,yy in zip(x,y)))
        net.append(S.mean(xx['net']-yy['net'] for xx,yy in zip(x,y)))
    near(S.mean(gains),z['gain']);near(S.mean(cost),z['NN_delta']);near(S.mean(net),z['net_gain'])
    assert sum(x>0 for x in gains)==z['better'] and sum(x<0 for x in gains)==z['worse']
out=dict(requests=n,NN=nn,visits=vis,roots=len(rows0),totals=dict(total),scope='exact resource/receipt readback and root-repeat means, no model refit')
(D/'HALF_AUDIT.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(out)
