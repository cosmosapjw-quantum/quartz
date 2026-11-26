"""Post-lock readout decomposition on actual same-run outcomes, no policy refit."""
from trial import *
ll=labelmap('TEST');rows0={}
for t in rows('TEST_LIVE'):
    i=t['root']['id']
    if i in ll:rows0[i]=t['results']
assert set(rows0)==set(ll)
ids=sorted(ll)
def ci(v):
    rng=random.Random(SEED+918);b=sorted(sum(rng.choices(v,k=len(v)))/len(v) for _ in range(4000))
    return [b[99],b[3899]]
def stat(v):return dict(n=len(v),mean=S.mean(v),ci=ci(v),better=sum(x>0 for x in v),worse=sum(x<0 for x in v))
out={}
for k,b in [('prior','short'),('continue','serial'),('priced','short')]:
    choose=[];read=[];total=[];natives=[];learned=[];changed=0
    for i in ids:
        u=ll[i];d=rows0[i];x=[d[k+':'+str(j)] for j in range(2)];y=[d[b+':'+str(j)] for j in range(2)]
        a=S.mean((u[z['native_best']]-u[w['best']])/2 for z,w in zip(x,y))
        r=S.mean((u[z['best']]-u[z['native_best']])/2 for z in x)
        c=S.mean((u[z['best']]-u[w['best']])/2 for z,w in zip(x,y))
        assert a+r==c
        choose.append(a);read.append(r);total.append(c)
        natives+= [u[z['native_best']]/2 for z in x];learned+=[u[z['best']]/2 for z in x];changed+=sum(z['best']!=z['native_best'] for z in x)
    out[k+':'+b]=dict(selection_native=stat(choose),readout=stat(read),total=stat(total),native_utility=S.mean(natives),learned_utility=S.mean(learned),readout_changed=changed)
# Relative log score decomposition: a worse conditional forecast is not proof true MI=0.
out['scope']='post-freeze root-repeat diagnostic, same-run readout difference, no refit or policy selection'
save('READOUT_DIAGNOSTIC.json',out);print(json.dumps(out,sort_keys=True),flush=True)
