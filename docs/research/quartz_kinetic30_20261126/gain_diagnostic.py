"""Post-freeze diagnostic: the native-gain moment is not the same estimand as conditional log loss."""
from trial import *
m=read('MODEL.json');ll=labelmap('TEST')
v=[[] for _ in range(8)]
for t in rows('TEST_ATLAS'):
    r=t['root'];i=r['id']
    if i not in ll:continue
    a=datum(t,ll[i]);u=a['u'];ds=neighbors(m['donors'],a)
    prior,ww,ep,p,q0,base=prior_law(m['donors'],a,m['k'],m['eps'],ds)
    _,_,rr=raw_channel(m['donors'],a,m['k'],m['eps'],ds)
    mm=prior['prior'];L=len(u)
    for c,raw0 in enumerate(rr):
        pz=raw0['mass'];J=raw0['j'];al=m['alpha'][c]
        D0=sum(pz[z]*mm[z//2] for z in range(2*L))-mm[0]
        D1=sum(J[z][z//2] for z in range(2*L))-mm[0]
        mix=(1-al)*D0+al*D1
        emp=sum(w*S.mean((m['donors'][k]['u'][z//2]-m['donors'][k]['u'][0])/2 for z in m['donors'][k]['z'][c]) for k,w in ww)
        uniform=ep*(sum(base)/L-base[0])
        close(D1,emp+uniform)
        j,_=projected(raw0,mm,al,m['guard'][c],prior['donors'])
        close(mix,sum(j[z][z//2] for z in range(2*L))-mm[0])
        actual=S.mean((u[z//2]-u[0])/2 for z in a['z'][c])
        v[c].append((D0,D1,mix,D1-D0,actual))
out=[]
for c,ls in enumerate(v):
    t=list(zip(*ls));actual=t[-1];z=dict(arm=c,n=len(ls),alpha=m['alpha'][c],actual=S.mean(actual),covariance_contribution=S.mean(t[3]))
    for i,key in enumerate(('independence','raw_joint','shrunk')):
        p=t[i];z[key]=dict(mean=S.mean(p),bias=S.mean(x-y for x,y in zip(p,actual)),rmse=math.sqrt(S.mean((x-y)**2 for x,y in zip(p,actual))),corr=S.correlation(p,actual) if S.pstdev(p)>0 and S.pstdev(actual)>0 else None)
    out.append(z)
p=BIN.parent/'GAIN_DIAGNOSTIC.json'
assert not p.exists()
p.write_text(json.dumps(dict(scope='post-freeze heldout moment decomposition, no policy selection or refitting',arms=out),indent=2,sort_keys=True)+'\n')
print(json.dumps(out,sort_keys=True),flush=True)
