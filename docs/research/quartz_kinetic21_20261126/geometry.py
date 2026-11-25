"""Retrospective exact algebra on K19 modeled joint law and K20 actual root observations."""
import pathlib,json,gzip,math,statistics as S,itertools,collections as C,hashlib
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=ROOT/'docs/research'
ST=[(0,a/2,b/2) for a,b in itertools.product(range(-2,3),repeat=2) if max(0,a,b)-min(0,a,b)<=2]
TAU=(.05,.1,.25,.5)
def phi(x,t=.25):
    a=max(x);w=[math.exp((v-a)/t) for v in x];z=math.fsum(w)
    return a+t*math.log(z/len(x)),[v/z for v in w]
def kl(a,b):return math.fsum(x*math.log(x/y) for x,y in zip(a,b) if x)
def mean(p):return [math.fsum(w*v[i] for w,v in zip(p,ST)) for i in range(3)]
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def close(a,b,eps=1e-10):assert abs(a-b)<=eps,(a,b)
def joint(J):
    p=[sum(r) for r in J];mu=mean(p);v=max(mu);a=max(range(3),key=mu.__getitem__);H=sum(w*max(s) for w,s in zip(p,ST))
    prob=[sum(r[z] for r in J) for z in range(3)]
    post=[mean([r[z]/prob[z] for r in J]) for z in range(3)]
    for i in range(3):close(sum(w*m[i] for w,m in zip(prob,post)),mu[i])
    ds=[[x-y for x,y in zip(m,mu)] for m in post]
    hard=sum(w*max(m) for w,m in zip(prob,post))-v
    variance=[sum(w*(d[i]-d[a])**2 for w,d in zip(prob,ds)) for i in range(3)]
    gap=[v-m for m in mu];bound=sum(s/(2*(math.sqrt(g*g+s)+g)) if g or s else 0 for s,g in zip(variance,gap))
    assert -1e-10<=hard<=min(bound,H-v)+1e-10
    mi=sum(J[i][z]*math.log(J[i][z]/(p[i]*prob[z])) for i in range(19) for z in range(3) if J[i][z])
    out=dict(hard=hard,EVPI=H-v,bound=min(bound,H-v),mi=mi,temperatures={})
    for t in TAU:
        f,q=phi(mu,t);val=geom=quad=rem=upper=0.
        for w,m,d in zip(prob,post,ds):
            f1,q1=phi(m,t);dr=sum(x*y for x,y in zip(q,d));g=t*kl(q,q1)
            close(f1-f,dr+g)
            val+=w*(f1-f);geom+=w*g
            quad+=w*(sum(x*y*y for x,y in zip(q,d))-dr*dr)/(2*t)
            rem+=w*(max(d)-min(d))**3/(6*t*t)
            upper+=w*(max(d)-min(d))**2/(8*t)
        close(val,geom);assert abs(hard-val)<=t*math.log(3)+1e-10 and abs(val-quad)<=rem+1e-10 and val<=upper+1e-10
        out['temperatures'][str(t)]=dict(soft=val,geom=geom,quad=quad,remainder_bound=rem,upper=upper)
    return out
def snap(s):
    moves=[a for a,p in s['candidates']];e={x['move']:x for x in s['edges']};n=[e.get(a,{}).get('n',0) for a in moves];w=[e.get(a,{}).get('w',0) for a in moves]
    assert sum(n)==s['root_n']
    return moves,n,[(b/(a+4)+1)/2 for a,b in zip(n,w)]
def flow(a,b):
    ca,n,x=snap(a);cb,nn,y=snap(b);assert ca==cb
    N=sum(n);M=sum(nn)-N;assert M>0
    g=M/(N+M);r=[v/N for v in n];j=[(v-u)/M for u,v in zip(n,nn)];z=[v/(N+M) for v in nn]
    assert min(j)>=0
    close(sum(abs(u-v) for u,v in zip(z,r)),g*sum(abs(u-v) for u,v in zip(r,j)))
    f,p=phi(x);f1,q=phi(y);dr=sum(u*(v-w) for u,v,w in zip(p,y,x));geo=.25*kl(p,q);close(f1-f,dr+geo)
    # Weighted JS and KL to fixed uniform. Zero-mass entries excluded.
    un=[1/len(r)]*len(r);js=(1-g)*kl(r,z)+g*kl(j,z)
    close(kl(z,un)-kl(r,un),g*(kl(j,un)-kl(r,un))-js)
    return dict(drift=dr,geometry=geo,phi_delta=f1-f,js=js)
def main():
    p19=R/'quartz_kinetic19_20261126/PREDICTIONS.jsonl.gz';p20=R/'quartz_kinetic20_20261126/ATLAS.jsonl.gz'
    allj=[];ranks=C.Counter()
    for r in raw(p19):
        rr={c:dict(joint(a['J']),gain=a['gain'],pred_gain=a['pred_gain']) for c,a in r['channels'].items()}
        for a in rr.values():close(a['hard'],a['pred_gain'])
        h=max(rr,key=lambda c:rr[c]['hard'])
        for t in TAU:ranks[str(t)]+=max(rr,key=lambda c:rr[c]['temperatures'][str(t)]['soft'])==h
        ranks['MI']+=max(rr,key=lambda c:rr[c]['mi'])==h
        allj.extend(rr.values())
    old=[]
    for r in raw(p20):
        for a in r['fixed']:
            prev=a['evidence']['prefix'];seq=[]
            for ph in a['evidence']['phases']:
                seq.append(flow(prev,ph['result']));prev=ph['result']
            totals={k:sum(z[k] for z in seq) for k in seq[0]};close(totals['phi_delta'],totals['geometry']+totals['drift'])
            old.append(totals)
    s=dict(k19_roots=len(allj)//6,k19_channels=len(allj),rank_agreement=dict(ranks),mean_hard=S.mean(x['hard'] for x in allj),mean_mi=S.mean(x['mi'] for x in allj),mean_bound=S.mean(x['bound'] for x in allj),mean_realized_gain=S.mean(x['gain'] for x in allj),temperatures={str(t):{k:S.mean(x['temperatures'][str(t)][k] for x in allj) for k in ('soft','geom','quad','remainder_bound','upper')} for t in TAU},k20_paths=len(old),k20_blocks=len(old)*3,k20_mean={k:S.mean(x[k] for x in old) for k in old[0]},k20_negative_drift=sum(x['drift']<0 for x in old),k20_negative_phi=sum(x['phi_delta']<0 for x in old),sources={str(p.relative_to(ROOT)):sha(p) for p in (p19,p20)},scope='retrospective exact model identities, observed score and occupation geometry; not new heldout fit or physical entropy production')
    (D/'RETROSPECTIVE.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print(json.dumps(s))
if __name__=='__main__':main()
