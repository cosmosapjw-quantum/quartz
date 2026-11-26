"""Post-lock descriptive signed flip gain and natural-frame sampling diagnostic."""
from experiment import *
def frac_mean(v):return sum(v,F(0))/len(v)
def bootstrap(v,seed=2026112837):
    rng=random.Random(seed);q=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(4000))
    return [float(q[99]),float(q[3899])]
def main():
    fr=read('FRAME.json');lab=load_labels('TRAIN_CAL')|load_labels('HELDOUT');out={}
    for group in GROUPS:
        rr=[r for r in fr if r['root']['job']['group']==group];n=len(rr);hh={}
        for h in (96,112,128,256):
            d=[];cost=[];flip=[];rgs=[]
            for r in rr:
                i=r['root']['job']['id'];u=U(lab.get(i));a,b=r['arms']['L'+str(h)],r['arms']['L80']
                z=delta(a['best'],b['best'],u);d.append(z);cost.append(a['NN_calls']-b['NN_calls'])
                flip.append(a['best']!=b['best'])
                if flip[-1]:rgs.append([delta(a['best'],b['best'],U(lab[i],j)) for j in range(4)])
            k=sum(flip);ds=[x for x,f in zip(d,flip) if f];assert all(x==0 for x,f in zip(d,flip) if not f)
            g=frac_mean(d);cond=frac_mean(ds) if k else None
            if k:assert F(k,n)*cond==g
            cp=F(sum(cost),n);hh[str(h)]=dict(flip=k,n=n,p_flip=k/n,mean_gain=float(g),gain_ci=bootstrap(d),conditional_gain=float(cond) if k else None,
                positive=sum(x>0 for x in ds),negative=sum(x<0 for x in ds),zero=sum(x==0 for x in ds),extra_NN=float(cp),signed_gain_per_NN=float(g/cp) if cp>0 else None,
                mixed_reference_sign=sum(min(x)<0<max(x) for x in rgs),reference_disagree=sum(len(set(x))>1 for x in rgs))
        out[group]=hh
    rr=[r for r in fr if r['root']['job']['group']=='NATURAL'];stable=[r for r in rr if len(r['actions'])==1];hard=[r for r in rr if len(r['actions'])>1];N=len(rr);M=len(stable);n=min(M,max(2,(M+3)//4))
    def u(r,h):return U(lab[r['root']['job']['id']])[str(r['arms']['L'+str(h)]['best'])]
    sy=[u(r,80) for r in stable];hy=[u(r,80) for r in hard];truth=frac_mean(sy+hy)
    rng=random.Random(SEED+12345);vals=[];naive=[];vs=[]
    for _ in range(2048):
        x=rng.sample(sy,n)
        e=estimate_absolute(N,[(M,x),(len(hy),hy)])
        vals.append(e['mean']);vs.append(e['variance']);naive.append(frac_mean(x+hy))
    th=F(M,N)**2*(1-F(n,M))*sum((x-frac_mean(sy))**2 for x in sy)/(M-1)/n
    diag=dict(N=N,stable=M,sample=n,hard=len(hard),true=float(truth),HT_mean=float(frac_mean(vals)),HT_empirical_bias=float(frac_mean(vals)-truth),naive_mean=float(frac_mean(naive)),
         theoretical_variance=float(th),empirical_variance=float(frac_mean([(x-truth)**2 for x in vals])),mean_estimated_variance=float(frac_mean(vs)),reps=len(vals))
    save('DECOMPOSITION.json',dict(groups=out,natural_resampling=diag,scope='post-lock descriptive analysis, reference-conditioned signed gains not true minimax or adaptive refit'))
    print('DECOMP',json.dumps(dict(groups=out,natural_resampling=diag),sort_keys=True),flush=True)
if __name__=='__main__':main()
