"""Small independent recursive finite-horizon reference."""
TOL=1e-12
def plan(state,row,costs,horizon):
    best=max(row(c)[0] for c in state); choice=None
    if horizon==0:return best,choice
    for i,c in enumerate(state):
        if c[0] is not None:continue
        score=-costs[i]
        for y,p in enumerate(row(c)[1:]):
            if not p:continue
            h=list(c[1]);h[y]+=1
            ss=list(state);ss[i]=(None,tuple(h),*c[2:])
            score+=p*plan(ss,row,costs,horizon-1)[0]
        if score>best+TOL:best,choice=score,i
    return best,choice
