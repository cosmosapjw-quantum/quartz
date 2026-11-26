from fractions import Fraction as F
def solve(points,B):
    """points[z][h]=(incremental cost, gain in quarter-score units)."""
    B=F(B);out={};segments=[]
    if B<0:return dict(feasible=False,prob={z:{20:F(1)} for z in points},cost=F(0),gain=F(0),dual=None,dual_bound=None)
    for z,hs in sorted(points.items()):
        xs={}
        for h,(k,g) in sorted(hs.items()):
            k,g=F(k),F(g);assert k>=0
            if k not in xs or g>xs[k][1]:xs[k]=(h,g)
        assert 0 in xs
        hull=[]
        for k,(h,g) in sorted(xs.items()):
            if hull and g<=hull[-1][1]:continue
            while len(hull)>=2:
                x,y,_=hull[-2];a,b,_=hull[-1]
                if (b-y)*(k-a)>(g-b)*(a-x):break
                hull.pop()
            hull.append((k,g,h))
        out[z]={hull[0][2]:F(1)}
        for i,(a,b) in enumerate(zip(hull,hull[1:])):
            k0,g0,h0=a;k1,g1,h1=b
            segments.append(((g1-g0)/(k1-k0),z,i,k1-k0,h0,h1))
    left=B;lam=F(0)
    for slope,z,i,k,h0,h1 in sorted(segments,key=lambda s:(-s[0],s[1],s[2])):
        if left<=0:
            lam=slope;break
        q=min(F(1),left/k);assert out[z]=={h0:F(1)}
        out[z]={h1:F(1)} if q==1 else {h0:1-q,h1:q}
        left-=q*k
        if q<1:lam=slope;break
    cost=sum(q*F(points[z][h][0]) for z,hh in out.items() for h,q in hh.items())
    gain=sum(q*F(points[z][h][1]) for z,hh in out.items() for h,q in hh.items())
    dual=lam*B+sum(max(F(g)-lam*F(k) for k,g in hs.values()) for hs in points.values())
    assert cost<=B and gain==dual and lam>=0 and lam*(B-cost)==0,(points,B,out,cost,gain,dual,lam)
    return dict(feasible=True,prob=out,cost=cost,gain=gain,dual=lam,dual_bound=dual)
