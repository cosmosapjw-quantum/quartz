"""Independent R07 finite check of scalar envelopes against exact stored-float algebra."""
from fractions import Fraction as Q
from pathlib import Path
import datetime,hashlib,json,math,random,subprocess,sys,time
UP=lambda x: math.nextafter(x,math.inf)
DN=lambda x: math.nextafter(x,-math.inf)
ua=lambda a,b:UP(a+b)
da=lambda a,b:DN(a+b)
um=lambda a,b:UP(a*b)
dm=lambda a,b:DN(a*b)
ud=lambda a,b:UP(a/b)
dd=lambda a,b:DN(a/b)
u=2.0**-53; sigma=sys.float_info.min
def gf(k):
    ku=um(float(k),u)
    den=DN(1.0-ku)
    return ud(ku,den),ud(um(float(k),sigma),den)
def exact_A(side,rho):
    n=side*side
    a=[[Q(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a[i][i]=Q(1+4*rho)
        for j in range(n):
            if abs(i//side-j//side)+abs(i%side-j%side)==1:a[i][j]=Q(-rho)
    return a
def mv(a,x):return [sum((v*w for v,w in zip(row,x)),Q(0)) for row in a]
def dot(a,b):return sum((v*w for v,w in zip(a,b)),Q(0))
def solve(a,b):
    assert all(isinstance(v,Q) for row in a for v in row) and all(isinstance(v,Q) for v in b)
    rows=[list(row)+[v] for row,v in zip(a,b)];n=len(b)
    for k in range(n):
        assert rows[k][k]
        pivot=rows[k][k];rows[k]=[v/pivot for v in rows[k]]
        for i in range(n):
            if i!=k:
                z=rows[i][k]
                rows[i]=[v-z*w for v,w in zip(rows[i],rows[k])]
    return [row[-1] for row in rows]
def plain_stencil(side,rho,x):
    out=[]
    for i in range(len(x)):
        y=(1.0+4.0*rho)*x[i]
        if i//side>0:y-=rho*x[i-side]
        if i//side+1<side:y-=rho*x[i+side]
        if i%side>0:y-=rho*x[i-1]
        if i%side+1<side:y-=rho*x[i+1]
        out.append(y)
    return out
def square_sum(x):
    out=0.0
    for v in x:out+=v*v
    return out
def pos_dot_interval(h,n):
    g,f=gf(2*n)
    lo=max(0.0,dd(max(0.0,DN(h-f)),ua(1.0,g)))
    hi=ud(ua(h,f),DN(1.0-g))
    return lo,hi
def admitted(b,x):
    bmax=max(map(abs,b));xmax=max(map(abs,x))
    if bmax==0:return xmax==0
    return 2.0**-400<=bmax<=2.0**100 and xmax<=2.0**200 and xmax<=bmax*2.0**100
def enclosing(side,rho,b,x,scale):
    assert admitted(b,x)
    n=len(b);B=max(map(abs,b));X=max(map(abs,x))
    if B==0:return dict(L=0.,U=0.,ell=(0.,0.),D=(0.,0.),P=(0.,0.),Q=(0.,0.),Ey=0.,Yhat=[0.]*n,relative_upper=0.,score=0.)
    yh=plain_stencil(side,rho,x);rh=[bi-yi for bi,yi in zip(b,yh)]
    Y=max(map(abs,yh));R=max(map(abs,rh))
    ph=qh=dh=0.
    for bi,xi,yi,ri in zip(b,x,yh,rh):
        ph+=bi*xi;qh+=xi*yi;dh+=ri*ri
    g,f=gf(2*n);g9,f9=gf(9);nf=float(n)
    ey=ua(um(um(g9,1.0+8.0*rho),X),f9)
    er=ua(ua(ey,um(u,ua(B,Y))),sigma)
    ep=ua(um(um(um(g,nf),B),X),f)
    eq=ua(ua(um(um(um(g,nf),X),Y),f),um(um(nf,X),ey))
    p=(DN(ph-ep),UP(ph+ep));q=(DN(qh-eq),UP(qh+eq))
    ell=(DN(dm(2.,p[0])-q[1]),UP(um(2.,p[1])-q[0]))
    d0l,d0u=pos_dot_interval(dh,n)
    perturb=um(nf,ua(um(um(2.,R),er),um(er,er)))
    dl=max(0.,DN(d0l-perturb));du=UP(d0u+perturb)
    rawlo=max(0.,da(ell[0],dd(dl,1.+8.*rho)))
    rawhi=min(scale,ua(ell[1],du))
    assert rawlo<=rawhi
    L=max(0.,dd(rawlo,scale));U=min(1.,ud(rawhi,scale))
    bl,bu=pos_dot_interval(square_sum(b),n)
    assert bl>0 and bu<=scale
    relative=ud(UP(math.sqrt(du)),DN(math.sqrt(bl)))
    score=(2.*ph-qh)/scale
    assert all(math.isfinite(v) for v in [L,U,relative,score])
    return dict(L=L,U=U,ell=ell,D=(dl,du),P=p,Q=q,Ey=ey,Yhat=yh,relative_upper=relative,score=score)
def scale_for(b):
    norm=dot(list(map(Q,b)),list(map(Q,b)))
    if not norm:return 1.
    scale=math.ldexp(1.,math.frexp(float(norm))[1]+1)
    while Q(scale)<2*norm:scale*=2.
    return scale
start=time.perf_counter();cases=[];counts=dict(cases=0,exact_stencil_components=0,exact_scalar_containments=0,residual_upper_checks=0)
rng=random.Random(2400723)
def check(label,side,rho,b,x):
    a=exact_A(side,rho);bf=list(map(Q,b));xf=list(map(Q,x))
    q=dot(bf,solve(a,bf));y=mv(a,xf);r=[bi-yi for bi,yi in zip(bf,y)]
    p=dot(bf,xf);z=dot(xf,y);ell=2*p-z;D=dot(r,r);scale=scale_for(b)
    v=enclosing(side,rho,b,x,scale)
    for yi,hi in zip(y,v['Yhat']):assert abs(yi-Q(hi))<=Q(v['Ey'])
    for name,value in [('P',p),('Q',z),('ell',ell),('D',D)]:
        assert Q(v[name][0])<=value<=Q(v[name][1]),(label,name)
    assert Q(v['L'])<=q/Q(scale)<=Q(v['U']),(label,'utility')
    norm=dot(bf,bf)
    if norm:assert D<=Q(v['relative_upper'])**2*norm,(label,'relative')
    else:assert D==0 and v['relative_upper']==0
    counts['cases']+=1;counts['exact_stencil_components']+=len(b);counts['exact_scalar_containments']+=5;counts['residual_upper_checks']+=1
    cases.append({'label':label,'side':side,'rho':rho,'scale':scale,'utility_exact':str(q/Q(scale)),'L':v['L'],'U':v['U'],'relative_upper':v['relative_upper']})
for side in [2,3]:
    n=side*side
    for rho in [0,1,16,64]:
        for exponent in [-400,0,100]:
            amp=math.ldexp(1.,exponent)
            b=[((-1.)**i)*((i+1)/n)*amp for i in range(n)]
            for mode in ['zero','dyadic_step','perturbed']:
                x=[0.]*n if mode=='zero' else [v/64. for v in b] if mode=='dyadic_step' else [rng.uniform(-1,1)*amp*4 for _ in b]
                # Do not exceed the stated absolute x domain at the largest scale.
                check(f'{side}_{rho}_{exponent}_{mode}',side,rho,b,x)
for rho in [0,1,64]:
    check(f'zero_{rho}',2,rho,[0.]*4,[0.]*4)
check('mixed_subnormal',2,64,[1.,math.ldexp(1.,-1074),0.,0.],[math.ldexp(1.,-1074),0.,0.,0.])
check('max_supported_state',2,64,[math.ldexp(1.,100),0.,0.,0.],[math.ldexp(1.,200),0.,0.,0.])
assert not admitted([math.ldexp(1.,-600),0.,0.,0.],[0.]*4)
assert not admitted([1.,0.,0.,0.],[math.ldexp(1.,101),0.,0.,0.])
# New 3x3 exact example, independent of the prior 2x2 C23 check.
a=exact_A(3,16)
b1=[Q(i,16) for i in range(1,10)];b2=[Q(0)]*9;b2[4]=Q(1,32)
example=[]
for b in [b1,b2]:
    x=[v/64 for v in b];y=mv(a,x);r=[bi-yi for bi,yi in zip(b,y)]
    q=dot(b,solve(a,b));ell=2*dot(b,x)-dot(x,y);D=dot(r,r)
    assert q-ell==dot(r,solve(a,r))
    example.append({'q':str(q),'ell':str(ell),'D':str(D),'L':str(ell+D/129),'U':str(ell+D),'identity':str(q-ell-dot(r,solve(a,r)))})
assert Q(example[0]['L'])>Q(example[1]['U'])
root=Path(__file__).resolve().parents[3]
out={'status':'VERIFIED_FINITE_ENVELOPE_ASSERTIONS','assertion_tested':'Exact Fraction stencil/P/Q/ell/D/q and relative upper containment at all prescribed stored binary64 states; mathematical domain predicates reject two unsupported states. This is not the Rust implementation test.','counts':counts,'cases':cases,'exact_3x3_example':example,'strict_separation':True,'unsupported_predicates_checked':2,'argv':sys.argv,'python':sys.version,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dest=Path(__file__).resolve().parents[1]/'evidence/R07_FINITE_ENVELOPES.json'
dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
